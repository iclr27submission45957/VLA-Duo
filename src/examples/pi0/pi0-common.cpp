#include "pi0-common.h"

#include "log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Small helpers
// ============================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        LOG_ERR("Missing tensor: %s\n", name);
    }
    return t;
}

static void sincos_posemb(float t, int dim, float * out) {
    const int half = dim / 2;
    const float min_period = 4e-3f;
    const float max_period = 4.0f;
    for (int i = 0; i < half; i++) {
        float frac = (half > 1) ? (float)i / (half - 1) : 0.0f;
        float period = min_period * powf(max_period / min_period, frac);
        float x = t / period * 2.0f * (float)M_PI;
        out[i]        = sinf(x);
        out[half + i] = cosf(x);
    }
}

// Broadcast a 1-D bias [n] over a 2-D matmul output [n, m] with an explicit repeat.
static ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * mat, ggml_tensor * bias) {
    ggml_tensor * b2d = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, mat, ggml_repeat(ctx, b2d, mat));
}

// ============================================================
// Public dequant helpers
// ============================================================

void embd_lookup_f32(ggml_tensor * t, int token_id, float * out) {
    const int64_t n_embd = t->ne[0];
    const size_t row_bytes = ggml_row_size(t->type, n_embd);

    // Rows must be contiguous, and n_embd a multiple of the quant block size.
    if (row_bytes != (size_t) t->nb[1]) {
        GGML_ABORT("embd_lookup_f32: non-contiguous row (nb1=%zu, expected=%zu)",
                   (size_t) t->nb[1], row_bytes);
    }

    // Copy the whole table to the host once per tensor and index it there. A one-row device read can cost as much
    // as a full-table read. The table does not change during the process, so the static cache is safe.
    static std::map<const ggml_tensor *, std::vector<uint8_t>> host_cache;
    auto it = host_cache.find(t);
    if (it == host_cache.end()) {
        std::vector<uint8_t> buf(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        it = host_cache.emplace(t, std::move(buf)).first;
    }
    const uint8_t * row = it->second.data() + (size_t) token_id * row_bytes;
    if (t->type == GGML_TYPE_F32) { memcpy(out, row, row_bytes); return; }
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) row;
        for (int64_t j = 0; j < n_embd; j++) out[j] = ggml_fp16_to_fp32(h[j]);
        return;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) GGML_ABORT("embd_lookup_f32: type %s has no dequantizer", ggml_type_name(t->type));
    tt->to_float(row, out, n_embd);
}

// ============================================================
// GGUF weight loading
// ============================================================

static bool load_gguf_weights(const char * path,
                              ggml_context ** out_ctx,
                              ggml_backend_buffer_t * out_buf,
                              ggml_backend_t backend) {
    struct ggml_context * meta = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf_ctx = gguf_init_from_file(path, params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to load GGUF: %s\n", path);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG_INF("Loading %d tensors from %s\n", n_tensors, path);

    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * ctx_data = ggml_init(ctx_params);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * meta_tensor = ggml_get_tensor(meta, name);
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data, meta_tensor);
        ggml_set_name(data_tensor, name);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
    if (!buf) {
        LOG_ERR("Failed to allocate buffer for %s\n", path);
        ggml_free(ctx_data);
        ggml_free(meta);
        gguf_free(gguf_ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if (!f) {
        LOG_ERR("Failed to open %s\n", path);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_data);
        ggml_free(meta);
        gguf_free(gguf_ctx);
        return false;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        size_t nbytes = ggml_nbytes(cur);

        std::vector<uint8_t> read_buf(nbytes);
        fseek(f, offset, SEEK_SET);
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            LOG_ERR("Failed to read tensor %s\n", name);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
    }
    fclose(f);

    *out_ctx = ctx_data;
    *out_buf = buf;

    ggml_free(meta);
    gguf_free(gguf_ctx);
    return true;
}

bool load_pi0_model(pi0_model & m, const char * pali_path, const char * expert_path, ggml_backend_t backend) {
    if (!load_gguf_weights(pali_path, &m.ctx_pali, &m.buf_pali, backend)) {
        return false;
    }
    m.pali_embed      = get_tensor(m.ctx_pali, "token_embd.weight");
    m.pali_final_norm = get_tensor(m.ctx_pali, "output_norm.weight");
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.pali_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_pali, name);
        // Fused QKV [in, n_head*hd + 2*kv_dim] if present, else separate q/k/v.
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);    l.qkv_proj  = ggml_get_tensor(m.ctx_pali, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);  l.q_proj    = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);  l.k_proj    = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);  l.v_proj    = get_tensor(m.ctx_pali, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gateup.weight", il);  l.gateup_proj = ggml_get_tensor(m.ctx_pali, name);   // interleaved gate/up (optional)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = l.gateup_proj ? ggml_get_tensor(m.ctx_pali, name) : get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = l.gateup_proj ? ggml_get_tensor(m.ctx_pali, name) : get_tensor(m.ctx_pali, name);
        l.down_n_split = 0;   // K-split ffn_down (optional): take k0..kN as long as they are contiguous
        for (int ks = 0; ks < PI0_FFN_DOWN_MAX_SPLIT; ks++) {
            snprintf(name, sizeof(name), "blk.%d.ffn_down_k%d.weight", il, ks);
            ggml_tensor * t = ggml_get_tensor(m.ctx_pali, name);
            if (!t) break;
            l.down_k[ks] = t; l.down_n_split = ks + 1;
        }
        GGML_ASSERT(l.down_n_split != 1 && "ffn_down_k0 without ffn_down_k1 is not a valid split");
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);
        l.down_proj = l.down_n_split ? ggml_get_tensor(m.ctx_pali, name) : get_tensor(m.ctx_pali, name);
    }

    if (!load_gguf_weights(expert_path, &m.ctx_expert, &m.buf_expert, backend)) {
        return false;
    }
    // pi0.5 expert (adaRMS) is auto-detected by its final-norm modulation Dense.
    m.pi05 = ggml_get_tensor(m.ctx_expert, "output_norm.dense.weight") != nullptr;
    if (m.pi05) {
        m.expert_final_norm_dense_w = get_tensor(m.ctx_expert, "output_norm.dense.weight");
        m.expert_final_norm_dense_b = get_tensor(m.ctx_expert, "output_norm.dense.bias");
        LOG_INF("pi0: action expert is pi0.5 (adaRMS, %d action tokens, no state token)\n", PI0_ACTION_HORIZON);
    } else {
        m.expert_final_norm = get_tensor(m.ctx_expert, "output_norm.weight");
    }
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.expert_layers[il];
        if (m.pi05) {
            snprintf(name, sizeof(name), "blk.%d.attn_norm.dense.weight", il); l.attn_norm_dense_w = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_norm.dense.bias", il);   l.attn_norm_dense_b = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.dense.weight", il);  l.ffn_norm_dense_w  = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.dense.bias", il);    l.ffn_norm_dense_b  = get_tensor(m.ctx_expert, name);
        } else {
            snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_expert, name);
        }
        // Fused QKV [in, n_head*hd + 2*kv_dim] if present, else separate q/k/v.
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);    l.qkv_proj  = ggml_get_tensor(m.ctx_expert, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);  l.q_proj    = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);  l.k_proj    = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);  l.v_proj    = get_tensor(m.ctx_expert, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gateup.weight", il);  l.gateup_proj = ggml_get_tensor(m.ctx_expert, name);   // interleaved gate/up (optional)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = l.gateup_proj ? ggml_get_tensor(m.ctx_expert, name) : get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = l.gateup_proj ? ggml_get_tensor(m.ctx_expert, name) : get_tensor(m.ctx_expert, name);
        l.down_n_split = 0;   // K-split ffn_down (optional): take k0..kN as long as they are contiguous
        for (int ks = 0; ks < PI0_FFN_DOWN_MAX_SPLIT; ks++) {
            snprintf(name, sizeof(name), "blk.%d.ffn_down_k%d.weight", il, ks);
            ggml_tensor * t = ggml_get_tensor(m.ctx_expert, name);
            if (!t) break;
            l.down_k[ks] = t; l.down_n_split = ks + 1;
        }
        GGML_ASSERT(l.down_n_split != 1 && "ffn_down_k0 without ffn_down_k1 is not a valid split");
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);
        l.down_proj = l.down_n_split ? ggml_get_tensor(m.ctx_expert, name) : get_tensor(m.ctx_expert, name);
    }

    m.action_in_proj_w      = get_tensor(m.ctx_expert, "action_in_proj.weight");
    m.action_in_proj_b      = get_tensor(m.ctx_expert, "action_in_proj.bias");
    m.action_out_proj_w     = get_tensor(m.ctx_expert, "action_out_proj.weight");
    m.action_out_proj_b     = get_tensor(m.ctx_expert, "action_out_proj.bias");
    if (m.pi05) {
        // Fold the adaRMS +1 into the scale third of every modulation bias, so the graph computes
        // rms(x) * (1 + scale) + shift with one mul on the rms_norm output.
        auto fold_one = [](ggml_tensor * b) {
            if (!b) return;
            // reads/writes ggml_nbytes as raw floats, so it is only correct for an F32 bias
            GGML_ASSERT(b->type == GGML_TYPE_F32 && "adaRMS modulation bias must be F32 to fold the +1");
            std::vector<float> buf(ggml_nelements(b));
            ggml_backend_tensor_get(b, buf.data(), 0, ggml_nbytes(b));
            for (int i = 0; i < EXPERT_N_EMBD; i++) buf[i] += 1.0f;
            ggml_backend_tensor_set(b, buf.data(), 0, ggml_nbytes(b));
        };
        for (int il = 0; il < PI0_N_LAYER; il++) {
            fold_one(m.expert_layers[il].attn_norm_dense_b);
            fold_one(m.expert_layers[il].ffn_norm_dense_b);
        }
        fold_one(m.expert_final_norm_dense_b);
        m.time_mlp_in_w  = get_tensor(m.ctx_expert, "time_mlp_in.weight");
        m.time_mlp_in_b  = get_tensor(m.ctx_expert, "time_mlp_in.bias");
        m.time_mlp_out_w = get_tensor(m.ctx_expert, "time_mlp_out.weight");
        m.time_mlp_out_b = get_tensor(m.ctx_expert, "time_mlp_out.bias");
    } else {
        m.state_proj_w          = get_tensor(m.ctx_expert, "state_proj.weight");
        m.state_proj_b          = get_tensor(m.ctx_expert, "state_proj.bias");
        m.action_time_mlp_in_k0 = ggml_get_tensor(m.ctx_expert, "action_time_mlp_in_k0.weight");   // K-split (optional)
        m.action_time_mlp_in_k1 = ggml_get_tensor(m.ctx_expert, "action_time_mlp_in_k1.weight");
        m.action_time_mlp_in_w  = (m.action_time_mlp_in_k0 && m.action_time_mlp_in_k1) ? ggml_get_tensor(m.ctx_expert, "action_time_mlp_in.weight")
                                                                                       : get_tensor(m.ctx_expert, "action_time_mlp_in.weight");
        m.action_time_mlp_in_b  = get_tensor(m.ctx_expert, "action_time_mlp_in.bias");
        m.action_time_mlp_out_w = get_tensor(m.ctx_expert, "action_time_mlp_out.weight");
        m.action_time_mlp_out_b = get_tensor(m.ctx_expert, "action_time_mlp_out.bias");
    }

    return true;
}

void free_pi0_model(pi0_model & m) {
    if (m.buf_pali)   ggml_backend_buffer_free(m.buf_pali);
    if (m.buf_expert) ggml_backend_buffer_free(m.buf_expert);
    if (m.ctx_pali)   ggml_free(m.ctx_pali);
    if (m.ctx_expert) ggml_free(m.ctx_expert);
    m.buf_pali = m.buf_expert = nullptr;
    m.ctx_pali = m.ctx_expert = nullptr;
}

// ============================================================
// Graph fragments
// ============================================================

// PI0_LM_NOCONT=1 (default): skip the layout copies around the attention (fused-QKV slices, KV repack, Q permute).
// The consumers read strided rows directly.
static const int pi0_lm_nocont = []{ const char * e = getenv("PI0_LM_NOCONT"); return e ? atoi(e) : 1; }();

static ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions,
                                int n_head, int head_dim) {
    // x may be a row-sliced view of the fused-QKV output (PI0_LM_NOCONT). Keep its row stride, since ROPE reads
    // rows through nb[1].
    ggml_tensor * reshaped = ggml_is_contiguous(x)
        ? ggml_reshape_3d(ctx, x, head_dim, n_head, x->ne[1])
        : ggml_view_3d(ctx, x, head_dim, n_head, x->ne[1], (size_t) head_dim * ggml_element_size(x), x->nb[1], 0);
    // Gemma uses NEOX-style RoPE (rotate split halves), not the interleaved GGML_ROPE_TYPE_NORMAL.
    ggml_tensor * roped = ggml_rope_ext(ctx, reshaped, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, roped, n_head * head_dim, x->ne[1]);
}

// Whether attention uses ggml_flash_attn_ext (default) or the standard 3-step path. The expert graph builder needs
// it too because the mask dtype differs: the flash-attn kernels read the mask as F16 without a type check, and
// soft_max_ext takes it as F32.
// g_pi0_attn_mode is the pi0_session::attn_mode of the graph being built on this thread (-1 = process default).
// It is thread-local because pipeline stages build their graphs concurrently.
static thread_local int g_pi0_attn_mode = -1;
static bool pi0_use_flash_attn() {
    static const bool v = []{
        const char * env = getenv("PI0_NO_FLASH_ATTN");
        if (env && env[0] == '1') { fprintf(stderr, "PI0: standard attention (no flash)\n"); return false; }
        return true;
    }();
    if (g_pi0_attn_mode == 0) return true;
    if (g_pi0_attn_mode == 1) return false;
    return v;
}

// ---- pi0.5 adaRMS (action expert only) ----
static const int pi0_adarms_table = []{ const char * e = getenv("PI0_ADARMS_TABLE"); return e ? atoi(e) : 1; }();   // 1 (default) = precomputed table, 0 = Dense in the step graph

// cond [EXPERT_N_EMBD, 1] -> Dense -> (scale | shift | gate), each [width, 1], broadcast over the tokens.
struct adarms_mod { ggml_tensor * scale; ggml_tensor * shift; ggml_tensor * gate; };
static adarms_mod adarms_modulation(ggml_context * ctx, ggml_tensor * dense_w, ggml_tensor * dense_b,
                                    ggml_tensor * cond, int width) {
    ggml_tensor * mod = ggml_mul_mat(ctx, dense_w, cond);                                    // [3*width, 1]
    mod = ggml_add(ctx, mod, ggml_reshape_2d(ctx, dense_b, dense_b->ne[0], 1));
    adarms_mod r;
    r.scale = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 0 * width * sizeof(float)));
    r.shift = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 1 * width * sizeof(float)));
    r.gate  = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 2 * width * sizeof(float)));
    return r;
}
// rms(x) * (1 + scale) + shift. The +1 is folded into the bias at load time, so scale already holds (1 + scale).
static ggml_tensor * adarms_apply(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
    return ggml_add(ctx, ggml_mul(ctx, n, scale), shift);
}

// Gemma transformer layer with cache-resident K/V.
// New K/V (post-RoPE, F32) are written to cache rows [write_offset, write_offset + seq_len). ggml_cpy converts them
// to the cache type. Attention reads cache rows [0, attn_kv_len). The prefix pass uses write_offset = 0 and
// attn_kv_len = prefix_len. The expert uses write_offset = prefix_len and attends over the prefix and its suffix.
// The cpy ops are added to graph directly so they survive the expansion from the final output. ggml tracks the
// parent-view alias, so attention runs after the cpy writes.
// adarms_cond or ada_pre_attn set -> pi0.5 action-expert layer: adaRMS norms and gated residuals (x + y * gate).
static ggml_tensor * build_gemma_layer(ggml_context * ctx,
                                       ggml_cgraph * graph,
                                       ggml_tensor * input,
                                       const gemma_layer_weights & w,
                                       ggml_tensor * positions,
                                       ggml_tensor * attn_mask,
                                       int n_head, int n_kv_head, int head_dim,
                                       ggml_tensor * cache_k, ggml_tensor * cache_v,
                                       int write_offset, int attn_kv_len,
                                       ggml_tensor * adarms_cond = nullptr,
                                       const adarms_mod * ada_pre_attn = nullptr,      // PI0_ADARMS_TABLE leaves
                                       const adarms_mod * ada_pre_ffn  = nullptr) {
    const int seq_len = input->ne[1];
    const int kv_dim  = n_kv_head * head_dim;
    const int width   = input->ne[0];
    const bool ada    = (adarms_cond != nullptr) || (ada_pre_attn != nullptr);

    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * x;
    if (ada) {
        adarms_mod md = ada_pre_attn ? *ada_pre_attn
                                     : adarms_modulation(ctx, w.attn_norm_dense_w, w.attn_norm_dense_b, adarms_cond, width);
        x = adarms_apply(ctx, input, md.scale, md.shift);
        attn_gate = md.gate;
    } else {
        x = ggml_rms_norm(ctx, input, 1e-6f);
        x = ggml_mul(ctx, x, w.attn_norm);
    }

    ggml_tensor * q, * k, * v;
    bool rope_done = false;
    // PI0_LM_ONEROPE: one RoPE over the Q and K heads of the fused QKV product (a strided view).
    // -1 (default) = only when the weights are on HTP, whose kernels read strided rows natively.
    static const int pi0_one_rope = []{ const char * e = getenv("PI0_LM_ONEROPE"); return e ? atoi(e) : -1; }();
    const bool layer_on_htp = w.qkv_proj && w.qkv_proj->buffer &&
        strncmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(w.qkv_proj->buffer)), "HTP", 3) == 0;
    const bool one_rope = pi0_lm_nocont && (pi0_one_rope < 0 ? layer_on_htp : pi0_one_rope != 0);
    if (w.qkv_proj) {
        // Fused QKV: one GEMM, then q/k/v are row views of its output.
        const int q_dim = n_head * head_dim;
        ggml_tensor * qkv = ggml_mul_mat(ctx, w.qkv_proj, x);            // [q_dim+2*kv_dim, seq] f32
        q = ggml_view_2d(ctx, qkv, q_dim,  seq_len, qkv->nb[1], 0);
        k = ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t) q_dim          * sizeof(float));
        v = ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t)(q_dim + kv_dim) * sizeof(float));
        // PI0_LM_NOCONT=1 passes the row views to rope and the KV-cache cpy, which read strided rows. 0 copies them.
        if (!pi0_lm_nocont) {
            q = ggml_cont(ctx, q);
            k = ggml_cont(ctx, k);
            v = ggml_cont(ctx, v);
        } else if (kv_dim == head_dim && one_rope) {
            // One rope over the leading [head_dim, n_head + 1] block of the fused projection. The same NEOX rotation
            // applies to every head, so the Q heads and the single K head rotate together and V is untouched.
            // q/k become row views of the rotated block (token stride (n_head+1)*head_dim).
            ggml_tensor * qk = ggml_view_3d(ctx, qkv, head_dim, n_head + n_kv_head, seq_len,
                                            (size_t) head_dim * sizeof(float), qkv->nb[1], 0);
            qk = ggml_rope_ext(ctx, qk, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            q = ggml_view_2d(ctx, qk, q_dim,  seq_len, qk->nb[2], 0);
            k = ggml_view_2d(ctx, qk, kv_dim, seq_len, qk->nb[2], (size_t) q_dim * sizeof(float));
            rope_done = true;
        }
    } else {
        q = ggml_mul_mat(ctx, w.q_proj, x);
        k = ggml_mul_mat(ctx, w.k_proj, x);
        v = ggml_mul_mat(ctx, w.v_proj, x);
    }

    if (!rope_done) {
        q = apply_rope(ctx, q, positions, n_head, head_dim);
        k = apply_rope(ctx, k, positions, n_kv_head, head_dim);
    }

    // ---- Write new K/V into the cache (F32 -> kv_type conversion) ----
    const size_t row_bytes_k = cache_k->nb[1];
    const size_t row_bytes_v = cache_v->nb[1];

    ggml_tensor * k_dst = ggml_view_2d(ctx, cache_k, kv_dim, seq_len,
        row_bytes_k, (size_t) write_offset * row_bytes_k);
    ggml_tensor * v_dst = ggml_view_2d(ctx, cache_v, kv_dim, seq_len,
        row_bytes_v, (size_t) write_offset * row_bytes_v);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k, k_dst));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v, v_dst));

    // ---- Attention reads cache[0..attn_kv_len) ----
    ggml_tensor * attn_k = ggml_view_2d(ctx, cache_k, kv_dim, attn_kv_len, row_bytes_k, 0);
    ggml_tensor * attn_v = ggml_view_2d(ctx, cache_v, kv_dim, attn_kv_len, row_bytes_v, 0);

    q      = ggml_is_contiguous(q) ? ggml_reshape_3d(ctx, q, head_dim, n_head, seq_len)
                                   : ggml_view_3d(ctx, q, head_dim, n_head, seq_len, (size_t) head_dim * sizeof(float), q->nb[1], 0);
    attn_k = ggml_reshape_3d(ctx, attn_k, head_dim, n_kv_head, attn_kv_len);
    attn_v = ggml_reshape_3d(ctx, attn_v, head_dim, n_kv_head, attn_kv_len);

    // ggml_flash_attn_ext expects k/v as [head_dim, kv_len, n_kv_head].
    if (pi0_lm_nocont && n_kv_head == 1) {
        // one KV head: [head_dim, 1, kv_len] -> [head_dim, kv_len, 1] is a relabel of the contiguous cache rows
        attn_k = ggml_reshape_3d(ctx, ggml_reshape_2d(ctx, attn_k, head_dim, attn_kv_len), head_dim, attn_kv_len, 1);
        attn_v = ggml_reshape_3d(ctx, ggml_reshape_2d(ctx, attn_v, head_dim, attn_kv_len), head_dim, attn_kv_len, 1);
    } else {
        attn_k = ggml_cont(ctx, ggml_permute(ctx, attn_k, 0, 2, 1, 3));
        attn_v = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // PI0_NO_FLASH_ATTN=1 or attn_mode selects the standard 3-step attention (mul_mat + soft_max + mul_mat).
    const bool use_flash_attn = pi0_use_flash_attn();

    ggml_tensor * attn_out;
    if (use_flash_attn) {
        // flash_attn_ext requires q as [head_dim, seq_len, n_head], and q is [head_dim, n_head, seq_len], so permute.
        // With PI0_LM_NOCONT the flash kernels read the permuted q as a strided view, without a copy.
        ggml_tensor * q_for_fa = pi0_lm_nocont ? ggml_permute(ctx, q, 0, 2, 1, 3)
                                               : ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // → [head_dim, seq_len, n_head]
        // flash_attn output is [head_dim, n_head, seq_len], as the reshape_2d below expects.
        attn_out = ggml_flash_attn_ext(ctx, q_for_fa, attn_k, attn_v, attn_mask, scale, 0.0f, 0.0f);
    } else {
        // Standard 3-step attention.
        // Layout:
        //   q_std:  [head_dim, seq_len,      n_head]
        //   attn_k: [head_dim, attn_kv_len,  n_kv_head]
        //   attn_v: [head_dim, attn_kv_len,  n_kv_head]
        ggml_tensor * q_std = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

        // V product. Reduction dim (n_kv) needs to be innermost in src0.
        //   v_t = transpose(attn_v) → [attn_kv_len, head_dim, n_kv_head]
        ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, attn_v));

        // PI0_ATTN_COLLAPSE=1: with a single k/v head, fold the n_head query heads into the N dimension and run one
        // GEMM instead of the broadcast batched GEMM. Same math. Requires n_kv_head == 1 and a mask shared by all heads.
        static const bool use_collapse = []{
            const char * env = getenv("PI0_ATTN_COLLAPSE");
            if (env && env[0] == '1') { fprintf(stderr, "PI0: collapsed attention (MQA fold)\n"); return true; }
            return false;
        }();

        ggml_tensor * kqv;
        if (use_collapse && n_kv_head == 1) {
            ggml_tensor * q_coll = ggml_reshape_2d(ctx, q_std, head_dim, (int64_t) seq_len * n_head);
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_coll);    // [n_kv, n_q*n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // The collapsed kq is [n_kv, seq_len*n_head] with column j = (query j%seq_len, head j/seq_len), so the
            // [n_kv, seq_len] mask is tiled n_head times along N. The prefix has no mask.
            ggml_tensor * mask_coll = attn_mask ? ggml_repeat(ctx, attn_mask, kq) : nullptr;
            kq = ggml_soft_max_ext(ctx, kq, mask_coll, scale, 0.0f);
            ggml_tensor * kqv_coll = ggml_mul_mat(ctx, v_t, kq);     // [head_dim, n_q*n_head]
            kqv = ggml_reshape_3d(ctx, kqv_coll, head_dim, seq_len, n_head);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_std);     // [n_kv, n_q, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // No causal mask. The prefix attention is bidirectional (attn_mask is NULL there).
            kq = ggml_soft_max_ext(ctx, kq, attn_mask, scale, 0.0f);
            kqv = ggml_mul_mat(ctx, v_t, kq);                        // [head_dim, seq_len, n_head]
        }

        attn_out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    }
    attn_out = ggml_reshape_2d(ctx, attn_out, n_head * head_dim, seq_len);

    ggml_tensor * attn_proj = ggml_mul_mat(ctx, w.o_proj, attn_out);
    input = ada ? ggml_add(ctx, input, ggml_mul(ctx, attn_proj, attn_gate))   // gated residual (adaRMS)
                : ggml_add(ctx, input, attn_proj);

    ggml_tensor * ffn_gate_mod = nullptr;
    if (ada) {
        adarms_mod md = ada_pre_ffn ? *ada_pre_ffn
                                    : adarms_modulation(ctx, w.ffn_norm_dense_w, w.ffn_norm_dense_b, adarms_cond, width);
        x = adarms_apply(ctx, input, md.scale, md.shift);
        ffn_gate_mod = md.gate;
    } else {
        x = ggml_rms_norm(ctx, input, 1e-6f);
        x = ggml_mul(ctx, x, w.ffn_norm);
    }

    // ffn_down, optionally K-split (blk.N.ffn_down_k*.weight, tools/gguf_tools.py split-k): one GEMM per K slice over
    // a column view of the hidden, with the f32 partial sums added in order (plus the residual when folded).
    // The residual or accumulator is the first operand of each add, so gallocr places the sum in its buffer.
    auto ffn_down = [&](ggml_tensor * h, ggml_tensor * residual) -> ggml_tensor * {
        if (w.down_n_split >= 2) {
            const int    n  = w.down_n_split;
            const int64_t kh = w.down_k[0]->ne[0];
            GGML_ASSERT(kh * n == h->ne[0]);
            ggml_tensor * acc = residual;
            for (int i = 0; i < n; i++) {
                GGML_ASSERT(w.down_k[i]->ne[0] == kh);
                ggml_tensor * hi = ggml_view_2d(ctx, h, kh, h->ne[1], h->nb[1], (size_t) i * kh * ggml_element_size(h));
                ggml_tensor * pi = ggml_mul_mat(ctx, w.down_k[i], hi);
                acc = acc ? ggml_add(ctx, acc, pi) : pi;
            }
            return acc;
        }
        GGML_ASSERT(w.down_proj);
        ggml_tensor * p = ggml_mul_mat(ctx, w.down_proj, h);
        return residual ? ggml_add(ctx, residual, p) : p;
    };
    // PI0_FFN_GU=1 (default): use the interleaved gate/up weight (blk.N.ffn_gateup.weight, tools/gguf_tools.py
    // fuse-gateup). HTP fuses the matmul and GEGLU (HTP_OP_MUL_MAT_GEGLU). It is used on HTP whenever the GGUF has
    // ffn_gateup, and on other backends only when the GGUF has no separate gate/up (the graph then runs unfused).
    static const int   pi0_ffn_gu_          = []{ const char * e = getenv("PI0_FFN_GU");            return e ? atoi(e) : 1; }();
    static const int   pi0_ffn_f16_gu       = []{ const char * e = getenv("PI0_FFN_F16");           return e ? atoi(e) : 0; }();
    const ggml_tensor * ffn_w0 = w.gateup_proj ? w.gateup_proj : w.gate_proj;
    const bool ffn_on_htp_gu = ffn_w0 && ffn_w0->buffer &&
        strncmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(ffn_w0->buffer)), "HTP", 3) == 0;
    const bool use_gu = pi0_ffn_gu_ && w.gateup_proj && (ffn_on_htp_gu || !w.gate_proj);
    // The NPU layouts (interleaved gate/up only, K-split ffn_down) feed strided views to MUL_MAT. HTP and CPU accept
    // them, OpenCL does not. A GPU stage needs the GPU weight set (-m or --gpu-model-dir).
    GGML_ASSERT((ffn_on_htp_gu || !((use_gu && !w.gate_proj) || w.down_n_split >= 2) ||
                 (ffn_w0 && ffn_w0->buffer && ggml_backend_buffer_is_host(ffn_w0->buffer))) &&
                "NPU-layout FFN weights (ffn_gateup only / K-split ffn_down) run on HTP or CPU only; pass a GPU set");
    // PI0_FFN_FOLD_RESIDUAL: add the layer residual inside ffn_down (fused MUL_MAT+ADD on HTP). -1 (default) = HTP
    // only. Never with adaRMS, whose gating needs the bare FFN output.
    static const int pi0_fold_res = []{ const char * e = getenv("PI0_FFN_FOLD_RESIDUAL"); return e ? atoi(e) : -1; }();
    const bool fold_residual = !ada && (pi0_fold_res < 0 ? ffn_on_htp_gu : pi0_fold_res != 0);
    ggml_tensor * ffn = nullptr;
    if (use_gu) {
        const int64_t ff = w.gateup_proj->ne[1] / 2;
        ggml_tensor * gu = ggml_mul_mat(ctx, w.gateup_proj, x);                                        // [2*ff, seq] (not materialised on HTP)
        ggml_tensor * g  = ggml_view_3d(ctx, gu, 32, ff / 32, seq_len, 64 * sizeof(float), gu->nb[1], 0);
        ggml_tensor * u  = ggml_view_3d(ctx, gu, 32, ff / 32, seq_len, 64 * sizeof(float), gu->nb[1], 32 * sizeof(float));
        ffn = ggml_reshape_2d(ctx, ggml_geglu_split(ctx, g, u), ff, seq_len);
        if (pi0_ffn_f16_gu && ffn_on_htp_gu) {   // HTP only. The CPU q4_0 matmul asserts on an F16 src1.
            ffn = ggml_cast(ctx, ffn, GGML_TYPE_F16);
        }
        ffn = ffn_down(ffn, fold_residual ? input : nullptr);
    } else {
    GGML_ASSERT(w.gate_proj && w.up_proj && "this GGUF only carries ffn_gateup: run it on HTP with PI0_FFN_GU=1");
    ggml_tensor * gate = ggml_mul_mat(ctx, w.gate_proj, x);
    ggml_tensor * up   = ggml_mul_mat(ctx, w.up_proj,   x);
    // Gemma FFN is GeGLU with the tanh GELU approximation, not SwiGLU.
    // PI0_FFN_F16=1: keep the FFN hidden in F16. The residual stream stays F32 because Gemma overflows F16.
    static const int pi0_ffn_f16 = []{ const char * e = getenv("PI0_FFN_F16"); return e ? atoi(e) : 0; }();
    // Only HTP has the F16 GEGLU and F16-activation matmul. The CPU q4_0 matmul asserts on an F16 src1, so decide per
    // layer from the weights' buffer type.
    const bool ffn_on_htp = w.gate_proj->buffer &&
        strncmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(w.gate_proj->buffer)), "HTP", 3) == 0;
    if (pi0_ffn_f16 && ffn_on_htp) {
        gate = ggml_cast(ctx, gate, GGML_TYPE_F16);
        up   = ggml_cast(ctx, up,   GGML_TYPE_F16);
    }
    ffn = ggml_geglu_split(ctx, gate, up);
    ffn = ffn_down(ffn, fold_residual ? input : nullptr);
    }  // end of the separate gate/up path

    if (fold_residual) {
        return ffn;   // residual already added
    }
    input = ada ? ggml_add(ctx, input, ggml_mul(ctx, ffn, ffn_gate_mod))
                : ggml_add(ctx, input, ffn);
    return input;
}

// pi0 suffix input [EXPERT_N_EMBD, 51]: the state token and the action tokens after the action_time MLP.
static ggml_tensor * build_suffix_tokens(ggml_context * ctx,
                                         ggml_cgraph * graph,
                                         pi0_model & m,
                                         ggml_tensor * state_in,       // [PI0_ACTION_DIM, 1]
                                         ggml_tensor * noisy_in,       // [PI0_ACTION_DIM, 50]
                                         ggml_tensor * time_in,        // [EXPERT_N_EMBD, 50] (concat path) or [EXPERT_N_EMBD, 1] (K-split path)
                                         ggml_tensor * suffix_buf) {   // pinned [EXPERT_N_EMBD, 51] or null (-> CONCAT)
    // state token: [EXPERT_N_EMBD, 1]
    ggml_tensor * state_tok = ggml_mul_mat(ctx, m.state_proj_w, state_in);
    state_tok = add_bias(ctx, state_tok, m.state_proj_b);

    // action tokens (pre-MLP): [EXPERT_N_EMBD, 50]
    ggml_tensor * action_tok = ggml_mul_mat(ctx, m.action_in_proj_w, noisy_in);
    action_tok = add_bias(ctx, action_tok, m.action_in_proj_b);

    // MLP: Linear([action_tok | time_emb]) -> SiLU -> Linear
    ggml_tensor * h;
    if (m.action_time_mlp_in_k0 && m.action_time_mlp_in_k1 && time_in->ne[1] == 1) {
        // K-split weight: W_in [a | t] = W_k0 a + W_k1 t. The time part is one GEMV on sincos(t) [EXPERT_N_EMBD, 1]
        // (+ bias), broadcast-added to the action GEMM, so no CONCAT is needed.
        ggml_tensor * tvec = ggml_mul_mat(ctx, m.action_time_mlp_in_k1, time_in);                               // [EXPERT_N_EMBD, 1]
        tvec = ggml_add(ctx, tvec, ggml_reshape_2d(ctx, m.action_time_mlp_in_b, EXPERT_N_EMBD, 1));
        h = ggml_add(ctx, ggml_mul_mat(ctx, m.action_time_mlp_in_k0, action_tok), tvec);                        // broadcast over the 50 tokens
    } else {
        GGML_ASSERT(m.action_time_mlp_in_w && "this GGUF only carries the K-split action_time_mlp_in");
        // concat with time embedding along feature dim 0 -> [2*EXPERT_N_EMBD, 50]
        ggml_tensor * concat = ggml_concat(ctx, action_tok, time_in, 0);
        h = ggml_mul_mat(ctx, m.action_time_mlp_in_w, concat);
        h = add_bias(ctx, h, m.action_time_mlp_in_b);
    }
    h = ggml_silu(ctx, h);

    ggml_tensor * action_emb = ggml_mul_mat(ctx, m.action_time_mlp_out_w, h);
    action_emb = add_bias(ctx, action_emb, m.action_time_mlp_out_b);

    // Prepend the state token along the sequence dim -> [EXPERT_N_EMBD, 51]. With suffix_buf, two column copies
    // replace the CONCAT. suffix_buf must not come from gallocr, which recycles a tensor written only through views.
    if (suffix_buf) {
        ggml_build_forward_expand(graph, ggml_cpy(ctx, state_tok,
            ggml_view_2d(ctx, suffix_buf, EXPERT_N_EMBD, 1, suffix_buf->nb[1], 0)));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, action_emb,
            ggml_view_2d(ctx, suffix_buf, EXPERT_N_EMBD, PI0_ACTION_HORIZON, suffix_buf->nb[1], suffix_buf->nb[1])));
        return suffix_buf;
    }
    return ggml_concat(ctx, state_tok, action_emb, 1);
}

// ============================================================
// KV cache storage
// ============================================================

static void destroy_kv_storage(pi0_session & s) {
    if (s.kv_buf) { ggml_backend_buffer_free(s.kv_buf); s.kv_buf = nullptr; }
    if (s.kv_ctx) { ggml_free(s.kv_ctx);                s.kv_ctx = nullptr; }
    for (int il = 0; il < PI0_N_LAYER; il++) {
        s.cache_k[il] = nullptr;
        s.cache_v[il] = nullptr;
    }
    s.kv_alloc_prefix_len = 0;
    s.kv_alloc_type       = GGML_TYPE_F32;
}

// Allocate per-layer K/V cache of shape [kv_dim, prefix_len + suffix_len] in s.kv_type.
// No-op if existing storage already matches.
static bool ensure_kv_storage(pi0_session & s, ggml_backend_t backend, int prefix_len) {
    const int suffix_len = PI0_ACTION_HORIZON + 1;
    const int total_kv   = prefix_len + suffix_len;

    if (s.kv_buf
        && s.kv_alloc_prefix_len == prefix_len
        && s.kv_alloc_type       == s.kv_type) {
        return true;
    }

    destroy_kv_storage(s);

    const int kv_dim = PALI_N_KV_HEAD * PALI_HEAD_DIM; // = EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM = 256

    size_t ctx_size = ggml_tensor_overhead() * (2 * PI0_N_LAYER + 8);
    ggml_init_params params = { ctx_size, nullptr, true };
    s.kv_ctx = ggml_init(params);

    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[64];
        s.cache_k[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_k_%d", il);
        ggml_set_name(s.cache_k[il], name);

        s.cache_v[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_v_%d", il);
        ggml_set_name(s.cache_v[il], name);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    s.kv_buf = ggml_backend_alloc_ctx_tensors_from_buft(s.kv_ctx, buft);
    if (!s.kv_buf) {
        LOG_ERR("Failed to allocate KV cache buffer (type=%s, prefix_len=%d)\n",
                ggml_type_name(s.kv_type), prefix_len);
        ggml_free(s.kv_ctx);
        s.kv_ctx = nullptr;
        return false;
    }

    s.kv_alloc_prefix_len = prefix_len;
    s.kv_alloc_type       = s.kv_type;
    LOG_INF("KV cache allocated: %d layers × [%d, %d] %s (%.2f MB)\n",
            PI0_N_LAYER, kv_dim, total_kv, ggml_type_name(s.kv_type),
            2.0 * PI0_N_LAYER * ggml_nbytes(s.cache_k[0]) / (1024.0 * 1024.0));
    return true;
}

// ============================================================
// Forward passes
// ============================================================

static void destroy_prefix_graph(pi0_session & s) {
    if (s.pgraph_alloc) { ggml_gallocr_free(s.pgraph_alloc); s.pgraph_alloc = nullptr; }
    if (s.pgraph_ctx)   { ggml_free(s.pgraph_ctx);          s.pgraph_ctx   = nullptr; }
    s.pgraph = nullptr; s.pgraph_input = nullptr; s.pgraph_positions = nullptr; s.pgraph_out = nullptr;
    s.pgraph_prefix_len = 0; s.pgraph_backend = nullptr; s.pgraph_kv_ctx = nullptr;
    s.pgraph_kv_type = GGML_TYPE_F32; s.pgraph_cache_k0 = nullptr;
    s.pgraph_il_from = 0; s.pgraph_il_to = PI0_N_LAYER;
}

bool run_paligemma_prefix(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                          const float * embeddings, int prefix_len) {
    return run_paligemma_prefix_range(m, s, backend, embeddings, prefix_len, 0, PI0_N_LAYER, nullptr);
}

bool run_paligemma_prefix_range(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                                const float * embeddings, int prefix_len, int il_from, int il_to,
                                std::vector<float> * out_hidden) {
    if (il_from < 0 || il_to > PI0_N_LAYER || il_from >= il_to) {
        LOG_ERR("run_paligemma_prefix_range: bad layer range [%d, %d)\n", il_from, il_to);
        return false;
    }
    g_pi0_attn_mode = s.attn_mode;
    if (!ensure_kv_storage(s, backend, prefix_len)) {
        return false;
    }
    s.prefix_len = prefix_len;

    static const int graph_cache = []{ const char * e = getenv("PI0_PREFIX_GRAPH_CACHE"); return e ? atoi(e) : 1; }();

    // s.pgraph_kv_ctx alone does not identify the KV storage, since a re-created context can get the same address.
    // Also compare the dtype and the cache tensor so a graph with views into a freed buffer is never reused.
    const bool reuse = graph_cache && s.pgraph && s.pgraph_prefix_len == prefix_len
                    && s.pgraph_il_from == il_from && s.pgraph_il_to == il_to
                    && s.pgraph_backend == backend && s.pgraph_kv_ctx == s.kv_ctx
                    && s.pgraph_kv_type == s.kv_type && s.pgraph_cache_k0 == s.cache_k[0];
    if (!reuse) {
        destroy_prefix_graph(s);

        size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
        ggml_init_params params = { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PALI_N_EMBD, prefix_len);
        ggml_set_name(input, "prefix_input");
        ggml_set_input(input);

        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_len);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        // The Gemma sqrt(hidden) embedding scale applies to text tokens only, not to image features.
        // The caller applies it when assembling the prefix.
        ggml_tensor * cur = input;

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);

        // Each layer writes K/V to cache_k/v[il][0..prefix_len) and self-attention reads those rows.
        for (int il = il_from; il < il_to; il++) {
            cur = build_gemma_layer(ctx, graph, cur, m.pali_layers[il], positions, nullptr,
                PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM,
                s.cache_k[il], s.cache_v[il],
                /*write_offset=*/0, /*attn_kv_len=*/prefix_len);
        }

        if (il_to == PI0_N_LAYER) {
            cur = ggml_rms_norm(ctx, cur, 1e-6f);
            cur = ggml_mul(ctx, cur, m.pali_final_norm);
        } else {
            // partial range: output the residual for the next device. The copy makes it a real (non-view) node.
            cur = ggml_cont(ctx, cur);
        }
        ggml_set_name(cur, "prefix_out");
        ggml_set_output(cur);
        ggml_build_forward_expand(graph, cur);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            LOG_ERR("Failed to allocate prefix graph\n");
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        s.pgraph_ctx = ctx;   s.pgraph = graph;   s.pgraph_alloc = alloc;
        s.pgraph_input = input; s.pgraph_positions = positions; s.pgraph_out = cur;
        s.pgraph_prefix_len = prefix_len; s.pgraph_backend = backend; s.pgraph_kv_ctx = s.kv_ctx;
        s.pgraph_kv_type = s.kv_type; s.pgraph_cache_k0 = s.cache_k[0];
        s.pgraph_il_from = il_from; s.pgraph_il_to = il_to;
    }
    ggml_tensor * input     = s.pgraph_input;
    ggml_tensor * positions = s.pgraph_positions;
    ggml_tensor * cur       = s.pgraph_out;
    ggml_cgraph * graph     = s.pgraph;

    ggml_backend_tensor_set(input, embeddings, 0, sizeof(float) * PALI_N_EMBD * prefix_len);
    std::vector<int32_t> pos_data(prefix_len);
    for (int i = 0; i < prefix_len; i++) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, sizeof(int32_t) * prefix_len);

    const ggml_status hst = ggml_backend_graph_compute(backend, graph);
    if (hst != GGML_STATUS_SUCCESS) {
        LOG_ERR("Prefix compute failed\n");
        destroy_prefix_graph(s);
        return false;
    }
    if (out_hidden && il_to < PI0_N_LAYER) {
        out_hidden->resize((size_t) PALI_N_EMBD * prefix_len);
        ggml_backend_tensor_get(cur, out_hidden->data(), 0, sizeof(float) * out_hidden->size());
    }

    // The KV stays in s.cache_k/v on the backend. The graph is kept unless PI0_PREFIX_GRAPH_CACHE=0.
    if (!graph_cache) {
        destroy_prefix_graph(s);
    }
    return true;
}

// ---- Persistent expert graph (built once per (prefix_len, kv_type), reused across steps) ----

static void destroy_expert_graph(pi0_session & s) {
    if (s.graph_alloc)     { ggml_gallocr_free(s.graph_alloc);           s.graph_alloc     = nullptr; }
    if (s.graph_ctx)       { ggml_free(s.graph_ctx);                    s.graph_ctx       = nullptr; }
    if (s.graph_const_buf) { ggml_backend_buffer_free(s.graph_const_buf); s.graph_const_buf = nullptr; }
    if (s.graph_const_ctx) { ggml_free(s.graph_const_ctx);              s.graph_const_ctx = nullptr; }
    s.graph = nullptr;
    s.graph_prefix_len = 0;
    s.graph_kv_type    = GGML_TYPE_F32;
    s.in_state = s.in_noisy = s.in_time = s.in_positions = nullptr;
    s.in_attn_mask = s.out_v_t = s.suffix_buf = nullptr;
    s.in_hidden = s.out_hidden = nullptr;
    s.graph_il_from = 0; s.graph_il_to = PI0_N_LAYER;
    for (auto & site : s.ada_leaf) for (auto & t : site) t = nullptr;
    s.ada_table.clear(); s.ada_table_step = -1;
}

// Context for the expert graph's constant-across-steps inputs (see pi0_session::graph_const_ctx). Its tensors
// are allocated with alloc_const_inputs before the graph is built, so gallocr treats them as external.
static ggml_context * new_const_input_ctx(pi0_session & s) {
    ggml_init_params params = { ggml_tensor_overhead() * 128, nullptr, true };   // positions/mask + up to 111 adaRMS leaves
    s.graph_const_ctx = ggml_init(params);
    return s.graph_const_ctx;
}
static bool alloc_const_inputs(pi0_session & s, ggml_backend_t backend) {
    s.graph_const_buf = ggml_backend_alloc_ctx_tensors(s.graph_const_ctx, backend);
    if (!s.graph_const_buf) { LOG_ERR("Failed to allocate the expert graph's constant inputs\n"); return false; }
    return true;
}

// pi0.5 suffix: action_in_proj(noisy) only, no state token. The adaRMS conditioning vector
// cond = swish(time_mlp_out(swish(time_mlp_in(sincos(t))))) is computed in the graph from time_in [EXPERT_N_EMBD, 1].
static ggml_tensor * build_adarms_cond(ggml_context * ctx, pi0_model & m, ggml_tensor * time_in) {
    ggml_tensor * h = ggml_mul_mat(ctx, m.time_mlp_in_w, time_in);
    h = add_bias(ctx, h, m.time_mlp_in_b);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, m.time_mlp_out_w, h);
    h = add_bias(ctx, h, m.time_mlp_out_b);
    return ggml_silu(ctx, h);                                             // [EXPERT_N_EMBD, 1]
}
static ggml_tensor * build_suffix_tokens_pi05(ggml_context * ctx, pi0_model & m,
                                              ggml_tensor * noisy_in, ggml_tensor * time_in,
                                              ggml_tensor ** cond_out) {
    ggml_tensor * action_tok = ggml_mul_mat(ctx, m.action_in_proj_w, noisy_in);
    action_tok = add_bias(ctx, action_tok, m.action_in_proj_b);           // [EXPERT_N_EMBD, 50]
    if (cond_out) *cond_out = build_adarms_cond(ctx, m, time_in);         // nullptr = PI0_ADARMS_TABLE (leaves instead)
    return action_tok;
}

// PI0_ADARMS_TABLE: run cond -> 37 x Dense once per timestep on the same backend and keep the
// 37 x [3*EXPERT_N_EMBD] results per step on the host. Same values as the per-step graph.
static bool compute_adarms_table(pi0_model & m, pi0_session & s, ggml_backend_t backend) {
    const int S = pi0_session::ADA_SITES, W = EXPERT_N_EMBD;
    ggml_init_params ip = { ggml_tensor_overhead() * 1024 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, 1);
    ggml_set_name(tin, "ada_tab_time_in"); ggml_set_input(tin);
    ggml_tensor * cond = build_adarms_cond(ctx, m, tin);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 2048, false);
    std::vector<ggml_tensor *> outs(S);
    for (int i = 0; i < S; i++) {
        ggml_tensor * dw, * db;
        if (i < 2 * PI0_N_LAYER) {
            auto & L = m.expert_layers[i / 2];
            dw = (i % 2 == 0) ? L.attn_norm_dense_w : L.ffn_norm_dense_w;
            db = (i % 2 == 0) ? L.attn_norm_dense_b : L.ffn_norm_dense_b;
        } else { dw = m.expert_final_norm_dense_w; db = m.expert_final_norm_dense_b; }
        ggml_tensor * mod = ggml_mul_mat(ctx, dw, cond);                                   // [3*W, 1]
        mod = ggml_add(ctx, mod, ggml_reshape_2d(ctx, db, db->ne[0], 1));                  // bias already holds the +1 of (1+scale)
        ggml_set_output(mod); outs[i] = mod;
        ggml_build_forward_expand(g, mod);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(ga, g) || !ggml_gallocr_alloc_graph(ga, g)) {
        LOG_ERR("adaRMS table: graph allocation failed\n"); ggml_gallocr_free(ga); ggml_free(ctx); return false;
    }
    s.ada_table.assign((size_t) PI0_NUM_STEPS * S * 3 * W, 0.0f);
    std::vector<float> te(W);
    for (int step = 0; step < PI0_NUM_STEPS; step++) {
        const float t = 1.0f + step * (-1.0f / PI0_NUM_STEPS);                             // the fixed flow-matching schedule
        sincos_posemb(t, W, te.data());
        ggml_backend_tensor_set(tin, te.data(), 0, sizeof(float) * W);
        if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
            LOG_ERR("adaRMS table: compute failed at step %d\n", step); ggml_gallocr_free(ga); ggml_free(ctx); return false;
        }
        for (int i = 0; i < S; i++)
            ggml_backend_tensor_get(outs[i], &s.ada_table[((size_t) step * S + i) * 3 * W], 0, sizeof(float) * 3 * W);
    }
    ggml_gallocr_free(ga); ggml_free(ctx);
    s.ada_table_step = -1;
    LOG_INF("adaRMS: precomputed %d sites x %d steps of (1+scale|shift|gate) once (%.1f MB); per-step Dense GEMVs removed\n",
            S, PI0_NUM_STEPS, s.ada_table.size() * sizeof(float) / 1e6);
    return true;
}

static bool build_expert_graph_pi05(pi0_model & m, pi0_session & s, ggml_backend_t backend, int prefix_len) {
    const int suffix_len = PI0_ACTION_HORIZON;                 // 50 action tokens, no state token
    const int total_kv   = prefix_len + suffix_len;

    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    s.graph_ctx = ggml_init(params);
    ggml_context * ctx = s.graph_ctx;

    s.in_state = nullptr;
    s.in_noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI0_ACTION_DIM, PI0_ACTION_HORIZON);
    ggml_set_name(s.in_noisy, "noisy_in"); ggml_set_input(s.in_noisy);
    s.in_time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, 1);   // sincos(t), the MLP runs in-graph
    ggml_set_name(s.in_time, "time_in"); ggml_set_input(s.in_time);
    s.in_positions = ggml_new_tensor_1d(new_const_input_ctx(s), GGML_TYPE_I32, suffix_len);   // uploaded once
    ggml_set_name(s.in_positions, "suffix_positions"); ggml_set_input(s.in_positions);
    const bool ada_table = pi0_adarms_table != 0;
    if (ada_table) {
        // One [EXPERT_N_EMBD, 1] leaf per (site, scale|shift|gate), in the constant-input buffer (not the gallocr
        // arena). They are re-uploaded whenever the step changes.
        static const char * nm[3] = { "scale", "shift", "gate" };
        for (int i = 0; i < pi0_session::ADA_SITES; i++)
            for (int j = 0; j < 3; j++) {
                s.ada_leaf[i][j] = ggml_new_tensor_2d(s.graph_const_ctx, GGML_TYPE_F32, EXPERT_N_EMBD, 1);
                ggml_format_name(s.ada_leaf[i][j], "ada_%d_%s", i, nm[j]); ggml_set_input(s.ada_leaf[i][j]);
            }
    }
    if (!alloc_const_inputs(s, backend)) { destroy_expert_graph(s); return false; }
    s.in_attn_mask = nullptr;   // pi0.5 suffix attention is fully visible (prefix + all action tokens)

    s.graph = ggml_new_graph_custom(ctx, 16384, false);

    const int il_from = s.expert_il_from, il_to = s.expert_il_to;
    ggml_tensor * cond = nullptr;
    ggml_tensor * cur  = nullptr;
    if (il_from == 0 || !ada_table) {
        cur = build_suffix_tokens_pi05(ctx, m, s.in_noisy, s.in_time, ada_table ? nullptr : &cond);
        // Pin cond (read by all 37 norms) and the suffix input with ggml_set_output. Otherwise gallocr reuses their
        // buffers and later steps of the persistent graph read garbage.
        if (cond) { ggml_set_name(cond, "adarms_cond"); ggml_set_output(cond); }
        ggml_set_name(cur,  "suffix_tok");  ggml_set_output(cur);
    }
    if (il_from > 0) {
        // Second part of a split step: start from the other device's residual. Without the adaRMS table the suffix
        // prep above is still built for cond, and its token output is unused.
        s.in_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, suffix_len);
        ggml_set_name(s.in_hidden, "hidden_in"); ggml_set_input(s.in_hidden);
        cur = s.in_hidden;
    }

    auto leaf = [&](int site) { return adarms_mod{ s.ada_leaf[site][0], s.ada_leaf[site][1], s.ada_leaf[site][2] }; };
    for (int il = il_from; il < il_to; il++) {
        adarms_mod pa = ada_table ? leaf(2 * il) : adarms_mod{}, pf = ada_table ? leaf(2 * il + 1) : adarms_mod{};
        cur = build_gemma_layer(ctx, s.graph, cur, m.expert_layers[il],
            s.in_positions, /*attn_mask=*/nullptr,
            EXPERT_N_HEAD, EXPERT_N_KV_HEAD, EXPERT_HEAD_DIM,
            s.cache_k[il], s.cache_v[il],
            /*write_offset=*/prefix_len, /*attn_kv_len=*/total_kv, /*adarms_cond=*/cond,
            ada_table ? &pa : nullptr, ada_table ? &pf : nullptr);
    }

    if (il_to < PI0_N_LAYER) {
        // first part of a split step: hand the residual to the other device
        s.out_hidden = ggml_cont(ctx, cur);
        ggml_set_name(s.out_hidden, "hidden_out"); ggml_set_output(s.out_hidden);
        ggml_build_forward_expand(s.graph, s.out_hidden);
    } else {
    // Final adaRMS norm (gate unused), then action_out_proj over all 50 tokens.
    {
        adarms_mod md = ada_table ? leaf(2 * PI0_N_LAYER)
                                  : adarms_modulation(ctx, m.expert_final_norm_dense_w, m.expert_final_norm_dense_b, cond, EXPERT_N_EMBD);
        cur = adarms_apply(ctx, cur, md.scale, md.shift);
    }
    s.out_v_t = ggml_mul_mat(ctx, m.action_out_proj_w, cur);
    s.out_v_t = add_bias(ctx, s.out_v_t, m.action_out_proj_b);
    ggml_set_name(s.out_v_t, "v_t"); ggml_set_output(s.out_v_t);
    ggml_build_forward_expand(s.graph, s.out_v_t);
    }

    s.graph_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(s.graph_alloc, s.graph) || !ggml_gallocr_alloc_graph(s.graph_alloc, s.graph)) {
        LOG_ERR("Failed to allocate persistent pi0.5 expert graph\n");
        destroy_expert_graph(s);
        return false;
    }
    std::vector<int32_t> pos_data(suffix_len);
    for (int i = 0; i < suffix_len; i++) pos_data[i] = prefix_len + i;
    ggml_backend_tensor_set(s.in_positions, pos_data.data(), 0, sizeof(int32_t) * suffix_len);
    if (ada_table && !compute_adarms_table(m, s, backend)) { destroy_expert_graph(s); return false; }

    s.graph_prefix_len = prefix_len;
    s.graph_kv_type    = s.kv_type;
    s.graph_il_from = il_from; s.graph_il_to = il_to;
    return true;
}

static bool build_expert_graph(pi0_model & m, pi0_session & s, ggml_backend_t backend, int prefix_len) {
    destroy_expert_graph(s);
    if (m.pi05) {
        return build_expert_graph_pi05(m, s, backend, prefix_len);
    }

    const int suffix_len = PI0_ACTION_HORIZON + 1;            // 51 = state + 50 actions
    const int total_kv   = prefix_len + suffix_len;

    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    s.graph_ctx = ggml_init(params);
    ggml_context * ctx = s.graph_ctx;

    // ---- Inputs ----
    s.in_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI0_ACTION_DIM, 1);
    ggml_set_name(s.in_state, "state_in");
    ggml_set_input(s.in_state);

    s.in_noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI0_ACTION_DIM, PI0_ACTION_HORIZON);
    ggml_set_name(s.in_noisy, "noisy_in");
    ggml_set_input(s.in_noisy);

    // sincos(t): one column with the K-split action_time_mlp_in (GEMV + broadcast add), otherwise broadcast to
    // all 50 positions on the host so the concat path's time MLP runs as one batched matmul.
    const bool time_ksplit = m.action_time_mlp_in_k0 && m.action_time_mlp_in_k1;
    s.in_time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, time_ksplit ? 1 : PI0_ACTION_HORIZON);
    ggml_set_name(s.in_time, "time_in");
    ggml_set_input(s.in_time);

    // positions and mask are uploaded once and read by every step -> own buffer (pi0_session::graph_const_ctx)
    ggml_context * cctx = new_const_input_ctx(s);
    s.in_positions = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, suffix_len);
    ggml_set_name(s.in_positions, "suffix_positions");
    ggml_set_input(s.in_positions);

    // The flash-attn kernels read the mask as F16 without a type check, so flash attention needs an F16 mask.
    // soft_max_ext takes an F32 mask.
    s.in_attn_mask = ggml_new_tensor_2d(cctx, pi0_use_flash_attn() ? GGML_TYPE_F16 : GGML_TYPE_F32, total_kv, suffix_len);
    ggml_set_name(s.in_attn_mask, "attn_mask");
    ggml_set_input(s.in_attn_mask);
    // state + action tokens are copied into this pinned buffer (no CONCAT)
    s.suffix_buf = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, EXPERT_N_EMBD, suffix_len);
    ggml_set_name(s.suffix_buf, "suffix_tokens");
    if (!alloc_const_inputs(s, backend)) { destroy_expert_graph(s); return false; }

    s.graph = ggml_new_graph_custom(ctx, 16384, false);

    // ---- Suffix tokens (state_proj, action_in_proj, time MLP), or the other device's residual for the second
    //      part of a split step ----
    const int il_from = s.expert_il_from, il_to = s.expert_il_to;
    ggml_tensor * cur;
    if (il_from == 0) {
        cur = build_suffix_tokens(ctx, s.graph, m, s.in_state, s.in_noisy, s.in_time, s.suffix_buf);
    } else {
        s.in_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, suffix_len);
        ggml_set_name(s.in_hidden, "hidden_in"); ggml_set_input(s.in_hidden);
        cur = s.in_hidden;
    }

    // ---- expert layers [il_from, il_to): new K/V go to cache[prefix_len:total_kv], attention reads cache[0:total_kv] ----
    for (int il = il_from; il < il_to; il++) {
        cur = build_gemma_layer(ctx, s.graph, cur, m.expert_layers[il],
            s.in_positions, s.in_attn_mask,
            EXPERT_N_HEAD, EXPERT_N_KV_HEAD, EXPERT_HEAD_DIM,
            s.cache_k[il], s.cache_v[il],
            /*write_offset=*/prefix_len, /*attn_kv_len=*/total_kv);
    }

    if (il_to < PI0_N_LAYER) {
        s.out_hidden = ggml_cont(ctx, cur);
        ggml_set_name(s.out_hidden, "hidden_out"); ggml_set_output(s.out_hidden);
        ggml_build_forward_expand(s.graph, s.out_hidden);
    } else {
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.expert_final_norm);

    // Drop the leading state token and keep the 50 action positions for action_out_proj.
    ggml_tensor * action_hidden = ggml_view_2d(ctx, cur, EXPERT_N_EMBD, PI0_ACTION_HORIZON,
        cur->nb[1], cur->nb[1] * 1);

    s.out_v_t = ggml_mul_mat(ctx, m.action_out_proj_w, action_hidden);
    s.out_v_t = add_bias(ctx, s.out_v_t, m.action_out_proj_b);
    ggml_set_name(s.out_v_t, "v_t");
    ggml_set_output(s.out_v_t);

    ggml_build_forward_expand(s.graph, s.out_v_t);
    }
    s.graph_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(s.graph_alloc, s.graph) || !ggml_gallocr_alloc_graph(s.graph_alloc, s.graph)) {
        LOG_ERR("Failed to allocate persistent expert graph\n");
        destroy_expert_graph(s);
        return false;
    }

    // ---- Upload constant-across-steps inputs once ----

    // positions: prefix_len .. prefix_len + suffix_len - 1
    std::vector<int32_t> pos_data(suffix_len);
    for (int i = 0; i < suffix_len; i++) pos_data[i] = prefix_len + i;
    ggml_backend_tensor_set(s.in_positions, pos_data.data(), 0, sizeof(int32_t) * suffix_len);

    // Attention mask. The prefix is always visible. The state token attends to itself only.
    // Action tokens attend to the state and all actions.
    std::vector<float> mask_data(total_kv * suffix_len, 0.0f);
    const float NEG_INF = -INFINITY;
    for (int srow = 0; srow < suffix_len; srow++) {
        for (int kv = 0; kv < total_kv; kv++) {
            if (kv < prefix_len) {
                mask_data[srow * total_kv + kv] = 0.0f;
            } else {
                int suffix_kv = kv - prefix_len;
                if (srow == 0) {
                    mask_data[srow * total_kv + kv] = (suffix_kv == 0) ? 0.0f : NEG_INF;
                } else {
                    mask_data[srow * total_kv + kv] = 0.0f;
                }
            }
        }
    }
    if (s.in_attn_mask->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> mask_h(mask_data.size());
        ggml_fp32_to_fp16_row(mask_data.data(), mask_h.data(), (int64_t) mask_data.size());
        ggml_backend_tensor_set(s.in_attn_mask, mask_h.data(), 0, sizeof(ggml_fp16_t) * mask_h.size());
    } else {
        ggml_backend_tensor_set(s.in_attn_mask, mask_data.data(), 0, sizeof(float) * mask_data.size());
    }

    s.graph_prefix_len = prefix_len;
    s.graph_kv_type    = s.kv_type;
    s.graph_il_from = il_from; s.graph_il_to = il_to;
    return true;
}

int pi0_expert_suffix_len(const pi0_model & m) { return m.pi05 ? PI0_ACTION_HORIZON : PI0_ACTION_HORIZON + 1; }

bool run_expert_step(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                     const float * state, const float * noisy_actions, float timestep,
                     float * v_t_out) {
    return run_expert_step_range(m, s, backend, state, noisy_actions, timestep, nullptr, nullptr, v_t_out);
}

bool run_expert_step_range(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                           const float * state, const float * noisy_actions, float timestep,
                           const float * hidden_in, float * hidden_out, float * v_t_out) {
    g_pi0_attn_mode = s.attn_mode;
    if (s.expert_il_from < 0 || s.expert_il_to > PI0_N_LAYER || s.expert_il_from >= s.expert_il_to) {
        LOG_ERR("run_expert_step_range: bad layer range [%d, %d)\n", s.expert_il_from, s.expert_il_to); return false;
    }
    if (s.expert_il_from > 0 && !hidden_in)            { LOG_ERR("run_expert_step_range: hidden_in required for il_from > 0\n"); return false; }
    if (s.expert_il_to < PI0_N_LAYER && !hidden_out)   { LOG_ERR("run_expert_step_range: hidden_out required for il_to < %d\n", PI0_N_LAYER); return false; }
    // (Re)build the graph if not present or if (prefix_len, kv_type, layer range) changed.
    if (!s.graph
        || s.graph_prefix_len != s.prefix_len
        || s.graph_kv_type    != s.kv_type
        || s.graph_il_from    != s.expert_il_from
        || s.graph_il_to      != s.expert_il_to) {
        if (!build_expert_graph(m, s, backend, s.prefix_len)) return false;
    }
    const int suffix_len = pi0_expert_suffix_len(m);
    const bool tail_half = s.expert_il_from > 0;   // graph starts from the residual: the suffix-prep inputs are not in it
    if (tail_half) {
        ggml_backend_tensor_set(s.in_hidden, hidden_in, 0, sizeof(float) * EXPERT_N_EMBD * suffix_len);
    }

    // Per-step inputs: state, noisy_actions, time. gallocr does not allocate leaves the graph does not reach, so a
    // tail-half graph must not be fed the suffix-prep inputs.
    if (!tail_half) ggml_backend_tensor_set(s.in_noisy, noisy_actions, 0,
        sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);

    std::vector<float> time_emb(EXPERT_N_EMBD);
    sincos_posemb(timestep, EXPERT_N_EMBD, time_emb.data());
    if (m.pi05) {
        if (!s.ada_table.empty()) {
            // PI0_ADARMS_TABLE: upload this step's 111 modulation leaves when the step changes.
            int step = (int) lroundf((1.0f - timestep) * PI0_NUM_STEPS);
            if (step < 0) step = 0; if (step > PI0_NUM_STEPS - 1) step = PI0_NUM_STEPS - 1;
            if (step != s.ada_table_step) {
                const int S = pi0_session::ADA_SITES, W = EXPERT_N_EMBD;
                for (int i = 0; i < S; i++)
                    for (int j = 0; j < 3; j++)
                        ggml_backend_tensor_set(s.ada_leaf[i][j], &s.ada_table[(((size_t) step * S + i) * 3 + j) * W], 0, sizeof(float) * W);
                s.ada_table_step = step;
            }
        } else {
            // pi0.5 without the table: sincos(t) [EXPERT_N_EMBD, 1] feeds the in-graph time MLP for the adaRMS cond.
            // It is built for both parts, so in_time is always reachable.
            ggml_backend_tensor_set(s.in_time, time_emb.data(), 0, sizeof(float) * EXPERT_N_EMBD);
        }
    } else if (tail_half) {
        // pi0 tail half: state/time only enter through the suffix prep, which this graph does not contain
    } else {
        ggml_backend_tensor_set(s.in_state, state, 0, sizeof(float) * PI0_ACTION_DIM);
        if (s.in_time->ne[1] == 1) {
            ggml_backend_tensor_set(s.in_time, time_emb.data(), 0, sizeof(float) * EXPERT_N_EMBD);   // K-split time MLP
        } else {
            // sincos(t) once, broadcast to all 50 positions, then upload as one tensor.
            std::vector<float> time_emb_broadcast(EXPERT_N_EMBD * PI0_ACTION_HORIZON);
            for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
                memcpy(&time_emb_broadcast[t * EXPERT_N_EMBD], time_emb.data(),
                    sizeof(float) * EXPERT_N_EMBD);
            }
            ggml_backend_tensor_set(s.in_time, time_emb_broadcast.data(), 0,
                sizeof(float) * EXPERT_N_EMBD * PI0_ACTION_HORIZON);
        }
    }

    if (ggml_backend_graph_compute(backend, s.graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Expert compute failed\n");
        return false;
    }

    if (s.expert_il_to < PI0_N_LAYER) {
        ggml_backend_tensor_get(s.out_hidden, hidden_out, 0, sizeof(float) * EXPERT_N_EMBD * suffix_len);
    } else {
        ggml_backend_tensor_get(s.out_v_t, v_t_out, 0,
            sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);
    }
    return true;
}

void pi0_session_free(pi0_session & s) {
    destroy_expert_graph(s);
    destroy_prefix_graph(s);
    destroy_kv_storage(s);
    s.prefix_len = 0;
}

// ---- KV handoff between sessions ----

bool pi0_session_prepare_kv(pi0_session & s, ggml_backend_t backend, int prefix_len) {
    if (!ensure_kv_storage(s, backend, prefix_len)) return false;
    s.prefix_len = prefix_len;
    return true;
}

static size_t kv_prefix_row_bytes(const pi0_session & s) {
    return s.cache_k[0] ? (size_t) s.prefix_len * s.cache_k[0]->nb[1] : 0;
}

bool pi0_kv_prefix_get(const pi0_session & s, std::vector<uint8_t> & blob, int il_from, int il_to) {
    if (!s.kv_buf || s.prefix_len <= 0) { LOG_ERR("pi0_kv_prefix_get: no KV cache to read\n"); return false; }
    if (il_from < 0 || il_to > PI0_N_LAYER || il_from >= il_to) { LOG_ERR("pi0_kv_prefix_get: bad layer range\n"); return false; }
    const size_t n = kv_prefix_row_bytes(s);
    blob.resize(2 * (size_t) (il_to - il_from) * n);
    for (int il = il_from; il < il_to; il++) {
        const size_t o = 2 * (size_t) (il - il_from) * n;
        ggml_backend_tensor_get(s.cache_k[il], blob.data() + o,     0, n);
        ggml_backend_tensor_get(s.cache_v[il], blob.data() + o + n, 0, n);
    }
    return true;
}

bool pi0_kv_prefix_set(pi0_session & s, const std::vector<uint8_t> & blob, int il_from, int il_to) {
    if (!s.kv_buf || s.prefix_len <= 0) { LOG_ERR("pi0_kv_prefix_set: KV cache not allocated (call pi0_session_prepare_kv)\n"); return false; }
    if (il_from < 0 || il_to > PI0_N_LAYER || il_from >= il_to) { LOG_ERR("pi0_kv_prefix_set: bad layer range\n"); return false; }
    const size_t n = kv_prefix_row_bytes(s);
    if (blob.size() != 2 * (size_t) (il_to - il_from) * n) {
        LOG_ERR("pi0_kv_prefix_set: blob is %zu bytes, cache expects %zu for layers [%d,%d) (kv_type/prefix_len mismatch between the sessions)\n",
                blob.size(), 2 * (size_t) (il_to - il_from) * n, il_from, il_to);
        return false;
    }
    for (int il = il_from; il < il_to; il++) {
        const size_t o = 2 * (size_t) (il - il_from) * n;
        ggml_backend_tensor_set(s.cache_k[il], blob.data() + o,     0, n);
        ggml_backend_tensor_set(s.cache_v[il], blob.data() + o + n, 0, n);
    }
    return true;
}

// ============================================================
// Utility
// ============================================================

std::string pi0_format_prompt_pi05(const std::string & task, const float * state) {
    // 256 equal bins over [-1, 1] as in the openpi tokenizer. Values below -1 map to 0, at or above 1 to 255.
    // The zero state maps to 128.
    std::string s = "Task: " + task + ", State:";
    for (int i = 0; i < PI0_ACTION_DIM; i++) {
        const float v = state ? state[i] : 0.0f;
        int idx = (int) floorf((v + 1.0f) * 128.0f);      // bin i covers [-1 + i/128, -1 + (i+1)/128)
        if (idx < 0)   idx = 0;
        if (idx > 255) idx = 255;
        s += " " + std::to_string(idx);
    }
    s += ";\nAction: ";
    return s;
}

std::string resolve_model_dir(const std::string & path) {
    std::string dir = path;
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    if (dir.find(".gguf") != std::string::npos) {
        auto pos = dir.rfind('/');
        dir = (pos != std::string::npos) ? dir.substr(0, pos) : ".";
    }
    return dir;
}

// ============================================================
// Backend selection
// ============================================================

static const char * dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

void pi0_dump_backends() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    LOG_INF("Available backend devices (%zu):\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        LOG_INF("  [%zu] %-12s %-5s %s\n",
                i,
                ggml_backend_dev_name(dev),
                dev_type_str(ggml_backend_dev_type(dev)),
                ggml_backend_dev_description(dev));
    }
}

ggml_backend_t pi0_init_backend(const std::string & name) {
    ggml_backend_load_all();

    if (name.empty() || name == "cpu" || name == "CPU") {
        LOG_INF("Backend: CPU\n");
        return ggml_backend_cpu_init();
    }

    if (name == "auto" || name == "gpu" || name == "GPU") {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (b) {
            LOG_INF("Backend: %s (auto-selected GPU)\n", ggml_backend_name(b));
            return b;
        }
        LOG_WRN("No GPU backend available; falling back to CPU\n");
        return ggml_backend_cpu_init();
    }

    if (ggml_backend_t b = ggml_backend_init_by_name(name.c_str(), nullptr)) {
        LOG_INF("Backend: %s\n", ggml_backend_name(b));
        return b;
    }

    // Fallback: case-insensitive substring match against device names, so --backend OpenCL matches GPUOpenCL.
    auto lower = [](std::string s) {
        for (auto & c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        return s;
    };
    const std::string needle = lower(name);
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (lower(ggml_backend_dev_name(dev)).find(needle) != std::string::npos) {
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) {
                LOG_INF("Backend: %s (matched '%s' by substring)\n",
                        ggml_backend_dev_name(dev), name.c_str());
                return b;
            }
        }
    }

    LOG_ERR("Backend '%s' not found.\n", name.c_str());
    pi0_dump_backends();
    return nullptr;
}

// ============================================================
// Tensor-type reporting
// ============================================================

static void dump_ctx_types(ggml_context * ctx, const char * label) {
    if (!ctx) return;
    std::map<int, int>    counts; // ggml_type -> count
    std::map<int, size_t> bytes;
    size_t total_bytes = 0;
    int    total_count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        counts[(int) t->type]++;
        bytes[(int) t->type] += ggml_nbytes(t);
        total_bytes += ggml_nbytes(t);
        total_count++;
    }
    LOG_INF("[%s] %d tensors, %.2f MB total\n",
            label, total_count, total_bytes / (1024.0 * 1024.0));
    for (auto & kv : counts) {
        ggml_type ty = (ggml_type) kv.first;
        LOG_INF("    %-10s %4d tensors  %8.2f MB\n",
                ggml_type_name(ty), kv.second, bytes[kv.first] / (1024.0 * 1024.0));
    }
}

void pi0_dump_tensor_types(const pi0_model & m) {
    dump_ctx_types(m.ctx_pali,   "PaliGemma 2B");
    dump_ctx_types(m.ctx_expert, "Action Expert");
}

// ============================================================
// Shared CLI flag stripping
// ============================================================

pi0_cli pi0_strip_cli_args(int & argc, char ** argv) {
    pi0_cli c;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            c.backend = argv[++i];
        } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
            c.kv_type = argv[++i];
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return c;
}

ggml_type pi0_ggml_type_from_string(const std::string & s) {
    auto eq = [&s](const char * lit) {
        if (s.size() != strlen(lit)) return false;
        for (size_t i = 0; i < s.size(); i++) {
            char a = s[i];
            char b = lit[i];
            if (a >= 'A' && a <= 'Z') a = char(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = char(b - 'A' + 'a');
            if (a != b) return false;
        }
        return true;
    };
    if (eq("f32"))  return GGML_TYPE_F32;
    if (eq("f16"))  return GGML_TYPE_F16;
    if (eq("bf16")) return GGML_TYPE_BF16;
    if (eq("q8_0")) return GGML_TYPE_Q8_0;
    if (eq("q4_0")) return GGML_TYPE_Q4_0;
    if (eq("q4_1")) return GGML_TYPE_Q4_1;
    if (eq("q5_0")) return GGML_TYPE_Q5_0;
    if (eq("q5_1")) return GGML_TYPE_Q5_1;
    return GGML_TYPE_COUNT;
}
