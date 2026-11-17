#pragma once

// PI0 shared types and API.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// ---- Constants ----

static constexpr int PI0_ACTION_DIM     = 32;
static constexpr int PI0_ACTION_HORIZON = 50;
static constexpr int PI0_N_LAYER        = 18;
static constexpr int PI0_FFN_DOWN_MAX_SPLIT = 8;   // max ffn_down K-split slices ffn_down_k0..k7 (tools/gguf_tools.py split-k)
static constexpr int PI0_NUM_STEPS      = 10;

// PaliGemma 2B
static constexpr int PALI_N_EMBD    = 2048;
static constexpr int PALI_N_HEAD    = 8;
static constexpr int PALI_N_KV_HEAD = 1;
static constexpr int PALI_HEAD_DIM  = 256;
static constexpr int PALI_N_FF      = 16384;

// Action Expert 300M
static constexpr int EXPERT_N_EMBD    = 1024;
static constexpr int EXPERT_N_HEAD    = 8;
static constexpr int EXPERT_N_KV_HEAD = 1;
static constexpr int EXPERT_HEAD_DIM  = 256;
static constexpr int EXPERT_N_FF      = 4096;

// pi0.5 adaRMS modulation width: scale | shift | gate
static constexpr int PI0_ADARMS_MOD_DIM = 3 * EXPERT_N_EMBD;

// ---- Weight structures ----

struct gemma_layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * q_proj    = nullptr;
    ggml_tensor * k_proj    = nullptr;
    ggml_tensor * v_proj    = nullptr;
    ggml_tensor * qkv_proj  = nullptr; // fused [in, n_head*hd + 2*kv_dim], used instead of q/k/v when set
    ggml_tensor * o_proj    = nullptr;
    ggml_tensor * gate_proj = nullptr;
    ggml_tensor * up_proj   = nullptr;
    ggml_tensor * gateup_proj = nullptr; // gate/up rows interleaved in 32-row tiles [in, 2*ff] (tools/gguf_tools.py fuse-gateup)
    ggml_tensor * down_proj = nullptr;
    // ffn_down split along K into n [in/n, d] slices (tools/gguf_tools.py split-k). The n GEMMs' f32 partial sums
    // are added in sequence. down_proj may be null when set.
    ggml_tensor * down_k[PI0_FFN_DOWN_MAX_SPLIT] = {};
    int           down_n_split = 0;   // 0 = no K-split available, use down_proj
    // pi0.5 action expert only: adaRMS modulation Dense [EXPERT_N_EMBD -> 3*width] (+bias) per norm.
    // When set, attn_norm/ffn_norm stay null and the layer uses adaRMS and gated residuals.
    ggml_tensor * attn_norm_dense_w = nullptr;
    ggml_tensor * attn_norm_dense_b = nullptr;
    ggml_tensor * ffn_norm_dense_w  = nullptr;
    ggml_tensor * ffn_norm_dense_b  = nullptr;
};

struct pi0_model {
    // PaliGemma 2B
    ggml_tensor * pali_embed = nullptr;
    gemma_layer_weights pali_layers[PI0_N_LAYER] = {};
    ggml_tensor * pali_final_norm = nullptr;

    // Action Expert 300M
    gemma_layer_weights expert_layers[PI0_N_LAYER] = {};
    ggml_tensor * expert_final_norm = nullptr;

    // Action projections. They may be quantized, so read them only via ggml_mul_mat in graphs.
    ggml_tensor * action_in_proj_w      = nullptr;
    ggml_tensor * action_in_proj_b      = nullptr;
    ggml_tensor * action_out_proj_w     = nullptr;
    ggml_tensor * action_out_proj_b     = nullptr;
    ggml_tensor * state_proj_w          = nullptr;
    ggml_tensor * state_proj_b          = nullptr;
    ggml_tensor * action_time_mlp_in_w  = nullptr;
    ggml_tensor * action_time_mlp_in_b  = nullptr;
    ggml_tensor * action_time_mlp_in_k0 = nullptr; // K-split of action_time_mlp_in (gguf_tools.py split-k): k0 acts on the action
    ggml_tensor * action_time_mlp_in_k1 = nullptr; // tokens, k1 on the time embedding. _w may be null when both are set.
    ggml_tensor * action_time_mlp_out_w = nullptr;
    ggml_tensor * action_time_mlp_out_b = nullptr;

    // pi0.5 action expert (pi05 == true): no state token or action_time_mlp. adaRMS norms (and a final adaRMS norm)
    // are conditioned on cond = swish(time_mlp_out(swish(time_mlp_in(sincos(t))))).
    bool          pi05                     = false;
    ggml_tensor * expert_final_norm_dense_w = nullptr; // [EXPERT_N_EMBD, 3*EXPERT_N_EMBD]
    ggml_tensor * expert_final_norm_dense_b = nullptr;
    ggml_tensor * time_mlp_in_w            = nullptr;
    ggml_tensor * time_mlp_in_b            = nullptr;
    ggml_tensor * time_mlp_out_w           = nullptr;
    ggml_tensor * time_mlp_out_b           = nullptr;

    // Backend resources
    ggml_context * ctx_pali   = nullptr;
    ggml_context * ctx_expert = nullptr;
    ggml_backend_buffer_t buf_pali   = nullptr;
    ggml_backend_buffer_t buf_expert = nullptr;
};

// ---- Per-inference runtime state ----
//
// Owns the backend-resident per-layer K/V cache (element type kv_type) and the persistent graphs.
//
// Lifecycle:
//   1. Default-construct, then set kv_type (default F32).
//   2. run_paligemma_prefix() allocates the KV cache as needed and writes positions 0..prefix_len-1.
//      The F32 -> kv_type conversion happens in the graph via ggml_cpy.
//   3. run_expert_step() writes positions prefix_len..prefix_len+suffix_len-1 and attends over the full cache.
//      Its graph is built lazily and rebuilt only when prefix_len, kv_type or the layer range changes.
//   4. pi0_session_free() releases the cache and the graphs.

struct pi0_session {
    // KV cache configuration. Set BEFORE run_paligemma_prefix.
    ggml_type kv_type = GGML_TYPE_F32;
    // Attention for the graphs built through this session: -1 = process default (PI0_NO_FLASH_ATTN),
    // 0 = ggml_flash_attn_ext, 1 = standard 3-step attention. Per session, so sessions on different backends can differ.
    int attn_mode = -1;

    int prefix_len = 0; // length of the prefix written by run_paligemma_prefix

    // Persistent KV cache. Each cache_k/v[il] is [kv_dim, prefix_len + suffix_len] in kv_type, with
    // suffix_len = PI0_ACTION_HORIZON + 1. The prefix rows are written once per inference. The suffix rows are
    // overwritten every diffusion step.
    int kv_alloc_prefix_len = 0;
    ggml_type kv_alloc_type = GGML_TYPE_F32;
    ggml_context        * kv_ctx           = nullptr;
    ggml_backend_buffer_t kv_buf           = nullptr;
    ggml_tensor         * cache_k[PI0_N_LAYER] = {};
    ggml_tensor         * cache_v[PI0_N_LAYER] = {};

    // Persistent expert graph (rebuilt if prefix_len, kv_type or the layer range changes).
    // expert_il_from/to: the expert layers the step graph covers (default all). For a step split across two devices,
    // the first part emits the suffix residual (out_hidden) and the second part takes it (in_hidden) and emits v_t.
    // Each expert layer reads only the suffix K/V it wrote, so the parts share only the prefix K/V
    // (pi0_kv_prefix_set) and the [EXPERT_N_EMBD x suffix_len] residual.
    int       expert_il_from   = 0;
    int       expert_il_to     = PI0_N_LAYER;
    int       graph_il_from    = 0;
    int       graph_il_to      = PI0_N_LAYER;
    ggml_tensor * in_hidden    = nullptr;
    ggml_tensor * out_hidden   = nullptr;
    int       graph_prefix_len = 0;
    ggml_type graph_kv_type    = GGML_TYPE_F32;
    ggml_context  * graph_ctx   = nullptr;
    ggml_cgraph   * graph       = nullptr;
    ggml_gallocr_t  graph_alloc = nullptr;

    ggml_tensor * in_state     = nullptr;
    ggml_tensor * in_noisy     = nullptr;
    ggml_tensor * in_time      = nullptr;
    ggml_tensor * in_positions = nullptr;
    ggml_tensor * in_attn_mask = nullptr;
    ggml_tensor * out_v_t      = nullptr;
    // Constant-across-steps inputs (positions, attention mask) live in their own buffer, not in the gallocr arena.
    // gallocr may reuse an input's memory after its last consumer, which would corrupt an input uploaded only once.
    ggml_context         * graph_const_ctx = nullptr;
    ggml_backend_buffer_t  graph_const_buf = nullptr;
    ggml_tensor          * suffix_buf      = nullptr;   // pi0: pinned [EXPERT_N_EMBD, 51] that receives the state and action tokens
    // pi0.5 adaRMS modulation table (PI0_ADARMS_TABLE=1). The (1+scale | shift | gate) vectors of the 37 norm sites
    // depend only on the timestep, and the schedule is fixed (t = 1 + step * -1/PI0_NUM_STEPS). They are computed
    // once per graph build for all steps and fed per step as [EXPERT_N_EMBD, 1] leaves.
    static constexpr int ADA_SITES = 2 * PI0_N_LAYER + 1;                // attn-norm + ffn-norm per layer, + final norm
    ggml_tensor        * ada_leaf[2 * PI0_N_LAYER + 1][3] = {};         // [site][scale|shift|gate]
    std::vector<float>   ada_table;                                      // [PI0_NUM_STEPS][ADA_SITES][3*EXPERT_N_EMBD]
    int                  ada_table_step = -1;

    // Persistent prefix graph (rebuilt if prefix_len, the layer range, the backend or the KV storage changes).
    // PI0_PREFIX_GRAPH_CACHE=0 rebuilds it on every call.
    int             pgraph_prefix_len = 0;
    int             pgraph_il_from    = 0;               // layer range the cached graph covers
    int             pgraph_il_to      = PI0_N_LAYER;
    ggml_type       pgraph_kv_type    = GGML_TYPE_F32;   // the cached graph bakes in the KV dtype and the cache addresses
    ggml_tensor *   pgraph_cache_k0   = nullptr;
    ggml_backend_t  pgraph_backend    = nullptr;
    ggml_context  * pgraph_kv_ctx     = nullptr;   // identity of the KV storage the graph was built against
    ggml_context  * pgraph_ctx        = nullptr;
    ggml_cgraph   * pgraph            = nullptr;
    ggml_gallocr_t  pgraph_alloc      = nullptr;
    ggml_tensor   * pgraph_input      = nullptr;
    ggml_tensor   * pgraph_positions  = nullptr;
    ggml_tensor   * pgraph_out        = nullptr;
};

void pi0_session_free(pi0_session & s);

// ---- KV cache handoff between two sessions on different backends ----
// Allocate the KV storage of s for prefix_len on backend without running the prefix (the prefix rows are copied in
// from another session), and set s.prefix_len so run_expert_step() can build its graph.
bool pi0_session_prepare_kv(pi0_session & s, ggml_backend_t backend, int prefix_len);
// Copy the prefix rows [0, prefix_len) of each layer's K and V to a host blob (per layer: K rows, then V rows), or
// from such a blob into another session's cache. Both sessions must share kv_type and prefix_len.
bool pi0_kv_prefix_get(const pi0_session & s, std::vector<uint8_t> & blob, int il_from = 0, int il_to = PI0_N_LAYER);
bool pi0_kv_prefix_set(pi0_session & s, const std::vector<uint8_t> & blob, int il_from = 0, int il_to = PI0_N_LAYER);

// Parse a ggml_type from a CLI string ("f32", "f16", "q8_0", "q4_0", ...).
// Returns GGML_TYPE_COUNT on unrecognized input.
ggml_type pi0_ggml_type_from_string(const std::string & s);

// ---- API ----

bool load_pi0_model(pi0_model & m,
                    const char * pali_path,
                    const char * expert_path,
                    ggml_backend_t backend);

void free_pi0_model(pi0_model & m);

// PaliGemma prefix pass. Allocates s.cache_k/v as needed and writes positions 0..prefix_len-1.
bool run_paligemma_prefix(pi0_model & m,
                          pi0_session & s,
                          ggml_backend_t backend,
                          const float * embeddings, // [PALI_N_EMBD * prefix_len]
                          int prefix_len);

// Prefix pass over layers [il_from, il_to) only. input is the prefix embeddings when il_from == 0, else the residual
// [PALI_N_EMBD * prefix_len] from the device that ran the earlier layers. K/V of these layers go to s.cache_k/v.
// The caller copies the other layers' K/V in with pi0_kv_prefix_set(). When il_to < PI0_N_LAYER and out_hidden is
// set, the residual after layer il_to-1 is read back into *out_hidden. The final norm runs only when
// il_to == PI0_N_LAYER. run_paligemma_prefix() is the full range with no read-back.
bool run_paligemma_prefix_range(pi0_model & m,
                                pi0_session & s,
                                ggml_backend_t backend,
                                const float * input,
                                int prefix_len,
                                int il_from, int il_to,
                                std::vector<float> * out_hidden);

// One Action Expert denoising step. Suffix prep (state_proj, action_in_proj, time MLP) runs in the graph.
bool run_expert_step(pi0_model & m,
                     pi0_session & s,
                     ggml_backend_t backend,
                     const float * state,         // [PI0_ACTION_DIM], ignored (may be null) for a pi0.5 expert
                     const float * noisy_actions, // [PI0_ACTION_DIM * PI0_ACTION_HORIZON]
                     float timestep,
                     float * v_t_out);            // [PI0_ACTION_DIM * PI0_ACTION_HORIZON]

// Layer-range variant (s.expert_il_from / s.expert_il_to). hidden_in is required when expert_il_from > 0
// ([EXPERT_N_EMBD * suffix_len], suffix_len = 51 for pi0, 50 for pi0.5). hidden_out receives the residual when
// expert_il_to < PI0_N_LAYER, and v_t_out is then untouched. run_expert_step() is the full range.
bool run_expert_step_range(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                           const float * state, const float * noisy_actions, float timestep,
                           const float * hidden_in, float * hidden_out, float * v_t_out);
int  pi0_expert_suffix_len(const pi0_model & m);   // 51 (pi0: state + 50 actions) or 50 (pi0.5)

// Dequantize a single embedding row (token_id) into out_f32 (length t->ne[0]).
// Works for F32, F16, and any quantized type registered with ggml_get_type_traits.
void embd_lookup_f32(ggml_tensor * t, int token_id, float * out_f32);

std::string resolve_model_dir(const std::string & path);
// pi0.5 prompt (openpi tokenizer convention): "Task: <task>, State: <32 ints>;\nAction: ", with the state
// discretised into 256 bins over [-1, 1]. state == nullptr means the zero state.
// pi0.5 has no state token. The robot state reaches the model only through this prompt, so pass the real state.
std::string pi0_format_prompt_pi05(const std::string & task, const float * state);

// ---- Backend selection ----
//
// Resolve a backend by name: "cpu", "auto"/"gpu" (first available GPU), or a
// concrete name like "OpenCL", "Vulkan", "Metal", "CUDA". Returns nullptr if
// the requested backend is unavailable (use pi0_dump_backends() to see what is).

ggml_backend_t pi0_init_backend(const std::string & name);

// List available backend devices to stderr.
void pi0_dump_backends();

// ---- Reporting ----
//
// Print a per-context histogram of tensor types (F16, Q4_0, Q4_K, ...) and the total weight footprint.

void pi0_dump_tensor_types(const pi0_model & m);

// ---- Shared CLI flag stripping ----
//
// Removes the pi0-wide flags (--backend, --kv-type) from argv before common_params_parse sees them.

struct pi0_cli {
    std::string backend = "cpu";
    std::string kv_type = "f32"; // see pi0_ggml_type_from_string
};

pi0_cli pi0_strip_cli_args(int & argc, char ** argv);
