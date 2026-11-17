// llama-pi0-opmat-bench: operator-level GPU/NPU co-execution matrices (the co-run profile of the cut model).
//
//   g_ij = rate of GPU (OpenCL) op i while the NPU (HTP0) loops op j
//   h_ij = rate of NPU (HTP0) op i while the GPU (OpenCL) loops op j
//
// Run by `vladuo.sh profile quick|full` (one process per chunk). vladuo.py merges the logs.
//
// Op sets
//   --set quick  (default) the cut-model input. GPU rows: the six SigLIP ops per view (N=256, f16 weights). NPU: the 18
//                HTP ops of batched SigLIP (N=768), the PaliGemma prefix (N=774) and the action expert (N=51). pi0
//                shapes only. GPU vis_attn zero-pads head_dim 72 -> 80 (scale 1/sqrt(72)). NPU fm_ff_dn is the single
//                ffn_down 4096->1024 on f16 activations + residual.
//   --set full   the 20 atom types of the cut model on each device (GPU = what the OpenCL sessions run, NPU = the
//                deployed HTP graph), shapes of --model pi0|pi05 (prefix N 774|912, suffix N 51|50, expert kv 825|962).
//
// Measurement: one weight context per op (U(-0.1, 0.1), q4_0 stored in the HTP REPACK buffer). Only the two graphs of
// the current cell are alive. Each graph holds enough copies of its op that one submit takes >= --rep-ms (at most 32
// copies). A background thread loops the other device's graph while the foreground runs 3 timed passes. The median
// pass is reported in GFLOPS. Every row also prints its 'idle' rate (other device idle). Cells outside --cols, or whose
// graph the backend does not support, print n/a. Only ratios are meaningful (dispatch overheads differ from the fused
// model graphs).
//
// Small NPU ops looping as background (QUICK vis_ln vis_gelu pre_rmsn pre_rope, FULL v_norm p_norm e_norm) can hang
// the DSP queue of the process. Those columns print n/a unless --small-bg, which raises their copy cap to 2048 (bounded
// by 512 MB of graph memory) so that one submit lasts >= --rep-ms. Small NPU ops as foreground rows are safe.
//
// FULL graphs reproduce the node sequence of each deployed atom. The support check runs after allocation. The GLU/CPY
// nodes of the interleaved gate/up are exempt because ggml-hexagon fuses them into the matmul.
//
// cGFlop per call: matmul 2*M*K*N, attention 4*N*kv*hd*heads*batch (logical hd 72 for SigLIP), norms 5*K*N. FULL atoms
// add GELU/GEGLU 8*ff*N and rope 6*hd*heads*N. Patch-embed counts the useful K = 3*14*14 = 588 on both devices.
//
// Usage: llama-pi0-opmat-bench [--set quick|full] [--model pi0|pi05] --which g|h [--rows SEL] [--cols SEL]
//                              [--iters 12] [--rep-ms 8] [--small-bg] [--verbose]
//        llama-pi0-opmat-bench --list [--model pi0|pi05]        (op tables, no backend)
//        llama-pi0-opmat-bench --dry ...                        (build the graphs without a backend, print the plan)
//        llama-pi0-opmat-bench --selftest-parse
//   SEL = comma list of indices and half-open ranges a:b, e.g. 0:4,6,9:11.
//   env: the deployed HTP/OpenCL flags (vladuo.sh ENV_OPMAT). This file reads none itself.

#include "pi0-common.h"

#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

enum op_kind {
    K_MM, K_MM_KSPLIT, K_GEGLU, K_FA, K_LN, K_GELU, K_RMSN, K_ROPE,   // QUICK forms (several reused by FULL)
    K_VIT_ATTN,     // GPU SigLIP attention, clip.cpp non-flash path with the PI0_VIT_IMG head_dim pad
    K_PATCH_CONV,   // GPU patch embedding: ggml_conv_2d + transpose cont + bias + pos
    K_PATCH_FAST,   // NPU batched patch embedding: PAD x, PAD w, MUL_MAT, bias, broadcast pos
    K_QKV3,         // SigLIP q/k/v: 3 x (mm + bias) on one input
    K_LN_MM,        // post-LN + projector mm + bias
    K_ADARMS,       // pi0.5 adaRMS norm: rms_norm * scale + shift
    K_LM_QKV,       // Gemma QKV + rope (var 0: unfused q/k/v, 1: fused + two strided ropes, 2: fused + one rope)
    K_ATTN_COLL,    // GPU MQA-collapse attention (QK / softmax [+ tiled mask] / AV), f16 K/V
    K_FA_LM,        // NPU flash attention with the one-rope Q view, f16 K/V [+ f16 mask]
    K_GEGLU2,       // separate gate / up mm + GEGLU
};

struct OpDef {
    const char * name;
    const char * desc;       // form
    op_kind      kind;
    ggml_type    wt;         // weight type (matmul kinds)
    int64_t      K, M, N;    // matmul dims, attention: K = head_dim, M = n_kv, N = n_q
    int          nh, nkvh, nb;   // heads / kv heads / batch (attention, LM QKV), K_MM_KSPLIT: nh = K-split parts
    bool         bias, residual, in_f16, out_f16, mask;
    int          count    = 0;       // calls per frame on this device
    bool         small    = false;   // small NPU op: background column only with --small-bg
    bool         full     = false;   // FULL atom: cGFlop includes the element-wise nodes
    bool         res_last = false;   // residual ADD as (product, residual)
    bool         gelu     = false;   // K_MM: tanh-GELU after the bias
    bool         gated    = false;   // pi0.5 gated residual: residual + product * gate[M,1]
    int          var      = 0;       // K_LM_QKV variant
};

static constexpr int     LM_HD     = 256;                 // Gemma head_dim (prefix and expert)
static constexpr int     VIT_IMG   = 224, VIT_PATCH = 14, VIT_C = 3, VIT_NPOS = 256;
static constexpr int64_t VIT_KPAD  = 32;                  // HMX K alignment of the batched patch matmul
static constexpr int     CAP_COPIES = 32, CAP_COPIES_SMALL_BG = 2048;
static constexpr double  SMALL_BG_MAX_BYTES = 512.0 * 1024 * 1024;

// ---- QUICK: GPU (OpenCL) SigLIP per view, N=256 ------------------------------------------------------------------
static const OpDef QUICK_GPU[] = {
    { "vis_qkvo", "mm +bias [f16]",                       K_MM,       GGML_TYPE_F16, 1152, 1152, 256, 0, 0, 0, true,  false, false, false, false, 324 },
    { "vis_ff_up","mm +bias [f16]",                       K_MM,       GGML_TYPE_F16, 1152, 4320, 256, 0, 0, 0, true,  false, false, false, false, 81 },
    { "vis_ff_dn","mm +bias +res [f16]",                  K_MM,       GGML_TYPE_F16, 4320, 1152, 256, 0, 0, 0, true,  true,  false, false, false, 81 },
    { "vis_attn", "QK (hd 72->80 zero pad)/softmax/AV f32", K_VIT_ATTN, GGML_TYPE_F32, 72, 256, 256, 16, 16, 1, false, false, false, false, false, 81 },
    { "vis_ln",   "layernorm*w+b",                        K_LN,       GGML_TYPE_F32, 1152, 0, 256, 0, 0, 0, false, false, false, false, false, 162 },
    { "vis_gelu", "gelu",                                 K_GELU,     GGML_TYPE_F32, 4320, 0, 256, 0, 0, 0, false, false, false, false, false, 81 },
};
// ---- QUICK: NPU (HTP0) batched SigLIP N=768, prefix N=774, expert N=51 ---------------------------------------------
static const OpDef QUICK_NPU[] = {
    { "vis_qkvo", "mm +bias [f16]",                       K_MM,       GGML_TYPE_F16, 1152, 1152, 768, 0, 0, 0, true,  false, false, false, false, 108 },
    { "vis_ff_up","mm +bias [f16]",                       K_MM,       GGML_TYPE_F16, 1152, 4320, 768, 0, 0, 0, true,  false, false, false, false, 27 },
    { "vis_ff_dn","mm +bias +res [f16]",                  K_MM,       GGML_TYPE_F16, 4320, 1152, 768, 0, 0, 0, true,  true,  false, false, false, 27 },
    { "vis_attn", "flash-attn 3 views f32 (native pad)",  K_FA,       GGML_TYPE_F32, 72, 256, 256, 16, 16, 3, false, false, false, false, false, 27 },
    { "vis_ln",   "layernorm*w+b (fused)",                K_LN,       GGML_TYPE_F32, 1152, 0, 768, 0, 0, 0, false, false, false, false, false, 54, true },
    { "vis_gelu", "gelu",                                 K_GELU,     GGML_TYPE_F32, 4320, 0, 768, 0, 0, 0, false, false, false, false, false, 27, true },
    { "pre_qkv",  "mm [q4_0]",                            K_MM,       GGML_TYPE_Q4_0, 2048, 2560, 774, 0, 0, 0, false, false, false, false, false, 18 },
    { "pre_o",    "mm +res [q4_0]",                       K_MM,       GGML_TYPE_Q4_0, 2048, 2048, 774, 0, 0, 0, false, true,  false, false, false, 18 },
    { "pre_ff_up","gate/up GeGLU epilogue, f16 store [q4_0]", K_GEGLU, GGML_TYPE_Q4_0, 2048, 16384, 774, 0, 0, 0, false, false, false, true, false, 18 },
    { "pre_ff_dn","4-way K-split, f16 acts, +res [q4_0]", K_MM_KSPLIT, GGML_TYPE_Q4_0, 16384, 2048, 774, 4, 0, 0, false, true, true, false, false, 18 },
    { "pre_attn", "flash-attn f16 kv + f16 mask",         K_FA,       GGML_TYPE_F16, 256, 774, 774, 8, 1, 1, false, false, false, false, true, 18 },
    { "pre_rmsn", "rmsnorm*w (fused)",                    K_RMSN,     GGML_TYPE_F32, 2048, 0, 774, 0, 0, 0, false, false, false, false, false, 37, true },
    { "pre_rope", "rope neox",                            K_ROPE,     GGML_TYPE_F32, 256, 9, 774, 0, 0, 0, false, false, false, false, false, 18, true },
    { "fm_qkv",   "mm [q4_0]",                            K_MM,       GGML_TYPE_Q4_0, 1024, 2560, 51, 0, 0, 0, false, false, false, false, false, 180 },
    { "fm_o",     "mm +res [q4_0]",                       K_MM,       GGML_TYPE_Q4_0, 2048, 1024, 51, 0, 0, 0, false, true,  false, false, false, 180 },
    { "fm_ff_up", "gate/up GeGLU epilogue [q4_0]",        K_GEGLU,    GGML_TYPE_Q4_0, 1024, 4096, 51, 0, 0, 0, false, false, false, false, false, 180 },
    { "fm_ff_dn", "single ffn_down, f16 acts, +res [q4_0]", K_MM,     GGML_TYPE_Q4_0, 4096, 1024, 51, 0, 0, 0, false, true,  true,  false, false, 180 },
    { "fm_attn",  "flash-attn f16 kv + f16 mask",         K_FA,       GGML_TYPE_F16, 256, 825, 51, 8, 1, 1, false, false, false, false, true, 180 },
};

// ---- FULL: the 20 atoms per device -----------------------------------------------------------------------------------
static std::vector<OpDef> full_ops(bool npu, bool pi05) {
    const int64_t P = pi05 ? 912 : 774, S = pi05 ? 50 : 51, KV = P + S;
    const int64_t NV = npu ? 768 : 256;   // SigLIP tokens per call: 3 views batched on the NPU, one view on the GPU
    const ggml_type F16 = GGML_TYPE_F16, F32 = GGML_TYPE_F32, Q4 = GGML_TYPE_Q4_0;
    auto op = [](const char * name, const char * desc, op_kind kind, ggml_type wt, int64_t K, int64_t M, int64_t N, int count) {
        OpDef o = { name, desc, kind, wt, K, M, N, 0, 0, 0, false, false, false, false, false };
        o.count = count; o.full = true;
        return o;
    };
    std::vector<OpDef> v;
    if (npu) {
        OpDef a = op("v_patch", "PAD x/w 588->608, mm f32 +bias +pos", K_PATCH_FAST, F32, 588, 1152, NV, 1); a.nb = 3; v.push_back(a);
    } else {
        v.push_back(op("v_patch", "conv_2d (im2col+mm) f32 +bias +pos", K_PATCH_CONV, F32, 588, 1152, NV, 3));
    }
    { OpDef a = op("v_norm", npu ? "layernorm*w+b (fused)" : "layernorm*w+b", K_LN, F32, 1152, 0, NV, npu ? 54 : 162); a.small = npu; v.push_back(a); }
    { OpDef a = op("v_qkv", "3 x mm +bias [f16]", K_QKV3, F16, 1152, 1152, NV, npu ? 27 : 81); a.bias = true; v.push_back(a); }
    if (npu) {
        OpDef a = op("v_attn", "flash-attn 3 views f32 (native pad)", K_FA, F32, 72, 256, 256, 27); a.nh = 16; a.nkvh = 16; a.nb = 3; v.push_back(a);
    } else {
        OpDef a = op("v_attn", "QK (hd 72->80 zero pad)/softmax/AV f32", K_VIT_ATTN, F32, 72, 256, 256, 81); a.nh = 16; a.nkvh = 16; a.nb = 1; v.push_back(a);
    }
    { OpDef a = op("v_out",    "mm +bias +res [f16]",      K_MM, F16, 1152, 1152, NV, npu ? 27 : 81); a.bias = a.residual = a.res_last = true; v.push_back(a); }
    { OpDef a = op("v_ffn_up", "mm +bias +gelu [f16]",     K_MM, F16, 1152, 4320, NV, npu ? 27 : 81); a.bias = a.gelu = true; v.push_back(a); }
    { OpDef a = op("v_ffn_dn", "mm +bias +res [f16]",      K_MM, F16, 4320, 1152, NV, npu ? 27 : 81); a.bias = a.residual = true; v.push_back(a); }
    { OpDef a = op("v_proj",   "post-LN + mm +bias [f16]", K_LN_MM, F16, 1152, 2048, NV, npu ? 1 : 3); a.bias = true; v.push_back(a); }
    { OpDef a = op("p_norm", npu ? "rmsnorm*w (fused)" : "rmsnorm*w", K_RMSN, F32, 2048, 0, P, 37); a.small = npu; v.push_back(a); }
    { OpDef a = op("p_qkv", npu ? "fused mm + one rope [q4_0]" : "q/k/v mm + 2 ropes [q4_0]", K_LM_QKV, Q4, 2048, 2560, P, 18);
      a.nh = 8; a.nkvh = 1; a.var = npu ? 2 : 0; v.push_back(a); }
    { OpDef a = op("p_attn", npu ? "flash-attn f16 kv (one-block)" : "MQA-collapse QK/softmax/AV f16 kv", npu ? K_FA_LM : K_ATTN_COLL, F16, LM_HD, P, P, 18);
      a.nh = 8; a.nkvh = 1; a.nb = 1; v.push_back(a); }
    { OpDef a = op("p_out", "mm +res [q4_0]", K_MM, Q4, 2048, 2048, P, 18); a.residual = true; v.push_back(a); }
    if (npu) { OpDef a = op("p_ffn_up", "gate/up GeGLU epilogue, f16 store [q4_0]", K_GEGLU, Q4, 2048, 16384, P, 18); a.out_f16 = true; v.push_back(a); }
    else     { v.push_back(op("p_ffn_up", "gate + up mm + geglu [q4_0]", K_GEGLU2, Q4, 2048, 16384, P, 18)); }
    if (npu) { OpDef a = op("p_ffn_dn", pi05 ? "2-way K-split, f16 acts, +res [q4_0]" : "4-way K-split, f16 acts, +res [q4_0]", K_MM_KSPLIT, Q4, 16384, 2048, P, 18);
               a.nh = pi05 ? 2 : 4; a.in_f16 = a.residual = true; v.push_back(a); }
    else     { OpDef a = op("p_ffn_dn", "mm +res [q4_0]", K_MM, Q4, 16384, 2048, P, 18); a.residual = true; v.push_back(a); }
    { OpDef a = pi05 ? op("e_norm", "adaRMS: rmsnorm*scale+shift", K_ADARMS, F32, 1024, 0, S, 370)
                     : op("e_norm", npu ? "rmsnorm*w (fused)" : "rmsnorm*w", K_RMSN, F32, 1024, 0, S, 370);
      a.small = npu; v.push_back(a); }
    { OpDef a = op("e_qkv", npu ? "fused mm + one rope [q4_0]" : "fused mm + 2 strided ropes [q4_0]", K_LM_QKV, Q4, 1024, 2560, S, 180);
      a.nh = 8; a.nkvh = 1; a.var = npu ? 2 : 1; v.push_back(a); }
    { const char * d = npu ? (pi05 ? "flash-attn f16 kv" : "flash-attn f16 kv + f16 mask")
                           : (pi05 ? "MQA-collapse QK/softmax/AV f16 kv" : "MQA-collapse f16 kv + tiled f32 mask");
      OpDef a = op("e_attn", d, npu ? K_FA_LM : K_ATTN_COLL, F16, LM_HD, KV, S, 180);
      a.nh = 8; a.nkvh = 1; a.nb = 1; a.mask = !pi05; v.push_back(a); }
    { OpDef a = op("e_out", pi05 ? "mm *gate +res [q4_0]" : "mm +res [q4_0]", K_MM, Q4, 2048, 1024, S, 180); a.residual = true; a.gated = pi05; v.push_back(a); }
    if (npu) { OpDef a = op("e_ffn_up", "gate/up GeGLU epilogue, f16 store [q4_0]", K_GEGLU, Q4, 1024, 4096, S, 180); a.out_f16 = true; v.push_back(a); }
    else     { v.push_back(op("e_ffn_up", "gate + up mm + geglu [q4_0]", K_GEGLU2, Q4, 1024, 4096, S, 180)); }
    { OpDef a = op("e_ffn_dn", npu ? (pi05 ? "mm f16 acts *gate +res [q4_0]" : "mm f16 acts +res [q4_0]") : (pi05 ? "mm *gate +res [q4_0]" : "mm +res [q4_0]"),
                   K_MM, Q4, 4096, 1024, S, 180);
      a.residual = true; a.in_f16 = npu; a.gated = pi05; v.push_back(a); }
    return v;
}

static bool g_verbose = false;

static double gflop_of(const OpDef & o) {
    const double K = (double) o.K, M = (double) o.M, N = (double) o.N;
    switch (o.kind) {
        case K_MM:         return (2.0 * M * K * N + (o.gelu ? 8.0 * M * N : 0.0)) / 1e9;
        case K_MM_KSPLIT:  return 2.0 * M * K * N / 1e9;
        case K_GEGLU:      return (2.0 * (2 * M) * K * N + (o.full ? 8.0 * M * N : 0.0)) / 1e9;
        case K_GEGLU2:     return (2.0 * (2 * M) * K * N + 8.0 * M * N) / 1e9;
        case K_FA: case K_VIT_ATTN: case K_ATTN_COLL: case K_FA_LM:
                           return 4.0 * N * M * K * o.nh * std::max(o.nb, 1) / 1e9;
        case K_LN: case K_RMSN: case K_ADARMS:
                           return 5.0 * K * N / 1e9;
        case K_GELU:       return 8.0 * K * N / 1e9;
        case K_ROPE:       return 6.0 * K * M * N / 1e9;
        case K_PATCH_CONV: case K_PATCH_FAST:
                           return 2.0 * M * K * N / 1e9;
        case K_QKV3:       return 3.0 * 2.0 * M * K * N / 1e9;
        case K_LN_MM:      return (5.0 * K * N + 2.0 * M * K * N) / 1e9;
        case K_LM_QKV:     return (2.0 * M * K * N + 6.0 * LM_HD * (o.nh + o.nkvh) * N) / 1e9;
    }
    return 0.0;
}

static std::string dims_of(const OpDef & o) {
    char b[64];
    const long long K = o.K, M = o.M, N = o.N;
    switch (o.kind) {
        case K_FA: case K_VIT_ATTN: case K_ATTN_COLL: case K_FA_LM:
            snprintf(b, sizeof(b), "seq=%lld kv=%lld h=%d/%d d=%lld b=%d", N, M, o.nh, o.nkvh, K, std::max(o.nb, 1)); break;
        case K_LN: case K_RMSN: case K_ADARMS: case K_GELU:
            snprintf(b, sizeof(b), "dim=%lld N=%lld", K, N); break;
        case K_ROPE:
            snprintf(b, sizeof(b), "d=%lld h=%lld seq=%lld", K, M, N); break;
        case K_GEGLU: case K_GEGLU2:
            snprintf(b, sizeof(b), "%lld->2x%lld N=%lld", K, M, N); break;
        case K_MM_KSPLIT:
            snprintf(b, sizeof(b), "%lld->%lld N=%lld ks=%d", K, M, N, o.nh); break;
        case K_PATCH_CONV:
            snprintf(b, sizeof(b), "%dx%dx%d->%lld N=%lld", VIT_C, VIT_PATCH, VIT_PATCH, M, N); break;
        default:
            snprintf(b, sizeof(b), "%lld->%lld N=%lld", K, M, N); break;
    }
    return b;
}

// ---- backend buffer types ---------------------------------------------------------------------------------------
static ggml_backend_buffer_type_t repack_buft(ggml_backend_t be) {
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return nullptr;
    typedef ggml_backend_buffer_type_t * (*get_extra_bufts_t)(ggml_backend_dev_t);
    auto fn = (get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!fn) return nullptr;
    ggml_backend_buffer_type_t * bufts = fn(dev);
    for (; bufts && *bufts; bufts++) {
        if (strstr(ggml_backend_buft_name(*bufts), "REPACK")) return *bufts;
    }
    return nullptr;
}

// ---- weights: one context per op, allocated once, resident for the whole process -------------------------------
struct OpWeights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> w;      // matmul weights (K-split: parts | GEGLU: interleaved gate/up | QKV3: q, k, v)
    std::vector<ggml_tensor *> b;      // biases, parallel to w
    ggml_tensor *         vec_w = nullptr, * vec_b = nullptr, * pos = nullptr;
};

static void fill_tensor(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> uni(-0.1f, 0.1f);
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> d(n); for (auto & x : d) x = uni(rng);
        ggml_backend_tensor_set(t, d.data(), 0, d.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> d(n); for (auto & x : d) x = ggml_fp32_to_fp16(uni(rng));
        ggml_backend_tensor_set(t, d.data(), 0, d.size() * sizeof(ggml_fp16_t));
    } else {   // quantized 2-D weight [K, M]
        const int64_t K = t->ne[0], M = t->ne[1];
        std::vector<float> f((size_t) K * M); for (auto & x : f) x = uni(rng);
        const size_t qsz = ggml_row_size(t->type, K) * M;
        std::vector<uint8_t> q(qsz);
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, M, K, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, qsz);
    }
}

// be == nullptr: create the tensors only (--dry)
static OpWeights make_weights(ggml_backend_t be, const OpDef & o, bool is_npu) {
    OpWeights W;
    W.ctx = ggml_init({ ggml_tensor_overhead() * 16, nullptr, true });
    auto mat = [&](int64_t k, int64_t m) { W.w.push_back(ggml_new_tensor_2d(W.ctx, o.wt, k, m)); };
    auto bia = [&](int64_t m) { W.b.push_back(ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, m)); };
    switch (o.kind) {
        case K_MM:
            mat(o.K, o.M);
            if (o.bias) bia(o.M);
            break;
        case K_MM_KSPLIT:
            for (int i = 0; i < o.nh; i++) mat(o.K / o.nh, o.M);
            break;
        case K_GEGLU:
            mat(o.K, 2 * o.M);
            break;
        case K_GEGLU2:
            mat(o.K, o.M); mat(o.K, o.M);
            break;
        case K_QKV3:
            for (int i = 0; i < 3; i++) { mat(o.K, o.M); bia(o.M); }
            break;
        case K_LN: case K_RMSN:
            W.vec_w = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, o.K);
            if (o.kind == K_LN) W.vec_b = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, o.K);
            break;
        case K_LN_MM:
            W.vec_w = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, o.K);
            W.vec_b = ggml_new_tensor_1d(W.ctx, GGML_TYPE_F32, o.K);
            mat(o.K, o.M); bia(o.M);
            break;
        case K_LM_QKV:
            if (o.var == 0) { mat(o.K, (int64_t) o.nh * LM_HD); mat(o.K, (int64_t) o.nkvh * LM_HD); mat(o.K, (int64_t) o.nkvh * LM_HD); }
            else            { mat(o.K, o.M); }
            break;
        case K_PATCH_CONV:   // [KW, KH, IC, OC] as in the mmproj GGUF
            W.w.push_back(ggml_new_tensor_4d(W.ctx, o.wt, VIT_PATCH, VIT_PATCH, VIT_C, o.M));
            bia(o.M);
            W.pos = ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, o.M, VIT_NPOS);
            break;
        case K_PATCH_FAST:   // the same kernel seen as [IC*KH*KW, OC]
            mat(o.K, o.M); bia(o.M);
            W.pos = ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, o.M, VIT_NPOS);
            break;
        default: break;
    }
    if (!be) return W;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    if (is_npu && o.wt == GGML_TYPE_Q4_0) {
        ggml_backend_buffer_type_t rb = repack_buft(be);
        if (rb) buft = rb; else LOG_ERR("%s: no REPACK buffer type on the NPU backend, q4_0 will NOT hit HMX\n", o.name);
    }
    if (ggml_get_first_tensor(W.ctx)) {
        W.buf = ggml_backend_alloc_ctx_tensors_from_buft(W.ctx, buft);
        if (!W.buf) { LOG_ERR("weight alloc failed for %s\n", o.name); exit(1); }
        std::mt19937 rng(123);
        for (ggml_tensor * t = ggml_get_first_tensor(W.ctx); t; t = ggml_get_next_tensor(W.ctx, t)) fill_tensor(t, rng);
        if (g_verbose) LOG_INF("weights %-10s -> %s\n", o.name, ggml_backend_buffer_name(W.buf));
    }
    return W;
}

// ---- graphs: activations + nodes, built per cell and freed after it ----------------------------------------------
struct OpG {
    ggml_context *        ctx = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_backend_t        be  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    bool                  supported = true;
    int                   reps = 1;      // copies of the op in the graph (work per submit = reps x op)
};

static ggml_tensor * new_input(ggml_context * ctx, ggml_type t, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    ggml_tensor * x = ggml_new_tensor_4d(ctx, t, n0, n1, n2, n3);
    ggml_set_input(x);
    return x;
}

static ggml_tensor * rope_neox(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, int n_dims) {
    return ggml_rope_ext(ctx, x, pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

// be == nullptr: build only (--dry), no allocation / support check / input upload
static OpG make_graph(ggml_backend_t be, const OpWeights & W, const OpDef & o, int reps = 1) {
    OpG g; g.be = be; g.reps = reps;
    g.ctx = ggml_init({ ggml_tensor_overhead() * 64 * reps + ggml_graph_overhead_custom(64 * reps, false), nullptr, true });
    ggml_context * ctx = g.ctx;
    std::vector<ggml_tensor *> inputs;
    g.gf = ggml_new_graph_custom(ctx, 64 * reps, false);
    auto in = [&](ggml_type t, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
        ggml_tensor * x = new_input(ctx, t, n0, n1, n2, n3); inputs.push_back(x); return x;
    };
    for (int rep = 0; rep < reps; rep++) {
    std::vector<ggml_tensor *> outs;   // expanded in this order (= the deployed graph order)
    switch (o.kind) {
        case K_MM: {
            ggml_tensor * x = in(o.in_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * out = ggml_mul_mat(ctx, W.w[0], x);
            if (o.bias)  out = ggml_add(ctx, out, W.b[0]);
            if (o.gelu)  out = ggml_gelu(ctx, out);
            if (o.gated) out = ggml_mul(ctx, out, in(GGML_TYPE_F32, o.M, 1));
            if (o.residual) { ggml_tensor * r = in(GGML_TYPE_F32, o.M, o.N); out = o.res_last ? ggml_add(ctx, out, r) : ggml_add(ctx, r, out); }
            outs.push_back(out);
        } break;
        case K_MM_KSPLIT: {
            ggml_tensor * h = in(o.in_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * acc = nullptr;
            if (o.residual) { acc = in(GGML_TYPE_F32, o.M, o.N); }
            const int64_t kh = o.K / o.nh;
            for (int i = 0; i < o.nh; i++) {
                ggml_tensor * hi = ggml_view_2d(ctx, h, kh, o.N, h->nb[1], (size_t) i * kh * ggml_element_size(h));
                ggml_tensor * pi = ggml_mul_mat(ctx, W.w[i], hi);
                acc = acc ? ggml_add(ctx, acc, pi) : pi;
            }
            outs.push_back(acc);
        } break;
        case K_GEGLU: {   // pi0-common pattern: MUL_MAT [2ff, N] -> 2 x view_3d -> geglu_split -> reshape [-> cast f16]
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            const int64_t ff = o.M;
            ggml_tensor * gu = ggml_mul_mat(ctx, W.w[0], x);
            ggml_tensor * gg = ggml_view_3d(ctx, gu, 32, ff / 32, o.N, 64 * sizeof(float), gu->nb[1], 0);
            ggml_tensor * uu = ggml_view_3d(ctx, gu, 32, ff / 32, o.N, 64 * sizeof(float), gu->nb[1], 32 * sizeof(float));
            ggml_tensor * out = ggml_reshape_2d(ctx, ggml_geglu_split(ctx, gg, uu), ff, o.N);
            if (o.out_f16) out = ggml_cast(ctx, out, GGML_TYPE_F16);
            outs.push_back(out);
        } break;
        case K_GEGLU2: {  // separate gate / up weights (OpenCL sets)
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * gate = ggml_mul_mat(ctx, W.w[0], x);
            ggml_tensor * up   = ggml_mul_mat(ctx, W.w[1], x);
            outs.push_back(ggml_geglu_split(ctx, gate, up));
        } break;
        case K_FA: {
            ggml_tensor * q, * k, * v;
            if (o.wt == GGML_TYPE_F32) {
                // SigLIP (native pad): q/k/v are the raw [hd, n_head, n_pos, batch] projection outputs handed over as
                // permute(0,2,1,3) views -> [hd, n_pos, n_head, batch] with the heads of one token contiguous
                // (the HMX f32-K/V path requires q->nb[1] >= q->nb[2])
                ggml_tensor * q0 = in(GGML_TYPE_F32, o.K, o.nh,   o.N, o.nb);
                ggml_tensor * k0 = in(GGML_TYPE_F32, o.K, o.nkvh, o.M, o.nb);
                ggml_tensor * v0 = in(GGML_TYPE_F32, o.K, o.nkvh, o.M, o.nb);
                q = ggml_permute(ctx, q0, 0, 2, 1, 3); k = ggml_permute(ctx, k0, 0, 2, 1, 3); v = ggml_permute(ctx, v0, 0, 2, 1, 3);
            } else {
                q = in(GGML_TYPE_F32, o.K, o.N, o.nh,   o.nb);
                k = in(o.wt,          o.K, o.M, o.nkvh, o.nb);
                v = in(o.wt,          o.K, o.M, o.nkvh, o.nb);
            }
            ggml_tensor * m = o.mask ? in(GGML_TYPE_F16, o.M, o.N) : nullptr;
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) o.K), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            outs.push_back(out);
        } break;
        case K_VIT_ATTN: {   // clip.cpp build_attn, non-flash path: q/k/v [hd, n_head, n_pos] -> PI0_VIT_IMG pad of K/Q
            ggml_tensor * qc = in(GGML_TYPE_F32, o.K, o.nh, o.N);
            ggml_tensor * kc = in(GGML_TYPE_F32, o.K, o.nh, o.M);
            ggml_tensor * vc = in(GGML_TYPE_F32, o.K, o.nh, o.M);
            ggml_tensor * q = ggml_permute(ctx, qc, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx, kc, 0, 2, 1, 3);
            ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, vc, 1, 2, 0, 3));
            const int pad = (int) ((16 - o.K % 16) % 16);
            ggml_tensor * kq = pad ? ggml_mul_mat(ctx, ggml_pad(ctx, k, pad, 0, 0, 0), ggml_pad(ctx, q, pad, 0, 0, 0))
                                   : ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf((float) o.K), 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
            ggml_tensor * cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            outs.push_back(ggml_cont_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]));
        } break;
        case K_LN: {
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            outs.push_back(ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), W.vec_w), W.vec_b));
        } break;
        case K_LN_MM: {
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * t = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), W.vec_w), W.vec_b);
            outs.push_back(ggml_add(ctx, ggml_mul_mat(ctx, W.w[0], t), W.b[0]));
        } break;
        case K_GELU: {
            outs.push_back(ggml_gelu(ctx, in(GGML_TYPE_F32, o.K, o.N)));
        } break;
        case K_RMSN: {
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            outs.push_back(ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), W.vec_w));
        } break;
        case K_ADARMS: {     // adaRMS with the per-step table leaves: rms_norm(x) * scale + shift, scale/shift [K, 1]
            ggml_tensor * x  = in(GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * sc = in(GGML_TYPE_F32, o.K, 1);
            ggml_tensor * sh = in(GGML_TYPE_F32, o.K, 1);
            outs.push_back(ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), sc), sh));
        } break;
        case K_ROPE: {
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.M, o.N);
            ggml_tensor * pos = in(GGML_TYPE_I32, o.N);
            outs.push_back(rope_neox(ctx, x, pos, (int) o.K));
        } break;
        case K_PATCH_CONV: {   // clip.cpp build_inp, conv_2d branch, one view
            ggml_tensor * img = in(GGML_TYPE_F32, VIT_IMG, VIT_IMG, VIT_C);
            ggml_tensor * x = ggml_conv_2d(ctx, W.w[0], img, VIT_PATCH, VIT_PATCH, 0, 0, 1, 1);
            x = ggml_reshape_2d(ctx, x, VIT_NPOS, o.M);
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            x = ggml_add(ctx, x, W.b[0]);
            outs.push_back(ggml_add(ctx, x, W.pos));
        } break;
        case K_PATCH_FAST: {   // clip.cpp build_inp, batched branch: the im2col rows (CPU) are the input
            ggml_tensor * i2c = in(GGML_TYPE_F32, o.K, o.N);
            const int64_t pad = (VIT_KPAD - o.K % VIT_KPAD) % VIT_KPAD;
            ggml_tensor * x  = pad ? ggml_pad(ctx, i2c,    (int) pad, 0, 0, 0) : i2c;
            ggml_tensor * w2 = pad ? ggml_pad(ctx, W.w[0], (int) pad, 0, 0, 0) : W.w[0];
            ggml_tensor * t = ggml_add(ctx, ggml_mul_mat(ctx, w2, x), W.b[0]);
            ggml_tensor * t3 = ggml_reshape_3d(ctx, t, o.M, o.N / o.nb, o.nb);
            ggml_tensor * p3 = ggml_reshape_3d(ctx, W.pos, o.M, VIT_NPOS, 1);
            outs.push_back(ggml_reshape_2d(ctx, ggml_add(ctx, t3, p3), o.M, o.N));
        } break;
        case K_QKV3: {
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            for (int i = 0; i < 3; i++) outs.push_back(ggml_add(ctx, ggml_mul_mat(ctx, W.w[i], x), W.b[i]));
        } break;
        case K_LM_QKV: {
            const int nh = o.nh, nkv = o.nkvh;
            ggml_tensor * x = in(GGML_TYPE_F32, o.K, o.N);
            ggml_tensor * pos = in(GGML_TYPE_I32, o.N);
            auto rope_rows = [&](ggml_tensor * t, int heads) {   // pi0-common apply_rope (contiguous or row-sliced view)
                ggml_tensor * r3 = ggml_is_contiguous(t) ? ggml_reshape_3d(ctx, t, LM_HD, heads, t->ne[1])
                    : ggml_view_3d(ctx, t, LM_HD, heads, t->ne[1], (size_t) LM_HD * ggml_element_size(t), t->nb[1], 0);
                return ggml_reshape_2d(ctx, rope_neox(ctx, r3, pos, LM_HD), (int64_t) heads * LM_HD, t->ne[1]);
            };
            if (o.var == 0) {          // unfused q/k/v, the K/V-cache writes (k, v) come first as in the deployed graph
                ggml_tensor * q = ggml_mul_mat(ctx, W.w[0], x);
                ggml_tensor * k = ggml_mul_mat(ctx, W.w[1], x);
                ggml_tensor * v = ggml_mul_mat(ctx, W.w[2], x);
                ggml_tensor * qr = rope_rows(q, nh);
                outs.push_back(rope_rows(k, nkv));
                outs.push_back(v);
                outs.push_back(qr);
            } else {
                ggml_tensor * qkv = ggml_mul_mat(ctx, W.w[0], x);
                if (o.var == 1) {      // row-sliced q/k views, one rope each
                    ggml_tensor * q = ggml_view_2d(ctx, qkv, (int64_t) nh  * LM_HD, o.N, qkv->nb[1], 0);
                    ggml_tensor * k = ggml_view_2d(ctx, qkv, (int64_t) nkv * LM_HD, o.N, qkv->nb[1], (size_t) nh * LM_HD * sizeof(float));
                    ggml_tensor * qr = rope_rows(q, nh);
                    outs.push_back(rope_rows(k, nkv));
                    outs.push_back(qr);
                } else {               // one rope over the leading [hd, nh + nkv] block (q heads + the k head)
                    ggml_tensor * qk = ggml_view_3d(ctx, qkv, LM_HD, nh + nkv, o.N, (size_t) LM_HD * sizeof(float), qkv->nb[1], 0);
                    outs.push_back(rope_neox(ctx, qk, pos, LM_HD));
                }
            }
        } break;
        case K_ATTN_COLL: {   // pi0-common standard attention with PI0_ATTN_COLLAPSE (one kv head), f16 cache
            const int64_t hd = o.K, kv = o.M, n = o.N;
            ggml_tensor * q = in(GGML_TYPE_F32, hd, o.nh, n);         // roped q [hd, n_head, seq]
            ggml_tensor * k = in(GGML_TYPE_F16, hd, kv, 1);
            ggml_tensor * v = in(GGML_TYPE_F16, hd, kv, 1);
            ggml_tensor * m = o.mask ? in(GGML_TYPE_F32, kv, n) : nullptr;
            ggml_tensor * q_std = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            ggml_tensor * v_t   = ggml_cont(ctx, ggml_transpose(ctx, v));
            ggml_tensor * q_coll = ggml_reshape_2d(ctx, q_std, hd, n * o.nh);
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q_coll);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            ggml_tensor * m_coll = m ? ggml_repeat(ctx, m, kq) : nullptr;
            kq = ggml_soft_max_ext(ctx, kq, m_coll, 1.0f / sqrtf((float) hd), 0.0f);
            ggml_tensor * kqv = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, v_t, kq), hd, n, o.nh);
            outs.push_back(ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), hd * o.nh, n));
        } break;
        case K_FA_LM: {       // pi0-common flash attention on HTP: Q = strided view of the one-rope output
            const int64_t hd = o.K, kv = o.M, n = o.N;
            ggml_tensor * qk = in(GGML_TYPE_F32, hd, o.nh + o.nkvh, n);
            ggml_tensor * q = ggml_view_2d(ctx, qk, hd * o.nh, n, qk->nb[2], 0);
            q = ggml_view_3d(ctx, q, hd, o.nh, n, (size_t) hd * sizeof(float), q->nb[1], 0);
            ggml_tensor * k = in(GGML_TYPE_F16, hd, kv, 1);
            ggml_tensor * v = in(GGML_TYPE_F16, hd, kv, 1);
            ggml_tensor * m = o.mask ? in(GGML_TYPE_F16, kv, n) : nullptr;
            ggml_tensor * out = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), k, v, m, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            outs.push_back(ggml_reshape_2d(ctx, out, hd * o.nh, n));
        } break;
    }
    for (ggml_tensor * out : outs) { ggml_set_output(out); ggml_build_forward_expand(g.gf, out); }
    }   // reps
    if (!be) return g;
    g.buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!g.buf) { LOG_ERR("graph alloc failed for %s on %s\n", o.name, ggml_backend_name(be)); g.supported = false; return g; }
    // support check AFTER allocation: the Hexagon supports_op looks at the operands' buffer types
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
        ggml_tensor * n = ggml_graph_node(g.gf, i);
        // views / reshapes / permutes are no-ops. ggml-hexagon fuses the GLU and the f16 CPY of the interleaved gate/up
        // pattern into the MUL_MAT, so they are never dispatched standalone.
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
            (o.kind == K_GEGLU && (n->op == GGML_OP_GLU || n->op == GGML_OP_CPY))) continue;
        if (!ggml_backend_supports_op(be, n)) {
            if (g_verbose) LOG_INF("%s: op %s (%s) not supported on %s\n", o.name, ggml_op_name(n->op), n->name, ggml_backend_name(be));
            g.supported = false;
        }
    }
    for (ggml_tensor * t : inputs) {
        const int64_t n = ggml_nelements(t);
        if (t->type == GGML_TYPE_F32)      { std::vector<float> d(n, 0.01f); ggml_backend_tensor_set(t, d.data(), 0, d.size() * sizeof(float)); }
        else if (t->type == GGML_TYPE_F16) { std::vector<ggml_fp16_t> d(n, ggml_fp32_to_fp16(o.mask ? 0.0f : 0.01f)); ggml_backend_tensor_set(t, d.data(), 0, d.size() * sizeof(ggml_fp16_t)); }
        else if (t->type == GGML_TYPE_I32) { std::vector<int32_t> d(n); for (int64_t i = 0; i < n; i++) d[i] = (int32_t) i; ggml_backend_tensor_set(t, d.data(), 0, d.size() * sizeof(int32_t)); }
    }
    return g;
}

static void free_graph(OpG & g) {
    if (g.buf) ggml_backend_buffer_free(g.buf);
    if (g.ctx) ggml_free(g.ctx);
    g.buf = nullptr; g.ctx = nullptr; g.gf = nullptr;
}

// ---- timing ---------------------------------------------------------------------------------------------------
// Foreground rate (GFLOPS) with an optional background graph looping on the other backend.
static double measure(OpG & fg, double c_op, OpG * bg, int iters) {
    const double c = c_op * fg.reps;
    std::atomic<bool> stop{ false };
    std::thread bgt;
    if (bg) {
        bgt = std::thread([&] {
            while (!stop.load(std::memory_order_relaxed)) { ggml_backend_graph_compute(bg->be, bg->gf); }
            ggml_backend_synchronize(bg->be);
        });
    }
    // Warm up under the background load for >= 300 ms so that the clocks have ramped. Then 3 timed passes of at least
    // `iters` calls and at least 200 ms each, so that short ops are not timed inside the clock ramp.
    const int64_t w0 = ggml_time_us();
    do { ggml_backend_graph_compute(fg.be, fg.gf); ggml_backend_synchronize(fg.be); } while (ggml_time_us() - w0 < 300000);
    double rates[3];
    for (int p = 0; p < 3; p++) {
        const int64_t t0 = ggml_time_us();
        int n = 0;
        do {
            for (int i = 0; i < iters; i++) ggml_backend_graph_compute(fg.be, fg.gf);
            ggml_backend_synchronize(fg.be);
            n += iters;
        } while (ggml_time_us() - t0 < 200000);
        const double sec = (ggml_time_us() - t0) / 1e6;
        rates[p] = sec > 0 ? c * n / sec : 0.0;
    }
    stop.store(true);
    if (bg) bgt.join();
    std::sort(rates, rates + 3);
    return rates[1];
}

static double g_rep_ms = 8.0;   // --rep-ms: target work per graph submit

static double submit_ms(OpG & g) {   // solo time of one submit (2 warm-up + 4 timed)
    for (int i = 0; i < 2; i++) ggml_backend_graph_compute(g.be, g.gf);
    ggml_backend_synchronize(g.be);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < 4; i++) ggml_backend_graph_compute(g.be, g.gf);
    ggml_backend_synchronize(g.be);
    return (ggml_time_us() - t0) / 4000.0;
}

// Build the op's graph with enough copies that one submit takes >= g_rep_ms (sized from a solo reps=1 graph).
// cap > CAP_COPIES (small NPU background ops under --small-bg): the copies are bounded by SMALL_BG_MAX_BYTES of graph
// memory, and the copy count is refined on the built graph until one submit lasts >= g_rep_ms, because the reps=1
// time of a tiny op is mostly per-submit overhead.
static OpG make_graph_amortised(ggml_backend_t be, const OpWeights & W, const OpDef & o, int cap = CAP_COPIES) {
    OpG g1 = make_graph(be, W, o, 1);
    if (!g1.supported || g_rep_ms <= 0) return g1;
    const double ms = submit_ms(g1);
    if (cap <= CAP_COPIES) {
        int reps = (int) std::min((double) cap, std::max(1.0, std::ceil(g_rep_ms / std::max(ms, 0.05))));
        if (reps == 1) return g1;
        free_graph(g1);
        OpG g = make_graph(be, W, o, reps);
        if (g_verbose) LOG_INF("%s on %s: %.2f ms/submit at reps=1 -> reps=%d\n", o.name, ggml_backend_name(be), ms, reps);
        return g;
    }
    const double lim = std::max((double) CAP_COPIES,
        std::min((double) cap, std::floor(SMALL_BG_MAX_BYTES / std::max((double) ggml_backend_buffer_get_size(g1.buf), 1.0))));
    int reps = (int) std::min(lim, std::max(1.0, std::ceil(g_rep_ms / std::max(ms, g_rep_ms / lim))));
    OpG g = g1;
    double t = ms;
    for (int it = 0; it < 4 && reps != g.reps; it++) {
        free_graph(g);
        g = make_graph(be, W, o, reps);
        if (!g.supported) return g;
        t = submit_ms(g);
        if (t >= g_rep_ms) break;
        reps = (int) std::min(lim, std::ceil(reps * 1.1 * g_rep_ms / std::max(t, 1e-3)));
    }
    if (g_verbose) LOG_INF("%s on %s: %.2f ms/submit at reps=1 -> reps=%d, %.2f ms/submit (small-bg cap %.0f)\n",
                           o.name, ggml_backend_name(be), ms, g.reps, t, lim);
    return g;
}

// ---- --rows / --cols: comma list of indices i and half-open ranges a:b (clamped to [0, n)) ------------------------
static bool parse_sel(const char * s, int n, std::vector<bool> & sel) {
    sel.assign(n, s == nullptr);
    if (!s) return true;
    const std::string str(s);
    if (str.empty()) return false;
    size_t p = 0;
    bool any = false;
    while (true) {
        const size_t e = str.find(',', p);
        const std::string tok = str.substr(p, e == std::string::npos ? std::string::npos : e - p);
        int a, b; char c;
        if (sscanf(tok.c_str(), "%d:%d%c", &a, &b, &c) == 2 && tok.find_first_not_of("0123456789:") == std::string::npos) {
            a = std::max(0, a); b = std::min(n, b);
            if (a >= b) return false;
            for (int i = a; i < b; i++) sel[i] = true;
        } else if (!tok.empty() && tok.find_first_not_of("0123456789") == std::string::npos) {
            a = atoi(tok.c_str());
            if (a >= n) return false;
            sel[a] = true;
        } else {
            return false;
        }
        any = true;
        if (e == std::string::npos) break;
        p = e + 1;
    }
    return any;
}

static int selftest_parse() {
    struct T { const char * s; int n; const char * want; };   // want: selected indices as "i,j,..", nullptr = reject
    static const T tests[] = {
        { "0:4", 18, "0,1,2,3" },        { "6:9,13", 18, "6,7,8,13" },   { "1,3,5", 18, "1,3,5" },
        { "16:18,0", 18, "0,16,17" },    { "0:4,6:9,9:11", 18, "0,1,2,3,6,7,8,9,10" },
        { "12", 20, "12" },              { "0:99", 6, "0,1,2,3,4,5" },  { "3,3", 6, "3" },
        { "", 18, nullptr },             { "a", 18, nullptr },          { "3:3", 18, nullptr },
        { "5:2", 18, nullptr },          { "1,,2", 18, nullptr },       { "18", 18, nullptr },
        { "-1", 18, nullptr },           { "1:2:3", 18, nullptr },      { "2,", 18, nullptr },
        { "0:4x", 18, nullptr },         { " 1", 18, nullptr },
    };
    int pass = 0, n = 0;
    for (const T & t : tests) {
        std::vector<bool> sel;
        const bool ok = parse_sel(t.s, t.n, sel);
        std::string got;
        if (ok) for (int i = 0; i < t.n; i++) if (sel[i]) got += (got.empty() ? "" : ",") + std::to_string(i);
        const bool good = t.want ? (ok && got == t.want) : !ok;
        printf("%-4s parse('%s', n=%d) -> %s\n", good ? "ok" : "FAIL", t.s, t.n, ok ? got.c_str() : "reject");
        pass += good; n++;
    }
    std::vector<bool> all;
    const bool all_ok = parse_sel(nullptr, 5, all) && std::count(all.begin(), all.end(), true) == 5;
    printf("%-4s parse(<none>, n=5) -> all\n", all_ok ? "ok" : "FAIL");
    pass += all_ok; n++;
    printf("selftest-parse: %d/%d %s\n", pass, n, pass == n ? "PASS" : "FAIL");
    return pass == n ? 0 : 1;
}

static void print_ops(const std::vector<OpDef> & ops, const char * dev) {
    for (const OpDef & o : ops) {
        printf("%-10s %-42s %-32s %11.6f %5d  [%s]\n", o.name, o.desc, dims_of(o).c_str(), gflop_of(o), o.count, dev);
    }
}

static void print_kernel_table(bool quick, const char * model, const std::vector<OpDef> & gpu, const std::vector<OpDef> & npu) {
    if (quick) printf("\n=== kernel types (QUICK: GPU set = SigLIP per view; NPU set = shipped HTP ops; pi0 shapes) ===\n");
    else       printf("\n=== kernel types (FULL %s: 20 atoms per device; GPU = OpenCL sessions, NPU = deployed HTP graph) ===\n", model);
    printf("%-10s %-42s %-32s %11s %5s\n", "type", "form", "dims", "cGFlop", "n");
    print_ops(gpu, "gpu");
    print_ops(npu, "npu");
    const char * small = quick ? "vis_ln vis_gelu pre_rmsn pre_rope" : "v_norm p_norm e_norm";
    printf("(cGFlop per call; n = calls per frame on that device; NPU background columns only with --small-bg: %s)\n", small);
}

static void usage(const char * argv0) {
    printf("usage: %s [--set quick|full] [--model pi0|pi05] --which g|h [--rows SEL] [--cols SEL]\n"
           "          [--iters N] [--rep-ms MS] [--small-bg] [--verbose] [--dry]\n"
           "       %s --list [--model pi0|pi05]\n"
           "       %s --selftest-parse\n"
           "  SEL: comma list of indices and half-open ranges a:b (e.g. 0:4,6,9:11); default all\n"
           "  g: GPU rows x NPU background cols; h: NPU rows x GPU background cols\n"
           "  --small-bg: measure the small NPU ops as background (DSP-queue hang risk; default n/a)\n"
           "  --dry: build the graphs without a backend and print the chunk plan (dry = would be measured)\n",
           argv0, argv0, argv0);
}

int main(int argc, char ** argv) {
    int iters = 12;
    std::string which, set = "quick", model = "pi0";
    const char * rows = nullptr, * cols = nullptr;
    bool list = false, dry = false, small_bg = false;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        const bool has = i + 1 < argc;
        if      (strcmp(a, "--iters") == 0 && has)  iters = std::atoi(argv[++i]);
        else if (strcmp(a, "--which") == 0 && has)  which = argv[++i];
        else if (strcmp(a, "--rows") == 0 && has)   rows = argv[++i];
        else if (strcmp(a, "--cols") == 0 && has)   cols = argv[++i];
        else if (strcmp(a, "--set") == 0 && has)    set = argv[++i];
        else if (strcmp(a, "--model") == 0 && has)  model = argv[++i];
        else if (strcmp(a, "--rep-ms") == 0 && has) g_rep_ms = std::atof(argv[++i]);
        else if (strcmp(a, "--verbose") == 0)       g_verbose = true;
        else if (strcmp(a, "--small-bg") == 0)      small_bg = true;
        else if (strcmp(a, "--list") == 0)          list = true;
        else if (strcmp(a, "--dry") == 0)           dry = true;
        else if (strcmp(a, "--selftest-parse") == 0) return selftest_parse();
        else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown or incomplete argument: %s\n", a); usage(argv[0]); return 1; }
    }
    if ((set != "quick" && set != "full") || (model != "pi0" && model != "pi05") || iters < 1) {
        fprintf(stderr, "bad --set / --model / --iters\n"); usage(argv[0]); return 1;
    }
    const bool quick = set == "quick", pi05 = model == "pi05";
    const std::vector<OpDef> q_gpu(std::begin(QUICK_GPU), std::end(QUICK_GPU)), q_npu(std::begin(QUICK_NPU), std::end(QUICK_NPU));
    const std::vector<OpDef> f_gpu = full_ops(false, pi05), f_npu = full_ops(true, pi05);
    if (list) {
        print_kernel_table(true, "pi0", q_gpu, q_npu);
        print_kernel_table(false, model.c_str(), f_gpu, f_npu);
        return 0;
    }
    if (which != "g" && which != "h") { fprintf(stderr, "--which g|h is required\n"); usage(argv[0]); return 1; }
    if (quick && pi05) fprintf(stderr, "note: --set quick always uses pi0 shapes (--model only selects the FULL shapes)\n");
    const std::vector<OpDef> & GPU = quick ? q_gpu : f_gpu;
    const std::vector<OpDef> & NPU = quick ? q_npu : f_npu;
    const int n_gpu = (int) GPU.size(), n_npu = (int) NPU.size();
    const bool do_g = which == "g";

    // rows = foreground indices, cols = background indices (per device list)
    const int n_fg = do_g ? n_gpu : n_npu, n_bg = do_g ? n_npu : n_gpu;
    std::vector<bool> rsel, csel;
    if (!parse_sel(rows, n_fg, rsel) || !parse_sel(cols, n_bg, csel)) { fprintf(stderr, "bad --rows/--cols (SEL = i,a:b,...; indices < %d / %d)\n", n_fg, n_bg); return 1; }
    // a background column is measured when selected and not a small NPU op without --small-bg
    auto bg_on = [&](int j) { return csel[j] && !(do_g && NPU[j].small && !small_bg); };
    auto bg_cap = [&](int j) { return (do_g && NPU[j].small && small_bg) ? CAP_COPIES_SMALL_BG : CAP_COPIES; };

    std::string rs = rows ? rows : "all", cs = cols ? cols : "all";
    printf("# llama-pi0-opmat-bench set=%s model=%s which=%s rows=%s cols=%s iters=%d rep-ms=%.1f small-bg=%d%s\n",
           set.c_str(), quick ? "pi0" : model.c_str(), which.c_str(), rs.c_str(), cs.c_str(), iters, g_rep_ms, small_bg ? 1 : 0, dry ? " dry" : "");

    ggml_backend_t be_ocl = nullptr, be_htp = nullptr;
    if (!dry) {
        ggml_backend_load_all();
        be_ocl = pi0_init_backend("OpenCL");
        be_htp = pi0_init_backend("HTP0");
        if (!be_ocl || !be_htp) { LOG_ERR("need BOTH OpenCL and HTP0 backends (ocl=%p htp=%p)\n", (void *) be_ocl, (void *) be_htp); return 1; }
        LOG_INF("OpenCL + HTP0 up; set=%s, %d GPU ops x %d NPU ops, iters=%d, which=%s\n", set.c_str(), n_gpu, n_npu, iters, which.c_str());
    }

    std::vector<OpWeights> W_gpu(n_gpu), W_npu(n_npu);
    std::vector<bool> need_gpu(n_gpu, false), need_npu(n_npu, false);
    for (int i = 0; i < n_fg; i++) if (rsel[i]) (do_g ? need_gpu : need_npu)[i] = true;
    for (int j = 0; j < n_bg; j++) if (bg_on(j)) (do_g ? need_npu : need_gpu)[j] = true;
    for (int i = 0; i < n_gpu; i++) if (need_gpu[i]) W_gpu[i] = make_weights(be_ocl, GPU[i], false);
    for (int i = 0; i < n_npu; i++) if (need_npu[i]) W_npu[i] = make_weights(be_htp, NPU[i], true);

    print_kernel_table(quick, model.c_str(), GPU, NPU);

    if (dry) {   // graph construction check (same builders, no backend) + the node sequence with --verbose
        auto show = [&](const OpDef & o, const OpWeights & W, const char * dev) {
            OpG g = make_graph(nullptr, W, o, 1);
            std::string seq;
            for (int k = 0; k < ggml_graph_n_nodes(g.gf); k++) {
                const ggml_tensor * n = ggml_graph_node(g.gf, k);
                if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
                seq += std::string(seq.empty() ? "" : " ") + (n->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(n)) :
                                                              n->op == GGML_OP_GLU   ? ggml_glu_op_name(ggml_get_glu_op(n)) : ggml_op_name(n->op));
                if (n->type != GGML_TYPE_F32) seq += std::string("(") + ggml_type_name(n->type) + ")";
            }
            if (g_verbose) printf("# graph %-9s [%s] %s\n", o.name, dev, seq.c_str());
            free_graph(g);
        };
        for (int i = 0; i < n_gpu; i++) if (need_gpu[i]) show(GPU[i], W_gpu[i], "gpu");
        for (int i = 0; i < n_npu; i++) if (need_npu[i]) show(NPU[i], W_npu[i], "npu");
    }

    auto fmt = [&](double v, bool ok) {
        char b[16];
        if (dry && ok) snprintf(b, sizeof(b), "%8s", "dry");
        else if (ok && v > 0) snprintf(b, sizeof(b), v >= 100 ? "%8.0f" : v >= 10 ? "%8.1f" : "%8.2f", v);   // >= 3 significant digits
        else snprintf(b, sizeof(b), "%8s", "n/a");
        return std::string(b);
    };

    // g_ij: GPU op i (fg, rows) while NPU op j (bg, cols) loops. h_ij: NPU op i while GPU op j loops. 'idle' = solo.
    const std::vector<OpDef> & FG = do_g ? GPU : NPU, & BG = do_g ? NPU : GPU;
    std::vector<OpWeights> & W_fg = do_g ? W_gpu : W_npu, & W_bg = do_g ? W_npu : W_gpu;
    ggml_backend_t be_fg = do_g ? be_ocl : be_htp, be_bg = do_g ? be_htp : be_ocl;
    if (do_g) printf("\n=== g_ij : GPU(OpenCL) %sop i while NPU runs op j  (GFLOPS; iters=%d, median of 3) ===\n", quick ? "vision " : "", iters);
    else      printf("\n=== h_ij : NPU(HTP0) op i while GPU runs %sop j  (GFLOPS; iters=%d, median of 3) ===\n", quick ? "vision " : "", iters);
    printf("%-10s %8s", "fg \\ bg", "idle");
    for (int j = 0; j < n_bg; j++) printf(" %8s", BG[j].name);
    printf("\n");
    for (int i = 0; i < n_fg; i++) {
        if (!rsel[i]) continue;
        const double c = gflop_of(FG[i]);
        OpG fg;
        if (!dry) fg = make_graph_amortised(be_fg, W_fg[i], FG[i]);
        printf("%-10s %s", FG[i].name, fmt(!dry && fg.supported ? measure(fg, c, nullptr, iters) : 0, fg.supported).c_str());
        fflush(stdout);
        for (int j = 0; j < n_bg; j++) {
            if (!bg_on(j) || !fg.supported) { printf(" %8s", "n/a"); continue; }
            if (dry) { printf(" %s", fmt(0, true).c_str()); continue; }
            OpG bg = make_graph_amortised(be_bg, W_bg[j], BG[j], bg_cap(j));
            printf(" %s", fmt(bg.supported ? measure(fg, c, &bg, iters) : 0, bg.supported).c_str());
            free_graph(bg);
            fflush(stdout);
        }
        printf("\n"); fflush(stdout);
        free_graph(fg);
    }
    printf("\n(col 'idle' = other device idle; n/a = not measured in this chunk, unsupported, or a small NPU background op without --small-bg)\n");
    return 0;
}
