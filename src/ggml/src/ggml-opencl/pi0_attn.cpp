// PI0 attention and ViT GEMMs for the Adreno OpenCL backend.
//
// Routes the two collapsed-MQA attention GEMMs through the f16-weight image-B kernel
// (kernel_mul_mat_Ab_Bi_8x4_f16, the f16 analog of the q4_0 Ab_Bi GEMM):
//     QK = mul_mat(K_cache, Q)     -> scores [n_kv, n_q*n_head]
//     AV = mul_mat(V_t,    scores) -> out    [head_dim, n_q*n_head]
// The "weights" are the f16 KV cache. The activations (Q / scores) are read through an image1d_buffer.
//
// This file is #included into ggml-opencl.cpp and uses that file's context struct, kernel handles
// and helpers. CMake does not compile it standalone. All paths are opt-in and off by default.
//
// The Ab_Bi kernel wants src0 as [K][M] (M-contiguous) and src1 as the transposed f16 "Bi" image.
// Both operands are [head_dim, seq] (reduction-contiguous), so both are transposed per call:
//     src0 (f16) -> kernel_transpose_16_buf -> A_T buffer [K][M] (k*M + m)
//     src1 (f32) -> kernel_transpose_32_16  -> B image (f16, Bi, N zero-padded to a multiple of 8)

// PI0_ATTN_IMG: 0/unset = off, any other value = route both QK and AV.
static bool pi0_attn_img_on() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("PI0_ATTN_IMG"); v = (e && atoi(e) != 0) ? 1 : 0; }
    return v != 0;
}

// Eligible = a PI0 collapsed attention GEMM: f16 src0, f32 src1, 2D (no batch), contiguous,
// N = dst->ne[1] >= 32, and one of
//   QK: ne00 (reduction = head_dim) == 256, ne01 (M = n_kv)         >= 64
//   AV: ne01 (M = head_dim)         == 256, ne00 (reduction = n_kv) >= 64
// The head_dim == 256 and 2D gates exclude the batched ViT attention and the FFN/projection GEMMs.
static bool pi0_attn_img_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!pi0_attn_img_on()) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) dst->ne[1];
    if (N < 32) return false;
    const bool qk = (K == 256 && M >= 64);   // reduce over head_dim
    const bool av = (M == 256 && K >= 64);   // M = head_dim
    return qk || av;
}

// PI0_VIT_IMG: route ViT (SigLIP) projection/FFN GEMMs through the same f16 image-B kernel.
// Weights stay f16. The 1D launch handles any M and the Kt padding below handles any K, so the
// weights need no padding. 0 or unset = off, 1 = on.
static int pi0_vit_img_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("PI0_VIT_IMG"); v = e ? atoi(e) : 0; }
    return v;
}

// Eligible = a ViT proj/FFN GEMM: f16 src0 (weight), f32 src1 (activations), 2D (no batch),
// contiguous, N = dst->ne[1] >= 32. M and K in [512, 8192] keep this to the SigLIP width/MLP
// GEMMs (1152/4304). This excludes attention (head_dim = 72) and lm_head (M = vocab).
// Quantized LM/expert weights fail the f16 type check.
static bool pi0_vit_img_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (pi0_vit_img_mode() == 0) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) dst->ne[1];
    return M >= 512 && M <= 8192 && K >= 512 && K <= 8192 && N >= 32;
}

// ---------------------------------------------------------------------------------------------------------------
// PI0_VIT_XMEM=1 (on top of PI0_VIT_IMG=1): static SigLIP projection / FFN weights go through the Adreno constant-RAM
// f16 GEMM (kernels/gemm_xmem_f16_f32_os8.cl, uniform slice, lws 128) instead of the Ab_Bi f16 kernel. The weight
// operand of every FMA is read from constant RAM, so it needs no register load. Both kernels accumulate in fp16.
//   * weights: rewritten once, in place, to the constant-load layout (same byte size because M % 32 == 0), as
//     PI0_VIT_WEIGHT_PRETRANS does for its transpose. No second copy of the weights is kept.
//   * activations / result: RGBA half image2d [N x K/4] / [N x M/4], one cached image per distinct size.
//   * per call: pack_src -> GEMM -> store, back to back on the in-order queue. PI0_VIT_XMEM=2 drops the store
//     pass and the result image (the GEMM writes f32 itself).
// Off by default.
// ---------------------------------------------------------------------------------------------------------------
// 0 or unset = off, 1 = result through a half image and a store pass, 2 = the GEMM stores f32 itself.
static int pi0_vit_xmem_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("PI0_VIT_XMEM"); v = e ? atoi(e) : 0; }
    return v;
}

// Eligible = a ViT weight GEMM (static f16 leaf, the PI0_VIT_IMG shape gate) that the xmem kernel can take:
// K % 8 == 0 (two K-texels per step), M % 32 == 0 (in-place layout, no padding rows), N >= 128 (one work-group spans
// 128 columns), image heights within the device limit.
static bool pi0_vit_xmem_eligible(ggml_backend_opencl_context * backend_ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (pi0_vit_xmem_mode() == 0 || backend_ctx->kernel_pi0_xmem_gemm == nullptr || backend_ctx->pi0_xmem_max_image_h == 0) return false;
    if (src0->op != GGML_OP_NONE || !pi0_vit_img_eligible(src0, src1, dst)) return false;
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) dst->ne[1];
    if (K % 8 != 0 || M % 32 != 0 || N < 128) return false;
    return (size_t) (M / 4) <= backend_ctx->pi0_xmem_max_image_h && (size_t) (K / 4) <= backend_ctx->pi0_xmem_max_image_h;
}

static void ggml_cl_mul_mat_pi0_vit_xmem(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    cl_context context = backend_ctx->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int K = (int) src0->ne[0];  // reduction
    int M = (int) src0->ne[1];  // output rows
    int N = (int) dst->ne[1];   // output columns (tokens)
    int kpack = K / 4;
    int npack = M / 4;          // M % 32 == 0: no padding rows, npack % 8 == 0
    int os    = 8;              // 8 half4 accumulators = 32 output rows per fiber

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("pi0: ViT xmem GEMM active (M=%d K=%d N=%d)\n", M, K, N);
    }

    cl_int           status;
    cl_buffer_region region;
    region.origin = offset0;
    region.size   = (size_t) K * M * sizeof(ggml_fp16_t);
    cl_mem w_sub = clCreateSubBuffer(extra0->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);

    if (!extra0->pi0_wt_xmem) {
        // First use of this weight: pack [M][K] f16 into the constant-load layout in the A_T scratch, then copy it back
        // over the weight. The in-order queue runs prepack -> copy -> GEMM, so the scratch is free for the next call.
        backend_ctx->prealloc_pi0_at.allocate(context, region.size + 64);
        cl_mem   packed = backend_ctx->prealloc_pi0_at.buffer;
        cl_ulong zero   = 0;
        cl_kernel pk = backend_ctx->kernel_pi0_xmem_prepack;
        CL_CHECK(clSetKernelArg(pk, 0, sizeof(cl_mem),   &packed));
        CL_CHECK(clSetKernelArg(pk, 1, sizeof(cl_mem),   &w_sub));
        CL_CHECK(clSetKernelArg(pk, 2, sizeof(cl_ulong), &zero));
        CL_CHECK(clSetKernelArg(pk, 3, sizeof(int),      &K));
        CL_CHECK(clSetKernelArg(pk, 4, sizeof(int),      &M));
        CL_CHECK(clSetKernelArg(pk, 5, sizeof(int),      &kpack));
        CL_CHECK(clSetKernelArg(pk, 6, sizeof(int),      &npack));
        CL_CHECK(clSetKernelArg(pk, 7, sizeof(int),      &os));
        size_t pg[1] = { (size_t) kpack * (size_t) npack };
        backend_ctx->enqueue_ndrange_kernel(pk, 1, pg, NULL, dst);
        CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, packed, w_sub, 0, 0, region.size, 0, NULL, NULL));
        extra0->pi0_wt_xmem = true;
    }

    // mode 2: the GEMM writes f32 [N][M] itself -> no result image, no store pass
    const bool direct = pi0_vit_xmem_mode() == 2 && backend_ctx->kernel_pi0_xmem_gemm_buf != nullptr;
    cl_mem src_img = backend_ctx->pi0_xmem_src_imgs.get(context, (size_t) N, (size_t) kpack);
    cl_mem dst_img = direct ? nullptr : backend_ctx->pi0_xmem_dst_imgs.get(context, (size_t) N, (size_t) npack);
    cl_mem C_d     = nullptr;
    if (direct) {
        region.origin = offsetd;
        region.size   = (size_t) M * N * sizeof(float);
        C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        CL_CHECK(status);
    }

    // activations: f32 [N][K] -> half image [N x K/4]
    {
        cl_kernel ps = backend_ctx->kernel_pi0_xmem_pack_src;
        CL_CHECK(clSetKernelArg(ps, 0, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(ps, 1, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(ps, 2, sizeof(cl_mem),   &src_img));
        CL_CHECK(clSetKernelArg(ps, 3, sizeof(int),      &K));
        CL_CHECK(clSetKernelArg(ps, 4, sizeof(int),      &N));
        size_t lws[2] = { 16, 16 };
        size_t gws[2] = { (((size_t) N + 15) / 16) * 16, (((size_t) kpack + 15) / 16) * 16 };
        backend_ctx->enqueue_ndrange_kernel(ps, 2, gws, lws, dst);
    }

    // GEMM: one fiber = one column x 32 rows. A 128-fiber work-group shares one constant-RAM slice (slice_mode 3: the
    // slot of sub-group 0 is published through slot_tbl and read back, which makes the slice address work-group-uniform).
    {
        backend_ctx->prealloc_pi0_xmem_const.allocate(context, 6144);
        int nmajor = 0, slice_mode = 3;
        const size_t lws_n    = 128;
        const size_t z_values = (size_t) (npack + os - 1) / (size_t) os;
        size_t lws[3] = { lws_n, 1, 1 };
        size_t gws[3] = { z_values * lws_n, ((size_t) N + lws_n - 1) / lws_n, 1 };
        backend_ctx->prealloc_pi0_xmem_slot.allocate(context, z_values * gws[1] * sizeof(cl_int) + 64);
        cl_kernel gk = direct ? backend_ctx->kernel_pi0_xmem_gemm_buf : backend_ctx->kernel_pi0_xmem_gemm;
        CL_CHECK(clSetKernelArg(gk, 0, sizeof(cl_mem), &w_sub));
        CL_CHECK(clSetKernelArg(gk, 1, sizeof(cl_mem), &backend_ctx->prealloc_pi0_xmem_const.buffer));
        CL_CHECK(clSetKernelArg(gk, 2, sizeof(cl_mem), &src_img));
        CL_CHECK(clSetKernelArg(gk, 3, sizeof(cl_mem), direct ? &C_d : &dst_img));
        CL_CHECK(clSetKernelArg(gk, 4, sizeof(int),    &N));
        CL_CHECK(clSetKernelArg(gk, 5, sizeof(int),    &npack));
        CL_CHECK(clSetKernelArg(gk, 6, sizeof(int),    &kpack));
        CL_CHECK(clSetKernelArg(gk, 7, sizeof(int),    &nmajor));
        CL_CHECK(clSetKernelArg(gk, 8, sizeof(int),    &slice_mode));
        CL_CHECK(clSetKernelArg(gk, 9, sizeof(cl_mem), &backend_ctx->prealloc_pi0_xmem_slot.buffer));
        backend_ctx->enqueue_ndrange_kernel(gk, 3, gws, lws, dst);
    }

    // result: half image [N x M/4] -> f32 [N][M] (mode 1 only)
    if (!direct) {
        cl_kernel st = backend_ctx->kernel_pi0_xmem_store;
        CL_CHECK(clSetKernelArg(st, 0, sizeof(cl_mem),   &dst_img));
        CL_CHECK(clSetKernelArg(st, 1, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(st, 2, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(st, 3, sizeof(int),      &M));
        CL_CHECK(clSetKernelArg(st, 4, sizeof(int),      &N));
        size_t lws[2] = { 16, 16 };
        size_t gws[2] = { (((size_t) N + 15) / 16) * 16, (((size_t) npack + 15) / 16) * 16 };
        backend_ctx->enqueue_ndrange_kernel(st, 2, gws, lws, dst);
    }

    if (C_d) { CL_CHECK(clReleaseMemObject(C_d)); }
    CL_CHECK(clReleaseMemObject(w_sub));
}

// f16 x f32 GEMM through the f16 image-B kernel, for the PI0 attention and ViT GEMMs.
static void ggml_cl_mul_mat_pi0_attn_img(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    cl_context context = backend_ctx->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;
    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offset1 = extra1->offset + src1->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    // PI0_VIT_XMEM: a static ViT weight takes the constant-RAM kernel instead. A weight keeps one layout for its whole
    // life. One that PI0_VIT_WEIGHT_PRETRANS already transposed stays on this path. One in the xmem layout must never
    // reach the Ab_Bi kernel below (it would be read as a transposed matrix).
    if (!extra0->pi0_wt_transposed && pi0_vit_xmem_eligible(backend_ctx, src0, src1, dst)) {
        ggml_cl_mul_mat_pi0_vit_xmem(backend, src0, src1, dst);
        return;
    }
    GGML_ASSERT(!extra0->pi0_wt_xmem && "PI0_VIT_XMEM: weight is in the xmem layout but this GEMM is not xmem-eligible");

    const int M = (int) src0->ne[1]; // output rows  (QK: n_kv, AV: head_dim)
    const int K = (int) src0->ne[0]; // reduction    (QK: head_dim, AV: n_kv)
    const int N = (int) dst->ne[1];  // output cols  (n_q * n_head)

    cl_int           status;
    cl_buffer_region region;

    // --- src0 (f16 KV) -> A_T [K][M] M-contiguous, via kernel_transpose_16_buf ---
    // out[x*ldo + y] = in[y*ldi + x], x in [0,K), y in [0,M). ldi=K, ldo=M -> A_T[k*M + m].
    // +64 pad bytes so a partial M-tile (M % 4 != 0) can over-read up to 3 halfs.
    region.origin = offset0;
    region.size   = (size_t) K * M * sizeof(ggml_fp16_t);
    cl_mem src0_sub = clCreateSubBuffer(extra0->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // A_T is a reused grow-once scratch, not a per-call buffer. The in-order queue serializes
    // transpose -> GEMM -> next transpose, so one buffer is safe to reuse (as prealloc_act_trans).
    backend_ctx->prealloc_pi0_at.allocate(context, region.size + 64);
    cl_mem A_T = backend_ctx->prealloc_pi0_at.buffer;

    // PI0_VIT_WEIGHT_PRETRANS=1: a ViT proj/FFN weight is a static f16 leaf, so its [K][M] transpose is
    // loop-invariant. Transpose it once and store it in place in the weight's own device buffer (as the
    // q4_0 path does for its quant/scale buffers). Only weights qualify (op == NONE and ViT-shaped).
    // Attention KV is a dynamic activation and keeps the per-call transpose. Off by default.
    static const bool wt_pretrans = [] { const char * e = getenv("PI0_VIT_WEIGHT_PRETRANS"); return e && e[0] == '1'; }();
    const bool is_vit_weight = wt_pretrans && src0->op == GGML_OP_NONE && pi0_vit_img_eligible(src0, src1, dst);
    cl_mem A_gemm = A_T;   // default: GEMM reads A from the freshly transposed scratch

    if (is_vit_weight && extra0->pi0_wt_transposed) {
        // Transposed on a previous forward. src0_sub already holds [K][M].
        A_gemm = src0_sub;
    } else {
        int ldi = K, ldo = M;
        cl_kernel tk = backend_ctx->kernel_transpose_16_buf;
        CL_CHECK(clSetKernelArg(tk, 0, sizeof(cl_mem), &src0_sub));
        CL_CHECK(clSetKernelArg(tk, 1, sizeof(cl_mem), &A_T));
        CL_CHECK(clSetKernelArg(tk, 2, sizeof(int),    &ldi));
        CL_CHECK(clSetKernelArg(tk, 3, sizeof(int),    &ldo));
        size_t tg[3] = { (size_t) K, (size_t) M, 1 };
        backend_ctx->enqueue_ndrange_kernel(tk, 2, tg, NULL, dst);
        if (is_vit_weight) {
            // Persist the transpose: copy A_T back into the weight's own buffer (once, first forward).
            CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, A_T, src0_sub, 0, 0,
                (size_t) K * M * sizeof(ggml_fp16_t), 0, NULL, NULL));
            extra0->pi0_wt_transposed = true;
            A_gemm = src0_sub;
        }
    }

    // --- src1 (f32 Q/scores) -> B image (f16 Bi, N zero-padded), via kernel_transpose_32_16 ---
    // The B image packs 4 reduction elements per RGBA texel, so the input row width must be a multiple of 4.
    // When K % 4 != 0, src1 is first copied into a K-padded temp [N][Kt], Kt = ceil(K/4)*4. The GEMM still
    // reduces over the real K (k arg below), so the pad columns are never read and need not be zeroed.
    // PI0_VIT_2D=1: 2D-launch f16 kernel (one N-tile per work-group). The kernel guards its own M tail, so
    // any M % 4 == 0 works without weight padding.
    static const bool vit_2d = [] { const char * e = getenv("PI0_VIT_2D"); return e && e[0] == '1'; }();
    const int npad_mult = 8;
    const int extra_elements = N % npad_mult;
    const int padding = extra_elements > 0 ? npad_mult - extra_elements : 0;
    const int Kt = (K + 3) & ~3; // reduction width rounded up to a multiple of 4 (for the texels)
    region.origin = offset1;
    region.size   = (size_t) K * N * sizeof(float);
    cl_mem B_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    cl_mem B_in_buf = B_sub;   // buffer the input image views (B_sub, or the K-padded temp)
    cl_mem B_pad    = NULL;
    if (Kt != K) {
        B_pad = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) Kt * N * sizeof(float), NULL, &status);
        CL_CHECK(status);
        // strided copy: N rows of K floats from src1 [N][K] into temp [N][Kt] (pad cols unread).
        size_t src_origin[3] = {0,0,0}, dst_origin[3] = {0,0,0};
        size_t rgn[3] = { (size_t) K * sizeof(float), (size_t) N, 1 };
        CL_CHECK(clEnqueueCopyBufferRect(backend_ctx->queue, B_sub, B_pad, src_origin, dst_origin, rgn,
            (size_t) K * sizeof(float), 0, (size_t) Kt * sizeof(float), 0, 0, NULL, NULL));
        B_in_buf = B_pad;
    }
    region.origin = 0;
    region.size   = (size_t) Kt * (N + padding) * sizeof(float) / 2; // /2 for FP16
    backend_ctx->prealloc_act_trans.allocate(context, region.size);
    cl_mem B_d = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    cl_image_format fmt_in  = { CL_RGBA, CL_FLOAT };
    cl_image_desc   desc_in = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)(Kt * N / 4), 0,0,0,0,0,0,0, { B_in_buf } };
    cl_mem B_in_img = clCreateImage(context, 0, &fmt_in, &desc_in, NULL, &status);
    CL_CHECK(status);
    cl_image_format fmt_out  = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc   desc_out = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)(Kt * (N + padding) / 4), 0,0,0,0,0,0,0, { B_d } };
    cl_mem B_image1d = clCreateImage(context, 0, &fmt_out, &desc_out, NULL, &status);
    CL_CHECK(status);
    {
        int height_B = N / 4; if (height_B == 0) { height_B = 1; }
        int width_B  = Kt / 4;
        int padded_height_B = (N + padding) / 4;
        cl_kernel tk = backend_ctx->kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(tk, 0, sizeof(cl_mem), &B_in_img));
        CL_CHECK(clSetKernelArg(tk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(tk, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(tk, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(tk, 4, sizeof(int),    &padded_height_B));
        size_t tl[2] = { 1, 16 };
        size_t tg[2] = { (size_t) width_B, (size_t) padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(tk, 2, tg, tl, dst);
    }

    // --- C output sub-buffer ---
    region.origin = offsetd;
    region.size   = (size_t) M * N * sizeof(float);
    cl_mem C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);

    // --- GEMM ---
    // 1D-flattened launch: total fibers = ceil(N/8) * ceil(M/4), rounded up to lws=128. Only the
    // last work-group is partial. The kernel maps flat id -> (gy, gx) and returns early for gy >= n_tiles.
    const int padded_N = N + padding;
    const size_t lws1    = 128;
    const size_t m_tiles = (size_t) ((M + 3) / 4);
    const size_t n_tiles = (size_t) ((N + 7) / 8);
    const size_t total   = m_tiles * n_tiles;
    const size_t wgs     = (total + lws1 - 1) / lws1;

    // Split-K for shapes with few work-groups: split the K reduction k_split ways along a second
    // grid dim, then sum the partials. Gate: fewer than 48 work-groups and K >= 256.
    int k_split = 1;
    if (wgs < 48 && K >= 256) {
        k_split = 2;
    }

    if (k_split > 1) {
        const int plane = M * N;
        cl_mem partials = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) k_split * plane * sizeof(float), NULL, &status);
        CL_CHECK(status);
        cl_kernel sk = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f16_splitk;
        CL_CHECK(clSetKernelArg(sk, 0, sizeof(cl_mem), &A_gemm));
        CL_CHECK(clSetKernelArg(sk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(sk, 2, sizeof(cl_mem), &partials));
        CL_CHECK(clSetKernelArg(sk, 3, sizeof(int),    &M));
        CL_CHECK(clSetKernelArg(sk, 4, sizeof(int),    &padded_N));
        CL_CHECK(clSetKernelArg(sk, 5, sizeof(int),    &K));
        CL_CHECK(clSetKernelArg(sk, 6, sizeof(int),    &N));
        CL_CHECK(clSetKernelArg(sk, 7, sizeof(int),    &k_split));
        size_t sk_gws[2] = { ((total + lws1 - 1) / lws1) * lws1, (size_t) k_split };
        size_t sk_lws[2] = { lws1, 1 };
        backend_ctx->enqueue_ndrange_kernel(sk, 2, sk_gws, sk_lws, dst);

        cl_kernel rk = backend_ctx->kernel_splitk_reduce;
        CL_CHECK(clSetKernelArg(rk, 0, sizeof(cl_mem), &partials));
        CL_CHECK(clSetKernelArg(rk, 1, sizeof(cl_mem), &C_d));
        CL_CHECK(clSetKernelArg(rk, 2, sizeof(int),    &plane));
        CL_CHECK(clSetKernelArg(rk, 3, sizeof(int),    &k_split));
        size_t rd_gws[1] = { (size_t)(((plane + 63) / 64) * 64) };
        size_t rd_lws[1] = { 64 };
        backend_ctx->enqueue_ndrange_kernel(rk, 1, rd_gws, rd_lws, dst);
        CL_CHECK(clReleaseMemObject(partials));
    } else if (vit_2d && (M % 4 == 0) && N >= 8) {
        // 2D launch: gy = gid(0) = N-tile, gx = gid(1) = M-tile. gws[1] is rounded up to the local size and
        // the kernel returns early for gx_2 >= M, so M only has to be a multiple of the 4-wide vload.
        cl_kernel fk = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f16_2d;
        CL_CHECK(clSetKernelArg(fk, 0, sizeof(cl_mem), &A_gemm));
        CL_CHECK(clSetKernelArg(fk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(fk, 2, sizeof(cl_mem), &C_d));
        CL_CHECK(clSetKernelArg(fk, 3, sizeof(int),    &M));
        CL_CHECK(clSetKernelArg(fk, 4, sizeof(int),    &padded_N));
        CL_CHECK(clSetKernelArg(fk, 5, sizeof(int),    &K));
        CL_CHECK(clSetKernelArg(fk, 6, sizeof(int),    &N));
        const size_t m_tiles = (size_t) (M / 4);
        const size_t lws_m   = 128;
        size_t gws[3] = { (size_t) ((N + 7) / 8), ((m_tiles + lws_m - 1) / lws_m) * lws_m, 1 };
        size_t lws[3] = { 1, lws_m, 1 };
        backend_ctx->enqueue_ndrange_kernel(fk, 3, gws, lws, dst);
    } else {
        cl_kernel fk = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f16;
        CL_CHECK(clSetKernelArg(fk, 0, sizeof(cl_mem), &A_gemm));
        CL_CHECK(clSetKernelArg(fk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(fk, 2, sizeof(cl_mem), &C_d));
        CL_CHECK(clSetKernelArg(fk, 3, sizeof(int),    &M));
        CL_CHECK(clSetKernelArg(fk, 4, sizeof(int),    &padded_N));
        CL_CHECK(clSetKernelArg(fk, 5, sizeof(int),    &K));
        CL_CHECK(clSetKernelArg(fk, 6, sizeof(int),    &N));
        size_t gws[1] = { ((total + lws1 - 1) / lws1) * lws1 };
        size_t lws[1] = { lws1 };
        backend_ctx->enqueue_ndrange_kernel(fk, 1, gws, lws, dst);
    }

    CL_CHECK(clReleaseMemObject(src0_sub));
    // A_T is the reused prealloc_pi0_at scratch. Do not release it.
    CL_CHECK(clReleaseMemObject(B_sub));
    if (B_pad) { CL_CHECK(clReleaseMemObject(B_pad)); }
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(B_in_img));
    CL_CHECK(clReleaseMemObject(B_image1d));
    CL_CHECK(clReleaseMemObject(C_d));
}
