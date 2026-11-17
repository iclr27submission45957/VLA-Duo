// PI0 benchmark: per-phase timing over repeated inferences.
//
// Usage:
//   llama-pi0-bench -m <model_dir> --image img.jpg -p "instruction" [-n 20] [--warmup 3]
//
// --warmup W inferences are discarded and the next -n M are measured. The run prints the per-phase table and one
// machine-readable RESULT line (means over the M measured inferences).

#include "pi0-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

// Tag profiled HTP ops by pipeline phase (GGML_HEXAGON_PROFILE). No-op if the Hexagon backend is not linked.
extern "C" void ggml_hexagon_set_profile_phase(const char *) __attribute__((weak));
static inline void pi0_profile_phase(const char * p) {
    if (ggml_hexagon_set_profile_phase) ggml_hexagon_set_profile_phase(p);
}
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---- Timing infrastructure ----

enum Phase {
    PHASE_VISION_ENCODE,
    PHASE_PREFIX_PASS,
    PHASE_EXPERT_STEP,
    PHASE_DIFFUSION_TOTAL,
    PHASE_INFERENCE_TOTAL,
    PHASE_COUNT
};

static const char * phase_names[PHASE_COUNT] = {
    "Vision Encode",
    "Prefix Pass",
    "Expert Step (x10)", // suffix prep is folded into the expert graph
    "Diffusion Total",
    "Inference Total",
};

struct PhaseStats {
    std::vector<double> samples_ms;

    double total() const {
        return std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    }
    double avg() const {
        return samples_ms.empty() ? 0.0 : total() / (double)samples_ms.size();
    }
    double min_val() const {
        return samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end());
    }
    double max_val() const {
        return samples_ms.empty() ? 0.0 : *std::max_element(samples_ms.begin(), samples_ms.end());
    }
    double stddev() const {
        if (samples_ms.size() < 2) return 0.0;
        double m = avg();
        double sum_sq = 0.0;
        for (double v : samples_ms) sum_sq += (v - m) * (v - m);
        return std::sqrt(sum_sq / (double)(samples_ms.size() - 1));
    }
};

static void print_table(PhaseStats stats[PHASE_COUNT], int n_iter, int n_warmup) {
    double total_inference = stats[PHASE_INFERENCE_TOTAL].total();

    printf("\n=== PI0 Benchmark (%d iterations, %d warmup) ===\n\n", n_iter, n_warmup);
    printf("%-20s | %10s | %8s | %8s | %8s | %8s | %7s\n",
           "Phase", "Total ms", "Avg ms", "Min ms", "Max ms", "Std ms", "% Total");
    printf("---------------------+------------+----------+----------+----------+----------+--------\n");

    for (int i = 0; i < PHASE_COUNT; i++) {
        if (i == PHASE_DIFFUSION_TOTAL) {
            printf("---------------------+------------+----------+----------+----------+----------+--------\n");
        }
        double pct = (total_inference > 0) ? 100.0 * stats[i].total() / total_inference : 0.0;
        printf("%-20s | %10.1f | %8.2f | %8.2f | %8.2f | %8.2f | %5.1f%%\n",
               phase_names[i],
               stats[i].total(),
               stats[i].avg(),
               stats[i].min_val(),
               stats[i].max_val(),
               stats[i].stddev(),
               pct);
    }

    printf("\n");
    if (total_inference > 0) {
        printf("Throughput: %.2f inferences/sec\n", n_iter / (total_inference / 1000.0));
    }
}

// ---- Custom arg parsing ----
// Strip bench-specific args (-n, --warmup) from argv before passing to common_params_parse,
// which would reject unknown arguments.

struct bench_params {
    int  n_iter      = 20;
    int  n_warmup    = 3;
};

static bench_params strip_bench_args(int & argc, char ** argv) {
    bench_params bp;
    int out = 1; // keep argv[0]
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--bench-n") == 0) && i + 1 < argc) {
            bp.n_iter = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            bp.n_warmup = std::atoi(argv[++i]);
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return bp;
}

// ---- Main ----

int main(int argc, char ** argv) {
    // Strip the shared pi0 flags (--backend, --kv-type) first, then the bench-specific ones.
    pi0_cli      cli = pi0_strip_cli_args(argc, argv);
    bench_params bp  = strip_bench_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 benchmark — per-phase timing\n\n");
        printf("Usage: %s -m <model_dir> --image img1.jpg,img2.jpg,img3.jpg \\\n", argv[0]);
        printf("         -p <instruction> [-n 20] [--warmup 3] [--backend cpu|auto|<name>]\n");
        printf("         [--kv-type f32|f16|q8_0|q4_0]\n\n");
        printf("  -n <int>       Number of measured inferences (default: 20)\n");
        printf("  --warmup <int> Discarded warm-up inferences before them (default: 3)\n");
        printf("  --backend      Backend: 'cpu' (default), 'auto'/'gpu', or a concrete name.\n");
        printf("  --kv-type      KV cache element type: f32 (default), f16, q8_0, q4_0, ...\n");
        printf("\nNote: use comma-separated paths for --image (not repeated flags)\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    std::string model_dir = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    // The fused-QKV PaliGemma file feeds only the pi0 loader. llama.cpp's gemma loader (tokenizer) needs the
    // unfused pi0-gemma-2b.gguf.
    std::string pali_pi0_path = model_dir + "/pi0-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_pi0_path.c_str(), "rb")) { fclose(f); } else { pali_pi0_path = pali_path; }

    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }

    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    // Mirror --backend cpu onto llama's own loader so it does not offload the tokenizer model to the GPU.
    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers   = 0;
        params.mmproj_use_gpu = false;
    }

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("Benchmark: %d iterations + %d warmup\n", bp.n_iter, bp.n_warmup);

    // ---- Initialize (one-time) ----

    ggml_backend_t backend = pi0_init_backend(cli.backend);
    if (!backend) {
        LOG_ERR("Failed to init backend '%s'\n", cli.backend.c_str());
        return 1;
    }

    pi0_model model;
    if (!load_pi0_model(model, pali_pi0_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    pi0_dump_tensor_types(model);

    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();
    if (!llm_model || !llm_ctx) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    // Load images
    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // Build prompt and tokenize
    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) {
        prompt += mtmd_default_marker();
    }
    if (model.pi05) params.prompt = pi0_format_prompt_pi05(params.prompt, nullptr); // pi0.5 prompt with the zero state
    prompt += params.prompt;

    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                      bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        LOG_ERR("Failed to tokenize\n");
        return 1;
    }

    // Text embeddings are dequantized per row via embd_lookup_f32.
    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> emb_row(PALI_N_EMBD);

    // Per-inference state. The KV cache lives on the backend and is reused across iterations.
    pi0_session session;
    session.kv_type = pi0_ggml_type_from_string(cli.kv_type);
    if (session.kv_type == GGML_TYPE_COUNT) {
        LOG_ERR("Unknown --kv-type '%s' (try f32, f16, q8_0, q4_0)\n", cli.kv_type.c_str());
        return 1;
    }
    LOG_INF("KV cache type: %s\n", ggml_type_name(session.kv_type));

    // Robot state (zeros)
    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);
    const float dt = -1.0f / PI0_NUM_STEPS;

    LOG_INF("Starting benchmark...\n");

    // ---- Benchmark loop ----

    PhaseStats stats[PHASE_COUNT];
    const int total_runs = bp.n_warmup + bp.n_iter;
    double action_rms = 0.0;   // rms of the action chunk of the first measured inference

    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < bp.n_warmup);
        if (!is_warmup && run == bp.n_warmup) {
            LOG_INF("Warmup done. Measuring %d iterations...\n", bp.n_iter);
        }

        int64_t t_iter_start = ggml_time_us();

        // ---- Phase: Vision Encode ----
        pi0_profile_phase("vision");
        int64_t t0 = ggml_time_us();

        std::vector<float> prefix_embeddings;
        int actual_prefix_len = 0;

        // PI0_VIT_BATCH=1: encode all image chunks in one vision graph and slice the result per chunk below.
        static const int pi0_vit_batch = []{ const char * e = getenv("PI0_VIT_BATCH"); return e ? atoi(e) : 0; }();
        std::vector<const mtmd_input_chunk *> img_chunks;
        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img_chunks.push_back(c);
        }
        const float * batched_img_embd = nullptr;
        size_t batched_off = 0;
        if (pi0_vit_batch && img_chunks.size() > 1) {
            if (mtmd_encode_image_chunks(ctx_vision.get(), img_chunks.data(), img_chunks.size()) != 0) {
                LOG_ERR("Batched vision encode failed at run %d\n", run);
                return 1;
            }
            batched_img_embd = mtmd_get_output_embd(ctx_vision.get());
        }

        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                // Gemma scales text token embeddings by sqrt(hidden). SigLIP image features stay unscaled.
                const float embd_scale = sqrtf((float) PALI_N_EMBD);
                for (size_t t = 0; t < n_tokens; t++) {
                    embd_lookup_f32(model.pali_embed, tokens[t], emb_row.data());
                    for (float & e : emb_row) e *= embd_scale;
                    prefix_embeddings.insert(prefix_embeddings.end(), emb_row.begin(), emb_row.end());
                    actual_prefix_len++;
                }
            } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
                const float * img_embd = nullptr;
                if (batched_img_embd) {
                    img_embd = batched_img_embd + batched_off;
                    batched_off += n_img_tokens * PALI_N_EMBD;
                } else {
                    if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
                        LOG_ERR("Vision encode failed at run %d\n", run);
                        return 1;
                    }
                    img_embd = mtmd_get_output_embd(ctx_vision.get());
                }
                prefix_embeddings.insert(prefix_embeddings.end(),
                    img_embd, img_embd + n_img_tokens * PALI_N_EMBD);
                actual_prefix_len += (int)n_img_tokens;
            }
        }

        int64_t t1 = ggml_time_us();

        // ---- Phase: Prefix Pass ----
        pi0_profile_phase("prefix");
        if (!run_paligemma_prefix(model, session, backend, prefix_embeddings.data(), actual_prefix_len)) {
            LOG_ERR("Prefix pass failed at run %d\n", run);
            return 1;
        }
        int64_t t2 = ggml_time_us();

        // ---- Phase: Diffusion Loop ----
        pi0_profile_phase("diffusion");
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        // PI0_NOISE_BIN=<file>: read the initial noise as raw float32 [HORIZON][DIM], else draw it from mt19937(42).
        static const std::string noise_bin = []{ const char * e = getenv("PI0_NOISE_BIN"); return e ? std::string(e) : std::string(); }();
        if (!noise_bin.empty()) {
            FILE * nf = fopen(noise_bin.c_str(), "rb");
            size_t got = nf ? fread(x_t.data(), sizeof(float), x_t.size(), nf) : 0;
            if (nf) fclose(nf);
            if (got != x_t.size()) { LOG_ERR("PI0_NOISE_BIN: cannot read %zu floats from %s\n", x_t.size(), noise_bin.c_str()); return 1; }
        } else {
            for (auto & v : x_t) v = normal(rng);
        }

        double expert_step_ms = 0.0;
        std::vector<float> v_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);

        for (int step = 0; step < PI0_NUM_STEPS; step++) {
            float t = 1.0f + step * dt;

            int64_t ts0 = ggml_time_us();
            if (!run_expert_step(model, session, backend,
                                 robot_state.data(), x_t.data(), t, v_t.data())) {
                LOG_ERR("Expert step failed at run %d step %d\n", run, step);
                return 1;
            }
            int64_t ts1 = ggml_time_us();
            expert_step_ms += (ts1 - ts0) / 1000.0;

            for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
                x_t[i] += dt * v_t[i];
            }
        }

        int64_t t3 = ggml_time_us();

        // ---- Dump the final action of the first measured inference ----
        if (!is_warmup && run == bp.n_warmup) {
            fprintf(stderr, "ACTION_DUMP: x_t[0:7] = ");
            for (int i = 0; i < PI0_ACTION_DIM && i < 7; i++) {
                fprintf(stderr, "%.6f ", x_t[i]);
            }
            fprintf(stderr, "\n");
            double sum = 0, sumsq = 0;
            for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
                sum += x_t[i]; sumsq += (double)x_t[i] * x_t[i];
            }
            action_rms = std::sqrt(sumsq / (PI0_ACTION_DIM * PI0_ACTION_HORIZON));
            fprintf(stderr, "ACTION_DUMP: full mean=%.6f rms=%.6f n=%d\n",
                sum / (PI0_ACTION_DIM * PI0_ACTION_HORIZON), action_rms,
                PI0_ACTION_DIM * PI0_ACTION_HORIZON);
            // PI0_DUMP_ACTIONS=<path>: write the raw f32 [PI0_ACTION_DIM * PI0_ACTION_HORIZON] actions
            if (const char * ap = getenv("PI0_DUMP_ACTIONS")) {
                if (FILE * f = fopen(ap, "wb")) {
                    fwrite(x_t.data(), sizeof(float), PI0_ACTION_DIM * PI0_ACTION_HORIZON, f);
                    fclose(f);
                    fprintf(stderr, "ACTION_DUMP: wrote %s\n", ap);
                }
            }
        }

        // ---- Record stats ----
        if (!is_warmup) {
            double vision_ms  = (t1 - t0) / 1000.0;
            double prefix_ms  = (t2 - t1) / 1000.0;
            double diff_ms    = (t3 - t2) / 1000.0;
            double total_ms   = (t3 - t_iter_start) / 1000.0;

            stats[PHASE_VISION_ENCODE].samples_ms.push_back(vision_ms);
            stats[PHASE_PREFIX_PASS].samples_ms.push_back(prefix_ms);
            stats[PHASE_EXPERT_STEP].samples_ms.push_back(expert_step_ms);
            stats[PHASE_DIFFUSION_TOTAL].samples_ms.push_back(diff_ms);
            stats[PHASE_INFERENCE_TOTAL].samples_ms.push_back(total_ms);

            LOG_INF("  [%d/%d] iter %.1f ms (vision %.1f, prefix %.1f, diffusion %.1f)\n",
                run - bp.n_warmup + 1, bp.n_iter,
                total_ms, vision_ms, prefix_ms, diff_ms);
        }
    }

    // ---- Print results ----
    print_table(stats, bp.n_iter, bp.n_warmup);
    printf("RESULT tool=bench backend=%s model=%s warmup=%d measured=%d vision_ms=%.1f prefix_ms=%.1f fm_ms=%.1f "
           "total_ms=%.1f total_min_ms=%.1f total_max_ms=%.1f rms=%.6f\n",
           cli.backend.c_str(), model.pi05 ? "pi05" : "pi0", bp.n_warmup, bp.n_iter,
           stats[PHASE_VISION_ENCODE].avg(), stats[PHASE_PREFIX_PASS].avg(), stats[PHASE_DIFFUSION_TOTAL].avg(),
           stats[PHASE_INFERENCE_TOTAL].avg(), stats[PHASE_INFERENCE_TOTAL].min_val(),
           stats[PHASE_INFERENCE_TOTAL].max_val(), action_rms);
    fflush(stdout);

    // Cleanup
    pi0_session_free(session);
    free_pi0_model(model);
    ggml_backend_free(backend);

    return 0;
}
