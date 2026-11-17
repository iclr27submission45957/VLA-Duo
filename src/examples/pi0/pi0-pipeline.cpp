// PI0 GPU/NPU periodic pipeline.
//
// Consecutive inferences run as a pipeline over the OpenCL GPU and the Hexagon NPU. Frame N+1's first stage overlaps
// frame N's second stage, so the steady-state period is bounded by the busier device, not the sum:
//
//   Image - ViT head(GPU) - ViT tail + Prefix + FM(NPU) - Acting
//                    Image - ViT head(GPU) - ViT tail + Prefix + FM(NPU) - Acting
//
// Schedules:
//   two-stage   one cut, G->N (default: GPU head, NPU rest, --vit-split) or N->G (--direction NG with --vit-split,
//               --lm-split, --fm-split or --fm-layer-split), eager admission, double-buffered hand-off queue
//   plan        k-cut pipeline (--plan + --plan-order): k+1 segments on the two devices, static per-device order
//
// Burst protocol: one untimed warm-up pass builds and allocates every graph. Then a stream of N = W + M + D frames
// runs: W warm-up frames (--warmup), M measured frames (--measure) and D drain frames. D = 3 for two-stage (2 in the
// hand-off queue + 1 held by the head while the queue is full). For plans D is the deepest negative frame offset of
// --plan-order. Period = (complete[W+M-1] - complete[W-1]) / M. Latency = mean of (complete - admitted) over frames
// W..W+M-1. The drain frames keep the head as busy during the last measured frames as in steady state.
// Two-stage latency grows with the hand-off backlog (eager admission), so it depends on the window.
// Plan latencies are bounded by the static order.
// The run prints a summary and one machine-readable RESULT line.
//
// Usage:
//   llama-pi0-pipeline -m <model_dir> --image img.jpg -p "instruction" --backend HTP0 [--vit-split L] [--warmup 3] [--measure 20]
//
// Use a build without per-phase profiling. The profiling setters touch shared globals and race across the two workers.

#include "pi0-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

// ---- Small helpers ----

static inline double us_to_ms(int64_t us) { return us / 1000.0; }

static void action_stats(const std::vector<float> & x, double & mean, double & rms) {
    double sum = 0, sumsq = 0;
    for (float v : x) { sum += v; sumsq += (double) v * v; }
    mean = sum / (double) x.size();
    rms  = std::sqrt(sumsq / (double) x.size());
}

// One frame-plan segment, built once on the main thread so that the GPU worker never reads the NPU-resident
// embedding table. Text segments are constant across frames. Image segments are re-encoded each frame.
struct chunk_seg {
    bool is_image = false;
    const mtmd_input_chunk * chunk = nullptr; // valid iff is_image
    std::vector<float> text_embd;             // valid iff !is_image (already scaled)
    int n_tokens = 0;
};

// One frame handed from the first-device stage to the second (host buffers).
struct frame_item {
    int frame_id = 0;                // -1 = end sentinel
    std::vector<float> embd;         // prefix embeddings [PALI_N_EMBD * prefix_len] (unused with --vit-split)
    int prefix_len = 0;
    // --vit-split L: per-image SigLIP hidden state after layer L-1 from the head graph. The second device consumes it
    // in the next period, so the two devices never wait on each other within a period. Empty: `embd` holds the prefix.
    std::vector<std::vector<float>> hidden;
    // deep cuts: noisy action state + prefix KV rows of the first-device layers (pi0_kv_prefix_get layout)
    std::vector<float>   x_t;
    std::vector<uint8_t> kv;
    std::vector<float>   hidden_lm;   // --lm-split: the residual after prefix layer lm_split-1 [PALI_N_EMBD * prefix_len]
                                      // --fm-layer-split: the suffix residual after expert layer j-1 [EXPERT_N_EMBD * suffix_len]
};

// Bounded blocking FIFO for the hand-off.
class frame_queue {
    std::mutex m;
    std::condition_variable cv_not_full, cv_not_empty;
    std::queue<frame_item> q;
    size_t cap;
public:
    int64_t consumer_wait_us = 0; // total time the consumer spent blocked (stall)
    explicit frame_queue(size_t c) : cap(c) {}

    void push(frame_item it) {
        std::unique_lock<std::mutex> lk(m);
        cv_not_full.wait(lk, [&] { return q.size() < cap; });
        q.push(std::move(it));
        cv_not_empty.notify_one();
    }
    frame_item pop() {
        std::unique_lock<std::mutex> lk(m);
        int64_t w0 = ggml_time_us();
        cv_not_empty.wait(lk, [&] { return !q.empty(); });
        consumer_wait_us += ggml_time_us() - w0;
        frame_item it = std::move(q.front());
        q.pop();
        cv_not_full.notify_one();
        return it;
    }
};
// ---- CLI ----

struct pipe_params {
    int  warmup      = 3;     // streamed warm-up frames (not measured)
    int  measure     = 20;    // measured frames
    int  vit_split   = -1;    // two-stage: # of ViT layers kept on the FIRST device (-1 = all, no split)
    // Direction of the single cut. GN: first device = GPU (SigLIP head), second = NPU (tail + projector + prefix + FM).
    // NG: first device = NPU, second = GPU. The LM and the expert then run on OpenCL, so the model dir must be a
    // non-fused GPU set and the OpenCL LM flags must be set.
    std::string direction = "GN";
    // N->G cut on the expert side (atom p = 47 + fm_split). The NPU runs ViT + projector + prefix + the first fm_split
    // denoising steps. OpenCL runs the remaining steps against a copy of the prefix KV. -1 = off.
    int  fm_split = -1;
    // N->G cut inside the LM (p = 29 + lm_split). The first device runs ViT + prefix layers [0, lm_split) and hands the
    // residual + those layers' KV to OpenCL, which runs layers [lm_split, 18) and all denoising steps.
    // Range 1..17 (18 is --fm-split 0). -1 = off.
    int  lm_split = -1;
    // N->G cut inside the first denoising step (fine atom c = 272 + 6 * j). The first device runs ViT + prefix + expert
    // layers [0, j) of step 0 and hands the suffix residual to OpenCL, which runs layers [j, 18) of step 0 and steps
    // 1..9. Range 1..17. -1 = off.
    int  fm_layer_split = -1;
    std::string gpu_model_dir; // model dir for the OpenCL LM / expert (a non-fused GPU set), default = -m
    std::string plan;         // --plan: k-cut pipeline (see kc_seg), overrides --vit-split, excludes the other cut options
    std::string plan_order;   // --plan-order: static per-device order for --plan, "c=1;G:1/0,3/-1;N:0/0,4/-2,2/0"
    double period_ms = 0;     // two-stage: frame f is admitted at t0 + f * period (a busy head starts it late). 0 = eager
    double abort_ms  = 0;     // two-stage: stop admitting once 5 consecutive frames exceed this latency. 0 = off
};

// ---- k-cut plans (--plan) ----
// A plan = k+1 contiguous segments of the 57 layer-level atoms of one inference, each on the GPU or the NPU, run as a
// (k+1)-stage pipeline (the format vladuo.py kcut prints):
//   V = 29 atoms (0 = patch embedding, 1..27 = SigLIP layers 0..26, 28 = post-LN + projector),
//   L = 18 PaliGemma layers (a cut inside L hands on the residual and the KV rows of every layer travel with the frame),
//   F = 10 flow-matching steps.
// Syntax: "G1V,N6V,G22V,N18L10F". Brackets, '+', ',' and spaces are separators, so "G[1V] N[6V] G[22V] N[18L+10F]"
// parses as is.
// Hand-offs: a cut inside SigLIP passes the raw hidden state [1152, n_patches] x images (fp32). The cut at atom 29
// passes the prefix embeddings. A cut inside F passes the noisy action state + the prefix KV of all 18 layers
// (pi0_kv_prefix_get layout).
static constexpr int KC_V = 29;
static constexpr int KC_M = KC_V + PI0_N_LAYER + PI0_NUM_STEPS;

struct kc_seg {
    char dev = 'G';          // 'G' = OpenCL, 'N' = the --backend device (HTP0)
    int  a = 0, b = 0;       // atoms [a, b)
    int  v0 = 0, v1 = 0;     // SigLIP atoms [v0, v1) (v0 == v1: none)
    bool lm = false;         // all 18 PaliGemma layers
    int  l0 = 0, l1 = 0;     // PaliGemma layers [l0, l1) (l0 == l1: none), a partial range hands the residual + KV rows on
    int  f0 = 0, f1 = 0;     // flow-matching steps [f0, f1)
    std::string name;        // "G[1V]", "N[18L+10F]"
};

static bool parse_kc_plan(const std::string & spec, std::vector<kc_seg> & out, std::string & err) {
    out.clear();
    int pos = 0, num = -1;
    std::string parts;
    auto close_seg = [&]() -> bool {
        if (out.empty()) return true;
        kc_seg & g = out.back();
        if (num >= 0) { err = "a count without its V/L/F letter in segment " + std::to_string(out.size()); return false; }
        g.b = pos;
        if (g.b <= g.a) { err = "segment " + std::to_string(out.size()) + " is empty"; return false; }
        g.name = std::string(1, g.dev) + "[" + parts + "]";
        parts.clear();
        return true;
    };
    for (char c : spec) {
        if (c == 'G' || c == 'N') {
            if (!close_seg()) return false;
            kc_seg g; g.dev = c; g.a = pos;
            out.push_back(g);
        } else if (c >= '0' && c <= '9') {
            if (out.empty()) { err = "the plan must start with G or N"; return false; }
            num = (num < 0 ? 0 : num * 10) + (c - '0');
        } else if (c == 'V' || c == 'L' || c == 'F') {
            if (out.empty() || num <= 0) { err = std::string("'") + c + "' needs a positive count before it"; return false; }
            const int lo = c == 'V' ? 0 : c == 'L' ? KC_V : KC_V + PI0_N_LAYER;
            const int hi = c == 'V' ? KC_V : c == 'L' ? KC_V + PI0_N_LAYER : KC_M;
            if (pos < lo || pos + num > hi) {
                err = std::string("atoms [") + std::to_string(pos) + "," + std::to_string(pos + num) + ") are not all " + c +
                      " atoms (V = [0,29), L = [29,47), F = [47,57); parts must follow in this order)";
                return false;
            }
            parts += (parts.empty() ? "" : "+") + std::to_string(num) + c;
            pos += num; num = -1;
        } else if (c == '[' || c == ']' || c == '+' || c == ',' || c == ' ') {
            if (num >= 0) { err = "a count without its V/L/F letter"; return false; }
        } else {
            err = std::string("unexpected character '") + c + "'"; return false;
        }
    }
    if (!close_seg()) return false;
    if (out.size() < 2) { err = "a plan needs at least 2 segments (k >= 1)"; return false; }
    if (pos != KC_M) { err = "the segments cover " + std::to_string(pos) + " atoms, not " + std::to_string(KC_M); return false; }
    for (size_t s = 0; s < out.size(); s++) {
        kc_seg & g = out[s];
        // neighbouring segments may sit on the same device (the hand-off then stays on that device)
        g.v0 = std::min(g.a, KC_V); g.v1 = std::min(g.b, KC_V);
        const int l0 = std::max(std::min(g.a, KC_V + PI0_N_LAYER), KC_V), l1 = std::max(std::min(g.b, KC_V + PI0_N_LAYER), KC_V);
        g.l0 = l0 - KC_V; g.l1 = l1 - KC_V;
        g.lm = l1 - l0 == PI0_N_LAYER;
        g.f0 = std::max(g.a, KC_V + PI0_N_LAYER) - (KC_V + PI0_N_LAYER);
        g.f1 = std::max(g.b, KC_V + PI0_N_LAYER) - (KC_V + PI0_N_LAYER);
    }
    return true;
}

// --plan-order "c=<frames per cycle>;G:<s>/<off>,...;N:<s>/<off>,...": every segment must sit in its own device's list
// once per residue of off mod c, so that every (segment, frame) runs exactly once.
// "sync" anywhere in the spec: a device starts cycle p only after both devices finished cycle p-1. With frame offsets
// 0 / -1 the observation-to-action latency then stays <= 2 cycles.
static bool parse_kc_order(const std::string & spec, const std::vector<kc_seg> & kc, int & c, bool & sync,
                           std::vector<std::pair<int, int>> & og, std::vector<std::pair<int, int>> & on, std::string & err) {
    c = 0; sync = false; og.clear(); on.clear();
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(';', i);
        if (j == std::string::npos) j = spec.size();
        const std::string tok = spec.substr(i, j - i);
        i = j + 1;
        if (tok.empty()) continue;
        if (tok.rfind("c=", 0) == 0) { c = std::atoi(tok.c_str() + 2); continue; }
        if (tok == "sync") { sync = true; continue; }
        if (tok.size() < 2 || (tok[0] != 'G' && tok[0] != 'N') || tok[1] != ':') { err = "bad token '" + tok + "'"; return false; }
        auto & lst = tok[0] == 'G' ? og : on;
        size_t k = 2;
        while (k < tok.size()) {
            size_t e = tok.find(',', k);
            if (e == std::string::npos) e = tok.size();
            const std::string ent = tok.substr(k, e - k);
            const size_t sl = ent.find('/');
            if (sl == std::string::npos) { err = "entry '" + ent + "' is not s/off"; return false; }
            lst.emplace_back(std::atoi(ent.substr(0, sl).c_str()), std::atoi(ent.substr(sl + 1).c_str()));
            k = e + 1;
        }
    }
    if (c < 1) { err = "missing c=<frames per cycle>"; return false; }
    const int S = (int) kc.size();
    std::vector<std::vector<int>> seen(S, std::vector<int>(c, 0));
    for (char d : {'G', 'N'}) {
        for (const auto & e : d == 'G' ? og : on) {
            if (e.first < 0 || e.first >= S) { err = "segment " + std::to_string(e.first) + " does not exist"; return false; }
            if (kc[e.first].dev != d) { err = "segment " + std::to_string(e.first) + " " + kc[e.first].name + " is listed for the wrong device"; return false; }
            seen[e.first][((e.second % c) + c) % c]++;
        }
    }
    for (int s = 0; s < S; s++) for (int r = 0; r < c; r++) {
        if (seen[s][r] != 1) { err = "segment " + std::to_string(s) + " appears " + std::to_string(seen[s][r]) + " times for frames = " + std::to_string(r) + " mod c"; return false; }
    }
    return true;
}

static pipe_params strip_pipe_args(int & argc, char ** argv) {
    pipe_params pp;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            pp.warmup = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
            pp.measure = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--period") == 0 && i + 1 < argc) {
            pp.period_ms = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--abort-latency") == 0 && i + 1 < argc) {
            pp.abort_ms = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--direction") == 0 && i + 1 < argc) {
            pp.direction = argv[++i];
            if (pp.direction != "GN" && pp.direction != "NG") { fprintf(stderr, "--direction must be GN or NG\n"); exit(1); }
        } else if (strcmp(argv[i], "--vit-split") == 0 && i + 1 < argc) {
            pp.vit_split = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fm-split") == 0 && i + 1 < argc) {
            pp.fm_split = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lm-split") == 0 && i + 1 < argc) {
            pp.lm_split = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fm-layer-split") == 0 && i + 1 < argc) {
            pp.fm_layer_split = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-model-dir") == 0 && i + 1 < argc) {
            pp.gpu_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--plan") == 0 && i + 1 < argc) {
            pp.plan = argv[++i];
        } else if (strcmp(argv[i], "--plan-order") == 0 && i + 1 < argc) {
            pp.plan_order = argv[++i];
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return pp;
}

int main(int argc, char ** argv) {
    pi0_cli     cli = pi0_strip_cli_args(argc, argv);
    pipe_params pp  = strip_pipe_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 heterogeneous GPU/NPU periodic pipeline (VLA-Duo)\n\n");
        printf("Usage: %s -m <model_dir> --image img1.jpg,... -p <instruction> \\\n", argv[0]);
        printf("         --backend HTP0 [--vit-split L] [--warmup 3] [--measure 20]\n\n");
        printf("  --backend <name>     pi0 backend of the NPU stage (HTP0)\n");
        printf("  --kv-type <type>     KV cache element type (f16)\n");
        printf("  --warmup <int>       streamed warm-up frames before the measured ones (default: 3)\n");
        printf("  --measure <int>      measured frames (default: 20); the stream adds the drain frame(s)\n");
        printf("  --period <ms>        sustained run (two-stage): admit frame f at t0 + f * period instead of eagerly\n");
        printf("  --abort-latency <ms> sustained run: stop once 5 consecutive frames exceed this latency\n");
        printf("  --direction GN|NG    single-cut direction: GN = GPU head -> NPU rest (default); NG = NPU head -> GPU rest\n");
        printf("  --vit-split <int>    # of SigLIP layers on the first device; the rest + projector + prefix + FM run on the\n");
        printf("                       second device (default: -1 = whole ViT on the GPU, G->N only); with --direction NG\n");
        printf("                       the LM and expert run on OpenCL from the -m dir, so -m must be a GPU set (gguf/<m>-gpu)\n");
        printf("  --lm-split <K>       N->G cut inside the LM (p = 29 + K): the NPU runs ViT + prefix layers [0,K), OpenCL\n");
        printf("                       runs layers [K,%d) + all denoising steps (residual + K layers' KV copied)\n", PI0_N_LAYER);
        printf("  --fm-split <J>       N->G cut on the expert side (p = 47 + J): the NPU runs ViT + prefix + the first J\n");
        printf("                       denoising steps, OpenCL the remaining %d - J against a copy of the prefix KV\n", PI0_NUM_STEPS);
        printf("  --fm-layer-split <j> N->G cut inside the first denoising step (c = 272 + 6j): the NPU runs ViT + prefix +\n");
        printf("                       expert layers [0,j) of step 0, OpenCL runs layers [j,%d) of step 0 and steps 1..%d\n", PI0_N_LAYER, PI0_NUM_STEPS - 1);
        printf("  --gpu-model-dir <d>  GGUF dir for the OpenCL LM / expert (a non-fused GPU set); default: the -m dir\n");
        printf("  --plan <spec>        k-cut pipeline, e.g. \"G1V,N6V,G22V,N18L10F\" (= \"G[1V] N[6V] G[22V] N[18L+10F]\"): k+1 segments of the\n");
        printf("                       57 atoms (29 V = patch embed + 27 SigLIP layers + projector, 18 L, 10 F), alternating devices;\n");
        printf("                       ignores --vit-split; needs --plan-order\n");
        printf("  --plan-order <spec>  static order for --plan: \"c=1;G:1/0,3/-1;N:0/0,4/-2,2/0\" = per cycle of c frames each device\n");
        printf("                       runs segment s of frame c*cycle+off in this order; \"sync\" adds a cycle barrier\n");
        printf("\nEnv: PI0_PIPE_TRACE=<csv> per-frame trace, PI0_ACTIONS_OUT=<file> actions of the last frame (float32),\n");
        printf("     PI0_NOISE_BIN=<file> initial noise (float32 [50][32]).\n");
    };


    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    std::string model_dir   = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    std::string pali_pi0_path = model_dir + "/pi0-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_pi0_path.c_str(), "rb")) { fclose(f); } else { pali_pi0_path = pali_path; }

    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }
    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers   = 0;
        params.mmproj_use_gpu = false;
    }

    // --plan replaces every other cut option. An extra --vit-split is ignored.
    std::vector<kc_seg> kc;
    const bool kcut = !pp.plan.empty();
    bool kc_gpu_lm = false, kc_npu_v = false;   // OpenCL runs L or F atoms / the NPU runs SigLIP atoms
    if (kcut) {
        std::string err;
        if (!parse_kc_plan(pp.plan, kc, err)) { LOG_ERR("--plan \"%s\": %s\n", pp.plan.c_str(), err.c_str()); return 1; }
        if (pp.vit_split > 0) LOG_INF("--plan: ignoring --vit-split %d\n", pp.vit_split);
        pp.vit_split = -1;
        if (pp.fm_split >= 0 || pp.lm_split >= 0 || pp.fm_layer_split > 0 || pp.direction != "GN") {
            LOG_ERR("--plan excludes --fm-split/--lm-split/--fm-layer-split/--direction\n");
            return 1;
        }
        if (pp.plan_order.empty()) { LOG_ERR("--plan needs --plan-order\n"); return 1; }
        for (const kc_seg & g : kc) {
            if (g.dev == 'G' && (g.l1 > g.l0 || g.f1 > g.f0)) kc_gpu_lm = true;
            if (g.dev == 'N' && g.v1 > g.v0)           kc_npu_v  = true;
        }
    }
    int kord_c = 0;                                   // --plan-order: frames per cycle
    bool kord_sync = false;                           // --plan-order "sync": cycle barrier between the two devices
    std::vector<std::pair<int, int>> kord_g, kord_n;  // (segment, frame offset) per device, in run order
    if (!pp.plan_order.empty()) {
        std::string err;
        if (!kcut) { LOG_ERR("--plan-order needs --plan\n"); return 1; }
        if (!parse_kc_order(pp.plan_order, kc, kord_c, kord_sync, kord_g, kord_n, err)) { LOG_ERR("--plan-order \"%s\": %s\n", pp.plan_order.c_str(), err.c_str()); return 1; }
    }

    // Stream window: W warm-up + M measured + D drain frames.
    if (pp.warmup < 1 || pp.measure < 1) { LOG_ERR("--warmup and --measure must be >= 1\n"); return 1; }
    if ((pp.period_ms > 0 || pp.abort_ms > 0) && kcut) { LOG_ERR("--period / --abort-latency are two-stage only (not with --plan)\n"); return 1; }
    if (pp.period_ms < 0 || pp.abort_ms < 0) { LOG_ERR("--period and --abort-latency must be >= 0\n"); return 1; }
    int drain = 3;   // two-stage: hand-off queue depth 2 + the frame the head holds while the queue is full
    if (kcut) {
        drain = 1;
        for (const auto & e : kord_g) drain = std::max(drain, -e.second);
        for (const auto & e : kord_n) drain = std::max(drain, -e.second);
        if (kord_c > 1 && (pp.warmup % kord_c != 0 || pp.measure % kord_c != 0)) {
            LOG_ERR("--plan-order c=%d: --warmup (%d) and --measure (%d) must be multiples of c\n", kord_c, pp.warmup, pp.measure);
            return 1;
        }
    }
    const int N = pp.warmup + pp.measure + drain;

    const bool ng   = (pp.direction == "NG");
    const bool deep = pp.fm_split >= 0 || pp.lm_split >= 0 || pp.fm_layer_split > 0;   // N->G cut on the LM / expert side
    // lm_k = PaliGemma layers on the first device (18 = all), fm_j = denoising steps on the first device (0 = none),
    // fl_j = expert layers of the next step on the first device (0 = none, only with fm_j == 0 and lm_k == 18).
    const int lm_k = pp.lm_split >= 0 ? pp.lm_split : PI0_N_LAYER;
    const int fm_j = pp.fm_split >= 0 ? pp.fm_split : 0;
    const int fl_j = pp.fm_layer_split > 0 ? pp.fm_layer_split : 0;
    const int deep_p = 1 + 27 + 1 + lm_k + fm_j;   // layer-level atom index of the cut
    const int deep_pfine = (lm_k < PI0_N_LAYER) ? 164 + 6 * lm_k : 272 + 108 * fm_j + 6 * fl_j;
    if (deep) {
        if ((pp.fm_split >= 0) + (pp.lm_split >= 0) + (pp.fm_layer_split > 0) > 1) { LOG_ERR("--fm-split, --lm-split and --fm-layer-split are mutually exclusive\n"); return 1; }
        if (fl_j > PI0_N_LAYER - 1) { LOG_ERR("--fm-layer-split must be in [1, %d]\n", PI0_N_LAYER - 1); return 1; }
        if (fm_j > PI0_NUM_STEPS) { LOG_ERR("--fm-split must be in [0, %d]\n", PI0_NUM_STEPS); return 1; }
        if (pp.lm_split >= 0 && (lm_k < 1 || lm_k > PI0_N_LAYER - 1)) { LOG_ERR("--lm-split must be in [1, %d] (use --fm-split 0 for all layers on the NPU)\n", PI0_N_LAYER - 1); return 1; }
        if (pp.vit_split > 0) { LOG_ERR("--vit-split does not combine with --fm-split/--lm-split/--fm-layer-split\n"); return 1; }
        if (!ng) { LOG_ERR("--fm-split/--lm-split/--fm-layer-split are N->G cuts (NPU: ViT + prefix [+ steps], OpenCL: the rest); pass --direction NG\n"); return 1; }
    }
    if (ng && !deep && pp.vit_split <= 0) { LOG_ERR("--direction NG needs --vit-split >= 1 (SigLIP layers on the NPU head), --lm-split or --fm-split\n"); return 1; }
    // layer-level cut p and fine atom index c of a two-stage run
    const int cut_p = deep ? deep_p : (pp.vit_split > 0 ? pp.vit_split + 1 : 29);
    const int cut_c = deep ? deep_pfine : (pp.vit_split > 0 ? 6 * cut_p - 5 : 164);

    // Backend of the SigLIP tail + projector + prefix + FM. GN: the NPU (cli.backend). NG: OpenCL.
    // Deep cuts keep the prefix on cli.backend and run the rest on OpenCL (backend_gpu).
    const std::string lm_backend_name = (ng && !deep) ? std::string("gpu") : cli.backend;
    ggml_backend_t backend = pi0_init_backend(lm_backend_name);
    if (!backend) {
        LOG_ERR("Failed to init backend '%s'\n", lm_backend_name.c_str());
        return 1;
    }
    LOG_INF("pi0 backend (Prefix+FM): %s | Vision(mtmd) use_gpu=%d | direction %s\n",
            lm_backend_name.c_str(), (int) params.mmproj_use_gpu, pp.direction.c_str());

    pi0_model model;
    if (!load_pi0_model(model, pali_pi0_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    pi0_dump_tensor_types(model);

    // Deep cuts / plans with GPU LM atoms: a second pi0 model on OpenCL. The HTP model dir has fused / K-split layouts
    // the OpenCL kernels do not take, so this LM / expert comes from --gpu-model-dir.
    ggml_backend_t backend_gpu = nullptr;
    pi0_model      model_gpu;
    if (deep || kc_gpu_lm) {
        backend_gpu = pi0_init_backend("gpu");
        if (!backend_gpu) { LOG_ERR("--fm-split: failed to init the OpenCL backend\n"); return 1; }
        const std::string gdir  = pp.gpu_model_dir.empty() ? model_dir : resolve_model_dir(pp.gpu_model_dir);
        const std::string gpali = gdir + "/pi0-gemma-2b.gguf";
        const std::string gexp  = gdir + "/pi0-action-expert.gguf";
        if (!load_pi0_model(model_gpu, gpali.c_str(), gexp.c_str(), backend_gpu)) {
            LOG_ERR("--fm-split: failed to load the OpenCL expert from %s\n", gdir.c_str());
            return 1;
        }
        if (deep) {
            LOG_INF("deep cut p=%d (p_fine=%d): %s runs ViT + prefix layers [0,%d) + steps [0,%d) + expert layers [0,%d) of step %d; OpenCL runs the rest from %s\n",
                    deep_p, deep_pfine, cli.backend.c_str(), lm_k, fm_j, fl_j, fm_j, gdir.c_str());
        } else {
            LOG_INF("--plan: OpenCL LM / expert from %s\n", gdir.c_str());
        }
    }

    auto llama_init = common_init_from_params(params);
    llama_model * llm_model = llama_init->model();
    if (!llm_model) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu; // GPU stage lives here (OpenCL)
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    // --vit-split: a second vision context on the NPU. Each device needs its own context because the layer range is
    // per-context state and the two workers run concurrently. The clip_ctx constructor reads MTMD_BACKEND_DEVICE, so it
    // is set around the init.
    mtmd::context_ptr ctx_vision_tail;
    if (pp.vit_split > 0) {
        const char * prev = getenv("MTMD_BACKEND_DEVICE");
        const std::string saved = prev ? prev : "";
        setenv("MTMD_BACKEND_DEVICE", cli.backend.c_str(), 1);
        ctx_vision_tail.reset(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
        if (saved.empty()) unsetenv("MTMD_BACKEND_DEVICE");
        else               setenv("MTMD_BACKEND_DEVICE", saved.c_str(), 1);
        if (!ctx_vision_tail.get()) {
            LOG_ERR("Failed to load the tail vision model on %s\n", cli.backend.c_str());
            return 1;
        }
    }
    // Deep cuts: the whole ViT runs on the first device (cli.backend) in one more context with the full layer range.
    mtmd::context_ptr ctx_vision_npu;
    if (deep || kc_npu_v) {   // --plan: the NPU's SigLIP segments run here
        const char * prev = getenv("MTMD_BACKEND_DEVICE");
        const std::string saved = prev ? prev : "";
        setenv("MTMD_BACKEND_DEVICE", cli.backend.c_str(), 1);
        ctx_vision_npu.reset(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
        if (saved.empty()) unsetenv("MTMD_BACKEND_DEVICE");
        else               setenv("MTMD_BACKEND_DEVICE", saved.c_str(), 1);
        if (!ctx_vision_npu.get()) { LOG_ERR("--fm-split: failed to load the vision model on %s\n", cli.backend.c_str()); return 1; }
    }
    // Head (first device) and tail (second device) contexts. ctx_vision is the OpenCL context and ctx_vision_tail the
    // NPU one. GN uses them as named, NG swaps them.
    mtmd_context * ctx_head = ctx_vision.get();
    mtmd_context * ctx_tail = ctx_vision_tail.get();
    if (ng) std::swap(ctx_head, ctx_tail);
    if (pp.vit_split > 0) {
        mtmd_set_vision_layer_range(ctx_head, 0,            pp.vit_split);
        mtmd_set_vision_layer_range(ctx_tail, pp.vit_split, -1);
        LOG_INF("vit-split [%s]: %s runs ViT layers [0,%d), %s runs [%d,end) + projector + prefix + FM\n",
                pp.direction.c_str(), ng ? cli.backend.c_str() : "GPU", pp.vit_split,
                ng ? "GPU" : cli.backend.c_str(), pp.vit_split);
    }

    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) prompt += mtmd_default_marker();
    if (model.pi05) params.prompt = pi0_format_prompt_pi05(params.prompt, nullptr); // pi0.5 prompt, zero state
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

    // Frame plan. Text embeddings are looked up once here so that the GPU worker stays OpenCL-only.
    std::vector<chunk_seg> plan;
    int prefix_len = 0;
    {
        const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
        std::vector<float> emb_row(PALI_N_EMBD);
        const float embd_scale = sqrtf((float) PALI_N_EMBD);
        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            chunk_seg seg;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                seg.is_image = false;
                seg.n_tokens = (int) n_tokens;
                seg.text_embd.reserve(n_tokens * PALI_N_EMBD);
                for (size_t t = 0; t < n_tokens; t++) {
                    embd_lookup_f32(model.pali_embed, tokens[t], emb_row.data());
                    for (float & e : emb_row) e *= embd_scale;
                    seg.text_embd.insert(seg.text_embd.end(), emb_row.begin(), emb_row.end());
                }
            } else {
                seg.is_image = true;
                seg.chunk    = chunk;
                seg.n_tokens = (int) mtmd_input_chunk_get_n_tokens(chunk);
            }
            prefix_len += seg.n_tokens;
            plan.push_back(std::move(seg));
        }
    }
    LOG_INF("prefix_len = %d tokens\n", prefix_len);

    pi0_session session;
    session.kv_type = pi0_ggml_type_from_string(cli.kv_type);
    if (session.kv_type == GGML_TYPE_COUNT) {
        LOG_ERR("Unknown --kv-type '%s'\n", cli.kv_type.c_str());
        return 1;
    }

    pi0_session session_gpu;               // deep cuts / plans: OpenCL LM / expert, KV copied in per frame
    session_gpu.kv_type   = session.kv_type;
    session_gpu.attn_mode = 1;             // standard attention on OpenCL
    // --fm-layer-split: the NPU session's expert graph covers layers [0, fl_j) (residual out). A second OpenCL session
    // covers [fl_j, 18) of step 0 (residual in), so the full-range OpenCL graph of the later steps is not rebuilt every
    // frame. Both OpenCL sessions get the prefix KV.
    pi0_session session_gpu_tail;
    session_gpu_tail.kv_type = session.kv_type; session_gpu_tail.attn_mode = 1;
    if (fl_j > 0) { session.expert_il_to = fl_j; session_gpu_tail.expert_il_from = fl_j; }
    // SigLIP-layer N->G cut: `session` runs on OpenCL (the -m set must be a GPU set) with standard attention
    if (ng && !deep) session.attn_mode = 1;

    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);
    const float dt = -1.0f / PI0_NUM_STEPS;
    // Context of the full-ViT encode_frame(): OpenCL, or the first device under deep cuts.
    mtmd_context * ctx_frame = deep ? ctx_vision_npu.get() : ctx_vision.get();

    // --vit-split head: only ViT layers [0, vit_split), returns the raw hidden state per image. clip writes it into the
    // output-embd buffer, which fits it because that buffer is sized for the 2048-wide projection.
    const int VIT_N_EMBD = 1152; // SigLIP So400m trunk width
    auto encode_head = [&](std::vector<std::vector<float>> & hidden) -> bool {
        hidden.clear();
        for (const chunk_seg & seg : plan) {
            if (!seg.is_image) continue;
            if (mtmd_encode_chunk(ctx_head, seg.chunk) != 0) {
                LOG_ERR("Vision head encode failed\n");
                return false;
            }
            const float * h = mtmd_get_output_embd(ctx_head);
            hidden.emplace_back(h, h + (size_t) seg.n_tokens * VIT_N_EMBD);
        }
        return true;
    };

    // --vit-split tail: ViT layers [vit_split, end) + projector for all images in one graph launch (clip builds one
    // chain per image), then stitch the prefix like encode_frame.
    auto encode_tail = [&](const std::vector<std::vector<float>> & hidden,
                           std::vector<float> & out) -> int {
        out.clear();
        out.reserve((size_t) prefix_len * PALI_N_EMBD);
        if (hidden.empty()) { LOG_ERR("vit-split: missing hidden state\n"); return -1; }

        std::vector<float> all;
        all.reserve(hidden.size() * hidden[0].size());
        for (const auto & h : hidden) all.insert(all.end(), h.begin(), h.end());
        mtmd_set_vision_hidden(ctx_tail, all.data(), all.size(), (int) hidden.size());

        const mtmd_input_chunk * first_img = nullptr;
        for (const chunk_seg & seg : plan) { if (seg.is_image) { first_img = seg.chunk; break; } }
        if (mtmd_encode_chunk(ctx_tail, first_img) != 0) {
            LOG_ERR("Vision tail encode failed\n");
            return -1;
        }
        const float * emb = mtmd_get_output_embd(ctx_tail);

        size_t img_i = 0;
        for (const chunk_seg & seg : plan) {
            if (seg.is_image) {
                const float * src = emb + img_i * (size_t) seg.n_tokens * PALI_N_EMBD;
                out.insert(out.end(), src, src + (size_t) seg.n_tokens * PALI_N_EMBD);
                img_i++;
            } else {
                out.insert(out.end(), seg.text_embd.begin(), seg.text_embd.end());
            }
        }
        return prefix_len;
    };

    // Encode one frame's images and stitch in the precomputed text embeddings. Returns prefix_len, or -1.
    auto encode_frame = [&](std::vector<float> & out) -> int {
        if (pp.vit_split > 0) {
            // warm-up: head and tail back to back on this thread
            std::vector<std::vector<float>> h;
            if (!encode_head(h)) return -1;
            return encode_tail(h, out);
        }
        out.clear();
        out.reserve((size_t) prefix_len * PALI_N_EMBD);
        // PI0_VIT_BATCH=1: all images through one vision graph
        static const int pi0_vit_batch = []{ const char * e = getenv("PI0_VIT_BATCH"); return e ? atoi(e) : 0; }();
        const float * batched = nullptr;
        size_t batched_off = 0;
        if (pi0_vit_batch) {
            std::vector<const mtmd_input_chunk *> imgs;
            for (const chunk_seg & seg : plan) if (seg.is_image) imgs.push_back(seg.chunk);
            if (imgs.size() > 1) {
                if (mtmd_encode_image_chunks(ctx_frame, imgs.data(), imgs.size()) != 0) {
                    LOG_ERR("Batched vision encode failed\n");
                    return -1;
                }
                batched = mtmd_get_output_embd(ctx_frame);
            }
        }
        for (const chunk_seg & seg : plan) {
            if (seg.is_image) {
                if (batched) {
                    out.insert(out.end(), batched + batched_off, batched + batched_off + (size_t) seg.n_tokens * PALI_N_EMBD);
                    batched_off += (size_t) seg.n_tokens * PALI_N_EMBD;
                    continue;
                }
                if (mtmd_encode_chunk(ctx_frame, seg.chunk) != 0) {
                    LOG_ERR("Vision encode failed\n");
                    return -1;
                }
                float * img = mtmd_get_output_embd(ctx_frame);
                out.insert(out.end(), img, img + (size_t) seg.n_tokens * PALI_N_EMBD);
            } else {
                out.insert(out.end(), seg.text_embd.begin(), seg.text_embd.end());
            }
        }
        return prefix_len;
    };

    // Initial flow-matching noise: fixed seed, or PI0_NOISE_BIN=<file> (raw float32 [HORIZON][DIM]).
    auto init_noise = [&](std::vector<float> & x_t) -> bool {
        x_t.assign(PI0_ACTION_DIM * PI0_ACTION_HORIZON, 0.0f);
        static const std::string noise_bin = []{ const char * e = getenv("PI0_NOISE_BIN"); return e ? std::string(e) : std::string(); }();
        if (!noise_bin.empty()) {
            FILE * nf = fopen(noise_bin.c_str(), "rb");
            size_t got = nf ? fread(x_t.data(), sizeof(float), x_t.size(), nf) : 0;
            if (nf) fclose(nf);
            if (got != x_t.size()) { LOG_ERR("PI0_NOISE_BIN: cannot read %zu floats from %s\n", x_t.size(), noise_bin.c_str()); return false; }
        } else {
            std::mt19937 rng(42);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (auto & v : x_t) v = normal(rng);
        }
        return true;
    };

    // Deep N->G stages. Stage 1 (cli.backend): prefix layers [0, lm_k) and, when lm_k == 18, the first fm_j denoising
    // steps. It reads the hand-off back to the host: the KV rows of these layers plus the residual (lm_k < 18) or the
    // noisy action state (lm_k == 18).
    // Stage 2 (OpenCL): writes the KV into its session, runs layers [lm_k, 18) from the residual (if any) and the
    // remaining denoising steps. fm_j == PI0_NUM_STEPS: no hand-off (all on the NPU).
    const bool deep_handoff = lm_k < PI0_N_LAYER || fm_j < PI0_NUM_STEPS;
    auto run_lm_stage1 = [&](const float * embd, int plen, std::vector<float> & x_t, std::vector<uint8_t> & kv,
                             std::vector<float> & hidden, double * prefix_ms, double * fm_ms, double * kvget_ms) -> bool {
        int64_t t0 = ggml_time_us();
        if (!run_paligemma_prefix_range(model, session, backend, embd, plen, 0, lm_k, lm_k < PI0_N_LAYER ? &hidden : nullptr)) return false;
        int64_t t1 = ggml_time_us();
        x_t.clear();
        if (lm_k == PI0_N_LAYER) {
            if (!init_noise(x_t)) return false;
            std::vector<float> v_t(x_t.size());
            for (int step = 0; step < fm_j; step++) {
                float t = 1.0f + step * dt;
                if (!run_expert_step(model, session, backend, robot_state.data(), x_t.data(), t, v_t.data())) return false;
                for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
            }
            if (fl_j > 0) {   // expert layers [0, fl_j) of step fm_j on the first device -> suffix residual
                hidden.resize((size_t) EXPERT_N_EMBD * pi0_expert_suffix_len(model));
                if (!run_expert_step_range(model, session, backend, robot_state.data(), x_t.data(), 1.0f + fm_j * dt,
                                           nullptr, hidden.data(), nullptr)) return false;
            }
        }
        int64_t t2 = ggml_time_us();
        if (deep_handoff && !pi0_kv_prefix_get(session, kv, 0, lm_k)) return false;
        int64_t t3 = ggml_time_us();
        if (prefix_ms) *prefix_ms = us_to_ms(t1 - t0);
        if (fm_ms)     *fm_ms     = us_to_ms(t2 - t1);
        if (kvget_ms)  *kvget_ms  = us_to_ms(t3 - t2);   // includes the residual read-back under --lm-split
        return true;
    };
    auto run_lm_stage2 = [&](std::vector<float> & x_t, const std::vector<uint8_t> & kv, const std::vector<float> & hidden, int plen,
                             double & mean, double & rms, double * kvset_ms, double * pre2_ms, double * fm_ms) -> bool {
        if (kvset_ms) *kvset_ms = 0;
        if (pre2_ms)  *pre2_ms  = 0;
        if (fm_ms)    *fm_ms    = 0;
        if (deep_handoff) {
            int64_t t0 = ggml_time_us();
            if (!pi0_session_prepare_kv(session_gpu, backend_gpu, plen)) return false;
            if (!pi0_kv_prefix_set(session_gpu, kv, 0, lm_k)) return false;
            int64_t t1 = ggml_time_us();
            if (lm_k < PI0_N_LAYER) {
                if (hidden.size() != (size_t) PALI_N_EMBD * plen) { LOG_ERR("lm-split: missing residual\n"); return false; }
                if (!run_paligemma_prefix_range(model_gpu, session_gpu, backend_gpu, hidden.data(), plen, lm_k, PI0_N_LAYER, nullptr)) return false;
            }
            int64_t t2 = ggml_time_us();
            if (x_t.empty() && !init_noise(x_t)) return false;
            std::vector<float> v_t(x_t.size());
            int first_full = fm_j;
            if (fl_j > 0) {   // layers [fl_j, 18) of step fm_j from the residual, on the second OpenCL session
                if (!pi0_session_prepare_kv(session_gpu_tail, backend_gpu, plen)) return false;
                if (!pi0_kv_prefix_set(session_gpu_tail, kv, 0, lm_k)) return false;
                if (hidden.size() != (size_t) EXPERT_N_EMBD * pi0_expert_suffix_len(model_gpu)) { LOG_ERR("fm-layer-split: missing suffix residual\n"); return false; }
                if (!run_expert_step_range(model_gpu, session_gpu_tail, backend_gpu, robot_state.data(), x_t.data(), 1.0f + fm_j * dt,
                                           hidden.data(), nullptr, v_t.data())) return false;
                for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
                first_full = fm_j + 1;
            }
            for (int step = first_full; step < PI0_NUM_STEPS; step++) {
                float t = 1.0f + step * dt;
                if (!run_expert_step(model_gpu, session_gpu, backend_gpu, robot_state.data(), x_t.data(), t, v_t.data())) return false;
                for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
            }
            int64_t t3 = ggml_time_us();
            if (kvset_ms) *kvset_ms = us_to_ms(t1 - t0);
            if (pre2_ms)  *pre2_ms  = us_to_ms(t2 - t1);
            if (fm_ms)    *fm_ms    = us_to_ms(t3 - t2);
        }
        action_stats(x_t, mean, rms);
        return true;
    };
    // warm-up: both deep stages back to back on this thread
    auto run_deep = [&](const float * embd, int plen, double & mean, double & rms) -> bool {
        std::vector<float> x, h; std::vector<uint8_t> kv;
        if (!run_lm_stage1(embd, plen, x, kv, h, nullptr, nullptr, nullptr)) return false;
        return run_lm_stage2(x, kv, h, plen, mean, rms, nullptr, nullptr, nullptr);
    };

    // Prefix + PI0_NUM_STEPS Euler steps on `backend`. Returns the action mean/rms and optionally the phase wall times.
    auto run_npu = [&](const float * embd, int plen, double & mean, double & rms,
                       double * prefix_ms = nullptr, double * fm_ms = nullptr) -> bool {
        int64_t tp0 = ggml_time_us();
        if (!run_paligemma_prefix(model, session, backend, embd, plen)) return false;
        int64_t tp1 = ggml_time_us();
        std::vector<float> x_t;
        if (!init_noise(x_t)) return false;
        std::vector<float> v_t(x_t.size());
        for (int step = 0; step < PI0_NUM_STEPS; step++) {
            float t = 1.0f + step * dt;
            if (!run_expert_step(model, session, backend, robot_state.data(), x_t.data(), t, v_t.data())) {
                return false;
            }
            for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
        }
        int64_t tp2 = ggml_time_us();
        if (prefix_ms) *prefix_ms = us_to_ms(tp1 - tp0);
        if (fm_ms)     *fm_ms     = us_to_ms(tp2 - tp1);
        action_stats(x_t, mean, rms);
        return true;
    };

    // ---- --plan: k-cut pipeline (one worker thread per device, static per-device order) ----
    if (kcut) {
        const int S = (int) kc.size();
        const int VIT_N_LAYER = KC_V - 2;   // 27 SigLIP blocks between the patch-embedding and projector atoms
        std::vector<const mtmd_input_chunk *> imgs;
        for (const chunk_seg & sg : plan) {
            if (!sg.is_image) continue;
            if (!imgs.empty() && (size_t) sg.n_tokens != mtmd_input_chunk_get_n_tokens(imgs[0])) {
                LOG_ERR("--plan: the images must have the same size\n"); return 1;
            }
            imgs.push_back(sg.chunk);
        }
        const int n_img = (int) imgs.size();
        if (n_img == 0) { LOG_ERR("--plan: no image in the prompt\n"); return 1; }
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(imgs[0]);
        const size_t per_h = n_tok * VIT_N_EMBD;    // SigLIP hidden floats per image
        const size_t per_e = n_tok * PALI_N_EMBD;   // projected floats per image
        // the prefix KV travels with the frame when an F segment or a partial L segment does not run all 18 layers
        bool kv_handoff = false;
        for (const kc_seg & g : kc) if (g.f1 > g.f0 && !g.lm) kv_handoff = true;
        for (const kc_seg & g : kc) if (g.l1 > g.l0 && !g.lm) kv_handoff = true;

        // kv = this frame's prefix KV of all 18 layers (pi0_kv_prefix_get layout, rows merged in as partial L segments run),
        // res = the PaliGemma residual after the last layer run so far (cut inside L)
        struct kc_frame { std::vector<float> hidden, embd, x_t, res; std::vector<uint8_t> kv; double mean = 0, rms = 0; };
        struct kc_dev   { mtmd_context * vctx; bool batched; pi0_model * m; pi0_session * s; ggml_backend_t be; };
        // GPU: one SigLIP graph per image. NPU: one graph for all images (PI0_VIT_BATCH=1 also batches a hidden-state input).
        const kc_dev dev_g = { ctx_vision.get(),     false, &model_gpu, &session_gpu, backend_gpu };
        const kc_dev dev_n = { ctx_vision_npu.get(), true,  &model,     &session,     backend     };
        struct kc_ms { double v = 0, l = 0, f = 0, kv = 0; };   // vision / LM / FM wall ms of one segment run (l, f include kv)
        // A segment with a partial PaliGemma range gets its own session. A session caches one prefix graph per layer range
        // and rebuilds it when the range changes. The frame's KV travels in kc_frame::kv, so the extra session only costs
        // its own KV storage.
        std::vector<pi0_session> seg_sess(S);
        for (int s = 0; s < S; s++) {
            seg_sess[s].kv_type   = session.kv_type;
            seg_sess[s].attn_mode = kc[s].dev == 'G' ? session_gpu.attn_mode : session.attn_mode;
        }
        auto sess_of = [&](int s) -> pi0_session & {
            return (kc[s].l1 > kc[s].l0 && !kc[s].lm) ? seg_sess[s] : (kc[s].dev == 'G' ? session_gpu : session);
        };

        auto run_seg = [&](int s, kc_frame & fr, kc_ms * ms) -> bool {
            const kc_seg & g = kc[s];
            const kc_dev & d = g.dev == 'G' ? dev_g : dev_n;
            const int64_t t0 = ggml_time_us();
            double kv_ms = 0;
            if (g.v1 > g.v0) {
                const bool in_h  = g.v0 >= 1;            // after the patch-embedding atom: the input is a hidden state
                const bool out_h = g.v1 <= KC_V - 1;     // before the projector atom: the output is a hidden state
                const int  il0   = in_h  ? g.v0 - 1 : 0;
                const int  il1   = out_h ? g.v1 - 1 : VIT_N_LAYER;
                if (in_h && fr.hidden.size() != per_h * n_img) { LOG_ERR("--plan %s: missing SigLIP hidden state\n", g.name.c_str()); return false; }
                mtmd_set_vision_segment(d.vctx, il0, il1, in_h, out_h);
                const size_t per_o = out_h ? per_h : per_e;
                std::vector<float> out(per_o * n_img);
                if (d.batched) {
                    int rc;
                    if (in_h) {
                        mtmd_set_vision_hidden(d.vctx, fr.hidden.data(), fr.hidden.size(), n_img);
                        rc = mtmd_encode_chunk(d.vctx, imgs[0]);
                    } else {
                        rc = mtmd_encode_image_chunks(d.vctx, imgs.data(), imgs.size());
                    }
                    if (rc != 0) { LOG_ERR("--plan %s: vision encode failed\n", g.name.c_str()); return false; }
                    const float * o = mtmd_get_output_embd(d.vctx);
                    std::copy(o, o + out.size(), out.begin());
                } else {
                    for (int b = 0; b < n_img; b++) {
                        if (in_h) mtmd_set_vision_hidden(d.vctx, fr.hidden.data() + b * per_h, per_h, 1);
                        if (mtmd_encode_chunk(d.vctx, imgs[b]) != 0) { LOG_ERR("--plan %s: vision encode failed (image %d)\n", g.name.c_str(), b); return false; }
                        const float * o = mtmd_get_output_embd(d.vctx);
                        std::copy(o, o + per_o, out.begin() + b * per_o);
                    }
                }
                if (out_h) {
                    fr.hidden.swap(out);
                } else {
                    // projected image embeddings + the precomputed text embeddings = the prefix (encode_tail's stitching)
                    fr.hidden.clear();
                    fr.embd.clear();
                    fr.embd.reserve((size_t) prefix_len * PALI_N_EMBD);
                    int b = 0;
                    for (const chunk_seg & sg : plan) {
                        if (sg.is_image) { fr.embd.insert(fr.embd.end(), out.begin() + b * per_e, out.begin() + (b + 1) * per_e); b++; }
                        else             { fr.embd.insert(fr.embd.end(), sg.text_embd.begin(), sg.text_embd.end()); }
                    }
                }
            }
            const int64_t t1 = ggml_time_us();
            if (g.lm) {
                if (fr.embd.size() != (size_t) prefix_len * PALI_N_EMBD) { LOG_ERR("--plan %s: missing prefix embeddings\n", g.name.c_str()); return false; }
                if (!run_paligemma_prefix(*d.m, *d.s, d.be, fr.embd.data(), prefix_len)) return false;
                fr.embd.clear();
                if (kv_handoff) {
                    const int64_t k0 = ggml_time_us();
                    if (!pi0_kv_prefix_get(*d.s, fr.kv, 0, PI0_N_LAYER)) return false;
                    kv_ms += us_to_ms(ggml_time_us() - k0);
                }
            } else if (g.l1 > g.l0) {
                // PaliGemma layers [l0, l1) only, from the prefix embeddings (l0 == 0) or the previous L part's residual.
                // Each layer's attention needs only its own KV, which this pass writes. The rows are copied into the frame's
                // 18-layer KV blob at once, because this device may run another frame before this frame's FM steps.
                const std::vector<float> & in = g.l0 == 0 ? fr.embd : fr.res;
                if (in.size() != (size_t) prefix_len * PALI_N_EMBD) { LOG_ERR("--plan %s: missing %s\n", g.name.c_str(), g.l0 == 0 ? "prefix embeddings" : "PaliGemma residual"); return false; }
                std::vector<float> out_res;
                if (!run_paligemma_prefix_range(*d.m, sess_of(s), d.be, in.data(), prefix_len, g.l0, g.l1, g.l1 < PI0_N_LAYER ? &out_res : nullptr)) return false;
                if (g.l0 == 0) fr.embd.clear();
                fr.res.swap(out_res);
                const int64_t k0 = ggml_time_us();
                std::vector<uint8_t> part;
                if (!pi0_kv_prefix_get(sess_of(s), part, g.l0, g.l1)) return false;
                const size_t row = part.size() / (2 * (size_t) (g.l1 - g.l0));   // bytes of one layer's K (or V) rows
                if (fr.kv.size() != 2 * (size_t) PI0_N_LAYER * row) fr.kv.assign(2 * (size_t) PI0_N_LAYER * row, 0);
                std::copy(part.begin(), part.end(), fr.kv.begin() + 2 * (size_t) g.l0 * row);
                kv_ms += us_to_ms(ggml_time_us() - k0);
            }
            const int64_t t2 = ggml_time_us();
            if (g.f1 > g.f0) {
                if (!g.lm) {
                    // this device's KV cache may hold another frame's prefix by now: write this frame's back
                    const int64_t k0 = ggml_time_us();
                    if (!pi0_session_prepare_kv(sess_of(s), d.be, prefix_len)) return false;
                    if (!pi0_kv_prefix_set(sess_of(s), fr.kv, 0, PI0_N_LAYER)) return false;
                    kv_ms += us_to_ms(ggml_time_us() - k0);
                }
                if (g.f0 == 0 && !init_noise(fr.x_t)) return false;
                if (fr.x_t.size() != (size_t) PI0_ACTION_DIM * PI0_ACTION_HORIZON) { LOG_ERR("--plan %s: missing action state\n", g.name.c_str()); return false; }
                std::vector<float> v_t(fr.x_t.size());
                for (int step = g.f0; step < g.f1; step++) {
                    const float t = 1.0f + step * dt;
                    if (!run_expert_step(*d.m, sess_of(s), d.be, robot_state.data(), fr.x_t.data(), t, v_t.data())) return false;
                    for (size_t i = 0; i < fr.x_t.size(); i++) fr.x_t[i] += dt * v_t[i];
                }
                if (g.f1 == PI0_NUM_STEPS) { action_stats(fr.x_t, fr.mean, fr.rms); fr.kv.clear(); }
            }
            const int64_t t3 = ggml_time_us();
            if (ms) { ms->v = us_to_ms(t1 - t0); ms->l = us_to_ms(t2 - t1); ms->f = us_to_ms(t3 - t2); ms->kv = kv_ms; }
            return true;
        };
        auto seg_ms_str = [](const kc_ms & m) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%.1f (V %.1f L %.1f F %.1f, KV %.1f)", m.v + m.l + m.f, m.v, m.l, m.f, m.kv);
            return std::string(buf);
        };

        printf("\n========================================================================\n");
        printf(" PI0 k-cut pipeline, k = %d: ", S - 1);
        for (int s = 0; s < S; s++) printf("%s%s", s ? " " : "", kc[s].name.c_str());
        printf("\n cuts at atoms");
        for (int s = 1; s < S; s++) printf(" %d", kc[s].a);
        printf("  (V [0,29) L [29,47) F [47,57)); G = OpenCL, N = %s; prefix_len=%d\n", cli.backend.c_str(), prefix_len);
        printf(" schedule: static order --plan-order \"%s\"\n", pp.plan_order.c_str());
        printf(" frames=%d (warm-up %d + measured %d + drain %d)\n", N, pp.warmup, pp.measure, drain);
        printf("========================================================================\n");

        // warm-up pass, not counted: every segment once, back to back on this thread (builds every graph of the plan)
        {
            kc_frame fr;
            for (int s = 0; s < S; s++) {
                kc_ms m;
                if (!run_seg(s, fr, &m)) { LOG_ERR("--plan warm-up failed in segment %s\n", kc[s].name.c_str()); return 1; }
                LOG_INF("warm-up %-14s %s ms\n", kc[s].name.c_str(), seg_ms_str(m).c_str());
            }
            LOG_INF("Warmup ok (action mean=%.6f rms=%.6f)\n", fr.mean, fr.rms);
        }

        {
            std::mutex km;
            std::condition_variable kcv;
            std::vector<std::vector<char>>    kdone(S, std::vector<char>(N, 0)), kstart(S, std::vector<char>(N, 0));
            std::vector<std::vector<int64_t>> t0s(S, std::vector<int64_t>(N, 0)), t1s(S, std::vector<int64_t>(N, 0));
            std::vector<std::vector<kc_ms>>   msx(S, std::vector<kc_ms>(N));
            std::vector<kc_frame> frs(N);
            std::vector<int64_t>  start_us(N, 0), complete_us(N, 0);
            bool kok = true;

            FILE * trace_f = nullptr;
            if (const char * tp = getenv("PI0_PIPE_TRACE")) {
                trace_f = fopen(tp, "w");
                if (trace_f) {
                    // the first 10 columns keep the two-stage trace layout (vision_ms / npu_ms = the frame's GPU / NPU load,
                    // split = k, t_infer = max of the two), then one wall time per segment
                    fprintf(trace_f, "frame,admitted_ms,complete_ms,ii_ms,latency_ms,vision_ms,npu_ms,split,t_infer_ms,t_output_ms");
                    for (int s = 0; s < S; s++) fprintf(trace_f, ",s%d_ms", s);
                    fprintf(trace_f, "\n");
                    fflush(trace_f);
                } else {
                    LOG_ERR("PI0_PIPE_TRACE: cannot open %s\n", tp);
                }
            }
            const int64_t wall0 = ggml_time_us();

            // bookkeeping after segment s of frame f ended at t1 (called with km held)
            auto finish = [&](int s, int f, bool good, int64_t t1) {
                t1s[s][f]   = t1;
                kdone[s][f] = 1;
                if (!good) {
                    LOG_ERR("--plan: segment %s failed on frame %d\n", kc[s].name.c_str(), f);
                    kok = false;
                } else if (s == S - 1) {
                    complete_us[f] = t1;
                    if (trace_f) {
                        double lg = 0, ln = 0;
                        for (int j = 0; j < S; j++) (kc[j].dev == 'G' ? lg : ln) += us_to_ms(t1s[j][f] - t0s[j][f]);
                        fprintf(trace_f, "%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%.1f,0.0", f, us_to_ms(start_us[f] - wall0),
                                us_to_ms(t1 - wall0), f > 0 ? us_to_ms(t1 - complete_us[f - 1]) : 0.0,
                                us_to_ms(t1 - start_us[f]), lg, ln, S - 1, std::max(lg, ln));
                        for (int j = 0; j < S; j++) fprintf(trace_f, ",%.1f", us_to_ms(t1s[j][f] - t0s[j][f]));
                        fprintf(trace_f, "\n");
                        fflush(trace_f);
                    }
                }
            };

            // --plan-order: a static order per device = one cycle of the steady state. In cycle p the device runs its
            // entries (s, off) for frame f = c * p + off in the listed order and waits only for segment s-1 of f.
            // There is no run-time choice, so a run cannot drift into another steady state.
            int cyc_done[2] = { -1, -1 };   // last cycle each device (0 = G, 1 = N) has finished ("sync" barrier)
            auto worker_static = [&](char dev) {
                const std::vector<std::pair<int, int>> & ord = dev == 'G' ? kord_g : kord_n;
                const int me = dev == 'G' ? 0 : 1;
                int min_off = 1 << 30;
                for (const auto & e : ord) min_off = std::min(min_off, e.second);
                std::unique_lock<std::mutex> lk(km);
                for (int p = 0; kok && !ord.empty() && kord_c * p + min_off < N; p++) {
                    if (kord_sync && p > 0) {
                        // the next period starts when both chains of the previous one are done
                        while (kok && cyc_done[1 - me] < p - 1) {
                            if (kcv.wait_for(lk, std::chrono::seconds(60)) == std::cv_status::timeout && cyc_done[1 - me] < p - 1) {
                                LOG_ERR("--plan-order sync: %c waited 60 s for cycle %d of the other device\n", dev, p - 1);
                                kok = false;
                            }
                        }
                        if (!kok) break;
                    }
                    for (const auto & e : ord) {
                        const int s = e.first, f = kord_c * p + e.second;
                        if (!kok) break;
                        if (f < 0 || f >= N) continue;
                        while (kok && s > 0 && !kdone[s - 1][f]) {
                            if (kcv.wait_for(lk, std::chrono::seconds(60)) == std::cv_status::timeout && !kdone[s - 1][f]) {
                                LOG_ERR("--plan-order: %c waited 60 s for segment %d of frame %d (the order deadlocks)\n", dev, s - 1, f);
                                kok = false;
                            }
                        }
                        if (!kok) break;
                        kstart[s][f] = 1;
                        t0s[s][f] = ggml_time_us();
                        if (s == 0) start_us[f] = t0s[s][f];
                        lk.unlock();
                        const bool good = run_seg(s, frs[f], &msx[s][f]);
                        const int64_t t1 = ggml_time_us();
                        lk.lock();
                        finish(s, f, good, t1);
                        kcv.notify_all();
                    }
                    cyc_done[me] = p;
                    kcv.notify_all();
                }
                cyc_done[me] = 1 << 29;   // finished (or failed): never hold the other device at the barrier
                kcv.notify_all();
            };
            std::thread th_g(worker_static, 'G'), th_n(worker_static, 'N');
            th_g.join();
            th_n.join();
            const int64_t wall1 = ggml_time_us();
            if (trace_f) fclose(trace_f);
            if (!kok) { LOG_ERR("--plan pipeline failed\n"); return 1; }

            // measured window: completions W..W+M-1, period over the M gaps ending there, drain frames excluded
            const int k0   = pp.warmup - 1;
            const int last = pp.warmup + pp.measure - 1;
            const double nf = (double) pp.measure;
            const double II_ms = us_to_ms(complete_us[last] - complete_us[k0]) / nf;
            double lat = 0, lat_min = 1e18, lat_max = 0, kv_sum = 0;
            for (int f = k0 + 1; f <= last; f++) {
                const double l = us_to_ms(complete_us[f] - start_us[f]);
                lat += l; lat_min = std::min(lat_min, l); lat_max = std::max(lat_max, l);
                for (int s = 0; s < S; s++) kv_sum += msx[s][f].kv;
            }
            lat /= nf;
            const bool drain_free = start_us[N - 1] < complete_us[last];
            const double wall_ms = us_to_ms(wall1 - wall0);
            printf("\n[pipeline] %d frames in %.1f ms  ->  %.2f Hz effective\n", N, wall_ms, N * 1000.0 / wall_ms);
            printf("           steady-state period %.1f ms  (%.2f Hz) over the measured frames %d..%d\n", II_ms, 1000.0 / II_ms, k0 + 1, last);
            printf("           ii per frame:");
            for (int f = 1; f < N; f++) printf(" %d:%.1f", f, us_to_ms(complete_us[f] - complete_us[f - 1]));
            printf("\n           segment times over frames %d..%d (ms, mean | min / max):\n", k0 + 1, last);
            double dg = 0, dn = 0, bg = 0, bn = 0;
            for (int s = 0; s < S; s++) {
                kc_ms m; double mn = 1e18, mx = 0;
                for (int f = k0 + 1; f <= last; f++) {
                    const kc_ms & x = msx[s][f];
                    m.v += x.v; m.l += x.l; m.f += x.f; m.kv += x.kv;
                    const double w = us_to_ms(t1s[s][f] - t0s[s][f]);
                    mn = std::min(mn, w); mx = std::max(mx, w);
                }
                m.v /= nf; m.l /= nf; m.f /= nf; m.kv /= nf;
                printf("             s%d %-14s %s | %.1f / %.1f\n", s, kc[s].name.c_str(), seg_ms_str(m).c_str(), mn, mx);
                (kc[s].dev == 'G' ? dg : dn) += m.v + m.l + m.f;
            }
            for (int s = 0; s < S; s++) for (int f = 0; f < N; f++) (kc[s].dev == 'G' ? bg : bn) += us_to_ms(t1s[s][f] - t0s[s][f]);
            printf("           device load per frame: GPU %.1f | NPU %.1f ms (bound by %s) | busy over the run GPU %.0f%% NPU %.0f%%\n",
                   dg, dn, dg >= dn ? "GPU" : "NPU", 100.0 * bg / wall_ms, 100.0 * bn / wall_ms);
            printf("           steady-state latency %.1f ms (obs->action, min %.1f / max %.1f, L/T %.2f, window f=%d..%d)\n",
                   lat, lat_min, lat_max, lat / II_ms, k0 + 1, last);
            printf("           drain_free=%d (frame %d admitted before frame %d completed)\n", (int) drain_free, N - 1, last);
            printf("           timeline (ms since start; segment s: start-end%s):\n", N > 40 ? ", first 40 frames" : "");
            for (int f = 0; f < std::min(N, 40); f++) {
                printf("             f%d", f);
                for (int s = 0; s < S; s++) printf("  s%d %c %.0f-%.0f", s, kc[s].dev, us_to_ms(t0s[s][f] - wall0), us_to_ms(t1s[s][f] - wall0));
                printf("\n");
            }
            double r_min = 1e18, r_max = 0;
            for (int f = 0; f < N; f++) { r_min = std::min(r_min, frs[f].rms); r_max = std::max(r_max, frs[f].rms); }
            printf("           action mean=%.6f rms=%.6f (rms over the %d frames %.6f .. %.6f)\n", frs[N - 1].mean, frs[N - 1].rms, N, r_min, r_max);
            if (const char * ap = getenv("PI0_ACTIONS_OUT")) {
                FILE * af = fopen(ap, "wb");
                const size_t nw = af ? fwrite(frs[N - 1].x_t.data(), sizeof(float), frs[N - 1].x_t.size(), af) : 0;
                if (af) fclose(af);
                printf("           actions of frame %d -> %s (%zu floats)\n", N - 1, ap, nw);
            }
            if (const char * tp = getenv("PI0_PIPE_TRACE")) printf("           per-frame trace -> %s\n", tp);
            const bool g_first = kc[0].dev == 'G';
            printf("RESULT tool=pipeline mode=plan dir=%s p=%d c=- k=%d head=%s warmup=%d measured=%d drain=%d frames=%d "
                   "period_ms=%.1f latency_ms=%.1f lt=%.3f stage1_ms=%.1f stage2_ms=%.1f kv_ms=%.1f drain_free=%d rms=%.6f\n",
                   g_first ? "GN" : "NG", kc[1].a, S - 1, g_first ? "GPU" : "NPU", pp.warmup, pp.measure, drain, N,
                   II_ms, lat, lat / II_ms, g_first ? dg : dn, g_first ? dn : dg, kv_sum / nf, (int) drain_free, frs[N - 1].rms);
            fflush(stdout);
        }
        printf("\n");

        for (int s = 0; s < S; s++) if (kc[s].l1 > kc[s].l0 && !kc[s].lm) pi0_session_free(seg_sess[s]);
        pi0_session_free(session);
        free_pi0_model(model);
        ggml_backend_free(backend);
        if (backend_gpu) {
            pi0_session_free(session_gpu);
            free_pi0_model(model_gpu);
            ggml_backend_free(backend_gpu);
        }
        return 0;
    }

    // ---- Warm-up pass (not counted): one frame through both stages back to back on this thread ----
    {
        std::vector<float> e;
        double mn, rs;
        const bool okw = encode_frame(e) >= 0 && (deep ? run_deep(e.data(), prefix_len, mn, rs) : run_npu(e.data(), prefix_len, mn, rs));
        if (!okw) {
            LOG_ERR("Warmup failed\n");
            return 1;
        }
        LOG_INF("Warmup ok (action mean=%.6f rms=%.6f)\n", mn, rs);
    }

    // stage 1 = the first device (head), stage 2 = the second device
    const char * dev1 = ng ? "NPU" : "GPU";
    const char * dev2 = ng ? "GPU" : "NPU";
    printf("\n========================================================================\n");
    if (deep) {
        printf(" PI0 two-stage pipeline, direction NG, cut p=%d c=%d: stage 1 = %s (ViT + prefix layers [0,%d) + FM steps [0,%d) + expert layers [0,%d) of step %d), stage 2 = GPU(OpenCL) (the rest; KV%s copied)\n",
               cut_p, cut_c, cli.backend.c_str(), lm_k, fm_j, fl_j, fm_j, (lm_k < PI0_N_LAYER || fl_j > 0) ? " + residual" : "");
    } else {
        printf(" PI0 two-stage pipeline, direction %s, cut p=%d c=%d: stage 1 = %s (SigLIP head), stage 2 = %s (tail + prefix + FM)\n",
               pp.direction.c_str(), cut_p, cut_c, ng ? cli.backend.c_str() : "GPU(OpenCL)", ng ? "GPU(OpenCL)" : cli.backend.c_str());
    }
    printf(" frames=%d (warm-up %d + measured %d + drain %d)  diffusion steps=%d  prefix_len=%d\n",
           N, pp.warmup, pp.measure, drain, PI0_NUM_STEPS, prefix_len);
    printf("========================================================================\n");

    // ---- Two-thread pipeline over a stream of frames (eager admission, double-buffered hand-off) ----
    {
        frame_queue q(2); // double-buffer
        std::vector<double> vision_ms(N, 0), npu_ms(N, 0), kvget_ms(N, 0), kvset_ms(N, 0);   // stage 1 / stage 2 wall
        std::vector<int64_t> complete_us(N, 0), start_us(N, 0), handoff_us(N, 0); // admitted / handed to stage 2
        std::atomic<bool> ok{true};
        std::atomic<bool> aborted{false};   // --abort-latency hit: the head stops admitting
        int n_over = 0;                     // consecutive frames above --abort-latency (second-stage thread only)
        double pipe_mean = 0, pipe_rms = 0;

        int64_t wall0 = ggml_time_us();

        // PI0_PIPE_TRACE=<path>: per-frame trace, opened before the workers start and flushed after every frame so that a
        // hung DSP call cannot lose the data. Times are ms since wall0. ii_ms = gap to the previous completion.
        FILE * trace_f = nullptr;
        if (const char * tp = getenv("PI0_PIPE_TRACE")) {
            trace_f = fopen(tp, "w");
            if (trace_f) {
                fprintf(trace_f, "frame,admitted_ms,complete_ms,ii_ms,latency_ms,vision_ms,npu_ms,split,t_infer_ms,t_output_ms\n");
                fflush(trace_f);
            } else {
                LOG_ERR("PI0_PIPE_TRACE: cannot open %s\n", tp);
            }
        }

        // completion bookkeeping (second-stage worker thread)
        auto on_complete = [&](int f, int64_t t1, double mn, double rs) {
            complete_us[f] = t1;
            if (trace_f) {
                // frames complete in admission order (single consumer on a FIFO), so f-1 is final here
                const double ii = f > 0 ? us_to_ms(complete_us[f] - complete_us[f - 1]) : 0.0;
                fprintf(trace_f, "%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%.1f,%.1f\n", f, us_to_ms(start_us[f] - wall0),
                        us_to_ms(complete_us[f] - wall0), ii, us_to_ms(complete_us[f] - start_us[f]),
                        vision_ms[f], npu_ms[f], (!deep && pp.vit_split > 0) ? pp.vit_split : 0,
                        std::max(vision_ms[f], npu_ms[f]), 0.0);
                fflush(trace_f);
            }
            pipe_mean = mn; pipe_rms = rs;   // last completed frame
            if (pp.abort_ms > 0) {
                n_over = us_to_ms(complete_us[f] - start_us[f]) > pp.abort_ms ? n_over + 1 : 0;
                if (n_over >= 5) aborted = true;
            }
        };

        std::thread gpu_worker([&] {
            for (int f = 0; f < N && ok && !aborted; f++) {
                frame_item it;
                it.frame_id = f;
                if (pp.period_ms > 0) {   // fixed-period admission: wait for the slot of frame f, start at once if late
                    const int64_t due = wall0 + (int64_t) (f * pp.period_ms * 1000.0);
                    while (ggml_time_us() < due && !aborted) {
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    if (aborted) break;
                }
                int64_t t0 = ggml_time_us();
                start_us[f] = t0;
                if (deep) {
                    // first-device stage of a deep cut, hands out x_t / residual + prefix KV
                    it.prefix_len = encode_frame(it.embd);
                    if (it.prefix_len > 0 && !run_lm_stage1(it.embd.data(), it.prefix_len, it.x_t, it.kv, it.hidden_lm, nullptr, nullptr, &kvget_ms[f])) { ok = false; break; }
                    it.embd.clear(); it.embd.shrink_to_fit();
                } else if (pp.vit_split > 0) {
                    // head only, the tail runs in the second device's period
                    if (!encode_head(it.hidden)) { ok = false; break; }
                    it.prefix_len = prefix_len;
                } else {
                    it.prefix_len = encode_frame(it.embd);
                }
                vision_ms[f] = us_to_ms(ggml_time_us() - t0);
                if (it.prefix_len < 0) { ok = false; break; }
                q.push(std::move(it));
                handoff_us[f] = ggml_time_us();
            }
            { frame_item sentinel; sentinel.frame_id = -1; q.push(std::move(sentinel)); }
        });

        std::thread npu_worker([&] {
            for (;;) {
                frame_item it = q.pop();
                if (it.frame_id < 0) break;
                int64_t t0 = ggml_time_us();
                double mn, rs;
                if (deep) {
                    if (!run_lm_stage2(it.x_t, it.kv, it.hidden_lm, it.prefix_len, mn, rs, &kvset_ms[it.frame_id], nullptr, nullptr)) { ok = false; break; }
                } else {
                    if (pp.vit_split > 0 && encode_tail(it.hidden, it.embd) < 0) { ok = false; break; }
                    if (!run_npu(it.embd.data(), it.prefix_len, mn, rs)) { ok = false; break; }
                }
                int64_t t1 = ggml_time_us();
                npu_ms[it.frame_id]     = us_to_ms(t1 - t0);
                on_complete(it.frame_id, t1, mn, rs);
            }
        });

        gpu_worker.join();
        npu_worker.join();
        int64_t wall1 = ggml_time_us();
        if (trace_f) { fclose(trace_f); trace_f = nullptr; }
        if (!ok) { LOG_ERR("pipeline stage failed\n"); return 1; }

        if (pp.period_ms > 0) {
            // sustained run: report the completed frames. The verdict comes from the trace (vladuo.py summary sustain).
            int n_done = 0;
            double lat_max = 0, lat_last = 0, ti_last = 0;
            for (int f = 0; f < N && complete_us[f] > 0; f++) {
                n_done   = f + 1;
                lat_last = us_to_ms(complete_us[f] - start_us[f]);
                lat_max  = std::max(lat_max, lat_last);
                ti_last  = std::max(vision_ms[f], npu_ms[f]);
            }
            const double el_s = n_done > 0 ? us_to_ms(complete_us[n_done - 1] - wall0) / 1000.0 : 0.0;
            printf("\n[sustain] period %.1f ms: %d of %d frames in %.1f s%s, latency max %.1f ms, last %.1f ms\n",
                   pp.period_ms, n_done, N, el_s, aborted ? " (stopped: latency limit)" : "", lat_max, lat_last);
            printf("RESULT tool=pipeline mode=sustain dir=%s p=%d c=%d head=%s period_set_ms=%.1f abort_ms=%.1f frames=%d "
                   "aborted=%d elapsed_s=%.1f latency_max_ms=%.1f latency_last_ms=%.1f t_infer_last_ms=%.1f rms=%.6f\n",
                   pp.direction.c_str(), cut_p, cut_c, dev1, pp.period_ms, pp.abort_ms, n_done, (int) aborted.load(), el_s,
                   lat_max, lat_last, ti_last, pipe_rms);
            fflush(stdout);
        } else {
            double wall_ms = us_to_ms(wall1 - wall0);
            double sum_vision = 0, sum_npu = 0;
            for (int f = 0; f < N; f++) { sum_vision += vision_ms[f]; sum_npu += npu_ms[f]; }

            // Measured window: completions W..W+M-1. The period is the mean gap over the M completions ending there.
            // The drain frames are not measured. They keep the head working (or waiting for a free buffer, as in steady
            // state) until the last measured frame completes.
            const int k    = pp.warmup - 1;
            const int last = pp.warmup + pp.measure - 1;
            const double nf = (double) pp.measure;
            const double II_ms = us_to_ms(complete_us[last] - complete_us[k]) / nf;
            const double gpu_busy = 100.0 * sum_vision / wall_ms;
            const double npu_busy = 100.0 * sum_npu    / wall_ms;
            const double stall_ms = us_to_ms(q.consumer_wait_us);
            // Observation-to-action latency: mean over the measured frames of (action complete - observation admitted).
            // With eager admission it grows with the hand-off backlog, so it depends on the window.
            double lat_sum = 0, lat_min = 1e18, lat_max = 0;
            double g_sum = 0, g_min = 1e18, g_max = 0, n_sum = 0, n_min = 1e18, n_max = 0, kv_sum = 0;
            for (int f = k + 1; f <= last; f++) {
                const double l = us_to_ms(complete_us[f] - start_us[f]);
                lat_sum += l; lat_min = std::min(lat_min, l); lat_max = std::max(lat_max, l);
                g_sum += vision_ms[f]; g_min = std::min(g_min, vision_ms[f]); g_max = std::max(g_max, vision_ms[f]);
                n_sum += npu_ms[f];    n_min = std::min(n_min, npu_ms[f]);    n_max = std::max(n_max, npu_ms[f]);
                kv_sum += kvget_ms[f] + kvset_ms[f];
            }
            const double lat_ms = lat_sum / nf;
            // drain-free: the head handed off its last frame only after the last measured frame completed, so it never ran
            // out of frames while a measured frame was in stage 2
            const bool drain_free = handoff_us[N - 1] >= complete_us[last];

            printf("\n[pipeline] %d frames in %.1f ms  ->  %.2f Hz effective\n", N, wall_ms, N * 1000.0 / wall_ms);
            printf("           steady-state period %.1f ms  (%.2f Hz) over the measured frames %d..%d\n", II_ms, 1000.0 / II_ms, k + 1, last);
            printf("           stage 1 %s %.1f ms (min %.1f / max %.1f) | stage 2 %s %.1f ms (min %.1f / max %.1f) | bound by %s\n",
                   dev1, g_sum / nf, g_min, g_max, dev2, n_sum / nf, n_min, n_max, (g_sum >= n_sum) ? dev1 : dev2);
            if (deep) {
                printf("           hand-off KV get + set %.1f ms/frame\n", kv_sum / nf);
            }
            printf("           busy over the run: stage 1 %s %.0f%% | stage 2 %s %.0f%% | stage 2 stall %.1f ms\n",
                   dev1, gpu_busy, dev2, npu_busy, stall_ms);
            printf("           latency %.1f ms (admitted->action, min %.1f / max %.1f, L/T %.2f, window f=%d..%d; eager admission:\n"
                   "           grows with the hand-off backlog, window-dependent)\n",
                   lat_ms, lat_min, lat_max, lat_ms / II_ms, k + 1, last);
            printf("           drain_free=%d (frame %d handed off after frame %d completed)\n", (int) drain_free, N - 1, last);
            printf("           action mean=%.6f rms=%.6f\n", pipe_mean, pipe_rms);
            if (const char * tp = getenv("PI0_PIPE_TRACE")) {
                printf("           per-frame trace -> %s (written incrementally, %d frames)\n", tp, N);
            }
            printf("RESULT tool=pipeline mode=two-stage dir=%s p=%d c=%d head=%s warmup=%d measured=%d drain=%d frames=%d "
                   "period_ms=%.1f latency_ms=%.1f lt=%.3f stage1_ms=%.1f stage2_ms=%.1f kv_ms=%.1f drain_free=%d rms=%.6f\n",
                   pp.direction.c_str(), cut_p, cut_c, dev1, pp.warmup, pp.measure, drain, N,
                   II_ms, lat_ms, lat_ms / II_ms, g_sum / nf, n_sum / nf, kv_sum / nf, (int) drain_free, pipe_rms);
            fflush(stdout);
        }
    }

    printf("\n");

    pi0_session_free(session);
    free_pi0_model(model);
    ggml_backend_free(backend);
    if (deep) {
        pi0_session_free(session_gpu);
        pi0_session_free(session_gpu_tail);
        free_pi0_model(model_gpu);
        ggml_backend_free(backend_gpu);
    }
    return 0;
}
