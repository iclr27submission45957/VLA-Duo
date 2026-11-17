#!/usr/bin/env bash
# Build, model preparation, device runs (one phone over adb), profiling, cut replay and summaries. See --help.
set -euo pipefail
shopt -s inherit_errexit

REL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORK=${VLADUO_WORK:-$REL/work}; OUT=${VLADUO_OUT:-$REL/out}
MODELS=${VLADUO_MODELS:-$REL/models}; RESULTS=${VLADUO_RESULTS:-$REL/results}
DEV_ROOT=${DEV_ROOT:-/data/local/tmp/vladuo}; IMAGE=${VLADUO_IMAGE:-vladuo-env:1}
DRYRUN=${DRYRUN:-}; PY=${PYTHON:-python3}
B8414=5744d7ec430e2f875a393770195fda530560773f; HEX=178a6c44937154dc4c4eff0d166f4a044c4fceba
IMG=inputs/cam_high.jpg,inputs/cam_left_wrist.jpg,inputs/cam_right_wrist.jpg; PROMPT='pick up the red block'
GGUFS="pi0-mmproj.gguf pi0-gemma-2b.gguf pi0-gemma-2b-fused.gguf pi0-action-expert.gguf"
BINS="llama-pi0-bench llama-pi0-pipeline llama-pi0-opmat-bench"
LIBS="libggml.so libggml-base.so libggml-cpu.so libggml-opencl.so libggml-hexagon.so libllama.so libmtmd.so libggml-htp-v79.so libggml-htp-v81.so"

usage() {
	cat <<'EOF'
usage: ./vladuo.sh <command> [options]       DRYRUN=1 or --dry-run: print the docker/adb commands and jobs, run nothing

host (Docker, no phone)
  build                              fetch llama.cpp 5744d7ec + ggml-hexagon, apply patches/vla-duo.patch, copy src/,
                                     build the Android binaries, both HTP skels and the host llama-quantize -> out/
  models [--heads hf16|q4|both] [--model pi0|pi05|all]
                                     download the public checkpoints + tokenizer, build the GGUF sets -> models/sets
  replay rates|select|curve|kcut [ARGS]   cut model on a profile (tools/vladuo.py), e.g. replay select --soc sd8elite --model pi0
  promote FILE [--key SOC/CFG]       add a QUICK profile.json to data/profiles.json (never overwrites a key)
  summary burst|table2|table3|table4|table6|kcut|fig6|sustain [--soc S] [--cfg C] [--heads H] [--window A:B] [DIR...]
                                     tables from pulled batches -> results/<soc>/summary/ (fig6 runs in the container)
  check [--fast]                     offline checks: file list, scrub, patch apply, build outputs, model md5
                                     (--fast skips it), tools/vladuo.py check; prints PASS/FAIL

device (adb, one phone; batches run detached on the phone and are pulled to results/<soc>/<batch>/)
  push [--heads hf16|q4|both] [--model M] [--force]         binaries, libraries, GGUF sets, inputs (md5-verified)
  profile quick [--cfg C]                                   cut-model inputs: 26 opmat + 10 calibration jobs (~30 min)
  profile full [--cfg C] [--model pi0|pi05] [--small-bg]    full 20 x 20 co-run matrices (1.5-2.3 h)
  burst npu|gpu [--model pi0|pi05|all]                      Table 2 single-device rows
  burst duo [--model M] [--cut auto|P|LABEL|"ARGS"]          Table 2 periodic row (auto = replay-selected cut)
  burst table3 [--model M]                                  Tables 3/6: tested executable cuts around the selection
  burst ladder                                              Table 4 rungs N1-N3, G1-G3 (pi0)
  burst sweep --dir GN|NG --cuts LIST [--model M]           Fig. 6 points; LIST = p values / ranges a-b / fl<j> (NG)
  kcut [--model M] [--kmax 9] [--plans FILE]                multi-cut table: DP plans + measured runs
  sustain --mode duo|npu --period T [--dur S] [--model M] [--cut ...] [--gpu-cap MHz]
                                                            fixed-period run for S seconds (300). --gpu-cap needs root
  common: --soc sd8elite|sd8gen5 (default: from the phone), --cfg default|noxmem|xmem (default: default on
          sd8elite, noxmem on sd8gen5. xmem: Adreno 840 ViT kernel), --runs N (3), --heads hf16|q4 (default hf16)

internal
  _build                             build steps, run inside the container by `build`
  _pull SOC BATCH                    resume polling a running batch and pull it

environment: VLADUO_WORK VLADUO_OUT VLADUO_MODELS VLADUO_RESULTS (default <release>/{work,out,models,results}),
  DEV_ROOT (/data/local/tmp/vladuo), ANDROID_SERIAL, LLAMA_CPP_GIT (URL, or a clone under VLADUO_WORK),
  HF_LOCAL_PI0 HF_LOCAL_PI05 (local model.safetensors), VLADUO_IMAGE (vladuo-env:1), TH_C (28), ZONES,
  VLADUO_SCRUB_EXTRA (file of extra private scrub patterns), PYTHON (python3)
EOF
}

die() { echo "vladuo.sh: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found"; }
dry() { [ -n "$DRYRUN" ]; }
q() { printf ' %q' "$@"; }
adb_() { if dry; then echo "+ adb$(q "$@")"; else adb "$@"; fi; }
ash() { if dry; then echo "+ adb shell '$1'"; else timeout "${ASH_TMO:-120}" adb shell "$1" </dev/null | tr -d '\r'; fi; }

# ---------------------------------------------------------------------------------------------------------------------
# build steps inside the container
# ---------------------------------------------------------------------------------------------------------------------
# shellcheck disable=SC1083,SC2086
_build() {
set -euo pipefail
B8414=5744d7ec430e2f875a393770195fda530560773f; HEX=178a6c44937154dc4c4eff0d166f4a044c4fceba
UP=${LLAMA_CPP_GIT:-https://github.com/ggml-org/llama.cpp}; S=/work/llama.cpp
[ -d $S/.git ] || git clone --filter=blob:none --no-checkout "$UP" $S
for c in $B8414 $HEX; do git -C $S cat-file -e $c^{commit} 2>/dev/null || git -C $S fetch --filter=blob:none origin $c; done
git -C $S checkout -f --detach $B8414 && git -C $S clean -fdxq
rm -rf $S/ggml/src/ggml-hexagon && git -C $S checkout $HEX -- ggml/src/ggml-hexagon
git -C $S apply --whitespace=nowarn /rel/patches/vla-duo.patch
cp -r /rel/src/. $S/ && cp $S/docs/backend/snapdragon/CMakeUserPresets.json $S/
cd $S
cmake --preset arm64-android-snapdragon-release -B /work/build-android -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /work/build-android --target llama-pi0-bench llama-pi0-pipeline llama-pi0-opmat-bench htp-v79 htp-v81 -j"$(nproc)"
cmake -S . -B /work/build-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
      -DGGML_OPENCL=OFF -DGGML_HEXAGON=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /work/build-host --target llama-quantize -j"$(nproc)"
rm -rf /out/* && mkdir -p /out/android/bin /out/android/lib /out/host
cp /work/build-android/bin/llama-pi0-{bench,pipeline,opmat-bench} /out/android/bin/
cp /work/build-android/bin/lib{ggml,ggml-base,ggml-cpu,ggml-opencl,ggml-hexagon,llama,mtmd}.so /out/android/lib/
cp /work/build-android/ggml/src/ggml-hexagon/libggml-htp-v{79,81}.so /out/android/lib/
cp /work/build-host/bin/llama-quantize /out/host/ && cp -r gguf-py /out/gguf-py
{ echo "llama.cpp $B8414 (master, build number 8414); ggml-hexagon from $HEX"; md5sum /rel/patches/vla-duo.patch
  (cd /out && md5sum android/bin/* android/lib/* host/*); } > /out/BUILD_INFO.txt
}

# ---------------------------------------------------------------------------------------------------------------------
# container runs
# ---------------------------------------------------------------------------------------------------------------------
ENVX=(); VOLX=()
ensure_image() {
	if dry; then echo "+ docker image inspect $IMAGE >/dev/null || docker build -t $IMAGE - < Dockerfile"; return; fi
	need docker
	docker image inspect "$IMAGE" >/dev/null 2>&1 && return
	echo "building $IMAGE from Dockerfile (no build context)"
	docker build -t "$IMAGE" - <"$REL/Dockerfile"
}
crun() {   # docker run with the release mounts. ENVX/VOLX add -e and -v arguments
	local w o m
	w=$(realpath -m "$WORK"); o=$(realpath -m "$OUT"); m=$(realpath -m "$MODELS")
	dry || mkdir -p "$w" "$o" "$m"
	local cmd=(docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp ${ENVX[@]+"${ENVX[@]}"} -v "$REL":/rel:ro -v "$w":/work
		-v "$o":/out -v "$m":/models ${VOLX[@]+"${VOLX[@]}"} -w /rel "$IMAGE" "$@")
	if dry; then echo "+$(q "${cmd[@]}")"; else "${cmd[@]}"; fi
}

cmd_build() {
	local g=${LLAMA_CPP_GIT:-} a w
	if [ -n "$g" ] && [ -d "$g" ]; then   # a local clone is visible in the container only under /work
		a=$(cd "$g" && pwd -P); w=$(realpath -m "$WORK")
		case "$a/" in "$w"/*) g=/work/${a#"$w"/} ;; *) die "LLAMA_CPP_GIT=$g: a local clone must be under VLADUO_WORK ($w)" ;; esac
	fi
	[ -n "$g" ] && ENVX=(-e "LLAMA_CPP_GIT=$g")
	ensure_image
	crun bash /rel/vladuo.sh _build
	dry || { cat "$OUT/BUILD_INFO.txt"; echo "build: done (out = $OUT)"; }
}

cmd_models() {
	local f
	if ! dry && { [ ! -x "$OUT/host/llama-quantize" ] || [ ! -d "$OUT/gguf-py" ]; }; then die "run ./vladuo.sh build first ($OUT)"; fi
	ENVX=(-e PYTHONPATH=/out/gguf-py)
	for f in "${HF_LOCAL_PI0:-}:pi0" "${HF_LOCAL_PI05:-}:pi05"; do
		[ -n "${f%:*}" ] || continue
		[ -f "${f%:*}" ] || die "${f%:*}: not a file"
		VOLX+=(-v "$(realpath "${f%:*}")":/hf/"${f##*:}".safetensors:ro)
	done
	ensure_image
	crun python tools/gguf_tools.py prepare --models /models --heads "$HEADS" --model "$MODEL"
	dry && return
	(cd "$MODELS" && md5sum -c --quiet --ignore-missing "$REL/data/models.md5") || die "md5 mismatch (see data/models.md5)"
	echo "models: md5 OK ($MODELS/sets)"
}

# ---------------------------------------------------------------------------------------------------------------------
# device: soc, env presets, command templates
# ---------------------------------------------------------------------------------------------------------------------
device_ready() {
	dry && return
	need adb
	[ "$(adb get-state 2>/dev/null)" = device ] || die "no phone in adb (set ANDROID_SERIAL if several are attached)"
}
resolve_soc() {
	local m=""
	if dry; then
		[ -n "$SOC" ] || { SOC=sd8elite; echo "(dry run: --soc not given, assuming sd8elite)" >&2; }
	else
		device_ready
		m=$(adb shell getprop ro.soc.model </dev/null | tr -d '\r')
		case $m in SM8750*) m=sd8elite ;; SM8850*) m=sd8gen5 ;; *) m="" ;; esac
		[ -n "$SOC" ] || SOC=$m
		[ -n "$SOC" ] || die "unsupported SoC (pass --soc)"
		[ -z "$m" ] || [ "$m" = "$SOC" ] || die "--soc $SOC but the phone is $m"
	fi
	case $SOC in sd8elite | sd8gen5) ;; *) die "--soc sd8elite|sd8gen5" ;; esac
	[ -n "$CFG" ] || { [ "$SOC" = sd8gen5 ] && CFG=noxmem || CFG=default; }
	case $SOC/$CFG in sd8elite/default | sd8gen5/xmem | sd8gen5/noxmem) ;; *) die "--cfg $CFG is not a kernel config of $SOC" ;; esac
	local xe=""; [ "$CFG" = xmem ] && xe=PI0_VIT_XMEM=2
	XE=$xe
	LDENV="LD_LIBRARY_PATH=lib:/vendor/lib64 LD_PRELOAD=/system/lib64/libc++.so:/system/lib64/libbinder.so OCL_ICD_FILENAMES=/vendor/lib64/libOpenCL_adreno.so ADSP_LIBRARY_PATH=$DEV_ROOT/lib"
	ENV_NPU="MTMD_BACKEND_DEVICE=HTP0 PI0_VIT_HD_PAD=2 PI0_FFN_F16=1 GGML_HEXAGON_OPPOLL=1 PI0_VIT_PATCH_FAST=1 PI0_VIT_BATCH=1 GGML_HEXAGON_FA_ONEBLOCK=1"
	local duo="PI0_VIT_HD_PAD=2 PI0_FFN_F16=1 GGML_HEXAGON_OPPOLL=1 PI0_VIT_IMG=1 PI0_VIT_WEIGHT_PRETRANS=1 PI0_VIT_BATCH=1 GGML_HEXAGON_FA_ONEBLOCK=1 PI0_VIT_2D=1"
	ENV_DUO="$duo${xe:+ $xe}"
	ENV_GPU="PI0_VIT_IMG=1 PI0_VIT_WEIGHT_PRETRANS=1 PI0_VIT_2D=1 PI0_ATTN_IMG=1 PI0_ATTN_COLLAPSE=1 PI0_SPLITK=1 PI0_NO_FLASH_ATTN=1"
	ENV_GPULM="PI0_ATTN_IMG=1 PI0_ATTN_COLLAPSE=1 PI0_SPLITK=1"
	ENV_OPMAT="$duo PI0_VIT_PATCH_FAST=1 $ENV_GPULM${xe:+ $xe}"
	HS=""; [ "$HEADS" != q4 ] || HS=-q4h
}
run_heads() { [ "$HEADS" != both ] || die "--heads both is for push and models; device runs take --heads hf16|q4"; }

bench_cmd() {   # ENV BACKEND SET WARMUP N [FLAGS]
	echo "env $LDENV $1 ./bin/llama-pi0-bench --backend $2 --kv-type f16 --warmup $4 -n $5${6:+ $6} -m gguf/$3 --image $IMG -p \"$PROMPT\""
}
pipe_cmd() {    # ENV SET ARGS
	echo "env $LDENV $1 ./bin/llama-pi0-pipeline --backend HTP0 --kv-type f16 --warmup 3 --measure 20${3:+ $3} -m gguf/$2 --image $IMG -p \"$PROMPT\""
}
duo_cmd() {     # MODEL CUT TAG
	local a tr="PI0_PIPE_TRACE=runs/$B/$3.csv"
	a=$(cut_args "$2")
	case $a in
	*--vit-split*--direction* | *--direction*--vit-split*)   # N->G at a SigLIP layer: the LM and expert run on OpenCL from -m
		pipe_cmd "$ENV_DUO $ENV_GPULM PI0_NO_FLASH_ATTN=1 $tr" "$1-gpu$HS" "$a" ;;
	--direction*) pipe_cmd "$ENV_DUO $ENV_GPULM $tr" "$1-npu$HS" "--direction NG --gpu-model-dir gguf/$1-gpu$HS ${a#--direction NG }" ;;
	*) pipe_cmd "$ENV_DUO $tr" "$1-npu$HS" "$a" ;;
	esac
}
kcut_cmd() {    # MODEL TAG PLAN ORDER GPU_LM
	local e=$ENV_DUO g=""
	[ "$5" = 1 ] && { e="$e $ENV_GPULM"; g=" --gpu-model-dir gguf/$1-gpu$HS"; }
	echo "env $LDENV $e PI0_PIPE_TRACE=runs/$B/$2.csv PI0_ACTIONS_OUT=runs/$B/$2.bin ./bin/llama-pi0-pipeline --backend HTP0 --kv-type f16 --warmup 3 --measure 20 --plan \"$3\" --plan-order \"$4\"$g -m gguf/$1-npu$HS --image $IMG -p \"$PROMPT\""
}
sustain_cmd() {  # MODEL CUT TAG PERIOD DUR MODE: fixed-period admission for DUR s, stop at latency 2T (duo) / T (npu)
	local c lim=$4 m=$((($5 * 1000 + $4 - 1) / $4))
	[ "$6" = duo ] && lim=$((2 * $4))
	c=$(duo_cmd "$1" "$2" "$3")
	echo "${c/--warmup 3 --measure 20/--warmup 1 --measure $m --period $4 --abort-latency $lim}"
}
ladder_cmd() {  # RUNG SET
	case $1 in
	N1) bench_cmd "MTMD_BACKEND_DEVICE=HTP0 PI0_LM_NOCONT=0 PI0_PREFIX_GRAPH_CACHE=0 PI0_HTP_LNFUSE=0 PI0_LM_ONEROPE=0 PI0_FFN_FOLD_RESIDUAL=0 PI0_FFN_GU=0" HTP0 "$2" 3 20 ;;
	N2) bench_cmd "MTMD_BACKEND_DEVICE=HTP0 PI0_VIT_PATCH_FAST=1 PI0_VIT_HD_PAD=1 GGML_HEXAGON_OPPOLL=1 PI0_PREFIX_GRAPH_CACHE=1 PI0_LM_NOCONT=1 PI0_LM_ONEROPE=1 PI0_FFN_F16=1 PI0_HTP_LNFUSE=0 PI0_FFN_FOLD_RESIDUAL=0 PI0_FFN_GU=0" HTP0 "$2" 3 20 ;;
	N3) bench_cmd "$ENV_NPU" HTP0 "$2" 3 20 ;;
	G1) bench_cmd "PI0_NO_FLASH_ATTN=1" OpenCL "$2" 3 20 "-fa off" ;;
	G2) bench_cmd "PI0_NO_FLASH_ATTN=1 PI0_ATTN_COLLAPSE=1 PI0_VIT_WEIGHT_PRETRANS=1" OpenCL "$2" 3 20 ;;
	G3) bench_cmd "$ENV_GPU" OpenCL "$2" 3 20 ;;
	esac
}
opmat_cmd() {   # SET MODEL WHICH ROWS COLS [--small-bg]
	local m=""; [ "$1" != full ] || m=" --model $2"
	echo "env $LDENV $ENV_OPMAT ./bin/llama-pi0-opmat-bench --set $1$m --iters 12 --rep-ms 8 --which $3 --rows $4 --cols $5${6:+ $6}"
}

# cut labels (tools/vladuo.py): GNp<p> (G->N, p = 2..29), NGp<p> (N->G, p = 2..28, 30..56), NGfl<j> (inside step 0)
cut_args() {
	local p
	case $1 in
	GNp29) echo "" ;;
	GNp[0-9]*) p=${1#GNp}; ((p >= 2 && p <= 28)) || die "not an executable G->N cut: $1"; echo "--vit-split $((p - 1))" ;;
	NGfl[0-9]*) p=${1#NGfl}; ((p >= 1 && p <= 17)) || die "not an executable cut: $1"; echo "--direction NG --fm-layer-split $p" ;;
	NGp[0-9]*)
		p=${1#NGp}
		if ((p >= 2 && p <= 28)); then echo "--direction NG --vit-split $((p - 1))"
		elif ((p >= 30 && p <= 46)); then echo "--direction NG --lm-split $((p - 29))"
		elif ((p >= 47 && p <= 56)); then echo "--direction NG --fm-split $((p - 47))"
		else die "not an executable N->G cut: $1"; fi ;;
	*) die "unknown cut $1 (GNp<p>, NGp<p>, NGfl<j>)" ;;
	esac
}
args_label() {  # "--direction NG --lm-split 17" -> NGp46
	local d=GN v="" l="" f="" j=""
	# shellcheck disable=SC2086
	set -- $1
	while (($#)); do
		case $1 in
		--direction) d=${2:?}; shift ;; --vit-split) v=${2:?}; shift ;; --lm-split) l=${2:?}; shift ;;
		--fm-split) f=${2:?}; shift ;; --fm-layer-split) j=${2:?}; shift ;; *) die "unsupported cut argument $1" ;;
		esac
		shift
	done
	if [ -n "$l" ]; then echo "NGp$((29 + l))"; elif [ -n "$f" ]; then echo "NGp$((47 + f))"
	elif [ -n "$j" ]; then echo "NGfl$j"; elif [ -z "$v" ]; then echo "${d}p29"; else echo "${d}p$((v + 1))"; fi
}

# ---------------------------------------------------------------------------------------------------------------------
# batches: jobs.tsv (tag run kind model set cut cfg soc cmd) + the device runner vd_run.sh
# ---------------------------------------------------------------------------------------------------------------------
TMPD=""
mk_tmpd() { [ -n "$TMPD" ] || { TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT; }; }   # not inside $(...)
batch_init() {
	B="$1-$(date +%Y%m%d-%H%M%S)"; RD="$RESULTS/$SOC/$B"
	mk_tmpd
	if dry; then JOBS=$TMPD/jobs.tsv; else mkdir -p "$RD"; JOBS=$RD/jobs.tsv; fi
	CONF=$TMPD/conf
	printf 'tag\trun\tkind\tmodel\tset\tcut\tcfg\tsoc\tcmd\n' >"$JOBS"
}
addjob() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$JOBS"; }
conf() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:--}" "${6:--}" "${7:--}" >>"$CONF"; }
mkcmd() {   # KIND MODEL SET CUT TAG X1 X2 X3
	case $1 in
	npu) bench_cmd "$ENV_NPU" HTP0 "$3" 3 20 ;;
	gpu) bench_cmd "$ENV_GPU" OpenCL "$3" 3 20 ;;
	duo | sweep) duo_cmd "$2" "$4" "$5" ;;
	kcut) kcut_cmd "$2" "$5" "$6" "$7" "$8" ;;
	sustain-*) sustain_cmd "$2" "$4" "$5" "$6" "$7" "${1#sustain-}" ;;
	ladder-*) ladder_cmd "${1#ladder-}" "$3" ;;
	*) die "internal: kind $1" ;;
	esac
}
add_runs() {   # every config of $CONF once per run, odd runs in order, even runs reversed
	local r kind model set cut x1 x2 x3 tag cmd
	for ((r = 1; r <= RUNS; r++)); do
		if ((r % 2)); then cat "$CONF"; else tac "$CONF"; fi >"$CONF.ord"
		while IFS=$'\t' read -r kind model set cut x1 x2 x3; do
			tag=${kind}_${model}_${cut}_${CFG}_r$r
			cmd=$(mkcmd "$kind" "$model" "$set" "$cut" "$tag" "$x1" "$x2" "$x3")
			addjob "$tag" "$r" "$kind" "$model" "$set" "$cut" "$CFG" "$SOC" "$cmd"
		done <"$CONF.ord"
	done
	rm -f "$CONF.ord"
}

runner_text() {
	cat <<'EOF'
#!/system/bin/sh
# Device runner written by vladuo.sh. From the device root: sh vd_run.sh runs/<batch>/jobs.tsv
# jobs.tsv columns: tag run kind model set cut cfg soc cmd. Per job: <tag>.log (+ .csv trace/profile, .bin actions)
# and a meta.tsv row (exit, timeout, uptime, die temperatures, battery level/status). Writes runner.log, DONE at the end.
# Standard job: cool below TH, idle, wake, one env layer, 900 s cap, kill by PID.
# Invalid runs (exit, timeout, no RESULT, drain_free=0, discharging, battery level drop, rms != first valid run of the
# config) are re-run once at the end as tag + "b". quick-*/full-* jobs follow the lighter profiling rules.
trap "" TERM HUP
case $0 in */*) cd "${0%/*}" || exit 1 ;; esac
J=$1; D=${J%/*}; LOG=$D/runner.log; META=$D/meta.tsv; RR=$D/rerun.tsv; T=$(printf '\t')
TH=$((${TH_C:-28} * 1000)); IDLE=${IDLE:-120}; WAITMAX=${WAITMAX:-1800}
log() { echo "$(date +%T) $*" >> "$LOG"; }
up() { cut -d. -f1 /proc/uptime; }
ZP=""
for t in ${ZONES:-nsphmx-0 gpuss-0}; do
  for z in /sys/class/thermal/thermal_zone*; do [ "$(cat "$z/type" 2>/dev/null)" = "$t" ] && ZP="$ZP $z/temp"; done
done
temp() { m=0; for p in $ZP; do v=$(cat "$p" 2>/dev/null); [ "${v:-0}" -gt "$m" ] 2>/dev/null && m=$v; done; echo "$m"; }
bat() { dumpsys battery | awk '$1 == "level:" { l = $2 } $1 == "status:" { s = $2 } END { print l + 0, s + 0 }'; }
busy() { [ -n "$(pidof llama-pi0-bench llama-pi0-pipeline llama-pi0-opmat-bench 2>/dev/null)" ]; }
cool() {
  n=0; while [ "$(temp)" -ge "$TH" ] && [ "$n" -lt "$WAITMAX" ]; do sleep 10; n=$((n + 10)); done
  if [ "$n" -ge "$WAITMAX" ]; then log "cooling cap ${WAITMAX} s reached at $(temp)"; fi
}
wake() { input keyevent KEYCODE_WAKEUP; sleep 2; }
guard() {
  n=0
  while busy; do
    [ "$n" = 0 ] && log "another llama-pi0 process is running; waiting"
    [ "$n" -ge 3600 ] && { note="${note:+$note,}busy"; return 1; }
    sleep 20; n=$((n + 20))
  done
}
run() {   # TAG TIMEOUT_S EXT: $cmd -> $D/TAG.EXT, sets EX and TO
  rm -f htp_profiling.csv
  sh -c "exec $cmd" < /dev/null > "$D/$1.$3" 2>&1 &
  P=$!; n=0; TO=0
  while kill -0 "$P" 2>/dev/null; do
    sleep 2; n=$((n + 2))
    if [ "$n" -ge "$2" ]; then TO=1; kill -9 "$P" 2>/dev/null; log "$1: timeout after $2 s, killed pid $P"; break; fi
  done
  wait "$P"; EX=$?
  if [ "$TO" = 1 ]; then n=0; while busy && [ "$n" -lt 60 ]; do sleep 2; n=$((n + 2)); done; fi
  case $cmd in *GGML_HEXAGON_PROFILE=1*) [ -f htp_profiling.csv ] && mv htp_profiling.csv "$D/$1.csv" ;; esac
}
pre() { b=$(bat); lv0=${b% *}; st0=${b#* }; t0=$(temp); s0=$(up); }
post() { b=$(bat); lv1=${b% *}; st1=${b#* }; }
meta() {
  [ -f "$META" ] || printf 'tag\texit\ttimeout\tstart_s\tend_s\tdie0\tdie1\tlevel0\tlevel1\tstatus0\tstatus1\tnote\n' > "$META"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$EX" "$TO" "$s0" "$(up)" "$t0" "$(temp)" \
    "$lv0" "$lv1" "$st0" "$st1" "${note:--}" >> "$META"
}
valid() {   # validity rules, sets why
  why=""
  [ "$EX" = 0 ] || why="$why exit=$EX"
  [ "$TO" = 0 ] || why="$why timeout"
  res=$(grep '^RESULT ' "$D/$tag.log" | tail -n 1)
  [ -n "$res" ] || why="$why no-RESULT"
  case " $res " in *" drain_free=0 "*) why="$why drain" ;; esac
  [ "$st1" = 3 ] && why="$why discharging"
  [ "$lv1" -lt "$lv0" ] && why="$why battery-level"
  rms=$(echo "$res" | sed -n 's/.* rms=\([^ ]*\).*/\1/p'); key="$kind|$model|$set|$cut|$cfg"
  if [ -z "$why" ] && [ -n "$rms" ]; then
    ref=$(awk -v k="$key" '$1 == k { print $2; exit }' "$D/rms.ref" 2>/dev/null)
    if [ -z "$ref" ]; then echo "$key $rms" >> "$D/rms.ref"
    elif awk -v a="$rms" -v b="$ref" 'BEGIN { d = (a - b) / b; if (d < 0) d = -d; exit !(d > 1e-4) }'; then why="$why rms=$rms/$ref"; fi
  fi
}
std() {
  note=""; cool; sleep "$IDLE"; wake; guard
  pre; run "$tag" 900 log; post; valid
  [ -n "$why" ] && note="${note:+$note,}invalid:${why# }"
  meta
  if [ -n "$why" ] && [ -z "$RERUN" ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}b" "$run" "$kind" "$model" "$set" "$cut" "$cfg" "$soc" "$cmd" >> "$RR"
  fi
}
opm() {   # QUICK opmat process: cool once before the first, 180 s cap, 8 s pause
  note=""; [ -n "$OPM" ] || { cool; OPM=1; }; guard
  pre; run "$tag" 180 log; post; meta; sleep 8
}
cal() {   # QUICK calibration leg: cool + idle once before the block, then cool + wake per leg
  note=""; cool; [ -n "$CAL" ] || { sleep "$IDLE"; CAL=1; }; wake; guard
  pre; run "$tag" 900 log; post; meta
}
full() {  # FULL process: guard, 240 s cap, one retry after a timeout or a foreign process, 8 s pause
  note=""; guard; pre; run "$tag" 240 log
  if [ "$TO" = 1 ] || busy; then
    [ "$TO" = 1 ] && note=timeout || note=busy-after
    mv "$D/$tag.log" "$D/$tag.log.1"; sleep 30; guard; run "$tag" 240 log; note="$note,retried"
  fi
  post; meta; sleep 8
}
KG=/sys/class/kgsl/kgsl-3d0
gcap() {   # GPU_CAP (MHz) -> the highest GPU power level at or below it, needs root. The old level goes to runs/.gpucap
  f=$(su -c "cat $KG/gpu_available_frequencies" 2>/dev/null) || return 1
  lv=$(echo "$f" | tr ' ' '\n' | awk -v c="$GPU_CAP" 'NF { if (p == "" && $1 <= c * 1000000) p = n; n++ } END { print p }')
  [ -n "$lv" ] || return 1
  [ -f runs/.gpucap ] || su -c "cat $KG/max_pwrlevel" > runs/.gpucap || return 1
  su -c "echo $lv > $KG/max_pwrlevel" || return 1
  mx=$(su -c "cat $KG/max_gpuclk" 2>/dev/null)
  [ "${mx:-0}" -le $((GPU_CAP * 1000000)) ] 2>/dev/null || { grestore; return 1; }
  log "gpu cap $GPU_CAP MHz: power level $lv, max clock $mx Hz"
}
grestore() { [ -f runs/.gpucap ] || return 0; su -c "echo $(cat runs/.gpucap) > $KG/max_pwrlevel" && rm -f runs/.gpucap; }
sus() {   # sustained run: cool, idle, optional GPU cap, die temperature every second in <tag>.temp, 3900 s cap
  note=""; cool; sleep "$IDLE"; wake; guard
  if [ -n "$GPU_CAP" ] && ! gcap; then note=gpu-cap-failed; EX=1; TO=0; pre; post; meta; return; fi
  ( while :; do echo "$(up) $(temp)"; sleep 1; done ) > "$D/$tag.temp" 2>/dev/null &
  SP=$!
  pre; run "$tag" 3900 log; post
  kill -9 "$SP" 2>/dev/null
  grestore
  meta
}
canary() {  # cool, then the canary. Its NPU idle rate must stay within 8 % of the first canary
  note=""; cool; guard; pre; run "$tag" 240 txt; post
  v=$(awk '/^=== h_ij/ { h = 1; next } h && /^fg / { getline; print $2; exit }' "$D/$tag.txt")
  case $v in '' | *[!0-9.]*) v=0 ;; esac
  bad=""
  if [ -z "$CREF" ]; then [ "$v" != 0 ] && CREF=$v; note="canary $v (reference)"
  elif awk -v a="$v" -v b="$CREF" 'BEGIN { d = (a - b) / b; if (d < 0) d = -d; exit !(d > 0.08) }'; then
    bad=1; note="canary $v vs $CREF: re-running the processes since the last good canary"
  else note="canary $v ok"; fi
  meta
  if [ -n "$bad" ] && [ -s "$D/since.tsv" ] && [ -z "$RERUN" ]; then
    ctag=$tag; ccmd=$cmd; RERUN=1; mv "$D/since.tsv" "$D/since.run"
    while IFS="$T" read -r tag run kind model set cut cfg soc cmd <&4; do
      log "$tag re-run"; mv "$D/$tag.log" "$D/$tag.log.stale" 2>/dev/null; full
    done 4< "$D/since.run"
    rm -f "$D/since.run"; tag=${ctag}b; cmd=$ccmd; canary; RERUN=""; tag=$ctag
  fi
  : > "$D/since.tsv"
}
one() {
  log "$tag start"
  case $kind in
    quick-C*) cal ;;
    quick-*) opm ;;
    full-canary*) canary ;;
    sustain-*) sus ;;
    full-*) full; [ -n "$RERUN" ] || echo "$tag$T$run$T$kind$T$model$T$set$T$cut$T$cfg$T$soc$T$cmd" >> "$D/since.tsv" ;;
    *) std ;;
  esac
  log "$tag end exit=$EX timeout=$TO${note:+ $note}"
}
LK=runs/.lock   # one batch at a time: a second runner would interleave its jobs with this one's cool/idle gaps
if ! mkdir "$LK" 2>/dev/null; then
  op=$(cat "$LK/pid" 2>/dev/null)
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ABORT: runner pid $op ($(cat "$LK/batch" 2>/dev/null)) is still running"; echo aborted > "$D/DONE"; exit 1; fi
fi
echo $$ > "$LK/pid"; echo "$D" > "$LK/batch"
if [ -f runs/.gpucap ]; then grestore; log "restored the GPU power level left by an interrupted batch"; fi
b=$(bat)
log "batch $D: soc=$(getprop ro.soc.model) high_performance_mode_on=$(settings get system high_performance_mode_on 2>/dev/null)/$(settings get global high_performance_mode_on 2>/dev/null) battery(level status)=$b die=$(temp) zones=${ZP:-none} TH=$TH idle=${IDLE}s"
if ! dumpsys battery | grep -qE '(AC|USB) powered: true'; then log "ABORT: not powered (connect the charger)"; echo aborted > "$D/DONE"; rm -rf "$LK"; exit 1; fi
[ -n "$ZP" ] || log "warning: no thermal zone found (set ZONES); cooling waits are skipped"
svc power stayon true; wake
note=""; guard || { log "ABORT: another llama-pi0 process keeps running"; echo aborted > "$D/DONE"; rm -rf "$LK"; exit 1; }
RERUN=""; OPM=""; CAL=""; CREF=""; : > "$D/since.tsv"; grep -v '^tag' "$J" > "$D/queue.tsv"
while IFS="$T" read -r tag run kind model set cut cfg soc cmd <&3; do
  case $tag in '' | '#'*) continue ;; esac
  one
done 3< "$D/queue.tsv"
if [ -s "$RR" ]; then
  log "re-running $(wc -l < "$RR") invalid run(s) once"; cat "$RR" >> "$J"; RERUN=1
  while IFS="$T" read -r tag run kind model set cut cfg soc cmd <&3; do one; done 3< "$RR"
fi
rm -f "$D/queue.tsv" "$D/since.tsv"
svc power stayon false
log "batch end: battery(level status)=$(bat) die=$(temp)"
echo ok > "$D/DONE"
rm -rf "$LK"
EOF
}

no_batch() {   # one batch at a time. push must not replace files under a running batch either
	local r
	r=$(ash "ps -A -o PID,ARGS | grep '[v]d_run.sh' || true")
	[ -z "$r" ] || die "a batch is running on the phone:
$r
wait for it (./vladuo.sh _pull $SOC <batch>) or stop the runner and the binary with adb shell kill -9 <pid>"
}
batch_run() {
	local n
	n=$(($(wc -l <"$JOBS") - 1))
	rm -f "$CONF"
	# only the runner goes to the background, with all three fds redirected: a backgrounded `cd && ...` list would be a
	# subshell holding adb's output pipe, and `adb shell` would not return until the batch ends
	local start="cd $DEV_ROOT && { ${TH_C:+TH_C=$TH_C }${ZONES:+ZONES=\"$ZONES\" }${GPU_CAP:+GPU_CAP=$GPU_CAP }setsid nohup sh vd_run.sh runs/$B/jobs.tsv > /dev/null 2>&1 < /dev/null & }"
	if dry; then
		echo "==> jobs.tsv: batch $B, $n jobs (device $DEV_ROOT/runs/$B, host $RD) <=="
		cat "$JOBS"
		echo "==> vd_run.sh <=="
		runner_text
		echo "==> adb <=="
		ash "[ -x $DEV_ROOT/bin/llama-pi0-pipeline ] && echo ok"
		ash "dumpsys battery"
		ash "mkdir -p $DEV_ROOT/runs/$B"
		adb_ push "$WORK/vd_run.sh" "$DEV_ROOT/vd_run.sh"
		adb_ push "$RD/jobs.tsv" "$DEV_ROOT/runs/$B/jobs.tsv"
		ash "$start"
		echo "+ every 30 s: adb shell 'cat $DEV_ROOT/runs/$B/DONE; tail -n 1 $DEV_ROOT/runs/$B/runner.log'"
		adb_ pull "$DEV_ROOT/runs/$B" "$RESULTS/$SOC/"
		rm -f "$JOBS"
		return
	fi
	mkdir -p "$WORK"
	runner_text >"$WORK/vd_run.sh"
	[ "$(ash "[ -x $DEV_ROOT/bin/llama-pi0-pipeline ] && echo ok")" = ok ] || die "nothing pushed to $DEV_ROOT (./vladuo.sh push)"
	n=$(ash "dumpsys battery")
	grep -qE '(AC|USB) powered: true' <<<"$n" || die "the phone is not powered (connect the charger)"
	no_batch
	n=$(($(wc -l <"$JOBS") - 1))
	ash "mkdir -p $DEV_ROOT/runs/$B" >/dev/null || die "adb shell failed (phone reachable?)"
	adb push "$WORK/vd_run.sh" "$DEV_ROOT/vd_run.sh" >/dev/null
	adb push "$JOBS" "$DEV_ROOT/runs/$B/jobs.tsv" >/dev/null
	ASH_TMO=30 ash "$start" >/dev/null || die "could not start the runner on the phone"
	echo "batch $B: $n jobs started on the phone; results -> $RD"
	wait_pull "$SOC" "$B"
}
wait_pull() {   # SOC BATCH: poll DONE every 30 s, then pull. The batch keeps running if adb drops
	local st
	if dry; then
		echo "+ every 30 s: adb shell 'cd $DEV_ROOT/runs/$2 && cat DONE; tail -n 1 runner.log' until DONE"
		adb_ pull "$DEV_ROOT/runs/$2" "$RESULTS/$1/"
		return
	fi
	echo "polling every 30 s; Ctrl-C is safe (the batch keeps running): resume with ./vladuo.sh _pull $1 $2"
	while :; do
		sleep 30
		st=$(timeout 60 adb shell "cd $DEV_ROOT/runs/$2 && cat DONE 2>/dev/null; tail -n 1 runner.log" </dev/null 2>/dev/null | tr -d '\r') || continue
		[ -n "$st" ] && printf '  %s\n' "${st##*$'\n'}"
		case $st in ok* | aborted*) break ;; esac
	done
	mkdir -p "$RESULTS/$1"
	adb pull "$DEV_ROOT/runs/$2" "$RESULTS/$1/" >/dev/null
	case $st in aborted*) die "batch aborted on the phone: see $RESULTS/$1/$2/runner.log" ;; esac
	echo "pulled $RESULTS/$1/$2 ($(awk -F'\t' 'NR > 1 && $2 != "0" { n++ } END { print n + 0 }' "$RESULTS/$1/$2/meta.tsv") non-zero exits)"
}

# ---------------------------------------------------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------------------------------------------------
cmd_push() {
	local sets="" s f src key k have exp got bad=0
	declare -A first
	resolve_soc
	case $HEADS in hf16) sets="pi0-npu pi0-gpu pi0-stock pi05-npu pi05-gpu" ;;
	q4) sets="pi0-npu-q4h pi0-gpu-q4h pi0-stock-q4h pi05-npu-q4h pi05-gpu-q4h" ;;
	both) sets="pi0-npu pi0-gpu pi0-stock pi05-npu pi05-gpu pi0-npu-q4h pi0-gpu-q4h pi0-stock-q4h pi05-npu-q4h pi05-gpu-q4h" ;; esac
	[ "$MODEL" = all ] || sets=$(for s in $sets; do case $s in "$MODEL"-*) echo "$s" ;; esac; done | xargs)
	if ! dry; then
		for f in $BINS; do [ -f "$OUT/android/bin/$f" ] || die "run ./vladuo.sh build first"; done
		for s in $sets; do [ -d "$MODELS/sets/$s" ] || die "missing $MODELS/sets/$s (./vladuo.sh models --heads $HEADS --model $MODEL)"; done
	fi
	dry || no_batch
	echo "push to $DEV_ROOT (soc $SOC, sets: $sets)"
	ash "mkdir -p $DEV_ROOT/bin $DEV_ROOT/lib $DEV_ROOT/inputs $DEV_ROOT/runs $(for s in $sets; do printf '%s ' "$DEV_ROOT/gguf/$s"; done)"
	put() {   # LOCAL REMOTE: skip when the phone already has a file of the same size (md5 is verified below)
		if [ -z "$FORCE" ] && ! dry && [ "$(ash "stat -c %s $2 2>/dev/null")" = "$(stat -Lc %s "$1")" ]; then return; fi
		adb_ push "$1" "$2"
	}
	for f in $BINS; do put "$OUT/android/bin/$f" "$DEV_ROOT/bin/$f"; done
	ash "chmod 755 $DEV_ROOT/bin/*"
	for f in $LIBS; do put "$OUT/android/lib/$f" "$DEV_ROOT/lib/$f"; done
	for f in cam_high cam_left_wrist cam_right_wrist; do put "$REL/data/inputs/$f.jpg" "$DEV_ROOT/inputs/$f.jpg"; done
	for s in $sets; do
		for f in $GGUFS; do
			src=$MODELS/sets/$s/$f; key=$s/$f
			[ -e "$src" ] && key=$(stat -Lc %d:%i "$src")   # set members are hard links: push each file once
			if [ -n "${first[$key]:-}" ]; then ash "ln -sf ../${first[$key]} $DEV_ROOT/gguf/$s/$f"
			else first[$key]=$s/$f; put "$src" "$DEV_ROOT/gguf/$s/$f"; fi
		done
	done
	mk_tmpd; runner_text >"$TMPD/vd_run.sh"
	adb_ push "$TMPD/vd_run.sh" "$DEV_ROOT/vd_run.sh"
	if dry; then echo "+ verify: adb shell 'cd $DEV_ROOT && md5sum bin/* lib/* gguf/*/*.gguf' against out/BUILD_INFO.txt and data/models.md5"; return; fi
	echo "verifying md5 on the phone"
	while read -r got f; do
		case $f in bin/* | lib/*) exp=$(awk -v p="android/$f" '$2 == p { print $1 }' "$OUT/BUILD_INFO.txt") ;;
		gguf/*) k=${f#gguf/}; exp=$(awk -v p="sets/$k" '$2 == p { print $1 }' "$REL/data/models.md5"); [ -n "$exp" ] || exp=$(md5sum <"$MODELS/sets/$k" | cut -d' ' -f1) ;;
		*) continue ;; esac
		if [ -z "$exp" ]; then echo "  unexpected file on the phone: $f (remove it)"; bad=1
		elif [ "$got" != "$exp" ]; then echo "  md5 mismatch: $f ($got, expected $exp)"; bad=1; fi
		have=1
	done < <(ASH_TMO=1800 ash "cd $DEV_ROOT && md5sum bin/* lib/* $(for s in $sets; do printf 'gguf/%s/*.gguf ' "$s"; done)")
	[ -n "${have:-}" ] || die "md5sum on the phone failed"
	[ "$bad" = 0 ] || die "push: md5 mismatch (re-run with --force)"
	echo "push: OK (device md5 = host md5)"
}

# ---------------------------------------------------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------------------------------------------------
cmd_profile() {
	local what=${ARGS[0]:-} r c i=0 tag k n cols chunks
	run_heads; resolve_soc
	case $what in
	quick)
		batch_init quick
		for r in 0 1 2 3; do for c in 0:4 6:9 9:11 13:16 16:18; do
			i=$((i + 1)); tag=$(printf 'quick-g%02d' $i)
			addjob "${tag}_pi0_-_${CFG}_r1" 1 "$tag" pi0 - - "$CFG" "$SOC" "$(opmat_cmd quick pi0 g $r $c)"
		done; done
		i=0
		for r in 0:3 3:6 6:9 9:11 13:15 15:18; do
			i=$((i + 1))
			addjob "quick-h${i}_pi0_-_${CFG}_r1" 1 "quick-h$i" pi0 - - "$CFG" "$SOC" "$(opmat_cmd quick pi0 h $r 0:5)"
		done
		calj() { addjob "quick-$1_$2_$4_${CFG}_r1" 1 "quick-$1" "$2" "$3" "$4" "$CFG" "$SOC" "$5"; }
		calj C1 pi0 "pi0-npu$HS" - "$(bench_cmd "$ENV_NPU" HTP0 "pi0-npu$HS" 3 5)"
		calj C2 pi0 "pi0-npu$HS" - "$(bench_cmd "$ENV_NPU GGML_HEXAGON_PROFILE=1" HTP0 "pi0-npu$HS" 1 1)"
		calj C3 pi05 "pi05-npu$HS" - "$(bench_cmd "$ENV_NPU" HTP0 "pi05-npu$HS" 3 5)"
		# C4 hand-off: read from the last, GPU-idle frame of a short run (1 + 3 + 3 drain frames)
		c=$(duo_cmd pi0 GNp28 "quick-C4_pi0_GNp28_${CFG}_r1"); c=${c/--warmup 3 --measure 20/--warmup 1 --measure 3}
		calj C4 pi0 "pi0-npu$HS" GNp28 "$c"
		k=5
		for m in pi0 pi05; do for c in NGp47 NGp46; do
			calj "C$k" $m "$m-npu$HS" $c "$(duo_cmd $m $c "quick-C${k}_${m}_${c}_${CFG}_r1")"; k=$((k + 1))
		done; done
		# C9/C10 give the cut model its GPU layer times, so they use the kernel config of the VLA-Duo runs
		calj C9 pi0 "pi0-gpu$HS" - "$(bench_cmd "$ENV_GPU${XE:+ $XE}" OpenCL "pi0-gpu$HS" 3 5)"
		calj C10 pi05 "pi05-gpu$HS" - "$(bench_cmd "$ENV_GPU${XE:+ $XE}" OpenCL "pi05-gpu$HS" 3 5)"
		batch_run
		dry && return
		"$PY" -B "$REL/tools/vladuo.py" profile "$RD" --soc "$SOC" --cfg "$CFG"
		echo "next: ./vladuo.sh promote $RD/profile.json   (adds key $SOC/$CFG if absent), or replay with --profile $RD/profile.json"
		;;
	full)
		[ "$MODEL" = all ] && MODEL=pi0
		batch_init "full-$MODEL"
		chunks="0,2,3 4,5,6 7,9,10 11,12,13 15,17,18 19"
		n=0
		fadd() {   # KIND ROWS COLS WHICH [--small-bg]. Adds a canary before the first process and after every 10th
			if ((n % 10 == 0)); then
				k=$(printf 'full-canary%02d' $((n / 10)))
				addjob "${k}_${MODEL}_-_${CFG}_r1" 1 "$k" "$MODEL" - - "$CFG" "$SOC" "$(opmat_cmd full "$MODEL" h 12 2)"
			fi
			addjob "$1_${MODEL}_-_${CFG}_r1" 1 "$1" "$MODEL" - - "$CFG" "$SOC" "$(opmat_cmd full "$MODEL" "$4" "$2" "$3" "${5:-}")"
			n=$((n + 1))
		}
		for r in $(seq 0 19); do i=0; for cols in $chunks; do fadd "$(printf 'full-g%02dc%d' "$r" $i)" "$r" "$cols,16" g; i=$((i + 1)); done; done
		for r in $(seq 0 2 18); do fadd "$(printf 'full-h%02dc0' "$r")" "$r:$((r + 2))" 0:10 h; fadd "$(printf 'full-h%02dc1' "$r")" "$r:$((r + 2))" 10:20 h; done
		if [ -n "$SMALLBG" ]; then for r in $(seq 0 19); do fadd "$(printf 'full-s%02d' "$r")" "$r" 1,8,14,16 g --small-bg; done; fi
		k=$(printf 'full-canary%02d' $((n / 10 + (n % 10 > 0))))
		addjob "${k}_${MODEL}_-_${CFG}_r1" 1 "$k" "$MODEL" - - "$CFG" "$SOC" "$(opmat_cmd full "$MODEL" h 12 2)"
		batch_run
		dry && return
		local qd x
		qd=""; for x in "$RESULTS/$SOC"/quick-*/; do [ ! -f "$x/jobs.tsv" ] || qd=$x; done
		if [ -n "$qd" ]; then
			"$PY" -B "$REL/tools/vladuo.py" profile "$qd" --soc "$SOC" --cfg "$CFG" --full "$RD" --out "$RD/profile.json"
		else
			echo "no QUICK run for $SOC yet: after ./vladuo.sh profile quick, run"
			echo "  python3 tools/vladuo.py profile <quick dir> --soc $SOC --cfg $CFG --full $RD --out $RD/profile.json"
		fi
		;;
	*) die "profile quick|full" ;;
	esac
}

# ---------------------------------------------------------------------------------------------------------------------
# bursts and k-cut
# ---------------------------------------------------------------------------------------------------------------------
select_json() {   # MODEL FILE: replay selection on the profile of SOC/CFG
	local o
	if ! o=$("$PY" -B "$REL/tools/vladuo.py" select --soc "$SOC" --cfg "$CFG" --model "$1" --json "$2" 2>&1); then
		echo "$o" >&2; die "no replay selection for $SOC/$CFG: run ./vladuo.sh profile quick --cfg $CFG and promote it, or pass a manual --cut"
	fi
	echo "$o" >&2
	dry || echo "$o" >>"$RD/select.txt"
}
jcuts() { "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(" ".join([d["selected"]["cut"]] if sys.argv[2] == "sel" else [r["cut"] for r in d["tested"]]))' "$1" "$2"; }

cmd_burst() {
	local what=${ARGS[0]:-} m models lab x p tmp
	run_heads; resolve_soc
	models="pi0 pi05"; [ "$MODEL" = all ] || models=$MODEL
	case $what in
	npu | gpu)
		batch_init "burst-$what"
		for m in $models; do conf "$what" "$m" "$m-$what$HS" -; done ;;
	duo | table3)
		batch_init "burst-$what"; tmp=$TMPD/select.json
		for m in $models; do
			if [ "$what" = table3 ] || [ "$CUT" = auto ]; then
				select_json "$m" "$tmp"
				for lab in $(jcuts "$tmp" "$([ "$what" = duo ] && echo sel || echo tested)"); do conf duo "$m" "$m-npu$HS" "$lab"; done
			else
				case $CUT in
				[0-9]*) lab=GNp$CUT ;; GNp* | NGp* | NGfl*) lab=$CUT ;; --*) lab=$(args_label "$CUT") ;; *) die "--cut auto|P|LABEL|\"ARGS\"" ;;
				esac
				cut_args "$lab" >/dev/null
				echo "$m: manual cut $lab (not replay-selected)" >&2
				conf duo "$m" "$m-npu$HS" "$lab"
			fi
		done
		rm -f "$tmp" ;;
	ladder)
		batch_init burst-ladder
		for x in N1:pi0-stock N2:pi0-gpu N3:pi0-npu G1:pi0-stock G2:pi0-gpu G3:pi0-gpu; do conf "ladder-${x%%:*}" pi0 "${x#*:}$HS" -; done ;;
	sweep)
		[ "$DIR" = GN ] || [ "$DIR" = NG ] || die "burst sweep --dir GN|NG --cuts LIST"
		[ -n "$CUTS" ] || die "burst sweep needs --cuts (e.g. 2-28, 30-46, fl3)"
		[ "$DIR" = NG ] || [[ ! ",$CUTS," =~ ,fl ]] || die "fl<j> cuts are N->G only (--dir NG)"
		batch_init burst-sweep
		for m in $models; do
			for x in ${CUTS//,/ }; do
				case $x in
				fl*) lab=NGfl${x#fl}; cut_args "$lab" >/dev/null; conf sweep "$m" "$m-npu$HS" "$lab" ;;
				*-*) for p in $(seq "${x%-*}" "${x#*-}"); do lab=${DIR}p$p; cut_args "$lab" >/dev/null; conf sweep "$m" "$m-npu$HS" "$lab"; done ;;
				*) lab=${DIR}p$x; cut_args "$lab" >/dev/null; conf sweep "$m" "$m-npu$HS" "$lab" ;;
				esac
			done
		done ;;
	*) die "burst npu|gpu|duo|table3|ladder|sweep" ;;
	esac
	add_runs
	batch_run
	dry || echo "next: ./vladuo.sh summary <table> --soc $SOC $RD"
}

cmd_sustain() {
	local m models lab tmp
	run_heads; resolve_soc
	case $MODE in duo | npu) ;; *) die "sustain --mode duo|npu --period T" ;; esac
	[[ $PERIOD =~ ^[1-9][0-9]*$ ]] || die "sustain needs --period T (ms, integer)"
	if ! [[ $DUR =~ ^[1-9][0-9]*$ ]] || ((DUR > 3600)); then die "--dur S (seconds, 1..3600)"; fi
	[ -z "$GPU_CAP" ] || [[ $GPU_CAP =~ ^[1-9][0-9]*$ ]] || die "--gpu-cap MHz (integer)"
	[ -n "$RUNS_OPT" ] || RUNS=1
	models="pi0 pi05"; [ "$MODEL" = all ] || models=$MODEL
	batch_init "sustain-$MODE"; tmp=$TMPD/select.json
	for m in $models; do
		if [ "$MODE" = npu ]; then
			lab=GNp2   # NPU-only, paced: the GPU runs the patch embedding and one SigLIP layer
		elif [ "$CUT" = auto ]; then
			select_json "$m" "$tmp"; lab=$(jcuts "$tmp" sel)
		else
			case $CUT in
			[0-9]*) lab=GNp$CUT ;; GNp* | NGp* | NGfl*) lab=$CUT ;; --*) lab=$(args_label "$CUT") ;; *) die "--cut auto|P|LABEL|\"ARGS\"" ;;
			esac
		fi
		cut_args "$lab" >/dev/null
		conf "sustain-$MODE" "$m" "$m-npu$HS" "$lab" "$PERIOD" "$DUR"
	done
	rm -f "$tmp"
	add_runs
	batch_run
	dry || echo "next: ./vladuo.sh summary sustain --soc $SOC $RD"
}

cmd_kcut() {
	local m k plan order gl
	run_heads; resolve_soc
	batch_init kcut
	if [ -n "$PLANS" ]; then cp "$PLANS" "$CONF.plans"
	else
		echo "k-cut DP ($SOC/$CFG, kmax $KMAX; about 1 min per model)" >&2
		"$PY" -B "$REL/tools/vladuo.py" kcut --soc "$SOC" --cfg "$CFG" --model "$MODEL" --kmax "$KMAX" >"$CONF.plans"
	fi
	dry || cp "$CONF.plans" "$RD/plans.tsv"
	while IFS=$'\t ' read -r m k _ plan order gl; do
		[[ $k =~ ^[0-9]+$ ]] || continue
		[ "$MODEL" = all ] || [ "$MODEL" = "$m" ] || continue
		if [ "${gl:-}" != 0 ] && [ "${gl:-}" != 1 ]; then gl=0; [[ ",$plan" =~ ,G[^,]*[LF] ]] && gl=1; fi   # a GPU segment with LM/FM atoms
		conf kcut "$m" "$m-npu$HS" "k$k" "$plan" "$order" "$gl"
	done <"$CONF.plans"
	rm -f "$CONF.plans"
	add_runs
	batch_run
	dry || echo "next: ./vladuo.sh summary kcut --soc $SOC $RD"
}

# ---------------------------------------------------------------------------------------------------------------------
# replay, promote, summary
# ---------------------------------------------------------------------------------------------------------------------
cmd_promote() {
	local f=${ARGS[0]:-} key=$KEY
	[ -f "$f" ] || die "promote FILE [--key SOC/CFG]"
	[ -n "$key" ] || key=$("$PY" -c 'import json, sys; p = json.load(open(sys.argv[1])); print(p["soc"] + "/" + p["kernel_config"])' "$f")
	"$PY" -B "$REL/tools/vladuo.py" promote "$f" --key "$key"
}

cmd_summary() {
	local what=${ARGS[0]:-} d dirs=() x out
	case $what in burst | table2 | table3 | table4 | table6 | kcut | fig6 | sustain) ;; *) die "summary burst|table2|table3|table4|table6|kcut|fig6|sustain" ;; esac
	for x in "${ARGS[@]:1}"; do dirs+=("$(realpath "$x")"); done
	if [ -z "$SOC" ]; then
		if ((${#dirs[@]})); then SOC=$(basename "$(dirname "${dirs[0]}")"); else for x in sd8elite sd8gen5; do [ -d "$RESULTS/$x" ] && SOC=$x && break; done; fi
	fi
	case $SOC in sd8elite | sd8gen5) ;; *) die "summary: pass --soc sd8elite|sd8gen5" ;; esac
	if ! ((${#dirs[@]})); then
		if [ "$what" = kcut ]; then for d in "$RESULTS/$SOC"/kcut-*/; do [ ! -f "$d/jobs.tsv" ] || dirs=("${d%/}"); done
		elif [ "$what" = sustain ]; then for d in "$RESULTS/$SOC"/sustain-*/; do [ -f "$d/jobs.tsv" ] && dirs+=("${d%/}"); done
		else for d in "$RESULTS/$SOC"/burst-*/; do [ -f "$d/jobs.tsv" ] && dirs+=("${d%/}"); done; fi
	fi
	((${#dirs[@]})) || die "no batch directories under $RESULTS/$SOC"
	out=$RESULTS/$SOC/summary
	local opt=(--soc "$SOC")
	[ -n "$CFG" ] && opt+=(--cfg "$CFG")
	[ -n "$HEADS_OPT" ] && opt+=(--heads "$HEADS")
	[ -n "$WINDOW" ] && opt+=(--window "$WINDOW")
	if [ "$what" = fig6 ]; then   # matplotlib lives in the container
		local r; r=$(realpath -m "$RESULTS")
		for x in "${!dirs[@]}"; do case ${dirs[$x]}/ in "$r"/*) dirs[x]=/results/${dirs[$x]#"$r"/} ;; *) die "fig6: ${dirs[$x]} is not under VLADUO_RESULTS" ;; esac; done
		VOLX=(-v "$r":/results); ensure_image
		crun python tools/vladuo.py summary fig6 "${dirs[@]}" "${opt[@]}" --out "/results/$SOC/summary"
	else
		"$PY" -B "$REL/tools/vladuo.py" summary "$what" "${dirs[@]}" "${opt[@]}" --out "$out"
	fi
}

# ---------------------------------------------------------------------------------------------------------------------
# check (offline)
# ---------------------------------------------------------------------------------------------------------------------
NFAIL=0
res() { printf '%-5s %s\n' "$1" "$2"; [ "$1" != FAIL ] || NFAIL=$((NFAIL + 1)); }

scrub() {   # prints every hit. Private patterns go in VLADUO_SCRUB_EXTRA (a file outside the release)
	local ex=(--exclude-dir=work --exclude-dir=models --exclude-dir=out --exclude-dir=results --exclude-dir=.git)
	local p1='/(hom[e]|User[s])/[a-z]'
	local ip='\b(100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])|192\.168|172\.(1[6-9]|2[0-9]|3[01])|10)\.[0-9]{1,3}\.[0-9]{1,3}(\.[0-9]{1,3})?\b'
	(
		set +e
		cd "$REL"
		grep -rnIiE "${ex[@]}" -e "$p1" .
		grep -rnIE "${ex[@]}" -e "$ip" .
		grep -rnIE "${ex[@]}" --include='*.sh' --include='*.md' --include='*.py' -e ':[0-9]{5}\b' .
		grep -nE '^((From|Author|Date|[A-Za-z-]+-[Bb]y):|index )' patches/vla-duo.patch
		grep -la 'Exif' data/inputs/*.jpg
		if [ -n "${VLADUO_SCRUB_EXTRA:-}" ]; then grep -rnIiE "${ex[@]}" -f "$VLADUO_SCRUB_EXTRA" .; fi
		if [ -d "$OUT/android" ]; then strings "$OUT"/android/bin/* "$OUT"/android/lib/* 2>/dev/null | grep -E "/(hom[e]|User[s])/[a-z]" | sed 's/^/out: /'; fi
		if [ -n "${VLADUO_SCRUB_EXTRA:-}" ] && [ -d "$OUT/android" ]; then strings "$OUT"/android/bin/* "$OUT"/android/lib/* | grep -iE -f "$VLADUO_SCRUB_EXTRA" | sed 's/^/out: /'; fi
	) || true
}

cmd_check() {
	local want have hits n s t idx ok
	cd "$REL"
	# file list
	want="Dockerfile LICENSE README.md data/inputs/cam_high.jpg data/inputs/cam_left_wrist.jpg data/inputs/cam_right_wrist.jpg data/models.md5 data/profiles.json patches/vla-duo.patch src/examples/pi0/CMakeLists.txt src/examples/pi0/pi0-bench.cpp src/examples/pi0/pi0-common.cpp src/examples/pi0/pi0-common.h src/examples/pi0/pi0-opmat-bench.cpp src/examples/pi0/pi0-pipeline.cpp src/ggml/src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl src/ggml/src/ggml-opencl/pi0_attn.cpp tools/gguf_tools.py tools/vladuo.py vladuo.sh"
	have=$(find . \( -path ./work -o -path ./out -o -path ./models -o -path ./results -o -path ./.git \) -prune -o -type f -print | sed 's|^\./||' | LC_ALL=C sort | tr '\n' ' ')
	n=$(echo "$have" | wc -w)
	if [ "$have" = "$(echo "$want" | tr ' ' '\n' | LC_ALL=C sort | tr '\n' ' ')" ]; then res PASS "files: the $n release files"
	else res FAIL "files: $n files, differs from the release list: $(comm -3 <(echo "$want" | tr ' ' '\n' | LC_ALL=C sort) <(echo "$have" | tr ' ' '\n' | LC_ALL=C sort) | tr -s '\t\n' '  ')"; fi
	# scrub
	hits=$(scrub)
	if [ -z "$hits" ]; then res PASS "scrub: no hits (built-in patterns${VLADUO_SCRUB_EXTRA:+ + $(grep -c . "$VLADUO_SCRUB_EXTRA") patterns from VLADUO_SCRUB_EXTRA})"
	else res FAIL "scrub:"; echo "$hits" | head -n 40 | sed 's/^/      /'; fi
	n=$(grep -rnE --exclude-dir={work,models,out,results,.git} '20[0-9]{2}-[01][0-9]-[0-3][0-9]' . | wc -l || true)
	[ "$n" = 0 ] || res WARN "dates: $n lines with a date (warning only)"
	# patch on a fresh llama.cpp 5744d7ec + ggml-hexagon tree, via a temporary index of the build clone (working tree untouched)
	s=$WORK/llama.cpp
	if [ -d "$s/.git" ] && git -C "$s" cat-file -e "$HEX^{commit}" 2>/dev/null; then
		idx=$(mktemp)
		if GIT_INDEX_FILE=$idx git -C "$s" read-tree $B8414 && GIT_INDEX_FILE=$idx git -C "$s" rm -r -q -f --cached ggml/src/ggml-hexagon &&
			GIT_INDEX_FILE=$idx git -C "$s" read-tree --prefix=ggml/src/ggml-hexagon/ "$HEX:ggml/src/ggml-hexagon" &&
			GIT_INDEX_FILE=$idx git -C "$s" apply --cached --check "$REL/patches/vla-duo.patch"; then res PASS "patch: applies to llama.cpp ${B8414:0:8} + ggml-hexagon@${HEX:0:9}"
		else res FAIL "patch: git apply --check failed"; fi
		rm -f "$idx"
	else res SKIP "patch: no llama.cpp clone in $s (./vladuo.sh build first)"; fi
	# build outputs (checks a-f)
	if [ -d "$OUT/android/bin" ] && ! { command -v file && command -v readelf && command -v strings; } >/dev/null; then res SKIP "out: needs file, readelf, strings"
	elif [ -d "$OUT/android/bin" ]; then
		ok=1
		for t in $BINS; do file -b "$OUT/android/bin/$t" | grep -q 'ELF 64-bit.*ARM aarch64' || { ok=0; echo "      $t: not an aarch64 ELF"; }; done
		[ "$(cd "$OUT/android/lib" && printf '%s\n' * | LC_ALL=C sort | tr '\n' ' ')" = "$(echo "$LIBS" | tr ' ' '\n' | LC_ALL=C sort | tr '\n' ' ')" ] || { ok=0; echo "      lib/ is not exactly the 9 libraries"; }
		for t in $LIBS; do
			case $t in libggml-htp-*) file -b "$OUT/android/lib/$t" | grep -q 'ELF 32-bit.*QUALCOMM DSP6' ;; *) file -b "$OUT/android/lib/$t" | grep -q 'ELF 64-bit.*ARM aarch64' ;; esac || { ok=0; echo "      $t: wrong ELF type"; }
		done
		file -b "$OUT/host/llama-quantize" | grep -q 'ELF 64-bit.*x86-64' || { ok=0; echo "      host/llama-quantize: not x86-64"; }
		if [ ! -d "$OUT/gguf-py/gguf" ] || [ ! -f "$OUT/BUILD_INFO.txt" ]; then ok=0; echo "      gguf-py/ or BUILD_INFO.txt missing"; fi
		if [ $ok = 1 ]; then res PASS "out (a): ELF types, 9 libraries, gguf-py, BUILD_INFO.txt"; else res FAIL "out (a)"; fi
		t=$(readelf -d "$OUT/android/bin/llama-pi0-pipeline" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -vE '^(libmtmd|libllama|libggml[a-z0-9-]*|libc|libm|libdl|liblog)\.so$' || true)
		if [ -z "$t" ]; then res PASS "out (b): NEEDED of llama-pi0-pipeline"; else res FAIL "out (b): unexpected NEEDED $t"; fi
		t=$(strings "$OUT"/android/bin/* "$OUT"/android/lib/* | grep -cE "/(hom[e]|User[s])/[a-z]" || true)
		if [ "$t" = 0 ]; then res PASS "out (c): strings scan"; else res FAIL "out (c): $t identifying strings"; fi
		if [ -d "$s/.git" ]; then
			t=$(git -C "$s" status --porcelain | cut -c4- | grep -vE '^(ggml/src/ggml-hexagon/|ggml/src/ggml-opencl/(ggml-opencl\.cpp|CMakeLists\.txt|pi0_attn\.cpp|kernels/mul_mat_Ab_Bi_8x4\.cl|kernels/transpose\.cl|kernels/gemm_xmem_f16_f32_os8\.cl)$|tools/mtmd/(clip\.cpp|clip\.h|clip-graph\.h|clip-impl\.h|models/siglip\.cpp|mtmd\.cpp|mtmd\.h)$|examples/CMakeLists\.txt$|examples/pi0/|CMakeUserPresets\.json$)' || true)
			if [ -z "$t" ]; then res PASS "out (d): build tree touches only the patched/new paths"; else res FAIL "out (d): $(echo "$t" | tr '\n' ' ')"; fi
		fi
		if command -v docker >/dev/null && docker image inspect "$IMAGE" >/dev/null 2>&1; then
			if docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$(realpath "$OUT")":/out:ro "$IMAGE" sh -c "/out/host/llama-quantize 2>&1 | grep -q usage && python -c \"import sys; sys.path.insert(0, '/out/gguf-py'); import gguf, sentencepiece, numpy, yaml\"" >/dev/null 2>&1
			then res PASS "out (a, e): llama-quantize runs and gguf/sentencepiece/numpy/yaml import in $IMAGE"; else res FAIL "out (a, e): container checks"; fi
		else res SKIP "out (e): image $IMAGE not built"; fi
	else res SKIP "out: no build outputs in $OUT"; fi
	if grep -qE '<64 hex' "$REL/Dockerfile"; then res FAIL "Dockerfile (f): unpinned digest"; else res PASS "Dockerfile (f): digests pinned"; fi
	# models
	if [ -d "$MODELS/sets" ] && [ -z "$FAST" ]; then
		n=$(cd "$MODELS" && { md5sum -c --ignore-missing "$REL/data/models.md5" 2>/dev/null | grep -c ': OK$' || true; })
		if (cd "$MODELS" && md5sum -c --quiet --ignore-missing "$REL/data/models.md5" >/dev/null 2>&1); then res PASS "models: md5sum -c, $n files OK"
		else res FAIL "models: md5sum -c (cd $MODELS && md5sum -c $REL/data/models.md5)"; fi
	elif [ -n "$FAST" ]; then res SKIP "models: --fast"
	else res SKIP "models: no $MODELS/sets"; fi
	# cut model reference values
	if "$PY" -B "$REL/tools/vladuo.py" check >"${TMPDIR:-/tmp}/vladuo-check.$$" 2>&1; then res PASS "tools/vladuo.py check"
	else res FAIL "tools/vladuo.py check:"; tail -n 20 "${TMPDIR:-/tmp}/vladuo-check.$$" | sed 's/^/      /'; fi
	rm -f "${TMPDIR:-/tmp}/vladuo-check.$$"
	if [ "$NFAIL" = 0 ]; then echo "check: PASS"; else echo "check: FAIL ($NFAIL)"; exit 1; fi
}

# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------
SOC=""; CFG=""; XE=""; MODEL=all; RUNS=3; RUNS_OPT=""; HEADS=hf16; HEADS_OPT=""; CUT=auto; DIR=""; CUTS=""; KMAX=9; PLANS=""; SMALLBG=""; FORCE=""; FAST=""
KEY=""; WINDOW=""; MODE=""; PERIOD=""; DUR=300; GPU_CAP=""; ARGS=()
CMD=${1:-}; [ $# -gt 0 ] && shift
case $CMD in
_build) _build; exit ;;
replay)
	case ${1:-} in rates | select | curve | kcut) ;; *) die "replay rates|select|curve|kcut [ARGS] (see tools/vladuo.py <sub> --help)" ;; esac
	exec "$PY" -B "$REL/tools/vladuo.py" "$@" ;;
"" | -h | --help | help) usage; exit 0 ;;
esac
while [ $# -gt 0 ]; do
	case $1 in
	--soc) SOC=${2:?--soc}; shift ;; --cfg) CFG=${2:?--cfg}; shift ;; --model) MODEL=${2:?--model}; shift ;;
	--runs) RUNS=${2:?--runs}; RUNS_OPT=1; shift ;; --heads) HEADS=${2:?--heads}; HEADS_OPT=1; shift ;; --cut) CUT=${2:?--cut}; shift ;;
	--dir) DIR=${2:?--dir}; shift ;; --cuts) CUTS=${2:?--cuts}; shift ;; --kmax) KMAX=${2:?--kmax}; shift ;;
	--plans) PLANS=$(realpath "${2:?--plans}"); shift ;; --key) KEY=${2:?--key}; shift ;; --window) WINDOW=${2:?--window}; shift ;;
	--mode) MODE=${2:?--mode}; shift ;; --period) PERIOD=${2:?--period}; shift ;; --dur) DUR=${2:?--dur}; shift ;;
	--gpu-cap) GPU_CAP=${2:?--gpu-cap}; shift ;;
	--small-bg) SMALLBG=1 ;; --force) FORCE=1 ;; --fast) FAST=1 ;; --dry-run) DRYRUN=1 ;;
	-h | --help) usage; exit 0 ;;
	-*) die "unknown option $1 (./vladuo.sh --help)" ;;
	*) ARGS+=("$1") ;;
	esac
	shift
done
case $MODEL in pi0 | pi05 | all) ;; *) die "--model pi0|pi05|all" ;; esac
case $HEADS in hf16 | q4 | both) ;; *) die "--heads hf16|q4|both" ;; esac
[[ $RUNS =~ ^[1-9][0-9]*$ ]] || die "--runs N"
case $CMD in
build) cmd_build ;;
models) cmd_models ;;
push) cmd_push ;;
profile) cmd_profile ;;
burst) cmd_burst ;;
sustain) cmd_sustain ;;
kcut) cmd_kcut ;;
promote) cmd_promote ;;
summary) cmd_summary ;;
check) cmd_check ;;
_pull)
	[ ${#ARGS[@]} = 2 ] || die "_pull SOC BATCH"
	device_ready; wait_pull "${ARGS[0]}" "${ARGS[1]}" ;;
*) die "unknown command $CMD (./vladuo.sh --help)" ;;
esac
