#!/usr/bin/env python3
"""VLA-Duo model preparation: public pi0 / pi0.5 checkpoints -> the GGUF sets loaded by examples/pi0.

  convert          model.safetensors -> <p>-mmproj.gguf, <p>-gemma-2b.gguf, <p>-action-expert.gguf (p = pi0 | pi05)
  quantize-expert  action expert -> Q4_0 (llama-quantize does not know the pi0-action-expert architecture)
  fuse-qkv         blk.N.attn_{q,k,v}.weight -> blk.N.attn_qkv.weight
  fuse-gateup      blk.N.ffn_{gate,up}.weight -> blk.N.ffn_gateup.weight, output rows interleaved in 32-row tiles
  split-k          split weights along the input (K) axis into <name>_k0.._k{N-1}
  pad-mmproj       zero-pad the SigLIP FFN width (n_ff) to a multiple of --align
  heads16          copy of a Q4_0 expert whose action-head weights come from the F16 expert (self-verifying)
  prepare          download + verify the public sources, run the recipe, link sets/, check data/models.md5

Every transform is exact: whole quant rows/blocks are moved, never re-quantized, and padding is zero.
gguf-py is taken from PYTHONPATH (out/gguf-py of the build).
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("gguf_tools.py: gguf-py not importable; run with PYTHONPATH=<out>/gguf-py")

QT = gguf.GGMLQuantizationType
SELF = str(Path(__file__).resolve())
ARRAY = gguf.GGUFValueType.ARRAY


# ---------------------------------------------------------------------------------------------------------------
# GGUF rewrite helpers (metadata, order, alignment and raw bytes of untouched tensors are kept)

def _write(r, dst, tensors, kv=None):
    """Write `tensors` [(name, ndarray, ggml type)] to dst with the metadata of reader r (kv: value overrides)."""
    w = gguf.GGUFWriter(dst, arch=str(r.fields["general.architecture"].contents()), endianess=r.endianess)
    al = r.fields.get("general.alignment")
    if al is not None:
        w.data_alignment = al.contents()
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        val = kv[f.name] if kv and f.name in kv else f.contents()
        w.add_key_value(f.name, val, vt, sub_type=f.types[-1] if vt == ARRAY else None)
    for name, data, qt in tensors:
        w.add_tensor(name, data, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _copy(t):
    return (t.name, np.array(t.data), t.tensor_type)


QKV = (".attn_q.weight", ".attn_k.weight", ".attn_v.weight")


def _fuse_qkv(r, fused_fn, other_fn, qkv_first):
    """Each layer's attn_q/k/v.weight -> one attn_qkv.weight (rows q, k, v) at attn_q's slot, or all first."""
    by = {t.name: t for t in r.tensors}
    bases = {t.name[:-len(QKV[0])] for t in r.tensors if t.name.endswith(QKV[0])}
    out = []
    for t in r.tensors:
        base = next((t.name[:-len(s)] for s in QKV if t.name.endswith(s)), None)
        if base in bases:
            if t.name.endswith(QKV[0]):
                out.append(fused_fn(base + ".attn_qkv.weight", [by[base + s] for s in QKV]))
            continue
        out.append(other_fn(t))
    if qkv_first:
        out = [x for x in out if x[0].endswith(".attn_qkv.weight")] + \
              [x for x in out if not x[0].endswith(".attn_qkv.weight")]
    return out


# ---------------------------------------------------------------------------------------------------------------
# transforms

def cmd_quantize_expert(a):
    """RTN Q4_0 of every 2-D weight whose in-features are a multiple of 32. Norms and biases are copied.
    --fuse-qkv concatenates q/k/v before quantization."""
    qt = {"q4_0": QT.Q4_0}[a.qtype]
    r = gguf.GGUFReader(a.src)

    def quant(name, arr):
        return (name, gguf.quants.quantize(arr.astype(np.float32), qt), qt)

    def other(t):
        if t.name.endswith(".weight") and len(t.shape) == 2 and int(t.shape[0]) % 32 == 0:
            return quant(t.name, np.array(t.data))
        return _copy(t)

    def fused(name, parts):
        return quant(name, np.concatenate([np.array(p.data) for p in parts], axis=0))

    if a.fuse_qkv:
        out = _fuse_qkv(r, fused, other, a.qkv_first)
    else:
        out = [other(t) for t in r.tensors]
    _write(r, a.dst, out)
    print(f"{a.qtype}: {len(out)} tensors -> {a.dst}")


def cmd_fuse_qkv(a):
    """Row concatenation of the stored q, k, v rows (exact for block quants: blocks run along K)."""
    r = gguf.GGUFReader(a.src)

    def fused(name, parts):
        assert len({p.tensor_type for p in parts}) == 1 and len({int(p.shape[0]) for p in parts}) == 1, name
        return (name, np.concatenate([np.array(p.data) for p in parts], axis=0), parts[0].tensor_type)

    out = _fuse_qkv(r, fused, _copy, a.qkv_first)
    _write(r, a.dst, out)
    print(f"fused qkv, {len(out)} tensors -> {a.dst}")


def cmd_fuse_gateup(a):
    """ffn_gateup rows [64j, 64j+32) = gate rows [32j, 32j+32), rows [64j+32, 64j+64) = up rows [32j, 32j+32):
    the HMX output tile is 32 wide, so gelu(gate)*up runs in the matmul store epilogue (HTP_OP_MUL_MAT_GEGLU)."""
    tile = 32
    r = gguf.GGUFReader(a.src)
    by = {t.name: t for t in r.tensors}
    g_s, u_s = ".ffn_gate.weight", ".ffn_up.weight"
    bases = {n[:-len(g_s)] for n in by if n.endswith(g_s) and n[:-len(g_s)] + u_s in by}
    out = []
    for t in r.tensors:
        base = next((t.name[:-len(s)] for s in (g_s, u_s) if t.name.endswith(s)), None)
        if base in bases:
            if t.name.endswith(g_s):
                g, u = by[base + g_s], by[base + u_s]
                assert g.tensor_type == u.tensor_type and tuple(g.shape) == tuple(u.shape), base
                gd, ud = np.array(g.data), np.array(u.data)
                ff = gd.shape[0]
                assert ff % tile == 0, (base, ff)
                il = np.stack([gd.reshape(ff // tile, tile, *gd.shape[1:]), ud.reshape(ff // tile, tile, *ud.shape[1:])],
                              axis=1).reshape((2 * ff,) + gd.shape[1:])
                out.append((base + ".ffn_gateup.weight", il, g.tensor_type))
            continue
        out.append(_copy(t))
    _write(r, a.dst, out)
    print(f"interleaved gate/up in {len(bases)} layers -> {a.dst}")


def cmd_split_k(a):
    """NAME[:N] (N = 2 by default): split along K into N slices of whole quant blocks. The graph sums the partial
    products in f32. `ffn_down` means every blk.N.ffn_down.weight, other names are exact tensor names."""
    sel = {}
    for tok in a.names:
        name, _, n = tok.partition(":")
        sel[name] = int(n) if n else 2
    r = gguf.GGUFReader(a.src)
    names = {t.name for t in r.tensors}
    missing = [n for n in sel if n != "ffn_down" and n not in names]
    if missing:
        sys.exit(f"split-k: tensors not found: {missing}")
    out = []
    for t in r.tensors:
        n = sel.get(t.name) or (sel.get("ffn_down") if t.name.endswith(".ffn_down.weight") else None)
        if not n:
            out.append(_copy(t))
            continue
        blk = gguf.GGML_QUANT_SIZES[QT(t.tensor_type)][0]
        assert n >= 2 and (int(t.shape[0]) // blk) % n == 0, (t.name, n)
        data = np.array(t.data)                          # [out rows, row bytes | K]
        step = data.shape[1] // n
        base = t.name[:-len(".weight")]
        out += [(f"{base}_k{i}.weight", np.ascontiguousarray(data[:, i * step:(i + 1) * step]), t.tensor_type)
                for i in range(n)]
    _write(r, a.dst, out)
    print(f"K-split -> {a.dst}")


def cmd_pad_mmproj(a):
    """SigLIP n_ff 4304 -> 4320: HMX needs ne00 % 32 == 0 and ne01 % 32 == 0 for f16 weights. Zero rows of ffn_up
    (+ zero bias) give hidden lanes equal to 0, gelu(0) = 0, and zero K-columns of ffn_down add nothing."""
    r = gguf.GGUFReader(a.src)
    key = "clip.vision.feed_forward_length"
    n_ff = int(r.fields[key].contents())
    pad = (n_ff + a.align - 1) // a.align * a.align - n_ff
    out = []
    for t in r.tensors:
        d = t.data
        if t.name.endswith("ffn_up.weight"):
            d = np.pad(d, ((0, pad), (0, 0)))
        elif t.name.endswith("ffn_up.bias"):
            d = np.pad(d, (0, pad))
        elif t.name.endswith("ffn_down.weight"):
            d = np.pad(d, ((0, 0), (0, pad)))
        out.append((t.name, d, QT(t.tensor_type)))
    _write(r, a.dst, out, kv={key: n_ff + pad})
    print(f"n_ff {n_ff} -> {n_ff + pad} -> {a.dst}")


HEADS = ("action_in_proj.weight", "action_out_proj.weight", "state_proj.weight", "action_time_mlp_in.weight",
         "action_time_mlp_in_k0.weight", "action_time_mlp_in_k1.weight", "action_time_mlp_out.weight",
         "time_mlp_in.weight", "time_mlp_out.weight")


def cmd_heads16(a):
    """Tensor-for-tensor copy of the Q4_0 expert (metadata, order, alignment, raw bytes) in which the action-head
    weights are the F16 donor's bytes. Verifies the result."""
    q4, f16 = gguf.GGUFReader(a.q4), gguf.GGUFReader(a.f16)
    donor = {t.name: t for t in f16.tensors}
    heads = [t.name for t in q4.tensors if t.name in HEADS]
    assert heads, "no head tensor found"
    for t in q4.tensors:
        if t.name in heads:
            d = donor[t.name]
            assert t.tensor_type == QT.Q4_0 and d.tensor_type == QT.F16 and list(t.shape) == list(d.shape), t.name
    src = [donor[t.name] if t.name in heads else t for t in q4.tensors]
    _write(q4, a.dst, [(t.name, t.data, t.tensor_type) for t in src])

    new = gguf.GGUFReader(a.dst)

    def kv(r):
        return [(f.name, [x.name for x in f.types], f.contents()) for f in r.fields.values()
                if not f.name.startswith("GGUF.")]
    assert kv(new) == kv(q4), "metadata differs"
    assert [t.name for t in new.tensors] == [t.name for t in q4.tensors], "tensor order differs"
    for o, n in zip(q4.tensors, new.tensors):
        ref = donor[o.name] if o.name in heads else o
        assert n.tensor_type == ref.tensor_type and list(n.shape) == list(o.shape) and \
            n.data.tobytes() == ref.data.tobytes(), o.name
        if o.name in heads:
            deq = gguf.quants.dequantize(o.data, o.tensor_type).astype(np.float32).reshape(-1)
            hf = np.asarray(n.data, dtype=np.float32).reshape(-1)
            rel = float(np.sqrt(np.mean((deq - hf) ** 2)) / np.sqrt(np.mean(hf ** 2)))
            print(f"  head {o.name:30s} {list(map(int, o.shape))}  Q4_0 {o.n_bytes} B -> F16 {n.n_bytes} B"
                  f"  Q4-vs-F16 rel-RMS {rel * 100:.2f} %")
    print(f"  OK: {len(q4.tensors) - len(heads)} tensors byte-identical to the Q4 file, {len(heads)} heads = F16 donor"
          f" bytes, metadata identical -> {a.dst} ({os.path.getsize(a.dst)} B)")


# ---------------------------------------------------------------------------------------------------------------
# convert (HF safetensors -> f16 GGUF)

class _SafeTensors:
    def __init__(self, path, round_bf16):
        p = Path(path)
        self.f = open(p / "model.safetensors" if p.is_dir() else p, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.round_bf16 = round_bf16

    def __call__(self, name):
        info = self.header[name]
        s, e = info["data_offsets"]
        self.f.seek(self.base + s)
        raw = self.f.read(e - s)
        shape = info["shape"]
        if info["dtype"] == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16).reshape(shape).astype(np.uint32) << 16).view(np.float32)
        if info["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        a = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if self.round_bf16:                              # F32 -> bf16, round to nearest even
            u = a.view(np.uint32)
            a = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
        return a


# tensors used by ggml_mul_mat, stored as f16 with --type f16 (everything else f32)
MUL_MAT_NAMES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight", "attn_out.weight", "ffn_up.weight",
    "ffn_down.weight", "mm.0.weight", "attn_output.weight", "ffn_gate.weight",
    "action_in_proj.weight", "action_out_proj.weight", "state_proj.weight", "action_time_mlp_in.weight",
    "action_time_mlp_out.weight", "attn_norm.dense.weight", "ffn_norm.dense.weight", "output_norm.dense.weight",
    "time_mlp_in.weight", "time_mlp_out.weight")

GEMMA_LAYER = (("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
               ("attn_output", "self_attn.o_proj"), ("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"),
               ("ffn_down", "mlp.down_proj"))


def _put(w, name, data, f16):
    w.add_tensor(name, data.astype(np.float16 if f16 and name.endswith(MUL_MAT_NAMES) else np.float32))


def _finish(w, path):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"written {path} ({os.path.getsize(path) / 2**20:.1f} MiB)")


def _gemma_kv(w, n_embd, n_ff):
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_block_count(18)
    w.add_head_count(8)
    w.add_head_count_kv(1)
    w.add_key_length(256)
    w.add_value_length(256)
    w.add_layer_norm_rms_eps(1e-6)


def _conv_mmproj(ld, path, f16):
    w = gguf.GGUFWriter(str(path), arch="clip")
    w.add_type(gguf.GGUFType.MMPROJ)
    w.add_string("general.name", "pi0")
    w.add_clip_projector_type("pi0")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_vision_image_size(224)
    w.add_vision_patch_size(14)
    w.add_vision_embedding_length(1152)
    w.add_vision_feed_forward_length(4304)
    w.add_vision_block_count(27)
    w.add_vision_head_count(16)
    w.add_vision_attention_layernorm_eps(1e-6)
    w.add_vision_image_mean([0.5, 0.5, 0.5])
    w.add_vision_image_std([0.5, 0.5, 0.5])
    w.add_vision_use_gelu(True)
    w.add_uint32("clip.vision.projection_dim", 0)
    vt = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    names = [("v.patch_embd", "embeddings.patch_embedding"), ("v.position_embd.weight", "embeddings.position_embedding.weight"),
             ("v.post_ln", "post_layernorm")]
    for il in range(27):
        names += [(f"v.blk.{il}.{d}", f"encoder.layers.{il}.{s}") for d, s in (
            ("ln1", "layer_norm1"), ("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"),
            ("attn_v", "self_attn.v_proj"), ("attn_out", "self_attn.out_proj"), ("ln2", "layer_norm2"),
            ("ffn_up", "mlp.fc1"), ("ffn_down", "mlp.fc2"))]
    for d, s in names:
        for wb in ((".weight", ".bias") if not d.endswith(".weight") else ("",)):
            _put(w, d + wb, ld(f"{vt}.{s}{wb}"), f16)
    mm = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    _put(w, "mm.0.weight", ld(mm + ".weight"), f16)
    _put(w, "mm.0.bias", ld(mm + ".bias"), f16)
    _finish(w, path)


def _add_tokenizer(w, tok_path, legacy_kv):
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(tok_path))
    tokens, scores, types = [], [], []
    for i in range(sp.GetPieceSize()):
        tokens.append(sp.IdToPiece(i).encode("utf-8"))
        scores.append(sp.GetScore(i))
        types.append(2 if sp.IsUnknown(i) else 3 if sp.IsControl(i) else 5 if sp.IsUnused(i) else 6 if sp.IsByte(i) else 1)
    w.add_string("tokenizer.ggml.model", "llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_uint32("tokenizer.ggml.bos_token_id", sp.bos_id())
    w.add_uint32("tokenizer.ggml.eos_token_id", sp.eos_id())
    w.add_uint32("tokenizer.ggml.padding_token_id", sp.pad_id())
    w.add_uint32("tokenizer.ggml.unknown_token_id", sp.unk_id())
    if not legacy_kv:
        # PaliGemma's SentencePiece model adds no dummy space prefix ('pick' -> 18075, not '▁pick' 4788).
        # llama.cpp defaults add_space_prefix to true for SPM.
        w.add_bool("tokenizer.ggml.add_space_prefix", False)


def _conv_gemma(ld, path, f16, tok_path, legacy_kv):
    w = gguf.GGUFWriter(str(path), arch="gemma")
    w.add_type(gguf.GGUFType.MODEL)
    w.add_string("general.name", "pi0-paligemma-2b")
    w.add_context_length(2048)
    _gemma_kv(w, 2048, 16384)
    w.add_vocab_size(257152)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if f16 else gguf.LlamaFileType.ALL_F32)
    _add_tokenizer(w, tok_path, legacy_kv)
    lm = "paligemma_with_expert.paligemma.model.language_model"
    _put(w, "token_embd.weight", ld("paligemma_with_expert.paligemma.lm_head.weight"), f16)   # tied embedding
    for il in range(18):
        p = f"{lm}.layers.{il}"
        _put(w, f"blk.{il}.attn_norm.weight", ld(p + ".input_layernorm.weight") + 1.0, False)  # Gemma (1 + w)
        _put(w, f"blk.{il}.ffn_norm.weight", ld(p + ".post_attention_layernorm.weight") + 1.0, False)
        for d, s in GEMMA_LAYER:
            _put(w, f"blk.{il}.{d}.weight", ld(f"{p}.{s}.weight"), f16)
    _put(w, "output_norm.weight", ld(lm + ".norm.weight") + 1.0, False)
    _finish(w, path)


def _conv_expert(ld, path, f16, pi05):
    """pi0: Gemma-300M with (1 + w) RMSNorm, state_proj and action_time_mlp. pi0.5: adaRMS norms (Linear width ->
    3 * width '.dense', no per-channel scale, (1 + scale) applied in the graph) and time_mlp, no state token."""
    w = gguf.GGUFWriter(str(path), arch="pi0-action-expert")
    w.add_type(gguf.GGUFType.MODEL)
    w.add_string("general.name", "pi05-action-expert" if pi05 else "pi0-action-expert")
    _gemma_kv(w, 1024, 4096)
    w.add_uint32("pi0.action_dim", 32)
    w.add_uint32("pi0.action_horizon", 50)
    if pi05:
        w.add_bool("pi0.pi05", True)
        w.add_uint32("pi0.adarms_cond_dim", 1024)
    ex = "paligemma_with_expert.gemma_expert.model"
    for il in range(18):
        p = f"{ex}.layers.{il}"
        for d, s in (("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
            if pi05:
                _put(w, f"blk.{il}.{d}.dense.weight", ld(f"{p}.{s}.dense.weight"), f16)
                _put(w, f"blk.{il}.{d}.dense.bias", ld(f"{p}.{s}.dense.bias"), False)
            else:
                _put(w, f"blk.{il}.{d}.weight", ld(f"{p}.{s}.weight") + 1.0, False)
        for d, s in GEMMA_LAYER:
            _put(w, f"blk.{il}.{d}.weight", ld(f"{p}.{s}.weight"), f16)
    if pi05:
        _put(w, "output_norm.dense.weight", ld(ex + ".norm.dense.weight"), f16)
        _put(w, "output_norm.dense.bias", ld(ex + ".norm.dense.bias"), False)
        heads = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    else:
        _put(w, "output_norm.weight", ld(ex + ".norm.weight") + 1.0, False)
        heads = ("action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in", "action_time_mlp_out")
    for h in heads:
        _put(w, h + ".weight", ld(h + ".weight"), f16)
        _put(w, h + ".bias", ld(h + ".bias"), False)
    _finish(w, path)


def cmd_convert(a):
    ld = _SafeTensors(a.model, a.round_bf16)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    f16 = a.type == "f16"
    p = "pi05" if a.pi05 else "pi0"
    _conv_mmproj(ld, out / f"{p}-mmproj.gguf", f16)
    _conv_gemma(ld, out / f"{p}-gemma-2b.gguf", f16, a.tokenizer, a.legacy_tokenizer_kv)
    _conv_expert(ld, out / f"{p}-action-expert.gguf", f16, a.pi05)


# ---------------------------------------------------------------------------------------------------------------
# prepare

SOURCES = {   # key: (path under DIR, URL, hash, digest, size, local override)
    "pi0": ("src/pi0_base/model.safetensors",
            "https://huggingface.co/lerobot/pi0_base/resolve/6e245399ad403ccf02b8112b73cfdc2633068f38/model.safetensors",
            "sha256", "8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303", 14005618584,
            "/hf/pi0.safetensors"),
    "pi05": ("src/pi05_base/model.safetensors",
             "https://huggingface.co/lerobot/pi05_base/resolve/7de663972b7817d2c4cf2d84c821153dfea772e9/model.safetensors",
             "sha256", "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f", 14467165872,
             "/hf/pi05.safetensors"),
    "tok": ("src/paligemma_tokenizer.model", "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model",
            "md5", "1420adc9856720a559e8a87284b195e2", 4264023, None),
}

# step = KIND OUTPUT ARGS. C = convert, LQ = llama-quantize, Q = transform, '*' = needed only for hf16 heads.
RECIPE = {
    "pi0": """
C   f16/pi0                 --model {src} --output-dir {out} --type f16 --tokenizer {tok} --round-bf16 --legacy-tokenizer-kv
LQ  t/pi0-lm-q4.gguf        f16/pi0/pi0-gemma-2b.gguf {out} Q4_0
Q   t/pi0-lm-qkv.gguf       fuse-qkv --qkv-first t/pi0-lm-q4.gguf {out}
Q   t/pi0-lm-gu.gguf        fuse-gateup t/pi0-lm-qkv.gguf {out}
Q   t/pi0-lm-npu.gguf       split-k t/pi0-lm-gu.gguf {out} ffn_down:4
Q   t/pi0-ex-q4.gguf        quantize-expert f16/pi0/pi0-action-expert.gguf {out} q4_0 --fuse-qkv --qkv-first
Q   t/pi0-ex-gu.gguf        fuse-gateup t/pi0-ex-q4.gguf {out}
Q   t/pi0-ex-npu-q4h.gguf   split-k t/pi0-ex-gu.gguf {out} action_time_mlp_in.weight
Q*  t/pi0-ex-f16-ks.gguf    split-k f16/pi0/pi0-action-expert.gguf {out} action_time_mlp_in.weight
Q*  t/pi0-ex-npu-hf16.gguf  heads16 t/pi0-ex-npu-q4h.gguf t/pi0-ex-f16-ks.gguf {out}
Q*  t/pi0-ex-gpu-hf16.gguf  heads16 t/pi0-ex-q4.gguf f16/pi0/pi0-action-expert.gguf {out}
Q   t/pi0-mmproj-pad.gguf   pad-mmproj f16/pi0/pi0-mmproj.gguf {out}
""",
    "pi05": """
C   f16/pi05                --model {src} --output-dir {out} --type f16 --tokenizer {tok} --pi05
LQ  t/pi05-lm-q4.gguf       f16/pi05/pi05-gemma-2b.gguf {out} Q4_0
Q   t/pi05-lm-qkv.gguf      fuse-qkv t/pi05-lm-q4.gguf {out}
Q   t/pi05-lm-gu.gguf       fuse-gateup t/pi05-lm-qkv.gguf {out}
Q   t/pi05-lm-npu.gguf      split-k t/pi05-lm-gu.gguf {out} ffn_down
Q   t/pi05-ex-q4.gguf       quantize-expert f16/pi05/pi05-action-expert.gguf {out} q4_0 --fuse-qkv
Q   t/pi05-ex-npu-q4h.gguf  fuse-gateup t/pi05-ex-q4.gguf {out}
Q*  t/pi05-ex-npu-hf16.gguf heads16 t/pi05-ex-npu-q4h.gguf f16/pi05/pi05-action-expert.gguf {out}
Q*  t/pi05-ex-gpu-hf16.gguf heads16 t/pi05-ex-q4.gguf f16/pi05/pi05-action-expert.gguf {out}
Q   t/pi05-mmproj-pad.gguf  pad-mmproj f16/pi05/pi05-mmproj.gguf {out}
""",
}

SET_FILES = ("pi0-mmproj.gguf", "pi0-gemma-2b.gguf", "pi0-gemma-2b-fused.gguf", "pi0-action-expert.gguf")
SETS = {   # hf16 sets. The -q4h twin takes ex-npu-q4h (npu) or ex-q4 (gpu, stock) as its expert
    "pi0-npu": ("t/pi0-mmproj-pad.gguf", "t/pi0-lm-q4.gguf", "t/pi0-lm-npu.gguf", "t/pi0-ex-npu-hf16.gguf"),
    "pi0-gpu": ("t/pi0-mmproj-pad.gguf", "t/pi0-lm-q4.gguf", "t/pi0-lm-qkv.gguf", "t/pi0-ex-gpu-hf16.gguf"),
    "pi0-stock": ("f16/pi0/pi0-mmproj.gguf", "t/pi0-lm-q4.gguf", "t/pi0-lm-qkv.gguf", "t/pi0-ex-gpu-hf16.gguf"),
    "pi05-npu": ("t/pi05-mmproj-pad.gguf", "t/pi05-lm-q4.gguf", "t/pi05-lm-npu.gguf", "t/pi05-ex-npu-hf16.gguf"),
    "pi05-gpu": ("t/pi05-mmproj-pad.gguf", "t/pi05-lm-q4.gguf", "t/pi05-lm-qkv.gguf", "t/pi05-ex-gpu-hf16.gguf"),
}


_DIGESTS = {}


def _digest(path, algo):
    st = os.stat(path)
    k = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, algo)
    if k not in _DIGESTS:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _DIGESTS[k] = h.hexdigest()
    return _DIGESTS[k]


def _source(d, key, local):
    """Path of a verified source: the local override, else DIR/<rel> (downloaded when missing)."""
    rel, url, algo, want, size, default_local = SOURCES[key]
    if local is None and default_local and os.path.isfile(default_local):
        local = default_local
    path = Path(local) if local else d / rel
    if not path.is_file():
        if local:
            sys.exit(f"[prepare] {path}: not found")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        print(f"[prepare] download {url}", flush=True)
        subprocess.run(["curl", "-L", "--fail", "-o", str(part), url], check=True)
        os.replace(part, path)
    got = _digest(path, algo) if path.stat().st_size == size else f"size {path.stat().st_size}"
    if got != want:
        sys.exit(f"[prepare] {path}: {algo} {got} != {want} (delete it and run again)")
    print(f"[prepare] {path}: {algo} ok", flush=True)
    return path


def _md5_list():
    lst = Path(__file__).resolve().parents[1] / "data" / "models.md5"
    known, diag = {}, {}
    for line in lst.read_text().splitlines() if lst.is_file() else []:
        f = line.lstrip("# ").split()
        if len(f) == 2 and len(f[0]) == 32:
            (diag if line.startswith("#") else known)[f[1]] = f[0]
    return known, diag


def cmd_prepare(a):
    d = Path(a.models).resolve()
    d.mkdir(parents=True, exist_ok=True)
    lq = a.llama_quantize or str(Path(gguf.__file__).resolve().parents[2] / "host" / "llama-quantize")
    models = ["pi0", "pi05"] if a.model == "all" else [a.model]
    tok = _source(d, "tok", None)
    for m in models:
        src = _source(d, m, a.pi0_safetensors if m == "pi0" else a.pi05_safetensors)
        for line in RECIPE[m].strip().splitlines():
            kind, out, *args = line.split()
            if kind.endswith("*") and a.heads == "q4":
                continue
            if (d / out).exists():
                print(f"[prepare] {out}: exists, skipped", flush=True)
                continue
            part = d / (out + ".part")
            if part.is_dir():
                shutil.rmtree(part)
            part.parent.mkdir(parents=True, exist_ok=True)
            args = [x.format(src=src, tok=tok, out=part) for x in args]
            if kind.startswith("C"):
                cmd = [sys.executable, "-B", SELF, "convert"] + args
            elif kind == "LQ":
                cmd = [lq] + args
            else:
                cmd = [sys.executable, "-B", SELF] + args
            print(f"[prepare] {kind} {' '.join(args)}", flush=True)
            subprocess.run(cmd, cwd=d, check=True)
            os.replace(part, d / out)
    # sets: hard links, file names fixed by the loaders
    want = [s for s in SETS if s.split("-")[0] in models]
    q4h = {"t/pi0-ex-npu-hf16.gguf": "t/pi0-ex-npu-q4h.gguf", "t/pi0-ex-gpu-hf16.gguf": "t/pi0-ex-q4.gguf",
           "t/pi05-ex-npu-hf16.gguf": "t/pi05-ex-npu-q4h.gguf", "t/pi05-ex-gpu-hf16.gguf": "t/pi05-ex-q4.gguf"}
    sets = {}
    for s in want:
        if a.heads in ("hf16", "both"):
            sets[s] = SETS[s]
        if a.heads in ("q4", "both"):
            sets[s + "-q4h"] = SETS[s][:3] + (q4h[SETS[s][3]],)
    for s, files in sets.items():
        (d / "sets" / s).mkdir(parents=True, exist_ok=True)
        for name, srcf in zip(SET_FILES, files):
            dst = d / "sets" / s / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(d / srcf, dst)
    # md5 check against data/models.md5
    known, diag = _md5_list()
    bad = 0
    for s in sets:
        for name in SET_FILES:
            rel = f"sets/{s}/{name}"
            got = _digest(d / rel, "md5")
            ok = "NEW" if rel not in known else "OK" if known[rel] == got else "FAILED"
            bad += ok == "FAILED"
            print(f"[prepare] {ok:6s} {got}  {rel}")
    if bad:
        for rel, md5 in diag.items():
            if (d / rel).is_file():
                print(f"[prepare] stage {rel}: {'OK' if _digest(d / rel, 'md5') == md5 else 'FAILED'}")
        sys.exit(f"[prepare] {bad} file(s) differ from data/models.md5")
    print(f"[prepare] done: {len(sets)} sets in {d / 'sets'}")


# ---------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="HF model.safetensors -> 3 f16 GGUF files")
    p.add_argument("--model", required=True, help="directory with model.safetensors, or the file itself")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--type", default="f16", choices=["f16", "f32"])
    p.add_argument("--tokenizer", required=True, help="paligemma_tokenizer.model (SentencePiece)")
    p.add_argument("--pi05", action="store_true", help="pi0.5 checkpoint (adaRMS action expert); outputs pi05-*.gguf")
    p.add_argument("--round-bf16", action="store_true", help="round F32 source tensors to bf16 (nearest even)")
    p.add_argument("--legacy-tokenizer-kv", action="store_true", help="omit tokenizer.ggml.add_space_prefix")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("quantize-expert", help="action expert -> Q4_0")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("qtype", choices=["q4_0"])
    p.add_argument("--fuse-qkv", action="store_true")
    p.add_argument("--qkv-first", action="store_true", help="write the attn_qkv tensors first")
    p.set_defaults(fn=cmd_quantize_expert)

    p = sub.add_parser("fuse-qkv", help="attn_q/k/v -> attn_qkv")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--qkv-first", action="store_true", help="write the attn_qkv tensors first")
    p.set_defaults(fn=cmd_fuse_qkv)

    p = sub.add_parser("fuse-gateup", help="ffn_gate/up -> ffn_gateup (32-row interleave)")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(fn=cmd_fuse_gateup)

    p = sub.add_parser("split-k", help="split tensors along K")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("names", nargs="+", metavar="NAME[:N]", help="ffn_down (every blk.N.ffn_down.weight) or a tensor name")
    p.set_defaults(fn=cmd_split_k)

    p = sub.add_parser("pad-mmproj", help="zero-pad SigLIP n_ff")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--align", type=int, default=32)
    p.set_defaults(fn=cmd_pad_mmproj)

    p = sub.add_parser("heads16", help="Q4_0 expert with F16 action heads")
    p.add_argument("q4")
    p.add_argument("f16")
    p.add_argument("dst")
    p.set_defaults(fn=cmd_heads16)

    p = sub.add_parser("prepare", help="download, convert, transform, link sets/, check md5")
    p.add_argument("--models", required=True, help="models directory (nothing is written outside it)")
    p.add_argument("--heads", default="hf16", choices=["hf16", "q4", "both"])
    p.add_argument("--model", default="all", choices=["pi0", "pi05", "all"])
    p.add_argument("--pi0-safetensors", help="local lerobot/pi0_base model.safetensors (default /hf/pi0.safetensors)")
    p.add_argument("--pi05-safetensors", help="local lerobot/pi05_base model.safetensors (default /hf/pi05.safetensors)")
    p.add_argument("--llama-quantize", help="default: <out>/host/llama-quantize next to the gguf-py in use")
    p.set_defaults(fn=cmd_prepare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
