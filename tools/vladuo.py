#!/usr/bin/env python3
"""VLA-Duo cut model, profile builder and result summaries (stdlib only; matplotlib only for `summary fig6`).

  rates     phase-level co-run rates and the six SigLIP-side pairs (OPMAT_VIT) of a profile
  select    executable cut chosen by replay (--fine: all-atom search)
  curve     model curve over every cut
  kcut      exact-k two-in-flight (2T) k-cut DP -> --plan / --plan-order
  check     reproduce the shipped reference values (no device)
  opmat     merge raw llama-pi0-opmat-bench chunk logs into a ratio table
  htp-agg   aggregate a GGML_HEXAGON_PROFILE=1 htp_profiling.csv per operator (--w-npu: W_NPU weights)
  profile   build profile.json from a `vladuo.sh profile quick` directory
  promote   add a profile.json to data/profiles.json under a new key
  summary   burst | table2 | table3 | table4 | table6 | kcut | fig6 | sustain

Profiles: data/profiles.json maps "<soc>/<kernel config>" to a profile; --profile FILE or --soc S [--cfg C] selects
one (default cfg: sd8elite -> default, sd8gen5 -> noxmem).
"""
import argparse, copy, csv, glob, hashlib, io, json, math, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, '..', 'data', 'profiles.json')
SHIPPED = ('sd8elite/default', 'sd8gen5/noxmem')
DEFAULT_CFG = {'sd8elite': 'default', 'sd8gen5': 'noxmem'}
MODELS = ('pi0', 'pi05')
PH = ('vit', 'prefix', 'fm')

# atom layout: patch, 27 SigLIP layers, projector, 18 PaliGemma layers, 10 flow-matching steps (layer level, m = 57).
# The fine layout splits each SigLIP / LM layer into 6 atoms and each step into 18 expert layers x 6 (m = 1352).
N_VIT, N_LM, N_FM = 27, 18, 10
M_LAYER = 1 + N_VIT + 1 + N_LM + N_FM
L0, L1 = 29, 47
ATOMS = ['LN1', 'QKV', 'ATTN', 'O', 'LN2', 'FFN']
LM_ATOMS = ['RMS1', 'QKV', 'ATTN', 'O', 'RMS2', 'FFN']
OP_OF = {'LN1': 'vis_ln', 'QKV': 'vis_qkvo', 'ATTN': 'vis_attn', 'O': 'vis_qkvo', 'LN2': 'vis_ln', 'FFN': 'vis_ff_up'}
W_NPU_KEYS = ['vis_qkvo', 'vis_ff_up', 'vis_ff_dn', 'vis_attn', 'vis_ln', 'vis_gelu', 'pre_qkv', 'pre_o', 'pre_ff_up',
              'pre_ff_dn', 'pre_attn', 'pre_geglu', 'pre_rmsn', 'pre_rope', 'fm_qkv', 'fm_o', 'fm_ff_up', 'fm_ff_dn',
              'fm_attn', 'fm_geglu']
# QUICK GPU SigLIP weights (27 layers x 3 views per frame): calls per layer-view, cGFlop per call (opmat bench --list)
QUICK_CALLS = {'vis_qkvo': 4, 'vis_ff_up': 1, 'vis_ff_dn': 1, 'vis_attn': 1}
QUICK_CGF = {'vis_qkvo': 0.679, 'vis_ff_up': 2.548, 'vis_ff_dn': 2.548, 'vis_attn': 0.302}
# FULL atoms (20 per device) and per-frame call counts (GPU, NPU)
FULL_ATOMS = ['v_patch', 'v_norm', 'v_qkv', 'v_attn', 'v_out', 'v_ffn_up', 'v_ffn_dn', 'v_proj', 'p_norm', 'p_qkv',
              'p_attn', 'p_out', 'p_ffn_up', 'p_ffn_dn', 'e_norm', 'e_qkv', 'e_attn', 'e_out', 'e_ffn_up', 'e_ffn_dn']
FULL_COUNTS = dict(v_patch=(3, 1), v_proj=(3, 1), v_norm=(162, 54), **{k: (81, 27) for k in (
    'v_qkv', 'v_attn', 'v_out', 'v_ffn_up', 'v_ffn_dn')}, p_norm=(37, 37), **{k: (18, 18) for k in (
    'p_qkv', 'p_attn', 'p_out', 'p_ffn_up', 'p_ffn_dn')}, e_norm=(370, 370), **{k: (180, 180) for k in (
    'e_qkv', 'e_attn', 'e_out', 'e_ffn_up', 'e_ffn_dn')})
FULL_OP = {'LN1': ['v_norm'], 'LN2': ['v_norm'], 'QKV': ['v_qkv'], 'ATTN': ['v_attn'], 'O': ['v_out'],
           'FFN': ['v_ffn_up', 'v_ffn_dn']}
# W_NPU mapping: htp-agg table rows (phase, HTP op, W or None = any) summed per opmat name (pi0 shapes)
W_MAP = [('vis_qkvo', 'Vision', 'MUL_MAT+ADD', '1152:1152'), ('vis_ff_up', 'Vision', 'MUL_MAT+ADD', '1152:4320'),
         ('vis_ff_dn', 'Vision', 'MUL_MAT+ADD', '4320:1152'), ('vis_attn', 'Vision', 'FLASH_ATTN_EXT', None),
         ('vis_ln', 'Vision', 'NORM+MUL+ADD', None), ('vis_ln', 'Vision', 'ADD', '1152:768'),
         ('vis_gelu', 'Vision', 'GELU', None), ('pre_qkv', 'Prefix', 'MUL_MAT', '2048:2560'),
         ('pre_o', 'Prefix', 'MUL_MAT+ADD', '2048:2048'), ('pre_ff_up', 'Prefix', 'MUL_MAT+GEGLU+CPY', None),
         ('pre_ff_dn', 'Prefix', 'MUL_MAT+ADD', '4096:2048'), ('pre_attn', 'Prefix', 'FLASH_ATTN_EXT', None),
         ('pre_rmsn', 'Prefix', 'RMS_NORM+MUL', None), ('pre_rope', 'Prefix', 'ROPE', None),
         ('fm_qkv', 'FM', 'MUL_MAT', '1024:2560'), ('fm_o', 'FM', 'MUL_MAT+ADD', '2048:1024'),
         ('fm_ff_up', 'FM', 'MUL_MAT+GEGLU+CPY', None), ('fm_ff_dn', 'FM', 'MUL_MAT+ADD', '4096:1024'),
         ('fm_attn', 'FM', 'FLASH_ATTN_EXT', None)]
OTHER = {'gpu': 'npu', 'npu': 'gpu'}


def die(msg):
    sys.exit(f'vladuo.py: {msg}')


def phase(n):
    """phase of an opmat name (vis_/pre_/fm_ QUICK, v_/p_/e_ FULL)"""
    return 'vit' if n.startswith('v') else ('prefix' if n.startswith('p') else 'fm')


# ---------------------------------------------------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------------------------------------------------
def dump_json(o, ind=0):
    """JSON with one line per innermost object (diff-friendly)."""
    pad = ' ' * ind
    deep = isinstance(o, dict) and any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()) for v in o.values())
    if deep or (isinstance(o, dict) and any(isinstance(v, dict) for v in o.values()) and len(json.dumps(o)) > 100):
        f = (lambda v: dump_json(v, ind + 1).lstrip()) if deep else json.dumps
        return pad + '{\n' + ',\n'.join(f'{pad} {json.dumps(k)}: {f(v)}' for k, v in o.items()) + '\n' + pad + '}'
    return pad + json.dumps(o)


def load_profiles(path=None):
    with open(path or PROFILES) as f:
        return json.load(f)


def resolve(a, need=True):
    """profile from --profile FILE or --soc/--cfg. Returns (key, profile)."""
    if getattr(a, 'profile', None):
        d = load_profiles(a.profile)
        if 'soc' in d: return os.path.basename(a.profile), d
        if len(d) == 1: return next(iter(d.items()))
        die(f'{a.profile} holds several profiles; use --soc/--cfg with --profiles')
    soc = getattr(a, 'soc', None) or 'sd8elite'
    cfg = getattr(a, 'cfg', None) or DEFAULT_CFG.get(soc, 'default')
    d, key = load_profiles(getattr(a, 'profiles', None)), f'{soc}/{cfg}'
    if key not in d:
        if not need: return key, None
        hint = ' (run `vladuo.sh profile quick` + `vladuo.sh promote`, or pass --cfg noxmem)' if key == 'sd8gen5/xmem' else ''
        die(f'no profile {key} in data/profiles.json{hint}; have: {", ".join(d)}')
    return key, d[key]


def validate(p, key=None):
    """profile schema check. Returns a list of problems."""
    err = []
    def num(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0
    if p.get('soc') not in ('sd8elite', 'sd8gen5'): err.append('soc')
    if p.get('kernel_config') not in ('default', 'noxmem', 'xmem'): err.append('kernel_config')
    if key and key != f"{p.get('soc')}/{p.get('kernel_config')}": err.append(f'key {key} != soc/kernel_config')
    for k in ('desc',): err += [] if isinstance(p.get(k), str) else [k]
    if not isinstance(p.get('provenance'), dict): err.append('provenance')
    for k in ('tau_ms', 'vit_fixed_ms', 'npu_patch_ms', 'handoff_ms', 'handoff_deep_ms'):
        if not num(p.get(k)) or not p.get(k): err.append(k)
    for m in MODELS:
        mm = (p.get('models') or {}).get(m) or {}
        for dev in ('gpu', 'npu'):
            for ph in ('vision', 'prefix', 'fm'):
                if not num(((mm.get('solo') or {}).get(dev) or {}).get(ph)): err.append(f'models.{m}.solo.{dev}.{ph}')
        for k in ('npu_stage47', 'gpu_stage47', 'T46'):
            if not num((mm.get('deep') or {}).get(k)): err.append(f'models.{m}.deep.{k}')
    w = p.get('w_npu') or {}
    if set(w) != set(W_NPU_KEYS) or not all(num(v) for v in w.values()): err.append('w_npu (20 keys)')
    s = p.get('sh_gpu') or {}
    if set(s) != set(ATOMS) or abs(sum(s.values()) - 1.0) > 1e-6: err.append('sh_gpu')
    if not isinstance(p.get('w_g_override', {}), dict): err.append('w_g_override')
    om = p.get('opmat') or {}
    if om.get('set') != 'quick' or not num(om.get('rep_ms')): err.append('opmat.set/rep_ms')
    for mk in ('g', 'h'):
        rows = om.get(mk)
        if not isinstance(rows, dict) or not rows or not all(isinstance(r, dict) and 'cells' in r and 'idle' in r for r in rows.values()):
            err.append(f'opmat.{mk}')
    if p.get('opmat_full') is not None and not isinstance(p.get('opmat_full'), dict): err.append('opmat_full')
    return err


# ---------------------------------------------------------------------------------------------------------------------
# op matrix: merge, ratio table, aggregation
# ---------------------------------------------------------------------------------------------------------------------
def _cell(v):
    return v == 'n/a' or re.match(r'^-?[0-9.]+(e[-+]?\d+)?$', v) is not None


def opmat_chunks(paths):
    """raw bench logs -> list of chunks [{'g': {row: {col: GFLOPS, 'idle': x}}, 'h': ..., 'solo': ..., 'cgf': ...}]"""
    out = []
    for path in paths:
        with open(path, errors='replace') as f:
            text = f.read()
        for part in re.split(r'(?m)^#####.*$', text):
            ch, cur, header = {'g': {}, 'h': {}, 'solo': {}, 'cgf': {}}, None, None
            for s in part.split('\n'):
                if s.startswith('=== g_ij'): cur, header = 'g', None; continue
                if s.startswith('=== h_ij'): cur, header = 'h', None; continue
                if s.startswith('=== solo'): cur = 'solo'; continue
                if s.startswith('=== kernel'): cur = 'kern'; continue
                if not s.strip(): continue
                if s.lstrip().startswith('fg \\ bg'): header = s.split()[3:]; continue
                f = s.split()
                if cur == 'kern':
                    m = re.match(r'^([a-z_]+)\s.*?\s(\d*\.\d+)\s+(?:\d+\s+)?\[(gpu|npu)\]\s*$', s)
                    if m: ch['cgf'][(m.group(3), m.group(1))] = float(m.group(2))
                elif cur == 'solo' and re.match(r'^[a-z_]+$', f[0]) and len(f) >= 3 and f[0] != 'type':
                    d = ch['solo'].setdefault(f[0], {})
                    if f[1] != 'n/a': d['gpu'] = float(f[1])
                    if f[2] != 'n/a': d['npu'] = float(f[2])
                elif cur in ('g', 'h') and header and re.match(r'^[a-z_]+$', f[0]) and len(f) > 1 and all(map(_cell, f[1:])):
                    row = ch[cur].setdefault(f[0], {})
                    for col, v in zip(header, f[1:]):
                        if v != 'n/a': row[col] = float(v)
            if ch['g'] or ch['h'] or ch['solo']: out.append(ch)
    return out


def opmat_merge(chunks):
    """merge the chunks of one run per row. For a repeated cell or idle the last chunk wins."""
    g, h, solo = {}, {}, {}
    for ch in chunks:
        for tgt, src in ((g, ch['g']), (h, ch['h'])):
            for r, cells in src.items(): tgt.setdefault(r, {}).update(cells)
        for k, d in ch['solo'].items(): solo.setdefault(k, {}).update(d)
    return g, h, solo


def ratio_text(g, h, solo, src):
    """ratio-to-idle table: one row per foreground op, its idle GFLOPS, then cell / idle per background op"""
    out = [f'# merged from {src}: G_vision x N_all sub-matrix, ratio to idle\n', '\n=== solo speeds (idle) ===\n',
           f"{'type':<12s}{'GPU':>8s}{'NPU':>8s}\n"]
    for k, d in solo.items(): out.append(f"{k:<12s}{d.get('gpu', float('nan')):8.0f}{d.get('npu', float('nan')):8.0f}\n")
    for name, m in (('g_ij : GPU(OpenCL) vision op i while NPU runs op j', g), ('h_ij : NPU(HTP0) op i while GPU runs vision op j', h)):
        cols = []
        for r in m.values():
            for c in r:
                if c != 'idle' and c not in cols: cols.append(c)
        out.append(f'\n=== {name} -- SLOWDOWN vs idle ===\n')
        hdr = 'fg \\ bg'
        out.append(f"{hdr:<12s}{'idle':>8s}" + ''.join(f'{c:>10s}' for c in cols) + '\n')
        for fg, r in m.items():
            idle = r.get('idle')
            out.append(f"{fg:<12s}{(f'{idle:8.0f}' if idle else '     n/a')}")
            for c in cols:
                v = r.get(c)
                out.append(f"{'n/a':>10s}" if v is None or not idle else f'{v / idle:10.2f}')
            out.append('\n')
    return ''.join(out)


def parse_ratio(text):
    """ratio table -> {'g': {row: {'idle': x, 'cells': {col: ratio}}}, 'h': ...}"""
    out, cur, cols = {'g': {}, 'h': {}}, None, None
    for line in text.split('\n'):
        if line.startswith('=== g_ij'): cur, cols = out['g'], None; continue
        if line.startswith('=== h_ij'): cur, cols = out['h'], None; continue
        if cur is None: continue
        t = line.split()
        if not t: continue
        if t[0] == 'fg': cols = t[4:]; continue
        if cols and re.match(r'^[a-z_]+$', t[0]) and len(t) > 2:
            idle = None if t[1] == 'n/a' else float(t[1])
            cur[t[0]] = {'idle': int(idle) if idle is not None and idle == int(idle) else idle,
                         'cells': {c: float(v) for c, v in zip(cols, t[2:]) if v != 'n/a'}}
    return out


def opmat_vit(prof):
    """QUICK phase pairs G[vit|b] (GPU rows over the keep-alive cell, clamp once) and N[a|vit] (idle-normalised)"""
    om, wn = prof['opmat'], prof['w_npu']
    g = {r: dict(v['cells'], idle=v['idle']) for r, v in om['g'].items()}
    h = {r: dict(v['cells'], idle=v['idle']) for r, v in om['h'].items()}
    wg = {k: 81 * QUICK_CALLS[k] * QUICK_CGF[k] / g[k]['idle'] * 1000.0 for k in QUICK_CALLS
          if g.get(k, {}).get('idle') and (k != 'vis_attn' or 'vis_attn' not in prof.get('w_g_override', {}))}
    wg.update(prof.get('w_g_override', {}))

    def agg(mat, fg_w, bg_w, fp, bp, keepalive):
        num = den = 0.0
        for fg, r in mat.items():
            if phase(fg) != fp or fg not in fg_w: continue
            base = (r.get('fm_attn') or max(v for c, v in r.items() if c != 'idle')) if keepalive else 1.0
            n2 = d2 = 0.0
            for bg, v in r.items():
                if bg == 'idle' or phase(bg) != bp or bg not in bg_w: continue
                n2 += bg_w[bg] * v / base; d2 += bg_w[bg]
            if d2 > 0: num += fg_w[fg] * n2 / d2; den += fg_w[fg]
        return num / den if den else float('nan')
    out = {f'G_vit|{b}': min(1.0, agg(g, wg, wn, 'vit', b, True)) for b in PH}
    out.update({f'N_{a}|vit': agg(h, wn, wg, a, 'vit', False) for a in PH})
    return out


def full_rates(full, Gq, Nq):
    """FULL matrix -> phase pairs and op-level SigLIP overlay. A missing (atom, phase) takes the QUICK value."""
    cnt, cgf = FULL_COUNTS, full['cgf']
    hrows = full['h']
    gidle = {}
    for ch in full['g']: gidle[ch['row']] = ch['idle']
    wG = {i: cnt[i][0] * cgf['gpu'][i] / gidle[i] for i in gidle if gidle[i] and i in cgf['gpu']}
    wN = {j: cnt[j][1] * cgf['npu'][j] / r['idle'] for j, r in hrows.items() if r.get('idle') and j in cgf['npu']}
    acc = {}
    for ch in full['g']:
        c = ch['cells']
        if not c: continue
        base = c.get('e_attn') or max(c.values())
        for j, v in c.items():
            if j in wN: acc.setdefault((ch['row'], phase(j)), {})[j] = v / base
    Gi = {}
    for (i, b), d in acc.items():
        den = sum(wN[j] for j in d)
        if den: Gi[(i, b)] = min(1.0, sum(wN[j] * v for j, v in d.items()) / den)
    Nj = {}
    for j, r in hrows.items():
        for b in PH:
            terms = [(wG[i], v / r['idle']) for i, v in r['cells'].items() if i in wG and phase(i) == b and r.get('idle')]
            den = sum(w for w, _ in terms)
            if den: Nj[(j, b)] = sum(w * x for w, x in terms) / den

    def wmean(pairs):
        den = sum(w for w, _ in pairs)
        return sum(w * x for w, x in pairs) / den if den else None
    G, N = dict(Gq), dict(Nq)
    for a in PH:
        for b in PH:
            v = wmean([(wG[i], Gi[(i, b)]) for i in wG if phase(i) == a and (i, b) in Gi])
            if v is not None: G[(a, b)] = v
            v = wmean([(wN[j], Nj[(j, b)]) for j in wN if phase(j) == a and (j, b) in Nj])
            if v is not None: N[(a, b)] = v
    Gf, Nf = {}, {}
    for a in ATOMS:
        for b in PH:
            v = wmean([(wG.get(i, 0), Gi[(i, b)]) for i in FULL_OP[a] if (i, b) in Gi])
            Gf[(a, b)] = v if v is not None else G[('vit', b)]
            terms = []
            for j in wN:
                if phase(j) != b: continue
                x = wmean([(wG.get(i, 0), hrows[j]['cells'][i] / hrows[j]['idle']) for i in FULL_OP[a] if i in hrows[j]['cells']])
                if x is not None: terms.append((wN[j], x))
            v = wmean(terms)
            Nf[(b, a)] = v if v is not None else N[(b, 'vit')]
    return G, N, Gf, Nf


def parse_full(paths, model='pi0'):
    """FULL chunk logs -> opmat_full dict (g kept per chunk for the keep-alive normalisation)"""
    chunks = opmat_chunks(paths)
    g, cgf, na = [], {'gpu': {}, 'npu': {}}, []
    for ch in chunks:
        for (dev, k), v in ch['cgf'].items(): cgf[dev][k] = v
        for r, cells in ch['g'].items():
            g.append({'row': r, 'idle': cells.get('idle'), 'cells': {c: v for c, v in cells.items() if c != 'idle'}})
    hm = opmat_merge(chunks)[1]
    h = {r: {'idle': c.get('idle'), 'cells': {k: v for k, v in c.items() if k != 'idle'}} for r, c in hm.items()}
    for i in FULL_ATOMS:
        for j in FULL_ATOMS:
            if not any(x['row'] == i and j in x['cells'] for x in g): na.append(f'g:{i}:{j}')
            if j not in h.get(i, {}).get('cells', {}): na.append(f'h:{i}:{j}')
    return {'set': 'full', 'model': model, 'g': g, 'h': h, 'cgf': cgf, 'na': na}


# ---------------------------------------------------------------------------------------------------------------------
# cut model (single cut)
# ---------------------------------------------------------------------------------------------------------------------
class One(dict):
    def __missing__(self, k): return 1.0


class Rates(dict):
    """G->N co-run rates: op-level SigLIP atoms fall back to their phase-level pair"""
    def __init__(self, base, fine): super().__init__(base); self.fine = fine

    def __missing__(self, k):
        a, p = k
        pa = 'vit' if a in ATOMS else a; pp = 'vit' if p in ATOMS else p
        if (a, pp) in self.fine: return self.fine[(a, pp)]
        if (pa, p) in self.fine: return self.fine[(pa, p)]
        return dict.get(self, (pa, pp), 1.0)


class Ph(dict):
    """N->G co-run rates: phase-level through each label's phase"""
    def __missing__(self, k):
        a, p = k
        pa = 'vit' if a in ATOMS else ('prefix' if a.startswith('pre:') else ('fm' if a.startswith('fm:') else a))
        pp = 'vit' if p in ATOMS else ('prefix' if p.startswith('pre:') else ('fm' if p.startswith('fm:') else p))
        return dict.get(self, (pa, pp), 1.0)


def simulate(seq, p, direction, G, N, tau):
    """Eager work-conserving replay of one period. Stream A = atoms [0,p) of the new inference on the head device.
    Stream B = atoms [p,m) of the previous one on the other device. Returns (period, tA, tB) in quanta of tau."""
    d1, d2 = ('gpu', 'npu') if direction == 'G>N' else ('npu', 'gpu')
    A = [(ph, g if d1 == 'gpu' else n) for (ph, g, n) in seq[:p]]
    B = [(ph, g if d2 == 'gpu' else n) for (ph, g, n) in seq[p:]]
    rem = [list(x) for x in A], [list(x) for x in B]
    dev = [d1, d2]
    done = [0.0, 0.0]; q = 0
    while rem[0] or rem[1]:
        q += 1
        active = [rem[i][0][0] if rem[i] else None for i in range(2)]
        for i in range(2):
            if not rem[i]: continue
            peer = active[1 - i]
            if peer is None: rate = 1.0
            elif dev[i] == 'gpu': rate = G[(active[i], peer)]
            else: rate = N[(active[i], peer)]
            budget = rate * tau
            while budget > 1e-12 and rem[i]:
                take = min(budget, rem[i][0][1])
                rem[i][0][1] -= take; budget -= take
                if rem[i][0][1] <= 1e-9: rem[i].pop(0)
            if not rem[i] and done[i] == 0.0: done[i] = q
    return q, done[0], done[1]


def layer_p(c):
    """layer-level cut of the boundary at fine atom c (handoff term)"""
    if c <= 0: return 0
    if c <= 163: return 1 + (c - 1) // 6
    if c == 164: return 29
    if c < 272: return 29 + (c - 164) // 6
    if c < 1352: return 47 + (c - 272) // 108
    return 57


def p_fine_of(p):
    """fine atom index of the layer-level cut p"""
    if p <= 0: return 0
    if p <= 1 + N_VIT: return 1 + 6 * (p - 1)
    if p == 1 + N_VIT + 1: return 164
    if p <= 1 + N_VIT + 1 + N_LM: return 164 + 6 * (p - 29)
    return 272 + 108 * (p - 47)


def executable():
    """(dir, fine atom c, layer p, runtime args, cut label, kind) for every cut llama-pi0-pipeline runs, + references"""
    out = [('G>N', 1 + 6 * L, L + 1, f'--vit-split {L}', f'GNp{L + 1}', 'duo') for L in range(1, 28)]
    out.append(('G>N', 164, 29, '', 'GNp29', 'duo'))
    out += [('N>G', 1 + 6 * L, L + 1, f'--direction NG --vit-split {L}', f'NGp{L + 1}', 'duo') for L in range(1, 28)]
    out += [('N>G', 164 + 6 * K, 29 + K, f'--direction NG --lm-split {K}', f'NGp{29 + K}', 'duo') for K in range(1, 18)]
    out += [('N>G', 272 + 108 * J, 47 + J, f'--direction NG --fm-split {J}', f'NGp{47 + J}', 'duo') for J in range(10)]
    out += [('N>G', 272 + 6 * j, 47, f'--direction NG --fm-layer-split {j}', f'NGfl{j}', 'duo') for j in range(1, 18)]
    out += [('G>N', 0, 0, 'NPU-only (llama-pi0-bench)', 'npu', 'single'), ('G>N', 1352, 57, 'GPU-only (llama-pi0-bench)', 'gpu', 'single')]
    return out


class Cell:
    """one (profile, model): atom sequences and calibrated co-run rates"""

    def __init__(self, prof, model, corun='quick'):
        self.prof, self.model, self.corun = prof, model, corun
        self.solo = prof['models'][model]['solo']
        self.deep = prof['models'][model]['deep']
        self.tau = float(prof['tau_ms'])
        self.vf = float(prof['vit_fixed_ms'])
        wn = prof['w_npu']
        npu_raw = {'LN1': wn['vis_ln'] / 2, 'QKV': wn['vis_qkvo'] * 0.75, 'ATTN': wn['vis_attn'], 'O': wn['vis_qkvo'] * 0.25,
                   'LN2': wn['vis_ln'] / 2, 'FFN': wn['vis_ff_up'] + wn['vis_ff_dn'] + wn['vis_gelu']}
        pre_raw = {'RMS1': wn['pre_rmsn'] / 2, 'QKV': wn['pre_qkv'] + wn['pre_rope'], 'ATTN': wn['pre_attn'], 'O': wn['pre_o'],
                   'RMS2': wn['pre_rmsn'] / 2, 'FFN': wn['pre_ff_up'] + wn['pre_ff_dn'] + wn.get('pre_geglu', 0)}
        fm_raw = {'RMS1': 1.0, 'QKV': wn['fm_qkv'], 'ATTN': wn['fm_attn'], 'O': wn['fm_o'], 'RMS2': 1.0,
                  'FFN': wn['fm_ff_up'] + wn['fm_ff_dn'] + wn.get('fm_geglu', 0)}
        self.sh_gpu = prof['sh_gpu']
        self.sh_npu = {k: v / sum(npu_raw.values()) for k, v in npu_raw.items()}
        self.sh_pre = {k: v / sum(pre_raw.values()) for k, v in pre_raw.items()}
        self.sh_fm = {k: v / sum(fm_raw.values()) for k, v in fm_raw.items()}
        G = {(a, b): 1.0 for a in PH for b in PH}; N = dict(G)
        self.note = self.calibrate_deep(G, N)
        self.om = {k: round(v, 3) for k, v in opmat_vit(prof).items()}   # 3 decimals: CURVE_SHA256 depends on it
        for b in PH:
            G[('vit', b)] = self.om[f'G_vit|{b}']; N[(b, 'vit')] = self.om[f'N_{b}|vit']
        self.G, self.N = G, N
        self.Gf, self.Nf = self.rates_for(G, N)
        if corun == 'full':
            if not prof.get('opmat_full'): die('--corun full needs a profile with opmat_full (vladuo.py profile --full)')
            self.G, self.N, self.Gf, self.Nf = full_rates(prof['opmat_full'], G, N)
            self.note = 'matrix-only model: Table 3 inputs change'

    def handoff(self, p):
        """measured handoff of the layer-level cut p: SigLIP/projector cut vs LM/FM cut"""
        if p <= 0 or p >= M_LAYER: return 0.0
        return self.prof['handoff_ms'] if p <= 1 + N_VIT + 1 else self.prof['handoff_deep_ms']

    def atoms(self, fm_sub=1):
        s, vf = self.solo, self.vf
        seq = [('vit', vf * 0.5, vf * 0.5)]
        seq += [('vit', (s['gpu']['vision'] - vf) / N_VIT, (s['npu']['vision'] - vf) / N_VIT)] * N_VIT
        seq += [('vit', vf * 0.5, vf * 0.5)]
        seq += [('prefix', s['gpu']['prefix'] / N_LM, s['npu']['prefix'] / N_LM)] * N_LM
        seq += [('fm', s['gpu']['fm'] / (N_FM * fm_sub), s['npu']['fm'] / (N_FM * fm_sub))] * (N_FM * fm_sub)
        return seq

    def calibrate_deep(self, G, N):
        """deep pairs from the p=47 / p=46 N->G frames. G[prefix|*] inverts T46 over the grid 0.80..1.00."""
        fr, s = self.deep, self.solo
        n_solo = s['npu']['vision'] + s['npu']['prefix'] + self.prof['handoff_deep_ms']
        n_deep = n_solo / fr['npu_stage47']
        g_fm = s['gpu']['fm'] / fr['gpu_stage47']
        for a in PH:
            for b in ('prefix', 'fm'): N[(a, b)] = n_deep
        for b in PH: G[('fm', b)] = g_fm
        seq, best, tau = self.atoms(), (None, 1e9), self.tau
        for i in range(0, 41):
            r = 0.80 + 0.005 * i
            for b in PH: G[('prefix', b)] = r
            P, tA, tB = simulate(seq, 46, 'N>G', G, N, tau)
            T = (P + math.ceil(self.handoff(46) / tau)) * tau
            if abs(T - self.deep['T46']) < best[1]: best = (r, abs(T - self.deep['T46']))
        for b in PH: G[('prefix', b)] = best[0]
        clamp = ' (clamped at the grid bound)' if best[0] in (0.80, 1.0) else ''
        return f'deep pairs: N[*|prefix,fm]={n_deep:.3f} G[fm|*]={g_fm:.3f} G[prefix|*]={best[0]:.3f}{clamp} (|dT46|={best[1]:.1f} ms)'

    def rates_for(self, Gp, Np):
        """op-level SigLIP rates G[(atom, phase)], N[(phase, atom)] from the QUICK matrix"""
        wn, om = self.prof['w_npu'], self.prof['opmat']
        g = {r: v['cells'] for r, v in om['g'].items()}
        h = {r: v['cells'] for r, v in om['h'].items()}
        G, N = {}, {}
        for a in ATOMS:
            row = g.get(OP_OF[a])
            for b in PH:
                if row:
                    base = row.get('fm_attn') or max(v for c, v in row.items() if c != 'idle')
                    num = den = 0.0
                    for bg, v in row.items():
                        if bg == 'idle' or phase(bg) != b or bg not in wn: continue
                        num += wn[bg] * v / base; den += wn[bg]
                    G[(a, b)] = min(1.0, num / den) if den else Gp[('vit', b)]
                else:
                    G[(a, b)] = Gp[('vit', b)]
            for ph in PH:
                num = den = 0.0
                for fg, r in h.items():
                    if phase(fg) != ph or OP_OF[a] not in r or fg not in wn: continue
                    num += wn[fg] * r[OP_OF[a]]; den += wn[fg]
                N[(ph, a)] = num / den if den else Np[(ph, 'vit')]
        return G, N

    def seq_gn(self):
        """G->N sequence: SigLIP layers split in 6 atoms, LM 18 atoms, FM 10 atoms"""
        seq, fine = self.atoms(), []
        vit_idx = [i for i, (ph, g, n) in enumerate(seq) if ph == 'vit']; layer_idx = set(vit_idx[1:1 + N_VIT])
        for i, (ph, g, n) in enumerate(seq):
            if i in layer_idx:
                for a in ATOMS: fine.append((a, g * self.sh_gpu[a], n * self.sh_npu[a]))
            else: fine.append((ph, g, n))
        return fine

    def seq_ng(self):
        """N->G sequence (m = 1352): SigLIP, LM and expert layers split in 6 atoms"""
        fine, n_vit = [], 0
        for (ph, g, n) in self.atoms(18):
            if ph == 'vit':
                n_vit += 1
                if 2 <= n_vit <= 28:
                    for a in ATOMS: fine.append((a, g * self.sh_gpu[a], n * self.sh_npu[a]))
                else: fine.append((ph, g, n))
            elif ph == 'prefix':
                for a in LM_ATOMS: fine.append(('pre:' + a, g * self.sh_pre[a], n * self.sh_pre[a]))
            else:
                for a in LM_ATOMS: fine.append(('fm:' + a, g * self.sh_fm[a], n * self.sh_fm[a]))
        return fine

    def rates(self, variant):
        """(G_gn, N_gn, G_ng, N_ng)"""
        if variant == 'solo': return One(), One(), One(), One()
        return Rates(self.G, self.Gf), Rates(self.N, self.Nf), Ph(self.G), Ph(self.N)

    def evaluate(self, variant='co-run'):
        """every executable cut + the two single-device references"""
        Ggn, Ngn, Gng, Nng = self.rates(variant)
        sgn, sng, tau = self.seq_gn(), self.seq_ng(), self.tau
        rows = []
        for d, c, p, args, cut, kind in executable():
            seq = sgn if d == 'G>N' else sng
            cs = len(seq) if c == 1352 else c
            Pq, tA, tB = simulate(seq, cs, d, *((Ggn, Ngn) if d == 'G>N' else (Gng, Nng)), tau)
            h = 0 if cs in (0, len(seq)) else math.ceil(self.handoff(layer_p(cs)) / tau)
            T = (Pq + h) * tau
            L = T + tB * tau if 0 < cs < len(seq) else (tA + tB) * tau
            rows.append(dict(dir=d, c=c, p=p, args=args, cut=cut, kind=kind, T=T, L=L, tA=tA * tau, tB=tB * tau))
        return rows

    def fine_search(self, variant='co-run'):
        """all-atom search: G->N over c = 1..164, N->G over the window c = 230..302 (first min)"""
        Ggn, Ngn, Gng, Nng = self.rates(variant)
        tau, fine = self.tau, self.seq_gn()
        gn = []
        for c in range(1, 165):
            Pq, tA, tB = simulate(fine, c, 'G>N', Ggn, Ngn, tau)
            T = (Pq + math.ceil(self.handoff(layer_p(c)) / tau)) * tau
            gn.append((c, fine[c - 1][0], T, tA * tau, tB * tau))
        best_gn = min(gn, key=lambda r: r[2])
        fine, ng = self.seq_ng(), []
        for c in range(230, 303):
            Pq, tA, tB = simulate(fine, c, 'N>G', Gng, Nng, tau)
            T = (Pq + math.ceil(self.handoff(layer_p(c)) / tau)) * tau
            ng.append((c, fine[c - 1][0], T, tA * tau, tB * tau))
        return best_gn, min(ng, key=lambda r: r[2]), gn

    def layer_curve(self):
        """layer-level model (m = 57, phase-level rates): every cut of both directions"""
        seq = self.atoms(); m = len(seq); rows = []
        for d in ('G>N', 'N>G'):
            for p in range(0, m + 1):
                if d == 'N>G' and p in (0, m): continue
                P, tA, tB = simulate(seq, p, d, self.G, self.N, self.tau)
                T = (P + math.ceil(self.handoff(p) / self.tau)) * self.tau
                L = T + tB * self.tau if 0 < p < m else (tA + tB) * self.tau
                rows.append((d, p, p_fine_of(p), T, L, tA * self.tau, tB * self.tau))
        return rows


def select(rows):
    """min T_est, ties -> min L_est, then min c, then G->N"""
    duo = sorted((r for r in rows if r['kind'] == 'duo'), key=lambda r: (r['T'], r['L'], r['c'], r['dir'] != 'G>N'))
    best_ng = next(r for r in duo if r['dir'] == 'N>G')
    return duo[0], duo[1], best_ng


def tested_set(rows, sel, solo, best_ng):
    """tested set: p_duo-2 .. p_duo+2 (same family, executable) + p_solo + best N->G"""
    by = {r['cut']: r for r in rows if r['kind'] == 'duo'}
    fam = sel['cut'].rstrip('0123456789')
    out = [f'{fam}{p}' for p in range(sel['p'] - 2, sel['p'] + 3) if f'{fam}{p}' in by and not sel['cut'].startswith('NGfl')]
    for c in (sel['cut'], solo['cut'], best_ng['cut']):
        if c not in out: out.append(c)
    return [c for c in out if c in by]


def short(r):
    return {k: r[k] for k in ('dir', 'p', 'c', 'cut', 'args', 'T', 'L')}


# ---------------------------------------------------------------------------------------------------------------------
# k-cut DP (exact k, two inferences in flight)
# ---------------------------------------------------------------------------------------------------------------------
class Rec:
    """DP record. hf: the head switched device (or started) at its current atom and has not run since.
    lt: atom of the tail's most recent switch or start."""
    __slots__ = ('time', 't', 'ns', 'parent', 'act', 'tfin', 'hf', 'lt')

    def __init__(self, time, t, ns, parent, act, tfin):
        self.time, self.t, self.ns, self.parent, self.act, self.tfin = time, t, ns, parent, act, tfin
        if parent is None:
            self.hf, self.lt = True, int(round(t)); return
        self.hf, self.lt = parent.hf, parent.lt
        if act.startswith('SWAP'):
            self.hf, self.lt = True, int(re.match(r'SWAP at head h\d+/tail a(\d+)', act).group(1))
        elif act.startswith('MOVE head'):
            self.hf = True
        elif act.startswith('MOVE tail'):
            self.lt = int(act.split(' at a')[1].split()[0])
        elif act.startswith(('both', 'head')):
            self.hf = False


def _ok(rec, chain, atom):
    """False if this chain's most recent switch (or its start) happened at the same atom (a no-op switch)"""
    return (not rec.hf) if chain == 'h' else rec.lt != atom


def _tail_flex(r):
    return abs(r.t - r.lt) > 1e-9


def pareto_add(lst, rec, eps_t=1e-6, eps_time=1e-6):
    """keep (time, tail progress) Pareto records. A record dominates only if it is at least as free to switch."""
    def dom(a, b):
        return (a.time <= b.time + eps_time and a.t >= b.t - eps_t and (not a.hf or b.hf)
                and (_tail_flex(a) or not _tail_flex(b)))
    for r in lst:
        if dom(r, rec): return False
    lst[:] = [r for r in lst if not dom(rec, r)]
    lst.append(rec)
    return True


def run_both(seq, h, dh, t, dt, G, N):
    """head atom h on dh while the tail (progress t on dt) runs. Returns (elapsed, new t, tail finish or None)."""
    m = len(seq)
    r_h = seq[h][1] if dh == 'gpu' else seq[h][2]
    elapsed, tail_fin = 0.0, None
    while r_h > 1e-9:
        if t >= m - 1e-9:
            elapsed += r_h; r_h = 0.0; break
        i = int(math.floor(t + 1e-9)); frac = t - i
        a_t = seq[i][1] if dt == 'gpu' else seq[i][2]
        r_t = (1.0 - frac) * a_t
        rh = G[(seq[h][0], seq[i][0])] if dh == 'gpu' else N[(seq[h][0], seq[i][0])]
        rt = G[(seq[i][0], seq[h][0])] if dt == 'gpu' else N[(seq[i][0], seq[h][0])]
        d_h = r_h / rh; d_t = r_t / rt
        step = min(d_h, d_t)
        elapsed += step
        r_h -= step * rh
        if step >= d_t - 1e-12:
            t = float(i + 1)
            if t >= m - 1e-9: tail_fin = elapsed
        else:
            t = i + frac + (step * rt) / a_t
        if d_h <= d_t + 1e-12: r_h = 0.0
    return elapsed, t, tail_fin


def solve_cuts(seq, p, D0, G, N, handoff, max_swaps):
    """min period per number of physical cuts k for cut p, head starting on D0. Idling, SWAP (+2) and MOVE (+1) are
    allowed. A head that ends on D0 adds 1. Returns {k: (T, rec, dh)}."""
    m = len(seq)
    def ms(i, d): return seq[i][1] if d == 'gpu' else seq[i][2]
    states = {(0, D0, OTHER[D0], 0): [Rec(0.0, float(p), 0, None, 'start', None)]}
    best = {}

    def finish(rec, dh):
        T = rec.time + (handoff(p) if dh == D0 else 0.0)
        k = rec.ns + (1 if dh == D0 else 0)
        if k not in best or T < best[k][0] - 1e-9: best[k] = (T, rec, dh)

    def add(key, rec):
        return pareto_add(states.setdefault(key, []), rec)

    for h in range(0, p + 1):
        pending = [k for k in list(states) if k[0] == h]
        seen = set()
        while pending:
            k = pending.pop()
            _, dh, dt, _ns = k
            for rec in list(states.get(k, [])):
                if (k, id(rec)) in seen: continue
                seen.add((k, id(rec)))
                tail_alive = rec.t < m - 1e-9
                if h >= p and not tail_alive:
                    finish(rec, dh); continue
                if h < p:
                    if tail_alive and dt != dh:
                        el, t2, tf = run_both(seq, h, dh, rec.t, dt, G, N)
                        nr = Rec(rec.time + el, t2, rec.ns, rec, f'both h{h}', (rec.time + tf) if tf is not None else rec.tfin)
                        add((h + 1, dh, dt, nr.ns), nr)
                    else:
                        nr = Rec(rec.time + ms(h, dh), rec.t, rec.ns, rec, f'head h{h} alone', rec.tfin)
                        add((h + 1, dh, dt, nr.ns), nr)
                    if tail_alive and dt != dh:
                        nr = Rec(rec.time + ms(h, dh), rec.t, rec.ns, rec, f'head h{h} alone(tail idle)', rec.tfin)
                        add((h + 1, dh, dt, nr.ns), nr)
                        i = int(math.floor(rec.t + 1e-9)); frac = rec.t - i
                        el = (1.0 - frac) * ms(i, dt); t2 = float(i + 1)
                        nr = Rec(rec.time + el, t2, rec.ns, rec, f'tail a{i} alone(head idle)', (rec.time + el) if t2 >= m - 1e-9 else rec.tfin)
                        if add((h, dh, dt, nr.ns), nr) and k not in pending: pending.append(k)
                    if tail_alive and dt != dh and rec.ns + 2 <= max_swaps and h > 0:
                        i = int(math.floor(rec.t + 1e-9)); frac = rec.t - i
                        el = (1.0 - frac) * ms(i, dt) if frac > 1e-9 else 0.0
                        i2 = i + 1 if frac > 1e-9 else i
                        if i2 < m and _ok(rec, 'h', h) and _ok(rec, 't', i2):
                            cost = handoff(h) + handoff(i2)
                            nr = Rec(rec.time + el + cost, float(i2), rec.ns + 2, rec, f'SWAP at head h{h}/tail a{i2} (+{cost:.1f} ms)', rec.tfin)
                            nk = (h, dt, dh, nr.ns)
                            if add(nk, nr) and nk not in pending: pending.append(nk)
                    if not tail_alive and h > 0 and dt != dh and rec.ns < max_swaps and _ok(rec, 'h', h):
                        cost = handoff(h)
                        nr = Rec(rec.time + cost, rec.t, rec.ns + 1, rec, f'MOVE head to {dt} at h{h} (+{cost:.1f} ms)', rec.tfin)
                        nk = (h, dt, dh, nr.ns)
                        if add(nk, nr) and nk not in pending: pending.append(nk)
                else:
                    i = int(math.floor(rec.t + 1e-9)); frac = rec.t - i
                    el = (1.0 - frac) * ms(i, dt) + sum(ms(j, dt) for j in range(i + 1, m))
                    finish(Rec(rec.time + el, float(m), rec.ns, rec, f'tail a{i}..{m - 1} alone on {dt}', rec.time + el), dh)
                    if rec.ns < max_swaps:
                        el1 = (1.0 - frac) * ms(i, dt) if frac > 1e-9 else 0.0
                        i2 = i + 1 if frac > 1e-9 else i
                        if i2 < m and (frac > 1e-9 or i2 == i):
                            j2 = i2 + (0 if frac > 1e-9 else 1)
                            if j2 <= m - 1:
                                el2 = el1 + (ms(i2, dt) if frac <= 1e-9 else 0.0)
                                nr2 = Rec(rec.time + el2, float(j2), rec.ns, rec, f'tail a{i}..{j2 - 1} alone on {dt}', rec.tfin)
                                if add(k, nr2) and k not in pending: pending.append(k)
                        if i2 < m and dt != dh and _ok(rec, 't', i2):
                            cost = handoff(i2)
                            nr = Rec(rec.time + el1 + cost, float(i2), rec.ns + 1, rec, f'MOVE tail to {dh} at a{i2} (+{cost:.1f} ms)', rec.tfin)
                            nk = (h, dh, dh, nr.ns)
                            if add(nk, nr) and nk not in pending: pending.append(nk)
    return best


def events(rec):
    out = []
    while rec is not None: out.append(rec.act); rec = rec.parent
    return out[::-1]


def describe(rec):
    acts = [a for a in events(rec) if 'SWAP' in a or 'MOVE' in a or 'idle' in a]
    return ' ; '.join(acts) if acts else 'eager single cut (both chains always running)'


def convert(seq, p, D0, rec):
    """DP record -> --plan segments (split at device changes and at p) + per-device chronological order
    (head = frame offset 0, tail = -1)"""
    m = len(seq)
    dh, dt = D0, OTHER[D0]
    sw_h, sw_t = [(0, D0)], [(p, OTHER[D0])]
    chron = [('h', 0, D0), ('t', p, OTHER[D0])]
    for a in events(rec):
        mm = re.match(r'SWAP at head h(\d+)/tail a(\d+)', a)
        if mm:
            h, i2 = int(mm.group(1)), int(mm.group(2))
            dh, dt = dt, dh
            sw_h.append((h, dh)); sw_t.append((i2, dt)); chron += [('h', h, dh), ('t', i2, dt)]
            continue
        mm = re.match(r'MOVE head to (\w+) at h(\d+)', a)
        if mm:
            dh = mm.group(1); h = int(mm.group(2)); sw_h.append((h, dh)); chron.append(('h', h, dh)); continue
        mm = re.match(r'MOVE tail to (\w+) at a(\d+)', a)
        if mm:
            dt = mm.group(1); i2 = int(mm.group(2)); sw_t.append((i2, dt)); chron.append(('t', i2, dt))
    dev = [None] * m
    for lst, lo, hi in ((sw_h, 0, p), (sw_t, p, m)):
        for j, (a0, d) in enumerate(lst):
            a1 = lst[j + 1][0] if j + 1 < len(lst) else hi
            for x in range(max(a0, lo), min(a1, hi)): dev[x] = d
    starts = sorted(set([0, p] + [a for a, _ in sw_h] + [a for a, _ in sw_t]))
    starts = [x for x in starts if x < m]
    segs = [[dev[a0], a0, (starts[i + 1] if i + 1 < len(starts) else m)] for i, a0 in enumerate(starts)]
    for d, a0, a1 in segs: assert all(dev[x] == d for x in range(a0, a1)), (a0, a1)
    idx = {a0: i for i, (_, a0, _) in enumerate(segs)}
    order = {'gpu': [], 'npu': []}
    for ch, a, d in chron:
        if a < m: order[d].append((idx[a], 0 if a < p else -1))
    return segs, order


def name_of(segs):
    out = []
    for d, a0, a1 in segs:
        v = max(0, min(a1, L0) - a0); l = max(0, min(a1, L1) - max(a0, L0)); f = max(0, a1 - max(a0, L1))
        out.append(('G' if d == 'gpu' else 'N') + ''.join(x for x in (f'{v}V' if v else '', f'{l}L' if l else '', f'{f}F' if f else '') if x))
    return ','.join(out)


def sim_order(segs, order, seq, F=40):
    """static order with the cycle barrier and solo segment times. Returns (period over frames 10..30, L/T), or None
    on deadlock."""
    ms = [sum((seq[x][1] if d == 'gpu' else seq[x][2]) for x in range(a0, a1)) for d, a0, a1 in segs]
    S = len(segs); done = [[None] * F for _ in range(S)]; start = [None] * F
    devs = [d for d in order if order[d]]
    free = {d: 0.0 for d in devs}; ptr = {d: (0, 0) for d in devs}; cend = {d: {} for d in devs}
    minoff = {d: min(o for _, o in order[d]) for d in devs}; fin = {d: False for d in devs}
    while not all(fin.values()):
        prog = False
        for d in devs:
            if fin[d]: continue
            p, i = ptr[d]
            if p + minoff[d] >= F: fin[d] = True; prog = True; continue
            if i == 0 and p > 0:
                if any((p - 1) not in cend[o] and not fin[o] for o in devs): continue
                free[d] = max([free[d]] + [cend[o][p - 1] for o in devs if (p - 1) in cend[o]])
            s, off = order[d][i]; f = p + off
            nxt = (p, i + 1) if i + 1 < len(order[d]) else (p + 1, 0)
            if 0 <= f < F:
                if s > 0 and done[s - 1][f] is None: continue
                t0 = max(free[d], done[s - 1][f] if s > 0 else 0.0)
                done[s][f] = t0 + ms[s]; free[d] = done[s][f]
                if s == 0: start[f] = t0
            if nxt[1] == 0: cend[d][p] = free[d]
            ptr[d] = nxt; prog = True
        if not prog: return None
    c = done[S - 1]
    T = (c[30] - c[10]) / 20
    return T, sum(c[f] - start[f] for f in range(10, 30)) / 20 / T


def kcut_plans(prof, model, kmax=9):
    """[(k, T_dp, plan, order, gpu_lm, info)] for k = 1..kmax with the canonical phase-level rates"""
    cell = Cell(prof, model)
    seq = cell.atoms()
    seq[0] = (seq[0][0], seq[0][1], float(prof['npu_patch_ms']))
    best = {}
    for D0 in ('gpu', 'npu'):
        for p in range(1, len(seq)):
            for k, (T, rec, dh) in solve_cuts(seq, p, D0, cell.G, cell.N, cell.handoff, kmax).items():
                if 1 <= k <= kmax and (k not in best or T < best[k][0]): best[k] = (T, D0, p, rec)
    out = []
    for k in sorted(best):
        T, D0, p, rec = best[k]
        segs, order = convert(seq, p, D0, rec)
        ospec = 'c=1;sync;G:' + ','.join(f'{s}/{o}' for s, o in order['gpu']) + ';N:' + ','.join(f'{s}/{o}' for s, o in order['npu'])
        gpu_lm = int(any(d == 'gpu' and a1 > L0 for d, a0, a1 in segs))
        r = sim_order(segs, order, seq)
        info = f"p={p} D0={D0} cuts={len(segs) - 1} sim_solo={'DEADLOCK' if r is None else f'{r[0]:.1f} L/T {r[1]:.2f}'} | {describe(rec)}"
        out.append((k, T, name_of(segs), ospec, gpu_lm, info))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# HTP per-op profile (GGML_HEXAGON_PROFILE=1)
# ---------------------------------------------------------------------------------------------------------------------
def _ne(tok):
    try: return [int(x) for x in tok.strip().split(':')]
    except ValueError: return []


def _dims(d):
    if '->' not in d: return [], []
    ins, out = d.split('->')
    return [_ne(t) for t in ins.split(' x ') if t.strip()], _ne(out)


def _prod(v):
    p = 1
    for d in v: p *= d
    return p


def _flop(op, dims):
    ins, out = _dims(dims)
    if not ins or not out: return 0
    nmm = op.count('MUL_MAT')
    if nmm == 1: return 2 * _prod(out) * (ins[0][0] if ins[0] else 0)
    if nmm > 1:
        K = ins[0][0]; Nn = ins[1][1] if len(ins[1]) > 1 else 1
        return 2 * K * Nn * sum(w[1] for i, w in enumerate(ins) if i != 1 and len(w) > 1)
    if op == 'FLASH_ATTN_EXT':
        q, k = ins[0], ins[1]
        if len(q) >= 3 and len(k) >= 2: return 4 * q[0] * q[2] * q[1] * k[1]
    return 0


def _op_name(op, n):
    def has(s): return s in n
    for s, name in (('patch_embd', 'patch_embd (conv matmul)'), ('position_embd', 'pos_embed add')):
        if has(s): return name
    if has('ln1.') or has('ln2.') or has('post_ln') or has('ln_pre') or has('ln_post'): return 'layernorm affine (mul+add bias)'
    if op == 'NORM': return 'layernorm (NORM)'
    if op == 'GELU' or (op == 'UNARY' and 'gelu' in n): return 'gelu'
    if has('attn_qkv.weight'): return 'attn_qkv_proj (fused q/k/v)'
    if has('attn_output.weight') or has('attn_out.weight'): return 'attn_o_proj (+residual)'
    if has('attn_q.weight') or has('attn_k.weight') or has('attn_v.weight'): return 'attn_q/k/v_proj (unfused)'
    for s, name in (('ffn_gate.weight', 'ffn_gate'), ('ffn_up.weight', 'ffn_up'), ('ffn_down.weight', 'ffn_down (+residual)'),
                    ('state_proj', 'state_proj'), ('action_in_proj', 'action_in_proj'), ('action_out_proj', 'action_out_proj'),
                    ('action_time_mlp', 'time_mlp')):
        if has(s): return name
    if op == 'FLASH_ATTN_EXT': return 'attn (flash)'
    if op == 'GEGLU': return 'geglu (gelu(gate)*up)'
    if op.startswith('RMS_NORM'): return 'rms_norm(+mul)'
    if op == 'ROPE': return 'rope'
    if op == 'CONT': return 'cont (KV repack)' if 'cache_' in n else 'cont (permute/copy)'
    if op == 'CPY': return 'cpy (KV write)' if 'cache_' in n else 'cpy (f32->f16)'
    return {'SET_ROWS': 'set_rows (KV write)', 'CONCAT': 'concat', 'REPEAT': 'repeat (bias)', 'ADD': 'add', 'SILU': 'silu',
            'SCALE': 'scale', 'GET_ROWS': 'get_rows'}.get(op, op.lower())


def _sig(dims):
    ins, out = _dims(dims)
    return (':'.join(map(str, ins[0])) if ins else '-', ':'.join(map(str, ins[1])) if len(ins) > 1 else '-',
            ':'.join(map(str, out)) if out else '-')


def htp_agg(path):
    """htp_profiling.csv -> per-operator table of the last (warm) iteration of every phase"""
    rows = []
    with open(path) as f:
        for r in csv.reader(f, delimiter='\t'):
            if len(r) >= 7 and r[5].isdigit():
                rows.append(dict(phase=r[0], op=r[1], names=r[2], dims=r[3], types=r[4], usec=int(r[5]), cyc=int(r[6])))
    blocks = []
    for r in rows:
        if not blocks or blocks[-1][0] != r['phase']: blocks.append([r['phase'], []])
        blocks[-1][1].append(r)
    PHS = [('vision', 'Vision'), ('prefix', 'Prefix'), ('diffusion', 'FM'), ('', 'tail(2tok)')]
    sel = {lbl: [b[1] for b in blocks if b[0] == tag][-1] for tag, lbl in PHS if any(b[0] == tag for b in blocks)}
    table, totals = [], {}
    for ph, blk in sel.items():
        agg = {}
        for r in blk:
            key = (_op_name(r['op'], r['names']), _sig(r['dims'])[2])
            e = agg.setdefault(key, dict(n=0, usec=0, flop=0, op='', types=''))
            e['n'] += 1; e['usec'] += r['usec']; e['flop'] += _flop(r['op'], r['dims'])
            if not e['op']: e['op'] = r['op']; e['W'], e['X'], e['Y'] = _sig(r['dims']); e['types'] = r['types']
        tot = sum(v['usec'] for v in agg.values())
        for (name, _), e in sorted(agg.items(), key=lambda kv: -kv[1]['usec']):
            gf = e['flop'] / (e['usec'] * 1e3) if e['usec'] and e['flop'] else 0
            table.append([ph, name, e['op'], e['W'], e['X'], e['Y'], e['types'], e['n'], f"{e['usec'] / 1000.0:.2f}",
                          f'{gf:.0f}' if gf else '-', f"{100.0 * e['usec'] / tot if tot else 0:.1f}"])
        totals[ph] = tot / 1000.0
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(['Phase', 'Operator', 'HTP op', 'W', 'X', 'Y', 'types', 'Num', 'dsp_ms', 'GFLOPS', 'Pct_of_phase'])
    w.writerows(table); w.writerow([])
    for ph, t in totals.items(): w.writerow([f'TOTAL {ph} (DSP)', '', '', '', '', '', '', '', f'{t:.1f}', '', '100'])
    return buf.getvalue()


def w_npu_of(table_csv):
    """htp-agg table -> the 20 W_NPU weights (W_MAP)"""
    out = {k: 0.0 for k in W_NPU_KEYS}
    for r in csv.reader(io.StringIO(table_csv)):
        if len(r) < 9 or r[0] == 'Phase': continue
        for name, ph, op, W in W_MAP:
            if r[0] == ph and r[2] == op and (W is None or r[3] == W): out[name] += float(r[8])
    return {k: round(v, 2) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------------
# logs, run batches, validity
# ---------------------------------------------------------------------------------------------------------------------
def _f(v):
    try: return float(v)
    except ValueError: return v


def parse_log(text):
    """RESULT line, else the bench phase table / pipeline summary lines"""
    for line in text.split('\n'):
        if line.startswith('RESULT '):
            return {k: _f(v) for k, v in (t.split('=', 1) for t in line.split()[1:] if '=' in t)}
    r = {}
    for name, key in (('Vision Encode', 'vision_ms'), ('Prefix Pass', 'prefix_ms'), ('Diffusion Total', 'fm_ms'), ('Inference Total', 'total_ms')):
        m = re.search(rf'{name}\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)', text)
        if m:
            r[key] = float(m.group(1)); r['tool'] = 'bench'
            if key == 'total_ms': r['total_min_ms'], r['total_max_ms'] = float(m.group(2)), float(m.group(3))
    m = re.search(r'Backend:\s*(\S+)', text)
    if m and r: r['backend'] = 'HTP0' if 'HTP' in m.group(1) else 'OpenCL'
    m = re.search(r'Benchmark: (\d+) iterations', text)
    if m and r: r['measured'] = float(m.group(1))
    m = re.search(r'steady-state II ([0-9.]+) ms', text)
    if m: r.update(tool='pipeline', period_ms=float(m.group(1)))
    m = re.search(r'stage GPU ([0-9.]+) ms.*?\| NPU ([0-9.]+) ms', text)
    if m: r['stage1_ms'], r['stage2_ms'] = float(m.group(1)), float(m.group(2))   # GPU / NPU here mean first / second device
    m = re.search(r'steady-state latency ([0-9.]+) ms', text)
    if m: r['latency_ms'] = float(m.group(1))
    m = re.findall(r'\brms=([0-9.]+)', text)
    if m: r['rms'] = float(m[-1])
    return r


def read_trace(path):
    try:
        with open(path) as f: rows = list(csv.DictReader(f))
    except OSError: return None
    if not rows or 'complete_ms' not in rows[0]: return None
    return [{k: float(v) for k, v in r.items() if v not in ('', None)} for r in rows]


def read_tsv(path):
    """tab-separated file with a header row -> list of dicts"""
    if not os.path.exists(path): return []
    with open(path) as f: lines = [l.rstrip('\n').split('\t') for l in f if l.strip() and not l.startswith('#')]
    if not lines: return []
    hdr = [h.strip().lower() for h in lines[0]]
    return [dict(zip(hdr, l)) for l in lines[1:]]


JOB_COLS = ['tag', 'run', 'kind', 'model', 'set', 'cut', 'cfg', 'soc', 'cmd']


def read_jobs(d):
    out = []
    with open(os.path.join(d, 'jobs.tsv')) as f:
        for line in f:
            x = line.rstrip('\n').split('\t')
            if not x[0] or x[0] == 'tag' or x[0].startswith('#'): continue
            x += [''] * (9 - len(x))
            out.append(dict(zip(JOB_COLS, x[:8] + ['\t'.join(x[8:])])))
    return out


def _meta_val(m, *names):
    for n in names:
        if n in m and m[n] != '': return m[n]
    return None


def load_runs(d):
    """one record per run attempt: jobs.tsv + <tag>.log (+ meta.tsv, traces). Without jobs.tsv, runs come from file names."""
    runs = []
    if os.path.exists(os.path.join(d, 'jobs.tsv')):
        meta = {}
        for m in read_tsv(os.path.join(d, 'meta.tsv')): meta[m.get('tag')] = m
        for j in read_jobs(d):
            p = os.path.join(d, j['tag'] + '.log')
            text = open(p, errors='replace').read() if os.path.exists(p) else ''
            res = parse_log(text)
            rr = dict(j, dir=d, res=res, has_result='RESULT ' in text, meta=meta.get(j['tag'], {}),
                      run=int(re.sub(r'\D', '', j['run']) or 1), trace=os.path.join(d, j['tag'] + '.csv'),
                      actions=os.path.join(d, j['tag'] + '.bin'))
            runs.append(rr)
        return runs
    legacy = []
    sm = os.path.join(d, 'summary.txt')
    if os.path.exists(sm):
        for line in open(sm, errors='replace'):
            m = re.match(r'^(pi05|pi0) (npu|gpu|p=(\d+)) rep=(\d+) (.*)$', line.rstrip('\n'))
            if m: legacy.append((m.group(1), m.group(2), m.group(3), int(m.group(4)), m.group(5), None))
    else:
        for p in sorted(glob.glob(os.path.join(d, '*.log'))):
            m = re.match(r'^(?:.*_)?(pi05|pi0)_(npu|gpu|p(\d+))_r(\d+)\.log$', os.path.basename(p))
            if m: legacy.append((m.group(1), m.group(2), m.group(3), int(m.group(4)), open(p, errors='replace').read(), p[:-4] + '.csv'))
        for p in sorted(glob.glob(os.path.join(d, '*.csv'))):
            m = re.match(r'^(?:.*_)?(pi05|pi0)_p(\d+)_r(\d+)\.csv$', os.path.basename(p))
            if m and not os.path.exists(p[:-4] + '.log'): legacy.append((m.group(1), 'p', m.group(2), int(m.group(3)), '', p))
    for model, what, p, rep, text, trace in legacy:
        res = parse_log(text)
        if 'period_ms' not in res and trace and what not in ('npu', 'gpu'):
            tr = read_trace(trace)
            if tr and len(tr) > 6:
                res.update(tool='pipeline', period_ms=(tr[-2]['complete_ms'] - tr[2]['complete_ms']) / (len(tr) - 4))
        kind = what if what in ('npu', 'gpu') else 'duo'
        cut = '-' if what in ('npu', 'gpu') else f'GNp{p}'
        runs.append(dict(tag=f'{kind}_{model}_{cut}_legacy_r{rep}', run=rep, kind=kind, model=model, set='', cut=cut,
                         cfg='legacy', soc='', cmd='', dir=d, res=res, has_result=True, meta={}, trace=trace or '', actions=''))
    return runs


def check_validity(runs):
    """validity rules (exit, timeout, RESULT, drain, battery, rms). Sets r['valid'] and r['why']."""
    ref = {}
    for r in runs:
        res, m, why = r['res'], r['meta'], []
        ex = _meta_val(m, 'exit', 'rc', 'exit_code')
        if ex is not None and ex not in ('0', '0.0'): why.append('timeout' if ex in ('124', '137', '143') else f'exit {ex}')
        if str(_meta_val(m, 'timeout') or '0') not in ('0', ''): why.append('timeout')
        if not r['has_result'] or not res.get('tool'): why.append('no RESULT')
        if res.get('drain_free') == 0: why.append('drain')
        st = str(_meta_val(m, 'status1', 'status_end', 'status') or '')
        if st.lower().startswith('discharg') or st == '3': why.append('discharging')
        l0, l1 = _meta_val(m, 'level0', 'level_start'), _meta_val(m, 'level1', 'level_end')
        if l0 not in (None, '') and l1 not in (None, '') and float(l1) < float(l0): why.append('battery level dropped')
        key = (r.get('dir'), r['kind'], r['model'], r['set'], r['cut'], r['cfg'])
        if not why and isinstance(res.get('rms'), float):
            if key in ref and abs(res['rms'] - ref[key]) / ref[key] > 1e-4: why.append(f"rms {res['rms']} vs {ref[key]}")
            ref.setdefault(key, res['rms'])
        r['valid'], r['why'] = not why, ', '.join(why)
    return runs


def value_of(r):
    res = r['res']
    return res.get('period_ms') if res.get('tool') == 'pipeline' else res.get('total_ms')


def configs(runs):
    """group run attempts per config over all batch directories. A re-run (tag + "b") replaces only an invalid attempt
    of the same batch and run index, so every valid run of every batch counts."""
    out = {}
    for r in runs:
        key = (r['kind'], r['model'], r['set'], r['cut'], r['cfg'])
        out.setdefault(key, []).append(r)
    res = {}
    for key, rs in out.items():
        per_run = {}
        for r in rs:
            i = (r.get('dir'), r['run'])
            if r['valid'] or i not in per_run: per_run[i] = r
        vals = [value_of(r) for r in per_run.values() if r['valid'] and value_of(r) is not None]
        res[key] = dict(attempts=rs, vals=vals, mean=sum(vals) / len(vals) if vals else None,
                        invalid=sum(not r['valid'] for r in rs), short=len(vals) < 3)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------------------------------
def fmt(v, nd=1):
    return '-' if v is None else f'{v:.{nd}f}'


def emit(a, name, header, cols, rows):
    """print a table. With --out also write <out>/<name>.md and .csv."""
    md = [f'# {name}', ''] + [f'- {h}' for h in header] + ['', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    text = '\n'.join(md) + '\n'
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, f'{name}.md'), 'w').write(text)
        with open(os.path.join(a.out, f'{name}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(rows)


CFG_KINDS = ('duo', 'sweep', 'kcut')   # kinds whose command depends on the kernel config (ENV_DUO)


def heads_of(r):
    return 'q4' if r.get('set', '').endswith('-q4h') else 'hf16'


def scope_runs(a, runs):
    """table2/3/4/6/fig6 use one kernel config and one head set: --cfg / --heads, else the single value found in the
    runs, else the SoC default / hf16 (with a warning). npu/gpu/ladder commands do not depend on the kernel config."""
    def pick(name, vals, default):
        if getattr(a, name, None): return getattr(a, name)
        vals = sorted(set(vals))
        if len(vals) == 1: return vals[0]
        if vals: print(f'warning: runs with several {name} values ({", ".join(vals)}); using {default} (pass --{name})', file=sys.stderr)
        return default
    a.cfg = pick('cfg', [r['cfg'] for r in runs if r['kind'] in CFG_KINDS and r['cfg'] not in ('legacy', '')],
                 DEFAULT_CFG.get(a.soc or 'sd8elite', 'default'))
    a.heads = pick('heads', [heads_of(r) for r in runs if r['cfg'] != 'legacy'], 'hf16')
    out = []
    for r in runs:
        if r['cfg'] != 'legacy':
            if heads_of(r) != a.heads or (r['kind'] in CFG_KINDS and r['cfg'] != a.cfg): continue
            if r['kind'] not in CFG_KINDS: r = dict(r, cfg='*')
        out.append(r)
    return out


def summary_header(a, runs, key=None, prof=None, rule=True):
    inv = sum(not r['valid'] for r in runs)
    h = ['protocol: 3 warm-up + 20 measured inferences per run, 3 runs per config, mean of the valid runs of all '
         'batches (cells with < 3 valid runs are marked *)',
         f'invalid runs: {inv} of {len(runs)} (rules: exit, timeout, RESULT, drain, battery, rms)']
    if getattr(a, 'heads', None) and getattr(a, 'cfg', None): h.append(f'GGUF sets: {a.heads} action heads; VLA-Duo runs with kernel config {a.cfg}')
    if rule: h.append('cut rule: executable cuts only, min T_est, ties -> min L_est -> min c -> G->N')
    if prof: h += [f'profile: {key} ({prof["desc"]})', 'provenance: ' + ', '.join(f'{k}={v}' for k, v in prof['provenance'].items()),
                   f'kernel config: {prof["kernel_config"]}']
    return h


def _mean_cell(c):
    return '-' if not c or c['mean'] is None else f"{c['mean']:.1f}{'*' if c['short'] else ''}"


def sum_burst(a, runs):
    rows = []
    for key, c in sorted(configs(runs).items()):
        per = ' / '.join(f"{fmt(value_of(r))}{'' if r['valid'] else ' (invalid: ' + r['why'] + ')'}" for r in c['attempts'])
        rows.append([key[0], key[1], key[3], key[4], per, _mean_cell(c)])
    emit(a, 'burst', summary_header(a, runs, rule=False), ['kind', 'model', 'cut', 'cfg', 'runs (ms)', 'mean (ms)'], rows)


def _profile_or_none(a):
    try: return resolve(a)
    except SystemExit as e:
        print(f'warning: {e}; model columns left empty', file=sys.stderr); return None, None


def _select_for(prof, model, corun):
    rows = Cell(prof, model).evaluate('solo' if corun == 'solo' else 'co-run')
    return rows, select(rows)


def sum_table2(a, runs):
    key, prof = _profile_or_none(a)
    cf = configs(runs); rows = []
    for model in MODELS:
        for kind, label in (('gpu', 'GPU only'), ('npu', 'NPU only')):
            c = next((v for k, v in cf.items() if k[0] == kind and k[1] == model), None)
            if c: rows.append([model, label, '-', _mean_cell(c), ' / '.join(fmt(x) for x in c['vals'])])
        duo = {k[3]: v for k, v in cf.items() if k[0] == 'duo' and k[1] == model}
        if not duo: continue
        sel = _select_for(prof, model, 'co-run')[1][0]['cut'] if prof else None
        cut = sel if sel in duo else (next(iter(duo)) if len(duo) == 1 else sel)
        lab = 'replay-selected' if cut == sel else 'manual (not replay-selected)'
        c = duo.get(cut)
        rows.append([model, f'periodic ({lab})', cut or '-', _mean_cell(c), ' / '.join(fmt(x) for x in (c or {}).get('vals', []))])
    emit(a, 'table2', summary_header(a, runs, key, prof), ['model', 'row', 'cut', 'T (ms)', 'runs'], rows)


def _by_cut(cf, model):
    """measured configs per cut: a `burst duo`/`table3` config wins over a `burst sweep` one of the same cut"""
    return {k[3]: v for kind in ('sweep', 'duo') for k, v in cf.items() if k[0] == kind and k[1] == model and v['mean'] is not None}


def sum_table36(a, runs, which):
    key, prof = _profile_or_none(a)
    if not prof: die('table3/table6 need a profile (--soc/--cfg or --profile)')
    cf = configs(runs); out = []
    for model in MODELS:
        cfm = _by_cut(cf, model)
        meas = {c: v['mean'] for c, v in cfm.items()}
        if not meas: continue
        co, solo = Cell(prof, model).evaluate('co-run'), Cell(prof, model).evaluate('solo')
        bco, bso = {r['cut']: r for r in co}, {r['cut']: r for r in solo}
        orc = min(meas, key=meas.get)
        rows = [('VLA-Duo', select(co)[0]['cut'], bco), ('Solo-profile', select(solo)[0]['cut'], bso), ('Measured oracle', orc, bco)]
        if which == 'table6': rows = rows[1:2] + rows[:1] + rows[2:]
        for name, cut, est in rows:
            T = meas.get(cut); Te = est[cut]['T'] if cut in est else None
            err = f'{100 * abs(Te - T) / T:.1f}' if T and Te else '-'
            reg = f'{100 * (T - meas[orc]) / meas[orc]:.2f}' if T else 'not measured'
            out.append([model, name, cut, est[cut]['args'] if cut in est else '', fmt(Te, 0), _mean_cell(cfm.get(cut)), err, reg])
    hdr = summary_header(a, runs, key, prof) + ['T_est: cut model at the executable cut (Solo row: all co-run rates 1); '
                                                 'oracle = best tested executable cut; regret vs the oracle']
    emit(a, which, hdr, ['model', 'row', 'cut', 'args', 'T_est', 'T (ms)', 'error %', 'regret %'], out)


def sum_table4(a, runs):
    cf = configs(runs); rows = []
    for rung, label in (('N1', 'baseline NPU'), ('N2', '+ graph/memory NPU'), ('N3', '+ kernel opt. NPU'),
                        ('G1', 'baseline GPU'), ('G2', '+ graph/memory GPU'), ('G3', '+ kernel opt. GPU')):
        c = next((v for k, v in cf.items() if k[0] == f'ladder-{rung}' and k[1] == 'pi0'), None)
        rows.append([rung, label, _mean_cell(c), ' / '.join(fmt(x) for x in (c or {}).get('vals', []))])
    duo = [(k, v) for k, v in cf.items() if k[0] == 'duo' and k[1] == 'pi0' and v['mean'] is not None]
    key, prof = _profile_or_none(a)
    sel = _select_for(prof, 'pi0', 'co-run')[1][0]['cut'] if prof else None
    h = next((v for k, v in duo if k[3] == sel), None)
    rows.append(['H', f'+ heterogeneous (Table 2 periodic, {sel or "-"})', _mean_cell(h), ' / '.join(fmt(x) for x in (h or {}).get('vals', []))])
    emit(a, 'table4', summary_header(a, runs, key, prof), ['rung', 'row', 'T (ms)', 'runs'], rows)


def read_plans(path):
    out = {}
    for line in open(path):
        w = line.split()
        if len(w) >= 5 and w[0] in MODELS and w[1].isdigit():
            out[(w[0], int(w[1]))] = dict(T=float(w[2]), plan=w[3], order=w[4], rep=w[5] if len(w) > 5 and not w[5].isdigit() else 'dp2')
    return out


def _actions(p):
    if not p or not os.path.exists(p): return None
    b = open(p, 'rb').read(); return struct.unpack(f'{len(b) // 4}f', b)


def _cos(x, y):
    return sum(i * j for i, j in zip(x, y)) / math.sqrt(sum(i * i for i in x) * sum(j * j for j in y))


SUS_HEAD, SUS_FLAT, SUS_PROJ_MIN = 0.97, 4.0, 5.0


def sustain_verdict(rows, T, limit, aborted, dur_s):
    """rows = (done_s, t_infer, latency) per frame. UNSUSTAINABLE: stopped at the latency limit, or the last 30 s mean
    T_infer is above T. SUSTAINABLE: at the end the last 90 s mean is below 0.97 T, that mean extrapolated 5 more minutes
    at its slope stays below T, the last 90 s latency is below the limit and no frame of the last 30 s reaches T."""
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    t1 = rows[-1][0]
    win = lambda lo, hi: [r for r in rows if t1 - hi < r[0] <= t1 - lo]
    m30, m90, m0 = (mean([r[1] for r in win(lo, hi)]) for lo, hi in ((0, 30), (0, 90), (90, 180)))
    slope = (m90 - m0) / 1.5 if m0 is not None else None
    info = dict(elapsed=t1 - rows[0][0], first60=mean([r[1] for r in rows if r[0] - rows[0][0] <= 60]), last90=m90,
                slope=slope, lat90=mean([r[2] for r in win(0, 90)]), lat_max=max(r[2] for r in rows))
    if aborted: return 'UNSUSTAINABLE', dict(info, reason='latency limit')
    if m30 > T: return 'UNSUSTAINABLE', dict(info, reason='T_infer above T')
    if info['elapsed'] < 0.9 * dur_s: return 'INCOMPLETE', dict(info, reason='ended early')
    proj = m90 + max(slope or 0.0, 0.0) * SUS_PROJ_MIN
    ok = m90 < SUS_HEAD * T and m30 < SUS_HEAD * T and proj < T and info['lat90'] < limit and max(r[1] for r in win(0, 30)) < T
    reason = ('' if ok else 'no margin') + ('' if slope is None or slope <= SUS_FLAT else (', ' if not ok else '') + 'rising')
    return ('SUSTAINABLE' if ok else 'UNSUSTAINABLE'), dict(info, reason=reason, projected=proj)


def sum_sustain(a, dirs):
    """vladuo.sh sustain batches: one row per run, verdict from the per-frame trace"""
    f1 = lambda x: '-' if x is None else f'{x:.1f}'
    rows = []
    for d in dirs:
        for r in load_runs(d):
            if not r['kind'].startswith('sustain-'): continue
            mode = r['kind'].split('-', 1)[1]
            m = re.search(r'--period (\d+)', r['cmd']); T = float(m.group(1)) if m else None
            m = re.search(r'--measure (\d+)', r['cmd']); dur = (int(m.group(1)) + 1) * T / 1000 if m and T else 0
            limit = 2 * T if mode == 'duo' else T
            tr = read_trace(r['trace']) or []
            fr = [(x['complete_ms'] / 1000, max(x['vision_ms'], x['npu_ms']) if mode == 'duo' else x['vision_ms'] + x['npu_ms'],
                   x['latency_ms']) for x in tr if x['frame'] >= 3]
            temps = []
            tp = os.path.join(d, r['tag'] + '.temp')
            if os.path.exists(tp):
                for line in open(tp):
                    x = line.split()
                    if len(x) == 2 and x[1].isdigit(): temps.append(int(x[1]) / 1000)
            if not fr or T is None:
                rows.append([r['model'], mode, r['cut'], f1(T), r['run'], '-', len(tr), '-', '-', '-', '-', '-', f1(max(temps) if temps else None), 'NO DATA', ''])
                continue
            v, i = sustain_verdict(fr, T, limit, bool(r['res'].get('aborted')), dur)
            rows.append([r['model'], mode, r['cut'], f1(T), r['run'], f'{i["elapsed"]:.0f}', len(tr), f1(i['first60']), f1(i['last90']),
                         f1(i['slope']), f1(i['lat90']), f1(i['lat_max']), f1(max(temps) if temps else None), v, i['reason']])
    hdr = ['fixed-period admission (frame f at t0 + f T), verdict from the per-frame trace (frames >= 3)',
           'T_infer = max(GPU, NPU stage) for duo, GPU + NPU stage for npu. Latency limit 2T (duo) or T (npu)']
    emit(a, 'sustain', hdr, ['model', 'mode', 'cut', 'T (ms)', 'run', 'elapsed (s)', 'frames', 'T_infer first 60 s', 'T_infer last 90 s',
                             'slope (ms/min)', 'latency last 90 s', 'latency max', 'die max (C)', 'verdict', 'note'], rows)


def sum_kcut(a, dirs):
    plans_path = a.plans or next((p for d in dirs for p in [os.path.join(d, 'plans.tsv')] if os.path.exists(p)), None)
    plans = read_plans(plans_path) if plans_path else {}
    win = tuple(int(x) for x in a.window.split(':')) if a.window else None
    runs = []
    for d in dirs:
        if os.path.exists(os.path.join(d, 'jobs.tsv')):
            for r in load_runs(d):
                m = re.match(r'k(\d+)$', r['cut'])
                if r['kind'] == 'kcut' and m: runs.append(dict(r, k=int(m.group(1))))
        else:
            for (model, k), pl in plans.items():
                for tr in sorted(glob.glob(os.path.join(d, f'kcut_*_{model}_k{k}_r{pl["rep"]}.csv'))):
                    runs.append(dict(tag=os.path.basename(tr)[:-4], run=1, kind='kcut', model=model, set='', cut=f'k{k}', cfg='legacy',
                                     res={'tool': 'pipeline'}, has_result=True, meta={}, trace=tr, actions=tr[:-4] + '_actions.bin', k=k))
    check_validity(runs)
    ref, rows = {}, []
    for r in sorted(runs, key=lambda r: (r['model'], r['k'], r['run'])):
        tr = read_trace(r['trace'])
        T = lt = None; drain = r['res'].get('drain_free')
        if win and tr:
            fr = {int(x['frame']): x for x in tr}
            if all(f in fr for f in range(win[0], win[1] + 1)):
                T = sum(fr[f]['ii_ms'] for f in range(win[0], win[1] + 1)) / (win[1] - win[0] + 1)
                lt = sum(fr[f]['latency_ms'] for f in range(win[0], win[1] + 1)) / (win[1] - win[0] + 1) / T
                drain = int(fr[win[1]]['complete_ms'] < tr[-1]['admitted_ms'])
        elif isinstance(r['res'].get('period_ms'), float):
            T = r['res']['period_ms']; lt = r['res'].get('lt')
        r['T'], r['lt'], r['drain'] = T, lt, drain
        act = _actions(r['actions'])
        if act is not None and r['k'] == 1: ref[(r['model'], r['run'])] = act
        r['cos'] = _cos(ref[(r['model'], r['run'])], act) if act is not None and (r['model'], r['run']) in ref else None
    groups = {}
    for r in runs: groups.setdefault((r['model'], r['k']), []).append(r)
    for (model, k), rs in sorted(groups.items()):
        ok = [r for r in rs if r['valid'] and r['T'] is not None]
        T = sum(r['T'] for r in ok) / len(ok) if ok else None
        Tdp = plans.get((model, k), {}).get('T')
        lts = [r['lt'] for r in ok if r['lt'] is not None]
        coss = [r['cos'] for r in ok if r['cos'] is not None]
        rows.append([model, k, fmt(Tdp), f"{fmt(T)}{'*' if len(ok) < 3 else ''}", ' / '.join(fmt(r['T']) for r in rs),
                     f'{100 * (T - Tdp) / Tdp:+.1f}' if T and Tdp else '-', fmt(sum(lts) / len(lts), 2) if lts else '-',
                     'ok' if all(r['drain'] in (1, 1.0, None) for r in ok) else 'DRAIN', f'{min(coss):.6f}' if coss else '-',
                     plans.get((model, k), {}).get('plan', '')])
    hdr = summary_header(a, runs, rule=False) + [f"window: {'frames %d..%d of the trace (diagnostic)' % win if win else 'RESULT period (3+20)'}",
                                                 'cos: actions vs the k=1 run of the same run index (expected >= 0.9998)']
    emit(a, 'kcut', hdr, ['model', 'k', 'T_dp', 'T (ms)', 'runs', 'diff %', 'L/T', 'drain', 'cos k=1', 'plan'], rows)


def sum_fig6(a, runs):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('summary fig6: matplotlib not installed, skipped', file=sys.stderr); return
    key, prof = _profile_or_none(a)
    if not prof: die('fig6 needs a profile')
    cf = configs(runs)
    fig, axs = plt.subplots(len(MODELS), 2, figsize=(7.0, 2.6 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        rows = Cell(prof, model).evaluate('co-run')
        meas = {c: v['mean'] for c, v in _by_cut(cf, model).items()}
        for j, d in enumerate(('G>N', 'N>G')):
            ax = axs[i][j]
            rs = sorted((r for r in rows if r['kind'] == 'duo' and r['dir'] == d), key=lambda r: r['c'])
            ax.plot([r['c'] for r in rs], [r['T'] for r in rs], '-', color='#e34948', lw=1.2, label='estimate')
            pts = [(r['c'], meas[r['cut']]) for r in rs if r['cut'] in meas]
            if pts: ax.plot(*zip(*pts), 'o', color='#68b24f', ms=3.5, label='measured')
            ax.set_xlabel(f'cut (atoms), {d}'); ax.set_ylabel(f'{model} period (ms)'); ax.legend(frameon=True, fontsize=7)
    fig.tight_layout()
    os.makedirs(a.out or '.', exist_ok=True)
    fig.savefig(os.path.join(a.out or '.', 'fig6.pdf')); print('wrote', os.path.join(a.out or '.', 'fig6.pdf'))


# ---------------------------------------------------------------------------------------------------------------------
# profile builder
# ---------------------------------------------------------------------------------------------------------------------
def _is_htp_csv(p):
    try:
        with open(p) as f: first = f.readline()
    except OSError: return False
    return first.startswith('phase\top')


def build_profile(d, soc, cfg, base=None):
    """QUICK run directory -> profile (fields not measured come from `base`, else error)"""
    shipped = load_profiles()
    prof = copy.deepcopy(base) if base else {k: copy.deepcopy(shipped['sd8elite/default'][k]) for k in ('tau_ms', 'vit_fixed_ms', 'npu_patch_ms', 'sh_gpu')}
    if not base: prof['provenance'] = {k: 'nominal, paper' for k in ('tau_ms', 'vit_fixed_ms', 'npu_patch_ms', 'sh_gpu')}
    prof.update(soc=soc, kernel_config=cfg, desc=f'QUICK profile {os.path.basename(os.path.normpath(d))}', opmat_full=None)
    tagm = f'measured {os.path.basename(os.path.normpath(d))}'
    prov, got, opm, c4 = prof['provenance'], set(), [], None
    jobs = read_jobs(d) if os.path.exists(os.path.join(d, 'jobs.tsv')) else \
        [dict(tag=os.path.basename(p)[:-4], model='', cut='', cmd='', kind='') for p in sorted(glob.glob(os.path.join(d, '*.log')))]
    models = prof.setdefault('models', {})
    for j in jobs:
        lp, cp = os.path.join(d, j['tag'] + '.log'), os.path.join(d, j['tag'] + '.csv')
        text = open(lp, errors='replace').read() if os.path.exists(lp) else ''
        if '=== g_ij' in text or '=== h_ij' in text: opm.append(lp); continue
        if _is_htp_csv(cp) or 'GGML_HEXAGON_PROFILE=1' in j['cmd'] or j['kind'].endswith('-C2'):
            if _is_htp_csv(cp): prof['w_npu'] = w_npu_of(htp_agg(cp)); prov['w_npu'] = tagm; got.add('w_npu')
            continue
        res = parse_log(text)
        model = j['model'] if j['model'] in MODELS else res.get('model')
        if model not in MODELS: continue
        mm = models.setdefault(model, {})
        if res.get('tool') == 'bench':
            if isinstance(res.get('measured'), float) and res['measured'] < 3: continue
            be = res.get('backend') or ('OpenCL' if '--backend OpenCL' in j['cmd'] or j['kind'] in ('gpu',) else 'HTP0')
            dev = 'npu' if 'HTP' in str(be) else 'gpu'
            mm.setdefault('solo', {})[dev] = {'vision': res['vision_ms'], 'prefix': res['prefix_ms'], 'fm': res['fm_ms']}
            got.add(f'models.{model}.solo.{dev}')
        elif res.get('tool') == 'pipeline':
            cut = j['cut'] or (f"{res.get('dir', '')}p{int(res['p'])}" if isinstance(res.get('p'), float) else '')
            dp = mm.setdefault('deep', {})
            if cut == 'NGp47':
                dp['npu_stage47'], dp['gpu_stage47'] = res['stage1_ms'], res['stage2_ms']; got |= {f'models.{model}.deep.npu_stage47', f'models.{model}.deep.gpu_stage47'}
                if model == 'pi0' and isinstance(res.get('kv_ms'), float):
                    prof['handoff_deep_ms'] = res['kv_ms']; prov['handoff_deep_ms'] = tagm; got.add('handoff_deep_ms')
            elif cut == 'NGp46':
                dp['T46'] = res['period_ms']; got.add(f'models.{model}.deep.T46')
            elif cut == 'GNp28' and model == 'pi0':
                tr = read_trace(cp)
                if tr: c4 = tr[-1]['npu_ms']
    if opm:
        g, h, solo = opmat_merge(opmat_chunks(opm))
        rt = parse_ratio(ratio_text(g, h, solo, os.path.basename(opm[0])))
        rep = re.search(r'--rep-ms\s+(\d+)', ' '.join(j['cmd'] for j in jobs))
        prof['opmat'] = dict(set='quick', rep_ms=int(rep.group(1)) if rep else 8, **rt); prov['opmat'] = tagm; got.add('opmat')
        if not base: prof.pop('w_g_override', None)
    if c4 is not None and 'models.pi0.solo.npu' in got:
        s = models['pi0']['solo']['npu']
        prof['handoff_ms'] = round(c4 - (s['prefix'] + s['fm']), 1); prov['handoff_ms'] = tagm; got.add('handoff_ms')
    for x in sorted(got):
        if x.startswith('models.'): prov[x] = tagm
    for k in ('handoff_ms', 'handoff_deep_ms', 'w_npu', 'opmat'): prov.setdefault(k, 'missing')
    order = ['soc', 'kernel_config', 'desc', 'provenance', 'tau_ms', 'vit_fixed_ms', 'npu_patch_ms', 'handoff_ms', 'handoff_deep_ms',
             'models', 'w_npu', 'sh_gpu', 'w_g_override', 'opmat', 'opmat_full']
    prof = {k: prof[k] for k in order if k in prof}
    return prof, got


# ---------------------------------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------------------------------
OPMAT_REF = {'sd8elite/default': (.876, .919, .973, .933, .961, .975), 'sd8gen5/noxmem': (.845, .906, .892, .934, .889, .942)}
RATES_REF = {('sd8elite/default', 'pi0'): (1.000, .806, .934), ('sd8elite/default', 'pi05'): (1.000, .777, .940),
             ('sd8gen5/noxmem', 'pi0'): (.990, .873, .962), ('sd8gen5/noxmem', 'pi05'): (1.000, .807, .946)}
FINE_REF = {('sd8elite/default', 'pi0'): ((132, 447), (135, 427), (272, 488)), ('sd8gen5/noxmem', 'pi0'): ((130, 399), (136, 360), (294, 406)),
            ('sd8elite/default', 'pi05'): ((142, 494), (148, 470), (272, 539)), ('sd8gen5/noxmem', 'pi05'): ((136, 429), (144, 386), (272, 449))}
SEL_REF = {('sd8elite/default', 'pi0'): ((23, 451, 891), (23, 428, 851)), ('sd8elite/default', 'pi05'): ((25, 495, 981), (25, 473, 941)),
           ('sd8gen5/noxmem', 'pi0'): ((22, 401, 795), (23, 362, 717)), ('sd8gen5/noxmem', 'pi05'): ((24, 429, 849), (25, 386, 763))}
CURVE_SHA256 = 'faa4908a31171643f82234a23815e129ae3e0b00d1a365b9cd6f5866bb701a82'
KCUT_REF = """pi0 1 449.4 G22V,N7V18L10F c=1;sync;G:0/0;N:1/-1
pi0 2 455.9 G20V,N8V,N1V18L9F,G1F c=1;sync;G:0/0,3/-1;N:2/-1,1/0
pi0 3 462.8 G20V,N7V,N2V18L8F,G1F,N1F c=1;sync;G:0/0,3/-1;N:2/-1,1/0,4/-1
pi0 4 465.9 G20V,N8V,G1V,N18L9F,G1F c=1;sync;G:0/0,4/-1,2/0;N:3/-1,1/0
pi0 5 464.2 G2V,N1V,G20V,N5V,G1V,N18L10F c=1;sync;G:0/0,4/-1,2/0;N:3/-1,1/0,5/-1
pi0 6 479.5 G15V,N7V,G3V,N4V2L,N16L3F,G1F,N5F,G1F c=1;sync;G:0/0,5/-1,2/0,7/-1;N:4/-1,1/0,6/-1,3/0
pi0 7 482.2 G17V,N7V,G3V,N1V,G1V,N18L4F,G1F,N5F c=1;sync;G:0/0,6/-1,2/0,4/0;N:5/-1,1/0,7/-1,3/0
pi0 8 498.9 N1V,G1V,N1V,G19V,N1V,G1V,N4V,G1V,N18L10F c=1;sync;G:5/-1,1/0,7/-1,3/0;N:0/0,6/-1,2/0,8/-1,4/0
pi0 9 496.8 G1V,N1V,G16V,N5V,G3V,N2V,G1V,N18L4F,G1F,N5F c=1;sync;G:0/0,6/-1,2/0,8/-1,4/0;N:5/-1,1/0,7/-1,3/0,9/-1
pi05 1 494.5 G25V,N4V18L10F c=1;sync;G:0/0;N:1/-1
pi05 2 504.8 G22V,N7V,N18L9F,G1F c=1;sync;G:0/0,3/-1;N:2/-1,1/0
pi05 3 509.5 G22V,N7V,N18L8F,G1F,N1F c=1;sync;G:0/0,3/-1;N:2/-1,1/0,4/-1
pi05 4 519.4 G22V,N6V,G1V,N18L9F,G1F c=1;sync;G:0/0,4/-1,2/0;N:3/-1,1/0
pi05 5 512.5 G1V,N1V,G23V,N3V,G1V,N18L10F c=1;sync;G:0/0,4/-1,2/0;N:3/-1,1/0,5/-1
pi05 6 525.5 G15V,N9V,G5V,N3L,N15L1F,G1F,N7F,G1F c=1;sync;G:0/0,5/-1,2/0,7/-1;N:4/-1,1/0,6/-1,3/0
pi05 7 538.1 G1V,N1V,G22V,N1V,G1V,N2V,G1V,N18L10F c=1;sync;G:0/0,6/-1,2/0,4/0;N:5/-1,1/0,7/-1,3/0
pi05 8 556.5 N1V,G1V,N1V,G19V,N1V,G1V,N4V,G1V,N18L10F c=1;sync;G:5/-1,1/0,7/-1,3/0;N:0/0,6/-1,2/0,8/-1,4/0
pi05 9 552.3 G1V,N1V,G17V,N2V,G5V,N2V,G1V,N18L1F,G1F,N8F c=1;sync;G:0/0,6/-1,2/0,8/-1,4/0;N:5/-1,1/0,7/-1,3/0,9/-1"""


def curve_lines(profs):
    out = []
    for key in SHIPPED:
        for model in MODELS:
            cell = Cell(profs[key], model)
            for variant in ('co-run', 'solo'):
                for r in cell.evaluate(variant):
                    out.append(f"{key},{model},{variant},{r['dir']},{r['c']},{r['p']},{r['T']:.0f},{r['L']:.0f}")
    return out


def run_check(a):
    profs, fails = load_profiles(a.profiles), []

    def report(item, ok, msg):
        print(f"[{'PASS' if ok else 'FAIL'}] {item}: {msg}", flush=True)
        if not ok: fails.append(item)
    for key in SHIPPED:
        e = validate(profs[key], key)
        report(f'0 schema {key}', not e, 'ok' if not e else ', '.join(e))
    for key in SHIPPED:
        v = opmat_vit(profs[key]); vals = [v[f'G_vit|{b}'] for b in PH] + [v[f'N_{b}|vit'] for b in PH]
        ok = all(abs(x - y) <= 0.0005 for x, y in zip(vals, OPMAT_REF[key]))
        report(f'1 OPMAT_VIT {key}', ok, 'G ' + '/'.join(f'{x:.4f}' for x in vals[:3]) + ' N ' + '/'.join(f'{x:.4f}' for x in vals[3:]))
    cells = {(k, m): Cell(profs[k], m) for k in SHIPPED for m in MODELS}
    for (key, model), c in cells.items():
        got = (c.G[('prefix', 'vit')], c.G[('fm', 'vit')], c.N[('vit', 'prefix')])
        same = all(c.G[('prefix', b)] == got[0] and c.G[('fm', b)] == got[1] for b in PH) and \
            all(c.N[(x, b)] == got[2] for x in PH for b in ('prefix', 'fm'))
        ok = same and all(abs(x - y) <= 0.001 for x, y in zip(got, RATES_REF[(key, model)]))
        report(f'2 rates {key} {model}', ok, 'G[pre|*] {:.3f} G[fm|*] {:.3f} N[*|pre,fm] {:.3f}'.format(*got))
    for (key, model), c in cells.items():
        co, cng, _ = c.fine_search('co-run'); so = c.fine_search('solo')[0]
        got = ((co[0], co[2]), (so[0], so[2]), (cng[0], cng[2]))
        report(f'3 fine {key} {model}', got == FINE_REF[(key, model)],
               'co-run {}/{:.0f}, solo {}/{:.0f}, best N>G {}/{:.0f}'.format(*[x for g in got for x in g]))
    lines = curve_lines(profs)
    sha = hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()
    if a.golden:
        gold = {}
        for r in csv.DictReader(open(a.golden)):
            if r['ng_rates'] != 'Ph' or float(r['tau']) != 1.0: continue
            key = 'sd8elite/default' if r['soc'] == 'OP13' else 'sd8gen5/noxmem'
            gold[(key, r['model'], r['variant'], r['direction'], int(r['c']))] = (int(r['p']), float(r['T']), float(r['L']))
        bad = [l for l in lines if (lambda f: gold.get((f[0], f[1], f[2], f[3], int(f[4]))) != (int(f[5]), float(f[6]), float(f[7])))(l.split(','))]
        report('4 golden curves', not bad and len(gold) == len(lines), f'{len(lines)} rows, {len(bad)} differ from {a.golden}' + (f' (first: {bad[0]})' if bad else ''))
    report('4 curve hash', sha == CURVE_SHA256, f'sha256 {sha}')
    for (key, model), c in cells.items():
        got = []
        for v in ('co-run', 'solo'):
            s = select(c.evaluate(v))[0]
            got.append((s['p'], round(s['T']), round(s['L'])) if s['dir'] == 'G>N' else ('N>G', s['p']))
        report(f'5 select {key} {model}', tuple(got) == SEL_REF[(key, model)],
               '; '.join(f'{v} p={g[0]} {g[1]}/{g[2]}' for v, g in zip(('co-run', 'solo'), got)))
    if not a.skip_kcut:
        ref = KCUT_REF.split('\n')
        for model in MODELS:
            got = [f'{model} {k} {T:.1f} {plan} {order}' for k, T, plan, order, gl, info in kcut_plans(profs['sd8elite/default'], model, 9)]
            want = [l for l in ref if l.startswith(model + ' ')]
            bad = [g for g, w in zip(got, want) if g != w]
            report(f'6 kcut {model}', got == want, f'{len(got)} plans, T_dp ' + ' '.join(g.split()[2] for g in got) + (f'; first diff: {bad[0]}' if bad else ''))
    if a.figdata:
        for model in MODELS:
            path = os.path.join(a.figdata, f'pred_{model}_v4.csv')
            want = [r for r in csv.DictReader(open(path)) if r['source'] == 'phase-probe']
            got = []
            for key in SHIPPED:
                name = '8 Elite (OnePlus 13)' if key.startswith('sd8elite') else '8 Elite Gen 5 (OnePlus 15)'
                got += [(name, d, str(p), str(pf), f'{T:.1f}', f'{L:.1f}', f'{tA:.1f}', f'{tB:.1f}') for d, p, pf, T, L, tA, tB in Cell(profs[key], model).layer_curve()]
            want = [(r['soc'], r['direction'], r['p'], r['p_fine'], r['T_est'], r['L_est'], r['tA'], r['tB']) for r in want]
            report(f'7 figdata {model}', got == want, f'{len(want)} rows, {sum(x != y for x, y in zip(got, want))} differ')
    print('check:', 'PASS' if not fails else 'FAIL (' + ', '.join(fails) + ')')
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def print_rates(key, prof, corun):
    om = opmat_vit(prof)
    print(f'== {key}: OPMAT_VIT G[vit|vit/prefix/fm] = ' + '/'.join(f"{om[f'G_vit|{b}']:.3f}" for b in PH) +
          '   N[vit/prefix/fm|vit] = ' + '/'.join(f"{om[f'N_{b}|vit']:.3f}" for b in PH))
    for model in MODELS:
        c = Cell(prof, model, corun)
        print(f'   {model}: {c.note}' + ('  [--corun solo: every co-run rate is 1.0; calibrated values shown]' if corun == 'solo' else ''))
        for a in PH:
            print('      ' + '  '.join(f'G[{a}|{b}]={c.G[(a, b)]:.3f}' for b in PH) + '   ' + '  '.join(f'N[{a}|{b}]={c.N[(a, b)]:.3f}' for b in PH))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def prof_args(p):
        p.add_argument('--profile', help='profile.json (single profile) instead of data/profiles.json')
        p.add_argument('--profiles', help=argparse.SUPPRESS)
        p.add_argument('--soc', choices=['sd8elite', 'sd8gen5'])
        p.add_argument('--cfg', choices=['default', 'noxmem', 'xmem'])
    p = sub.add_parser('rates'); prof_args(p); p.add_argument('--corun', choices=['quick', 'solo', 'full'], default='quick')
    p = sub.add_parser('select'); prof_args(p)
    p.add_argument('--model', choices=MODELS, required=True); p.add_argument('--corun', choices=['quick', 'solo', 'full'], default='quick')
    p.add_argument('--fine', action='store_true'); p.add_argument('--json')
    p = sub.add_parser('curve'); prof_args(p)
    p.add_argument('--model', choices=MODELS, required=True); p.add_argument('--corun', choices=['quick', 'solo', 'full'], default='quick')
    p.add_argument('--fine', action='store_true')
    p = sub.add_parser('kcut'); prof_args(p)
    p.add_argument('--model', choices=MODELS + ('all',), default='all'); p.add_argument('--kmax', type=int, default=9)
    p.add_argument('--verbose', action='store_true')
    p = sub.add_parser('check'); p.add_argument('--golden'); p.add_argument('--figdata'); p.add_argument('--skip-kcut', action='store_true')
    p.add_argument('--profiles', help=argparse.SUPPRESS)
    p = sub.add_parser('opmat'); p.add_argument('logs', nargs='+'); p.add_argument('--out')
    p = sub.add_parser('htp-agg'); p.add_argument('csv'); p.add_argument('--out'); p.add_argument('--w-npu', action='store_true')
    p = sub.add_parser('profile'); p.add_argument('dir'); p.add_argument('--soc', choices=['sd8elite', 'sd8gen5'], required=True)
    p.add_argument('--cfg', choices=['default', 'noxmem', 'xmem'], required=True)
    p.add_argument('--base', help='copy fields the directory does not measure from this shipped key (archived inputs only)')
    p.add_argument('--full', help='FULL run directory: adds opmat_full'); p.add_argument('--out')
    p = sub.add_parser('promote'); p.add_argument('file'); p.add_argument('--key', required=True); p.add_argument('--profiles')
    p = sub.add_parser('summary'); p.add_argument('what', choices=['burst', 'table2', 'table3', 'table4', 'table6', 'kcut', 'fig6', 'sustain'])
    p.add_argument('dirs', nargs='+'); prof_args(p); p.add_argument('--out'); p.add_argument('--plans'); p.add_argument('--window')
    p.add_argument('--heads', choices=['hf16', 'q4'], help='GGUF head set of the runs to use (default: the one found in the runs)')
    a = ap.parse_args(argv)

    if a.cmd == 'rates':
        key, prof = resolve(a); print_rates(key, prof, a.corun)
    elif a.cmd in ('select', 'curve'):
        key, prof = resolve(a)
        cell = Cell(prof, a.model, a.corun)
        variant = 'solo' if a.corun == 'solo' else 'co-run'
        if a.cmd == 'curve':
            w = csv.writer(sys.stdout, lineterminator='\n'); w.writerow(['dir', 'p', 'c', 'T_est', 'L_est', 'executable', 'args'])
            ex = {(r['dir'], r['c']): r for r in cell.evaluate(variant) if r['kind'] == 'duo'}
            if a.fine:
                for d, seq in (('G>N', cell.seq_gn()), ('N>G', cell.seq_ng())):
                    G, N = cell.rates(variant)[:2] if d == 'G>N' else cell.rates(variant)[2:]
                    for c in range(1, 165 if d == 'G>N' else 1352):
                        P, tA, tB = simulate(seq, c, d, G, N, cell.tau)
                        T = (P + math.ceil(cell.handoff(layer_p(c)) / cell.tau)) * cell.tau
                        r = ex.get((d, c))
                        w.writerow([d, layer_p(c), c, f'{T:.1f}', f'{T + tB * cell.tau:.1f}', int(r is not None), r['args'] if r else ''])
            else:
                for r in sorted(ex.values(), key=lambda r: (r['dir'], r['c'])):
                    w.writerow([r['dir'], r['p'], r['c'], f"{r['T']:.1f}", f"{r['L']:.1f}", 1, r['args']])
            return 0
        print(f'== {key} {a.model} corun={a.corun}: {cell.note}')
        if a.fine:
            for v in (('co-run', 'solo') if a.corun != 'solo' else ('solo',)):
                bg, bn, _ = cell.fine_search(v)
                lay = (bg[0] - 2) // 6 + 1
                print(f'  fine {v}: G->N atoms {bg[0]} (after {bg[1]} of layer {lay}) T_est {bg[2]:.0f} (GPU {bg[3]:.0f} | NPU {bg[4]:.0f})'
                      + (f';  N->G window 230..302: atoms {bn[0]} (after {bn[1]}) T_est {bn[2]:.0f}' if v == 'co-run' else ''))
            return 0
        rows = cell.evaluate(variant)
        sel, run2, bng = select(rows)
        solo = select(cell.evaluate('solo'))[0] if variant != 'solo' else sel
        ref = {r['cut']: r for r in rows if r['kind'] == 'single'}

        def line(r): return f"{r['dir']} p={r['p']} c={r['c']} [{r['args'] or 'no --vit-split'}] T_est {r['T']:.0f} L_est {r['L']:.0f}"
        print(f'  selected:  {line(sel)}  cut={sel["cut"]}')
        print(f'  runner-up: {line(run2)}')
        print(f'  best N->G: {line(bng)}')
        print(f"  references: NPU-only {ref['npu']['T']:.0f}, GPU-only {ref['gpu']['T']:.0f}")
        tested = tested_set(rows, sel, solo, bng)
        print('  table3 tested set: ' + ' '.join(tested))
        if a.json:
            by = {r['cut']: r for r in rows}
            out = dict(profile=key, model=a.model, corun=a.corun, selected=short(sel), runner_up=short(run2), best_ng=short(bng),
                       solo=short(solo), tested=[short(by[c]) for c in tested],
                       references={k: r['T'] for k, r in ref.items()})
            with open(a.json, 'w') as f: json.dump(out, f, indent=1)
    elif a.cmd == 'kcut':
        a.soc = a.soc or 'sd8elite'
        key, prof = resolve(a)
        print('model\tk\tT_dp\tplan\torder\tgpu_lm')
        for model in (MODELS if a.model == 'all' else (a.model,)):
            for k, T, plan, order, gl, info in kcut_plans(prof, model, a.kmax):
                print(f'{model}\t{k}\t{T:.1f}\t{plan}\t{order}\t{gl}', flush=True)
                if a.verbose: print(f'#   {info}', file=sys.stderr)
    elif a.cmd == 'check':
        return run_check(a)
    elif a.cmd == 'opmat':
        g, h, solo = opmat_merge(opmat_chunks(a.logs))
        src = os.path.basename(a.logs[0]) if len(a.logs) == 1 else f'{len(a.logs)} chunk logs'
        txt = ratio_text(g, h, solo, src)
        if a.out: open(a.out, 'w').write(txt)
        else: sys.stdout.write(txt)
        om = opmat_vit(dict(opmat=dict(set='quick', rep_ms=0, **parse_ratio(txt)), w_npu=load_profiles()['sd8elite/default']['w_npu'],
                            w_g_override={'vis_attn': 68.1}))
        print('OPMAT_VIT (paper W_NPU, vis_attn weight 68.1): G[vit|vit/prefix/fm] = ' + '/'.join(f"{om[f'G_vit|{b}']:.3f}" for b in PH)
              + '  N[vit/prefix/fm|vit] = ' + '/'.join(f"{om[f'N_{b}|vit']:.3f}" for b in PH), file=sys.stderr)
    elif a.cmd == 'htp-agg':
        t = htp_agg(a.csv) if not open(a.csv).readline().startswith('Phase,') else open(a.csv, newline='').read()
        out = json.dumps(w_npu_of(t)) + '\n' if a.w_npu else t
        if a.out: open(a.out, 'w', newline='').write(out)
        else: sys.stdout.write(out)
    elif a.cmd == 'profile':
        shipped = load_profiles()
        if a.base and a.base not in shipped: die(f'unknown --base {a.base}')
        prof, got = build_profile(a.dir, a.soc, a.cfg, shipped.get(a.base))
        if a.full:
            fl = [os.path.join(a.full, j['tag'] + '.log') for j in read_jobs(a.full)] if os.path.exists(os.path.join(a.full, 'jobs.tsv')) \
                else sorted(glob.glob(os.path.join(a.full, '*.log')))
            prof['opmat_full'] = parse_full([p for p in fl if os.path.exists(p)])
            open(os.path.join(a.full, 'opmat_full.json'), 'w').write(dump_json(prof['opmat_full']) + '\n')
            prof['provenance']['opmat_full'] = f'measured {os.path.basename(os.path.normpath(a.full))}'
        err = validate(prof, f'{a.soc}/{a.cfg}')
        if err: die('profile incomplete (' + ', '.join(err) + '); measured: ' + ', '.join(sorted(got)) + ' (use --base for archived inputs)')
        out = a.out or os.path.join(a.dir, 'profile.json')
        open(out, 'w').write(dump_json(prof) + '\n')
        print('wrote', out, '; measured:', ', '.join(sorted(got)))
        print_rates(f'{a.soc}/{a.cfg} (new)', prof, 'quick')
    elif a.cmd == 'promote':
        path = a.profiles or PROFILES
        d, p = load_profiles(path), load_profiles(a.file)
        if 'soc' not in p: die(f'{a.file} is not a single profile')
        if a.key in d: die(f'refusing to overwrite the existing key {a.key}')
        err = validate(p, a.key)
        if err: die(f'refusing {a.file} for {a.key}: ' + ', '.join(err))
        d[a.key] = p
        open(path, 'w').write(dump_json(d) + '\n'); print(f'added {a.key} to {path}')
    elif a.cmd == 'summary':
        if a.what == 'kcut': sum_kcut(a, a.dirs); return 0
        if a.what == 'sustain': sum_sustain(a, a.dirs); return 0
        runs = check_validity([r for d in a.dirs for r in load_runs(d)])
        if a.what != 'burst': runs = scope_runs(a, runs)
        {'burst': sum_burst, 'table2': sum_table2, 'table3': lambda a, r: sum_table36(a, r, 'table3'),
         'table6': lambda a, r: sum_table36(a, r, 'table6'), 'table4': sum_table4, 'fig6': sum_fig6}[a.what](a, runs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
