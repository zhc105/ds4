#!/usr/bin/env python3
"""Independent Flash-Next reference: a teacher-forced, token-at-a-time f32
forward straight from the HF safetensors and the .ngram sidecar, with no code
shared with ds4 or the converter.  Writes logits [n, V] to OUT.npy, which
compares against a DS4_QWEN_DUMP_LOGITS dump (they agree to ~1e-5).

    python3 tests/qwen_flash_next_ref.py ids.txt N_TOKENS OUT.npy [HF_DIR] [NGRAM]

Slow (about 40 s per token after the first, which caches the dequantised
non-expert weights, ~20 GB), so it is the tool for short prompts when the
production vLLM comparison leaves a doubt that its own FP4 noise cannot
explain.  Semantics follow the production vLLM package and llama.cpp's
qwen4exp: sigmoid GDN output gate, renormalised top-k routing, EOS-reset
n-gram window, dilated PLE conv.
"""
import sys, json, numpy as np
sys.path.insert(0, 'gguf-tools')
from hf_gguf import SafeTensors, nvfp4_dequant_modelopt, read_ngram_header, ue4m3_to_f32, bf16_to_f32

import re
ids = [int(x) for x in re.findall(r'-?\d+', open(sys.argv[1]).read())]
n_tok = int(sys.argv[2])
OUT = sys.argv[3]
HF = sys.argv[4] if len(sys.argv) > 4 else 'hf/Qwen3.8-Flash-Next-NVFP4'
NG = sys.argv[5] if len(sys.argv) > 5 else 'gguf/Qwen3.8-Flash-Next-NVFP4.ngram'
st = SafeTensors(HF); P = 'model.language_model'
cfg = json.load(open(HF + '/config.json'))['text_config']
eps = 1e-6; HC = 4; D = 2560
cache = {}
def W(n):
    if n not in cache: cache[n] = st.read_f32(n)
    return cache[n]
def lin(base, keep=True):
    if base in cache: return cache[base]
    if base + '.weight_scale' in st:
        w = nvfp4_dequant_modelopt(st.read(base + '.weight'), st.read(base + '.weight_scale'),
                                   float(st.read_f32(base + '.weight_scale_2').reshape(-1)[0]))
    else:
        w = W(base + '.weight')
    if keep: cache[base] = w
    return w
def silu(x): return x / (1 + np.exp(-x))
def sigmoid(x): return 1 / (1 + np.exp(-x))
def bf16_round(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000).astype(np.uint32).view(np.float32)
def rms(x, w=None, one=True):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    if w is None: return y
    return y * ((1.0 + w) if one else w)

def hc_mix(x4, base, inject=True):
    xn = rms(x4) * (1 + W(base + '.hc_norm.weight')).reshape(HC, D)
    lo = silu(lin(base + '.input_mix_weight_down') @ xn.reshape(-1) / HC)
    gate = sigmoid(lin(base + '.input_mix_weight_up') @ lo).reshape(HC, D)
    inj = lin(base + '.block_inject_weight') @ xn.reshape(-1) if inject else None
    return (xn * gate).mean(0), inj
def hc_combine(x4, y, inj):
    return x4 + y[None, :] * (2 * sigmoid(inj / HC))[:, None]

nk, nv, hd = cfg['linear_num_key_heads'], cfg['linear_num_value_heads'], cfg['linear_key_head_dim']
KC = cfg['linear_conv_kernel_dim']
class GDN:
    def __init__(self): self.hist = None; self.S = np.zeros((nv, hd, hd), np.float32)
    def __call__(self, h, base):
        qkv = lin(base + '.in_proj_qkv') @ h; z = lin(base + '.in_proj_z') @ h
        a = lin(base + '.in_proj_a') @ h; b = lin(base + '.in_proj_b') @ h
        cw = W(base + '.conv1d.weight').reshape(qkv.size, KC)
        if self.hist is None: self.hist = np.zeros((KC - 1, qkv.size), np.float32)
        acc = cw[:, -1] * qkv
        for j in range(KC - 1): acc += cw[:, j] * self.hist[j]
        self.hist = np.vstack([self.hist[1:], qkv[None, :]])
        mixed = silu(acc)
        q = mixed[:nk*hd].reshape(nk, hd); k = mixed[nk*hd:2*nk*hd].reshape(nk, hd); v = mixed[2*nk*hd:].reshape(nv, hd)
        q = q / np.sqrt((q*q).sum(-1, keepdims=True) + eps); k = k / np.sqrt((k*k).sum(-1, keepdims=True) + eps)
        A = W(base + '.A_log'); dtb = W(base + '.dt_bias'); nw = W(base + '.norm.weight')
        g = -np.exp(A) * np.log1p(np.exp(a + dtb)); beta = sigmoid(b)
        out = np.zeros((nv, hd), np.float32)
        for hv in range(nv):
            hk = hv // (nv // nk)
            S = self.S[hv] * np.exp(g[hv])
            S = S + np.outer(beta[hv] * (v[hv] - S @ k[hk]), k[hk])
            self.S[hv] = S
            o = (S @ q[hk]) / np.sqrt(hd)
            out[hv] = rms(o, nw, one=False) * sigmoid(z.reshape(nv, hd)[hv])
        return lin(base + '.out_proj') @ out.reshape(-1)

nh, nkv, ahd = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
n_rot = 64; base_theta = cfg['rope_parameters']['rope_theta']
def rope(x, pos):
    half = n_rot // 2
    i = np.arange(half); theta = pos * base_theta ** (-2.0 * i / n_rot)
    c, s = np.cos(theta), np.sin(theta)
    x0, x1 = x[..., :half].copy(), x[..., half:n_rot].copy()
    x = x.copy(); x[..., :half] = x0 * c - x1 * s; x[..., half:n_rot] = x0 * s + x1 * c
    return x
class ATTN:
    def __init__(self): self.K = []; self.V = []
    def __call__(self, h, base, pos):
        qg = (lin(base + '.q_proj') @ h).reshape(nh, 2 * ahd)
        k = (lin(base + '.k_proj') @ h).reshape(nkv, ahd); v = (lin(base + '.v_proj') @ h).reshape(nkv, ahd)
        q = rope(rms(qg[:, :ahd], W(base + '.q_norm.weight')), pos); gate = qg[:, ahd:]
        k = rope(rms(k, W(base + '.k_norm.weight')), pos)
        self.K.append(bf16_round(k)); self.V.append(bf16_round(v))   # the caches are bf16
        K = np.stack(self.K); Vv = np.stack(self.V)   # [t, nkv, hd]
        out = np.zeros((nh, ahd), np.float32)
        for hh in range(nh):
            g = hh // (nh // nkv)
            sc = K[:, g] @ q[hh] / np.sqrt(ahd); sc = np.exp(sc - sc.max()); sc /= sc.sum()
            out[hh] = (sc[:, None] * Vv[:, g]).sum(0) * sigmoid(gate[hh])
        return lin(base + '.o_proj') @ out.reshape(-1)

def moe(h, base):
    logits = W(base + '.gate.weight') @ h
    p = np.exp(logits - logits.max()); p /= p.sum()
    top = np.argsort(-p)[:cfg['num_experts_per_tok']]; w = p[top] / p[top].sum()
    y = np.zeros(D, np.float32)
    for e, we in zip(top, w):
        eb = f'{base}.experts.{e}'
        g = lin(eb + '.gate_proj', keep=False) @ h; u = lin(eb + '.up_proj', keep=False) @ h
        y += we * (lin(eb + '.down_proj', keep=False) @ (silu(g) * u))
    sb = base + '.shared_expert'
    s = lin(sb + '.down_proj') @ (silu(lin(sb + '.gate_proj') @ h) * (lin(sb + '.up_proj') @ h))
    return y + sigmoid(W(base + '.shared_expert_gate.weight') @ h)[0] * s

hdr = read_ngram_header(NG); table = np.memmap(NG, dtype=np.uint8, mode='r')
class PLE:
    def __init__(self): self.prev = [cfg['eos_token_id']] * (len(hdr['multipliers']) - 1); self.hist = None
    def __call__(self, x4, base, token):
        n_gram = len(hdr['multipliers']); n_head = len(hdr['head_offsets']); per = n_head // (n_gram - 1)
        eos = cfg['eos_token_id']; win = [token]; cut = False
        for s in range(1, n_gram):
            cut = cut or self.prev[s - 1] == eos
            win.append(eos if cut else self.prev[s - 1])
        self.prev = [token] + self.prev[:-1]
        rows = []
        for n in range(2, n_gram + 1):
            mixed = win[0] * hdr['multipliers'][0]
            for j in range(1, n): mixed ^= win[j] * hdr['multipliers'][j]
            mixed &= (1 << 64) - 1
            for g in range(per):
                hh = (n - 2) * per + g
                rows.append(mixed % hdr['head_vocab_sizes'][hh] + hdr['head_offsets'][hh])
        rb = hdr['row_bytes']
        emb = np.concatenate([table[4096 + r * rb: 4096 + (r + 1) * rb] for r in rows]).astype(np.uint8)
        val = ue4m3_to_f32(emb & 0x7f) * np.where(emb & 0x80, -1.0, 1.0) * hdr['scale']
        key = (lin(base + '.key_proj') @ val).reshape(HC, D); value = lin(base + '.value_proj') @ val
        key = rms(key) * (1 + W(base + '.norm_key.weight')).reshape(HC, D)
        q = rms(x4) * (1 + W(base + '.norm_query.weight')).reshape(HC, D)
        s = (key * q).sum(-1) / np.sqrt(D)
        gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
        gated = gate[:, None] * value[None, :]
        normalized = (rms(gated) * (1 + W(base + '.norm_conv.weight')).reshape(HC, D)).reshape(-1)
        kern = cfg['ple_conv_kernel_size']; dil = n_gram; nh_ = (kern - 1) * dil
        if self.hist is None: self.hist = np.zeros((nh_, HC * D), np.float32)
        cw = W(base + '.conv1d.weight').reshape(HC * D, kern)
        acc = cw[:, -1] * normalized
        for k in range(kern - 1):
            back = (kern - 1 - k) * dil
            acc += cw[:, k] * self.hist[nh_ - back]
        self.hist = np.vstack([self.hist[1:], normalized[None, :]])
        return x4 + gated + silu(acc).reshape(HC, D)

layer_types = cfg['layer_types']; ple_layers = [i - 1 for i in cfg['ple_layer_ids']]
gdn = [GDN() if t == 'linear_attention' else ATTN() for t in layer_types]; ple = PLE()
embd = st.read(f'{P}.embed_tokens.weight')
out = np.zeros((n_tok, embd.shape[0]), np.float32)
for pos in range(n_tok):
    tok = ids[pos]
    x4 = np.tile(bf16_to_f32(embd[tok:tok + 1])[0], (HC, 1)).astype(np.float32)
    for il in range(cfg['num_hidden_layers']):
        L = f'{P}.layers.{il}'
        if il in ple_layers: x4 = ple(x4, L + '.ple', tok)
        h, inj = hc_mix(x4, L + '.attn_hyper_connection')
        y = gdn[il](h, L + '.linear_attn') if layer_types[il] == 'linear_attention' else gdn[il](h, L + '.self_attn', pos)
        x4 = hc_combine(x4, y, inj)
        h, inj = hc_mix(x4, L + '.mlp_hyper_connection')
        x4 = hc_combine(x4, moe(h, L + '.mlp'), inj)
    mixed, _ = hc_mix(x4, f'{P}.hyper_connection_mixer', inject=False)
    out[pos] = lin('lm_head') @ mixed
    print(f'pos {pos} tok {tok} argmax {int(out[pos].argmax())}', flush=True)
    np.save(OUT, out[:pos + 1])
