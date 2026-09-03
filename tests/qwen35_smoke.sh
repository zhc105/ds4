#!/bin/sh
# Qwen3.5 CPU smoke test: teacher-forced logits of the ds4 CPU reference
# against llama.cpp on the same GGUF and the same token ids.
#
#   tests/qwen35_smoke.sh gguf/Qwen3.5-2B-NVFP4.gguf [gguf/Qwen3.5-2B-F16.gguf]
#
# The optional second GGUF is the llama.cpp side (use an F16 requantization
# of the same model to remove llama.cpp's 8-bit activation quantization from
# the comparison).  Needs LLAMA_CPP_DIR (default ~/workspace/llama.cpp) built
# with libllama in build/bin, and numpy.
set -e
DS4_MODEL=${1:?ds4 GGUF}
LLAMA_MODEL=${2:-$DS4_MODEL}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-$HOME/workspace/llama.cpp}
WORK=${TMPDIR:-/tmp}/qwen35-smoke.$$
PROMPT='请用中文和英文各写一句话介绍巴黎，然后给出一段 Python 代码：def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)。数字 2024 和 3.14159 也要出现。'
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cc -O2 -I "$LLAMA_CPP_DIR/include" -I "$LLAMA_CPP_DIR/ggml/include" \
   -o "$WORK/llama_logits_dump" tests/llama_logits_dump.c \
   -L "$LLAMA_CPP_DIR/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMA_CPP_DIR/build/bin"

./ds4 -m "$DS4_MODEL" --cpu --dump-tokens -p "$PROMPT" 2>/dev/null | head -1 | tr -d '[],' > "$WORK/ids.txt"
DS4_QWEN_DUMP_LOGITS="$WORK/ds4.bin" ./ds4 -m "$DS4_MODEL" --cpu -c 512 -n 1 --temp 0 -p "$PROMPT" > /dev/null 2>&1
"$WORK/llama_logits_dump" "$LLAMA_MODEL" "$WORK/ids.txt" "$WORK/llama.bin" 2>/dev/null

python3 - "$WORK/ds4.bin" "$WORK/llama.bin" <<'PY'
import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
n = b.size // a.size if b.size > a.size else 1
V = 248320
a = a.reshape(-1, V).astype(np.float64); b = b.reshape(-1, V).astype(np.float64)
def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))
la, lb = lsm(a), lsm(b)
kl = (np.exp(la) * (la - lb)).sum(-1)
agree = (a.argmax(-1) == b.argmax(-1)).mean()
print(f"qwen35-smoke: {a.shape[0]} positions, argmax agreement {agree*100:.1f}%, "
      f"mean KL {kl.mean():.2e}, max KL {kl.max():.2e}, max |dlogit| {np.abs(a-b).max():.3f}")
ok = agree == 1.0 and kl.max() < 1e-2
print("qwen35-smoke: PASS" if ok else "qwen35-smoke: FAIL")
sys.exit(0 if ok else 1)
PY
