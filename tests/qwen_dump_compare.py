#!/usr/bin/env python3
"""Compare two DS4_QWEN_DUMP_LOGITS dumps position by position.

Both backends dump every prompt position, so the rows line up from position
0.  --chunk C drops the first C-1 rows of the reference for dumps made by
older graph builds that only started at the end of the first prefill chunk.

    tests/qwen_dump_compare.py cpu.bin gpu.bin [--vocab 248320] [--max-kl 1e-6]

Prints max |dlogit|, mean and max KL(cpu || gpu) and the argmax agreement and
exits 1 when the KL or argmax thresholds are missed.
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("test")
ap.add_argument("--chunk", type=int, default=1, help="graph prefill chunk used for the test dump")
ap.add_argument("--vocab", type=int, default=248320)
ap.add_argument("--max-kl", type=float, default=1e-6)
args = ap.parse_args()

V = args.vocab
ref = np.fromfile(args.ref, dtype=np.float32).reshape(-1, V)
test = np.fromfile(args.test, dtype=np.float32).reshape(-1, V)
ref = ref[args.chunk - 1:]
n = min(len(ref), len(test))
if n == 0:
    sys.exit("no overlapping positions")
ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


lr, lt = log_softmax(ref), log_softmax(test)
kl = (np.exp(lr) * (lr - lt)).sum(axis=1)
agree = (ref.argmax(axis=1) == test.argmax(axis=1))
dlogit = np.abs(ref - test).max(axis=1)
print(f"positions {n}  max|dlogit| {dlogit.max():.3e}  KL mean {kl.mean():.3e} max {kl.max():.3e}  "
      f"argmax agreement {agree.mean() * 100:.1f}%")
worst = np.argsort(kl)[-3:][::-1]
for i in worst:
    print(f"  pos {i + args.chunk - 1}: KL {kl[i]:.3e}  dlogit {dlogit[i]:.3e}  "
          f"argmax ref {ref[i].argmax()} test {test[i].argmax()}")
ok = kl.max() <= args.max_kl and agree.all()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
