#!/usr/bin/env python3
"""Teacher-forced perplexity of a DS4_QWEN_DUMP_LOGITS dump.

    tests/qwen_ppl.py ids.json dump.bin [--from N] [--vocab 248320]

ids.json holds the prompt token ids (JSON list); row i of the dump is the
distribution after token i, scored on token i+1.  --from skips the first N
positions (e.g. the chat template) from the average.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ids")
ap.add_argument("dump")
ap.add_argument("--from", dest="start", type=int, default=0)
ap.add_argument("--vocab", type=int, default=248320)
args = ap.parse_args()

ids = json.load(open(args.ids))
V = args.vocab
logits = np.fromfile(args.dump, dtype=np.float32).reshape(-1, V).astype(np.float64)
n = min(len(logits), len(ids) - 1)
rows = logits[args.start:n]
targets = np.array(ids[args.start + 1:n + 1])
m = rows.max(axis=1, keepdims=True)
logp = rows - m - np.log(np.exp(rows - m).sum(axis=1, keepdims=True))
nll = -logp[np.arange(len(targets)), targets]
agree = (rows.argmax(axis=1) == targets).mean()
print(f"positions {len(targets)}  ppl {np.exp(nll.mean()):.2f}  mean nll {nll.mean():.4f}  "
      f"argmax hits {agree * 100:.1f}%")
