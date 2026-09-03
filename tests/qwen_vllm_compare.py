#!/usr/bin/env python3
"""Teacher-forced comparison of a ds4 Qwen logits dump against a vLLM server.

    ./ds4 -m MODEL.gguf --cpu --dump-tokens -p "$PROMPT" | head -1 > ids.txt
    DS4_QWEN_DUMP_LOGITS=ds4.bin ./ds4 -m MODEL.gguf --cpu -c 4096 -n 1 --temp 0 -p "$PROMPT"
    python3 tests/qwen_vllm_compare.py ids.txt ds4.bin [--url URL] [--model NAME] [--csv per_position.csv]

The same token ids are submitted to /v1/completions with echo and
prompt_logprobs, so both sides score every prompt position on identical
input and nothing depends on sampling.  vLLM only returns its top-k per
position (the server's --max-logprobs, 20 in production), so the metrics are
argmax agreement, top-k overlap, and the log-prob gap on the tokens vLLM
reports.  Row i of the ds4 dump is the distribution after prompt token i and
vLLM's prompt_logprobs[i + 1] is the same distribution; the last ds4 row has
no vLLM counterpart and is skipped.

vLLM's FP4 activation quantisation puts its own noise around 1e-2 in the
logits, so this catches wrong operators, not rounding differences.
"""
import argparse
import json
import re
import sys
import urllib.request

import numpy as np

V = 248320
PASS_ARGMAX = 0.98
PASS_OVERLAP = 18.0


def read_ids(path):
    with open(path) as fp:
        return [int(x) for x in re.findall(r"-?\d+", fp.read())]


def query_vllm(url, model, ids, top_k):
    body = {"model": model, "prompt": ids, "max_tokens": 0, "temperature": 0,
            "echo": True, "prompt_logprobs": top_k}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def vllm_rows(response, n):
    """One {token: logprob} dict per ds4 row, plus the rank-1 token of each."""
    choice = response["choices"][0]
    rows = [{int(t): v["logprob"] for t, v in e.items()} for e in choice["prompt_logprobs"][1:]]
    tops = [next(int(t) for t, v in e.items() if v["rank"] == 1) for e in choice["prompt_logprobs"][1:]]
    if len(rows) != n - 1:
        sys.exit(f"vLLM returned {len(rows)} prompt distributions for {n} tokens")
    return rows, tops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids")
    ap.add_argument("dump")
    ap.add_argument("--url", default="http://192.168.227.28:8888/v1/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--csv", help="write per-position metrics")
    ap.add_argument("--response", help="reuse a saved vLLM response instead of querying")
    ap.add_argument("--save-response", help="save the raw vLLM response JSON")
    args = ap.parse_args()

    ids = read_ids(args.ids)
    logits = np.fromfile(args.dump, dtype=np.float32)
    if logits.size % V != 0:
        sys.exit(f"dump size {logits.size} is not a multiple of the vocab {V}")
    logits = logits.reshape(-1, V).astype(np.float64)
    if logits.shape[0] != len(ids):
        sys.exit(f"dump has {logits.shape[0]} rows, ids file has {len(ids)} tokens")

    if args.response:
        with open(args.response) as fp:
            response = json.load(fp)
    else:
        response = query_vllm(args.url, args.model, ids, args.top_k)
        if args.save_response:
            with open(args.save_response, "w") as fp:
                json.dump(response, fp)
    rows, tops = vllm_rows(response, len(ids))

    k = args.top_k
    n = len(rows)
    agree = np.zeros(n, dtype=bool)
    overlap = np.zeros(n)
    gap_actual = np.zeros(n)
    gap_topk = np.zeros(n)
    ds4_rank_of_vllm_top = np.zeros(n, dtype=np.int64)
    for i in range(n):
        row = logits[i] - logits[i].max()
        logp = row - np.log(np.exp(row).sum())
        order = np.argsort(-logp)
        ds4_top = set(order[:k].tolist())
        vllm = rows[i]
        vllm_top = [t for t, _ in sorted(vllm.items(), key=lambda kv: -kv[1])[:k]]
        agree[i] = order[0] == tops[i]
        overlap[i] = len(ds4_top & set(vllm_top))
        actual = ids[i + 1]
        gap_actual[i] = logp[actual] - vllm[actual]
        gap_topk[i] = np.mean([abs(logp[t] - vllm[t]) for t in vllm_top])
        ds4_rank_of_vllm_top[i] = int(np.where(order == tops[i])[0][0]) + 1

    if args.csv:
        with open(args.csv, "w") as fp:
            fp.write("pos,token,argmax_agree,topk_overlap,gap_actual,gap_topk_mean,ds4_rank_of_vllm_argmax\n")
            for i in range(n):
                fp.write(f"{i},{ids[i + 1]},{int(agree[i])},{int(overlap[i])},{gap_actual[i]:.4f},"
                         f"{gap_topk[i]:.4f},{ds4_rank_of_vllm_top[i]}\n")

    print(f"qwen-vllm-compare: {n} positions, argmax agreement {agree.mean() * 100:.1f}%, "
          f"top-{k} overlap mean {overlap.mean():.1f} min {int(overlap.min())}, "
          f"|dlogp(actual)| mean {np.abs(gap_actual).mean():.3f} max {np.abs(gap_actual).max():.3f}, "
          f"|dlogp(top-{k})| mean {gap_topk.mean():.3f}")
    worst = np.argsort(-np.abs(gap_actual))[:5]
    for i in worst:
        print(f"  pos {i}: token {ids[i + 1]} agree={int(agree[i])} overlap={int(overlap[i])} "
              f"dlogp={gap_actual[i]:+.3f} ds4_rank_of_vllm_argmax={ds4_rank_of_vllm_top[i]}")
    ok = agree.mean() >= PASS_ARGMAX and overlap.mean() >= PASS_OVERLAP
    print("qwen-vllm-compare: PASS" if ok else "qwen-vllm-compare: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
