#!/usr/bin/env python3
"""Offline cache simulation over a DS4_PLE_TRACE n-gram row-id dump.

Answers one question before anyone writes a cache: for a given RAM budget, how
much does row-granular caching buy over the page cache the engine uses today?
The trace is the real fetch stream (ds4.c qwen_ple_trace), so every number here
is measured on the actual access sequence, not a model of it.

Reported:

  * per-head distinct rows and how concentrated the accesses are;
  * the novelty curve -- distinct rows against tokens processed.  This answers
    "does the working set saturate": a curve still climbing linearly at the end
    means no cache size ever reaches ~0 misses;
  * the compulsory (first-touch) rate, a floor under *any* policy's miss rate;
  * the exact LRU hit rate at every cache size at once, from LRU stack
    distances, for a row-granular cache and -- same trace, same bytes -- for a
    4 KiB page cache, which is the thing to beat;
  * the greedy-optimal split of a total row budget across the heads.

usage:
  ple_cache_sim.py TRACE [--row-bytes 160] [--page-bytes 4096]
                         [--budgets 0.25,1,4,10,24,48] [--max-tokens N]

Two things to keep in mind when reading the output.  The stack-distance pass is
O(n log n) in Python, so a million-token trace takes minutes; --max-tokens
bounds it and the hit-rate curve is usually stable well before that.  And a
"token" here is a row of the trace, which a speculative pass contributes k+1 of
(one per verified row, rejected drafts included), so the token count overstates
generated tokens by roughly the accepted length.
"""
from __future__ import annotations

import argparse
import heapq
import struct
import sys

import numpy as np

TRACE_MAGIC = b"DS4PLTRC"
ROW_META_BYTES = 8      # tag + slot overhead a row-granular cache needs per row


def read_trace(path):
    with open(path, "rb") as f:
        if f.read(8) != TRACE_MAGIC:
            sys.exit(f"{path}: not a DS4_PLE_TRACE dump")
        version, n_head = struct.unpack("<II", f.read(8))
        if version != 1:
            sys.exit(f"{path}: unsupported trace version {version}")
        raw = f.read()
    ids = np.frombuffer(raw, dtype="<u4")
    if ids.size % n_head:
        sys.exit(f"{path}: truncated (not a whole number of tokens)")
    return ids.reshape(-1, n_head), int(n_head)


def stack_distances(seq):
    """LRU stack distance of every access: the number of distinct items seen
    since the previous access to the same one.  -1 marks a first touch, which
    no cache can hit.  An access hits an LRU of capacity C exactly when its
    distance is < C, so this one pass yields every cache size.

    A Fenwick tree over access positions holds one mark per distinct item at
    its most recent access, so the distance is a range sum."""
    n = len(seq)
    bit = [0] * (n + 1)
    last = {}
    out = np.empty(n, dtype=np.int32)
    for i, x in enumerate(seq):
        p = last.get(x)
        if p is None:
            out[i] = -1
        else:
            s, j = 0, i
            while j > 0:
                s += bit[j]
                j -= j & -j
            t, j = 0, p + 1
            while j > 0:
                t += bit[j]
                j -= j & -j
            out[i] = s - t
            j = p + 1
            while j <= n:
                bit[j] -= 1
                j += j & -j
        j = i + 1
        while j <= n:
            bit[j] += 1
            j += j & -j
        last[x] = i
    return out


def hits_at(dist, capacity):
    """Hit count for an LRU of `capacity` items."""
    if capacity <= 0:
        return 0
    finite = dist[dist >= 0]
    return int(np.searchsorted(np.sort(finite), capacity, side="left"))


def novelty_curve(ids, n_head, points=24):
    """Distinct (head, row) pairs against tokens processed at log-spaced
    checkpoints.  Each head indexes its own region of the table, so a row id is
    only meaningful together with its head."""
    n_tok = len(ids)
    if n_tok < 2:
        return np.array([1]), np.array([1])
    marks = np.unique(np.geomspace(1, n_tok, points).astype(np.int64))
    columns = np.arange(n_head, dtype=np.int64)
    distinct, seen, cursor = [], set(), 0
    for c in marks:
        keys = ids[cursor:c].astype(np.int64) * n_head + columns
        seen.update(keys.reshape(-1).tolist())
        cursor = c
        distinct.append(len(seen))
    return marks, np.array(distinct)


def power_law_exponent(tokens, distinct):
    """d = a * t^b.  b near 0: bounded working set.  b near 1: every token
    brings new rows and no cache size suffices."""
    if len(tokens) < 4:
        return float("nan")
    x = np.log(tokens[2:].astype(float))
    y = np.log(np.maximum(distinct[2:], 1).astype(float))
    return float(np.polyfit(x, y, 1)[0])


def greedy_split(per_head_dist, budget_rows, grid_points=64):
    """Split `budget_rows` across heads to maximise total hits.

    The objective is separable and each head's hit curve is concave in capacity
    (a property of LRU), so consuming intervals in order of hits-per-row is
    optimal -- but a head's intervals are only offered once its earlier ones
    are taken, which a heap enforces."""
    heap, cursor = [], [0] * len(per_head_dist)
    steps = []
    for h, dist in enumerate(per_head_dist):
        finite = np.sort(dist[dist >= 0])
        top = max(int(finite[-1]) + 1, 1) if finite.size else 1
        grid = np.unique(np.geomspace(1, top, grid_points).astype(np.int64))
        steps.append((grid, np.searchsorted(finite, grid, side="left")))

    def push(h):
        grid, hits = steps[h]
        i = cursor[h]
        if i >= len(grid):
            return
        c0 = int(grid[i - 1]) if i else 0
        h0 = int(hits[i - 1]) if i else 0
        dc, dh = int(grid[i]) - c0, int(hits[i]) - h0
        if dc > 0:
            heapq.heappush(heap, (-dh / dc, h, dc, dh))

    for h in range(len(per_head_dist)):
        push(h)

    alloc = np.zeros(len(per_head_dist), dtype=np.int64)
    hits, left = 0.0, budget_rows
    while heap and left > 0:
        _, h, dc, dh = heapq.heappop(heap)
        take = min(dc, left)
        alloc[h] += take
        hits += dh * take / dc
        left -= take
        if take == dc:
            cursor[h] += 1
            push(h)
    return alloc, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--row-bytes", type=int, default=160,
                    help="bytes per n-gram row (Flash-Next: 160)")
    ap.add_argument("--page-bytes", type=int, default=4096,
                    help="page-cache granularity to compare against")
    ap.add_argument("--budgets", default="0.25,1,4,10,24,48",
                    help="comma-separated cache budgets in GiB")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="use only the first N tokens of the trace")
    args = ap.parse_args()

    ids, n_head = read_trace(args.trace)
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[: args.max_tokens]
    n_tok, n_acc = len(ids), len(ids) * n_head
    print(f"trace    {n_tok} tokens, {n_acc} row accesses, {n_head} heads")
    print(f"layout   {args.row_bytes} B/row, page cache granularity {args.page_bytes} B")

    print("\n== working set ==")
    marks, distinct = novelty_curve(ids, n_head)
    b = power_law_exponent(marks, distinct)
    print(f"distinct rows {int(distinct[-1])} over {n_acc} accesses"
          f"  ->  compulsory rate {100.0 * distinct[-1] / n_acc:.1f}%"
          f" (a floor under every policy)")
    verdict = ("bounded working set" if b < 0.35 else
               "still growing: no cache size reaches ~0 misses" if b > 0.6 else
               "growth is slowing")
    print(f"novelty fit d ~ t^{b:.3f}  ({verdict})")
    step = max(1, len(marks) // 8)
    print("tokens -> distinct: " + "  ".join(
        f"{int(t)}:{int(d)}" for t, d in zip(marks[::step], distinct[::step])))

    print("\n== per-head ==")
    print(f"{'head':>4} {'distinct':>10} {'accesses':>10} {'top1%':>7} {'top10%':>7}")
    per_head_dist = []
    for h in range(n_head):
        col = ids[:, h].astype(np.int64).tolist()
        per_head_dist.append(stack_distances(col))
        _, counts = np.unique(col, return_counts=True)
        counts = np.sort(counts)[::-1]
        d = len(counts)
        print(f"{h:>4} {d:>10} {len(col):>10}"
              f" {100 * counts[: max(1, d // 100)].sum() / counts.sum():>6.1f}%"
              f" {100 * counts[: max(1, d // 10)].sum() / counts.sum():>6.1f}%")

    # Rows are packed contiguously, so a row's file byte offset is
    # row_id * row_bytes up to a constant header, which does not move a page.
    page_ids = (ids.astype(np.int64) * args.row_bytes) // args.page_bytes
    page_dist = [stack_distances(page_ids[:, h].tolist()) for h in range(n_head)]

    print("\n== cache size -> hit rate (same trace, same bytes) ==")
    print(f"{'GiB':>6} {'rows/head':>10} {'row LRU':>9} {'page LRU':>9} {'row greedy':>11}")
    for g in [float(x) for x in args.budgets.split(",")]:
        budget = int(g * 2**30 / (args.row_bytes + ROW_META_BYTES))
        per_head = budget // n_head
        row_hits = sum(hits_at(d, per_head) for d in per_head_dist)
        page_cap = int(g * 2**30 / args.page_bytes) // n_head
        page_hits = sum(hits_at(d, page_cap) for d in page_dist)
        _, greedy_hits = greedy_split(per_head_dist, budget)
        print(f"{g:>6.2f} {per_head:>10} {100 * row_hits / n_acc:>8.1f}%"
              f" {100 * page_hits / n_acc:>8.1f}% {100 * greedy_hits / n_acc:>10.1f}%")

    alloc, _ = greedy_split(per_head_dist, int(float(args.budgets.split(",")[-1])
                                               * 2**30 / (args.row_bytes + ROW_META_BYTES)))
    print(f"\ngreedy split at the largest budget (rows/head): "
          + " ".join(str(int(a)) for a in alloc))

    print("\nRow caching only beats the page cache to the extent the hot set is"
          "\nconcentrated; compare the two LRU columns at your budget.")


if __name__ == "__main__":
    main()
