"""Live server test of batched Qwen decode: the same greedy requests sent
concurrently (decoded as the rows of one pass by a --batched-session N
server) and one at a time must produce the same tokens, since a batched
pass is bit-identical to the serialized one.

usage: qwen_batch_server_test.py http://HOST:8000 MODEL [N] [MAX_TOKENS]
"""
import json, sys, threading, time, urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
MODEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 96

PROMPTS = [
    "Write the integers from 1 to 200, separated by commas. Do not stop early.",
    "Write a compact C function that validates UTF-8, then explain each branch.",
    "List the first one hundred prime numbers and show no derivation.",
    "Describe how an LRU cache works using a detailed worked example.",
    "Write the first eighty Fibonacci numbers, one per line.",
    "Explain B-tree insertion with a concrete sequence of twenty keys.",
    "Generate SQL that creates and queries a small issue tracker schema.",
    "Compare TCP and UDP using six precise operational examples.",
]


def ask(i):
    # a distinct system line per request keeps the prefixes apart (no shared cache)
    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0, "reasoning_effort": "none",
            "messages": [{"role": "system", "content": f"You are assistant number {i}. Be direct."},
                         {"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    return (j["choices"][0]["message"].get("content") or ""), j["usage"]["completion_tokens"]


def concurrent():
    out = [None] * N

    def run(i):
        out[i] = ask(i)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(N)]
    t = time.time()
    for th in threads: th.start()
    for th in threads: th.join()
    return out, time.time() - t


together, t_batch = concurrent()
t = time.time()
alone = [ask(i) for i in range(N)]
t_serial = time.time() - t
fails = 0
for i in range(N):
    same = together[i] == alone[i]
    print(f"request {i}: {'same' if same else 'DIFFERENT'} tokens={alone[i][1]} -> {alone[i][0][:50]!r}")
    if not same:
        fails += 1
        print(f"  concurrent: {together[i][0][:200]!r}")
        print(f"  alone:      {alone[i][0][:200]!r}")
print(f"concurrent {t_batch:.1f}s, one at a time {t_serial:.1f}s")
print("qwen_batch_server_test:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
