"""Live server test of K/V page pool eviction: with a pool too small for
every slot's conversation, a request that needs pages must evict the least
recently used idle slot to disk (log line "kv pool full: evicted"), and the
evicted conversation must come back with the same greedy answer.

Start the server with two slots and a pool of three pages (Flash-Next
without the drafter: 49.5 MiB pages; the 2B's store is refused by the disk
cache, so it only shows the eviction, not the reload):
  ./ds4-server -m gguf/Qwen3.8-Flash-Next-NVFP4.gguf --ctx 8192 --batched-session 2 \
    --kv-pool-mb 150 --kv-disk-dir /tmp/ds4-kv-test --kv-disk-space-mb 4096 --port 8010
usage: qwen_kv_evict_test.py http://HOST:8010 MODEL STORY.txt SERVER.log

A (about 2300 tokens, two pages) and B (one page) run concurrently so both slots hold pages.
C shares B's prompt as a prefix, so the dispatcher puts it on B's slot and
extends it past the first page; the second page is not there, so A, idle and
older, must be evicted.
"""
import json, sys, threading, time, urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
MODEL = sys.argv[2]
STORY = open(sys.argv[3]).read()
LOG = sys.argv[4]


def ask(messages, tag, out=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": 24, "temperature": 0, "reasoning_effort": "none"}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    u = j["usage"]
    text = (j["choices"][0]["message"].get("content") or "").strip()
    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['prompt_tokens']} cached={cached} -> {text[:40]!r}", flush=True)
    if out is not None: out.append(text)
    return text


def log_since(mark):
    lines = open(LOG, errors="replace").read().splitlines()
    return [l for l in lines[mark:] if "kv pool" in l or "kv cache" in l], len(lines)


fails = 0


def check(cond, what):
    global fails
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        fails += 1


mark = len(open(LOG, errors="replace").read().splitlines())
a = [{"role": "user", "content": "Read this:\n" + STORY[:10500] + "\n\nSay 'one'."}]
b = [{"role": "user", "content": "Read this instead:\n" + STORY[12000:13000] + "\n\nSay 'two'."}]
c = [{"role": "user", "content": "Read this instead:\n" + STORY[12000:23000] + "\n\nSay 'three'."}]
a_out = []
ta = threading.Thread(target=ask, args=(a, "A1", a_out))
ta.start()
time.sleep(0.3)
ask(b, "B1")
ta.join()
ask(c, "C1 (on B's slot, needs a second page: evicts A)")
lines, mark = log_since(mark)
for l in lines: print("   ", l[l.find("ds4-server"):][:120])
check(any("kv pool full: evicted" in l for l in lines), "no eviction was logged when the pool ran out")
a2 = ask(a, "A2 (back from disk)")
lines, mark = log_since(mark)
for l in lines: print("   ", l[l.find("ds4-server"):][:120])
check(a_out and a2 == a_out[0], "the evicted conversation answers differently after reloading")
print("qwen_kv_evict_test:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
