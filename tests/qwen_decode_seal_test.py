"""Live server test: a segment boundary crossed while generating is sealed.

An agent's turns are mostly generation (reasoning, tool calls) and little
new input, so most 16K boundaries of its history are crossed by the decode,
not by a prefill.  The decode steps end on each boundary and the segment is
sealed there, from the live state, like a prefill piece's.

A prompt a little short of 16384 tokens is answered at length, so that the
generation crosses 16384.  The log must show the segment 0..16384 sealed
after the prompt was done, while generating.  Then, with RESTART given, the
server restarts (its shutdown writes the slot's tail) and the client sends
the turn back whole, reasoning included: it must resume at the full length
through that segment and the tail.

usage: qwen_decode_seal_test.py http://HOST:8000 STORY.txt SERVER.log [RESTART]
"""
import json, re, subprocess, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
URL = BASE + "/v1/chat/completions"
STORY = open(sys.argv[2]).read()
LOG = sys.argv[3]
RESTART = sys.argv[4] if len(sys.argv) > 4 else None
SEGMENT = 16384

fails = 0
def check(cond, what):
    global fails
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        fails += 1

def log_mark():
    return len(open(LOG, errors="replace").read().splitlines())

def log_since(mark):
    return open(LOG, errors="replace").read().splitlines()[mark:]

def ask(messages, max_tokens, tag):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0}
    t = time.time()
    j = json.loads(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    u = j["usage"]
    m = j["choices"][0]["message"]
    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['prompt_tokens']} cached={cached} "
          f"out={u['completion_tokens']}", flush=True)
    reply = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        reply["reasoning_content"] = m["reasoning_content"]
    return reply, u["prompt_tokens"], u["completion_tokens"], cached

# A prompt of about 15800 tokens (a unique opening, so nothing on disk begins
# it), answered at length past 16384.  A short answer is asked again with a
# prompt shorter by the difference.
opening = f"[decode seal test {time.time():.0f}] "
target = SEGMENT - 600
chars_per_token = 4.4
for attempt in range(3):
    chars = int(target * chars_per_token)
    messages = [{"role": "user", "content": opening + "Read this:\n" + STORY[:chars] +
                 "\n\nNow retell the story in detail, at least a thousand words."}]
    mark = log_mark()
    reply, prompt, out, _ = ask(messages, 2500, f"T1 (target {target} tokens)")
    if prompt < SEGMENT and prompt + out > SEGMENT:
        break
    chars_per_token = chars / max(prompt, 1)
    target = SEGMENT - max(out // 2, 200)
else:
    print("  could not cross the boundary while generating", flush=True)
    sys.exit(1)

lines = log_since(mark)
done = next((i for i, l in enumerate(lines) if "prompt done" in l), None)
sealed = [(i, l) for i, l in enumerate(lines) if "kv chain sealed tokens=0..16384" in l]
for _, l in sealed: print("   ", l[l.find("kv chain"):][:140])
check(sealed, "the boundary crossed while generating was not sealed")
check(sealed and done is not None and sealed[0][0] > done, "the segment was sealed before the prompt was done")

if RESTART:
    mark = log_mark()
    subprocess.run(RESTART, shell=True, check=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/v1/models").read()
            break
        except OSError:
            time.sleep(2)
    messages += [reply, {"role": "user", "content": "Thanks. Say 'ok'."}]
    _, p2, _, c2 = ask(messages, 20, "T2 (after a restart, the turn sent back whole)")
    lines = log_since(mark)
    hit = [l for l in lines if "kv chain hit" in l]
    for l in hit: print("   ", l[l.find("kv chain"):][:140])
    check(c2 >= prompt + out - 8, f"T2 resumed at {c2}, not at the turn's end ({prompt + out})")
    check(hit and "files=2" in hit[-1], "T2 did not resume through the sealed segment and the tail")

print("decode seal test:", "FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
