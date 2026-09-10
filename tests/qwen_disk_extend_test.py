"""Live server test of the per-conversation disk checkpoint: a conversation
that grows past the continued-store interval must extend its own file, a
second conversation gets a file of its own, and switching back resumes the
first from disk at its full length.  Reads the server log to see what the
store did.

usage: qwen_disk_extend_test.py http://HOST:8000 STORY.txt SERVER.log
"""
import json, os, re, sys, time, urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
STORY = open(sys.argv[2]).read()
LOG = sys.argv[3]

def ask(messages, tag):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": 40,
            "temperature": 0, "reasoning_effort": "none"}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}))
    j = json.loads(r.read())
    u = j["usage"]
    text = (j["choices"][0]["message"].get("content") or "").strip()
    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['prompt_tokens']} cached={cached} -> {text[:40]!r}", flush=True)
    return {"role": "assistant", "content": text}, u["prompt_tokens"], cached

def log_lines(since):
    lines = open(LOG, errors="replace").read().splitlines()
    return [l for l in lines[since:] if "kv cache" in l], len(lines)

fails = 0
def check(cond, what):
    global fails
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        fails += 1

mark = len(open(LOG, errors="replace").read().splitlines())
# Conversation A: about 22K tokens of story (a cold store at its anchor),
# then a turn that doubles it.
a = [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Read this:\n" + STORY[:100000] + "\n\nSay 'one'."}]
r1, p1, _ = ask(a, "A1 long story")
a += [r1, {"role": "user", "content": "Now read more:\n" + STORY[100000:180000] + "\n\nSay 'two'."}]
r2, p2, c2 = ask(a, "A2 grows")
check(c2 >= p1, "A2 did not continue from the live state")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv cache"):][:150])
stored = [l for l in lines if "kv cache stored" in l]
check(len(stored) <= 1, "conversation A should have at most one new file so far")

# Conversation B: a different story.  Storing A away must go to A's own
# file (extending it, or finding it already covered on a rerun), never to
# another one.
b = [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Read this instead:\n" + STORY[200000:300000] + "\n\nSay 'three'."}]
r3, p3, c3 = ask(b, "B1 another conversation")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv cache"):][:150])
check(any(("kv cache extended" in l or "kv cache covered" in l) and "reason=evict" in l for l in lines),
      "storing conversation A away did not go to its own file")
check(not any("kv cache stored" in l and "reason=evict" in l for l in lines),
      "storing conversation A away wrote a new file")

# Back to A: its file must hold the whole history.
a += [r2, {"role": "user", "content": "Say 'four'."}]
r4, p4, c4 = ask(a, "A3 back to the first conversation")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv cache"):][:150])
hit = [l for l in lines if "kv cache hit" in l]
check(hit and int(re.search(r"tokens=(\d+)", hit[0]).group(1)) >= p2 - 64,
      "A3 did not resume from A's file at its full length")
print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
