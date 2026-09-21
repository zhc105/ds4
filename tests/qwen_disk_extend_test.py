"""Live server test of the chain store: a conversation that grows past a
segment's length is sealed as its prompts are prefilled, giving its slot up
for a second conversation writes its tail and nothing else, and switching
back resumes the first from disk at its full length, segments and tail.
Reads the server log to see what the store did.

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
    return [l for l in lines[since:] if "kv chain" in l], len(lines)

fails = 0
def check(cond, what):
    global fails
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        fails += 1

mark = len(open(LOG, errors="replace").read().splitlines())
# Conversation A: about 22K tokens of story (one segment sealed on the way),
# then a turn that doubles it (one or two more).
a = [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Read this:\n" + STORY[:100000] + "\n\nSay 'one'."}]
r1, p1, _ = ask(a, "A1 long story")
a += [r1, {"role": "user", "content": "Now read more:\n" + STORY[100000:180000] + "\n\nSay 'two'."}]
r2, p2, c2 = ask(a, "A2 grows")
check(c2 >= p1, "A2 did not continue from the live state")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv chain"):][:150])
sealed = [l for l in lines if "kv chain sealed" in l]
check(len(sealed) >= 2 or any("kv chain hit" in l for l in lines),
      "a history of two segments' length sealed none (or, on a rerun, resumed none)")
check(not any("kv chain tail stored" in l for l in lines), "a tail was written while its slot held the conversation")

# Conversation B: a different story.  Giving A's slot up writes A's tail,
# the rows past its last segment, and no segment.
b = [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Read this instead:\n" + STORY[200000:300000] + "\n\nSay 'three'."}]
r3, p3, c3 = ask(b, "B1 another conversation")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv chain"):][:150])
tails = [l for l in lines if "kv chain tail stored" in l and "reason=evict" in l]
check(len(tails) == 1, "giving conversation A's slot up did not write exactly one tail")
check(tails and int(re.search(r"tokens=(\d+)\.\.", tails[0]).group(1)) > 0,
      "A's tail does not hang from a segment")

# Back to A: the chain and its tail hold the whole history.
a += [r2, {"role": "user", "content": "Say 'four'."}]
r4, p4, c4 = ask(a, "A3 back to the first conversation")
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv chain"):][:150])
hit = [l for l in lines if "kv chain hit" in l]
check(hit and int(re.search(r"tokens=(\d+)", hit[0]).group(1)) >= p2 - 64,
      "A3 did not resume from A's chain at its full length")
check(hit and "state=tail" in hit[0] and int(re.search(r"files=(\d+)", hit[0]).group(1)) >= 2,
      "A3 did not resume A's tail over its segments")
print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
