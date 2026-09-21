"""Live server test of the chain store: a conversation that grows past a
segment's length is sealed as its prompts are prefilled, a slot given up
writes its tail and nothing else, and the conversation resumes from disk at
its full length, segments and tail.  Reads the server log to see what the
store did.

A server with several slots keeps both conversations in memory, so what puts
them on disk here is a restart: RESTART is the command that restarts the
server (its shutdown stores every slot's tail).  Without it only the sealing
is checked.

usage: qwen_disk_extend_test.py http://HOST:8000 STORY.txt SERVER.log [RESTART]
"""
import json, os, re, subprocess, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
URL = BASE + "/v1/chat/completions"
STORY = open(sys.argv[2]).read()
LOG = sys.argv[3]
RESTART = sys.argv[4] if len(sys.argv) > 4 else None

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

# Conversation B: a different story, on a slot of its own or on A's.
b = [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Read this instead:\n" + STORY[200000:300000] + "\n\nSay 'three'."}]
r3, p3, c3 = ask(b, "B1 another conversation")
if not RESTART:
    print("no restart command: the disk round trip is not checked")
    print("FAILED" if fails else "PASS")
    sys.exit(1 if fails else 0)

# The server stops: every slot writes its tail, the rows past its last
# segment and no segment, A's hanging from the segments sealed above.
subprocess.run(RESTART, shell=True, check=True)
for _ in range(120):
    try:
        urllib.request.urlopen(BASE + "/v1/models").read()
        break
    except OSError:
        time.sleep(2)
lines, mark = log_lines(mark)
for l in lines: print("   ", l[l.find("kv chain"):][:150])
tails = [l for l in lines if "kv chain tail stored" in l and "reason=shutdown" in l]
check(not any("kv chain sealed" in l for l in lines), "a segment was sealed where nothing was prefilled")
check(any(int(re.search(r"tokens=(\d+)\.\.", l).group(1)) > 16000 for l in tails),
      "no tail hangs from A's second segment")

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
