"""Live test of a batched server resuming an edited history from a saved
state.  P1 -> answer, P2 (P1 + a turn) -> answer, then B = P1 + a different
turn: the server must resume B from the state saved after P1, not recompute
P1.  The batched prefill loop used to feed the engine 2048-token pieces
from position 0 in that case, so the log showed "prefill chunk 0/N" for
every piece of P1; and every piece archived a state, so P1's six pieces
flushed the five-entry archive and P1's own turn-boundary state with it.
Reads the server log.

usage: qwen_batched_resume_test.py http://HOST:8010 MODEL STORY.txt SERVER.log
Server: --batched-session 2 (any model).
"""
import json, re, sys, time, urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
MODEL = sys.argv[2]
STORY = open(sys.argv[3]).read()
LOG = sys.argv[4]


def ask(messages, tag):
    body = {"model": MODEL, "messages": messages, "max_tokens": 16, "temperature": 0, "reasoning_effort": "none"}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    text = (j["choices"][0]["message"].get("content") or "").strip()
    print(f"{tag}: {time.time()-t:.1f}s prompt={j['usage']['prompt_tokens']} -> {text[:40]!r}", flush=True)
    return {"role": "assistant", "content": text}


def log_since(mark):
    lines = open(LOG, errors="replace").read().splitlines()
    return lines[mark:], len(lines)


p1 = [{"role": "user", "content": "Read this:\n" + STORY[:50000] + "\n\nSay 'one'."}]
r1 = ask(p1, "P1 (about 11000 tokens, six pieces)")
p2 = p1 + [r1, {"role": "user", "content": "Now say 'two'."}]
ask(p2, "P2 extends P1")
mark = len(open(LOG, errors="replace").read().splitlines())
b = p1 + [r1, {"role": "user", "content": "Instead, say 'three'."}]
ask(b, "B edits the last turn (must resume from P1's saved state)")
lines, mark = log_since(mark)
starts = [l for l in lines if "prompt start" in l]
zero_chunks = [l for l in lines if re.search(r"prefill chunk 0/", l)]
for l in starts + zero_chunks[:3]: print("   ", l[l.find("ds4-server"):][:110])
fails = 0
m = re.search(r"ctx=(\d+)\.\.", starts[-1]) if starts else None
if not m or int(m.group(1)) < 5000:
    print("  FAIL: B did not resume from P1's saved state (ctx start %s)" % (m.group(1) if m else "?")); fails += 1
if len(zero_chunks) > 1:
    print("  FAIL: %d prefill pieces reported no progress: the prompt was recomputed from the start" % len(zero_chunks)); fails += 1
print("qwen_batched_resume_test:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
