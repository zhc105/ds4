"""Live server test of what a conversation's history is: the text its client
sent, and nothing generated after it until the client sends that back.

Two Responses clients hold the same kind of conversation, a tool call and
two plain turns, with thinking on:

  whole    sends every output item back, reasoning included
           (include: reasoning.encrypted_content, as Codex does);
  visible  drops the reasoning items, as a client that never asked for them.

The whole client must continue the live history: each request reuses all of
the request before it and what was generated after it.  The visible client
cannot, its replay leaves the live text where the reasoning began: it must
resume from the state saved before that generation, one token short of the
previous prompt, and recompute no more than the last turn's output.  With a
GPU build that is the usage's cached_tokens; a CPU build keeps no saved
states, so the visible client recomputes its prompt and only the whole one
is checked.  (What the two leave on disk is qwen_kv_evict_test.py's.)

usage: qwen_turn_history_test.py http://HOST:8000 SERVER.log [--cpu]
"""
import json, sys, urllib.request

BASE = sys.argv[1].rstrip("/")
LOG = sys.argv[2]
CPU = "--cpu" in sys.argv[3:]
URL = BASE + "/v1/responses"
TOOLS = [{"type": "function", "name": "get_time", "description": "Current time of a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]

fails = 0
def check(cond, what):
    global fails
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        fails += 1

def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

def ask(items, whole, tag):
    body = {"model": "qwen3.8-flash-next", "input": items, "max_output_tokens": 1500,
            "temperature": 0, "store": False, "tools": TOOLS}
    if whole:
        body["include"] = ["reasoning.encrypted_content"]
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req).read())
    u = j["usage"]
    cached = u.get("input_tokens_details", {}).get("cached_tokens", 0)
    kinds = [o["type"] for o in j["output"]]
    print(f"{tag}: input={u['input_tokens']} cached={cached} out={u['output_tokens']} {kinds}", flush=True)
    check(j.get("status") == "completed", f"{tag} did not complete: {j.get('status')}")
    return j["output"], u["input_tokens"], u["output_tokens"], cached

def log_mark():
    return len(open(LOG, errors="replace").read().splitlines())

def log_since(mark):
    return open(LOG, errors="replace").read().splitlines()[mark:]

def conversation(whole, salt):
    """Returns the requests made, so a second pass can send the very same ones."""
    name = "whole" if whole else "visible"
    items = [user(f"[{salt}] What time is it in Paris? Use the tool, think briefly.")]
    sent = []
    prev_in = prev_out = 0
    for turn in range(3):
        sent.append(list(items))
        out, n_in, n_out, cached = ask(items, whole, f"{name} turn {turn}")
        if turn > 0:
            if whole:
                check(cached == prev_in + prev_out,
                      f"{name} turn {turn} reused {cached}, not the {prev_in}+{prev_out} before it")
            elif not CPU:
                check(cached == prev_in - 1,
                      f"{name} turn {turn} reused {cached}, not the previous prompt less its last token ({prev_in - 1})")
        prev_in, prev_out = n_in, n_out
        for o in out:
            if o["type"] == "reasoning" and not whole:
                continue
            items.append(o)
        calls = [o for o in out if o["type"] == "function_call"]
        for c in calls:
            items.append({"type": "function_call_output", "call_id": c["call_id"], "output": "14:05"})
        if not calls:
            items.append(user("Thanks. And in Rome? No tool this time, just guess."))
    return sent

mark = log_mark()
for whole in (True, False):
    conversation(whole, "a" if whole else "b")

lines = log_since(mark)
stored_in_decode = [l for l in lines if "kv cache stored" in l and "reason=continued" in l and " gen=" in l]
check(not stored_in_decode, "a checkpoint was stored while generating")
check(not any("visible" in l and "kv cache" in l for l in lines), "a visible key is still in use")
check(not any("claims" in l and "engine resumes" in l for l in lines),
      "a cache source claimed more than the engine resumed")

print("turn history test:", "FAILED" if fails else "ok", flush=True)
sys.exit(1 if fails else 0)
