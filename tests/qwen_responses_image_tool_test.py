"""Live test of the Responses API turns a Codex session takes with pictures:
a long text turn, the model calling view_image, the tool answering with an
input_image block, a text turn after it, a second picture, and an edited
history.  The replay carries no reasoning (this client asks for none back),
so a turn that thought is not continued from the live history, which holds
reasoning the request does not: it resumes from the state saved before that
generation, one token short of the previous prompt, and recomputes no more
than the last turn's output.  None may recompute the history, and the
pictures must be seen.
Reads the server log when given: a prefill piece that reports no progress,
an engine that resumes short of what the cache claimed, a rejected or
failed request, all fail the test.

usage: qwen_responses_image_tool_test.py http://HOST:8000 STORY.txt [SERVER.log]
"""
import base64, json, re, struct, sys, time, urllib.error, urllib.request, zlib

URL = sys.argv[1].rstrip("/") + "/v1/responses"
STORY = open(sys.argv[2]).read()[:20000]
LOG = sys.argv[3] if len(sys.argv) > 3 else None
TOOLS = [{"type": "function", "name": "view_image",
          "description": "Attach a local image file to the conversation.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}]
fails = 0


def png(rgb):
    w = h = 256
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def check(cond, what):
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        globals()["fails"] += 1


def log_lines(mark):
    return open(LOG, errors="replace").read().splitlines()[mark:] if LOG else []


def log_check(mark, tag):
    lines = log_lines(mark)
    zero = [l for l in lines if "prefill chunk 0/" in l]
    errs = [l for l in lines if re.search(r"engine resumes at|prefill failed|rejected |cancelled during prefill", l)]
    check(len(zero) <= 1, f"{tag}: {len(zero)} prefill pieces from the start: the history was recomputed")
    check(not errs, f"{tag}: the server logged " + (errs[0][errs[0].find("ds4-server"):][:140] if errs else ""))


def ask(items, tag, expect=None, want_call=False):
    body = {"model": "qwen3.8-flash-next", "input": items, "tools": TOOLS,
            "max_output_tokens": 600, "temperature": 0, "stream": False}
    mark, t = len(log_lines(0)), time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                          {"Content-Type": "application/json"}), timeout=900)
        j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"{tag}: HTTP {e.code} {e.read()[:200]!r}", flush=True)
        globals()["fails"] += 1
        log_check(mark, tag)
        return [], 0, 0, None
    u = j["usage"]
    cached = u.get("input_tokens_details", {}).get("cached_tokens", 0)
    out = j["output"]
    text = " ".join(c.get("text", "") for o in out if o["type"] == "message" for c in o["content"]).strip()
    call = next((o for o in out if o["type"] == "function_call"), None)
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['input_tokens']} cached={cached} -> "
          f"{(call['name'] + ' ' + call['arguments']) if call else text[:60]!r}", flush=True)
    if expect:
        check(expect.lower() in text.lower(), f"{tag}: expected {expect!r} in the answer")
    if want_call:
        check(call is not None and call["name"] == "view_image", f"{tag}: the model did not call view_image")
    log_check(mark, tag)
    # Replay what Codex replays: the response's output items, reasoning included.
    return out, u["input_tokens"], cached, call


def tool_output(call, path, image):
    return {"type": "function_call_output", "call_id": call["call_id"],
            "output": [{"type": "input_text", "text": f"attached local image path: {path}"},
                       {"type": "input_image", "image_url": image, "detail": "auto"}]}


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


items = [user("Background reading:\n" + STORY + "\n\nSay 'ready' and nothing else.")]
out, p1, _, _ = ask(items, "R1 text")
items += out + [user("Call view_image on /tmp/red.png, then tell me the picture's dominant color in one word.")]
out, p2, c2, call = ask(items, "R2 asks for a picture", want_call=True)
check(c2 >= p1 - 1, "R2 recomputed more than R1's output")
if call:
    items += out + [tool_output(call, "/tmp/red.png", png((220, 30, 30)))]
    out, p3, c3, _ = ask(items, "R3 view_image returns the picture", expect="red")
    check(c3 >= p2 - 1, "R3 recomputed more than R2's output")
    r4 = [user("Now say 'thanks' and nothing else.")]
    items += out + r4
    out, p4, c4, _ = ask(items, "R4 text after the picture")
    check(c4 >= p3 - 1, "R4 recomputed more than R3's output")
    r4 += out
    items += out + [user("Call view_image on /tmp/blue.png, then name its dominant color in one word.")]
    out, p5, c5, call2 = ask(items, "R5 asks for a second picture", want_call=True)
    check(c5 >= p4 - 1, "R5 recomputed more than R4's output")
    if call2:
        items += out + [tool_output(call2, "/tmp/blue.png", png((30, 60, 220)))]
        out, p6, c6, _ = ask(items, "R6 view_image returns the second picture", expect="blue")
        check(c6 >= p5 - 1, "R6 recomputed more than R5's output")
        # Drop R4's exchange: the history diverges after R3's picture and must
        # resume from the state saved there, pictures and all, not start over.
        dropped = {id(i) for i in r4}
        edited = [i for i in items if id(i) not in dropped]
        out, p7, c7, _ = ask(edited, "R7 edited history (R4 dropped)", expect="blue")
        check(c7 > 0, "R7 prefilled from zero")
print("qwen_responses_image_tool_test:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
