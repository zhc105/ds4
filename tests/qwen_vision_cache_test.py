"""Live server test of the image loop an agent runs: a text turn, a turn
that adds an image, a text turn after it, the same history with the image
stripped, then another image.  Every turn but the first must reuse the
live or saved state instead of prefilling the story again, and the image
turns must see the pictures.  Reasoning is replayed as agents do; pass
--no-think to send reasoning_effort none, which makes the replay text equal
to the live text (the memory-text path instead of thinking-visible).

usage: qwen_vision_cache_test.py http://HOST:8000 STORY.txt [--no-think]
"""
import base64, json, os, struct, sys, time, urllib.request, zlib

NO_THINK = "--no-think" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
URL, STORY = args[0].rstrip("/") + "/v1/chat/completions", args[1]

def png(rgb):
    """A 256x256 solid-colour PNG as a data URI."""
    w = h = 256
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}

fails = 0

def ask(messages, tag, expect=None):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": 60, "temperature": 0}
    if NO_THINK:
        body["reasoning_effort"] = "none"
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}))
    j = json.loads(r.read())
    u = j["usage"]
    m = j["choices"][0]["message"]
    text = (m.get("content") or "").strip()
    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['prompt_tokens']} cached={cached} -> {text[:60]!r}", flush=True)
    if expect and expect.lower() not in text.lower():
        print(f"  FAIL: expected {expect!r} in the answer", flush=True)
        globals()["fails"] += 1
    reply = {"role": "assistant", "content": text}
    if m.get("reasoning_content"):
        reply["reasoning_content"] = m["reasoning_content"]
    return reply, u["prompt_tokens"], cached

def check(cond, what):
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        globals()["fails"] += 1

story = open(STORY).read()[:20000]
base = [{"role": "system", "content": "You are a terse assistant. Answer with one word when asked for one."},
        {"role": "user", "content": "Background reading:\n" + story + "\n\nSay 'ready' and nothing else."}]
a1, p1, _ = ask(base, "R1 text")
msgs = base + [a1, {"role": "user", "content": [{"type": "text", "text": "What is the dominant color of this image? One word."}, png((220, 30, 30))]}]
a2, p2, c2 = ask(msgs, "R2 adds an image", expect="red")
check(c2 >= p1, "R2 did not continue from the live state (cached < R1's prompt)")
msgs += [a2, {"role": "user", "content": "Now say 'thanks' and nothing else."}]
a3, p3, c3 = ask(msgs, "R3 text after the image")
check(c3 >= p2, "R3 did not continue from the live state")
stripped = base + [a1, {"role": "user", "content": "What is the dominant color of this image? One word. [image removed]"},
                   a2, {"role": "user", "content": "Now say 'thanks' and nothing else."}]
a4, p4, c4 = ask(stripped, "R4 image stripped")
check(c4 > 0, "R4 prefilled from zero")
stripped += [a4, {"role": "user", "content": [{"type": "text", "text": "And this one? One word."}, png((30, 60, 220))]}]
a5, p5, c5 = ask(stripped, "R5 adds another image", expect="blue")
check(c5 >= p4, "R5 did not continue from the live state")
stripped += [a5, {"role": "user", "content": "Which two colors did you see? Two words."}]
a6, p6, c6 = ask(stripped, "R6 text after the second image", expect="blue")
check(c6 >= p5, "R6 did not continue from the live state")
print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
