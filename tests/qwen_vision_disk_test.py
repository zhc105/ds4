"""Live server test of the disk KV cache with a picture in the history.

A state is its tokens and its pictures, and a disk key spells each picture
by its fingerprint.  Run `store`, restart the server (the shutdown writes the
live conversation to disk and the memory is gone), then run `resume`: the
replayed conversation must come back from disk past its picture without the
vision tower running, and the same conversation with another picture of the
same size (the same placeholder tokens) must not resume past where the
picture begins, and must see its own picture.

usage: qwen_vision_disk_test.py store|resume http://HOST:8000 STORY.txt STATE.json SERVER.log
"""
import base64, json, struct, sys, time, urllib.request, zlib

PHASE, URL, STORY, STATE, LOG = sys.argv[1], sys.argv[2].rstrip("/") + "/v1/chat/completions", sys.argv[3], sys.argv[4], sys.argv[5]
fails = 0


def check(cond, what):
    if not cond:
        print(f"  FAIL: {what}", flush=True)
        globals()["fails"] += 1


def png(rgb):
    w = h = 256
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def log_lines(mark):
    return open(LOG, errors="replace").read().splitlines()[mark:]


def ask(messages, tag):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": 40, "temperature": 0,
            "reasoning_effort": "none"}
    mark, t = len(log_lines(0)), time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}))
    j = json.loads(r.read())
    u = j["usage"]
    text = (j["choices"][0]["message"].get("content") or "").strip()
    cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    lines = log_lines(mark)
    towers = len([l for l in lines if "vision tower encoded" in l])
    disk = len([l for l in lines if "kv cache hit" in l])
    print(f"{tag}: {time.time()-t:.1f}s prompt={u['prompt_tokens']} cached={cached} tower={towers} disk_hits={disk} -> {text[:50]!r}", flush=True)
    return text, u["prompt_tokens"], cached, towers, disk


def conversation(rgb):
    story = open(STORY).read()[:20000]
    return [{"role": "system", "content": "You are a terse assistant. Answer with one word when asked for one."},
            {"role": "user", "content": [{"type": "text", "text": "Background reading:\n" + story +
                                          "\n\nWhat is the dominant color of this image? One word."}, png(rgb)]}]


if PHASE == "store":
    msgs = conversation((220, 30, 30))
    a1, p1, _, towers, _ = ask(msgs, "S1 a picture after the story")
    check("red" in a1.lower(), "S1 did not see the red picture")
    check(towers == 1, "S1 should encode its one picture")
    msgs += [{"role": "assistant", "content": a1}, {"role": "user", "content": "Say 'noted' and nothing else."}]
    a2, p2, c2, towers, _ = ask(msgs, "S2 text after the picture")
    check(c2 >= p1 and towers == 0, "S2 did not continue the live state")
    json.dump({"a1": a1, "a2": a2, "p1": p1, "p2": p2}, open(STATE, "w"))
    print("restart the server, then run resume")
else:
    st = json.load(open(STATE))
    tail = [{"role": "assistant", "content": st["a1"]}, {"role": "user", "content": "Say 'noted' and nothing else."},
            {"role": "assistant", "content": st["a2"]},
            {"role": "user", "content": "Which color was the image? One word."}]
    # another picture of the same size first: the file's key names the red one
    text, _, cached, towers, _ = ask(conversation((30, 60, 220)) + tail, "R1 the same turns with a blue picture")
    check(cached < st["p1"] - 60, f"R1 resumed at {cached}, past where its different picture begins")
    check(towers == 1 and "blue" in text.lower(), "R1 should encode and see its own picture")
    text, _, cached, towers, disk = ask(conversation((220, 30, 30)) + tail, "R2 the stored conversation")
    check(disk >= 1 and cached >= st["p1"], f"R2 resumed at {cached}: not from the disk state past the picture ({st['p1']})")
    check(towers == 0, "R2 ran the vision tower for a picture its state holds")
    check("red" in text.lower(), "R2 lost the picture")
print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
