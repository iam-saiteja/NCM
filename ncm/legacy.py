"""Import NCM v1 (.ncm binary format 2) files into a v2 Memory.

v1 vectors were 128-d random projections and cannot be reused, so texts are
re-embedded. Speaker comes from the v1 tag ("user"/"assistant") or a "role: "
text prefix; v1 timestamps were turn counters, so turns are spaced a minute apart
ending at the file's save time.

    from ncm.legacy import import_v1
    mem = import_v1("old/memory_store.ncm")
    mem.save("memory.ncm")
"""

import gzip
import io
import json
import struct

from ncm.memory import Memory


def _read_v1(path):
    data = open(path, "rb").read()
    buf = io.BytesIO(data)
    if buf.read(4) != b"NCM\x02" or struct.unpack(">B", buf.read(1))[0] != 2:
        raise ValueError(f"{path} is not an NCM v1 file")
    flags = struct.unpack(">H", buf.read(2))[0]
    _step, count, created = struct.unpack(">QIq", buf.read(20))
    profile = {"semantic_dim": 128, "emotional_dim": 3, "state_dim": 7}
    if flags & 0b10:
        profile = json.loads(buf.read(struct.unpack(">I", buf.read(4))[0]))
    if flags & 0b1000:
        buf.read(struct.unpack(">I", buf.read(4))[0])                  # auto-state, unused
    body = buf.read()
    mem = io.BytesIO(gzip.decompress(body) if flags & 0b1 else body)
    width = 2 if flags & 0b100 else 4
    dims = profile["semantic_dim"] + profile["emotional_dim"] + profile["state_dim"]
    dims += 5 if flags & 0b1000 else 0
    rows = []
    for _ in range(count):
        mem.read(struct.unpack(">H", mem.read(2))[0])                  # id
        ts, = struct.unpack(">q", mem.read(8))
        mem.read(4 + dims * width)                                     # strength + vectors
        text = mem.read(struct.unpack(">H", mem.read(2))[0]).decode("utf-8")
        tags = [mem.read(struct.unpack(">B", mem.read(1))[0]).decode("utf-8")
                for _ in range(struct.unpack(">B", mem.read(1))[0])]
        if flags & 0b10000:
            if struct.unpack(">B", mem.read(1))[0]:
                mem.read(struct.unpack(">H", mem.read(2))[0])
            mem.read(1)
        rows.append((ts, text, tags))
    return rows, created


def import_v1(path: str, embedder=None, config=None) -> Memory:
    rows, created = _read_v1(path)
    rows.sort(key=lambda r: r[0])
    last = rows[-1][0] if rows else 0
    out = Memory(embedder=embedder, config=config)
    texts, speakers, times = [], [], []
    for ts, text, tags in rows:
        speaker = next((t for t in tags if t in ("user", "assistant")), None)
        head, sep, rest = text.partition(": ")
        if sep and head in ("user", "assistant"):
            speaker, text = speaker or head, rest
        if text.strip() and not text.startswith("[UPDATE] Was:"):     # v1 conflict traces
            texts.append(text)
            speakers.append(speaker or "user")
            times.append(created - 60.0 * (last - ts))
    if texts:
        out.add_many(texts, speakers, times)
    return out
