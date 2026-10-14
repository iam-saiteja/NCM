"""NCM as a MemoryBench provider: a small local HTTP server that MemoryBench's `ncm`
provider (src/providers/ncm in the MemoryBench fork) talks to.

One Memory per MemoryBench container (one per question). Every message is stored as
one memory, with its speaker and its session's date (turns 30 s apart); no LLM is
used. Two-person transcripts (LoCoMo) use NCM's dialogue mode, user/assistant chats the
default configuration. Memories are saved after each ingest, so a restarted bridge
picks up where it left off.

  .venv/Scripts/python comparisons/memorybench_bridge.py --data ../memorybench/data/ncm

POST /health {}                               -> {"ncm": version, "embedder": name}
POST /ingest {tag, sessions: [{sessionId, date, dialogue, messages: [{speaker, content}]}]}
POST /search {tag, query, k}                  -> {"results": [{content, speaker, sessionId, date, time, rank, score}]}
POST /clear  {tag}
"""

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ncm                                                          # noqa: E402
from ncm import Embedder, Memory                                    # noqa: E402
from ncm.memory import Config                                       # noqa: E402

UNDATED_START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()  # ConvoMem has no dates: one day per session
TURN_GAP = 30.0


class CachedEmbedder:
    """The same turn is ingested once per question (LoCoMo: ~70 times); embed it once."""

    def __init__(self, inner):
        self.inner, self.name = inner, inner.name
        self.cache, self.lock = {}, threading.Lock()

    def __call__(self, texts, query: bool = False):
        texts = list(texts)
        keys = [(query, t) for t in texts]
        with self.lock:
            todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if todo:
            vecs = self.inner([t for _, t in todo], query=query)
            with self.lock:
                self.cache.update(zip(todo, vecs))
        with self.lock:
            return np.stack([self.cache[k] for k in keys]).astype(np.float32)


class Store:
    def __init__(self, root, embedder, keep=64):
        self.root, self.embedder, self.keep = root, embedder, keep
        self.mems, self.locks, self.lock = OrderedDict(), {}, threading.Lock()
        os.makedirs(root, exist_ok=True)

    def path(self, tag):
        return os.path.join(self.root, hashlib.sha1(tag.encode()).hexdigest()[:20] + ".ncm")

    def tag_lock(self, tag):
        with self.lock:
            return self.locks.setdefault(tag, threading.Lock())

    def get(self, tag, dialogue=None):
        with self.lock:
            if tag in self.mems:
                self.mems.move_to_end(tag)
                return self.mems[tag]
        p = self.path(tag)
        if os.path.exists(p):
            m = Memory.load(p, embedder=self.embedder)
        elif dialogue is None:
            raise KeyError(tag)
        else:
            m = Memory(p, embedder=self.embedder,
                       config=Config(lift_next=0.75, lift_width=3.0) if dialogue else Config())
        with self.lock:
            self.mems[tag] = m
            while len(self.mems) > self.keep:
                self.mems.popitem(last=False)                       # saved after every ingest
        return m

    def ingest(self, tag, sessions):
        ids = []
        with self.tag_lock(tag):
            m = self.get(tag, dialogue=bool(sessions and sessions[0].get("dialogue")))
            seen = {(x or {}).get("session") for x in m.metas}
            for s in sessions:
                if s["sessionId"] in seen:                          # a retried request
                    continue
                msgs = [x for x in s["messages"] if str(x.get("content") or "").strip()]
                if not msgs:
                    continue
                if s.get("date"):
                    start = datetime.fromisoformat(s["date"].replace("Z", "+00:00")).timestamp()
                else:
                    start = UNDATED_START + 86400 * len(seen)
                m.new_session()
                ids += m.add_many([x["content"] for x in msgs], [x["speaker"] for x in msgs],
                                  [start + TURN_GAP * i for i in range(len(msgs))],
                                  [{"session": s["sessionId"], "date": s.get("date")}] * len(msgs))
                seen.add(s["sessionId"])
            m.save()
        return [i for i in ids if i is not None]

    def search(self, tag, query, k):
        m = self.get(tag)
        out = []
        for rank, h in enumerate(m.recall(query, k=k)):
            meta = h.meta or {}
            out.append({"content": h.text, "speaker": h.speaker, "sessionId": meta.get("session"),
                        "date": meta.get("date"), "time": h.time, "rank": rank, "score": h.score})
        return out

    def clear(self, tag):
        with self.tag_lock(tag):
            with self.lock:
                self.mems.pop(tag, None)
            if os.path.exists(self.path(tag)):
                os.remove(self.path(tag))


def serve(store, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                route = self.path.strip("/")
                if route == "health":
                    out = {"ncm": ncm.__version__, "embedder": store.embedder.name}
                elif route == "ingest":
                    out = {"ids": store.ingest(body["tag"], body["sessions"])}
                elif route == "search":
                    out = {"results": store.search(body["tag"], body["query"], int(body.get("k", 30)))}
                elif route == "clear":
                    store.clear(body["tag"])
                    out = {"ok": True}
                else:
                    self.send_error(404)
                    return
                code, data = 200, json.dumps(out).encode()
            except KeyError as e:
                code, data = 404, json.dumps({"error": f"unknown container {e}"}).encode()
            except Exception as e:                                  # reported to MemoryBench, which retries
                code, data = 500, json.dumps({"error": repr(e)}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class Server(ThreadingHTTPServer):
        request_queue_size = 256            # the default 5 refuses connections under MemoryBench's concurrency
        daemon_threads = True

    Server(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="where the per-container .ncm files go")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    store = Store(a.data, CachedEmbedder(Embedder(a.embedder)))
    store.embedder(["warm up"])                 # load the model once, before threads race to load it
    print(f"NCM {ncm.__version__} bridge on 127.0.0.1:{a.port}, data in {a.data}", flush=True)
    serve(store, a.port)


if __name__ == "__main__":
    main()
