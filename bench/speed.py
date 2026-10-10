"""Speed and footprint: NCM's own overhead, with the encoder measured separately.

  python -m bench.speed                 # v2 at 1k..1M memories, plus encoder cost
  python -m bench.speed --v1            # the same for v1 (needs ../NCM-v1), in its own process
  python -m bench.speed --encoder       # embedding model cost, CPU and GPU

Memories are synthetic Zipf-distributed sentences with random unit vectors, so the
numbers isolate NCM's indexing and scoring from the embedding model.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DIM = 384


def corpus(n, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(20000)]
    p = 1.0 / np.arange(1, len(vocab) + 1)
    p /= p.sum()
    words = rng.choice(len(vocab), size=(n, 15), p=p)
    return [" ".join(vocab[j] for j in row) for row in words], rng


def unit(rng, n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Table:
    """Embedder stand-in: a precomputed vector per text."""
    name = "synthetic"

    def __init__(self, texts, vecs):
        self.t = dict(zip(texts, vecs))
        self.q = {}

    def __call__(self, texts, query=False):
        return np.stack([self._get(t, query) for t in texts])

    def _get(self, t, query):
        if query:
            return self.q[t]
        key = t.split(": ", 1)[-1]
        if key not in self.t:   # e.g. the affect anchor sentences
            self.t[key] = unit(np.random.default_rng(abs(hash(key)) % 2 ** 32), 1)[0]
        return self.t[key]


def pct(xs):
    return {"p50_ms": float(np.percentile(xs, 50) * 1e3), "p95_ms": float(np.percentile(xs, 95) * 1e3)}


def v2(sizes):
    sys.path.insert(0, os.path.join(HERE, ".."))
    from ncm import Memory
    out = {}
    for n in sizes:
        texts, rng = corpus(n)
        vecs = unit(rng, n)
        emb = Table(texts, vecs)
        m = Memory(embedder=emb)
        t0 = time.perf_counter()
        for i in range(0, n, 1000):
            m.add_many(texts[i:i + 1000], times=np.arange(i, min(i + 1000, n)) * 60.0)
        add_batch = time.perf_counter() - t0
        queries = [" ".join(texts[i].split()[:6]) for i in rng.integers(0, n, 50)]
        for q in queries:
            emb.q[q] = unit(rng, 1)[0]
        lat = []
        for q in queries:
            t1 = time.perf_counter()
            m.recall(q, k=10)
            lat.append(time.perf_counter() - t1)
        single = []
        extra, ev = corpus(200, seed=1)
        emb.t.update(zip(extra, unit(ev, 200)))
        for t in extra:
            t1 = time.perf_counter()
            m.add(t, time=1e9)
            single.append(time.perf_counter() - t1)
        path = os.path.join(tempfile.gettempdir(), f"ncm-speed-{n}.ncm")
        t1 = time.perf_counter()
        m.save(path)
        save_s = time.perf_counter() - t1
        t1 = time.perf_counter()
        Memory(path, embedder=emb)
        load_s = time.perf_counter() - t1
        out[n] = {"add_many_per_item_us": 1e6 * add_batch / n, "add_single": pct(single),
                  "recall": pct(lat), "save_s": save_s, "load_s": load_s,
                  "file_mb": os.path.getsize(path) / 2 ** 20,
                  "ram_mb_vectors": m.embeddings.nbytes / 2 ** 20}
        os.remove(path)
        print(n, json.dumps(out[n]), flush=True)
    return out


def v1(sizes):
    import types
    sys.path.insert(0, os.environ.get("NCM_V1", os.path.join(HERE, "..", "..", "NCM-v1")))
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=None)
    from ncm import MemoryEntry, MemoryStore, retrieve_top_k_fast
    out = {}
    for n in sizes:
        texts, rng = corpus(n)
        sem = unit(rng, n)[:, :128] * np.sqrt(3)
        sem /= np.linalg.norm(sem, axis=1, keepdims=True)
        store = MemoryStore()
        store.profile.max_size = n + 1
        t0 = time.perf_counter()
        for i, t in enumerate(texts):   # v1 add() also re-encodes the text for its auto-state; skipped here
            store.add(MemoryEntry(e_semantic=sem[i], e_emotional=np.ones(3, np.float32) / np.sqrt(3),
                                  s_snapshot=np.ones(7, np.float32) / np.sqrt(7), timestamp=i, text=t),
                      update_auto_state=False, contradiction_check=False)
        add_s = time.perf_counter() - t0
        lat = []
        for q in sem[rng.integers(0, n, 50)]:
            t1 = time.perf_counter()
            retrieve_top_k_fast(q, np.ones(3, np.float32) / np.sqrt(3), store, None, n, k=10)
            lat.append(time.perf_counter() - t1)
        out[n] = {"add_single_per_item_us": 1e6 * add_s / n, "recall": pct(lat)}
        print(n, json.dumps(out[n]), flush=True)
    return out


def encoder_cost():
    from sentence_transformers import SentenceTransformer
    texts, _ = corpus(512)
    res = {}
    for device in ("cpu", "cuda"):
        for name in ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"):
            m = SentenceTransformer(name, device=device)
            m.encode(texts[:8])
            one = []
            for t in texts[:30]:
                t1 = time.perf_counter()
                m.encode([t])
                one.append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            m.encode(texts, batch_size=64)
            res[f"{name.split('/')[-1]}@{device}"] = {
                "single": pct(one), "batch_per_item_ms": 1e3 * (time.perf_counter() - t1) / len(texts)}
            print(res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true")
    ap.add_argument("--encoder", action="store_true", help="time the embedding models (run on an idle GPU)")
    ap.add_argument("--sizes", default="1000,10000,100000,1000000")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    path = os.path.join(RES, "speed.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    if a.v1:
        data["v1"] = v1([s for s in sizes if s <= 100000])
    elif a.encoder:
        data["encoder"] = encoder_cost()
    else:
        data["v2"] = v2(sizes)
    os.makedirs(RES, exist_ok=True)
    json.dump(data, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
