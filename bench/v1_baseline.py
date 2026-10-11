"""Score the previous NCM (v1, main@9c1e851) on the same benchmarks.

Needs a checkout of the old code:  git worktree add --detach ../NCM-v1 9c1e851
Replays each haystack through the v1 store exactly as ollama_ncm_qwen2/chat_with_ncm.py
did (auto-state EMA, 7-d padded state, 128-d projection, default profile), with
MiniLM vectors served from the shared cache so v1 and v2 see identical embeddings.

  python -m bench.v1_baseline --dataset locomo
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.environ.get("NCM_V1", os.path.join(HERE, "..", "..", "NCM-v1"))
sys.path.insert(0, os.path.join(HERE, ".."))
from bench.data import LOADERS                                 # noqa: E402
from bench.embed import embed, model                           # noqa: E402
from bench.metrics import aggregate, score_query               # noqa: E402

TABLE = {}


class CachedST:
    """Stands in for SentenceTransformer: serves MiniLM vectors from the cache."""
    def __init__(self, *a, **k):
        pass

    def save(self, *a, **k):
        pass

    def encode(self, texts, **k):
        one = isinstance(texts, str)
        out = []
        for t in [texts] if one else texts:
            if t not in TABLE:
                TABLE[t] = model("minilm").encode(t, normalize_embeddings=True)
            out.append(TABLE[t])
        return out[0] if one else np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo")
    a = ap.parse_args()
    hays, queries = LOADERS[a.dataset]()
    docs = [t for h in hays for t in h.texts]
    TABLE.update(zip(docs, embed(docs, "minilm", "doc", tag=a.dataset)))
    qs = [q.question for q in queries]
    TABLE.update(zip(qs, embed(qs, "minilm", "query", tag=a.dataset)))

    model("minilm")  # load the real model before the stand-in shadows the import
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=CachedST)
    sys.path.insert(0, os.path.abspath(V1))
    import ncm
    assert os.path.abspath(V1) in os.path.abspath(ncm.__file__), ncm.__file__
    from ncm import MemoryEntry, MemoryStore, SentenceEncoder, retrieve_top_k_fast
    from ncm import auto_state
    auto_state._MODEL = CachedST()

    enc = SentenceEncoder(model_dir=os.path.join(HERE, ".cache", "v1model"))
    by_h = {}
    for q in queries:
        by_h.setdefault(q.haystack, []).append(q)
    rows, lat, n_q, rankings = [], 0.0, 0, {}
    for h in hays:
        if h.id not in by_h:
            continue
        store = MemoryStore()
        for t in h.texts:
            s7 = np.pad(store.auto_state.get_current_state(), (0, 2), constant_values=0.5)
            store.add(MemoryEntry(e_semantic=enc.encode(t), e_emotional=enc.encode_emotional(s7),
                                  s_snapshot=enc.encode_state(s7), timestamp=store.step, text=t))
            store.step += 1
        sess = np.asarray(h.sessions)
        order = {m.id: i for i, m in enumerate(store.get_all_safe())}
        for q in by_h[h.id]:
            s7 = np.pad(store.auto_state.get_current_state(), (0, 2), constant_values=0.5)
            t1 = time.perf_counter()
            res = retrieve_top_k_fast(enc.encode(q.question), enc.encode_emotional(s7), store,
                                      None, store.step, k=len(store))
            lat += time.perf_counter() - t1
            n_q += 1
            ranking = np.array([order[m.id] for _, _, m in res])
            rankings[f"{q.haystack}|{q.question}"] = ranking[:30].tolist()
            rows.append(score_query(ranking, q.gold, sess) | {"category": q.category})
    agg = aggregate(rows, by="category")
    agg["latency_ms"] = 1000 * lat / n_q
    print(json.dumps(agg["all"], indent=1))
    path = os.path.join(HERE, "results", f"retrieval-{a.dataset}-minilm.json")
    out = json.load(open(path))
    out["methods"]["ncm-v1"] = agg
    json.dump(out, open(path, "w"), indent=1)
    json.dump(rankings, open(path.replace("retrieval-", "rankings-v1-"), "w"))
    rp = path.replace("retrieval-", "rows-")
    r = json.load(open(rp))
    r["ncm-v1"] = rows
    json.dump(r, open(rp, "w"))


if __name__ == "__main__":
    main()
