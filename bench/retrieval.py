"""Retrieval benchmark: rank every turn of a haystack for each question.

  python -m bench.retrieval --dataset locomo --model minilm
  python -m bench.retrieval --dataset longmemeval --model bge-small --methods dense,ncm

Writes bench/results/retrieval-<dataset>-<model>.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.data import LOADERS                           # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import aggregate, paired_bootstrap, score_query  # noqa: E402
from ncm.lexical import BM25                             # noqa: E402
from ncm.embed import Reranker                           # noqa: E402
from ncm.memory import Config, score                     # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def rrf(*scores, k=60):
    out = 0
    for s in scores:
        r = np.empty(len(s))
        r[np.argsort(-s)] = np.arange(len(s))
        out = out + 1.0 / (k + r)
    return out


def methods():
    """Baselines, then the shipped scorer (ncm.memory.score) and its ablations."""
    m = {
        "dense": lambda c, q: c["D"] @ q["v"],
        "bm25": lambda c, q: c["bm25"].scores(q["text"]),
        "rrf": lambda c, q: rrf(c["D"] @ q["v"], c["bm25"].scores(q["text"])),
        "recency": lambda c, q: c["times"],
    }
    configs = {
        "ncm-v2": Config(),
        "ncm-v2-nolift": Config(lift_previous=0.0),
        "ncm-v2-dense-only": Config(lexical_weight=0.0),
        "ncm-v2-bidirectional": Config(lift_next=0.75, lift_width=3.0, lift_previous=0.75),
        "ncm-v2-forward-only": Config(lift_previous=0.0, lift_next=0.75, lift_width=1.0),
    }
    for lw in (0.25, 0.5, 1.0, 1.5):
        configs[f"ncm-v2-lex{lw}"] = Config(lexical_weight=lw)
    for name, cfg in configs.items():
        m[name] = (lambda cfg: lambda c, q: score(c["D"] @ q["v"], c["bm25"].scores(q["text"]), c["sess"], cfg))(cfg)
    m["ncm-v2-rerank"] = rerank_method(Config())
    return m


def rerank_method(cfg, top=30):
    """What Memory(reranker=True).recall does: the reranker re-orders the first stage's top 30."""
    rr = Reranker()

    def run(c, q):
        s = score(c["D"] @ q["v"], c["bm25"].scores(q["text"]), c["sess"], cfg)
        head = np.argsort(-s, kind="stable")[:top]
        r = rr(q["text"], [c["texts"][i] for i in head])
        out = s - s.max() - 1.0                       # everything else stays below the shortlist
        out[head[np.argsort(-r, kind="stable")]] = np.arange(len(head), 0, -1)
        return out
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo")
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--methods", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strip-speaker", action="store_true",
                    help="embed turns without their 'speaker: ' prefix (BM25 keeps it)")
    a = ap.parse_args()

    hays, queries = LOADERS[a.dataset]() if a.limit is None else LOADERS[a.dataset](limit=a.limit)
    all_m = methods()
    chosen = [s for s in a.methods.split(",") if s] or list(all_m)
    t0 = time.time()
    doc_texts = [t for h in hays for t in h.texts]
    if a.strip_speaker:
        D_all = embed([t.split(": ", 1)[-1] for t in doc_texts], a.model, "doc", tag=a.dataset)
    else:
        D_all = embed(doc_texts, a.model, "doc", tag=a.dataset)
    Q_all = embed([q.question for q in queries], a.model, "query", tag=a.dataset)
    print(f"embedded {len(doc_texts)} turns + {len(queries)} queries in {time.time()-t0:.0f}s", flush=True)

    ctx, off = {}, 0
    for h in hays:
        n = len(h.texts)
        sess = np.asarray(h.sessions)
        bm = BM25()
        for t in h.texts:
            bm.add(t)
        ctx[h.id] = {"D": D_all[off:off + n], "bm25": bm, "sess": sess, "texts": h.texts,
                     "times": np.asarray(h.times)}
        off += n

    rows = {m: [] for m in chosen}
    lat = {m: 0.0 for m in chosen}
    for q, v in zip(queries, Q_all):
        c = ctx[q.haystack]
        qd = {"v": v, "text": q.question, "time": q.time}
        for m in chosen:
            t1 = time.perf_counter()
            s = all_m[m](c, qd)
            ranking = np.argsort(-s, kind="stable")
            lat[m] += time.perf_counter() - t1
            rows[m].append(score_query(ranking, q.gold, c["sess"]) | {"category": q.category})

    name = f"{a.dataset}-{a.model}" + ("-nospk" if a.strip_speaker else "")
    res_path = os.path.join(RESULTS, f"retrieval-{name}.json")
    rows_path = os.path.join(RESULTS, f"rows-{name}.json")
    out = {"dataset": a.dataset, "model": a.model, "strip_speaker": a.strip_speaker,
           "n_queries": len(queries), "methods": {}}
    if os.path.exists(res_path):  # merge with earlier runs of other methods
        out["methods"] = json.load(open(res_path))["methods"]
        old_rows = json.load(open(rows_path))
        rows = {**{k: v for k, v in old_rows.items() if k not in rows}, **rows}
    for m in chosen:
        agg = aggregate(rows[m], by="category")
        agg["latency_ms"] = 1000 * lat[m] / len(queries)
        if m != "dense" and "dense" in rows and len(rows["dense"]) == len(rows[m]):
            agg["vs_dense"] = {k: paired_bootstrap([r[k] for r in rows["dense"]], [r[k] for r in rows[m]])
                               for k in ("recall@5", "ndcg@10", "mrr")}
        out["methods"][m] = agg
        a5 = agg["all"]
        print(f"{m:22s} R@5 {a5['recall@5']:.4f} R@10 {a5['recall@10']:.4f} nDCG {a5['ndcg@10']:.4f} "
              f"MRR {a5['mrr']:.4f} sessR@10 {a5['session_recall@10']:.4f} {agg['latency_ms']:.2f}ms", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(out, f, indent=1)
    with open(rows_path, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
