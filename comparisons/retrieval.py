"""Retrieval head-to-head: which memory comes back for each question?

Every conversation turn is a memory; each method ranks all of them for each question.
Scored on LoCoMo (all 1,531 questions) and LongMemEval-S (first 150 haystacks).

  python -m comparisons.retrieval            # writes comparisons/results/retrieval.json

Methods are what people actually deploy:
  vector RAG (Chroma / pgvector / FAISS style): cosine over one embedding model
  BM25 (Elasticsearch / OpenSearch default scoring)
  SPLADE (learned sparse retrieval, the modern successor to BM25)
  hybrid RRF (Elasticsearch, Weaviate, Qdrant hybrid search: reciprocal rank fusion)
  retrieve and rerank (vector top-30 -> cross-encoder, the Cohere-rerank pattern)
  NCM v1 and NCM v2
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import load_locomo                       # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import aggregate, paired_bootstrap, score_query  # noqa: E402
from comparisons.common import CACHE, lme_data, save    # noqa: E402
from ncm import Config                                   # noqa: E402
from ncm.lexical import BM25                             # noqa: E402
from ncm.memory import score                             # noqa: E402

SPLADE = "naver/splade-cocondenser-ensembledistil"
BGE_RERANKER = "BAAI/bge-reranker-base"
MSMARCO_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP = 30
# Every encoder reads at most 256 tokens of a memory (MiniLM's own limit), in batches of 32:
# longer inputs overflow a 4 GB laptop GPU into system RAM and run ~50x slower.
MAX_TOKENS = 256
BATCH = 32

_models = {}


def cross_encoder(name):
    if name not in _models:
        from sentence_transformers import CrossEncoder
        _models[name] = CrossEncoder(name, device="cuda", max_length=256)
    return _models[name]


def rerank(name, query, texts):
    return np.asarray(cross_encoder(name).predict([(query, t[:1000]) for t in texts], batch_size=64,
                                                  show_progress_bar=False), np.float32)


def splade_matrix(texts, kind, tag):
    """SPLADE vectors as a scipy CSR matrix, cached on disk."""
    import hashlib
    import scipy.sparse as sp
    key = hashlib.sha1(("\n".join(texts) + kind).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(CACHE, f"splade-{tag}-{kind}-{key}.npz")
    if os.path.exists(path):
        return sp.load_npz(path)
    if "splade" not in _models:
        from sentence_transformers import SparseEncoder
        _models["splade"] = SparseEncoder(SPLADE, device="cuda")
        _models["splade"].max_seq_length = MAX_TOKENS
    enc = _models["splade"].encode_query if kind == "query" else _models["splade"].encode_document
    rows = []
    for i in range(0, len(texts), 256):
        if i % 5120 == 0:
            print(f"  splade {tag} {kind} {i}/{len(texts)}", flush=True)
        t = enc(texts[i:i + 256], batch_size=32, convert_to_sparse_tensor=True).coalesce().cpu()
        idx, val = t.indices().numpy(), t.values().numpy()
        rows.append(sp.csr_matrix((val, (idx[0], idx[1])), shape=tuple(t.shape)))
    m = sp.vstack(rows).tocsr()
    os.makedirs(CACHE, exist_ok=True)
    sp.save_npz(path, m)
    return m


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def rrf(*scores, k=60):
    out = 0
    for s in scores:
        r = np.empty(len(s))
        r[np.argsort(-s, kind="stable")] = np.arange(len(s))
        out = out + 1.0 / (k + r)
    return out


def head_rerank(first, reranker, query, texts, top=TOP):
    """Top-`top` of `first` re-ordered by a cross-encoder; the rest keep their order below."""
    head = np.argsort(-first, kind="stable")[:top]
    out = first - first.max() - 1.0
    out[head[np.argsort(-rerank(reranker, query, [texts[i] for i in head]), kind="stable")]] = \
        np.arange(len(head), 0, -1)
    return out


METHODS = {
    # name: (description, function(c, q) -> scores)
    "vector RAG · MiniLM-L6 (22M)": lambda c, q: c["minilm"] @ q["minilm"],
    "vector RAG · bge-small (33M)": lambda c, q: c["bge-small"] @ q["bge-small"],
    "vector RAG · bge-base (110M)": lambda c, q: c["bge-base"] @ q["bge-base"],
    "BM25": lambda c, q: c["bm25"].scores(q["text"]),
    "SPLADE (learned sparse, 110M)": lambda c, q: np.asarray((c["splade"] @ q["splade"].T).todense()).ravel(),
    "hybrid RRF · BM25 + MiniLM": lambda c, q: rrf(c["minilm"] @ q["minilm"], c["bm25"].scores(q["text"])),
    "retrieve & rerank · MiniLM top-30 → bge-reranker-base": lambda c, q: head_rerank(
        c["minilm"] @ q["minilm"], BGE_RERANKER, q["text"], c["texts"]),
    "NCM v1": None,   # replayed from bench/results/rankings-v1-*.json (the v1 code is not imported)
    "NCM v2": lambda c, q: score(c["minilm"] @ q["minilm"], c["bm25"].scores(q["text"]), c["sess"], Config()),
    "NCM v2 + reranker (ms-marco-MiniLM, 22M)": lambda c, q: head_rerank(
        score(c["minilm"] @ q["minilm"], c["bm25"].scores(q["text"]), c["sess"], Config()),
        MSMARCO_RERANKER, q["text"], c["texts"]),
    "NCM v2 · bge-small embedder": lambda c, q: score(
        c["bge-small"] @ q["bge-small"], c["bm25"].scores(q["text"]), c["sess"], Config()),
    "NCM v2 · bge-base embedder": lambda c, q: score(
        c["bge-base"] @ q["bge-base"], c["bm25"].scores(q["text"]), c["sess"], Config()),
    "NCM v2 · SPLADE instead of BM25": lambda c, q: score(
        c["minilm"] @ q["minilm"], np.asarray((c["splade"] @ q["splade"].T).todense()).ravel(), c["sess"], Config()),
    "NCM v2 · SPLADE + BM25": lambda c, q: score(
        c["minilm"] @ q["minilm"], c["bm25"].scores(q["text"]), c["sess"], Config()) + 0.75 * _z(
        np.asarray((c["splade"] @ q["splade"].T).todense()).ravel()),
    "NCM v2 + bge-reranker-base": lambda c, q: head_rerank(
        score(c["minilm"] @ q["minilm"], c["bm25"].scores(q["text"]), c["sess"], Config()),
        BGE_RERANKER, q["text"], c["texts"]),
}


def build(dataset, hays, queries):
    import json
    tag = dataset
    doc_texts = [t for h in hays for t in h.texts]
    q_texts = [q.question for q in queries]
    import bench.embed as be
    if not getattr(be, "_capped", False):        # cap bge-base at 256 tokens, without loading it eagerly
        load = be.model

        def capped(name):
            m = load(name)
            m.max_seq_length = min(m.max_seq_length, MAX_TOKENS)
            return m
        be.model, be._capped = capped, True
    dense = {m: embed(doc_texts, m, "doc", tag=tag, batch_size=BATCH) for m in ("minilm", "bge-small", "bge-base")}
    qdense = {m: embed(q_texts, m, "query", tag=tag, batch_size=BATCH) for m in ("minilm", "bge-small", "bge-base")}
    sp_docs = splade_matrix(doc_texts, "doc", tag)
    sp_q = splade_matrix(q_texts, "query", tag)
    if _models.pop("splade", None) is not None:  # free the GPU for the rerankers (4 GB cards spill otherwise)
        import torch
        torch.cuda.empty_cache()
    ctx, off = {}, 0
    for h in hays:
        n = len(h.texts)
        bm = BM25()
        for t in h.texts:
            bm.add(t)
        ctx[h.id] = {m: dense[m][off:off + n] for m in dense} | {
            "bm25": bm, "sess": np.asarray(h.sessions), "texts": h.texts, "splade": sp_docs[off:off + n]}
        off += n
    qs = [{"text": q.question} | {m: qdense[m][i] for m in qdense} | {"splade": sp_q[i]}
          for i, q in enumerate(queries)]
    v1 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "bench", "results",
                                     f"rankings-v1-{dataset}-minilm.json")))
    return ctx, qs, v1


def run(dataset, hays, queries, only=None):
    ctx, qs, v1 = build(dataset, hays, queries)
    names = [m for m in METHODS if not only or m in only or m == "NCM v2"]
    rows = {m: [] for m in names}
    lat = {m: 0.0 for m in names}
    for qi, (q, qd) in enumerate(zip(queries, qs)):
        c = ctx[q.haystack]
        for name in names:
            fn = METHODS[name]
            t0 = time.perf_counter()
            if fn is None:
                top = v1[f"{q.haystack}|{q.question}"]
                seen = set(top)
                ranking = np.array(top + [i for i in range(len(c["texts"])) if i not in seen])
            else:
                ranking = np.argsort(-fn(c, qd), kind="stable")
            if qi:   # first query pays model loading
                lat[name] += time.perf_counter() - t0
            rows[name].append(score_query(ranking, q.gold, c["sess"]) | {"category": q.category})
        if qi % 200 == 0:
            print(dataset, qi, len(queries), flush=True)
    out = {}
    for name in names:
        agg = aggregate(rows[name])["all"]
        out[name] = {k: agg[k] for k in ("recall@5", "recall@10", "ndcg@10", "mrr")} | {
            "n": agg["n"], "query_ms_incl_models": 1000 * lat[name] / max(1, len(queries) - 1)}
        if name != "NCM v2":
            out[name]["ncm_v2_minus_this_r5"] = paired_bootstrap(
                [r["recall@5"] for r in rows[name]], [r["recall@5"] for r in rows["NCM v2"]])
    return out


def model_costs():
    """Encoding cost per memory and per query, CPU and GPU, on real LoCoMo turns."""
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
    from bench.embed import MODELS
    hays, _ = load_locomo()
    texts = [t for h in hays for t in h.texts][:512]
    out = {}
    for device in ("cpu", "cuda"):
        encoders = {name: SentenceTransformer(MODELS[name][0], device=device) for name in ("minilm", "bge-small", "bge-base")}
        encoders["splade"] = SparseEncoder(SPLADE, device=device)
        for name, m in encoders.items():
            m.encode(texts[:16])
            t0 = time.perf_counter()
            m.encode(texts, batch_size=64)
            per_doc = (time.perf_counter() - t0) / len(texts)
            t0 = time.perf_counter()
            for t in texts[:20]:
                m.encode([t])
            per_query = (time.perf_counter() - t0) / 20
            out[f"{name}@{device}"] = {"ms_per_memory_batched": 1000 * per_doc, "ms_per_query": 1000 * per_query}
        for name in (MSMARCO_RERANKER, BGE_RERANKER):
            ce = CrossEncoder(name, device=device, max_length=256)
            ce.predict([("q", t) for t in texts[:30]])
            t0 = time.perf_counter()
            for i in range(5):
                ce.predict([("what did she say about the trip?", t) for t in texts[i * 30:(i + 1) * 30]])
            out[f"{name.split('/')[-1]}-30pairs@{device}"] = {"ms_per_query": 1000 * (time.perf_counter() - t0) / 5}
        print(out, flush=True)
    return out


def index_bytes_per_memory(dataset, hays):
    doc_texts = [t for h in hays for t in h.texts]
    bm = BM25()
    for t in doc_texts:
        bm.add(t)
    postings = sum(p[2] for p in bm.postings.values())
    sp_nnz = splade_matrix(doc_texts, "doc", dataset).nnz
    n = len(doc_texts)
    text = sum(len(t.encode("utf-8")) for t in doc_texts) / n
    return {"text": text, "minilm_fp32": 384 * 4, "bge-base_fp32": 768 * 4,
            "bm25_postings": 8 * postings / n, "splade": 8 * sp_nnz / n}


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # names contain arrows; Windows consoles default to cp1252
    import json
    from comparisons.common import RESULTS
    path = os.path.join(RESULTS, "retrieval.json")
    out = json.load(open(path)) if os.path.exists(path) else {}   # resume: each part is saved when done
    only = [m for m in METHODS if m in sys.argv[1:]] or None
    if only:   # add methods to an existing run: python -m comparisons.retrieval "NCM v2 · SPLADE instead of BM25"
        for ds, key, loader in (("locomo", "locomo", load_locomo), ("longmemeval", "longmemeval_150", lme_data)):
            hays, queries = loader()
            out[key].update({k: v for k, v in run(ds, hays, queries, only).items() if k in only})
            save("retrieval.json", out)
        return
    if "locomo" not in out:
        hays, queries = load_locomo()
        out["locomo"] = run("locomo", hays, queries)
        out["index_bytes_per_memory_locomo"] = index_bytes_per_memory("locomo", hays)
        save("retrieval.json", out)
    if "longmemeval_150" not in out:
        hays, queries = lme_data()
        out["longmemeval_150"] = run("longmemeval", hays, queries)
        save("retrieval.json", out)
    if "model_costs" not in out:
        out["model_costs"] = model_costs()
        save("retrieval.json", out)
    for ds in ("locomo", "longmemeval_150"):
        for name, r in out[ds].items():
            print(f"{ds:16s} {name:58s} R@5 {r['recall@5']:.3f} R@10 {r['recall@10']:.3f} "
                  f"nDCG {r['ndcg@10']:.3f} MRR {r['mrr']:.3f} {r['query_ms_incl_models']:.1f} ms")


if __name__ == "__main__":
    main()
