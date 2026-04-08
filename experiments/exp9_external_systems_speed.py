"""
Experiment 9: Speed comparison across external systems and NCM.

Focus: retrieval speed (same corpus, same query batch), with setup/index time included.

Outputs:
- results/exp9_external_systems_speed.json
- results/exp9_external_systems_speed.txt
- results/exp9_external_systems_speed_latency.png
- results/exp9_external_systems_speed_qps.png
"""

import os
import sys
import json
import time
import random
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.retrieval import retrieve_top_k, retrieve_top_k_fast

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STATE_ARCHETYPES = {
    "calm_happy":      np.array([0.7, 0.8, 0.2, 0.8, 0.3, 0.7, 0.2], dtype=np.float32),
    "stressed_angry":  np.array([0.9, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9], dtype=np.float32),
    "sad_withdrawn":   np.array([0.2, 0.3, 0.6, 0.3, 0.7, 0.3, 0.4], dtype=np.float32),
    "excited_curious": np.array([0.9, 0.7, 0.3, 0.9, 0.4, 0.8, 0.3], dtype=np.float32),
    "fearful":         np.array([0.6, 0.2, 0.8, 0.3, 0.9, 0.2, 0.8], dtype=np.float32),
    "neutral":         np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    "confident":       np.array([0.8, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], dtype=np.float32),
    "exhausted":       np.array([0.1, 0.4, 0.7, 0.2, 0.6, 0.3, 0.6], dtype=np.float32),
}

SEMANTIC_CATEGORIES = {
    "betrayal": [
        "friend lied about the project deadline",
        "colleague took credit for my work",
        "trusted partner broke their promise",
        "discovered someone talking behind my back",
        "team member sabotaged the presentation",
        "roommate read my private messages",
        "mentor gave away my research idea",
        "business partner secretly withdrew funds",
    ],
    "achievement": [
        "finished the marathon in under 4 hours",
        "got promoted to team lead",
        "published my first research paper",
        "won the hackathon competition",
        "scored highest in the final exam",
        "completed the certification with distinction",
        "delivered the keynote speech successfully",
        "startup reached profitability this quarter",
    ],
    "loss": [
        "lost my wallet on the train",
        "pet cat passed away last week",
        "failed the certification exam",
        "project got cancelled after months of work",
        "missed the flight and lost the booking",
        "hard drive crashed and lost all data",
        "best friend moved to another country",
        "favorite restaurant closed permanently",
    ],
    "discovery": [
        "found an amazing new algorithm for sorting",
        "discovered a shortcut through the park",
        "learned about quantum computing principles",
        "found a bug that was causing data corruption",
        "realized the solution was simpler than expected",
        "stumbled upon a hidden coffee shop downtown",
        "figured out why the neural network was diverging",
        "discovered a pattern in the customer data",
    ],
    "social": [
        "had a great conversation with the new intern",
        "team lunch was really enjoyable today",
        "met an interesting person at the conference",
        "helped a stranger with directions downtown",
        "reconnected with an old school friend online",
        "organized a successful team building event",
        "received a heartfelt thank you note",
        "introduced two friends who became a couple",
    ],
    "conflict": [
        "argument with manager about project priorities",
        "disagreement about the technical approach taken",
        "heated debate during the team meeting today",
        "customer complaint escalated to upper management",
        "deadline conflict between two critical projects",
        "coworker refused to follow the agreed process",
        "budget dispute with the finance department",
        "code review turned into a personal argument",
    ],
}

QUERY_TEXTS = {
    "betrayal": ["someone I trusted deceived me", "a teammate betrayed confidence", "I was undermined by close collaborators"],
    "achievement": ["I accomplished an important goal", "a major success happened recently", "I achieved a milestone through effort"],
    "loss": ["I am dealing with an important loss", "something valuable was lost", "I experienced a painful setback"],
    "discovery": ["I uncovered something new and useful", "there was an insightful technical discovery", "I found a meaningful new pattern"],
    "social": ["I had positive social interaction", "there was meaningful connection with people", "a relationship-focused moment happened"],
    "conflict": ["there was interpersonal or work conflict", "I had a serious disagreement", "a tense dispute occurred"],
}


def percentile(vals, p):
    return float(np.percentile(np.array(vals, dtype=np.float64), p))


def build_store(encoder: SentenceEncoder, n_per_pair=25, seed=42):
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=50000))
    ts = 0
    for cat, texts in SEMANTIC_CATEGORIES.items():
        for sname, sbase in STATE_ARCHETYPES.items():
            for i in range(n_per_pair):
                text = texts[i % len(texts)]
                state = np.clip(sbase + rng.uniform(-0.05, 0.05, size=sbase.shape), 0.0, 1.0).astype(np.float32)
                e_sem = encoder.encode(text)
                e_sem = e_sem + rng.randn(e_sem.shape[0]).astype(np.float32) * 0.005
                e_sem = e_sem / (np.linalg.norm(e_sem) + 1e-8)
                mem = MemoryEntry(
                    e_semantic=e_sem,
                    e_emotional=encoder.encode_emotional(state),
                    s_snapshot=encoder.encode_state(state),
                    timestamp=ts,
                    text=text,
                    tags=[cat, sname],
                )
                store.add(mem)
                ts += 1
    store.step = ts
    return store


def build_query_batch(encoder: SentenceEncoder, size=1000, seed=123):
    rng = random.Random(seed)
    state_names = list(STATE_ARCHETYPES.keys())
    all_prompts = [p for arr in QUERY_TEXTS.values() for p in arr]

    queries = []
    for _ in range(size):
        text = rng.choice(all_prompts)
        sname = rng.choice(state_names)
        svec = STATE_ARCHETYPES[sname]
        queries.append({
            "text": text,
            "state_name": sname,
            "qs": encoder.encode(text),
            "qe": encoder.encode_emotional(svec),
            "qst": encoder.encode_state(svec),
        })
    return queries


def run_speed(name, setup_fn, query_fn, queries, warmup=50):
    t_setup0 = time.perf_counter()
    ctx = setup_fn()
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    for i in range(min(warmup, len(queries))):
        query_fn(ctx, queries[i], k=10)

    latencies = []
    t0 = time.perf_counter()
    for q in queries:
        q0 = time.perf_counter()
        query_fn(ctx, q, k=10)
        latencies.append((time.perf_counter() - q0) * 1000.0)
    total_ms = (time.perf_counter() - t0) * 1000.0

    avg = mean(latencies)
    return {
        "system": name,
        "setup_ms": round(setup_ms, 4),
        "avg_latency_ms": round(avg, 4),
        "p50_latency_ms": round(percentile(latencies, 50), 4),
        "p95_latency_ms": round(percentile(latencies, 95), 4),
        "p99_latency_ms": round(percentile(latencies, 99), 4),
        "throughput_qps": round(1000.0 / max(1e-9, avg), 2),
        "batch_total_ms": round(total_ms, 4),
        "queries": len(queries),
    }


def run():
    random.seed(42)
    np.random.seed(42)

    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    store = build_store(encoder, n_per_pair=25, seed=42)

    # enable non-uniform strengths
    for mid in list(store._memories.keys())[::17]:
        store.reinforce(mid, amount=0.6)

    queries = build_query_batch(encoder, size=1000, seed=123)

    # shared static caches for simple baselines
    store._rebuild_cache()
    ids = store._id_order
    docs = [store._memories[mid].text for mid in ids]
    sem = store._sem_cache
    ts = store._ts_cache

    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(os.path.join(ROOT_DIR, "models", "all-MiniLM-L6-v2"))

    # precompute dense query vectors for retrieval-only benchmark
    q_dense = dense_model.encode([q["text"] for q in queries], convert_to_numpy=True, show_progress_bar=False)
    q_dense = q_dense / (np.linalg.norm(q_dense, axis=1, keepdims=True) + 1e-8)
    for i, q in enumerate(queries):
        q["q_dense"] = q_dense[i]

    systems = []

    # BM25
    def bm25_setup():
        tok = [d.lower().split() for d in docs]
        return {"bm25": BM25Okapi(tok)}

    def bm25_query(ctx, q, k=10):
        s = ctx["bm25"].get_scores(q["text"].lower().split())
        _ = np.argsort(-s)[:k]

    systems.append(("bm25_text", bm25_setup, bm25_query))

    # TF-IDF
    def tfidf_setup():
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        x = v.fit_transform(docs)
        return {"vec": v, "X": x}

    def tfidf_query(ctx, q, k=10):
        qv = ctx["vec"].transform([q["text"]])
        s = (ctx["X"] @ qv.T).toarray().reshape(-1)
        _ = np.argsort(-s)[:k]

    systems.append(("tfidf_cosine", tfidf_setup, tfidf_query))

    # Dense SBERT cosine
    def dense_setup():
        d = dense_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        return {"doc": d}

    def dense_query(ctx, q, k=10):
        s = ctx["doc"] @ q["q_dense"]
        _ = np.argsort(-s)[:k]

    systems.append(("dense_sbert_cosine", dense_setup, dense_query))

    # RAG semantic-only
    def rag_sem_setup():
        return {}

    def rag_sem_query(ctx, q, k=10):
        d = 1.0 - (sem @ q["qs"])
        _ = np.argsort(d)[:k]

    systems.append(("rag_semantic_only", rag_sem_setup, rag_sem_query))

    # RAG semantic+recency
    def rag_sr_setup():
        return {}

    def rag_sr_query(ctx, q, k=10):
        d_sem = np.clip(1.0 - (sem @ q["qs"]), 0.0, 1.0)
        d_time = 1.0 - np.exp(-0.001 * np.maximum(0, store.step - ts))
        d = 0.8 * d_sem + 0.2 * d_time
        _ = np.argsort(d)[:k]

    systems.append(("rag_semantic_recency", rag_sr_setup, rag_sr_query))

    # recency-only
    def recency_setup():
        return {}

    def recency_query(ctx, q, k=10):
        d = np.maximum(0, store.step - ts)
        _ = np.argsort(d)[:k]

    systems.append(("recency_only", recency_setup, recency_query))

    # NCM full
    def ncm_setup():
        return {}

    def ncm_query(ctx, q, k=10):
        _ = retrieve_top_k(q["qs"], q["qe"], store, q["qst"], store.step, k=k, use_strength=True)

    systems.append(("ncm_full", ncm_setup, ncm_query))

    # NCM cached
    def ncm_cached_setup():
        store._rebuild_cache()
        return {}

    def ncm_cached_query(ctx, q, k=10):
        _ = retrieve_top_k_fast(q["qs"], q["qe"], store, q["qst"], store.step, k=k, use_strength=True)

    systems.append(("ncm_cached_full", ncm_cached_setup, ncm_cached_query))

    results = []
    for name, setup_fn, query_fn in systems:
        print(f"Benchmarking: {name}")
        results.append(run_speed(name, setup_fn, query_fn, queries, warmup=50))

    by_latency = sorted(results, key=lambda r: r["avg_latency_ms"])
    by_qps = sorted(results, key=lambda r: r["throughput_qps"], reverse=True)

    out = {
        "dataset": {"memories": len(store), "queries": len(queries)},
        "results": results,
        "ranking_by_latency": by_latency,
        "ranking_by_qps": by_qps,
        "notes": "Retrieval-focused benchmark. Query embeddings are precomputed for vector systems; lexical systems include transform/tokenization in query path.",
    }

    json_path = os.path.join(RESULTS_DIR, "exp9_external_systems_speed.json")
    txt_path = os.path.join(RESULTS_DIR, "exp9_external_systems_speed.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [
        "EXP9: External Systems Speed Comparison",
        "=" * 38,
        f"Dataset: {len(store)} memories, {len(queries)} queries",
        "",
        "Ranking by avg latency (lower is better)",
    ]
    for i, r in enumerate(by_latency, 1):
        lines.append(
            f"{i}. {r['system']}: avg={r['avg_latency_ms']:.4f}ms, p95={r['p95_latency_ms']:.4f}ms, "
            f"p99={r['p99_latency_ms']:.4f}ms, qps={r['throughput_qps']:.2f}, setup={r['setup_ms']:.2f}ms"
        )

    lines.append("")
    lines.append("Ranking by throughput (higher is better)")
    for i, r in enumerate(by_qps, 1):
        lines.append(
            f"{i}. {r['system']}: qps={r['throughput_qps']:.2f}, avg={r['avg_latency_ms']:.4f}ms, setup={r['setup_ms']:.2f}ms"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # plots
    names = [r["system"] for r in by_latency]
    avg_vals = [r["avg_latency_ms"] for r in by_latency]
    p95_vals = [r["p95_latency_ms"] for r in by_latency]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(names))
    w = 0.35
    plt.bar(x - w / 2, avg_vals, width=w, label="avg latency ms")
    plt.bar(x + w / 2, p95_vals, width=w, label="p95 latency ms")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.title("EXP9 Latency Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp9_external_systems_speed_latency.png"))
    plt.close()

    qps_sorted = sorted(results, key=lambda r: r["throughput_qps"], reverse=True)
    names_q = [r["system"] for r in qps_sorted]
    qps_vals = [r["throughput_qps"] for r in qps_sorted]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(names_q, qps_vals, color="#2E8B57")
    plt.gca().invert_yaxis()
    plt.title("EXP9 Throughput (QPS)")
    plt.xlabel("Queries per second")
    for b, v in zip(bars, qps_vals):
        plt.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp9_external_systems_speed_qps.png"))
    plt.close()

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp9_external_systems_speed_latency.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp9_external_systems_speed_qps.png')}")


if __name__ == "__main__":
    run()
