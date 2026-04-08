"""
Experiment 7: Standardized ranking + visualizations for memory systems.

Goal:
- Evaluate systems using commonly used retrieval metrics from IR/RAG practice
- Rank systems with a transparent composite score
- Produce publication-style visualizations

Systems compared:
- rag_semantic_only
- rag_semantic_recency
- recency_only
- semantic_emotional
- ncm_full
- ncm_cached_full

Outputs:
- results/exp7_standard_ranking.json
- results/exp7_standard_ranking.txt
- results/exp7_quality_metrics.png
- results/exp7_efficiency_metrics.png
- results/exp7_overall_ranking.png
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.retrieval import retrieve_top_k, retrieve_top_k_fast, retrieve_semantic_emotional

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
    "betrayal": [
        "someone I trusted deceived me",
        "a teammate betrayed confidence",
        "I was undermined by close collaborators",
    ],
    "achievement": [
        "I accomplished an important goal",
        "a major success happened recently",
        "I achieved a milestone through effort",
    ],
    "loss": [
        "I am dealing with an important loss",
        "something valuable was lost",
        "I experienced a painful setback",
    ],
    "discovery": [
        "I uncovered something new and useful",
        "there was an insightful technical discovery",
        "I found a meaningful new pattern",
    ],
    "social": [
        "I had positive social interaction",
        "there was meaningful connection with people",
        "a relationship-focused moment happened",
    ],
    "conflict": [
        "there was interpersonal or work conflict",
        "I had a serious disagreement",
        "a tense dispute occurred",
    ],
}


def build_store(encoder: SentenceEncoder, n_per_pair: int = 25, seed: int = 42):
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=50000))
    metadata = {}

    ts = 0
    for cat_name, texts in SEMANTIC_CATEGORIES.items():
        for state_name, state_base in STATE_ARCHETYPES.items():
            for i in range(n_per_pair):
                text = texts[i % len(texts)]
                state = np.clip(state_base + rng.uniform(-0.05, 0.05, size=state_base.shape), 0.0, 1.0).astype(np.float32)

                e_sem = encoder.encode(text)
                e_sem = e_sem + rng.randn(e_sem.shape[0]).astype(np.float32) * 0.005
                e_sem = e_sem / (np.linalg.norm(e_sem) + 1e-8)

                mem = MemoryEntry(
                    e_semantic=e_sem,
                    e_emotional=encoder.encode_emotional(state),
                    s_snapshot=encoder.encode_state(state),
                    timestamp=ts,
                    text=text,
                    tags=[cat_name, state_name],
                )
                store.add(mem)
                metadata[mem.id] = {"category": cat_name, "state": state_name, "text": text}
                ts += 1

    store.step = ts
    return store, metadata


def build_queries(seed: int = 123):
    rng = random.Random(seed)
    state_names = list(STATE_ARCHETYPES.keys())
    queries = []
    for cat, prompts in QUERY_TEXTS.items():
        for prompt in prompts:
            for _ in range(8):
                sname = rng.choice(state_names)
                queries.append({
                    "text": prompt,
                    "category": cat,
                    "state_name": sname,
                    "state_vec": STATE_ARCHETYPES[sname],
                })
    rng.shuffle(queries)
    return queries


def minmax(values):
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


def graded_rel(meta, q_cat, q_state):
    if meta["category"] == q_cat and meta["state"] == q_state:
        return 2
    if meta["category"] == q_cat:
        return 1
    return 0


def dcg(grades):
    out = 0.0
    for i, g in enumerate(grades):
        out += (2**g - 1) / np.log2(i + 2)
    return out


def evaluate_ranked(mids, metadata, q_cat, q_state, k):
    topk = mids[:k]
    binary = [1 if metadata[mid]["category"] == q_cat else 0 for mid in topk]
    graded = [graded_rel(metadata[mid], q_cat, q_state) for mid in topk]

    # Precision@k
    p_at_k = sum(binary) / k

    # Hit@k
    hit_k = 1.0 if any(binary) else 0.0

    # MRR@k (first category match)
    rr = 0.0
    for i, b in enumerate(binary):
        if b:
            rr = 1.0 / (i + 1)
            break

    # Recall@k over all category-relevant memories
    total_rel = sum(1 for m in metadata.values() if m["category"] == q_cat)
    rec_k = sum(binary) / max(1, total_rel)

    # AP@k
    num_hits = 0
    precisions = []
    for i, b in enumerate(binary, start=1):
        if b:
            num_hits += 1
            precisions.append(num_hits / i)
    ap_k = float(sum(precisions) / max(1, total_rel))

    # NDCG@k with graded relevance
    ideal_grades = sorted((graded_rel(m, q_cat, q_state) for m in metadata.values()), reverse=True)[:k]
    idcg = dcg(ideal_grades)
    ndcg_k = dcg(graded) / idcg if idcg > 0 else 0.0

    # state precision@k (strict)
    state_prec_k = sum(1 for mid in topk if metadata[mid]["category"] == q_cat and metadata[mid]["state"] == q_state) / k

    return {
        "precision": p_at_k,
        "hit": hit_k,
        "mrr": rr,
        "recall": rec_k,
        "map": ap_k,
        "ndcg": ndcg_k,
        "state_precision": state_prec_k,
    }


def make_retrievers(store):
    store._rebuild_cache()
    sem = store._sem_cache
    ts = store._ts_cache
    ids = store._id_order

    def rag_semantic_only(qs, qe, qst, step, k):
        d = 1.0 - (sem @ qs)
        idx = np.argsort(d)[:k]
        return [ids[i] for i in idx]

    def rag_semantic_recency(qs, qe, qst, step, k, alpha=0.8, decay=0.001):
        d_sem = np.clip(1.0 - (sem @ qs), 0.0, 1.0)
        dt = np.maximum(0, step - ts)
        d_time = 1.0 - np.exp(-decay * dt)
        d = alpha * d_sem + (1.0 - alpha) * d_time
        idx = np.argsort(d)[:k]
        return [ids[i] for i in idx]

    def recency_only(qs, qe, qst, step, k):
        d = np.maximum(0, step - ts)
        idx = np.argsort(d)[:k]
        return [ids[i] for i in idx]

    def semantic_emotional(qs, qe, qst, step, k):
        return [m.id for _, m in retrieve_semantic_emotional(qs, qe, store, k=k)]

    def ncm_full(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k(qs, qe, store, qst, step, k=k, use_strength=True)]

    def ncm_cached_full(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k_fast(qs, qe, store, qst, step, k=k, use_strength=True)]

    return {
        "rag_semantic_only": rag_semantic_only,
        "rag_semantic_recency": rag_semantic_recency,
        "recency_only": recency_only,
        "semantic_emotional": semantic_emotional,
        "ncm_full": ncm_full,
        "ncm_cached_full": ncm_cached_full,
    }


def system_index_footprint(store):
    store._rebuild_cache()
    sem_bytes = int(store._sem_cache.nbytes)
    emo_bytes = int(store._emo_cache.nbytes)
    state_bytes = int(store._state_cache.nbytes)
    ts_bytes = int(store._ts_cache.nbytes)
    str_bytes = int(store._str_cache.nbytes)
    return {
        "rag_semantic_only": sem_bytes,
        "rag_semantic_recency": sem_bytes + ts_bytes,
        "recency_only": ts_bytes,
        "semantic_emotional": sem_bytes + emo_bytes,
        "ncm_full": sem_bytes + emo_bytes + state_bytes + ts_bytes + str_bytes,
        "ncm_cached_full": sem_bytes + emo_bytes + state_bytes + ts_bytes + str_bytes,
    }


def run():
    random.seed(42)
    np.random.seed(42)

    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    store, metadata = build_store(encoder, n_per_pair=25, seed=42)

    # make strength meaningful
    mem_ids = list(store._memories.keys())
    for mid in mem_ids[::17]:
        store.reinforce(mid, amount=0.6)

    queries = build_queries(seed=123)
    retrievers = make_retrievers(store)
    footprints = system_index_footprint(store)

    ks = (10,)  # standardized top-k for ranking summary
    systems_order = [
        "rag_semantic_only",
        "rag_semantic_recency",
        "recency_only",
        "semantic_emotional",
        "ncm_full",
        "ncm_cached_full",
    ]

    rows = []

    for s in systems_order:
        print(f"Evaluating: {s}")
        vals = {
            "precision@10": [],
            "hit@10": [],
            "mrr@10": [],
            "recall@10": [],
            "map@10": [],
            "ndcg@10": [],
            "state_precision@10": [],
            "latency_ms": [],
        }

        for q in queries:
            qs = encoder.encode(q["text"])
            qe = encoder.encode_emotional(q["state_vec"])
            qst = encoder.encode_state(q["state_vec"])

            t0 = time.perf_counter()
            mids = retrievers[s](qs, qe, qst, store.step, k=max(ks))
            vals["latency_ms"].append((time.perf_counter() - t0) * 1000.0)

            metrics = evaluate_ranked(mids, metadata, q["category"], q["state_name"], 10)
            vals["precision@10"].append(metrics["precision"])
            vals["hit@10"].append(metrics["hit"])
            vals["mrr@10"].append(metrics["mrr"])
            vals["recall@10"].append(metrics["recall"])
            vals["map@10"].append(metrics["map"])
            vals["ndcg@10"].append(metrics["ndcg"])
            vals["state_precision@10"].append(metrics["state_precision"])

        rows.append({
            "system": s,
            "precision@10": round(mean(vals["precision@10"]), 4),
            "hit@10": round(mean(vals["hit@10"]), 4),
            "mrr@10": round(mean(vals["mrr@10"]), 4),
            "recall@10": round(mean(vals["recall@10"]), 4),
            "map@10": round(mean(vals["map@10"]), 4),
            "ndcg@10": round(mean(vals["ndcg@10"]), 4),
            "state_precision@10": round(mean(vals["state_precision@10"]), 4),
            "latency_ms_avg": round(mean(vals["latency_ms"]), 4),
            "throughput_qps_est": round(1000.0 / max(1e-9, mean(vals["latency_ms"])), 2),
            "index_footprint_bytes": int(footprints[s]),
        })

    # composite score
    quality_cols = ["ndcg@10", "recall@10", "mrr@10", "hit@10", "map@10", "state_precision@10"]
    latencies = [r["latency_ms_avg"] for r in rows]
    throughputs = [r["throughput_qps_est"] for r in rows]
    footprints_b = [r["index_footprint_bytes"] for r in rows]

    lat_norm = minmax(latencies)
    thr_norm = minmax(throughputs)
    mem_norm = minmax(footprints_b)

    # practical weighting (from common IR+systems practice):
    # relevance 60%, state-memory quality 20%, efficiency 20%
    for i, r in enumerate(rows):
        relevance = (
            0.20 * r["ndcg@10"] +
            0.15 * r["recall@10"] +
            0.10 * r["mrr@10"] +
            0.05 * r["hit@10"] +
            0.10 * r["map@10"]
        )
        state_q = 0.20 * r["state_precision@10"]
        efficiency = (
            0.10 * thr_norm[i] +
            0.05 * (1.0 - lat_norm[i]) +
            0.05 * (1.0 - mem_norm[i])
        )
        r["composite_score"] = round(relevance + state_q + efficiency, 4)

    ranked = sorted(rows, key=lambda x: x["composite_score"], reverse=True)

    # state-conditioning behavior (same semantic query, different states)
    compare_systems = ["rag_semantic_only", "rag_semantic_recency", "semantic_emotional", "ncm_full", "ncm_cached_full"]
    anchor_text = "I experienced betrayal by someone close"
    q_sem = encoder.encode(anchor_text)
    state_pairs = [("calm_happy", "stressed_angry"), ("confident", "fearful"), ("neutral", "exhausted")]

    conditioning = {}
    for a, b in state_pairs:
        qe_a, qst_a = encoder.encode_emotional(STATE_ARCHETYPES[a]), encoder.encode_state(STATE_ARCHETYPES[a])
        qe_b, qst_b = encoder.encode_emotional(STATE_ARCHETYPES[b]), encoder.encode_state(STATE_ARCHETYPES[b])
        row = {}
        for s in compare_systems:
            ids_a = retrievers[s](q_sem, qe_a, qst_a, store.step, 10)
            ids_b = retrievers[s](q_sem, qe_b, qst_b, store.step, 10)
            row[f"{s}_jaccard"] = round(jaccard_distance(ids_a, ids_b), 4)
        conditioning[f"{a}_vs_{b}"] = row

    references = [
        "https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html",
        "https://www.elastic.co/docs/reference/elasticsearch/rest-apis/search-rank-eval",
        "https://openreview.net/forum?id=wCu6T5xFjeJ",
        "https://github.com/beir-cellar/beir/wiki/Metrics-available",
        "https://recommenders-team.github.io/recommenders/evaluation.html",
        "https://www.tensorflow.org/recommenders/examples/basic_retrieval",
        "https://learn.microsoft.com/en-us/azure/search/search-performance-analysis",
        "https://mlcommons.org/benchmarks/inference-datacenter/",
    ]

    out = {
        "dataset": {
            "memories": len(store),
            "queries": len(queries),
            "query_type": "category paraphrases (not exact memory strings)",
        },
        "metrics_used": [
            "precision@10", "hit@10", "mrr@10", "recall@10", "map@10", "ndcg@10",
            "state_precision@10", "latency_ms_avg", "throughput_qps_est", "index_footprint_bytes"
        ],
        "composite_formula": {
            "relevance": "0.20*NDCG@10 + 0.15*Recall@10 + 0.10*MRR@10 + 0.05*Hit@10 + 0.10*MAP@10",
            "state": "0.20*StatePrecision@10",
            "efficiency": "0.10*norm(Throughput) + 0.05*(1-norm(Latency)) + 0.05*(1-norm(Memory))",
            "total": "Composite = relevance + state + efficiency",
        },
        "systems": rows,
        "ranking": ranked,
        "state_conditioning": conditioning,
        "references": references,
    }

    json_path = os.path.join(RESULTS_DIR, "exp7_standard_ranking.json")
    txt_path = os.path.join(RESULTS_DIR, "exp7_standard_ranking.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append("EXP7: Standardized Ranking for Current Memory Systems")
    lines.append("=" * 55)
    lines.append(f"Dataset: {out['dataset']['memories']} memories, {out['dataset']['queries']} queries")
    lines.append("")
    lines.append("Ranking (composite score)")
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"{i}. {r['system']}: score={r['composite_score']:.4f}, ndcg@10={r['ndcg@10']:.4f}, "
            f"recall@10={r['recall@10']:.4f}, state_p@10={r['state_precision@10']:.4f}, "
            f"latency_ms={r['latency_ms_avg']:.4f}, mem={r['index_footprint_bytes']}"
        )
    lines.append("")
    lines.append("State-conditioning Jaccard")
    for k, v in conditioning.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("References used for metrics")
    for u in references:
        lines.append(f"- {u}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Visualizations
    systems = [r["system"] for r in ranked]

    # 1) quality metrics plot
    q_metrics = ["ndcg@10", "recall@10", "mrr@10", "map@10", "state_precision@10"]
    x = np.arange(len(systems))
    width = 0.15

    plt.figure(figsize=(12, 5))
    for j, m in enumerate(q_metrics):
        vals = [next(rr for rr in ranked if rr["system"] == s)[m] for s in systems]
        plt.bar(x + (j - 2) * width, vals, width=width, label=m)
    plt.xticks(x, systems, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("EXP7 Quality Metrics by System")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp7_quality_metrics.png"))
    plt.close()

    # 2) efficiency plot
    lat = [next(rr for rr in ranked if rr["system"] == s)["latency_ms_avg"] for s in systems]
    thr = [next(rr for rr in ranked if rr["system"] == s)["throughput_qps_est"] for s in systems]
    mem = [next(rr for rr in ranked if rr["system"] == s)["index_footprint_bytes"] / (1024 * 1024) for s in systems]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 0.2, lat, width=0.35, label="latency_ms", color="#E15759")
    ax1.set_ylabel("Latency (ms)", color="#E15759")
    ax1.tick_params(axis='y', labelcolor="#E15759")

    ax2 = ax1.twinx()
    ax2.plot(x, thr, marker="o", label="throughput_qps", color="#4E79A7")
    ax2.plot(x + 0.2, mem, marker="s", label="index_footprint_mb", color="#59A14F")
    ax2.set_ylabel("Throughput (QPS) / Footprint (MB)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, ha="right")
    ax1.set_title("EXP7 Efficiency Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp7_efficiency_metrics.png"))
    plt.close(fig)

    # 3) overall ranking plot
    scores = [next(rr for rr in ranked if rr["system"] == s)["composite_score"] for s in systems]
    plt.figure(figsize=(10, 5))
    bars = plt.barh(systems, scores, color="#2E86AB")
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.05)
    plt.xlabel("Composite Score")
    plt.title("EXP7 Overall Ranking")
    for b, s in zip(bars, scores):
        plt.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{s:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp7_overall_ranking.png"))
    plt.close()

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp7_quality_metrics.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp7_efficiency_metrics.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp7_overall_ranking.png')}")


if __name__ == "__main__":
    run()
