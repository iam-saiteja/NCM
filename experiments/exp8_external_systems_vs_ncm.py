"""
Experiment 8: External memory/retrieval systems vs NCM.

Compares NCM against external/common systems:
- BM25 lexical retrieval (rank-bm25)
- TF-IDF cosine retrieval (scikit-learn)
- Dense SBERT retrieval (SentenceTransformer cosine)
- RAG semantic-only over NCM embeddings
- Recency-only memory
- NCM full
- NCM cached full

Outputs:
- results/exp8_external_systems_vs_ncm.json
- results/exp8_external_systems_vs_ncm.txt
- results/exp8_external_quality.png
- results/exp8_external_ranking.png
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


def minmax(values):
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def graded_rel(meta, q_cat, q_state):
    if meta["category"] == q_cat and meta["state"] == q_state:
        return 2
    if meta["category"] == q_cat:
        return 1
    return 0


def dcg(grades):
    s = 0.0
    for i, g in enumerate(grades):
        s += (2**g - 1) / np.log2(i + 2)
    return s


def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


def evaluate_at_k(mids, metadata, q_cat, q_state, k=10):
    topk = mids[:k]
    binary = [1 if metadata[mid]["category"] == q_cat else 0 for mid in topk]
    graded = [graded_rel(metadata[mid], q_cat, q_state) for mid in topk]

    precision = sum(binary) / k
    hit = 1.0 if any(binary) else 0.0

    rr = 0.0
    for i, b in enumerate(binary):
        if b:
            rr = 1.0 / (i + 1)
            break

    total_rel = sum(1 for m in metadata.values() if m["category"] == q_cat)
    recall = sum(binary) / max(1, total_rel)

    num_hits = 0
    precisions = []
    for i, b in enumerate(binary, start=1):
        if b:
            num_hits += 1
            precisions.append(num_hits / i)
    ap = float(sum(precisions) / max(1, total_rel))

    ideal = sorted((graded_rel(m, q_cat, q_state) for m in metadata.values()), reverse=True)[:k]
    ndcg = dcg(graded) / max(1e-12, dcg(ideal))

    state_prec = sum(1 for mid in topk if metadata[mid]["category"] == q_cat and metadata[mid]["state"] == q_state) / k

    return {
        "precision@10": precision,
        "hit@10": hit,
        "mrr@10": rr,
        "recall@10": recall,
        "map@10": ap,
        "ndcg@10": ndcg,
        "state_precision@10": state_prec,
    }


def build_store(encoder: SentenceEncoder, n_per_pair=25, seed=42):
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=50000))
    metadata = {}
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
                metadata[mem.id] = {"category": cat, "state": sname, "text": text}
                ts += 1
    store.step = ts
    return store, metadata


def build_queries(seed=123):
    rng = random.Random(seed)
    s_names = list(STATE_ARCHETYPES.keys())
    queries = []
    for cat, prompts in QUERY_TEXTS.items():
        for p in prompts:
            for _ in range(8):
                sname = rng.choice(s_names)
                queries.append({
                    "text": p,
                    "category": cat,
                    "state_name": sname,
                    "state_vec": STATE_ARCHETYPES[sname],
                })
    rng.shuffle(queries)
    return queries


def run():
    random.seed(42)
    np.random.seed(42)

    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    store, metadata = build_store(encoder, n_per_pair=25, seed=42)

    # make strength active
    for mid in list(store._memories.keys())[::17]:
        store.reinforce(mid, amount=0.6)

    queries = build_queries(seed=123)

    # Prepare corpora and caches
    store._rebuild_cache()
    ids = store._id_order
    id_to_idx = {mid: i for i, mid in enumerate(ids)}
    docs = [store._memories[mid].text for mid in ids]

    # BM25
    bm25_tokens = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(bm25_tokens)

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_tfidf = tfidf.fit_transform(docs)

    # Dense SBERT (external direct embedding baseline)
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(os.path.join(ROOT_DIR, "models", "all-MiniLM-L6-v2"))
    doc_dense = dense_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    doc_dense = doc_dense / (np.linalg.norm(doc_dense, axis=1, keepdims=True) + 1e-8)

    sem = store._sem_cache
    ts = store._ts_cache

    def rag_semantic_only(qs, qe, qst, step, k):
        d = 1.0 - (sem @ qs)
        return [ids[i] for i in np.argsort(d)[:k]]

    def rag_semantic_recency(qs, qe, qst, step, k):
        d_sem = np.clip(1.0 - (sem @ qs), 0.0, 1.0)
        d_time = 1.0 - np.exp(-0.001 * np.maximum(0, step - ts))
        d = 0.8 * d_sem + 0.2 * d_time
        return [ids[i] for i in np.argsort(d)[:k]]

    def recency_only(qs, qe, qst, step, k):
        d = np.maximum(0, step - ts)
        return [ids[i] for i in np.argsort(d)[:k]]

    def bm25_text(qs, qe, qst, step, k):
        qtxt = current_query_text.lower().split()
        scores = bm25.get_scores(qtxt)
        idx = np.argsort(-scores)[:k]
        return [ids[i] for i in idx]

    def tfidf_cosine(qs, qe, qst, step, k):
        qv = tfidf.transform([current_query_text])
        s = (X_tfidf @ qv.T).toarray().reshape(-1)
        idx = np.argsort(-s)[:k]
        return [ids[i] for i in idx]

    def dense_sbert_cosine(qs, qe, qst, step, k):
        q = dense_model.encode([current_query_text], convert_to_numpy=True, show_progress_bar=False)[0]
        q = q / (np.linalg.norm(q) + 1e-8)
        s = doc_dense @ q
        idx = np.argsort(-s)[:k]
        return [ids[i] for i in idx]

    def ncm_full(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k(qs, qe, store, qst, step, k=k, use_strength=True)]

    def ncm_cached_full(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k_fast(qs, qe, store, qst, step, k=k, use_strength=True)]

    systems = {
        "bm25_text": bm25_text,
        "tfidf_cosine": tfidf_cosine,
        "dense_sbert_cosine": dense_sbert_cosine,
        "rag_semantic_only": rag_semantic_only,
        "rag_semantic_recency": rag_semantic_recency,
        "recency_only": recency_only,
        "ncm_full": ncm_full,
        "ncm_cached_full": ncm_cached_full,
    }

    rows = []
    global current_query_text

    for s_name, fn in systems.items():
        print(f"Evaluating: {s_name}")
        acc = {
            "precision@10": [], "hit@10": [], "mrr@10": [], "recall@10": [],
            "map@10": [], "ndcg@10": [], "state_precision@10": [], "latency_ms": []
        }

        for q in queries:
            current_query_text = q["text"]
            qs = encoder.encode(q["text"])
            qe = encoder.encode_emotional(q["state_vec"])
            qst = encoder.encode_state(q["state_vec"])

            t0 = time.perf_counter()
            mids = fn(qs, qe, qst, store.step, 10)
            acc["latency_ms"].append((time.perf_counter() - t0) * 1000.0)

            m = evaluate_at_k(mids, metadata, q["category"], q["state_name"], k=10)
            for k, v in m.items():
                acc[k].append(v)

        rows.append({
            "system": s_name,
            "precision@10": round(mean(acc["precision@10"]), 4),
            "hit@10": round(mean(acc["hit@10"]), 4),
            "mrr@10": round(mean(acc["mrr@10"]), 4),
            "recall@10": round(mean(acc["recall@10"]), 4),
            "map@10": round(mean(acc["map@10"]), 4),
            "ndcg@10": round(mean(acc["ndcg@10"]), 4),
            "state_precision@10": round(mean(acc["state_precision@10"]), 4),
            "latency_ms_avg": round(mean(acc["latency_ms"]), 4),
            "throughput_qps_est": round(1000.0 / max(1e-9, mean(acc["latency_ms"])), 2),
        })

    # composite ranking
    lat = [r["latency_ms_avg"] for r in rows]
    qps = [r["throughput_qps_est"] for r in rows]
    lat_n = minmax(lat)
    qps_n = minmax(qps)

    for i, r in enumerate(rows):
        relevance = (
            0.20 * r["ndcg@10"] +
            0.15 * r["recall@10"] +
            0.10 * r["mrr@10"] +
            0.05 * r["hit@10"] +
            0.10 * r["map@10"]
        )
        state_q = 0.20 * r["state_precision@10"]
        efficiency = 0.15 * qps_n[i] + 0.20 * (1.0 - lat_n[i])
        # efficiency scaled to keep total near [0,1]
        efficiency *= 0.20
        r["composite_score"] = round(relevance + state_q + efficiency, 4)

    ranked = sorted(rows, key=lambda x: x["composite_score"], reverse=True)

    # state-conditioning jaccard
    state_pairs = [("calm_happy", "stressed_angry"), ("confident", "fearful"), ("neutral", "exhausted")]
    anchor_text = "I experienced betrayal by someone close"
    q_sem = encoder.encode(anchor_text)
    conditioning_systems = ["bm25_text", "tfidf_cosine", "dense_sbert_cosine", "rag_semantic_only", "ncm_full", "ncm_cached_full"]

    conditioning = {}
    for a, b in state_pairs:
        qa_emo, qa_st = encoder.encode_emotional(STATE_ARCHETYPES[a]), encoder.encode_state(STATE_ARCHETYPES[a])
        qb_emo, qb_st = encoder.encode_emotional(STATE_ARCHETYPES[b]), encoder.encode_state(STATE_ARCHETYPES[b])
        row = {}
        for s in conditioning_systems:
            current_query_text = anchor_text
            ids_a = systems[s](q_sem, qa_emo, qa_st, store.step, 10)
            ids_b = systems[s](q_sem, qb_emo, qb_st, store.step, 10)
            row[f"{s}_jaccard"] = round(jaccard_distance(ids_a, ids_b), 4)
        conditioning[f"{a}_vs_{b}"] = row

    out = {
        "dataset": {"memories": len(store), "queries": len(queries)},
        "systems": rows,
        "ranking": ranked,
        "state_conditioning": conditioning,
    }

    jpath = os.path.join(RESULTS_DIR, "exp8_external_systems_vs_ncm.json")
    tpath = os.path.join(RESULTS_DIR, "exp8_external_systems_vs_ncm.txt")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["EXP8: External Systems vs NCM", "=" * 30, f"Dataset: {len(store)} memories, {len(queries)} queries", "", "Ranking"]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"{i}. {r['system']}: score={r['composite_score']:.4f}, ndcg@10={r['ndcg@10']:.4f}, "
            f"recall@10={r['recall@10']:.4f}, state_p@10={r['state_precision@10']:.4f}, latency_ms={r['latency_ms_avg']:.4f}"
        )
    lines.append("")
    lines.append("State-conditioning Jaccard")
    for p, vals in conditioning.items():
        lines.append(f"- {p}: {vals}")

    with open(tpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Visuals
    systems_names = [r["system"] for r in ranked]

    # quality
    qm = ["ndcg@10", "recall@10", "state_precision@10"]
    x = np.arange(len(systems_names))
    w = 0.22
    plt.figure(figsize=(12, 5))
    for j, m in enumerate(qm):
        vals = [next(rr for rr in ranked if rr["system"] == s)[m] for s in systems_names]
        plt.bar(x + (j - 1) * w, vals, width=w, label=m)
    plt.xticks(x, systems_names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("EXP8 Quality Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp8_external_quality.png"))
    plt.close()

    # ranking
    scores = [next(rr for rr in ranked if rr["system"] == s)["composite_score"] for s in systems_names]
    plt.figure(figsize=(10, 5))
    bars = plt.barh(systems_names, scores, color="#3A7CA5")
    plt.gca().invert_yaxis()
    plt.xlabel("Composite score")
    plt.title("EXP8 Overall Ranking (External Systems + NCM)")
    for b, s in zip(bars, scores):
        plt.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{s:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp8_external_ranking.png"))
    plt.close()

    print(f"Saved: {jpath}")
    print(f"Saved: {tpath}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp8_external_quality.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'exp8_external_ranking.png')}")


if __name__ == "__main__":
    run()
