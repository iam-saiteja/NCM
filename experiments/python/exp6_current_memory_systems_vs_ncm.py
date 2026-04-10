"""
Experiment 6: NCM vs RAG and current production-style memory systems.

Baselines included:
- rag_semantic_only           (dense vector RAG)
- rag_semantic_recency        (dense + recency fusion)
- recency_only                (chat-memory style recent window)
- semantic_emotional          (semantic + affective context)
- ncm_full                    (semantic + emotional + state + time + strength)
- ncm_cached_full             (same as ncm_full via cached path)

Outputs:
- results/exp6_current_memory_systems_vs_ncm.json
- results/exp6_current_memory_systems_vs_ncm.txt
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.retrieval import (
    retrieve_top_k,
    retrieve_top_k_fast,
    retrieve_semantic_emotional,
)

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split('_')[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
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

# Query prompts are category-level paraphrases, not exact memory strings.
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


def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


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


def make_retrievers(store, encoder):
    store._rebuild_cache()

    sem = store._sem_cache
    emo = store._emo_cache
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
        # Typical chat memory pattern: prefer the most recent messages.
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


def evaluate_system(name, retrieve_fn, store, encoder, metadata, queries, ks=(1, 3, 5, 10)):
    category_precisions = {k: [] for k in ks}
    state_precisions = {k: [] for k in ks}
    latencies_ms = []

    for q in queries:
        q_sem = encoder.encode(q["text"])
        q_emo = encoder.encode_emotional(q["state_vec"])
        q_state = encoder.encode_state(q["state_vec"])

        t0 = time.perf_counter()
        mids = retrieve_fn(q_sem, q_emo, q_state, store.step, k=max(ks))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        cat_gt = {mid for mid, m in metadata.items() if m["category"] == q["category"]}
        state_gt = {mid for mid, m in metadata.items() if m["category"] == q["category"] and m["state"] == q["state_name"]}

        for k in ks:
            topk = mids[:k]
            category_precisions[k].append(sum(1 for mid in topk if mid in cat_gt) / k)
            state_precisions[k].append(sum(1 for mid in topk if mid in state_gt) / k)

    return {
        "system": name,
        "category_p_at_k": {str(k): round(mean(v), 4) for k, v in category_precisions.items()},
        "state_p_at_k": {str(k): round(mean(v), 4) for k, v in state_precisions.items()},
        "latency_ms_avg": round(mean(latencies_ms), 4),
    }


def run():
    random.seed(42)
    np.random.seed(42)

    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    store, metadata = build_store(encoder, n_per_pair=25, seed=42)  # 1200 memories

    # activate strength effects
    mem_ids = list(store._memories.keys())
    for mid in mem_ids[::17]:
        store.reinforce(mid, amount=0.6)

    queries = build_queries(seed=123)
    retrievers = make_retrievers(store, encoder)

    systems_order = [
        "rag_semantic_only",
        "rag_semantic_recency",
        "recency_only",
        "semantic_emotional",
        "ncm_full",
        "ncm_cached_full",
    ]

    results = {
        "dataset": {
            "memories": len(store),
            "queries": len(queries),
            "query_type": "category paraphrases (not exact memory strings)",
        },
        "systems": [],
    }

    for name in systems_order:
        print(f"Evaluating: {name}")
        results["systems"].append(
            evaluate_system(name, retrievers[name], store, encoder, metadata, queries)
        )

    # state-conditioning check on same semantic query, two states
    state_pairs = [
        ("calm_happy", "stressed_angry"),
        ("confident", "fearful"),
        ("neutral", "exhausted"),
    ]
    anchor_text = "I experienced betrayal by someone close"
    q_sem = encoder.encode(anchor_text)

    conditioning = {}
    compare_systems = ["rag_semantic_only", "rag_semantic_recency", "semantic_emotional", "ncm_full"]
    for a, b in state_pairs:
        qa_emo = encoder.encode_emotional(STATE_ARCHETYPES[a])
        qa_st = encoder.encode_state(STATE_ARCHETYPES[a])
        qb_emo = encoder.encode_emotional(STATE_ARCHETYPES[b])
        qb_st = encoder.encode_state(STATE_ARCHETYPES[b])

        row = {}
        for s in compare_systems:
            ids_a = retrievers[s](q_sem, qa_emo, qa_st, store.step, 10)
            ids_b = retrievers[s](q_sem, qb_emo, qb_st, store.step, 10)
            row[f"{s}_jaccard"] = round(jaccard_distance(ids_a, ids_b), 4)
        conditioning[f"{a}_vs_{b}"] = row

    results["state_conditioning"] = conditioning

    ranking = sorted(
        [
            {
                "system": r["system"],
                "category_avg": round(mean(r["category_p_at_k"].values()), 4),
                "state_avg": round(mean(r["state_p_at_k"].values()), 4),
                "latency_ms_avg": r["latency_ms_avg"],
            }
            for r in results["systems"]
        ],
        key=lambda x: (x["state_avg"], x["category_avg"], -x["latency_ms_avg"]),
        reverse=True,
    )
    results["standing"] = ranking

    json_path = os.path.join(RESULTS_DIR, "exp6_current_memory_systems_vs_ncm.json")
    txt_path = os.path.join(RESULTS_DIR, "exp6_current_memory_systems_vs_ncm.txt")
    png_path = os.path.join(RESULTS_DIR, "exp6_current_memory_systems_vs_ncm.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("EXP6: Current Memory Systems vs NCM")
    lines.append("=" * 36)
    lines.append(f"Dataset: {results['dataset']['memories']} memories, {results['dataset']['queries']} paraphrase queries")
    lines.append("")
    lines.append("Standings")
    for i, r in enumerate(ranking, start=1):
        lines.append(
            f"{i}. {r['system']}: state_avg={r['state_avg']:.4f}, category_avg={r['category_avg']:.4f}, latency_ms={r['latency_ms_avg']:.4f}"
        )
    lines.append("")
    lines.append("State-conditioning Jaccard (same semantic query, different states)")
    for pair, vals in conditioning.items():
        lines.append(f"- {pair}: {vals}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Visualization: rank quality and latency
    systems = [r["system"] for r in ranking]
    x = np.arange(len(systems))
    cat_vals = [r["category_avg"] for r in ranking]
    state_vals = [r["state_avg"] for r in ranking]
    lat_vals = [r["latency_ms_avg"] for r in ranking]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    width = 0.32
    ax1.bar(x - width / 2, cat_vals, width=width, label="Category avg P@k", color="#4E79A7")
    ax1.bar(x + width / 2, state_vals, width=width, label="State avg P@k", color="#59A14F")
    ax1.set_ylabel("Quality score")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, lat_vals, marker="o", color="#E15759", label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
    ax1.set_title("EXP6 Current Memory Systems vs NCM")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    run()
