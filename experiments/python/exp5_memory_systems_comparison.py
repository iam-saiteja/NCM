"""
Experiment 5: Memory Systems Comparison

Compares currently available memory systems in this repo:
1) Semantic only
2) Semantic + Emotional
3) NCM Full (state + time + strength)
4) NCM Full (state + time, no strength)
5) NCM Cached Full (fast path)

Outputs:
- results/exp5_memory_systems_comparison.json
- results/exp5_memory_systems_comparison.txt
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
    retrieve_semantic_only,
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


def evaluate_system(name, retrieve_fn, store, encoder, metadata, queries, ks=(1, 3, 5, 10)):
    category_precisions = {k: [] for k in ks}
    state_precisions = {k: [] for k in ks}
    latencies_ms = []

    for q in queries:
        q_sem = encoder.encode(q["text"])
        q_emo = encoder.encode_emotional(q["state_vec"])
        q_state = encoder.encode_state(q["state_vec"])

        t0 = time.perf_counter()
        ids = retrieve_fn(q_sem, q_emo, q_state, q["timestamp"], k=max(ks))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        cat_gt = {mid for mid, m in metadata.items() if m["category"] == q["category"]}
        state_gt = {mid for mid, m in metadata.items() if m["category"] == q["category"] and m["state"] == q["state_name"]}

        for k in ks:
            topk = ids[:k]
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
    store, metadata = build_store(encoder, n_per_pair=25, seed=42)  # 6*8*25 = 1200

    # reinforce a subset to activate strength-weighted behavior
    mem_ids = list(store._memories.keys())
    for mid in mem_ids[::17]:
        store.reinforce(mid, amount=0.6)

    keys = list(metadata.keys())
    random.shuffle(keys)
    eval_ids = keys[:240]

    queries = []
    for mid in eval_ids:
        m = metadata[mid]
        queries.append({
            "memory_id": mid,
            "text": m["text"],
            "category": m["category"],
            "state_name": m["state"],
            "state_vec": STATE_ARCHETYPES[m["state"]],
            "timestamp": store.step,
        })

    def get_ids_sem_only(qs, qe, qst, step, k):
        return [m.id for _, m in retrieve_semantic_only(qs, store, k=k)]

    def get_ids_sem_emo(qs, qe, qst, step, k):
        return [m.id for _, m in retrieve_semantic_emotional(qs, qe, store, k=k)]

    def get_ids_ncm_full(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k(qs, qe, store, qst, step, k=k, use_strength=True)]

    def get_ids_ncm_no_strength(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k(qs, qe, store, qst, step, k=k, use_strength=False)]

    def get_ids_ncm_cached(qs, qe, qst, step, k):
        return [m.id for _, _, m in retrieve_top_k_fast(qs, qe, store, qst, step, k=k, use_strength=True)]

    systems = [
        ("semantic_only", get_ids_sem_only),
        ("semantic_emotional", get_ids_sem_emo),
        ("ncm_full", get_ids_ncm_full),
        ("ncm_full_no_strength", get_ids_ncm_no_strength),
        ("ncm_cached_full", get_ids_ncm_cached),
    ]

    results = {"dataset": {"memories": len(store), "queries": len(queries)}, "systems": []}

    for name, fn in systems:
        print(f"Evaluating: {name}")
        results["systems"].append(evaluate_system(name, fn, store, encoder, metadata, queries))

    # State-conditioning check via Jaccard on same semantic query, different states
    state_pairs = [
        ("calm_happy", "stressed_angry"),
        ("confident", "fearful"),
        ("neutral", "exhausted"),
    ]
    anchor_text = "colleague took credit for my work"
    q_sem = encoder.encode(anchor_text)

    conditioning = {}
    for a, b in state_pairs:
        qe_a, qs_a = encoder.encode_emotional(STATE_ARCHETYPES[a]), encoder.encode_state(STATE_ARCHETYPES[a])
        qe_b, qs_b = encoder.encode_emotional(STATE_ARCHETYPES[b]), encoder.encode_state(STATE_ARCHETYPES[b])

        sem_a = get_ids_sem_only(q_sem, qe_a, qs_a, store.step, 10)
        sem_b = get_ids_sem_only(q_sem, qe_b, qs_b, store.step, 10)
        emo_a = get_ids_sem_emo(q_sem, qe_a, qs_a, store.step, 10)
        emo_b = get_ids_sem_emo(q_sem, qe_b, qs_b, store.step, 10)
        ncm_a = get_ids_ncm_full(q_sem, qe_a, qs_a, store.step, 10)
        ncm_b = get_ids_ncm_full(q_sem, qe_b, qs_b, store.step, 10)

        conditioning[f"{a}_vs_{b}"] = {
            "semantic_only_jaccard": round(jaccard_distance(sem_a, sem_b), 4),
            "semantic_emotional_jaccard": round(jaccard_distance(emo_a, emo_b), 4),
            "ncm_full_jaccard": round(jaccard_distance(ncm_a, ncm_b), 4),
        }

    results["state_conditioning"] = conditioning

    # ranking summary
    def avg_metric(sys_name, metric):
        row = next(r for r in results["systems"] if r["system"] == sys_name)
        return mean([row[metric]["1"], row[metric]["3"], row[metric]["5"], row[metric]["10"]])

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

    json_path = os.path.join(RESULTS_DIR, "exp5_memory_systems_comparison.json")
    txt_path = os.path.join(RESULTS_DIR, "exp5_memory_systems_comparison.txt")
    png_path = os.path.join(RESULTS_DIR, "exp5_memory_systems_comparison.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("EXP5: Memory Systems Comparison")
    lines.append("=" * 34)
    lines.append(f"Dataset: {results['dataset']['memories']} memories, {results['dataset']['queries']} queries")
    lines.append("")
    lines.append("Standings (higher state/category, lower latency)")
    for i, r in enumerate(ranking, start=1):
        lines.append(
            f"{i}. {r['system']}: state_avg={r['state_avg']:.4f}, category_avg={r['category_avg']:.4f}, latency_ms={r['latency_ms_avg']:.4f}"
        )
    lines.append("")
    lines.append("State-conditioning Jaccard (same text, different states)")
    for pair, vals in conditioning.items():
        lines.append(
            f"- {pair}: semantic={vals['semantic_only_jaccard']}, sem+emo={vals['semantic_emotional_jaccard']}, ncm={vals['ncm_full_jaccard']}"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Visualization: state/category quality + latency
    systems = [r["system"] for r in ranking]
    x = np.arange(len(systems))
    cat_vals = [r["category_avg"] for r in ranking]
    state_vals = [r["state_avg"] for r in ranking]
    lat_vals = [r["latency_ms_avg"] for r in ranking]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    width = 0.32
    b1 = ax1.bar(x - width / 2, cat_vals, width=width, label="Category avg P@k", color="#4E79A7")
    b2 = ax1.bar(x + width / 2, state_vals, width=width, label="State avg P@k", color="#59A14F")
    ax1.set_ylabel("Quality score")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    l1 = ax2.plot(x, lat_vals, marker="o", color="#E15759", label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
    ax1.set_title("EXP5 Memory Systems Comparison")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    run()
