"""
EXP16: Auto-State Integration Validation
=======================================

THREE INDEPENDENT CHECKS

1. Trajectory determinism. Feed a fixed 30-turn script through
   AutoStateTracker and compare the state at turns 10, 20 and 30 against
   pinned constants. This is a regression guard on the update rule, not an
   accuracy claim.

2. Era retrieval. The 30-turn script is hand-authored in three blocks of ten:
   turns 1-10 express stress, 11-20 express curiosity, 21-30 express positive
   affect. Era membership is therefore a HAND-AUTHORED label, not a corpus
   annotation, and it is disclosed as such wherever it is reported. Scoring is
   leave-one-out over all 30 turns.

3. Persistence. Save the store to .ncm, reload it, and confirm the tracker
   state, the adaptive weights and the shipped retrieval ranking survive the
   round trip bit-for-bit.

WHAT CHANGED AND WHY
The previous version of check 2 set the query's 5-dim state to
`states_at_era_end[era]`, that is, to the exact tracker state produced by the
target era's own turns. Because the composite distance rewards state
proximity, that hands the target era a perfectly aligned state signal derived
from the relevance label itself, so the check could not fail. It also ran only
three queries, one per era, and scored them with a bespoke 0.5/0.5 scorer
rather than the shipped retrieval path.

This version reports two conditions side by side:

  ncm_inferred  the agent's state is inferred from the QUERY TEXT ALONE by a
                fresh AutoStateTracker. Deployable, and able to fail.
  ncm_oracle    the previous behaviour, state set to the target era's end
                state. Retained ONLY as an upper bound, labelled as leaking.

Query count goes from 3 to 30 (leave-one-out over the script) plus the 3
original era probes, and all arms now go through
ncm.retrieval.retrieve_top_k_fast / retrieve_semantic_only.

Outputs
- experiments/results/exp16/exp16_auto_state_integration.json
- experiments/results/exp16/exp16_auto_state_integration.txt
- experiments/results/exp16/exp16_state_trajectory.png
- experiments/results/exp16/exp16_retrieval_trend.png
- experiments/results/exp16/exp16_persistence_validation.png
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import AutoStateTracker, MemoryEntry, MemoryStore, NCMFile, SentenceEncoder
from ncm.retrieval import retrieve_semantic_only, retrieve_top_k_fast

RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", "exp16")
os.makedirs(RESULTS_DIR, exist_ok=True)

TOL = 1e-5
DIMS = ["valence", "arousal", "dominance", "curiosity", "stress"]
ERAS = ("Era1", "Era2", "Era3")
ERA_NAMES = {"Era1": "stress", "Era2": "curiosity", "Era3": "positive affect"}
ARMS = ("semantic_only", "ncm_inferred", "ncm_oracle")

CONVERSATION_30 = [
    "I have so many deadlines and I am completely overwhelmed",
    "I failed my exam and I feel terrible about it",
    "There is too much pressure and I cannot handle it",
    "I am worried I will not finish this project in time",
    "Everything is going wrong and I feel helpless",
    "The deadline is tomorrow and I have barely started",
    "I am so anxious about the presentation next week",
    "I cannot sleep because I keep thinking about all my tasks",
    "My boss is angry and I feel like I am failing at everything",
    "I feel tense and stressed about everything in my life right now",
    "How does the human brain store long-term memories?",
    "I wonder what causes northern lights, the physics is fascinating",
    "What if we could simulate consciousness in a machine?",
    "Tell me about the history of the Roman empire",
    "I am curious about how black holes actually work",
    "What are the most interesting unsolved problems in mathematics?",
    "How do plants convert sunlight into energy through photosynthesis?",
    "I want to explore the theory of relativity in more depth",
    "What is the current state of quantum computing research?",
    "Can you explain how transformers work in machine learning?",
    "I finished my project and I feel amazing and relieved",
    "Today was a wonderful day and I am so happy",
    "I got great feedback and I feel confident and pleased",
    "Everything is going well and I feel content and at peace",
    "I had a delightful conversation with my friend today",
    "I completed all my tasks and I feel joyful and free",
    "Life feels wonderful right now, I am so grateful",
    "I feel calm, happy, and completely at ease with myself",
    "Today I received wonderful news and I feel delighted",
    "I am in a great mood, feeling cheerful and content",
]

EXPECTED_PROPOSED = {
    "turn10": np.array([0.4068923, 0.45871156, 0.49421537, 0.41032666, 0.63592273], dtype=np.float32),
    "turn20": np.array([0.46631497, 0.48791257, 0.50877178, 0.51426035, 0.5078938], dtype=np.float32),
    "turn30": np.array([0.52644837, 0.45842755, 0.4931564, 0.48029253, 0.4312407], dtype=np.float32),
}

# Hand-authored probe queries, one per era. Retained from the previous version
# so the era-probe numbers stay comparable across revisions.
QUERIES = {
    "Era1": "I am having a really hard time with everything right now",
    "Era2": "I want to learn and understand something new and interesting",
    "Era3": "I feel great and everything is going really well",
}


def era_of_turn(index: int) -> str:
    """0-based turn index to era label. Blocks of ten, by construction."""
    return "Era1" if index < 10 else ("Era2" if index < 20 else "Era3")


# ───────────────────────────────────────────
# CHECK 1: TRAJECTORY DETERMINISM
# ───────────────────────────────────────────

def trajectory_check() -> dict:
    tracker = AutoStateTracker()
    states = [tracker.get_current_state()]
    state_history = [{"turn": 0, "state": states[0].tolist()}]

    for text in CONVERSATION_30:
        states.append(tracker.update(text))
        state_history.append({"turn": len(states) - 1, "state": states[-1].tolist()})

    d10 = float(np.max(np.abs(states[10] - EXPECTED_PROPOSED["turn10"])))
    d20 = float(np.max(np.abs(states[20] - EXPECTED_PROPOSED["turn20"])))
    d30 = float(np.max(np.abs(states[30] - EXPECTED_PROPOSED["turn30"])))

    return {
        "turn10_max_abs_diff": d10,
        "turn20_max_abs_diff": d20,
        "turn30_max_abs_diff": d30,
        "tolerance": TOL,
        "pass": bool(d10 < TOL and d20 < TOL and d30 < TOL),
        "turn10": states[10].tolist(),
        "turn20": states[20].tolist(),
        "turn30": states[30].tolist(),
        "state_history": state_history,
    }


# ───────────────────────────────────────────
# CHECK 2: ERA RETRIEVAL
# ───────────────────────────────────────────

def build_store(encoder: SentenceEncoder) -> tuple[MemoryStore, dict[str, str], dict[str, int], dict[str, np.ndarray]]:
    """
    Store all 30 turns through the shipped write path.

    Returns the store plus id->era, id->turn-index, and the per-era end state
    (needed only by the disclosed oracle arm).
    """
    store = MemoryStore()
    era_of_id: dict[str, str] = {}
    index_of_id: dict[str, int] = {}
    era_end_state: dict[str, np.ndarray] = {}

    for i, text in enumerate(CONVERSATION_30):
        state_before = store.auto_state.get_current_state()
        mem = MemoryEntry(
            e_semantic=encoder.encode(text),
            e_emotional=encoder.encode_emotional(state_before),
            # s_snapshot is kept for file-format compatibility only. The
            # composite distance reads auto_state_snapshot, which add() writes.
            s_snapshot=encoder.encode_state(
                np.pad(state_before, (0, 2), mode="constant", constant_values=0.5)
            ),
            timestamp=int(store.step),
            text=text,
        )
        stored = store.add(mem, update_auto_state=True)
        era_of_id[stored.id] = era_of_turn(i)
        index_of_id[stored.id] = i
        store.step += 1

        if i in (9, 19, 29):
            era_end_state[era_of_turn(i)] = store.auto_state.get_current_state()

    return store, era_of_id, index_of_id, era_end_state


def _precision(labels: list[bool], k: int) -> float:
    top = labels[:k]
    return float(sum(top)) / float(len(top)) if top else 0.0


def _retrieve_labels(
    store: MemoryStore,
    q_sem: np.ndarray,
    q_emo: np.ndarray | None,
    q_state: np.ndarray | None,
    era: str,
    era_of_id: dict[str, str],
    exclude_id: str | None,
    k: int,
) -> list[bool]:
    """
    Retrieve k results through the shipped path, dropping the held-out memory,
    and return per-rank relevance against era membership.
    """
    over_fetch = k + (1 if exclude_id is not None else 0)

    if q_emo is None:
        hits = retrieve_semantic_only(q_sem, store, k=over_fetch)
    else:
        saved_state = store.auto_state.get_current_state()
        saved_turn = store.auto_state.turn
        store.auto_state.state = np.asarray(q_state, dtype=np.float32).copy()
        try:
            hits = retrieve_top_k_fast(
                q_sem, q_emo, store, q_state, int(store.step), k=over_fetch,
            )
        finally:
            store.auto_state.state = saved_state
            store.auto_state.turn = saved_turn

    entries = [h[-1] for h in hits]
    if exclude_id is not None:
        entries = [m for m in entries if m.id != exclude_id]
    entries = entries[:k]
    return [era_of_id.get(m.id, "") == era for m in entries]


def era_retrieval_check(encoder: SentenceEncoder) -> dict:
    store, era_of_id, index_of_id, era_end_state = build_store(encoder)
    memories = store.get_all_safe()

    # ---- leave-one-out over all 30 turns ----------------------------------
    loo: dict[str, dict[str, list[float]]] = {
        arm: {"p5": [], "p10": [], "p5_by_era": {e: [] for e in ERAS}} for arm in ARMS
    }

    for mem in memories:
        era = era_of_id[mem.id]
        q_sem = mem.e_semantic
        query_text = mem.text

        labels = _retrieve_labels(store, q_sem, None, None, era, era_of_id, mem.id, 10)
        loo["semantic_only"]["p5"].append(_precision(labels, 5))
        loo["semantic_only"]["p10"].append(_precision(labels, 10))
        loo["semantic_only"]["p5_by_era"][era].append(_precision(labels, 5))

        inferred = AutoStateTracker().update(query_text)
        labels = _retrieve_labels(
            store, q_sem, encoder.encode_emotional(inferred), inferred,
            era, era_of_id, mem.id, 10,
        )
        loo["ncm_inferred"]["p5"].append(_precision(labels, 5))
        loo["ncm_inferred"]["p10"].append(_precision(labels, 10))
        loo["ncm_inferred"]["p5_by_era"][era].append(_precision(labels, 5))

        oracle = era_end_state[era]
        labels = _retrieve_labels(
            store, q_sem, encoder.encode_emotional(oracle), oracle,
            era, era_of_id, mem.id, 10,
        )
        loo["ncm_oracle"]["p5"].append(_precision(labels, 5))
        loo["ncm_oracle"]["p10"].append(_precision(labels, 10))
        loo["ncm_oracle"]["p5_by_era"][era].append(_precision(labels, 5))

    def mean(xs: list[float]) -> float:
        return round(float(np.mean(xs)), 4) if xs else 0.0

    loo_summary = {
        arm: {
            "p5": mean(loo[arm]["p5"]),
            "p10": mean(loo[arm]["p10"]),
            "p5_by_era": {e: mean(loo[arm]["p5_by_era"][e]) for e in ERAS},
            "n_queries": len(loo[arm]["p5"]),
        }
        for arm in ARMS
    }

    # Under leave-one-out each query has 9 same-era peers among 29 candidates.
    random_p5 = round(9.0 / 29.0, 4)

    # ---- the 3 original hand-authored era probes -------------------------
    probes: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for era, query in QUERIES.items():
        q_sem = encoder.encode(query)

        labels = _retrieve_labels(store, q_sem, None, None, era, era_of_id, None, 10)
        probes["semantic_only"][era] = {
            "p5": round(_precision(labels, 5), 4), "p10": round(_precision(labels, 10), 4),
        }

        inferred = AutoStateTracker().update(query)
        labels = _retrieve_labels(
            store, q_sem, encoder.encode_emotional(inferred), inferred,
            era, era_of_id, None, 10,
        )
        probes["ncm_inferred"][era] = {
            "p5": round(_precision(labels, 5), 4), "p10": round(_precision(labels, 10), 4),
        }

        oracle = era_end_state[era]
        labels = _retrieve_labels(
            store, q_sem, encoder.encode_emotional(oracle), oracle,
            era, era_of_id, None, 10,
        )
        probes["ncm_oracle"][era] = {
            "p5": round(_precision(labels, 5), 4), "p10": round(_precision(labels, 10), 4),
        }

    probe_means = {
        arm: round(float(np.mean([probes[arm][e]["p5"] for e in ERAS])), 4) for arm in ARMS
    }

    return {
        "label_provenance": "HAND_AUTHORED: era membership is defined by the "
                            "position of each turn in a script written for this "
                            "experiment, not by any corpus annotation",
        "leave_one_out": {
            "arms": loo_summary,
            "random_guess_p5": random_p5,
            "p5_delta_inferred_minus_semantic": round(
                loo_summary["ncm_inferred"]["p5"] - loo_summary["semantic_only"]["p5"], 4
            ),
            "p5_delta_oracle_minus_inferred": round(
                loo_summary["ncm_oracle"]["p5"] - loo_summary["ncm_inferred"]["p5"], 4
            ),
        },
        "era_probes": {
            "n_queries": len(QUERIES),
            "queries": QUERIES,
            "per_era": probes,
            "mean_p5": probe_means,
            "p5_delta_inferred_minus_semantic": round(
                probe_means["ncm_inferred"] - probe_means["semantic_only"], 4
            ),
        },
        "oracle_disclosure": "ncm_oracle sets the query state to the target "
                             "era's end state, which is derived from the target "
                             "era's own turns. It leaks the relevance label and "
                             "is an upper bound, not a system result.",
    }


# ───────────────────────────────────────────
# CHECK 3: PERSISTENCE
# ───────────────────────────────────────────

def persistence_check(encoder: SentenceEncoder) -> dict:
    """Round-trip the store through .ncm and compare shipped-path retrieval."""
    store_pre = MemoryStore()
    for text in CONVERSATION_30[:20]:
        state_before = store_pre.auto_state.get_current_state()
        mem = MemoryEntry(
            e_semantic=encoder.encode(text),
            e_emotional=encoder.encode_emotional(state_before),
            s_snapshot=encoder.encode_state(
                np.pad(state_before, (0, 2), mode="constant", constant_values=0.5)
            ),
            timestamp=int(store_pre.step),
            text=text,
        )
        store_pre.add(mem, update_auto_state=True)
        store_pre.step += 1

    s_pre = store_pre.auto_state.get_current_state()
    w_state_pre, w_sem_pre = store_pre.auto_state.get_adaptive_weights()

    fd, ncm_path = tempfile.mkstemp(prefix="exp16_", suffix=".ncm")
    os.close(fd)
    try:
        NCMFile.save(store_pre, ncm_path, compress=True, fp16=False)
        store_post = NCMFile.load(ncm_path)
    finally:
        try:
            os.remove(ncm_path)
        except OSError:
            pass

    s_post = store_post.auto_state.get_current_state()
    w_state_post, w_sem_post = store_post.auto_state.get_adaptive_weights()

    max_state_diff = float(np.max(np.abs(s_pre - s_post)))
    turn_ok = bool(store_pre.auto_state.turn == store_post.auto_state.turn)
    alpha_ok = bool(np.max(np.abs(store_pre.auto_state.alpha - store_post.auto_state.alpha)) < TOL)
    weight_ok = bool(abs(w_state_pre - w_state_post) < TOL and abs(w_sem_pre - w_sem_post) < TOL)
    count_ok = bool(len(store_pre) == len(store_post))

    # Compare the shipped retrieval path, not a bespoke scorer.
    query = "I feel overwhelmed and anxious about my work"
    q_sem = encoder.encode(query)
    q_emo = encoder.encode_emotional(s_pre)
    k = min(10, len(store_pre))

    hits_pre = retrieve_top_k_fast(
        q_sem, q_emo, store_pre, encoder.encode_state(s_pre), int(store_pre.step), k=k,
    )
    hits_post = retrieve_top_k_fast(
        q_sem, q_emo, store_post, encoder.encode_state(s_post), int(store_post.step), k=k,
    )

    max_distance_diff = (
        float(max(abs(a[0] - b[0]) for a, b in zip(hits_pre, hits_post))) if hits_pre else 0.0
    )
    texts_pre = [h[-1].text for h in hits_pre]
    texts_post = [h[-1].text for h in hits_post]
    ranking_ok = bool(texts_pre == texts_post)
    top1_ok = bool(texts_pre[:1] == texts_post[:1])

    return {
        "memories_stored": len(store_pre),
        "k_compared": k,
        "max_state_diff": max_state_diff,
        "max_retrieval_distance_diff": max_distance_diff,
        "turn_ok": turn_ok,
        "alpha_ok": alpha_ok,
        "weights_ok": weight_ok,
        "count_ok": count_ok,
        "ranking_identical": ranking_ok,
        "top1_ok": top1_ok,
        "pass": bool(
            max_state_diff < TOL and max_distance_diff < TOL and turn_ok
            and alpha_ok and weight_ok and count_ok and ranking_ok
        ),
        "s_pre": [float(x) for x in s_pre],
        "retrieval_path": "ncm.retrieval.retrieve_top_k_fast",
    }


# ───────────────────────────────────────────
# VISUALIZATIONS
# ───────────────────────────────────────────

ARM_LABELS = {
    "semantic_only": "Semantic only",
    "ncm_inferred": "NCM (state from query)",
    "ncm_oracle": "NCM oracle (LABEL LEAK)",
}
ARM_COLORS = {
    "semantic_only": "#e74c3c",
    "ncm_inferred": "#2ecc71",
    "ncm_oracle": "#f39c12",
}


def plot_state_trajectory(trajectory_history) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    turns = [s["turn"] for s in trajectory_history]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]

    for i, dim in enumerate(DIMS):
        values = [s["state"][i] for s in trajectory_history]
        ax.plot(turns, values, marker="o", linewidth=2.5, markersize=5,
                label=dim.capitalize(), color=colors[i], alpha=0.85)

    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvspan(0.5, 10.5, alpha=0.08, color="#e74c3c")
    ax.axvspan(10.5, 20.5, alpha=0.08, color="#3498db")
    ax.text(5.5, 1.05, "Era1 stress", ha="center", fontsize=9, color="gray")
    ax.text(15.5, 1.05, "Era2 curiosity", ha="center", fontsize=9, color="gray")
    ax.text(25.5, 1.05, "Era3 positive", ha="center", fontsize=9, color="gray")

    ax.set_xlabel("Turn", fontsize=11)
    ax.set_ylabel("State value", fontsize=11)
    ax.set_title("EXP16 auto-state trajectory over a hand-authored 30-turn script",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=3, fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_state_trajectory.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def plot_retrieval_trend(trend: dict) -> str:
    loo = trend["leave_one_out"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(ERAS))
    width = 0.26
    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * width
        vals = [loo["arms"][arm]["p5_by_era"][e] for e in ERAS]
        bars = ax1.bar(x + offset, vals, width, label=ARM_LABELS[arm],
                       color=ARM_COLORS[arm], alpha=0.88, edgecolor="black", linewidth=1.1)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax1.axhline(loo["random_guess_p5"], color="black", linestyle=":", linewidth=1.6,
                label=f"Random guess ({loo['random_guess_p5']:.3f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{e}\n({ERA_NAMES[e]})" for e in ERAS])
    ax1.set_ylabel("P@5 (leave-one-out)", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Same-era retrieval precision by era", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    names = [ARM_LABELS[a] for a in ARMS]
    overall = [loo["arms"][a]["p5"] for a in ARMS]
    bars = ax2.bar(names, overall, color=[ARM_COLORS[a] for a in ARMS],
                   alpha=0.88, edgecolor="black", linewidth=1.3)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax2.axhline(loo["random_guess_p5"], color="black", linestyle=":", linewidth=1.6)
    ax2.set_ylabel("P@5 (all 30 queries)", fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(f"Overall, n={loo['arms']['ncm_inferred']['n_queries']} queries",
                  fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("EXP16 era retrieval. Era labels are HAND-AUTHORED; the oracle arm "
                 "leaks them into the query state.", fontsize=10, y=1.02)
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_retrieval_trend.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    return png_path


def plot_persistence_validation(persist: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = ["State\nmax diff", "Retrieval distance\nmax diff"]
    values = [persist["max_state_diff"], persist["max_retrieval_distance_diff"]]
    colors = ["#2ecc71" if v < TOL else "#e74c3c" for v in values]

    bars = ax1.bar(names, values, color=colors, alpha=0.88, edgecolor="black", linewidth=1.4)
    ax1.set_ylabel("Max absolute difference", fontsize=11)
    ax1.set_title("Round-trip numerical drift", fontsize=11, fontweight="bold")
    if all(v > 0 for v in values):
        ax1.set_yscale("log")
    else:
        ax1.set_ylim(0, max(max(values), TOL) * 2.0)
    ax1.axhline(TOL, color="gray", linestyle="--", linewidth=1.2, label=f"tolerance {TOL:g}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val * 1.5 if val > 0 else TOL * 0.1,
                 f"{val:.2e}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    labels = ["Turn", "Alpha", "Weights", "Count", "Ranking", "Top-1"]
    flags = [persist["turn_ok"], persist["alpha_ok"], persist["weights_ok"],
             persist["count_ok"], persist["ranking_identical"], persist["top1_ok"]]
    vals = [1.0 if f else 0.0 for f in flags]
    bars2 = ax2.bar(labels, vals, color=["#2ecc71" if v else "#e74c3c" for v in vals],
                    alpha=0.88, edgecolor="black", linewidth=1.4)
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["FAIL", "PASS"])
    ax2.set_title("Integrity checks", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                 "OK" if v else "X", ha="center", va="bottom",
                 fontsize=12, fontweight="bold",
                 color="green" if v else "red")

    fig.suptitle(f"EXP16 persistence: .ncm round trip over "
                 f"{persist['memories_stored']} memories, k={persist['k_compared']}",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_persistence_validation.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    return png_path


# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────

def main() -> None:
    print("[exp16] Loading encoder")
    encoder = SentenceEncoder(
        model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models")
    )
    backend = encoder.backend
    print(f"[exp16] Encoder backend: {backend}")
    if backend != "sentence-transformers":
        print("[exp16] ABORT: hash fallback carries no semantic structure, so the")
        print(f"[exp16]        retrieval checks would be meaningless. Reason: {encoder.backend_error}")
        return

    print("[exp16] Check 1: trajectory determinism")
    traj = trajectory_check()

    print("[exp16] Check 2: era retrieval (leave-one-out, n=30, plus 3 probes)")
    trend = era_retrieval_check(encoder)

    print("[exp16] Check 3: persistence round trip")
    persist = persistence_check(encoder)

    # The verdict covers only the two checks that have a pass/fail criterion.
    # Era retrieval reports magnitudes; it is not a pass/fail gate.
    verdict = bool(traj["pass"] and persist["pass"])

    results = {
        "experiment": "exp16_auto_state_integration",
        "encoder_backend": backend,
        "design_constants": {
            "dimensions": DIMS,
            "alpha": [0.15, 0.15, 0.15, 0.20, 0.25],
            "sigma": "(1 + cos(e,pos) - cos(e,neg)) / 2 clipped to [0,1]",
            "initial_state": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        "trajectory_check": traj,
        "era_retrieval": trend,
        "persistence_check": persist,
        "verdict": "PASS" if verdict else "FAIL",
        "verdict_scope": "trajectory_check and persistence_check only; "
                         "era_retrieval reports magnitudes and is not gated",
    }

    json_path = os.path.join(RESULTS_DIR, "exp16_auto_state_integration.json")
    txt_path = os.path.join(RESULTS_DIR, "exp16_auto_state_integration.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    loo = trend["leave_one_out"]
    probes = trend["era_probes"]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EXP16: Auto-State Integration Validation\n")
        f.write("========================================\n\n")
        f.write(f"Verdict: {results['verdict']}  "
                f"(scope: trajectory and persistence only)\n")
        f.write(f"Encoder backend: {backend}\n\n")

        f.write("Check 1: trajectory determinism (max abs diff vs pinned constants)\n")
        f.write(f"- Turn 10: {traj['turn10_max_abs_diff']:.2e}\n")
        f.write(f"- Turn 20: {traj['turn20_max_abs_diff']:.2e}\n")
        f.write(f"- Turn 30: {traj['turn30_max_abs_diff']:.2e}\n")
        f.write(f"- Tolerance: {TOL:g}    Pass: {traj['pass']}\n\n")

        f.write("Check 2: era retrieval\n")
        f.write("Era labels are HAND-AUTHORED: they come from the position of each\n")
        f.write("turn in a script written for this experiment.\n\n")
        f.write(f"Leave-one-out over all 30 turns "
                f"(random guess P@5 = {loo['random_guess_p5']:.4f}):\n")
        f.write(f"  {'arm':<16}{'P@5':>8}{'P@10':>8}   per-era P@5 "
                f"({', '.join(ERAS)})\n")
        for arm in ARMS:
            a = loo["arms"][arm]
            per_era = ", ".join(f"{a['p5_by_era'][e]:.3f}" for e in ERAS)
            f.write(f"  {arm:<16}{a['p5']:>8.4f}{a['p10']:>8.4f}   {per_era}\n")
        f.write(f"\n  ncm_inferred - semantic_only P@5: "
                f"{loo['p5_delta_inferred_minus_semantic']:+.4f}\n")
        f.write(f"  ncm_oracle - ncm_inferred P@5:    "
                f"{loo['p5_delta_oracle_minus_inferred']:+.4f}\n\n")

        f.write("Three hand-authored era probes (mean P@5 across the 3 eras):\n")
        for arm in ARMS:
            f.write(f"  {arm:<16}{probes['mean_p5'][arm]:.4f}\n")
        f.write(f"  ncm_inferred - semantic_only: "
                f"{probes['p5_delta_inferred_minus_semantic']:+.4f}\n\n")
        f.write("ncm_oracle sets the query state to the target era's end state, which\n")
        f.write("is produced by the target era's own turns. It leaks the relevance\n")
        f.write("label and bounds what the state channel could contribute. It is not\n")
        f.write("a system result.\n\n")

        f.write("Check 3: persistence (.ncm round trip)\n")
        f.write(f"- Memories stored: {persist['memories_stored']}, k compared: "
                f"{persist['k_compared']}\n")
        f.write(f"- max_state_diff: {persist['max_state_diff']:.2e}\n")
        f.write(f"- max_retrieval_distance_diff: "
                f"{persist['max_retrieval_distance_diff']:.2e}\n")
        f.write(f"- turn/alpha/weights/count: {persist['turn_ok']}/"
                f"{persist['alpha_ok']}/{persist['weights_ok']}/{persist['count_ok']}\n")
        f.write(f"- ranking identical: {persist['ranking_identical']}, "
                f"top-1 identical: {persist['top1_ok']}\n")
        f.write(f"- Retrieval path: {persist['retrieval_path']}\n")
        f.write(f"- Pass: {persist['pass']}\n\n")

        f.write("Reading of check 2\n")
        if loo["p5_delta_inferred_minus_semantic"] > 0:
            f.write("With the query state inferred from the query text alone, the\n")
            f.write("composite distance retrieves same-era memories more precisely than\n")
            f.write("semantic similarity alone on this script.\n")
        else:
            f.write("With the query state inferred from the query text alone, the\n")
            f.write("composite distance does NOT beat semantic similarity alone on this\n")
            f.write("script. The earlier positive result depended on the query state\n")
            f.write("being set from the target era itself.\n")

    print(f"[exp16] Saved: {json_path}")
    print(f"[exp16] Saved: {txt_path}")

    print("[exp16] Generating figures")
    if traj.get("state_history"):
        print(f"[exp16] Saved: {plot_state_trajectory(traj['state_history'])}")
    print(f"[exp16] Saved: {plot_retrieval_trend(trend)}")
    print(f"[exp16] Saved: {plot_persistence_validation(persist)}")

    print("[exp16] LOO P@5: " + ", ".join(
        f"{arm}={loo['arms'][arm]['p5']:.4f}" for arm in ARMS
    ) + f"  (random {loo['random_guess_p5']:.4f})")


if __name__ == "__main__":
    main()
