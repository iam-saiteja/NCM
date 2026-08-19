"""
EXP17: Same-Session Episodic Retrieval on Multi-Session Chat
===========================================================

WHAT THIS TESTS
- Given a held-out dialogue turn as the query, can the system retrieve the
  other turns from the SAME conversational session, out of a store holding
  every turn of that multi-session conversation?

WHY THIS GROUND TRUTH
The corpus (Multi-Session Chat style train.jsonl) carries an explicit
`session_id` on every session. Session membership is therefore a real label
that exists in the data and was not authored for this experiment. It is also
the property an episodic memory system should capture: turns from one sitting
belong together. Random-guess P@5 is roughly 1/num_sessions (~0.2 here), so
the metric is not saturated.

CORPUS PROVENANCE
`experiments/data/real_world_corpus/train.jsonl` holds 8939 records, each with
keys `id`, `init_personas`, `sessions`, and each session carrying `session_id`
and `personas`. The persona sentences are PersonaChat-derived, but the
multi-session structure with explicit `session_id` is what this experiment
uses as its label. Earlier documentation described the file as "PersonaChat"
with "8,940 conversations"; the record count is 8939 and the session
structure is what matters here.

IMPORTANT CONTROL
Sessions are contiguous blocks of turns, so session membership is correlated
with recency. A recency-only arm is therefore included. If recency alone
matches NCM, the state and temporal channels are not adding episodic
structure beyond "recently written", and this script reports that outcome
rather than hiding it.

ARMS
  semantic_only  shipped retrieve_semantic_only (standard dense-RAG baseline)
  recency_only   the k most recently written turns (temporal control)
  ncm_inferred   shipped retrieve_top_k_fast; the agent's 5-dim auto-state is
                 inferred from the QUERY TEXT ALONE by a fresh
                 AutoStateTracker. This is the deployable condition: nothing
                 about the target session is revealed to the query.
  ncm_oracle     shipped retrieve_top_k_fast; the auto-state is set to the
                 mean of the target session's stored auto-states. This LEAKS
                 the label and is reported ONLY as an upper bound on what the
                 state channel could contribute. It is not a system result.

REPLACES the previous version of this script, whose Precision@5 was computed
as len(top_5_list) / 5.0. That expression is identically 1.0 whenever the
store holds at least five memories, for every arm, so the previously reported
"NCM P@5 = 1.000, RAG P@5 = 1.000, improvement +0.000" measured list length
and not relevance. Timing also used time.time(), whose resolution on Windows
is around 15 ms and therefore coarser than the operation being timed;
perf_counter is used now.

Outputs
- experiments/results/exp17/exp17_real_world_scale.json
- experiments/results/exp17/exp17_real_world_scale.txt
- experiments/results/exp17/exp17_scale_retrieval_precision.png
- experiments/results/exp17/exp17_scale_performance_metrics.png
- experiments/results/exp17/exp17_scale_state_accuracy.png
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import AutoStateTracker, MemoryEntry, MemoryStore, SentenceEncoder
from ncm.retrieval import retrieve_semantic_only, retrieve_top_k_fast

RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", "exp17")
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(ROOT_DIR, "experiments", "data", "real_world_corpus", "train.jsonl")
DIMS = ["valence", "arousal", "dominance", "curiosity", "stress"]

SEED = 20260819
MAX_CONVERSATIONS = 100
MIN_SESSIONS_PER_CONVERSATION = 2
MIN_STORED_TURNS_IN_TARGET_SESSION = 3
K_LIST = (5, 10)
ARMS = ("semantic_only", "recency_only", "ncm_inferred", "ncm_oracle")
STATE_STABILITY_CONVERSATIONS = 20
STATE_STABILITY_TURNS_PER_CONVERSATION = 20


@dataclass
class Turn:
    speaker: str
    text: str
    session_id: int


@dataclass
class ConversationData:
    conv_id: int
    turns: list[Turn]

    @property
    def session_ids(self) -> list[int]:
        seen: list[int] = []
        for t in self.turns:
            if t.session_id not in seen:
                seen.append(t.session_id)
        return seen


# ───────────────────────────────────────────
# CORPUS
# ───────────────────────────────────────────

def load_conversations(corpus_path: str, max_conversations: int) -> list[ConversationData]:
    """
    Load conversations, keeping the session_id of every turn.

    The previous version flattened sessions and discarded session_id, which is
    why no relevance label was available to it.
    """
    conversations: list[ConversationData] = []

    try:
        handle = open(corpus_path, "r", encoding="utf-8")
    except FileNotFoundError:
        print(f"[exp17] ERROR: corpus not found at {corpus_path}")
        return []

    with handle as f:
        for line in f:
            if len(conversations) >= max_conversations:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns: list[Turn] = []
            for session_index, session in enumerate(data.get("sessions", [])):
                session_id = int(session.get("session_id", session_index))
                for turn in session.get("dialogue", []):
                    text = (turn.get("text") or "").strip()
                    if text:
                        turns.append(Turn(
                            speaker=turn.get("speaker", "Unknown"),
                            text=text,
                            session_id=session_id,
                        ))

            if turns:
                conversations.append(ConversationData(
                    conv_id=int(data.get("id", len(conversations))),
                    turns=turns,
                ))

    return conversations


# ───────────────────────────────────────────
# METRICS
# ───────────────────────────────────────────

def precision_at_k(retrieved_labels: list[bool], k: int) -> float:
    """Fraction of the top-k that are relevant. Denominator is min(k, len)."""
    top = retrieved_labels[:k]
    if not top:
        return 0.0
    return float(sum(top)) / float(len(top))


def recall_at_k(retrieved_labels: list[bool], k: int, n_relevant: int) -> float:
    if n_relevant <= 0:
        return 0.0
    return float(sum(retrieved_labels[:k])) / float(n_relevant)


def reciprocal_rank(retrieved_labels: list[bool]) -> float:
    for rank, is_relevant in enumerate(retrieved_labels, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_labels: list[bool], k: int, n_relevant: int) -> float:
    """Binary-gain NDCG@k. Ideal ranking puts min(k, n_relevant) hits first."""
    gains = [1.0 if hit else 0.0 for hit in retrieved_labels[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


# ───────────────────────────────────────────
# BENCHMARK
# ───────────────────────────────────────────

def _new_metric_accumulator() -> dict:
    acc = {f"p@{k}": [] for k in K_LIST}
    acc.update({f"r@{k}": [] for k in K_LIST})
    acc["ndcg@10"] = []
    acc["mrr"] = []
    acc["latency_ms"] = []
    return acc


def _score_arm(acc: dict, labels: list[bool], n_relevant: int, latency_ms: float) -> None:
    for k in K_LIST:
        acc[f"p@{k}"].append(precision_at_k(labels, k))
        acc[f"r@{k}"].append(recall_at_k(labels, k, n_relevant))
    acc["ndcg@10"].append(ndcg_at_k(labels, 10, n_relevant))
    acc["mrr"].append(reciprocal_rank(labels))
    acc["latency_ms"].append(latency_ms)


def _labels_from_entries(entries, target_session: int, session_of: dict[str, int]) -> list[bool]:
    return [session_of.get(mem.id, -1) == target_session for mem in entries]


def benchmark(conversations: list[ConversationData], encoder: SentenceEncoder) -> dict:
    """
    One store per conversation, holding every turn of that conversation.
    For each session with enough remaining turns, hold out one turn as the
    query and score each arm against same-session relevance.
    """
    rng = random.Random(SEED)
    acc = {arm: _new_metric_accumulator() for arm in ARMS}

    conversations_benchmarked = 0
    conversations_skipped_sessions = 0
    conversations_skipped_coverage = 0
    queries_evaluated = 0
    total_turns_stored = 0
    relevant_counts: list[int] = []
    store_sizes: list[int] = []
    # Random-guess P@k is an expectation over QUERIES, so it needs the store
    # size that each query actually saw. store_sizes below is per conversation
    # and has a different length, so it cannot be used for that ratio.
    per_query_relevant_fraction: list[float] = []
    recency_window_sessions: list[int] = []
    recency_window_is_target_only: list[float] = []
    max_k = max(K_LIST)

    for conv in conversations:
        session_ids = conv.session_ids
        if len(session_ids) < MIN_SESSIONS_PER_CONVERSATION:
            conversations_skipped_sessions += 1
            continue

        # Pick one held-out query turn per session, deterministically.
        held_out_index: dict[int, int] = {}
        for session_id in session_ids:
            candidate_indices = [i for i, t in enumerate(conv.turns) if t.session_id == session_id]
            # Need the query turn plus enough same-session turns left in the store.
            if len(candidate_indices) < MIN_STORED_TURNS_IN_TARGET_SESSION + 1:
                continue
            held_out_index[session_id] = rng.choice(candidate_indices)

        if not held_out_index:
            conversations_skipped_coverage += 1
            continue

        held_out_positions = set(held_out_index.values())

        # Build the store from every turn except the held-out queries.
        store = MemoryStore()
        session_of_memory: dict[str, int] = {}
        for position, turn in enumerate(conv.turns):
            if position in held_out_positions:
                continue
            state_before = store.auto_state.get_current_state()
            mem = MemoryEntry(
                e_semantic=encoder.encode(turn.text),
                e_emotional=encoder.encode_emotional(state_before),
                # s_snapshot is retained for file-format compatibility. The
                # composite distance uses auto_state_snapshot, which add()
                # writes, so the 0.5-padding here does not enter retrieval.
                s_snapshot=encoder.encode_state(
                    np.pad(state_before, (0, 2), mode="constant", constant_values=0.5)
                ),
                timestamp=int(store.step),
                text=turn.text,
            )
            stored = store.add(mem, update_auto_state=True)
            session_of_memory[stored.id] = turn.session_id
            store.step += 1

        if len(store) == 0:
            continue

        # Warm the vectorized cache before any timing. store.add() marks the
        # cache dirty, and retrieve_top_k_fast rebuilds it on its first call,
        # so without this the ncm_inferred arm paid the whole rebuild for the
        # first query of every conversation while ncm_oracle, which runs next
        # and does identical work, read it warm. That made the two NCM arms
        # look like they had different costs when only cache state differed.
        # retrieve_semantic_only does not use this cache; it rebuilds its own
        # (n, 128) matrix on every call, which is disclosed with the latencies.
        store._rebuild_cache()

        store_sizes.append(len(store))
        total_turns_stored += len(store)
        conversations_benchmarked += 1

        # Precompute the per-session mean auto-state for the oracle arm.
        session_state_mean: dict[int, np.ndarray] = {}
        for session_id in held_out_index:
            snapshots = [
                m.auto_state_snapshot
                for m in store.get_all_safe()
                if session_of_memory.get(m.id, -1) == session_id
                and m.auto_state_snapshot is not None
            ]
            if snapshots:
                session_state_mean[session_id] = np.mean(
                    np.stack(snapshots), axis=0
                ).astype(np.float32)

        saved_state = store.auto_state.get_current_state()
        saved_turn = store.auto_state.turn

        for session_id, position in held_out_index.items():
            query_text = conv.turns[position].text
            n_relevant = sum(
                1 for m in store.get_all_safe()
                if session_of_memory.get(m.id, -1) == session_id
            )
            if n_relevant < MIN_STORED_TURNS_IN_TARGET_SESSION:
                continue

            queries_evaluated += 1
            relevant_counts.append(n_relevant)
            per_query_relevant_fraction.append(n_relevant / float(len(store)))
            q_sem = encoder.encode(query_text)

            # --- semantic_only -------------------------------------------
            t0 = perf_counter()
            hits = retrieve_semantic_only(q_sem, store, k=max_k)
            latency = (perf_counter() - t0) * 1000.0
            labels = _labels_from_entries([h[-1] for h in hits], session_id, session_of_memory)
            _score_arm(acc["semantic_only"], labels, n_relevant, latency)

            # --- recency_only --------------------------------------------
            t0 = perf_counter()
            recent = sorted(store.get_all_safe(), key=lambda m: m.timestamp, reverse=True)[:max_k]
            latency = (perf_counter() - t0) * 1000.0
            labels = _labels_from_entries(recent, session_id, session_of_memory)
            _score_arm(acc["recency_only"], labels, n_relevant, latency)
            # Diagnostic for why recency P@5 and P@10 come out nearly equal.
            # Sessions are contiguous blocks of turns, so when the final
            # session holds at least max_k stored turns the whole recency
            # window sits inside it: every query from that session scores 1.0
            # at both k, and every other query scores 0.0. Recording the
            # number of distinct sessions in the window makes that visible
            # instead of leaving it to look like a scoring bug.
            recency_window_sessions.append(
                len({session_of_memory.get(m.id, -1) for m in recent})
            )
            recency_window_is_target_only.append(
                1.0 if all(session_of_memory.get(m.id, -1) == session_id for m in recent) else 0.0
            )

            # --- ncm_inferred: state read off the query text alone -------
            probe = AutoStateTracker()
            inferred_state = probe.update(query_text)
            store.auto_state.state = inferred_state.astype(np.float32).copy()
            q_emo = encoder.encode_emotional(inferred_state)
            t0 = perf_counter()
            hits = retrieve_top_k_fast(
                q_sem, q_emo, store,
                encoder.encode_state(inferred_state), int(store.step), k=max_k,
            )
            latency = (perf_counter() - t0) * 1000.0
            labels = _labels_from_entries([h[-1] for h in hits], session_id, session_of_memory)
            _score_arm(acc["ncm_inferred"], labels, n_relevant, latency)

            # --- ncm_oracle: label-leaking upper bound -------------------
            oracle_state = session_state_mean.get(session_id)
            if oracle_state is not None:
                store.auto_state.state = oracle_state.copy()
                q_emo_oracle = encoder.encode_emotional(oracle_state)
                t0 = perf_counter()
                hits = retrieve_top_k_fast(
                    q_sem, q_emo_oracle, store,
                    encoder.encode_state(oracle_state), int(store.step), k=max_k,
                )
                latency = (perf_counter() - t0) * 1000.0
                labels = _labels_from_entries([h[-1] for h in hits], session_id, session_of_memory)
                _score_arm(acc["ncm_oracle"], labels, n_relevant, latency)

            # Restore the tracker so the next query starts from the same place.
            store.auto_state.state = saved_state.copy()
            store.auto_state.turn = saved_turn

    def summarize(arm_acc: dict) -> dict:
        out = {}
        for metric, values in arm_acc.items():
            if not values:
                out[metric] = 0.0
                continue
            if metric == "latency_ms":
                out["latency_ms_median"] = round(float(np.median(values)), 4)
                out["latency_ms_p95"] = round(float(np.percentile(values, 95)), 4)
                out["latency_ms_mean"] = round(float(np.mean(values)), 4)
            else:
                out[metric] = round(float(np.mean(values)), 4)
        return out

    # Expectation of P@k for a retriever that draws k memories uniformly at
    # random, averaged over queries: E[P@k] = mean_q(n_relevant_q / |store_q|).
    # This is independent of k. The earlier form, mean(relevant_counts) /
    # mean(store_sizes), divided a per-query mean by a per-conversation mean,
    # so its two samples had different lengths (one entry per query versus one
    # per conversation) and the result was not an expectation over anything.
    random_baseline_p5 = (
        round(float(np.mean(per_query_relevant_fraction)), 4)
        if per_query_relevant_fraction else 0.0
    )

    return {
        "arms": {arm: summarize(acc[arm]) for arm in ARMS},
        "dataset": {
            "conversations_loaded": len(conversations),
            "conversations_benchmarked": conversations_benchmarked,
            "conversations_skipped_too_few_sessions": conversations_skipped_sessions,
            "conversations_skipped_no_eligible_session": conversations_skipped_coverage,
            "queries_evaluated": queries_evaluated,
            "total_turns_stored": total_turns_stored,
            "avg_turns_per_benchmarked_conversation": (
                round(total_turns_stored / conversations_benchmarked, 1)
                if conversations_benchmarked else 0.0
            ),
            "avg_relevant_per_query": (
                round(float(np.mean(relevant_counts)), 2) if relevant_counts else 0.0
            ),
            "avg_store_size": round(float(np.mean(store_sizes)), 1) if store_sizes else 0.0,
            "random_guess_precision_at_5": random_baseline_p5,
            "random_guess_definition": (
                "mean over queries of n_relevant / store_size; independent of k"
            ),
            "recency_window_mean_distinct_sessions": (
                round(float(np.mean(recency_window_sessions)), 3)
                if recency_window_sessions else 0.0
            ),
            "recency_window_fraction_entirely_target_session": (
                round(float(np.mean(recency_window_is_target_only)), 4)
                if recency_window_is_target_only else 0.0
            ),
        },
    }


def measure_state_stability(conversations: list[ConversationData]) -> dict:
    """
    Spread and entropy of the auto-state across diverse conversations.

    Reports the exact number of turns consumed rather than a hardcoded figure;
    the previous version's output text asserted "400 turns" regardless of how
    many turns the sampled conversations actually contained.
    """
    records = []
    conversations_used = 0

    for conv in conversations[:min(STATE_STABILITY_CONVERSATIONS, len(conversations))]:
        tracker = AutoStateTracker()
        conversations_used += 1
        for turn in conv.turns[:STATE_STABILITY_TURNS_PER_CONVERSATION]:
            s = tracker.update(turn.text)
            records.append({
                "spread": float(np.std(s)),
                "range": float(np.max(s) - np.min(s)),
                "entropy": float(-np.sum(s * np.log(s + 1e-8))),
            })

    if not records:
        return {
            "conversations_used": 0, "turns_consumed": 0, "total_samples": 0,
            "mean_spread": 0.0, "std_spread": 0.0, "min_spread": 0.0,
            "max_spread": 0.0, "mean_range": 0.0, "mean_entropy": 0.0,
        }

    spreads = [r["spread"] for r in records]
    ranges = [r["range"] for r in records]
    entropies = [r["entropy"] for r in records]

    return {
        "conversations_used": conversations_used,
        "turns_consumed": len(records),
        "total_samples": len(records),
        "mean_spread": round(float(np.mean(spreads)), 4),
        "std_spread": round(float(np.std(spreads)), 4),
        "min_spread": round(float(np.min(spreads)), 4),
        "max_spread": round(float(np.max(spreads)), 4),
        "mean_range": round(float(np.mean(ranges)), 4),
        "mean_entropy": round(float(np.mean(entropies)), 4),
    }


# ───────────────────────────────────────────
# VISUALIZATIONS
# ───────────────────────────────────────────

ARM_LABELS = {
    "semantic_only": "Semantic only\n(dense RAG)",
    "recency_only": "Recency only\n(temporal control)",
    "ncm_inferred": "NCM\n(state from query)",
    "ncm_oracle": "NCM oracle\n(LABEL LEAK)",
}
ARM_COLORS = {
    "semantic_only": "#e74c3c",
    "recency_only": "#95a5a6",
    "ncm_inferred": "#2ecc71",
    "ncm_oracle": "#f39c12",
}


def plot_retrieval_precision(bench: dict) -> str:
    arms = bench["arms"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    metrics = ["p@5", "p@10"]
    x = np.arange(len(metrics))
    width = 0.2

    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * width
        vals = [arms[arm].get(m, 0.0) for m in metrics]
        bars = ax1.bar(x + offset, vals, width, label=ARM_LABELS[arm],
                       color=ARM_COLORS[arm], alpha=0.88, edgecolor="black", linewidth=1.1)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    rnd = bench["dataset"]["random_guess_precision_at_5"]
    ax1.axhline(rnd, color="black", linestyle=":", linewidth=1.6,
                label=f"Random guess ({rnd:.3f})")
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_title("Same-session retrieval precision", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["P@5", "P@10"])
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    ranking = ["ndcg@10", "mrr", "r@10"]
    x2 = np.arange(len(ranking))
    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * width
        vals = [arms[arm].get(m, 0.0) for m in ranking]
        bars = ax2.bar(x2 + offset, vals, width, label=ARM_LABELS[arm],
                       color=ARM_COLORS[arm], alpha=0.88, edgecolor="black", linewidth=1.1)
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Ranking quality", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["NDCG@10", "MRR", "Recall@10"])
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"EXP17 Multi-Session Chat: {bench['dataset']['queries_evaluated']} queries, "
        f"{bench['dataset']['conversations_benchmarked']} conversations",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp17_scale_retrieval_precision.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    return png_path


def plot_performance_metrics(bench: dict) -> str:
    arms = bench["arms"]
    fig, ax = plt.subplots(figsize=(11, 5))

    names = [ARM_LABELS[a] for a in ARMS]
    medians = [arms[a].get("latency_ms_median", 0.0) for a in ARMS]
    p95s = [arms[a].get("latency_ms_p95", 0.0) for a in ARMS]

    x = np.arange(len(names))
    width = 0.36
    b1 = ax.bar(x - width / 2, medians, width, label="Median",
                color="#3498db", alpha=0.88, edgecolor="black", linewidth=1.2)
    b2 = ax.bar(x + width / 2, p95s, width, label="95th percentile",
                color="#9b59b6", alpha=0.88, edgecolor="black", linewidth=1.2)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Retrieval latency (ms, perf_counter)", fontsize=11)
    ax.set_title(
        f"EXP17 per-query retrieval latency (mean store size "
        f"{bench['dataset']['avg_store_size']:.0f} memories)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp17_scale_performance_metrics.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def plot_state_accuracy(stability: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))

    if stability["total_samples"] == 0:
        ax.text(0.5, 0.5, "No state data available", ha="center", va="center", fontsize=12)
    else:
        names = ["Mean std-dev\nacross 5 dims", "Mean max-min\nrange", "Mean entropy"]
        values = [stability["mean_spread"], stability["mean_range"], stability["mean_entropy"]]
        colors = ["#3498db", "#1abc9c", "#9b59b6"]
        bars = ax.barh(names, values, color=colors, alpha=0.88, edgecolor="black", linewidth=1.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", ha="left", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=11)
        ax.text(
            0.98, 0.06,
            f"{stability['turns_consumed']} turns from "
            f"{stability['conversations_used']} conversations\n"
            f"std-dev range [{stability['min_spread']:.4f}, {stability['max_spread']:.4f}]",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )

    ax.set_title("EXP17 auto-state dispersion across diverse conversations",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp17_scale_state_accuracy.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"[exp17] Loading corpus (seed={SEED})")
    conversations = load_conversations(CORPUS_PATH, MAX_CONVERSATIONS)
    print(f"[exp17] Loaded {len(conversations)} conversations")
    if not conversations:
        print("[exp17] ERROR: no conversations loaded. Aborting.")
        return

    encoder = SentenceEncoder(
        model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models")
    )
    backend = encoder.backend
    print(f"[exp17] Encoder backend: {backend}")
    if backend != "sentence-transformers":
        print("[exp17] ABORT: hash fallback carries no semantic structure, so any")
        print(f"[exp17]        retrieval number would be meaningless. Reason: {encoder.backend_error}")
        return

    print("[exp17] Running same-session retrieval benchmark")
    bench = benchmark(conversations, encoder)

    print("[exp17] Measuring auto-state dispersion")
    stability = measure_state_stability(conversations)

    arms = bench["arms"]
    ncm = arms["ncm_inferred"]
    sem = arms["semantic_only"]
    rec = arms["recency_only"]

    deltas = {
        "ncm_inferred_minus_semantic_p5": round(ncm["p@5"] - sem["p@5"], 4),
        "ncm_inferred_minus_semantic_ndcg10": round(ncm["ndcg@10"] - sem["ndcg@10"], 4),
        "ncm_inferred_minus_recency_p5": round(ncm["p@5"] - rec["p@5"], 4),
        "ncm_oracle_minus_ncm_inferred_p5": round(arms["ncm_oracle"]["p@5"] - ncm["p@5"], 4),
    }

    best_arm = max(ARMS, key=lambda a: arms[a]["p@5"])

    results = {
        "experiment": "exp17_same_session_episodic_retrieval",
        "config": {
            "seed": SEED,
            "corpus": "experiments/data/real_world_corpus/train.jsonl",
            "encoder_backend": backend,
            "max_conversations": MAX_CONVERSATIONS,
            "min_sessions_per_conversation": MIN_SESSIONS_PER_CONVERSATION,
            "min_stored_turns_in_target_session": MIN_STORED_TURNS_IN_TARGET_SESSION,
            "k_values": list(K_LIST),
            "relevance_definition": "a stored turn is relevant iff it shares the "
                                    "held-out query turn's session_id",
            "retrieval_paths": {
                "semantic_only": "ncm.retrieval.retrieve_semantic_only",
                "recency_only": "sort by MemoryEntry.timestamp descending",
                "ncm_inferred": "ncm.retrieval.retrieve_top_k_fast, auto-state "
                                "inferred from the query text alone",
                "ncm_oracle": "ncm.retrieval.retrieve_top_k_fast, auto-state set "
                              "to the target session's mean stored auto-state "
                              "(LABEL LEAK, upper bound only)",
            },
            "timer": "time.perf_counter",
        },
        "dataset": bench["dataset"],
        "arms": arms,
        "deltas": deltas,
        "best_arm_by_p5": best_arm,
        "state_dispersion": stability,
    }

    json_path = os.path.join(RESULTS_DIR, "exp17_real_world_scale.json")
    txt_path = os.path.join(RESULTS_DIR, "exp17_real_world_scale.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ds = bench["dataset"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EXP17: Same-Session Episodic Retrieval on Multi-Session Chat\n")
        f.write("===========================================================\n\n")
        f.write("Task: retrieve turns sharing the held-out query turn's session_id.\n")
        f.write("Relevance comes from the corpus session_id field, not from any\n")
        f.write("label authored for this experiment.\n\n")
        f.write(f"Seed: {SEED}    Encoder backend: {backend}\n")
        f.write("Timer: time.perf_counter\n\n")
        f.write("Dataset\n")
        f.write(f"- Conversations loaded: {ds['conversations_loaded']}\n")
        f.write(f"- Conversations benchmarked: {ds['conversations_benchmarked']}\n")
        f.write(f"- Skipped, fewer than {MIN_SESSIONS_PER_CONVERSATION} sessions: "
                f"{ds['conversations_skipped_too_few_sessions']}\n")
        f.write(f"- Skipped, no session with at least "
                f"{MIN_STORED_TURNS_IN_TARGET_SESSION} stored turns: "
                f"{ds['conversations_skipped_no_eligible_session']}\n")
        f.write(f"- Queries evaluated: {ds['queries_evaluated']}\n")
        f.write(f"- Turns stored in total: {ds['total_turns_stored']}\n")
        f.write(f"- Mean store size: {ds['avg_store_size']}\n")
        f.write(f"- Mean turns per benchmarked conversation: "
                f"{ds['avg_turns_per_benchmarked_conversation']}\n")
        f.write(f"- Mean relevant turns per query: {ds['avg_relevant_per_query']}\n")
        f.write(f"- Random-guess P@5: {ds['random_guess_precision_at_5']:.4f}"
                f"  ({ds['random_guess_definition']})\n\n")

        header = f"{'arm':<16}{'P@5':>8}{'P@10':>8}{'R@10':>8}{'NDCG@10':>10}{'MRR':>8}{'med ms':>9}{'p95 ms':>9}\n"
        f.write("Results\n")
        f.write(header)
        f.write("-" * (len(header) - 1) + "\n")
        for arm in ARMS:
            a = arms[arm]
            f.write(f"{arm:<16}{a['p@5']:>8.4f}{a['p@10']:>8.4f}{a['r@10']:>8.4f}"
                    f"{a['ndcg@10']:>10.4f}{a['mrr']:>8.4f}"
                    f"{a['latency_ms_median']:>9.3f}{a['latency_ms_p95']:>9.3f}\n")
        f.write("\nncm_oracle leaks the session label into the query state. It bounds\n")
        f.write("what the state channel could contribute and is not a system result.\n\n")

        f.write("Latency caveat\n")
        f.write("These are not a latency benchmark. Mean store size is only\n")
        f.write(f"{ds['avg_store_size']} memories, so every arm is far below a millisecond and\n")
        f.write("fixed per-call overhead dominates. The two paths also differ in\n")
        f.write("cache treatment: retrieve_top_k_fast reads a cache warmed once per\n")
        f.write("conversation, while retrieve_semantic_only rebuilds its own (n, 128)\n")
        f.write("matrix on every call. Exp4 is the latency measurement; read these\n")
        f.write("figures only as evidence that the composite path stays sub-millisecond\n")
        f.write("at this scale.\n\n")

        f.write("Why recency_only scores the same at k=5 and k=10\n")
        f.write("Sessions are contiguous blocks of turns, so the most recent turns\n")
        f.write("nearly always sit inside one session. The recency window holds\n")
        f.write(f"{ds['recency_window_mean_distinct_sessions']:.3f} distinct sessions on average, and\n")
        f.write(f"{ds['recency_window_fraction_entirely_target_session']:.4f} of queries have a window drawn\n")
        f.write("entirely from the target session. Such a query scores 1.0 at both k\n")
        f.write("and every other query scores 0.0, which is why the two columns agree.\n")
        f.write("It is a property of the corpus layout, not a scoring error.\n\n")

        f.write("Deltas\n")
        for name, val in deltas.items():
            f.write(f"- {name}: {val:+.4f}\n")
        f.write(f"- Highest P@5 arm: {best_arm}\n\n")

        f.write("Auto-state dispersion\n")
        f.write(f"- Turns consumed: {stability['turns_consumed']} from "
                f"{stability['conversations_used']} conversations\n")
        f.write(f"- Mean per-turn std-dev across the 5 dimensions: "
                f"{stability['mean_spread']:.4f} (sd {stability['std_spread']:.4f})\n")
        f.write(f"- Std-dev range: [{stability['min_spread']:.4f}, "
                f"{stability['max_spread']:.4f}]\n")
        f.write(f"- Mean max-min range: {stability['mean_range']:.4f}\n")
        f.write(f"- Mean entropy: {stability['mean_entropy']:.4f}\n\n")

        f.write("Reading of the result\n")
        if deltas["ncm_inferred_minus_semantic_p5"] > 0 and deltas["ncm_inferred_minus_recency_p5"] > 0:
            f.write("NCM's composite distance beats both the semantic-only baseline and\n")
            f.write("the recency control on P@5, so the gain is not attributable to\n")
            f.write("recency alone.\n")
        elif deltas["ncm_inferred_minus_semantic_p5"] > 0:
            f.write("NCM beats the semantic-only baseline on P@5 but does not beat the\n")
            f.write("recency control. Because sessions are contiguous blocks of turns,\n")
            f.write("the gain is consistent with the temporal channel alone and this\n")
            f.write("experiment does not isolate a state-specific contribution.\n")
        else:
            f.write("NCM does not beat the semantic-only baseline on P@5 under the\n")
            f.write("deployable condition where the query state is inferred from the\n")
            f.write("query text. The composite distance does not help on this task.\n")

    print(f"[exp17] Saved: {json_path}")
    print(f"[exp17] Saved: {txt_path}")

    print("[exp17] Generating figures")
    for path in (
        plot_retrieval_precision(bench),
        plot_performance_metrics(bench),
        plot_state_accuracy(stability),
    ):
        print(f"[exp17] Saved: {path}")

    print("[exp17] Summary (P@5): " + ", ".join(
        f"{arm}={arms[arm]['p@5']:.4f}" for arm in ARMS
    ))
    print(f"[exp17] Random-guess P@5: {ds['random_guess_precision_at_5']:.4f}")


if __name__ == "__main__":
    main()
