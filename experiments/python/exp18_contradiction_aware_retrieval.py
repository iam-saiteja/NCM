"""
EXP18: Contradiction-Aware Distance Penalty (CADP) Validation
==============================================================

Purpose
- Validate contradiction-aware retrieval for corrected facts without deleting history.
- Compare baseline manifold retrieval vs CADP on contradiction-heavy synthetic tasks.
- Check non-contradiction regression, latency overhead, and .ncm persistence of links.

Outputs
- experiments/results/exp18/exp18_cadp_validation.json
- experiments/results/exp18/exp18_cadp_validation.txt
- experiments/results/exp18/exp18_rank_accuracy.png
- experiments/results/exp18/exp18_latency_regression.png
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import MemoryEntry, MemoryProfile, MemoryStore, NCMFile, SentenceEncoder, retrieve_top_k_fast

RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", "exp18")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


def make_profile(name: str, cadp: bool, write_trace: bool = False) -> MemoryProfile:
    profile = MemoryProfile(name=name)
    profile.set_custom("enable_contradiction_awareness", cadp)
    profile.set_custom("contradiction_penalty", 0.20)
    profile.set_custom("contradiction_similarity_threshold", 0.82)
    profile.set_custom("write_conflict_trace", write_trace)
    profile.set_custom("contradiction_requires_marker", True)
    profile.set_custom("contradiction_query_gate", 1.0)
    return profile


def state7(store: MemoryStore) -> np.ndarray:
    s5 = store.auto_state.get_current_state().astype(np.float32)
    return np.pad(s5, (0, 2), mode="constant", constant_values=0.5)


def add_text(store: MemoryStore, encoder: SentenceEncoder, text: str, tags: list[str] | None = None) -> MemoryEntry:
    s7 = state7(store)
    m = MemoryEntry(
        e_semantic=encoder.encode(text),
        e_emotional=encoder.encode_emotional(s7),
        s_snapshot=encoder.encode_state(s7),
        timestamp=int(store.step),
        text=text,
        tags=tags or [],
    )
    store.add(m, update_auto_state=True)
    store.step += 1
    return m


def retrieve_texts(store: MemoryStore, encoder: SentenceEncoder, query: str, k: int = 5) -> list[str]:
    s7 = state7(store)
    q_sem = encoder.encode(query)
    q_emo = encoder.encode_emotional(s7)
    q_state = encoder.encode_state(s7)
    rows = retrieve_top_k_fast(q_sem, q_emo, store, q_state, store.step, k=k, use_strength=False)
    return [m.text for _, _, m in rows]


def rank_of(texts: list[str], needle: str) -> int:
    for i, t in enumerate(texts, 1):
        if t == needle:
            return i
    return 999


def scenario_single_correction(encoder: SentenceEncoder, trials: int = 50) -> dict[str, Any]:
    baseline_wins = 0
    cadp_wins = 0

    for i in range(trials):
        subject = f"project code {i}"
        old_value = f"A{i:03d}"
        new_value = f"B{i:03d}"

        old_text = f"My {subject} is {old_value}"
        new_text = f"Correction: my {subject} is {new_value}"
        query = f"What is my {subject} now?"

        base_store = MemoryStore(profile=make_profile("baseline", cadp=False))
        cadp_store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=False))

        # Noise memories
        for j in range(4):
            noise = f"I discussed unrelated topic {i}-{j} in a meeting"
            add_text(base_store, encoder, noise)
            add_text(cadp_store, encoder, noise)

        add_text(base_store, encoder, old_text)
        add_text(base_store, encoder, new_text)

        add_text(cadp_store, encoder, old_text)
        add_text(cadp_store, encoder, new_text)

        top_base = retrieve_texts(base_store, encoder, query, k=10)
        top_cadp = retrieve_texts(cadp_store, encoder, query, k=10)

        if rank_of(top_base, new_text) < rank_of(top_base, old_text):
            baseline_wins += 1
        if rank_of(top_cadp, new_text) < rank_of(top_cadp, old_text):
            cadp_wins += 1

    return {
        "trials": trials,
        "baseline_new_beats_old_rate": round(baseline_wins / trials, 3),
        "cadp_new_beats_old_rate": round(cadp_wins / trials, 3),
        "absolute_gain": round((cadp_wins - baseline_wins) / trials, 3),
    }


def scenario_chain_corrections(encoder: SentenceEncoder, trials: int = 40) -> dict[str, Any]:
    baseline_latest_top1 = 0
    cadp_latest_top1 = 0

    for i in range(trials):
        subject = f"office seat {i}"
        a = f"My {subject} is A{i:03d}"
        b = f"Update: my {subject} is B{i:03d}"
        c = f"Final correction: my {subject} is C{i:03d}"
        query = f"What is my current {subject}?"

        base_store = MemoryStore(profile=make_profile("baseline", cadp=False))
        cadp_store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=False))

        add_text(base_store, encoder, a)
        add_text(base_store, encoder, b)
        add_text(base_store, encoder, c)

        add_text(cadp_store, encoder, a)
        add_text(cadp_store, encoder, b)
        add_text(cadp_store, encoder, c)

        top_base = retrieve_texts(base_store, encoder, query, k=5)
        top_cadp = retrieve_texts(cadp_store, encoder, query, k=5)

        if top_base and top_base[0] == c:
            baseline_latest_top1 += 1
        if top_cadp and top_cadp[0] == c:
            cadp_latest_top1 += 1

    return {
        "trials": trials,
        "baseline_latest_top1_rate": round(baseline_latest_top1 / trials, 3),
        "cadp_latest_top1_rate": round(cadp_latest_top1 / trials, 3),
        "absolute_gain": round((cadp_latest_top1 - baseline_latest_top1) / trials, 3),
    }


def scenario_conflict_trace(encoder: SentenceEncoder, trials: int = 40) -> dict[str, Any]:
    trace_in_top3 = 0

    for i in range(trials):
        subject = f"hometown field {i}"
        old_text = f"My {subject} is Pune"
        new_text = f"Correction: my {subject} is Mumbai"
        query = f"Did my {subject} information change?"

        store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=True))
        add_text(store, encoder, old_text)
        add_text(store, encoder, new_text)

        top = retrieve_texts(store, encoder, query, k=3)
        if any("[UPDATE]" in t for t in top):
            trace_in_top3 += 1

    return {
        "trials": trials,
        "trace_top3_rate": round(trace_in_top3 / trials, 3),
    }


def scenario_non_contradiction_regression(encoder: SentenceEncoder) -> dict[str, Any]:
    base_store = MemoryStore(profile=make_profile("baseline", cadp=False))
    cadp_store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=False))

    facts = [
        "I enjoy hiking on weekends",
        "My favorite cuisine is Korean food",
        "I usually work out in the evening",
        "I am learning distributed systems",
        "I read science fiction novels",
        "I have a dentist appointment next Tuesday",
        "I use Linux for development",
        "I prefer quiet cafes for work",
        "I like jazz and classical music",
        "My commute takes about 25 minutes",
    ]
    queries = [
        "What kind of books do I read?",
        "What cuisine do I like?",
        "When do I work out?",
        "What OS do I use for development?",
        "How long is my commute?",
    ]

    for t in facts:
        add_text(base_store, encoder, t)
        add_text(cadp_store, encoder, t)

    unchanged_top1 = 0
    for q in queries:
        top_base = retrieve_texts(base_store, encoder, q, k=3)
        top_cadp = retrieve_texts(cadp_store, encoder, q, k=3)
        if top_base and top_cadp and top_base[0] == top_cadp[0]:
            unchanged_top1 += 1

    return {
        "queries": len(queries),
        "top1_unchanged": unchanged_top1,
        "top1_unchanged_ratio": round(unchanged_top1 / len(queries), 3),
    }


def scenario_latency(encoder: SentenceEncoder, n_queries: int = 80) -> dict[str, Any]:
    base_store = MemoryStore(profile=make_profile("baseline", cadp=False))
    cadp_store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=False))

    for i in range(120):
        t = f"Memory item {i} about project topic {i % 20}"
        add_text(base_store, encoder, t)
        add_text(cadp_store, encoder, t)

    queries = [f"What do we know about topic {i % 20}?" for i in range(n_queries)]

    t0 = time.perf_counter()
    for q in queries:
        retrieve_texts(base_store, encoder, q, k=5)
    base_ms = (time.perf_counter() - t0) * 1000 / n_queries

    t0 = time.perf_counter()
    for q in queries:
        retrieve_texts(cadp_store, encoder, q, k=5)
    cadp_ms = (time.perf_counter() - t0) * 1000 / n_queries

    return {
        "queries": n_queries,
        "baseline_avg_latency_ms": round(base_ms, 4),
        "cadp_avg_latency_ms": round(cadp_ms, 4),
        "latency_delta_ms": round(cadp_ms - base_ms, 4),
    }


def scenario_persistence(encoder: SentenceEncoder) -> dict[str, Any]:
    store = MemoryStore(profile=make_profile("cadp", cadp=True, write_trace=True))
    add_text(store, encoder, "My employee id is E-101")
    add_text(store, encoder, "Correction: my employee id is E-222")
    add_text(store, encoder, "I prefer remote work")

    before_links = sum(1 for m in store.get_all_safe() if m.contradicted_by is not None)
    before_traces = sum(1 for m in store.get_all_safe() if m.is_conflict_trace)

    fd, tmp_path = tempfile.mkstemp(prefix="exp18_", suffix=".ncm")
    os.close(fd)

    NCMFile.save(store, tmp_path, compress=True, fp16=False)
    loaded = NCMFile.load(tmp_path)

    after_links = sum(1 for m in loaded.get_all_safe() if m.contradicted_by is not None)
    after_traces = sum(1 for m in loaded.get_all_safe() if m.is_conflict_trace)

    return {
        "before_links": before_links,
        "after_links": after_links,
        "before_conflict_traces": before_traces,
        "after_conflict_traces": after_traces,
        "pass": bool(before_links == after_links and before_traces == after_traces),
    }


def make_plots(results: dict[str, Any]) -> dict[str, str]:
    # Plot 1: ranking accuracy gains
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Single\nCorrection", "Chain\nLatest@1"]
    baseline_vals = [
        results["single_correction"]["baseline_new_beats_old_rate"],
        results["chain_corrections"]["baseline_latest_top1_rate"],
    ]
    cadp_vals = [
        results["single_correction"]["cadp_new_beats_old_rate"],
        results["chain_corrections"]["cadp_latest_top1_rate"],
    ]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, baseline_vals, w, label="Baseline", color="#e67e22", alpha=0.85)
    ax.bar(x + w / 2, cadp_vals, w, label="CADP", color="#2ecc71", alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("EXP18 Ranking Accuracy on Contradiction Tasks", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    p1 = os.path.join(RESULTS_DIR, "exp18_rank_accuracy.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot 2: latency + non-contradiction stability
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = [
        results["latency"]["baseline_avg_latency_ms"],
        results["latency"]["cadp_avg_latency_ms"],
        results["non_contradiction_regression"]["top1_unchanged_ratio"] * 100.0,
    ]
    labels2 = ["Baseline\nLatency (ms)", "CADP\nLatency (ms)", "Top1 Unchanged\n(%)"]
    colors = ["#3498db", "#9b59b6", "#27ae60"]

    ax.bar(labels2, bars, color=colors, alpha=0.85)
    ax.set_title("EXP18 Latency & Regression Check", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    p2 = os.path.join(RESULTS_DIR, "exp18_latency_regression.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    return {
        "exp18_rank_accuracy": p1,
        "exp18_latency_regression": p2,
    }


def write_text_summary(results: dict[str, Any], out_path: str) -> None:
    lines = []
    lines.append("EXP18: Contradiction-Aware Distance Penalty (CADP) Validation")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"Single correction (new beats old): baseline={results['single_correction']['baseline_new_beats_old_rate']:.3f}, CADP={results['single_correction']['cadp_new_beats_old_rate']:.3f}")
    lines.append(f"Chain correction latest@1: baseline={results['chain_corrections']['baseline_latest_top1_rate']:.3f}, CADP={results['chain_corrections']['cadp_latest_top1_rate']:.3f}")
    lines.append(f"Conflict trace top-3 rate: {results['conflict_trace']['trace_top3_rate']:.3f}")
    lines.append(f"Non-contradiction top-1 unchanged ratio: {results['non_contradiction_regression']['top1_unchanged_ratio']:.3f}")
    lines.append(f"Latency avg (ms): baseline={results['latency']['baseline_avg_latency_ms']:.4f}, CADP={results['latency']['cadp_avg_latency_ms']:.4f}, delta={results['latency']['latency_delta_ms']:.4f}")
    lines.append(f"Persistence pass: {results['persistence']['pass']}")
    lines.append("")
    lines.append("Verdict: PASS" if results["verdict"] == "PASS" else "Verdict: FAIL")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run() -> dict[str, Any]:
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))

    results = {
        "single_correction": scenario_single_correction(encoder),
        "chain_corrections": scenario_chain_corrections(encoder),
        "conflict_trace": scenario_conflict_trace(encoder),
        "non_contradiction_regression": scenario_non_contradiction_regression(encoder),
        "latency": scenario_latency(encoder),
        "persistence": scenario_persistence(encoder),
    }

    pass_checks = [
        results["single_correction"]["cadp_new_beats_old_rate"] >= 0.90,
        results["chain_corrections"]["cadp_latest_top1_rate"] >= 0.90,
        results["conflict_trace"]["trace_top3_rate"] >= 0.80,
        results["non_contradiction_regression"]["top1_unchanged_ratio"] >= 0.95,
        results["persistence"]["pass"],
    ]
    results["verdict"] = "PASS" if all(pass_checks) else "FAIL"

    plot_paths = make_plots(results)
    results["plots"] = {k: os.path.relpath(v, ROOT_DIR).replace("\\", "/") for k, v in plot_paths.items()}

    out_json = os.path.join(RESULTS_DIR, "exp18_cadp_validation.json")
    out_txt = os.path.join(RESULTS_DIR, "exp18_cadp_validation.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_text_summary(results, out_txt)
    return results


if __name__ == "__main__":
    final = run()
    print(json.dumps(final, indent=2))
