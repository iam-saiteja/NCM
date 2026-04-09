"""
EXP12: Weight Sensitivity Analysis
==================================

Goal
- Sweep retrieval weights (alpha, beta, gamma, delta) on a real corpus slice.
- Show whether NCM is robust to weight changes or brittle around the defaults.

Evaluation
- Real-world multi-session chat corpus under experiments/data/real_world_corpus
- Measures average Recall@10, NDCG@10, MRR, and state divergence
- Uses the same query/state machinery as EXP11

Output
- results/exp12_weight_sensitivity.json
- results/exp12_weight_sensitivity.txt
- results/exp12_weight_sensitivity.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.profile import RetrievalWeights
from ncm.retrieval import retrieve_top_k_fast
from experiments.exp11_real_world_corpus_benchmark import (
    DEFAULT_CORPUS_DIR,
    build_query_records,
    build_state_projector,
    build_store,
    context_to_state,
    extract_retrieved_ids,
    jaccard_divergence,
    load_corpus,
    mrr,
    ndcg_at_k,
    relevant_ids_for_query,
    recall_at_k,
    SentenceEncoder,
)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


WEIGHT_PRESETS = [
    {"name": "default", "alpha": 0.40, "beta": 0.20, "gamma": 0.30, "delta": 0.10},
    {"name": "semantic_heavy", "alpha": 0.55, "beta": 0.15, "gamma": 0.20, "delta": 0.10},
    {"name": "semantic_light", "alpha": 0.25, "beta": 0.20, "gamma": 0.35, "delta": 0.20},
    {"name": "emotional_heavy", "alpha": 0.30, "beta": 0.40, "gamma": 0.20, "delta": 0.10},
    {"name": "state_heavy", "alpha": 0.30, "beta": 0.10, "gamma": 0.50, "delta": 0.10},
    {"name": "temporal_heavy", "alpha": 0.35, "beta": 0.15, "gamma": 0.20, "delta": 0.30},
    {"name": "no_temporal", "alpha": 0.45, "beta": 0.20, "gamma": 0.35, "delta": 0.00},
]


def log(message: str) -> None:
    print(f"[exp12 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate_weights(corpus_dir: str, max_chunks: int, query_stride: int, top_k: int, verbose: bool) -> dict:
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    corpus = load_corpus(corpus_dir, verbose=verbose)
    if not corpus:
        return {"status": "skipped", "reason": f"No corpus files found in {corpus_dir}"}

    if max_chunks > 0 and len(corpus) > max_chunks:
        corpus = corpus[:max_chunks]
        if verbose:
            log(f"Using first {len(corpus)} chunk(s) from the corpus")

    store = build_store(corpus, encoder, verbose=verbose, progress_every=1000)
    projector = build_state_projector(encoder.semantic_dim)
    queries = build_query_records(corpus, query_stride=query_stride)
    if verbose:
        log(f"Built {len(queries)} query chunk(s)")

    states = ["state_a", "state_b"]
    results = {
        "metadata": {
            "corpus_dir": corpus_dir,
            "num_chunks": len(corpus),
            "num_queries": len(queries),
            "query_stride": query_stride,
            "top_k": top_k,
            "timestamp": time.time(),
        },
        "weight_sweep": [],
    }

    for preset in WEIGHT_PRESETS:
        weights = RetrievalWeights(
            alpha=preset["alpha"],
            beta=preset["beta"],
            gamma=preset["gamma"],
            delta=preset["delta"],
        )
        store.profile.retrieval_weights = weights
        if verbose:
            log(f"Evaluating weights: {preset['name']} = {weights.to_dict()}")

        per_state = {}
        state_sets = {}
        for state_name in states:
            recalls = []
            ndcgs = []
            mrrs = []
            retrieved_sets = []
            for query_idx, query in enumerate(queries, start=1):
                relevant = relevant_ids_for_query(store, query)
                state_vec = context_to_state(
                    encoder,
                    projector,
                    query.context_before if state_name == "state_a" else (query.context_after or query.context_before),
                    query.text,
                )
                q_sem = encoder.encode(query.text)
                q_emo = encoder.encode_emotional(state_vec)
                q_state = encoder.encode_state(state_vec)
                retrieved = retrieve_top_k_fast(q_sem, q_emo, store, q_state, current_step=store.step, k=top_k)
                retrieved_ids = extract_retrieved_ids(retrieved)
                retrieved_sets.append(retrieved_ids)
                recalls.append(recall_at_k(retrieved_ids, relevant, k=top_k))
                ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k=top_k))
                mrrs.append(mrr(retrieved_ids, relevant))
                if verbose and query_idx % 100 == 0:
                    log(f"  {preset['name']}/{state_name}: {query_idx}/{len(queries)} queries")

            per_state[state_name] = {
                "recall@10": float(np.mean(recalls)),
                "ndcg@10": float(np.mean(ndcgs)),
                "mrr": float(np.mean(mrrs)),
                "retrieved_sets": retrieved_sets,
            }
            state_sets[state_name] = retrieved_sets

        divergences = [jaccard_divergence(a, b) for a, b in zip(state_sets["state_a"], state_sets["state_b"])]
        summary = {
            "preset": preset,
            "weights": weights.to_dict(),
            "by_state": {
                state_name: {
                    "recall@10": per_state[state_name]["recall@10"],
                    "ndcg@10": per_state[state_name]["ndcg@10"],
                    "mrr": per_state[state_name]["mrr"],
                }
                for state_name in states
            },
            "overall": {
                "avg_recall@10": float(np.mean([per_state[s]["recall@10"] for s in states])),
                "avg_ndcg@10": float(np.mean([per_state[s]["ndcg@10"] for s in states])),
                "avg_mrr": float(np.mean([per_state[s]["mrr"] for s in states])),
                "jaccard_divergence_mean": float(np.mean(divergences)),
                "jaccard_divergence_std": float(np.std(divergences)),
            },
        }
        results["weight_sweep"].append(summary)

    return results


def write_outputs(results: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp12_weight_sensitivity.json")
    txt_path = os.path.join(RESULTS_DIR, "exp12_weight_sensitivity.txt")
    png_path = os.path.join(RESULTS_DIR, "exp12_weight_sensitivity.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sweep = results["weight_sweep"]
    ranked = sorted(sweep, key=lambda item: item["overall"]["avg_ndcg@10"], reverse=True)
    default = next(item for item in sweep if item["preset"]["name"] == "default")

    lines = [
        "EXP12: Weight Sensitivity Analysis",
        "=" * 35,
        f"Corpus: {results['metadata']['corpus_dir']}",
        f"Chunks: {results['metadata']['num_chunks']}",
        f"Queries: {results['metadata']['num_queries']}",
        "",
        "Top configurations by avg NDCG@10:",
    ]
    for item in ranked[:5]:
        preset = item["preset"]
        overall = item["overall"]
        lines.append(
            f"- {preset['name']}: NDCG={overall['avg_ndcg@10']:.3f}, R@10={overall['avg_recall@10']:.3f}, JaccardΔ={overall['jaccard_divergence_mean']:.3f}, weights={item['weights']}"
        )
    lines.append("")
    lines.append(
        f"Default weights: NDCG={default['overall']['avg_ndcg@10']:.3f}, R@10={default['overall']['avg_recall@10']:.3f}, JaccardΔ={default['overall']['jaccard_divergence_mean']:.3f}"
    )
    lines.append("Interpretation:")
    lines.append("- If best/worst spread is small, NCM is robust to weight choice.")
    lines.append("- If spread is large, the weights need to be tuned carefully before publication.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    names = [item["preset"]["name"] for item in sweep]
    ndcgs = [item["overall"]["avg_ndcg@10"] for item in sweep]
    divergences = [item["overall"]["jaccard_divergence_mean"] for item in sweep]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    width = 0.36
    ax1.bar(x - width / 2, ndcgs, width, label="NDCG@10", color="#4C78A8")
    ax1.set_ylabel("NDCG@10")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, divergences, width, label="JaccardΔ", color="#F58518", alpha=0.85)
    ax2.set_ylabel("State divergence")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=2)
    ax1.set_title("EXP12 Weight Sensitivity Analysis")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n✓ Saved {json_path}")
    print(f"✓ Saved {txt_path}")
    print(f"✓ Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the weight sensitivity analysis.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR, help="Corpus directory")
    parser.add_argument("--max-chunks", type=int, default=800, help="Max chunks to keep for the sweep")
    parser.add_argument("--query-stride", type=int, default=20, help="Use every Nth chunk as a query")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval cutoff")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = evaluate_weights(
        corpus_dir=args.corpus_dir,
        max_chunks=args.max_chunks,
        query_stride=args.query_stride,
        top_k=args.top_k,
        verbose=args.verbose,
    )
    if results.get("status") == "skipped":
        print(f"[SKIP] {results['reason']}")
        return 0
    write_outputs(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
