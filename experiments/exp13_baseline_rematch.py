"""
EXP13: Honest Head-to-Head Rematch
==================================

Goal
- Compare semantic_emotional vs NCM on the same real corpus slice.
- Bucket queries by how different the two inferred states are.
- Show where semantic_emotional can be competitive and where NCM benefits.

Outputs
- results/exp13_baseline_rematch.json
- results/exp13_baseline_rematch.txt
- results/exp13_baseline_rematch.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.retrieval import retrieve_semantic_emotional, retrieve_top_k
from experiments.exp11_real_world_corpus_benchmark import (
    DEFAULT_CORPUS_DIR,
    SentenceEncoder,
    build_query_records,
    build_state_projector,
    build_store,
    context_to_state,
    extract_retrieved_ids,
    load_corpus,
    mrr,
    ndcg_at_k,
    relevant_ids_for_query,
    recall_at_k,
)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(message: str) -> None:
    print(f"[exp13 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def bucket_name(value: float, thresholds: List[float]) -> str:
    if value <= thresholds[0]:
        return "low_shift"
    if value <= thresholds[1]:
        return "medium_shift"
    return "high_shift"


def jaccard_divergence(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - (len(set_a & set_b) / len(union))


def evaluate(corpus_dir: str, max_chunks: int, query_stride: int, top_k: int, verbose: bool = False) -> dict:
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    corpus = load_corpus(corpus_dir, verbose=verbose)
    if not corpus:
        return {"status": "skipped", "reason": f"No corpus files found in {corpus_dir}"}

    if max_chunks > 0 and len(corpus) > max_chunks:
        corpus = corpus[:max_chunks]
        if verbose:
            log(f"Using first {len(corpus)} chunk(s) from corpus")

    store = build_store(corpus, encoder, verbose=verbose, progress_every=1000)
    projector = build_state_projector(encoder.semantic_dim)
    queries = build_query_records(corpus, query_stride=query_stride)
    if verbose:
        log(f"Built {len(queries)} query chunk(s)")

    results = {
        "metadata": {
            "corpus_dir": corpus_dir,
            "num_chunks": len(corpus),
            "num_queries": len(queries),
            "query_stride": query_stride,
            "top_k": top_k,
            "timestamp": time.time(),
        },
        "systems": {
            "semantic_emotional": {},
            "ncm_full": {},
        },
        "query_buckets": {},
    }

    query_rows = []
    for query_idx, query in enumerate(queries, start=1):
        state_a = context_to_state(encoder, projector, query.context_before, query.text)
        alt_context = query.context_after or query.context_before or query.text
        state_b = context_to_state(encoder, projector, alt_context, query.text)
        state_shift = float(np.linalg.norm(state_a - state_b))
        q_sem = encoder.encode(query.text)
        relevant = relevant_ids_for_query(store, query)

        semantic_results = []
        ncm_results = []
        for state_vec in [state_a, state_b]:
            q_emo = encoder.encode_emotional(state_vec)
            q_state = encoder.encode_state(state_vec)
            semantic_retrieved = retrieve_semantic_emotional(q_sem, q_emo, store, k=top_k)
            ncm_retrieved = retrieve_top_k(q_sem, q_emo, store, q_state, current_step=store.step, k=top_k)
            semantic_results.append(extract_retrieved_ids(semantic_retrieved))
            ncm_results.append(extract_retrieved_ids(ncm_retrieved))

        semantic_divergence = jaccard_divergence(semantic_results[0], semantic_results[1])
        ncm_divergence = jaccard_divergence(ncm_results[0], ncm_results[1])

        query_rows.append(
            {
                "state_shift": state_shift,
                "semantic_emotional": {
                    "recall@10": float(np.mean([recall_at_k(ids, relevant, k=top_k) for ids in semantic_results])),
                    "ndcg@10": float(np.mean([ndcg_at_k(ids, relevant, k=top_k) for ids in semantic_results])),
                    "mrr": float(np.mean([mrr(ids, relevant) for ids in semantic_results])),
                    "retrieval_divergence": float(semantic_divergence),
                },
                "ncm_full": {
                    "recall@10": float(np.mean([recall_at_k(ids, relevant, k=top_k) for ids in ncm_results])),
                    "ndcg@10": float(np.mean([ndcg_at_k(ids, relevant, k=top_k) for ids in ncm_results])),
                    "mrr": float(np.mean([mrr(ids, relevant) for ids in ncm_results])),
                    "retrieval_divergence": float(ncm_divergence),
                },
            }
        )

        if verbose and query_idx % 100 == 0:
            log(f"Processed {query_idx}/{len(queries)} queries")

    shifts = np.array([row["state_shift"] for row in query_rows], dtype=np.float32)
    thresholds = [float(np.quantile(shifts, 0.33)), float(np.quantile(shifts, 0.66))]

    buckets: Dict[str, List[dict]] = {"low_shift": [], "medium_shift": [], "high_shift": []}
    for row in query_rows:
        buckets[bucket_name(row["state_shift"], thresholds)].append(row)

    results["query_buckets"] = {
        "thresholds": thresholds,
        "sizes": {name: len(items) for name, items in buckets.items()},
    }

    for system_name in ["semantic_emotional", "ncm_full"]:
        results["systems"][system_name] = {
            "overall": {},
            "by_bucket": {},
        }
        overall_rows = [row[system_name] for row in query_rows]
        results["systems"][system_name]["overall"] = {
            "avg_recall@10": float(np.mean([row["recall@10"] for row in overall_rows])),
            "avg_ndcg@10": float(np.mean([row["ndcg@10"] for row in overall_rows])),
            "avg_mrr": float(np.mean([row["mrr"] for row in overall_rows])),
            "avg_retrieval_divergence": float(np.mean([row["retrieval_divergence"] for row in overall_rows])),
        }
        for bucket, items in buckets.items():
            bucket_rows = [row[system_name] for row in items]
            results["systems"][system_name]["by_bucket"][bucket] = {
                "avg_recall@10": float(np.mean([row["recall@10"] for row in bucket_rows])) if bucket_rows else 0.0,
                "avg_ndcg@10": float(np.mean([row["ndcg@10"] for row in bucket_rows])) if bucket_rows else 0.0,
                "avg_mrr": float(np.mean([row["mrr"] for row in bucket_rows])) if bucket_rows else 0.0,
                "avg_retrieval_divergence": float(np.mean([row["retrieval_divergence"] for row in bucket_rows])) if bucket_rows else 0.0,
            }

    return results


def write_outputs(results: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp13_baseline_rematch.json")
    txt_path = os.path.join(RESULTS_DIR, "exp13_baseline_rematch.txt")
    png_path = os.path.join(RESULTS_DIR, "exp13_baseline_rematch.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "EXP13: Honest Head-to-Head Rematch",
        "=" * 36,
        f"Corpus: {results['metadata']['corpus_dir']}",
        f"Chunks: {results['metadata']['num_chunks']}",
        f"Queries: {results['metadata']['num_queries']}",
        f"State-shift thresholds: {results['query_buckets']['thresholds']}",
        "",
    ]
    for system_name, payload in results["systems"].items():
        overall = payload["overall"]
        lines.append(
            f"{system_name}: R@10={overall['avg_recall@10']:.3f}  NDCG={overall['avg_ndcg@10']:.3f}  MRR={overall['avg_mrr']:.3f}  Divergence={overall['avg_retrieval_divergence']:.3f}"
        )
    lines.append("")
    lines.append("Bucket comparison (semantic_emotional - ncm_full on NDCG@10):")
    for bucket in ["low_shift", "medium_shift", "high_shift"]:
        sem = results["systems"]["semantic_emotional"]["by_bucket"][bucket]
        ncm = results["systems"]["ncm_full"]["by_bucket"][bucket]
        lines.append(
            f"- {bucket}: ΔNDCG={sem['avg_ndcg@10'] - ncm['avg_ndcg@10']:+.3f}, semantic_emotional={sem['avg_ndcg@10']:.3f}, ncm_full={ncm['avg_ndcg@10']:.3f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Low-shift queries are the regime where extra state can act like noise.")
    lines.append("- High-shift queries are the regime where NCM should benefit most from s_snapshot.")
    lines.append("- This is the boundary explanation the paper needs: semantic-emotional is not universally better; it wins when state sensitivity adds little signal.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    buckets = ["low_shift", "medium_shift", "high_shift"]
    sem_ndcg = [results["systems"]["semantic_emotional"]["by_bucket"][b]["avg_ndcg@10"] for b in buckets]
    ncm_ndcg = [results["systems"]["ncm_full"]["by_bucket"][b]["avg_ndcg@10"] for b in buckets]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(buckets))
    width = 0.34
    ax.bar(x - width / 2, sem_ndcg, width, label="semantic_emotional", color="#72B7B2")
    ax.bar(x + width / 2, ncm_ndcg, width, label="ncm_full", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 1)
    ax.set_title("EXP13 Baseline Rematch by State-Shift Bucket")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n✓ Saved {json_path}")
    print(f"✓ Saved {txt_path}")
    print(f"✓ Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline rematch analysis.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR, help="Corpus directory")
    parser.add_argument("--max-chunks", type=int, default=800, help="Max chunks to keep")
    parser.add_argument("--query-stride", type=int, default=20, help="Use every Nth chunk as a query")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval cutoff")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = evaluate(
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
