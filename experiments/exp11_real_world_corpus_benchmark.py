"""
EXP11: Real-World Corpus Benchmark
===================================

Purpose
- Benchmark NCM on messy local text exports instead of synthetic categories.
- Supports chat logs, diaries, journal exports, notes, and multi-session text.
- Measures both standard retrieval quality and state-conditioned divergence.

Default behavior
- Looks for a local corpus directory at `data/real_world_corpus`.
- If no corpus is present, exits cleanly with a skip message.

Supported inputs
- `.txt`, `.md`: treated as one session per file; paragraphs become chunks.
- `.json`, `.jsonl`: attempts to flatten common conversation/session structures.

Metrics
- Recall@10 against same-session relevance
- NDCG@10 and MRR
- State divergence (1 - Jaccard similarity) across two real context states
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.retrieval import retrieve_top_k, retrieve_top_k_fast, retrieve_semantic_only, retrieve_semantic_emotional

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_CORPUS_DIR = os.path.join(ROOT_DIR, "experiments", "data", "real_world_corpus")


@dataclass
class CorpusChunk:
    session_id: str
    source_path: str
    order: int
    text: str
    context_before: str
    context_after: str
    state_hint: str = ""


STATE_PROJECTION_SEED = 19


def log(message: str) -> None:
    print(f"[exp11 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def list_corpus_files(corpus_dir: str) -> List[Path]:
    root = Path(corpus_dir)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.suffix.lower() in {".txt", ".md", ".json", ".jsonl"} and path.is_file()]
    )


def chunk_text(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(parts) >= 2:
        return parts
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines if len(lines) >= 2 else ([text.strip()] if text.strip() else [])


def flatten_json_payload(payload, source_name: str) -> List[CorpusChunk]:
    chunks: List[CorpusChunk] = []

    def emit(session_id: str, items: Iterable[str], prefix: str = ""):
        items = [item.strip() for item in items if item and item.strip()]
        for idx, item in enumerate(items):
            chunks.append(
                CorpusChunk(
                    session_id=session_id,
                    source_path=source_name,
                    order=idx,
                    text=f"{prefix}{item}".strip(),
                    context_before="",
                    context_after="",
                )
            )

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, str):
                emit(f"{source_name}::list", [item])
                continue
            if isinstance(item, dict):
                session_id = str(item.get("session_id") or item.get("conversation_id") or f"{source_name}::{idx}")
                if isinstance(item.get("sessions"), list):
                    for session_idx, session in enumerate(item["sessions"]):
                        if not isinstance(session, dict):
                            continue
                        nested_session_id = str(
                            session.get("session_id")
                            or session.get("conversation_id")
                            or f"{session_id}::{session_idx}"
                        )
                        nested_messages = session.get("dialogue") or session.get("messages") or session.get("turns") or []
                        texts = []
                        for msg in nested_messages:
                            if isinstance(msg, str):
                                texts.append(msg)
                            elif isinstance(msg, dict):
                                texts.append(str(msg.get("content") or msg.get("text") or msg.get("message") or ""))
                        emit(nested_session_id, texts)
                    continue
                if isinstance(item.get("messages"), list):
                    texts = []
                    for msg in item["messages"]:
                        if isinstance(msg, str):
                            texts.append(msg)
                        elif isinstance(msg, dict):
                            texts.append(str(msg.get("content") or msg.get("text") or msg.get("message") or ""))
                    emit(session_id, texts)
                elif isinstance(item.get("turns"), list):
                    texts = []
                    for turn in item["turns"]:
                        if isinstance(turn, str):
                            texts.append(turn)
                        elif isinstance(turn, dict):
                            texts.append(str(turn.get("content") or turn.get("text") or turn.get("message") or ""))
                    emit(session_id, texts)
                else:
                    text = str(item.get("text") or item.get("content") or item.get("message") or "")
                    if text.strip():
                        emit(session_id, [text])
        return chunks

    if isinstance(payload, dict):
        if isinstance(payload.get("sessions"), list):
            for idx, session in enumerate(payload["sessions"]):
                if isinstance(session, dict):
                    session_id = str(session.get("session_id") or session.get("id") or f"{source_name}::{idx}")
                    messages = session.get("messages") or session.get("turns") or session.get("items") or []
                    texts = []
                    for msg in messages:
                        if isinstance(msg, str):
                            texts.append(msg)
                        elif isinstance(msg, dict):
                            texts.append(str(msg.get("content") or msg.get("text") or msg.get("message") or ""))
                    emit(session_id, texts)
        elif isinstance(payload.get("messages"), list):
            session_id = str(payload.get("session_id") or payload.get("conversation_id") or source_name)
            texts = []
            for msg in payload["messages"]:
                if isinstance(msg, str):
                    texts.append(msg)
                elif isinstance(msg, dict):
                    texts.append(str(msg.get("content") or msg.get("text") or msg.get("message") or ""))
            emit(session_id, texts)
        elif isinstance(payload.get("items"), list):
            session_id = str(payload.get("session_id") or payload.get("conversation_id") or source_name)
            texts = []
            for item in payload["items"]:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(str(item.get("content") or item.get("text") or item.get("message") or ""))
            emit(session_id, texts)
        else:
            text = str(payload.get("text") or payload.get("content") or payload.get("message") or "")
            if text.strip():
                emit(str(payload.get("session_id") or payload.get("conversation_id") or source_name), [text])
    return chunks


def load_corpus(corpus_dir: str, verbose: bool = False) -> List[CorpusChunk]:
    files = list_corpus_files(corpus_dir)
    if not files:
        return []

    if verbose:
        log(f"Found {len(files)} corpus file(s) in {corpus_dir}")

    records: List[CorpusChunk] = []
    for path in files:
        source_name = path.relative_to(Path(corpus_dir)).as_posix()
        suffix = path.suffix.lower()
        if verbose:
            log(f"Loading {source_name} ...")
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            parts = chunk_text(text)
            for idx, part in enumerate(parts):
                before = "\n\n".join(parts[max(0, idx - 2):idx])
                after = "\n\n".join(parts[idx + 1:idx + 3])
                records.append(
                    CorpusChunk(
                        session_id=source_name,
                        source_path=source_name,
                        order=idx,
                        text=part,
                        context_before=before,
                        context_after=after,
                    )
                )
            if verbose:
                log(f"Loaded {len(parts)} chunk(s) from {source_name}")
            continue

        try:
            if suffix == ".jsonl":
                payload = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            else:
                payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        flattened = flatten_json_payload(payload, source_name)
        for idx, chunk in enumerate(flattened):
            records.append(
                CorpusChunk(
                    session_id=chunk.session_id or source_name,
                    source_path=source_name,
                    order=idx,
                    text=chunk.text,
                    context_before=chunk.context_before,
                    context_after=chunk.context_after,
                    state_hint=chunk.state_hint,
                )
            )
        if verbose:
            log(f"Loaded {len(flattened)} chunk(s) from {source_name}")

    # Normalize chunk order within sessions and fill surrounding context
    by_session: dict[str, List[CorpusChunk]] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)

    normalized: List[CorpusChunk] = []
    for session_id, chunks in by_session.items():
        for idx, chunk in enumerate(chunks):
            before = "\n\n".join(item.text for item in chunks[max(0, idx - 2):idx])
            after = "\n\n".join(item.text for item in chunks[idx + 1:idx + 3])
            normalized.append(
                CorpusChunk(
                    session_id=session_id,
                    source_path=chunk.source_path,
                    order=idx,
                    text=chunk.text,
                    context_before=before,
                    context_after=after,
                    state_hint=chunk.state_hint,
                )
            )

    if verbose:
        log(f"Corpus normalized: {len(normalized)} chunks across {len(by_session)} session(s)")

    return normalized


def build_state_projector(semantic_dim: int, state_dim: int = 7, seed: int = STATE_PROJECTION_SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    proj = rng.randn(semantic_dim, state_dim).astype(np.float32)
    proj /= np.sqrt(state_dim)
    return proj


def context_to_state(encoder: SentenceEncoder, projector: np.ndarray, context_text: str, fallback_text: str) -> np.ndarray:
    source_text = context_text.strip() or fallback_text.strip() or "neutral context"
    semantic = encoder.encode(source_text)
    raw_state = semantic @ projector
    raw_state = 1.0 / (1.0 + np.exp(-raw_state))
    return encoder.encode_state(raw_state.astype(np.float32))


def build_store(
    corpus: List[CorpusChunk],
    encoder: SentenceEncoder,
    verbose: bool = False,
    progress_every: int = 5000,
) -> MemoryStore:
    profile = MemoryProfile(max_size=max(1000, len(corpus) + 10))
    store = MemoryStore(profile=profile)
    projector = build_state_projector(encoder.semantic_dim)

    if verbose:
        log(f"Building MemoryStore for {len(corpus)} chunk(s)")

    for idx, chunk in enumerate(corpus, start=1):
        semantic = encoder.encode(chunk.text)
        state_proxy = context_to_state(encoder, projector, chunk.context_before, chunk.text)
        emotional_proxy = encoder.encode_emotional(state_proxy)
        mem = MemoryEntry(
            e_semantic=semantic,
            e_emotional=emotional_proxy,
            s_snapshot=state_proxy,
            timestamp=idx,
            text=chunk.text,
            tags=[chunk.session_id, chunk.source_path],
        )
        store.add(mem)
        if verbose and progress_every > 0 and idx % progress_every == 0:
            log(f"Encoded/stored {idx}/{len(corpus)} chunks")

    store.step = len(corpus)
    if verbose:
        log(f"MemoryStore ready: size={len(store)} step={store.step}")
    return store


def build_query_records(corpus: List[CorpusChunk], query_stride: int) -> List[CorpusChunk]:
    if not corpus:
        return []
    queries: List[CorpusChunk] = []
    by_session: dict[str, List[CorpusChunk]] = {}
    for chunk in corpus:
        by_session.setdefault(chunk.session_id, []).append(chunk)

    for session_id, chunks in by_session.items():
        if len(chunks) < 4:
            continue
        for idx in range(1, len(chunks) - 1, query_stride):
            chunk = chunks[idx]
            queries.append(chunk)
    return queries


def relevant_ids_for_query(store: MemoryStore, query_chunk: CorpusChunk) -> set[str]:
    return {mem.id for mem in store.get_all_safe() if query_chunk.session_id in mem.tags and mem.text != query_chunk.text}


def recall_at_k(retrieved_ids: List[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, mem_id in enumerate(retrieved_ids[:k], start=1):
        if mem_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: set[str]) -> float:
    for rank, mem_id in enumerate(retrieved_ids, start=1):
        if mem_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def jaccard_divergence(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    similarity = len(set_a & set_b) / len(union)
    return 1.0 - similarity


def extract_retrieved_ids(retrieved: list) -> List[str]:
    """Support both retrieval formats:
    - NCM full/cached: (distance, probability, MemoryEntry)
    - Baselines:       (distance, MemoryEntry)
    """
    ids: List[str] = []
    for item in retrieved:
        if not isinstance(item, tuple):
            continue
        if len(item) == 3:
            mem = item[2]
        elif len(item) == 2:
            mem = item[1]
        else:
            continue
        if hasattr(mem, "id"):
            ids.append(mem.id)
    return ids


def pick_two_states(encoder: SentenceEncoder, projector: np.ndarray, query: CorpusChunk) -> tuple[np.ndarray, np.ndarray]:
    state_a = context_to_state(encoder, projector, query.context_before, query.text)
    alt_context = query.context_after or query.context_before or query.text
    state_b = context_to_state(encoder, projector, alt_context, query.text)
    return state_a, state_b


def run_benchmark(
    corpus_dir: str,
    query_stride: int,
    k: int,
    verbose: bool = False,
    progress_every: int = 5000,
    eval_progress_every: int = 250,
    max_chunks: int = 0,
) -> dict:
    if verbose:
        log("Initializing encoder (this is where model weights load)")
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    if verbose:
        log("Encoder initialized")

    corpus = load_corpus(corpus_dir, verbose=verbose)
    if not corpus:
        return {
            "status": "skipped",
            "reason": f"No corpus files found in {corpus_dir}",
            "corpus_dir": corpus_dir,
        }

    if max_chunks > 0 and len(corpus) > max_chunks:
        corpus = corpus[:max_chunks]
        if verbose:
            log(f"Applied max_chunks={max_chunks}; using {len(corpus)} chunk(s)")

    store = build_store(corpus, encoder, verbose=verbose, progress_every=progress_every)
    projector = build_state_projector(encoder.semantic_dim)
    queries = build_query_records(corpus, query_stride=query_stride)
    if verbose:
        log(f"Built {len(queries)} query chunk(s) with query_stride={query_stride}")
    if not queries:
        return {
            "status": "skipped",
            "reason": "Corpus did not contain enough multi-chunk sessions for queries",
            "corpus_dir": corpus_dir,
            "num_chunks": len(corpus),
        }

    systems = ["semantic_only", "semantic_emotional", "ncm_full", "ncm_cached"]
    states = ["state_a", "state_b"]
    results = {
        "metadata": {
            "corpus_dir": corpus_dir,
            "num_chunks": len(corpus),
            "num_queries": len(queries),
            "query_stride": query_stride,
            "top_k": k,
            "timestamp": time.time(),
        },
        "systems": {},
    }

    for system_name in systems:
        if verbose:
            log(f"Evaluating system: {system_name}")
        results["systems"][system_name] = {"by_state": {}, "state_divergence": {}, "overall": {}}
        state_metrics = {}
        for state_name in states:
            if verbose:
                log(f"  State pass: {state_name}")
            recalls = []
            ndcgs = []
            mrrs = []
            retrieved_sets = []
            for query_idx, query in enumerate(queries, start=1):
                relevant = relevant_ids_for_query(store, query)
                q_sem = encoder.encode(query.text)
                state_vec = pick_two_states(encoder, projector, query)[0 if state_name == "state_a" else 1]
                q_emo = encoder.encode_emotional(state_vec)
                q_state = encoder.encode_state(state_vec)

                if system_name == "semantic_only":
                    retrieved = retrieve_semantic_only(q_sem, store, k=k)
                elif system_name == "semantic_emotional":
                    retrieved = retrieve_semantic_emotional(q_sem, q_emo, store, k=k)
                elif system_name == "ncm_full":
                    retrieved = retrieve_top_k(q_sem, q_emo, store, q_state, current_step=store.step, k=k)
                else:
                    retrieved = retrieve_top_k_fast(q_sem, q_emo, store, q_state, current_step=store.step, k=k)

                retrieved_ids = extract_retrieved_ids(retrieved)
                retrieved_sets.append(retrieved_ids)
                recalls.append(recall_at_k(retrieved_ids, relevant, k=k))
                ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k=k))
                mrrs.append(mrr(retrieved_ids, relevant))

                if verbose and eval_progress_every > 0 and query_idx % eval_progress_every == 0:
                    log(f"    {system_name}/{state_name}: processed {query_idx}/{len(queries)} queries")

            state_metrics[state_name] = {
                "recall@10": float(np.mean(recalls)),
                "ndcg@10": float(np.mean(ndcgs)),
                "mrr": float(np.mean(mrrs)),
                "retrieved_sets": retrieved_sets,
            }
            results["systems"][system_name]["by_state"][state_name] = {
                "recall@10": float(np.mean(recalls)),
                "ndcg@10": float(np.mean(ndcgs)),
                "mrr": float(np.mean(mrrs)),
            }

        state_a_sets = state_metrics["state_a"]["retrieved_sets"]
        state_b_sets = state_metrics["state_b"]["retrieved_sets"]
        divergences = [jaccard_divergence(a, b) for a, b in zip(state_a_sets, state_b_sets)]
        divergence_mean = float(np.mean(divergences))
        divergence_std = float(np.std(divergences))

        results["systems"][system_name]["state_divergence"] = {
            "jaccard_divergence_mean": divergence_mean,
            "jaccard_divergence_std": divergence_std,
        }
        results["systems"][system_name]["overall"] = {
            "avg_recall@10": float(np.mean([state_metrics[s]["recall@10"] for s in states])),
            "avg_ndcg@10": float(np.mean([state_metrics[s]["ndcg@10"] for s in states])),
            "avg_mrr": float(np.mean([state_metrics[s]["mrr"] for s in states])),
            "jaccard_divergence_mean": divergence_mean,
        }

    return results


def write_outputs(results: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp11_real_world_corpus_benchmark.json")
    txt_path = os.path.join(RESULTS_DIR, "exp11_real_world_corpus_benchmark.txt")
    png_path = os.path.join(RESULTS_DIR, "exp11_real_world_corpus_benchmark.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "EXP11: Real-World Corpus Benchmark",
        "=" * 35,
        f"Corpus dir: {results['metadata']['corpus_dir']}",
        f"Chunks: {results['metadata'].get('num_chunks', 0)}",
        f"Queries: {results['metadata'].get('num_queries', 0)}",
        "",
    ]
    for system_name, payload in results.get("systems", {}).items():
        overall = payload.get("overall", {})
        divergence = payload.get("state_divergence", {})
        lines.append(
            f"{system_name}: R@10={overall.get('avg_recall@10', 0.0):.3f}  NDCG={overall.get('avg_ndcg@10', 0.0):.3f}  MRR={overall.get('avg_mrr', 0.0):.3f}  JaccardΔ={divergence.get('jaccard_divergence_mean', 0.0):.3f}"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    system_names = list(results.get("systems", {}).keys())
    recalls = [results["systems"][name]["overall"]["avg_recall@10"] for name in system_names]
    divergences = [results["systems"][name]["overall"]["jaccard_divergence_mean"] for name in system_names]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(system_names))
    width = 0.38
    ax1.bar(x - width / 2, recalls, width, label="Recall@10", color="#4C78A8")
    ax1.set_ylabel("Recall@10")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names, rotation=20)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, divergences, width, label="Jaccard divergence", color="#F58518", alpha=0.8)
    ax2.set_ylabel("State divergence")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=2)
    ax1.set_title("EXP11 Real-World Corpus Benchmark")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n✓ Saved {json_path}")
    print(f"✓ Saved {txt_path}")
    print(f"✓ Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-world corpus benchmark.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR, help="Directory containing .txt/.md/.json corpus exports")
    parser.add_argument("--query-stride", type=int, default=2, help="Use every Nth chunk as a query")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval cutoff")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
    parser.add_argument("--progress-every", type=int, default=5000, help="Progress interval for corpus encoding/store build")
    parser.add_argument("--eval-progress-every", type=int, default=250, help="Progress interval for evaluation query loop")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap on number of chunks (0 = no cap)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_dir = args.corpus_dir

    if not os.path.isdir(corpus_dir):
        print(f"[SKIP] Corpus directory not found: {corpus_dir}")
        print("       Add local .txt/.md/.json exports there, or pass --corpus-dir.")
        return 0

    results = run_benchmark(
        corpus_dir=corpus_dir,
        query_stride=args.query_stride,
        k=args.top_k,
        verbose=args.verbose,
        progress_every=args.progress_every,
        eval_progress_every=args.eval_progress_every,
        max_chunks=args.max_chunks,
    )
    if results.get("status") == "skipped":
        print(f"[SKIP] {results['reason']}")
        return 0

    write_outputs(results)

    print("\nSUMMARY")
    print("-" * 70)
    for system_name, payload in results["systems"].items():
        overall = payload["overall"]
        divergence = payload["state_divergence"]
        print(
            f"{system_name:20} R@10={overall['avg_recall@10']:.3f}  NDCG={overall['avg_ndcg@10']:.3f}  MRR={overall['avg_mrr']:.3f}  JaccardΔ={divergence['jaccard_divergence_mean']:.3f}"
        )

    print("\nInterpretation:")
    print("- Higher recall means better same-session retrieval on real exported text.")
    print("- Higher Jaccard divergence means the same query shifts more across states.")
    print("- NCM should only claim a real-world win if divergence stays high on messy, unseen data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
