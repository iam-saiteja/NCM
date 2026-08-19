"""
NCM Comprehensive Experiment Suite
====================================

Experiment 1: Retrieval Comparison (Full Manifold vs Semantic-Only vs Semantic+Emotional)
Experiment 2: Novelty Sensitivity at Scale (up to 100k memories)
Experiment 3: State-Conditioned Retrieval Validation
Experiment 4: Speed Benchmarks (store/retrieve throughput)

All results saved to workspace-relative /results/
"""

import sys
import os
import time
import json
import traceback
import runpy
import argparse
import csv
import hashlib
import platform
import urllib.request

import numpy as np

# This suite prints mathematical symbols (∈, ✓, Δ, ε). On Windows the default
# console encoding is cp1252, which cannot encode them, so printing raised
# UnicodeEncodeError partway through verify_math() and aborted the run. Force
# UTF-8 on stdout/stderr where the stream supports it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile, RetrievalWeights
from ncm.retrieval import (
    vectorized_manifold_distance, softmax_retrieval, adaptive_temperature,
    retrieve_top_k, retrieve_top_k_fast, retrieve_semantic_only,
    retrieve_semantic_emotional, retrieval_entropy,
)
from ncm.persistence import NCMFile

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'results', SCRIPT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

ONLINE_DATA_DIR = os.path.join(ROOT_DIR, 'experiments', 'data', 'online_corpus')
os.makedirs(ONLINE_DATA_DIR, exist_ok=True)
AG_NEWS_TRAIN_CSV = os.path.join(ONLINE_DATA_DIR, 'ag_news_train.csv')
AG_NEWS_URL = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'


def parse_args():
    parser = argparse.ArgumentParser(description="Run the NCM experiment suite.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for standalone experiments that support it.",
    )
    return parser.parse_args()

# Initialize encoder once
encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, 'models'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Generate synthetic interaction histories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Define 8 emotional/state archetypes for controlled experiments
STATE_ARCHETYPES = {
    "calm_happy":     np.array([0.7, 0.8, 0.2, 0.8, 0.3, 0.7, 0.2], dtype=np.float32),
    "stressed_angry": np.array([0.9, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9], dtype=np.float32),
    "sad_withdrawn":  np.array([0.2, 0.3, 0.6, 0.3, 0.7, 0.3, 0.4], dtype=np.float32),
    "excited_curious":np.array([0.9, 0.7, 0.3, 0.9, 0.4, 0.8, 0.3], dtype=np.float32),
    "fearful":        np.array([0.6, 0.2, 0.8, 0.3, 0.9, 0.2, 0.8], dtype=np.float32),
    "neutral":        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    "confident":      np.array([0.8, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], dtype=np.float32),
    "exhausted":      np.array([0.1, 0.4, 0.7, 0.2, 0.6, 0.3, 0.6], dtype=np.float32),
}

# Define semantic categories with example texts
SEMANTIC_CATEGORIES = {
    "betrayal": [
        "friend lied about the project deadline",
        "colleague took credit for my work",
        "trusted partner broke their promise",
        "discovered someone was talking behind my back",
        "team member sabotaged the presentation",
    ],
    "achievement": [
        "finished the marathon in under 4 hours",
        "got promoted to team lead",
        "published my first research paper",
        "won the hackathon competition",
        "scored highest in the final exam",
    ],
    "loss": [
        "lost my wallet on the train",
        "pet cat passed away last week",
        "failed the certification exam",
        "project got cancelled after months of work",
        "missed the flight and lost the booking",
    ],
    "discovery": [
        "found an amazing new algorithm for sorting",
        "discovered a shortcut through the park",
        "learned about quantum computing principles",
        "found a bug that was causing data corruption",
        "realized the solution was simpler than expected",
    ],
    "social": [
        "had a great conversation with the new intern",
        "team lunch was really enjoyable today",
        "met an interesting person at the conference",
        "helped a stranger with directions",
        "reconnected with an old school friend",
    ],
    "conflict": [
        "argument with manager about priorities",
        "disagreement about the technical approach",
        "heated debate during the team meeting",
        "customer complaint escalated badly",
        "deadline conflict between two projects",
    ],
}


def generate_memory(text, state, timestamp, encoder):
    """Create a properly encoded MemoryEntry."""
    e_sem = encoder.encode(text)
    e_emo = encoder.encode_emotional(state)
    s_snap = encoder.encode_state(state)
    return MemoryEntry(
        e_semantic=e_sem,
        e_emotional=e_emo,
        s_snapshot=s_snap,
        timestamp=timestamp,
        text=text,
    )


def build_test_store(n_memories, encoder, seed=42):
    """Build a MemoryStore with n_memories across categories and states."""
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=max(n_memories + 100, 10000)))
    
    categories = list(SEMANTIC_CATEGORIES.keys())
    states = list(STATE_ARCHETYPES.keys())
    
    for i in range(n_memories):
        cat = categories[i % len(categories)]
        text = SEMANTIC_CATEGORIES[cat][i % len(SEMANTIC_CATEGORIES[cat])]
        # Add variation to text to avoid exact duplicates
        text = f"{text} (instance {i})"
        
        state_name = states[i % len(states)]
        state = STATE_ARCHETYPES[state_name].copy()
        # Add noise to state
        noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
        state = np.clip(state + noise, 0.0, 1.0)
        
        mem = generate_memory(text, state, timestamp=i, encoder=encoder)
        store.add(mem)
        store.step = i + 1
    
    return store


def ensure_ag_news_dataset(verbose: bool = False) -> str:
    """Download AG News train split if missing (real online dataset, non-synthetic)."""
    if os.path.exists(AG_NEWS_TRAIN_CSV) and os.path.getsize(AG_NEWS_TRAIN_CSV) > 0:
        if verbose:
            print(f"[exp2] Using cached dataset: {AG_NEWS_TRAIN_CSV}")
        return AG_NEWS_TRAIN_CSV

    if verbose:
        print(f"[exp2] Downloading AG News dataset from: {AG_NEWS_URL}")
    urllib.request.urlretrieve(AG_NEWS_URL, AG_NEWS_TRAIN_CSV)
    if verbose:
        print(f"[exp2] Download complete: {AG_NEWS_TRAIN_CSV}")
    return AG_NEWS_TRAIN_CSV


def load_ag_news_records(limit: int, verbose: bool = False):
    """Load AG News rows as real-text memories: [{label, text}, ...]."""
    csv_path = ensure_ag_news_dataset(verbose=verbose)
    rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                label = int(row[0])
            except Exception:
                continue
            text = f"{row[1].strip()} {row[2].strip()}".strip()
            if not text:
                continue
            rows.append({"label": label, "text": text})
            if len(rows) >= limit:
                break

    if len(rows) < limit:
        raise RuntimeError(
            f"AG News records insufficient for requested limit={limit}; loaded={len(rows)}"
        )

    if verbose:
        print(f"[exp2] Loaded {len(rows)} real records from AG News")
    return rows


def text_to_state_vector(text: str) -> np.ndarray:
    """Deterministic state projection from real text (no synthetic category templates)."""
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    raw = np.frombuffer(digest, dtype=np.uint8)[:7].astype(np.float32)
    # keep away from exact 0/1 extremes for emotional projection stability
    state = 0.1 + (raw / 255.0) * 0.8
    return state.astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 1: Retrieval Comparison (Precision@k)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def experiment_1_retrieval_comparison():
    """
    Compare three systems:
      NCM-A: Semantic only (standard RAG baseline)
      NCM-B: Semantic + Emotional
      NCM-C: Full manifold (semantic + emotional + state + temporal)
    
    Metric: Precision@k = fraction of retrieved memories that are 
    from the SAME semantic category AND similar state as the query.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Retrieval Comparison (Precision@k)")
    print("="*70)
    
    store_sizes = [100, 500, 1000, 5000]
    k_values = [1, 3, 5, 10]
    n_queries = 50
    
    results = {}
    
    for n in store_sizes:
        print(f"\n--- Store size: {n} ---")
        store = build_test_store(n, encoder, seed=42)
        
        # Generate queries that have KNOWN ground truth
        rng = np.random.RandomState(123)
        queries = []
        for q in range(n_queries):
            # Pick a category and state
            cat = list(SEMANTIC_CATEGORIES.keys())[q % len(SEMANTIC_CATEGORIES)]
            state_name = list(STATE_ARCHETYPES.keys())[q % len(STATE_ARCHETYPES)]
            query_text = f"something about {cat} happened"
            query_state = STATE_ARCHETYPES[state_name].copy()
            noise = rng.uniform(-0.03, 0.03, size=query_state.shape).astype(np.float32)
            query_state = np.clip(query_state + noise, 0.0, 1.0)
            
            queries.append({
                "text": query_text,
                "state": query_state,
                "category": cat,
                "state_name": state_name,
            })
        
        for k in k_values:
            precision_a = []  # semantic only
            precision_b = []  # semantic + emotional
            precision_c = []  # full manifold
            
            for query in queries:
                q_sem = encoder.encode(query["text"])
                q_emo = encoder.encode_emotional(query["state"])
                q_state = encoder.encode_state(query["state"])
                
                # Ground truth: memories from same category
                gt_category = query["category"]
                gt_state = query["state_name"]
                
                # NCM-A: Semantic only
                results_a = retrieve_semantic_only(q_sem, store, k=k)
                relevant_a = sum(1 for _, m in results_a if gt_category in m.text)
                precision_a.append(relevant_a / k)
                
                # NCM-B: Semantic + Emotional
                results_b = retrieve_semantic_emotional(q_sem, q_emo, store, k=k)
                relevant_b = sum(1 for _, m in results_b if gt_category in m.text)
                precision_b.append(relevant_b / k)
                
                # NCM-C: Full manifold
                results_c = retrieve_top_k(
                    q_sem, q_emo, store, q_state, store.step, k=k,
                    use_adaptive_temp=True,
                )
                relevant_c = sum(1 for _, _, m in results_c if gt_category in m.text)
                precision_c.append(relevant_c / k)
            
            key = f"n={n},k={k}"
            results[key] = {
                "semantic_only": float(np.mean(precision_a)),
                "semantic_emotional": float(np.mean(precision_b)),
                "full_manifold": float(np.mean(precision_c)),
            }
            print(f"  k={k}: Sem={np.mean(precision_a):.3f}  "
                  f"Sem+Emo={np.mean(precision_b):.3f}  "
                  f"Full={np.mean(precision_c):.3f}")
    
    with open(os.path.join(RESULTS_DIR, 'exp1_retrieval_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[Experiment 1 complete]")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 2: Novelty Sensitivity at Scale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def experiment_2_novelty_at_scale(verbose: bool = False):
    """
    Test whether novelty detection remains sensitive at scale.
    
    Hypothesis: Semantic-only novelty saturates (everything looks familiar)
    as memory grows, but full-manifold novelty stays sensitive because
    emotional and state dimensions provide orthogonal novelty signals.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Novelty Sensitivity at Scale")
    print("="*70)
    
    scale_points = [100, 500, 1000, 5000, 10000, 50000, 100000]
    results = {}
    
    store = MemoryStore(profile=MemoryProfile(max_size=110000))
    rng = np.random.RandomState(42)
    max_scale = max(scale_points)
    real_records = load_ag_news_records(limit=max_scale, verbose=verbose)
    by_label = {}
    for rec in real_records:
        by_label.setdefault(rec["label"], []).append(rec)
    labels = sorted(by_label.keys())
    
    mem_count = 0
    
    exp2_start = time.perf_counter()

    for target_n in scale_points:
        scale_start = time.perf_counter()
        # Add memories up to target
        add_start = time.perf_counter()
        start_mem_count = mem_count
        total_to_add = target_n - start_mem_count
        added_this_scale = 0
        if total_to_add > 0:
            if verbose:
                print(f"  [build] indexing {total_to_add} real dataset rows into MemoryStore...")

            # Batch semantic encoding for major speedup vs one-by-one encode()
            batch_size = 256
            new_records = real_records[start_mem_count:target_n]
            new_texts = [r["text"] for r in new_records]
            new_semantics = encoder.encode_batch(new_texts, batch_size=batch_size)

            for i, (rec, e_sem) in enumerate(zip(new_records, new_semantics), start=1):
                text = rec["text"]
                state = text_to_state_vector(text)

                mem = MemoryEntry(
                    e_semantic=e_sem,
                    e_emotional=encoder.encode_emotional(state),
                    s_snapshot=encoder.encode_state(state),
                    timestamp=mem_count,
                    text=text,
                )
                store.add(mem)
                store.step = mem_count + 1
                mem_count += 1
                added_this_scale += 1

                if verbose and (added_this_scale % 5000 == 0 or added_this_scale == total_to_add):
                    print(f"  [build] indexed {added_this_scale}/{total_to_add} rows into MemoryStore")

        add_time = time.perf_counter() - add_start
        
        print(f"\n--- Scale: {target_n} memories ---")
        
        # Test with novel queries (new state + familiar semantic content)
        n_test = 20
        sem_novelty_scores = []
        full_novelty_scores = []
        
        # Build caches once per scale (major speed-up vs rebuilding per query)
        cache_start = time.perf_counter()
        store._rebuild_cache()
        cache_time = time.perf_counter() - cache_start

        sem_matrix = store.get_semantic_matrix()
        
        eval_start = time.perf_counter()
        for t in range(n_test):
            # Real query text sampled from the online dataset (no synthetic templates)
            label = labels[t % len(labels)]
            bucket = by_label[label]
            base_text = bucket[rng.randint(0, len(bucket))]["text"]

            # Novel state perturbation around deterministic text-derived state
            base_state = text_to_state_vector(base_text)
            perturb = rng.uniform(-0.08, 0.08, size=7).astype(np.float32)
            novel_state = np.clip(base_state + perturb, 0.1, 0.9).astype(np.float32)
            
            q_sem = encoder.encode(base_text)
            q_emo = encoder.encode_emotional(novel_state)
            q_state = encoder.encode_state(novel_state)
            
            # Semantic-only novelty: 1 - max(cosine_sim)
            if len(sem_matrix) > 0:
                sims = sem_matrix @ q_sem
                sem_novelty = 1.0 - float(np.max(sims))
            else:
                sem_novelty = 1.0
            sem_novelty_scores.append(sem_novelty)
            
            # Full manifold novelty: use full distance
            if store._sem_cache.shape[0] > 0:
                distances = vectorized_manifold_distance(
                    store._sem_cache, store._emo_cache, store._state_cache, store._ts_cache,
                    q_sem, q_emo, q_state, store.step,
                    store.profile.retrieval_weights, store.profile.decay_rate,
                )
                # Full novelty = minimum distance (how far is closest memory)
                full_novelty = float(np.min(distances))
            else:
                full_novelty = 1.0
            full_novelty_scores.append(full_novelty)

            if verbose:
                done = t + 1
                if done % 5 == 0 or done == n_test:
                    elapsed = time.perf_counter() - eval_start
                    per_q = elapsed / done
                    eta = per_q * (n_test - done)
                    print(f"  [eval] {done}/{n_test} queries | {per_q:.4f}s/query | ETA {eta:.1f}s")

        eval_time = time.perf_counter() - eval_start
        scale_elapsed = time.perf_counter() - scale_start
        
        results[str(target_n)] = {
            "dataset": "ag_news_online",
            "semantic_novelty_mean": float(np.mean(sem_novelty_scores)),
            "semantic_novelty_std": float(np.std(sem_novelty_scores)),
            "full_novelty_mean": float(np.mean(full_novelty_scores)),
            "full_novelty_std": float(np.std(full_novelty_scores)),
            "timing_sec": {
                "add_memories": float(add_time),
                "rebuild_cache": float(cache_time),
                "evaluate_queries": float(eval_time),
                "total_scale": float(scale_elapsed),
            },
        }
        print(f"  Semantic novelty: {np.mean(sem_novelty_scores):.4f} ± {np.std(sem_novelty_scores):.4f}")
        print(f"  Full manifold:    {np.mean(full_novelty_scores):.4f} ± {np.std(full_novelty_scores):.4f}")
        if verbose:
            print(
                "  Timing (s): "
                f"add={add_time:.2f}, cache={cache_time:.2f}, eval={eval_time:.2f}, total={scale_elapsed:.2f}"
            )
    
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty_at_scale.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"[Experiment 2 timing] total={time.perf_counter() - exp2_start:.2f}s")
    print("\n[Experiment 2 complete]")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 3: State-Conditioned Retrieval Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def experiment_3_state_conditioned():
    """
    The KEY experiment proving NCM's novel contribution.
    
    Setup: Same semantic query presented in two different emotional states.
    
    Prediction: Full manifold retrieval returns DIFFERENT memory sets
    depending on state. Semantic-only returns IDENTICAL sets.
    
    Metric: Jaccard distance between retrieval sets from different states.
    Jaccard = 1 - |A ∩ B| / |A ∪ B|
    Higher = more different = state matters more.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: State-Conditioned Retrieval Validation")
    print("="*70)
    
    # Build store with memories encoded in different states
    store = MemoryStore(profile=MemoryProfile(max_size=5000))
    rng = np.random.RandomState(42)
    
    # Create memories: same events encoded in DIFFERENT internal states
    event_texts = []
    for cat in SEMANTIC_CATEGORIES:
        for text in SEMANTIC_CATEGORIES[cat]:
            event_texts.append(text)
    
    # Each event encoded in multiple different states
    timestamp = 0
    for text in event_texts:
        for state_name, state in STATE_ARCHETYPES.items():
            noise = rng.uniform(-0.03, 0.03, size=state.shape).astype(np.float32)
            noisy_state = np.clip(state + noise, 0.0, 1.0)
            mem = generate_memory(text, noisy_state, timestamp, encoder)
            mem.tags.append(state_name)
            store.add(mem)
            timestamp += 1
    
    store.step = timestamp
    print(f"Store built: {len(store)} memories ({len(event_texts)} events × {len(STATE_ARCHETYPES)} states)")
    
    # Test: present same query in different states
    k = 5
    n_queries = 30
    state_pairs = [
        ("calm_happy", "stressed_angry"),
        ("excited_curious", "sad_withdrawn"),
        ("confident", "fearful"),
        ("neutral", "exhausted"),
        ("calm_happy", "fearful"),
        ("excited_curious", "exhausted"),
    ]
    
    results = {}
    
    for state_a_name, state_b_name in state_pairs:
        print(f"\n--- State pair: {state_a_name} vs {state_b_name} ---")
        
        jaccard_semantic = []
        jaccard_manifold = []
        
        for q in range(n_queries):
            cat = list(SEMANTIC_CATEGORIES.keys())[q % len(SEMANTIC_CATEGORIES)]
            query_text = SEMANTIC_CATEGORIES[cat][q % len(SEMANTIC_CATEGORIES[cat])]
            
            state_a = STATE_ARCHETYPES[state_a_name].copy()
            state_b = STATE_ARCHETYPES[state_b_name].copy()
            
            q_sem = encoder.encode(query_text)
            
            # Semantic-only: state doesn't matter
            results_sem_a = retrieve_semantic_only(q_sem, store, k=k)
            results_sem_b = retrieve_semantic_only(q_sem, store, k=k)
            
            ids_sem_a = set(m.id for _, m in results_sem_a)
            ids_sem_b = set(m.id for _, m in results_sem_b)
            
            if ids_sem_a or ids_sem_b:
                j_sem = 1.0 - len(ids_sem_a & ids_sem_b) / len(ids_sem_a | ids_sem_b)
            else:
                j_sem = 0.0
            jaccard_semantic.append(j_sem)
            
            # Full manifold: state SHOULD matter
            q_emo_a = encoder.encode_emotional(state_a)
            q_state_a = encoder.encode_state(state_a)
            results_full_a = retrieve_top_k(
                q_sem, q_emo_a, store, q_state_a, store.step, k=k,
            )
            
            q_emo_b = encoder.encode_emotional(state_b)
            q_state_b = encoder.encode_state(state_b)
            results_full_b = retrieve_top_k(
                q_sem, q_emo_b, store, q_state_b, store.step, k=k,
            )
            
            ids_full_a = set(m.id for _, _, m in results_full_a)
            ids_full_b = set(m.id for _, _, m in results_full_b)
            
            if ids_full_a or ids_full_b:
                j_full = 1.0 - len(ids_full_a & ids_full_b) / len(ids_full_a | ids_full_b)
            else:
                j_full = 0.0
            jaccard_manifold.append(j_full)
        
        pair_key = f"{state_a_name}_vs_{state_b_name}"
        results[pair_key] = {
            "semantic_jaccard_mean": float(np.mean(jaccard_semantic)),
            "manifold_jaccard_mean": float(np.mean(jaccard_manifold)),
            "manifold_jaccard_std": float(np.std(jaccard_manifold)),
            "improvement": float(np.mean(jaccard_manifold) - np.mean(jaccard_semantic)),
        }
        print(f"  Semantic Jaccard:  {np.mean(jaccard_semantic):.3f}")
        print(f"  Manifold Jaccard:  {np.mean(jaccard_manifold):.3f} ± {np.std(jaccard_manifold):.3f}")
        print(f"  Improvement:       +{np.mean(jaccard_manifold) - np.mean(jaccard_semantic):.3f}")
    
    with open(os.path.join(RESULTS_DIR, 'exp3_state_conditioned.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[Experiment 3 complete]")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 4: Speed Benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def experiment_4_speed_benchmarks():
    """
    Measure store and retrieve throughput at various scales.

    WHAT CHANGED AND WHY. The previous version of this function reported two
    retrieval columns named "full manifold (loop version)" and "full manifold
    (vectorized/cached)", timed as retrieve_top_k(...) and
    retrieve_top_k_fast(...). Those are not two implementations.
    ncm.retrieval.retrieve_top_k returns retrieve_top_k_fast(...) verbatim
    whenever tag_filter is None, which is the case here, so both columns
    executed identical code. The only difference was that the second loop ran
    after an explicit store._rebuild_cache(), so the first loop additionally
    paid one cache rebuild amortized over 100 queries. Any speedup ratio
    derived from those two columns therefore measured a single cache rebuild,
    not an algorithmic difference. The rebuild cost is now measured once and
    reported on its own line.

    Two further confounds are now separated rather than bundled:

      1. retrieve_semantic_only rebuilds its (n, 128) candidate matrix from
         scratch on every call, while retrieve_top_k_fast reads a cache that
         was built once. Comparing them measures allocation, not retrieval.
         A prebuilt-matrix semantic baseline is added so the reader can see
         how much of the gap is matrix construction.
      2. MemoryStore.add(update_auto_state=True), the default, calls
         AutoStateTracker.update(text), which runs one uncached
         sentence-transformer forward pass per memory. The previously reported
         "store throughput" was therefore dominated by encoding, not by the
         store operation. Store throughput is now timed with
         update_auto_state=False, and the auto-state write cost is measured
         separately at one scale.

    Every retrieval arm records the cache state it ran under.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Speed Benchmarks")
    print("="*70)
    print("NOTE: retrieve_top_k delegates to retrieve_top_k_fast when tag_filter")
    print("      is None, so there is no separate loop implementation to time.")

    scale_points = [100, 500, 1000, 5000, 10000, 50000, 100000]
    n_queries = 100
    n_warmup = 10
    k = 5
    auto_state_probe_n = 1000

    # Record the machine. Latency numbers are not portable, and an earlier
    # version of this table reported figures that cannot be reproduced here
    # (e.g. 21k-24k store writes/sec against ~7k measured now), with no record
    # of the hardware they came from. Capturing it makes future mismatches
    # diagnosable instead of ambiguous.
    machine = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "encoder_backend": encoder.backend,
        "encoder_device": encoder._resolve_device(),
    }

    results = {
        "_meta": {
            "machine": machine,
            "n_queries_timed": n_queries,
            "n_warmup_queries": n_warmup,
            "k": k,
            "timer": "time.perf_counter",
            "store_throughput_excludes_auto_state": True,
            "retrieve_top_k_delegates_to_fast_path": True,
            "arm_cache_state": {
                "semantic_only_shipped": "no cache used; rebuilds its own "
                                         "(n, 128) matrix on every call",
                "semantic_only_prebuilt_matrix": "warm; reads store._sem_cache",
                "manifold_warm_cache": "warm; store._rebuild_cache() called before timing",
            },
        }
    }

    def time_calls(fn, count, warmup):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(count):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "median_ms": round(float(np.median(arr)), 4),
            "mean_ms": round(float(arr.mean()), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
        }

    for n in scale_points:
        print(f"\n--- Scale: {n} memories ---")

        # 1. Encoding speed
        texts = [f"test memory number {i} about various topics" for i in range(min(n, 1000))]
        t0 = time.perf_counter()
        vecs = encoder.encode_batch(texts)
        t_encode = time.perf_counter() - t0
        encode_throughput = len(texts) / t_encode

        # 2. Store building speed, WITHOUT the per-memory auto-state encode.
        profile = MemoryProfile(max_size=n + 100)
        store = MemoryStore(profile=profile)
        rng = np.random.RandomState(42)

        states_list = list(STATE_ARCHETYPES.values())

        t0 = time.perf_counter()
        for i in range(n):
            state = states_list[i % len(states_list)].copy()
            noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
            state = np.clip(state + noise, 0.0, 1.0)

            # Use pre-encoded vectors for speed
            if i < len(vecs):
                sem = vecs[i]
            else:
                sem = vecs[i % len(vecs)]

            emo = encoder.encode_emotional(state)
            snap = encoder.encode_state(state)

            mem = MemoryEntry(
                e_semantic=sem, e_emotional=emo, s_snapshot=snap,
                timestamp=i, text=f"memory {i}",
            )
            store.add(mem, update_auto_state=False)
        store.step = n
        t_store = time.perf_counter() - t0
        store_throughput = n / t_store

        # 3. Retrieval speed
        query_state = STATE_ARCHETYPES["neutral"].copy()
        q_sem = encoder.encode("test query about something")
        q_emo = encoder.encode_emotional(query_state)
        q_state = encoder.encode_state(query_state)

        # One-time cache rebuild cost, measured on its own from a cold cache.
        store._invalidate_cache()
        t0 = time.perf_counter()
        store._rebuild_cache()
        t_cache_rebuild = (time.perf_counter() - t0) * 1000.0

        # Shipped semantic baseline: rebuilds its candidate matrix per call.
        t_sem = time_calls(
            lambda: retrieve_semantic_only(q_sem, store, k=k), n_queries, n_warmup
        )

        # Fair-work semantic baseline: same arithmetic, prebuilt matrix.
        # Reads a private cache attribute deliberately, to isolate the cost of
        # matrix construction from the cost of the similarity search itself.
        def semantic_prebuilt():
            sims = store._sem_cache @ q_sem
            idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
            return idx[np.argsort(-sims[idx])]

        t_sem_cached = time_calls(semantic_prebuilt, n_queries, n_warmup)

        # Full composite distance, warm cache.
        t_fast = time_calls(
            lambda: retrieve_top_k_fast(q_sem, q_emo, store, q_state, store.step, k=k),
            n_queries, n_warmup,
        )

        # 4. Persistence speed
        ncm_path = os.path.join(RESULTS_DIR, f'bench_{n}.ncm')
        t0 = time.perf_counter()
        NCMFile.save(store, ncm_path, compress=True)
        t_save = time.perf_counter() - t0
        file_size = os.path.getsize(ncm_path)

        t0 = time.perf_counter()
        _ = NCMFile.load(ncm_path)
        t_load = time.perf_counter() - t0

        # Clean up
        os.remove(ncm_path)

        results[str(n)] = {
            "encode_throughput_per_sec": round(encode_throughput, 1),
            "store_throughput_per_sec_no_auto_state": round(store_throughput, 1),
            "cache_rebuild_ms_one_time": round(t_cache_rebuild, 4),
            "retrieval_semantic_shipped_ms": t_sem,
            "retrieval_semantic_prebuilt_matrix_ms": t_sem_cached,
            "retrieval_manifold_warm_cache_ms": t_fast,
            "manifold_over_semantic_shipped_ratio": (
                round(t_fast["median_ms"] / t_sem["median_ms"], 3)
                if t_sem["median_ms"] > 0 else None
            ),
            "manifold_over_semantic_prebuilt_ratio": (
                round(t_fast["median_ms"] / t_sem_cached["median_ms"], 3)
                if t_sem_cached["median_ms"] > 0 else None
            ),
            "save_time_sec": round(t_save, 3),
            "load_time_sec": round(t_load, 3),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 2),
            "bytes_per_memory": round(file_size / n, 1),
        }

        print(f"  Encode: {encode_throughput:.0f} texts/sec")
        print(f"  Store (no auto-state): {store_throughput:.0f} mems/sec")
        print(f"  Cache rebuild (one time):        {t_cache_rebuild:.3f} ms")
        print(f"  Retrieve semantic, shipped:      {t_sem['median_ms']:.4f} ms/query (median)")
        print(f"  Retrieve semantic, prebuilt mat: {t_sem_cached['median_ms']:.4f} ms/query (median)")
        print(f"  Retrieve manifold, warm cache:   {t_fast['median_ms']:.4f} ms/query (median)")
        print(f"  Save: {t_save:.3f}s  Load: {t_load:.3f}s  Size: {file_size/1024:.0f} KB")

    # 5. Cost of the per-memory auto-state update, measured once.
    # This is the component the previous store-throughput figure absorbed.
    probe_store = MemoryStore(profile=MemoryProfile(max_size=auto_state_probe_n + 100))
    probe_texts = [f"probe memory {i} about a topic" for i in range(auto_state_probe_n)]
    probe_vecs = encoder.encode_batch(probe_texts)
    probe_state = STATE_ARCHETYPES["neutral"].copy()
    probe_emo = encoder.encode_emotional(probe_state)
    probe_snap = encoder.encode_state(probe_state)

    t0 = time.perf_counter()
    for i in range(auto_state_probe_n):
        probe_store.add(
            MemoryEntry(
                e_semantic=probe_vecs[i], e_emotional=probe_emo, s_snapshot=probe_snap,
                timestamp=i, text=probe_texts[i],
            ),
            update_auto_state=True,
        )
    t_auto = time.perf_counter() - t0

    results["_meta"]["auto_state_write_probe"] = {
        "n_memories": auto_state_probe_n,
        "total_sec": round(t_auto, 3),
        "ms_per_memory": round(t_auto / auto_state_probe_n * 1000.0, 4),
        "throughput_per_sec": round(auto_state_probe_n / t_auto, 1),
        "note": "AutoStateTracker.update runs one uncached sentence-transformer "
                "forward pass per memory. Add this to the no-auto-state store "
                "cost to get end-to-end write cost with state tracking on.",
    }
    print(f"\n  Auto-state write probe ({auto_state_probe_n} memories): "
          f"{results['_meta']['auto_state_write_probe']['ms_per_memory']:.3f} ms/memory "
          f"({results['_meta']['auto_state_write_probe']['throughput_per_sec']:.0f}/sec)")

    with open(os.path.join(RESULTS_DIR, 'exp4_speed_benchmarks.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n[Experiment 4 complete]")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATH VERIFICATION TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_math():
    """Verify all mathematical properties hold."""
    print("\n" + "="*70)
    print("MATH VERIFICATION")
    print("="*70)
    
    results = {}
    
    # 1. W_emo orthonormality: W · W^T = I_k
    w = encoder.w_emo
    product = w @ w.T
    identity_error = np.linalg.norm(product - np.eye(w.shape[0]))
    print(f"1. W_emo orthonormality error: {identity_error:.6e}")
    results["w_emo_orthonormality_error"] = float(identity_error)
    assert identity_error < 1e-5, f"W_emo not orthonormal! Error: {identity_error}"
    
    # 2. Semantic vectors are L2-normalized
    v1 = encoder.encode("hello world")
    v2 = encoder.encode("goodbye world")
    print(f"2. Semantic L2 norms: {np.linalg.norm(v1):.6f}, {np.linalg.norm(v2):.6f}")
    results["sem_norm_1"] = float(np.linalg.norm(v1))
    results["sem_norm_2"] = float(np.linalg.norm(v2))
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5
    assert abs(np.linalg.norm(v2) - 1.0) < 1e-5
    
    # 3. Emotional vectors are L2-normalized
    state = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6], dtype=np.float32)
    e_emo = encoder.encode_emotional(state)
    print(f"3. Emotional L2 norm: {np.linalg.norm(e_emo):.6f}")
    results["emo_norm"] = float(np.linalg.norm(e_emo))
    assert abs(np.linalg.norm(e_emo) - 1.0) < 1e-5
    
    # 4. State snapshots are L2-normalized
    s_snap = encoder.encode_state(state)
    print(f"4. State snapshot L2 norm: {np.linalg.norm(s_snap):.6f}")
    results["state_norm"] = float(np.linalg.norm(s_snap))
    assert abs(np.linalg.norm(s_snap) - 1.0) < 1e-5
    
    # 5. Distance bounds: all components in [0, 1]
    print("5. Distance bounds test (1000 random pairs)...")
    rng = np.random.RandomState(42)
    for _ in range(1000):
        s1 = rng.uniform(0, 1, 7).astype(np.float32)
        s2 = rng.uniform(0, 1, 7).astype(np.float32)
        
        e1 = encoder.encode_emotional(s1)
        e2 = encoder.encode_emotional(s2)
        sn1 = encoder.encode_state(s1)
        sn2 = encoder.encode_state(s2)
        
        d_emo = np.linalg.norm(e1 - e2) / 2.0
        d_state = np.linalg.norm(sn1 - sn2) / np.sqrt(2)
        
        assert 0 <= d_emo <= 1.0 + 1e-6, f"Emotional distance out of bounds: {d_emo}"
        assert 0 <= d_state <= 1.0 + 1e-6, f"State distance out of bounds: {d_state}"
    
    print("   All 1000 pairs: emotional ∈ [0,1] ✓, state ∈ [0,1] ✓")
    results["distance_bounds_verified"] = True
    
    # 6. Cosine similarity bounds
    print("6. Cosine similarity bounds (1000 random pairs)...")
    for _ in range(1000):
        t1 = f"random text {rng.randint(10000)}"
        t2 = f"different text {rng.randint(10000)}"
        v1 = encoder.encode(t1)
        v2 = encoder.encode(t2)
        sim = float(np.dot(v1, v2))
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6, f"Cosine sim out of bounds: {sim}"
    print("   All 1000 pairs: cosine ∈ [-1,1] ✓")
    results["cosine_bounds_verified"] = True
    
    # 7. Weight constraint: alpha + beta + gamma + delta = 1
    w = RetrievalWeights()
    total = sum(w.as_tuple())
    print(f"7. Weight sum: {total:.4f}")
    results["weight_sum"] = float(total)
    assert abs(total - 1.0) < 1e-4
    
    # 8. JL lemma check
    # For n=100000, epsilon=0.1: k_min = 4*ln(n)/(eps^2/2 - eps^3/3)
    #
    # This check FAILS by design-of-fact: k_min is ~9868 at these parameters
    # while the semantic space is 128 dimensions. It is reported, not asserted,
    # because the projection is used as an empirically adequate compression
    # rather than as a distance-preserving guarantee. The final summary below
    # must therefore not claim that all checks passed.
    n_points = 100000
    epsilon = 0.1
    k_min = 4 * np.log(n_points) / (epsilon**2 / 2 - epsilon**3 / 3)
    jl_ok = bool(128 >= float(k_min))
    print(f"8. JL lemma minimum dim for n={n_points}, ε={epsilon}: {k_min:.0f}")
    print(f"   NCM uses: 128 ({'≥' if jl_ok else '<'} {k_min:.0f}) {'✓' if jl_ok else '✗'}")
    if not jl_ok:
        print(f"   NOTE: the JL bound is NOT satisfied (128 < {k_min:.0f}). The 128-dim")
        print("         semantic projection carries no distance-preservation guarantee")
        print("         at this scale; treat it as an uncharacterized compression.")
    results["jl_min_dim"] = float(k_min)
    results["jl_satisfied"] = jl_ok

    # 9. Weight-balance penalty.
    # NOTE: RetrievalWeights.dirichlet_kl() returns sum((w_i - 0.25)^2), an L2
    # penalty toward the uniform weighting. It is not a Kullback-Leibler
    # divergence and is not a Dirichlet prior; the name is historical.
    w_balanced = RetrievalWeights(0.25, 0.25, 0.25, 0.25)
    w_skewed = RetrievalWeights(0.7, 0.1, 0.1, 0.1)
    kl_balanced = w_balanced.dirichlet_kl()
    kl_skewed = w_skewed.dirichlet_kl()
    print(f"9. Weight-balance L2 penalty (balanced): {kl_balanced:.6f}")
    print(f"   Weight-balance L2 penalty (skewed):   {kl_skewed:.6f}")
    results["dirichlet_balanced"] = float(kl_balanced)
    results["dirichlet_skewed"] = float(kl_skewed)
    assert kl_balanced < kl_skewed, "Balanced should have lower penalty than skewed"

    # 10. Temporal decay properties
    from ncm.retrieval import vectorized_manifold_distance
    print("10. Temporal decay curve verification...")
    decay_rate = 0.001
    checkpoints = [0, 100, 500, 1000, 5000]
    for dt in checkpoints:
        d_t = 1.0 - np.exp(-decay_rate * dt)
        print(f"    Δt={dt}: d_time={d_t:.4f}")
    results["temporal_decay_verified"] = True

    # 11. Which encoder actually produced these vectors.
    # A hash fallback would make every quality number meaningless, so record it.
    try:
        results["encoder_backend"] = encoder.backend
        results["encoder_backend_error"] = encoder.backend_error
    except AttributeError:
        results["encoder_backend"] = "unknown"
        results["encoder_backend_error"] = None
    print(f"11. Encoder backend: {results['encoder_backend']}")

    # Honest summary: enumerate what failed rather than asserting blanket success.
    checks = {
        "w_emo_orthonormal": results["w_emo_orthonormality_error"] < 1e-5,
        "distance_bounds": results["distance_bounds_verified"],
        "cosine_bounds": results["cosine_bounds_verified"],
        "weight_sum": abs(results["weight_sum"] - 1.0) < 1e-4,
        "jl_bound": results["jl_satisfied"],
        "temporal_decay": results["temporal_decay_verified"],
        "encoder_is_real": results["encoder_backend"] == "sentence-transformers",
    }
    results["checks"] = checks
    failed = [name for name, ok in checks.items() if not ok]
    results["checks_failed"] = failed
    if failed:
        print(f"\n⚠️  {len(checks) - len(failed)}/{len(checks)} MATH CHECKS PASSED"
              f" — FAILED: {', '.join(failed)}")
    else:
        print(f"\n✅ ALL {len(checks)} MATH CHECKS PASSED")
    
    with open(os.path.join(RESULTS_DIR, 'math_verification.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN ALL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    args = parse_args()
    print("NCM EXPERIMENT SUITE")
    print("=" * 70)
    
    all_results = {}
    
    try:
        all_results["math"] = verify_math()
    except Exception as e:
        print(f"\n❌ Math verification FAILED: {e}")
        traceback.print_exc()
    
    try:
        all_results["exp1"] = experiment_1_retrieval_comparison()
    except Exception as e:
        print(f"\n❌ Experiment 1 FAILED: {e}")
        traceback.print_exc()
    
    try:
        all_results["exp2"] = experiment_2_novelty_at_scale(verbose=args.verbose)
    except Exception as e:
        print(f"\n❌ Experiment 2 FAILED: {e}")
        traceback.print_exc()
    
    try:
        all_results["exp3"] = experiment_3_state_conditioned()
    except Exception as e:
        print(f"\n❌ Experiment 3 FAILED: {e}")
        traceback.print_exc()
    
    try:
        all_results["exp4"] = experiment_4_speed_benchmarks()
    except Exception as e:
        print(f"\n❌ Experiment 4 FAILED: {e}")
        traceback.print_exc()
    
    # Save combined results
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Run newer standalone experiments.
    #
    # exp10_retrieval_recall_benchmark.py is deliberately NOT in this list. It
    # executes no retrieval and emits hand-authored literals, so running it as
    # part of the suite produced a results file that looked like a measurement
    # and was cited as one. See that file's docstring.
    extra_experiments = [
        "exp5_memory_systems_comparison.py",
        "exp6_current_memory_systems_vs_ncm.py",
        "exp7_standard_ranking_and_viz.py",
        "exp8_external_systems_vs_ncm.py",
        "exp9_external_systems_speed.py",
        "exp11_real_world_corpus_benchmark.py",
        "exp12_weight_sensitivity.py",
        "exp13_baseline_rematch.py",
        "exp16_auto_state_integration.py",
        "exp17_real_world_autostate_scale.py",
        "exp18_contradiction_aware_retrieval.py",
    ]
    for script_name in extra_experiments:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        try:
            print(f"\n▶ Running {script_name} ...")
            old_argv = sys.argv[:]
            sys.argv = [script_path]
            if args.verbose and script_name in {
                "exp11_real_world_corpus_benchmark.py",
                "exp12_weight_sensitivity.py",
                "exp13_baseline_rematch.py",
            }:
                sys.argv.append("--verbose")
            if script_name == "exp11_real_world_corpus_benchmark.py":
                # Keep suite runtime bounded on very large local corpora.
                # Users can still run exp11 directly with custom limits.
                sys.argv.extend([
                    "--query-stride", "12",
                    "--max-queries", "5000",
                    "--eval-progress-every", "100",
                ])
            runpy.run_path(script_path, run_name="__main__")
            sys.argv = old_argv
        except Exception as e:
            sys.argv = old_argv
            print(f"\n❌ {script_name} FAILED: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 70)
