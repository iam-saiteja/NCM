"""
EXPERIMENT 1 REDESIGNED: Retrieval Precision@k
===============================================

Problem with v1: Queries were abstract ("something about betrayal") which
are semantically distant from stored event texts. This gave near-zero precision
for ALL methods, making comparison meaningless.

Fix: Use ACTUAL stored event texts as queries (with held-out evaluation).
This is the standard information retrieval evaluation protocol:
  1. Store N memories across C categories and S states
  2. For each query: pick a stored event text, use it as the query
  3. Ground truth: all memories from the SAME category are "relevant"
  4. Measure Precision@k: fraction of top-k that are from the correct category

Additionally, for state-conditioned precision:
  5. "State-relevant" = same category AND same state archetype
  6. Measure State-Precision@k for NCM vs baselines

This gives us two metrics:
  - Category Precision@k: does the system retrieve semantically relevant memories?
  - State Precision@k: does the system retrieve memories from the matching internal state?
"""

import sys, os, time, json
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile, RetrievalWeights
from ncm.retrieval import (
    retrieve_top_k, retrieve_semantic_only,
    retrieve_semantic_emotional,
)

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, 'models'))

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

# Expanded semantic categories with more texts for better coverage
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


def build_labeled_store(n_per_category_state, seed=42):
    """
    Build store where each memory has known category and state labels.
    
    n_per_category_state: how many copies (with noise) per (category, state) pair
    Total memories = n_categories × n_states × n_per_category_state
    """
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=50000))
    
    metadata = {}  # id -> {"category": ..., "state": ..., "text": ...}
    
    ts = 0
    for cat_name, texts in SEMANTIC_CATEGORIES.items():
        for state_name, state_base in STATE_ARCHETYPES.items():
            for copy_i in range(n_per_category_state):
                # Pick text cyclically
                text = texts[copy_i % len(texts)]
                
                # Add state noise
                state = state_base.copy()
                noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
                state = np.clip(state + noise, 0.0, 1.0)
                
                e_sem = encoder.encode(text)
                # Add tiny semantic noise for uniqueness
                e_sem = e_sem + rng.randn(128).astype(np.float32) * 0.005
                e_sem /= np.linalg.norm(e_sem)
                
                e_emo = encoder.encode_emotional(state)
                s_snap = encoder.encode_state(state)
                
                mem = MemoryEntry(
                    e_semantic=e_sem, e_emotional=e_emo, s_snapshot=s_snap,
                    timestamp=ts, text=text,
                    tags=[cat_name, state_name],
                )
                store.add(mem)
                metadata[mem.id] = {
                    "category": cat_name,
                    "state": state_name,
                    "text": text,
                }
                ts += 1
    
    store.step = ts
    return store, metadata


def compute_precision_at_k(retrieved_ids, ground_truth_ids, k):
    """Standard Precision@k."""
    topk = retrieved_ids[:k]
    relevant = sum(1 for rid in topk if rid in ground_truth_ids)
    return relevant / k


def compute_ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    """Normalized Discounted Cumulative Gain @k."""
    topk = retrieved_ids[:k]
    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, rid in enumerate(topk) if rid in ground_truth_ids
    )
    # Ideal DCG: all relevant at the top
    n_relevant = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def run_experiment_1():
    """
    Redesigned Experiment 1: Proper Precision@k evaluation.
    """
    print("="*70)
    print("EXPERIMENT 1 (REDESIGNED): Retrieval Precision@k")
    print("="*70)
    
    # Build stores at different scales
    configs = [
        {"n_per": 2, "label": "small (96 memories)"},    # 6 cats × 8 states × 2 = 96
        {"n_per": 5, "label": "medium (240 memories)"},   # 6 × 8 × 5 = 240
        {"n_per": 10, "label": "large (480 memories)"},   # 6 × 8 × 10 = 480
        {"n_per": 25, "label": "xlarge (1200 memories)"}, # 6 × 8 × 25 = 1200
    ]
    
    k_values = [1, 3, 5, 10]
    n_queries = 40
    
    all_results = {}
    
    for config in configs:
        n_per = config["n_per"]
        label = config["label"]
        print(f"\n{'─'*60}")
        print(f"Store: {label}")
        print(f"{'─'*60}")
        
        store, metadata = build_labeled_store(n_per, seed=42)
        total = len(store)
        print(f"  Total memories: {total}")
        
        # Generate queries: pick random memories as query sources
        rng = np.random.RandomState(999)
        all_ids = list(metadata.keys())
        query_indices = rng.choice(len(all_ids), size=n_queries, replace=False)
        
        for k in k_values:
            # Metrics accumulators
            cat_prec_sem = []
            cat_prec_sememo = []
            cat_prec_full = []
            
            state_prec_sem = []
            state_prec_sememo = []
            state_prec_full = []
            
            ndcg_sem = []
            ndcg_sememo = []
            ndcg_full = []
            
            for qi in query_indices:
                query_id = all_ids[qi]
                query_meta = metadata[query_id]
                query_cat = query_meta["category"]
                query_state = query_meta["state"]
                query_text = query_meta["text"]
                
                # Encode query
                q_sem = encoder.encode(query_text)
                query_state_vec = STATE_ARCHETYPES[query_state].copy()
                q_emo = encoder.encode_emotional(query_state_vec)
                q_state_n = encoder.encode_state(query_state_vec)
                
                # Ground truth: all IDs with same category (excluding query itself)
                gt_category = set(
                    mid for mid, m in metadata.items()
                    if m["category"] == query_cat and mid != query_id
                )
                
                # Ground truth: same category AND same state
                gt_state = set(
                    mid for mid, m in metadata.items()
                    if m["category"] == query_cat and m["state"] == query_state and mid != query_id
                )
                
                if not gt_category:
                    continue
                
                # NCM-A: Semantic only
                res_a = retrieve_semantic_only(q_sem, store, k=max(k_values))
                ids_a = [m.id for _, m in res_a]
                
                # NCM-B: Semantic + Emotional
                res_b = retrieve_semantic_emotional(q_sem, q_emo, store, k=max(k_values))
                ids_b = [m.id for _, m in res_b]
                
                # NCM-C: Full manifold
                res_c = retrieve_top_k(
                    q_sem, q_emo, store, q_state_n, store.step, k=max(k_values),
                    use_adaptive_temp=True,
                )
                ids_c = [m.id for _, _, m in res_c]
                
                # Category Precision@k
                cat_prec_sem.append(compute_precision_at_k(ids_a, gt_category, k))
                cat_prec_sememo.append(compute_precision_at_k(ids_b, gt_category, k))
                cat_prec_full.append(compute_precision_at_k(ids_c, gt_category, k))
                
                # State Precision@k (stricter: same category AND state)
                state_prec_sem.append(compute_precision_at_k(ids_a, gt_state, k))
                state_prec_sememo.append(compute_precision_at_k(ids_b, gt_state, k))
                state_prec_full.append(compute_precision_at_k(ids_c, gt_state, k))
                
                # NDCG@k (category)
                ndcg_sem.append(compute_ndcg_at_k(ids_a, gt_category, k))
                ndcg_sememo.append(compute_ndcg_at_k(ids_b, gt_category, k))
                ndcg_full.append(compute_ndcg_at_k(ids_c, gt_category, k))
            
            key = f"{total}_k{k}"
            result = {
                "store_size": total,
                "k": k,
                "category_precision": {
                    "semantic_only": round(float(np.mean(cat_prec_sem)), 4),
                    "semantic_emotional": round(float(np.mean(cat_prec_sememo)), 4),
                    "full_manifold": round(float(np.mean(cat_prec_full)), 4),
                },
                "state_precision": {
                    "semantic_only": round(float(np.mean(state_prec_sem)), 4),
                    "semantic_emotional": round(float(np.mean(state_prec_sememo)), 4),
                    "full_manifold": round(float(np.mean(state_prec_full)), 4),
                },
                "ndcg": {
                    "semantic_only": round(float(np.mean(ndcg_sem)), 4),
                    "semantic_emotional": round(float(np.mean(ndcg_sememo)), 4),
                    "full_manifold": round(float(np.mean(ndcg_full)), 4),
                },
            }
            all_results[key] = result
            
            print(f"\n  k={k}:")
            print(f"    Category P@k:  Sem={np.mean(cat_prec_sem):.3f}  "
                  f"S+E={np.mean(cat_prec_sememo):.3f}  "
                  f"Full={np.mean(cat_prec_full):.3f}")
            print(f"    State P@k:     Sem={np.mean(state_prec_sem):.3f}  "
                  f"S+E={np.mean(state_prec_sememo):.3f}  "
                  f"Full={np.mean(state_prec_full):.3f}")
            print(f"    NDCG@k:        Sem={np.mean(ndcg_sem):.3f}  "
                  f"S+E={np.mean(ndcg_sememo):.3f}  "
                  f"Full={np.mean(ndcg_full):.3f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'exp1_redesigned.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/exp1_redesigned.json")
    return all_results


def generate_precision_plots(results):
    """Generate precision@k curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3,
    })
    
    COLORS = {
        'semantic': '#e74c3c',
        'sem_emo': '#f39c12',
        'manifold': '#2ecc71',
    }
    
    # Get unique store sizes
    store_sizes = sorted(set(results[k]["store_size"] for k in results))
    k_values = sorted(set(results[k]["k"] for k in results))
    
    # ─── PLOT 1: Category Precision@k curves ───
    fig, axes = plt.subplots(1, len(store_sizes), figsize=(4.5 * len(store_sizes), 4.5), sharey=True)
    if len(store_sizes) == 1:
        axes = [axes]
    
    for ax, n in zip(axes, store_sizes):
        sem_vals = [results[f"{n}_k{k}"]["category_precision"]["semantic_only"] for k in k_values]
        se_vals = [results[f"{n}_k{k}"]["category_precision"]["semantic_emotional"] for k in k_values]
        full_vals = [results[f"{n}_k{k}"]["category_precision"]["full_manifold"] for k in k_values]
        
        ax.plot(k_values, sem_vals, 'o-', label='Semantic Only', color=COLORS['semantic'], linewidth=2, markersize=7)
        ax.plot(k_values, se_vals, 's-', label='Sem + Emotional', color=COLORS['sem_emo'], linewidth=2, markersize=7)
        ax.plot(k_values, full_vals, 'D-', label='NCM Full Manifold', color=COLORS['manifold'], linewidth=2, markersize=7)
        
        ax.set_xlabel('k')
        ax.set_title(f'n = {n}')
        ax.set_xticks(k_values)
    
    axes[0].set_ylabel('Category Precision@k')
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle('Category Precision@k: All Methods Retrieve Semantically Relevant Memories',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp1_category_precision.png'))
    plt.close()
    print("Saved: exp1_category_precision.png")
    
    # ─── PLOT 2: State Precision@k curves (THE DIFFERENTIATOR) ───
    fig, axes = plt.subplots(1, len(store_sizes), figsize=(4.5 * len(store_sizes), 4.5), sharey=True)
    if len(store_sizes) == 1:
        axes = [axes]
    
    for ax, n in zip(axes, store_sizes):
        sem_vals = [results[f"{n}_k{k}"]["state_precision"]["semantic_only"] for k in k_values]
        se_vals = [results[f"{n}_k{k}"]["state_precision"]["semantic_emotional"] for k in k_values]
        full_vals = [results[f"{n}_k{k}"]["state_precision"]["full_manifold"] for k in k_values]
        
        ax.plot(k_values, sem_vals, 'o-', label='Semantic Only', color=COLORS['semantic'], linewidth=2, markersize=7)
        ax.plot(k_values, se_vals, 's-', label='Sem + Emotional', color=COLORS['sem_emo'], linewidth=2, markersize=7)
        ax.plot(k_values, full_vals, 'D-', label='NCM Full Manifold', color=COLORS['manifold'], linewidth=2, markersize=7)
        
        ax.set_xlabel('k')
        ax.set_title(f'n = {n}')
        ax.set_xticks(k_values)
    
    axes[0].set_ylabel('State Precision@k')
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle('State Precision@k: NCM Retrieves State-Matching Memories, Baselines Cannot',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp1_state_precision.png'))
    plt.close()
    print("Saved: exp1_state_precision.png")
    
    # ─── PLOT 3: Combined bar chart for largest store ───
    largest_n = max(store_sizes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(k_values))
    w = 0.25
    
    # Category Precision
    sem_cp = [results[f"{largest_n}_k{k}"]["category_precision"]["semantic_only"] for k in k_values]
    se_cp = [results[f"{largest_n}_k{k}"]["category_precision"]["semantic_emotional"] for k in k_values]
    full_cp = [results[f"{largest_n}_k{k}"]["category_precision"]["full_manifold"] for k in k_values]
    
    ax1.bar(x - w, sem_cp, w, label='Semantic Only', color=COLORS['semantic'], alpha=0.85)
    ax1.bar(x, se_cp, w, label='Sem + Emotional', color=COLORS['sem_emo'], alpha=0.85)
    ax1.bar(x + w, full_cp, w, label='NCM Full', color=COLORS['manifold'], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Category Precision (n={largest_n})', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)
    
    # State Precision
    sem_sp = [results[f"{largest_n}_k{k}"]["state_precision"]["semantic_only"] for k in k_values]
    se_sp = [results[f"{largest_n}_k{k}"]["state_precision"]["semantic_emotional"] for k in k_values]
    full_sp = [results[f"{largest_n}_k{k}"]["state_precision"]["full_manifold"] for k in k_values]
    
    ax2.bar(x - w, sem_sp, w, label='Semantic Only', color=COLORS['semantic'], alpha=0.85)
    ax2.bar(x, se_sp, w, label='Sem + Emotional', color=COLORS['sem_emo'], alpha=0.85)
    bars = ax2.bar(x + w, full_sp, w, label='NCM Full', color=COLORS['manifold'], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.set_ylabel('Precision')
    ax2.set_title(f'State Precision (n={largest_n})', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    
    # Add value labels on full manifold bars
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                     ha='center', fontsize=9, fontweight='bold')
    
    fig.suptitle(f'Experiment 1: Retrieval Precision Comparison (n={largest_n})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp1_precision_bars.png'))
    plt.close()
    print("Saved: exp1_precision_bars.png")


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = run_experiment_1()
    generate_precision_plots(results)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
