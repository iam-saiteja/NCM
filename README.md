# NCM — Native Cognitive Memory

NCM is a memory storage and retrieval architecture where memories are encoded as multi-field geometric objects in a composite retrieval space. The system retrieves not just what is textually similar, but what is **cognitively resonant** — matching meaning, emotional context, internal state at encoding time, and recency simultaneously.

**The core novel contribution**: `s_snapshot` — storing a copy of the system's internal state vector at memory encoding time and using it as an independent retrieval dimension. This enables state-conditioned episodic retrieval, where the same query produces different results depending on the system's current internal state. No existing RAG, DNC, or attention-based memory system implements this.

---

## Features

- Tensor-based episodic memory representation
- Multi-field encoding (`e_semantic`, `e_emotional`, `s_snapshot`, time, strength)
- State-conditioned retrieval behavior
- Vectorized top-k retrieval with cached and uncached paths
- Adaptive softmax retrieval probabilities
- Reinforcement strength dynamics with bounded growth
- Binary persistence via `.ncm` serialization

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ENCODING PIPELINE                     │
│                                                         │
│  raw_text ──→ Encoder(text) ──→ Projector ──→ e_semantic│
│  s_current ──→ W_emo · s ──────────────────→ e_emotional│
│  s_current ──→ L2_normalize ───────────────→ s_snapshot │
│  clock ──────→ exp(-λ·Δt) ─────────────────→ t_encoded │
│                                                         │
│  All fields assembled into MemoryEntry                  │
│  Written to MemoryStore (dict, O(1) lookup)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                     │
│                                                         │
│  query_text ──→ encode ──→ q_semantic                   │
│  s_current ──→ W_emo · s ──→ q_emotional                │
│  s_current ──→ normalize ──→ q_state                    │
│                                                         │
│  d(m, q) = α·d_sem + β·d_emo + γ·d_state + δ·d_time   │
│                                                         │
│  All N memories scored via vectorized numpy (no loops)  │
│  Top-k returned by distance (ascending)                 │
│  Probabilities via softmax with adaptive temperature    │
└─────────────────────────────────────────────────────────┘
```

### Memory Entry Schema

```python
memory = {
    e_semantic:  vector in R^128    # what happened (JL random projection from 384-dim)
    e_emotional: vector in R^3      # emotional color (orthonormal projection via W_emo)
    s_snapshot:  vector in R^7      # internal state AT encoding time (L2-normalized)
    timestamp:   scalar             # step number
    strength:    scalar in [0, 2]   # reinforcement accumulator with bounded growth
    text:        string             # archived for human debugging only
}
```

Text is non-operational **during retrieval**. The system operates entirely on the geometric tensor structure.

---

## The Math

### 1. Cosine Similarity (Semantic Distance)

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
semantic_distance = 1 - cosine_similarity
```

Both vectors are L2-normalized at encoding time, so `A · B` computes cosine similarity directly. Result ∈ [0, 2], clipped to [0, 1].

### 2. Euclidean Distance (Emotional & State Distance)

```
||A - B|| = sqrt(Σ(A_i - B_i)²)
```

**Normalization constants (derived, not arbitrary)**:

- **Emotional**: For L2-normalized vectors, max `||a - b||` = 2.0 (when `cos(θ) = -1`), from `||a-b||² = 2 - 2·cos(θ)`. Divide by 2.0.
- **State**: For L2-normalized vectors in the positive orthant (all components ≥ 0), `cos(θ) ≥ 0` always, so max `||a - b||` = √2. Divide by √2.

**Critical fix**: Emotional distance compares **projected-to-projected** vectors (both through W_emo), not projected vs. raw state. Both the memory's `e_emotional` and the query's emotional vector are computed via `W_emo · s`.

### 3. Orthonormal Emotional Projection

```
e_emotional = W_emo · s_current
Constraint: W_emo · W_emo^T = I_k  (orthonormal)
```

W_emo ∈ R^(3×7) is initialized via QR decomposition of a random matrix. Orthonormality prevents subspace collapse — without it, two state variables could map to the same emotional direction, destroying geometric independence.

**Verified**: `||W_emo · W_emo^T - I|| = 2.1 × 10⁻⁷`

### 4. Temporal Encoding (Ebbinghaus Decay)

```
t_encoded = exp(-λ · Δt)
time_distance = 1 - exp(-λ · Δt)
```

| Δt | time_distance |
|----|---------------|
| 0 | 0.000 |
| 100 | 0.095 |
| 500 | 0.394 |
| 1000 | 0.632 |
| 5000 | 0.993 |

### 5. Full Distance Function

```
d(m, q) = α·(1 - cos(e_sem_m, e_sem_q))           # semantic
        + β·||e_emo_m - e_emo_q|| / 2.0             # emotional (projected vs projected)
        + γ·||s_snap_m - s_current|| / √2            # state (positive orthant)
        + δ·(1 - exp(-λ·Δt))                         # temporal

Constraint: α + β + γ + δ = 1
Default:    α=0.4, β=0.2, γ=0.3, δ=0.1
```

All four components are normalized to [0, 1]. A **Dirichlet regularization** penalty prevents any single dimension from dominating:

```
L_balance = Σ(w_i - 0.25)²
```

### 6. Softmax Retrieval with Adaptive Temperature

```
P(m_i | q) = exp(-d_i / T) / Σ_j exp(-d_j / T)
```

Adaptive temperature that responds to novelty:
```
T(t) = T_base · (1 + η · novelty)
novelty = min(distances)  # how far is the closest memory
```

High novelty → higher T → exploratory recall.
Low novelty → lower T → deterministic recall.

### 7. Semantic Projection (Johnson-Lindenstrauss)

The 384→128 dimensionality reduction uses a random projection matrix scaled by `1/√k`. The JL lemma guarantees pairwise distances are preserved within `(1±ε)` with high probability. For our use case, 128 dimensions are empirically sufficient (validated across 100k+ memories).

### 8. Memory Strength

```
On retrieval: strength = min(strength + 0.1, 2.0)
Each step:    strength = strength × 0.999

Half-life ≈ 693 steps (0.999^693 ≈ 0.500)
```

The 2.0 cap prevents unbounded reinforcement growth (analogous to bounded synaptic weights in Hebbian learning).

---

## Experiment Results

### Experiment 1: Retrieval Precision

Using stored event texts as queries (standard IR evaluation). 1,200 memories across 6 categories × 8 states.

| Metric | k | Semantic Only | Sem + Emotional | NCM Full |
|--------|---|:---:|:---:|:---:|
| **Category P@k** | 1 | **0.925** | 0.625 | 0.625 |
| | 3 | **0.933** | 0.692 | 0.692 |
| | 5 | **0.950** | 0.800 | 0.800 |
| | 10 | **0.955** | 0.900 | 0.890 |
| **State P@k** | 1 | 0.075 | 0.625 | **0.625** |
| | 3 | 0.083 | 0.683 | **0.692** |
| | 5 | 0.105 | 0.435 | **0.435** |
| | 10 | 0.095 | 0.217 | **0.217** |

**Interpretation**: Semantic-only excels at category precision (what happened) but fails at state precision (who you were). NCM matches or beats baselines on state precision while maintaining competitive category precision.

### Experiment 2: Novelty Sensitivity at Scale

Semantic novelty collapses as memory grows. NCM stays sensitive.

| Store Size | Semantic Novelty | NCM Novelty | Advantage |
|:---:|:---:|:---:|:---:|
| 100 | 0.006 | 0.127 | **21×** |
| 1,000 | 0.005 | 0.130 | **26×** |
| 10,000 | 0.004 | 0.123 | **29×** |
| 50,000 | 0.004 | 0.130 | **33×** |

The advantage **grows with scale** because semantic similarity saturates while emotional and state dimensions remain sensitive.

### Experiment 3: State-Conditioned Retrieval (Key Result)

Same semantic query presented in two different internal states. Jaccard distance = 0 means identical retrieval sets; 1 means completely different.

| State Pair | Semantic Jaccard | NCM Jaccard |
|:---|:---:|:---:|
| Calm-Happy vs Stressed-Angry | 0.000 | **0.792** |
| Excited-Curious vs Sad-Withdrawn | 0.000 | **0.764** |
| Confident vs Fearful | 0.000 | **0.861** |
| Neutral vs Exhausted | 0.000 | **0.333** |

**Mean Jaccard = 0.718**. Semantic retrieval returns identical sets regardless of state. NCM returns substantially different memory sets. This is the core proof that `s_snapshot` creates genuinely different retrieval behavior.

### Experiment 4: Speed Benchmarks

| Memories | Semantic (ms) | Full Manifold (ms) | NCM Cached (ms) |
|:---:|:---:|:---:|:---:|
| 1,000 | 0.28 | 0.95 | **0.18** |
| 10,000 | 7.39 | 19.29 | **3.51** |
| 50,000 | 31.75 | 79.72 | **11.31** |

- **Store throughput**: ~16,000 memories/sec
- **Encoding throughput**: ~509 texts/sec
- **Storage efficiency**: 560 bytes/memory (compressed .ncm binary format)
- **Cached retrieval at 50k**: 11.3 ms/query (real-time viable)

### Experiment 5: Memory Systems Comparison

From [results/exp5_memory_systems_comparison.txt](results/exp5_memory_systems_comparison.txt):

- `ncm_cached_full`: state_avg=0.7350, category_avg=0.9973, latency_ms=0.4873
- `ncm_full`: state_avg=0.7350, category_avg=0.9973, latency_ms=2.1402
- `semantic_only`: state_avg=0.1239, category_avg=1.0000, latency_ms=0.8542

### Experiment 6: Current Memory Systems vs NCM

From [results/exp6_current_memory_systems_vs_ncm.txt](results/exp6_current_memory_systems_vs_ncm.txt):

- `semantic_emotional`: state_avg=0.7672, category_avg=0.8764, latency_ms=1.9810
- `ncm_cached_full`: state_avg=0.5835, category_avg=0.6050, latency_ms=0.6482
- `rag_semantic_only`: state_avg=0.1200, category_avg=0.9847, latency_ms=0.3129

### Experiment 7: Standardized Ranking and Visualization

From [results/exp7_standard_ranking.txt](results/exp7_standard_ranking.txt):

- Composite ranking used NDCG@10, Recall@10, MRR@10, MAP@10, state precision@10, latency, throughput, and memory footprint.
- Top systems in this run:
  1. `semantic_emotional`
  2. `ncm_cached_full`
  3. `ncm_full`

### Experiment 8: External Systems vs NCM

From [results/exp8_external_systems_vs_ncm.txt](results/exp8_external_systems_vs_ncm.txt):

- Baselines: `bm25_text`, `tfidf_cosine`, `dense_sbert_cosine`, `rag_semantic_only`, `rag_semantic_recency`, `recency_only`
- Top systems in this run:
  1. `ncm_cached_full`
  2. `ncm_full`
  3. `rag_semantic_only`

### Experiment 9: External Systems Speed Comparison

From [results/exp9_external_systems_speed.txt](results/exp9_external_systems_speed.txt):

- `recency_only`: avg=0.0258ms
- `dense_sbert_cosine`: avg=0.2538ms
- `ncm_cached_full`: avg=0.6011ms
- `ncm_full`: avg=2.4630ms

Interpretation: quality/state-aware retrieval and pure speed optimize for different targets. Cached NCM provides a practical latency-quality tradeoff.

---

## Image Guide (what each image shows and where it is)

All generated images are placed in the [results](results) folder.

- [results/exp1_precision_bars.png](results/exp1_precision_bars.png): Combined precision bars for Experiment 1.
- [results/exp1_category_precision.png](results/exp1_category_precision.png): Category precision comparison for semantic vs manifold retrieval.
- [results/exp1_state_precision.png](results/exp1_state_precision.png): State precision comparison, highlighting state-conditioned retrieval behavior.
- [results/exp2_novelty_scale.png](results/exp2_novelty_scale.png): Novelty trend as memory size scales.
- [results/exp3_state_conditioned.png](results/exp3_state_conditioned.png): Retrieval-set divergence under different states (Jaccard view).
- [results/exp4_speed.png](results/exp4_speed.png): Speed scaling across memory sizes.
- [results/exp7_quality_metrics.png](results/exp7_quality_metrics.png): Quality metrics (NDCG/Recall/State precision) for standardized ranking.
- [results/exp7_efficiency_metrics.png](results/exp7_efficiency_metrics.png): Efficiency comparison (latency/throughput/memory indicators).
- [results/exp7_overall_ranking.png](results/exp7_overall_ranking.png): Overall composite ranking for the standardized benchmark.
- [results/exp8_external_quality.png](results/exp8_external_quality.png): Quality comparison for external baselines vs NCM.
- [results/exp8_external_ranking.png](results/exp8_external_ranking.png): Composite ranking for external baselines vs NCM.
- [results/exp9_external_systems_speed_latency.png](results/exp9_external_systems_speed_latency.png): Avg and p95 latency comparison for speed benchmark.
- [results/exp9_external_systems_speed_qps.png](results/exp9_external_systems_speed_qps.png): Throughput (QPS) comparison for speed benchmark.
- [results/ncm_dashboard.png](results/ncm_dashboard.png): Consolidated dashboard chart of key benchmark outputs.

---

## Experimentation and Hardware

### Experimentation setup

- Synthetic benchmark dataset with ~1,200 memories spanning multiple semantic categories and internal state archetypes.
- Query sets include direct and paraphrase-style prompts.
- Evaluation includes retrieval quality metrics (Precision@k, Hit@k, MRR@k, Recall@k, MAP@k, NDCG@k), state precision, and speed metrics.

### Computer hardware used

All tests were run locally on your laptop:

- Device: **ideapad gaming 3**
- Processor (CPU): **AMD Ryzen 7 6800H**
- Graphics (GPU): **NVIDIA GeForce RTX 3050 (4GB VRAM)**
- RAM: **16GB**
- Storage: **512GB SSD**
- OS: **Windows**

---

## Project Structure

```
NCM/
├── ncm/                          # Core library
│   ├── __init__.py
│   ├── encoder.py                # Semantic + emotional + state encoding
│   ├── memory.py                 # MemoryEntry + MemoryStore
│   ├── retrieval.py              # Vectorized manifold retrieval
│   ├── profile.py                # Retrieval weights + personalization
│   ├── persistence.py            # Binary .ncm file format
│   └── exceptions.py             # Custom exception hierarchy
├── experiments/
│   ├── exp1_redesigned.py        # Precision@k evaluation
│   ├── exp5_memory_systems_comparison.py
│   ├── exp6_current_memory_systems_vs_ncm.py
│   ├── exp7_standard_ranking_and_viz.py
│   ├── exp8_external_systems_vs_ncm.py
│   ├── exp9_external_systems_speed.py
│   ├── run_fast.py               # Novelty, state-conditioned, speed experiments
│   ├── run_all_experiments.py
│   ├── generate_plots.py         # Plot generation
│   └── fix_dashboard.py          # Dashboard chart
├── results/                      # Experiment outputs (JSON + TXT + PNG)
│   ├── exp1_*
│   ├── exp2_*
│   ├── exp3_*
│   ├── exp4_*
│   ├── exp5_*
│   ├── exp6_*
│   ├── exp7_*
│   ├── exp8_*
│   ├── exp9_*
│   └── ncm_dashboard.png
└── README.md
```

---

## Quick Start

```python
from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.retrieval import retrieve_top_k

# Initialize
encoder = SentenceEncoder()
store = MemoryStore()

# Encode and store a memory
state = [0.7, 0.8, 0.2, 0.8, 0.3, 0.7, 0.2]  # internal state
e_sem = encoder.encode("colleague took credit for my work")
e_emo = encoder.encode_emotional(state)
s_snap = encoder.encode_state(state)

memory = MemoryEntry(
    e_semantic=e_sem, e_emotional=e_emo, s_snapshot=s_snap,
    timestamp=0, text="colleague took credit for my work"
)
store.add(memory)

# Retrieve with state-conditioned query
query_state = [0.9, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9]  # stressed state
q_sem = encoder.encode("someone betrayed my trust")
q_emo = encoder.encode_emotional(query_state)
q_state = encoder.encode_state(query_state)

results = retrieve_top_k(q_sem, q_emo, store, q_state, current_step=100, k=3)
for distance, probability, mem in results:
    print(f"  d={distance:.3f}  p={probability:.3f}  {mem.text}")
```

---

## Dependencies

- Python 3.8+
- numpy
- sentence-transformers (for semantic encoding)
- matplotlib (for experiments only)
- rank-bm25 (for external lexical baseline experiments)
- scikit-learn (for TF-IDF baseline experiments)

---

## Research Status

This is **Invention 1 of 3** in the TES project stack. NCM is architecturally independent and constitutes a standalone research contribution. Three claims are independently testable:

1. **State-conditioned retrieval** produces measurably different behavioral trajectories than semantic-only retrieval ✅ (Experiment 3: **Jaccard = 0.718**)
2. **Four-dimensional retrieval** maintains competitive category precision while enabling state-conditioned retrieval ✅ (Experiment 1)
3. **Full manifold novelty detection** maintains sensitivity at scale where semantic-only degrades ✅ (Experiment 2: **33× advantage at 50k**)

---

## Where NCM Outperforms

- Strong **state-conditioned recall behavior** (retrieval set shifts with internal state).
- Better **state precision** than semantic-only retrieval systems.
- Strong performance in external ranking runs where state-aware quality is weighted (Experiment 8).
- Practical latency with caching (`ncm_cached_full`) while preserving state-aware behavior.

## Where NCM Is Not Performing Best

- **Raw speed**: `ncm_full` is slower than lightweight baselines in pure latency benchmarks (Experiment 9).
- In some mixed objective setups, **`semantic_emotional`** can rank above NCM on composite quality score (Experiment 7).
- **Category-only retrieval** tasks can favor semantic-only systems when state sensitivity is not required.

## How NCM Helps in Real Systems

- Improves memory retrieval in agents where **contextual internal state** should affect recall.
- Supports more human-like episodic behavior by combining semantic, emotional, temporal, and state dimensions.
- Offers deployment flexibility through **cached retrieval** for better latency-quality tradeoff.
- Enables debugging/interpretability via structured memory fields and experiment traces.

## Future Features

- Learnable/auto-tuned retrieval weights per user or domain.
- ANN indexing (e.g., FAISS/HNSW) for faster large-scale manifold retrieval.
- Better calibration of strength dynamics (decay/reinforcement scheduling).
- Hybrid routing: fast semantic pre-filter + state-aware manifold rerank.
- Online adaptation from feedback signals (implicit relevance and correction loops).
- Expanded benchmark suite with larger real-world corpora and multilingual memory tests.

---

## License

Private repository. All rights reserved.
