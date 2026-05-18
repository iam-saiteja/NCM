# NCM — Native Cognitive Memory

## Experimentally validated: Memory-Conditioned Behavior Shift

NCM is now experimentally validated to produce **different response behavior from the same base model** by changing only retrieved memory context (not model weights). This is demonstrated in:
- **Exp14 (real Ollama)**: measurable style/persona deltas under identical prompts.
- **Exp15 (large synthetic)**: stable persona-separation signal at scale (5k prompts, 5k memories/persona).

Latest validation added:
- **Exp16**: exact synthetic trajectory match, persistence round-trip, and retrieval trend preservation.
- **Exp17**: real-world corpus validation on 100 conversations / 2,009 stored turns with stable state evolution.
- **Exp18**: contradiction-aware retrieval validation showing deterministic corrected-fact dominance without deleting history.

NCM is a memory storage and retrieval architecture where memories are encoded as multi-field geometric objects in a composite retrieval space. Core stored fields include `e_semantic`, `e_emotional`, `s_snapshot`, `auto_state_snapshot`, `timestamp`, and `strength`. The system retrieves not just what is textually similar, but what is **cognitively resonant** — matching meaning, emotional context, internal state at encoding time, and recency simultaneously.

**Core retrieval contribution**: `s_snapshot` as an explicit retrieval dimension, now complemented by integrated `auto_state_snapshot` from `AutoStateTracker`. Together they enable state-conditioned episodic retrieval where the same semantic query can return different memories under different internal states, while preserving the existing manifold retrieval API.

## 🚀 Latest: 50-100x Performance Optimization (2026-04-10)

**All optimizations were designed to preserve mathematical correctness and retrieval accuracy where practical;** empirical validation for correctness and performance is provided in the experiments (see [CHANGELOG.md](CHANGELOG.md) and [experiments/EXPERIMENT_RESULTS.md](experiments/EXPERIMENT_RESULTS.md)).

### Performance Improvements
- **Batch Encoding**: 5-10x faster (GPU acceleration)
- **Distance Computation**: 15-50x faster (vectorization)
- **Memory Management**: 5-10x faster (upper triangle + SIMD)
- **Top-K Retrieval**: 2-5x faster (partition vs sort)
- **Corpus Loading**: 5-10x faster (batch encoding)
- **Experiment Runs**: 5-10x faster (query pre-encoding cache)
- **Aggregate**: **50-100x speedup** on typical benchmark workloads

-### Torch Runtime (CPU + GPU)
- NCM uses `sentence-transformers` (PyTorch backend when available) as the sentence-encoder runtime. The code supports CPU mode and a strict GPU-required option; when the encoder library is unavailable a deterministic hash-based fallback is used by the `SentenceEncoder` initialization.
- **CPU mode** is supported and documented (stable fallback path).
- **GPU mode** is supported and preferred for heavy workloads (batch encoding, corpus benchmarks).
- Exp11 is configured for **GPU-required** execution to avoid silent CPU fallback in long runs.

Install options:

```bash
# Standard install (CPU-compatible default)
pip install -r requirements.txt

# NVIDIA GPU (recommended for fastest runs)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

### Verification
```bash
python experiments/python/exp11_real_world_corpus_benchmark.py --max-chunks 50 --query-stride 4 --top-k 10
python experiments/python/exp12_weight_sensitivity.py --max-chunks 50 --query-stride 4 --top-k 10
python experiments/python/exp13_baseline_rematch.py --max-chunks 50 --query-stride 4 --top-k 10
```

### Project Documentation
- [CHANGELOG.md](CHANGELOG.md): what changed, commit-wise history
- [experiments/EXPERIMENT_RESULTS.md](experiments/EXPERIMENT_RESULTS.md): consolidated experiment table, visuals, and per-test links

### Project Layout (organized)
- Python experiment scripts: [experiments/python](experiments/python)
- Per-experiment outputs: [experiments/results](experiments/results)

## Runtime Defaults & How To Change Behavior

This project intentionally keeps several safety-first defaults in code while exposing knobs via `MemoryProfile` or function parameters. Below are the important defaults, where they live, and how to change them in your application.

- Contradiction-Aware Retrieval (CADP): Disabled by default. Defaults are implemented in `ncm/memory.py` and `ncm/retrieval.py`.
    - Default keys: `enable_contradiction_awareness=False`, `contradiction_penalty=0.0`, `contradiction_requires_marker=True`, `contradiction_similarity_threshold=0.85`.
    - To enable CADP and set the penalty (example):

```python
store.profile.set_custom('enable_contradiction_awareness', True)
store.profile.set_custom('contradiction_penalty', 0.20)
store.profile.set_custom('contradiction_requires_marker', True)
```

-- Per-memory `timestamp` vs store epoch: `MemoryEntry.timestamp` is preserved as provided by the creator code; `MemoryStore.add()` does NOT overwrite or auto-assign timestamps. Persistence writes a separate `store.step` in the file header. See `ncm/memory.py` (`MemoryStore.add`) and `ncm/persistence.py` (header write/read).
    - Example: experiment drivers set `timestamp` when creating `MemoryEntry` (see the pattern in [experiments/python/run_fast.py](experiments/python/run_fast.py#L60-L70)).
    - Best practice: set `memory.timestamp = store.step` when creating a memory if you want consistent store-level epoch semantics.

- Strength modulation clipping: hard-coded clip to `[0.5, 1.5]` in `ncm/retrieval.py` to avoid excessive amplification/suppression. To change, edit the clip bounds at the modulator location or make it configurable in your integration.

- Persistence defaults & truncation: `NCMFile.save()` defaults to `fp16=True`, sets autostate/contradiction flags, and enforces truncation limits (text ≤ 65535 bytes, max 255 tags, 255 bytes per tag). See `ncm/persistence.py` for details and tunables.
 - Persistence defaults & truncation: `NCMFile.save()` defaults to `compress=True, fp16=True`, sets presence flags for autostate and contradiction blocks, and enforces on-disk truncation limits. Key limits:
    - Text fields truncated to at most 65535 bytes on disk.
    - Tag lists limited to 255 entries; each tag is truncated to 255 bytes.
    - `contradicted_by` link bytes truncated to 65535 when written.
    - `store.step` is written into the file header (store-level epoch).
    See `ncm/persistence.py` for the exact implementation and tunables.

- Encoding gate (selective write): `novelty = 1 - max_cosine_similarity(query, memories)` and returns `1.0` for an empty store. Implementation: `ncm/encoder.py` (`encoding_gate`).
 - Encoding gate (selective write): two related implementations exist:
    1. **Semantic-only gate (encoder):** `SentenceEncoder.encoding_gate(query_vec, memory_vecs)` computes
        `novelty = clip(1.0 - max(cosine_sim(query, M)), 0.0, 1.0)` and returns `1.0` for an empty memory set. In short: the gate uses the *maximum cosine similarity* between the query and stored semantic vectors as the match score. See `ncm/encoder.py` for the exact code and docstring.
     2. **Semantic × State gate (store-level write check):** When `MemoryStore.add(..., gate_check=True)` is used, the implementation computes
         - `sem_sims = M_sem @ q_sem` and `sem_closeness = clip(sem_sims, 0.0, 1.0)`
         - `state_dists = ||M_state - s_query|| / sqrt(2)` and `state_closeness = 1.0 - clip(state_dists, 0.0, 1.0)`
         - `overlap = sem_closeness * state_closeness`
         - `novelty = 1.0 - max(overlap)` (or `1.0` when store empty)
         If `novelty < profile.write_threshold`, the memory is skipped. See `ncm/memory.py` (`MemoryStore.add`) for the exact implementation and rationale.

- W_emo orthonormality: `W_emo` is constructed by QR in `ncm/encoder.py`. The experiment suite records a small numeric error on the order of 1e-7 (see `experiments/results/run_all_experiments/math_verification.json`); treat this as an empirical measurement. To reproduce the check, run the validation script in `experiments/python` that computes `w_emo_orthonormality_error`.

 - Fast temporal approximation: opt-in `use_fast_temporal` parameter in retrieval functions. Default is `False`.
     - Approximation used: `exp(-decay_rate * Δt) ≈ 1 / (1 + decay_rate * Δt)` (faster to compute, small numeric error).
     - Tradeoff: faster retrieval at the cost of a slight temporal-term approximation error; enabled by passing `use_fast_temporal=True` to retrieval callers (e.g., `retrieve_top_k(..., use_fast_temporal=True)`).
     - See `ncm/retrieval.py` for the exact implementation and rationale.

If you want, I can also add short example snippets in `experiments/python` that demonstrate toggling each of these knobs; for now the README documents where to change them and how to enable them from user code.


## Features

- Tensor-based episodic memory representation
- Multi-field encoding (`e_semantic`, `e_emotional`, `s_snapshot`, `auto_state_snapshot`, time, strength)
- Optional contradiction metadata (`contradicted_by`, `is_conflict_trace`) for correction-aware recall (CADP). CADP is profile-configurable and conservative by default; see `ncm/memory.py` for keys like `contradiction_requires_marker` and `contradiction_similarity_threshold`.
- State-conditioned retrieval behavior
- Vectorized top-k retrieval with cached and uncached paths
- Adaptive softmax retrieval probabilities
- Reinforcement strength dynamics with bounded growth
- Binary persistence via `.ncm` serialization

### Implemented capabilities (documentation catch-up)

    - Selective write gate uses joint content+state novelty (`gate_check` + `write_threshold`) to skip true duplicates while keeping context-distinct episodes.
    - Implementation note: the semantic `encoding_gate` used by the selective write gate computes `novelty = 1 - max_cosine_similarity(query, memories)` and returns `1.0` for an empty store (see [ncm/encoder.py](ncm/encoder.py#L260-L268)). The gate is combined with state novelty to decide writes.
- Memory profiles are persisted inside `.ncm` files (dimensions, decay, temperature, thresholds, limits).
- `.ncm` format supports compression, optional FP16-on-disk vector storage, and compatibility-safe loading for legacy FP32 files.
 - Memory profiles are persisted inside `.ncm` files (dimensions, decay, temperature, thresholds, limits).
 - `.ncm` format supports compression, optional FP16-on-disk vector storage, and compatibility-safe loading for legacy FP32 files.
    - Persistence details: `NCMFile.save()` defaults to `fp16=True` and writes auto state and contradiction blocks when saving. The save path sets presence flags for autostate/contradiction and writes their blocks; loading will gracefully fall back to a neutral tracker if autostate parsing fails. On-disk limits: text fields are truncated to 65535 bytes, tag lists are limited to 255 entries, and each tag is truncated to 255 bytes. See the implementation for exact behavior and tunable flags in [ncm/persistence.py](ncm/persistence.py#L20-L140).
- Encoder runtime supports explicit device policy (`auto`/`cpu`/`cuda`) and strict GPU-required mode.
- Deterministic embedding fallback exists for environments where the sentence-transformer runtime is unavailable.
 - Encoder runtime supports explicit device policy (`auto`/`cpu`/`cuda`) and strict GPU-required mode. Device resolution: `auto` will prefer CUDA when `torch.cuda.is_available()` is True, otherwise `cpu`.
 - Deterministic embedding fallback exists for environments where the sentence-transformer runtime is unavailable. Implementation note: when `SentenceTransformer` fails to load the encoder falls back to a deterministic hash-based encoder for testing; if you set `SentenceEncoder(require_gpu=True)` and CUDA is unavailable initialization raises a `RuntimeError` instead of falling back. See `ncm/encoder.py` (`_ensure_initialized`, `_resolve_device`) for details.
- Memory lifecycle operations include reinforcement, decay, weakest-score eviction, and semantic consolidation.
- Tag-aware memory views are supported for scoped memory use cases.
- Explicit memory removal is supported for user-driven cleanup and moderation workflows.
- Profile metadata supports custom key/value fields for app-specific settings.
- Entropy-style recall confidence signals are available for uncertainty-aware behavior tuning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│             WRITE + AUTO-STATE PIPELINE                  │
│                                                         │
│  raw_text ──→ Encoder(text) ──→ Projector ──→ e_semantic│
│  raw_text ──→ AutoStateTracker.update() ─→ s_current(5D)│
│  s_current ──→ W_emo · s ──────────────────→ e_emotional│
│  s_current ──→ L2_normalize ───────────────→ s_snapshot │
│  s_current ────────────────────────────────→ auto_state_snapshot (5D)
│  clock ──────→ exp(-λ·Δt) ─────────────────→ t_encoded │
│                                                         │
│  All fields assembled into MemoryEntry                  │
│  Written to MemoryStore (dict, O(1) lookup)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL PIPELINE                     │
│                                                         │
│  query_text ──→ encode ──→ q_semantic                   │
│  store.auto_state.get_current_state() ───→ s_current(5D)│
│  s_current ──→ W_emo · s ────────────────→ q_emotional   │
│  s_current ──→ normalize ────────────────→ q_state       │
│  memory.auto_state_snapshot ─────────────→ d_state input │
│  memory.contradicted_by ────────────────→ d_contra input │
│                                                         │
│  d(m, q) = (1-λc)·(α·d_sem + β·d_emo + γ·d_state + δ·d_time) + λc·d_contra │
│                                                         │
│  All N memories scored via vectorized numpy (no loops)  │
│  Top-k returned by distance (ascending)                 │
│  Probabilities via softmax with adaptive temperature    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    PERSISTENCE PIPELINE                  │
│                                                         │
│  MemoryStore + profile + auto_state.to_dict()           │
│      └─→ NCMFile.save(..., FLAG_HAS_AUTOSTATE)          │
│      Note: per-memory `timestamp` values are provided by the caller when memories are created and preserved as-is; `NCMFile.save()` writes `store.step` into the file header as a separate store-level epoch and does not overwrite per-memory `timestamp` fields.
│                                                         │
│  NCMFile.load(...)                                       │
│      ├─ if auto_state exists: AutoStateTracker.from_dict│
│      └─ else: fresh neutral tracker [0.5 x 5]           │
└─────────────────────────────────────────────────────────┘
```

### Memory Entry Schema

```python
memory = {
    id: str,                      # UUID
    e_semantic:  vector in R^128    # what happened (JL random projection from 384-dim)
    e_emotional: vector in R^3      # emotional color (orthonormal projection via W_emo)
    s_snapshot:  vector in R^7      # encoder state snapshot (legacy/compat retrieval field)
    auto_state_snapshot: vector in R^5  # integrated auto-state at write time (val/aro/dom/cur/str). Note: when absent,
                                       # retrieval falls back to the normalized first-5 components of `s_snapshot`.
    contradicted_by: optional str      # points to newer correcting memory id
    is_conflict_trace: bool            # true for [UPDATE] trace memories
    timestamp:   scalar             # step number (assigned by the caller/creator). `MemoryStore.add()` does not auto-assign a timestamp; caller code or experiment runners must provide it.
    strength:    scalar (nominally in [0, 2])   # reinforcement accumulator; lifecycle ops cap growth at 2.0 but
                                              # constructor values are not validated on object creation.
                                              # Recommendation: initialize `strength` within `[0, 2]` or
                                              # use `MemoryStore.reinforce()` to adjust strengths; `NCMFile.load()` preserves stored strengths as-is.
    text:        string             # archived content (used for chat context and debugging)
    tags:        list[str]          # optional labels for scoped retrieval/filtering
}
```

Distance scoring is geometric (semantic/emotional/state/temporal). `text` is not used in distance math, but may be used by applications (e.g., chat context rendering).

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
# Constraint (implemented via QR): W_emo · W_emo^T ≈ I_k  (numerically orthonormal)
```

W_emo ∈ R^(3×7) is initialized via QR decomposition of a random matrix. Orthonormality prevents subspace collapse — without it, two state variables could map to the same emotional direction, destroying geometric independence.

**Note**: `W_emo` is constructed via QR decomposition and is numerically orthonormal up to floating-point precision. Empirical runs measured a small orthonormality error (on the order of 1e-7); see `experiments/results/run_all_experiments/math_verification.json` for the recorded value. Treat this as an experimental observation rather than a code invariant.

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

Note: the retrieval code supports an optional fast temporal approximation (`use_fast_temporal`) that
uses reciprocal-style approximations (e.g. `1/(1+x)`) for time contribution. This is an opt-in
performance/approximation tradeoff available in the retrieval API — see the implementation and
usage branches in [ncm/retrieval.py](ncm/retrieval.py#L63-L76) for details and tradeoffs.

### 5. Full Distance Function

```
d(m, q) = α·(1 - cos(e_sem_m, e_sem_q))           # semantic
        + β·||e_emo_m - e_emo_q|| / 2.0             # emotional (projected vs projected)
        + γ·||s_auto_m - s_current_auto|| / √2        # state (auto-state snapshot)
        + δ·(1 - exp(-λ·Δt))                         # temporal

Constraint: α + β + γ + δ = 1
Default:    α=0.4, β=0.2, γ=0.3, δ=0.1
```

All four components are normalized to [0, 1]. A **Dirichlet regularization** penalty prevents any single dimension from dominating:

In integrated mode, `s_auto_m` comes from per-memory `auto_state_snapshot` and `s_current_auto` comes from `store.auto_state.get_current_state()`. If a legacy memory lacks `auto_state_snapshot`, retrieval falls back to a compatible normalized state view.

```
L_balance = Σ(w_i - 0.25)²

Note: during retrieval a small "strength" modulation is applied to the final distance (reinforced memories are slightly easier to recall). The retrieval implementation clips the strength-derived modulator to the range [0.5, 1.5] before applying it to distances — see the implementation in [ncm/retrieval.py](ncm/retrieval.py#L120-L160) for the exact logic and rationale.
```

### 5.1 Contradiction-Aware Detection (CADP) — Correction Heuristics

When CADP is enabled via profile keys, the system detects candidate corrections using a conservative multi-step heuristic implemented in `ncm/memory.py`:

- Marker requirement (default): Corrections are only considered when the *new* text contains an explicit correction marker such as `correction`, `update`, `actually`, `instead`, `now`, or `revised`. This behavior is controlled by `contradiction_requires_marker` (default `True`).
- Subject extraction: the heuristic attempts to extract a short subject phrase from both old and new texts using regex patterns (e.g., `"my X is Y"`, `"the X is Y"`, or `"X is Y"`). If both subjects match, a lower semantic similarity threshold is allowed.

- Subject extraction: the heuristic attempts to extract a short subject phrase from both old and new texts using regex patterns and then compares the extracted subjects. The exact patterns used by `_extract_subject()` are implementation-specific (see `ncm/memory.py`) and currently are:

    - `r"\bmy\s+([a-z0-9][a-z0-9_\-\s]{1,40}?)\s+is\b"`
    - `r"\bthe\s+([a-z0-9][a-z0-9_\-\s]{1,40}?)\s+is\b"`
    - `r"\b([a-z0-9][a-z0-9_\-\s]{1,40}?)\s+is\b"`

    These patterns are intentionally simple and conservative (lowercase, limited length, allow alphanumerics, underscores, dashes, and spaces). If both extracted subjects match, the correction detector relaxes the semantic similarity requirement.

    Note: these regexes are heuristic and implementation-specific; adapt them in `ncm/memory.py` if your domain needs different subject extraction (e.g., named entities, longer phrases, or multilingual text).
- Semantic similarity: uses the dot product of L2-normalized semantic vectors as a cosine similarity score. Default similarity threshold is `contradiction_similarity_threshold=0.85`.
- Subject-aligned relax: when the extracted subjects match, the required similarity is relaxed to `min(threshold, 0.55)` to accept concise correction statements.
- Temporal and trace filters: only older memories with `timestamp < new.timestamp` and non-`is_conflict_trace` entries are considered for contradiction linking.
- On match: `old_memory.contradicted_by` is set to the new memory id. Optionally, a conflict-trace memory is written when `write_conflict_trace` is enabled in the profile; traces are created via `_make_conflict_trace()` and stored as `is_conflict_trace=True`.

These heuristics are deliberately conservative to avoid false positive corrections; tune the profile keys if you need more aggressive correction linking. See `ncm/memory.py` (`_is_correction_pair`, `_apply_contradiction_links`) for the exact implementation.

### 5.1 Contradiction-Aware Extension (CADP)

When contradiction-aware mode is enabled in `MemoryProfile.custom`, retrieval applies a correction penalty:

```
d_total(m, q) = (1 - λc)·d_base(m, q) + λc·I[m.contradicted_by != None]·g(q)

The contradiction penalty and query gate are configurable via `MemoryProfile.custom` (for example
`contradiction_penalty` and `contradiction_query_gate`). By default contradiction handling is
disabled (penalty = 0.0) unless explicitly enabled in the profile. When enabled, the runtime uses
conservative, configurable defaults exposed through the profile API (see `ncm/memory.py`): the
semantic similarity threshold is `0.85` by default (`contradiction_similarity_threshold`), correction
statements are expected to include an explicit marker by default (`contradiction_requires_marker = True`),
and automatic conflict-trace writing is opt-in (`write_conflict_trace = False`). The implementation also
permits subject-aligned corrections to pass with a lower threshold (implemented as `min(threshold, 0.55)`).
These are heuristics intended to make CADP safe by default; tune them via `MemoryProfile.set_custom()` for
different application needs.
```

This preserves old memories in storage while forcing corrected memories to dominate factual recall.
Write-time contradiction links are created only when the incoming text is correction-marked (e.g., `correction`, `update`, `actually`) and semantically/subject aligned with an older memory.

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
On explicit reinforcement (e.g. `store.reinforce(id, amount=0.1)`): strength = min(strength + amount, 2.0)
On consolidation (merge of highly-similar memories): absorbing memory strength increases by +0.05 (capped at 2.0)
Each `tick()` / step:    strength = strength × 0.999

Half-life ≈ 693 steps (0.999^693 ≈ 0.500)
```

Notes and links:
- Reinforcement is explicit: see `MemoryStore.reinforce()` implementation ([ncm/memory.py](ncm/memory.py#L381)).
- Periodic decay is implemented in `MemoryStore.decay_all()` and invoked from `tick()` ([ncm/memory.py](ncm/memory.py#L386)).
- Consolidation increases the stronger memory by +0.05 when merging similar memories ([ncm/memory.py](ncm/memory.py#L416)).
- Eviction uses a vectorized score combining `strength` and recency (exp(-0.001·age)); see `_evict_weakest()` ([ncm/memory.py](ncm/memory.py#L335)).
- Retrieval applies a small strength-derived modulation to final distances; the modulator is clipped to `[0.5, 1.5]` to avoid excessive amplification/suppression — see the exact implementation and rationale in [ncm/retrieval.py](ncm/retrieval.py#L136-L160).

The 2.0 cap prevents unbounded reinforcement growth (analogous to bounded synaptic weights in Hebbian learning). If you prefer different dynamics (e.g. faster reinforcement, slower decay), adjust `reinforce()`, `decay_all()`, or add a custom scheduler in your integration.

---

## Experiment Results

For detailed assessment of all experiment outputs (consolidated table, image-first summaries, and per-test result folders), visit [experiments/EXPERIMENT_RESULTS.md](experiments/EXPERIMENT_RESULTS.md).

## Auto-State Integration

### Design Overview

NCM’s auto-state module tracks a 5-dimensional affective state vector over time: valence, arousal, dominance, curiosity, and stress.
Each turn’s embedding is projected onto fixed positive/negative anchor pairs to produce a per-dimension familiarity signal `sigma_d in [0, 1]`, then integrated via an exponential moving average with dimension-specific learning rates `alpha_d = [0.15, 0.15, 0.15, 0.20, 0.25]`.
The resulting state `s_t` influences retrieval only through the existing manifold distance term `d_state` and an adaptive weighting scheme that increases state contribution when spread is high and de-emphasizes it near neutral states.

Reference: [experiments/results/exp16/exp16_auto_state_integration.txt](experiments/results/exp16/exp16_auto_state_integration.txt)

### EXP16 — Synthetic Validation (30-Turn Scripted Conversation)

EXP16 validates numerical correctness and retrieval impact of auto-state in a controlled 30-turn, three-era conversation (Stress -> Curiosity -> Positive Mixed).
The integrated implementation reproduces the locked simulation trajectory exactly: max absolute difference between expected and observed state at turns 10, 20, and 30 is `0.00e+00` on all five dimensions.
When auto-state is used as part of manifold distance, Precision@5 improves by `+0.400` in the stress-dominated era, with neutral effect (`+0.000`) in curiosity and mixed-positive eras, yielding an overall mean gain of `+0.133`.
A persistence stress test shows perfect `.ncm` round-trip: state components, adaptive weights, retrieval scores, and top-1 memory are identical before and after save/load.

Reference: [experiments/results/exp16/exp16_auto_state_integration.txt](experiments/results/exp16/exp16_auto_state_integration.txt)

### EXP17 — Real-World Scale (PersonaChat Sample)

EXP17 evaluates auto-state on real conversational data: 100 PersonaChat-style dialogues sampled from a corpus of 8,940 conversations, comprising 2,009 utterances stored and retrieved through NCM.
On this slice, both NCM (semantic + auto-state) and a semantic-only RAG baseline achieve saturated Precision@5 and Precision@10 of `1.000`, so auto-state does not change correctness but also does not regress it.
Retrieval latency remains production-ready: NCM with auto-state averages `~0.05 ms` per query versus `~0.02 ms` for the semantic-only baseline, an additional `~0.03 ms` that is negligible at human timescales.
Across 2,009 turns, mean state spread is `~0.0150` with range `[0.0022, 0.0473]`, and mean state entropy is `~1.7464`, indicating stable and informative 5D state behavior without collapse or saturation.

References:
- [experiments/results/exp17/exp17_real_world_scale.txt](experiments/results/exp17/exp17_real_world_scale.txt)
- [experiments/results/exp17/exp17_real_world_scale.json](experiments/results/exp17/exp17_real_world_scale.json)

### EXP18 — Contradiction-Aware Retrieval Validation (CADP)

EXP18 evaluates corrected-fact recall under contradiction-aware mode with optional conflict traces.
On synthetic correction tasks, baseline retrieval frequently returns stale facts first, while CADP consistently promotes the latest corrected memory:
- Single correction (`A -> B`): baseline `new>old` rate `0.08`, CADP `1.00`.
- Chain correction (`A -> B -> C`): baseline latest-top1 `0.00`, CADP latest-top1 `1.00`.
- Conflict traces: trace appears in top-3 for explicit update queries at rate `1.00`.
- Non-contradiction regression: top-1 unchanged ratio `1.00`.
- Persistence: contradiction links and conflict-trace flags round-trip through `.ncm` without loss.

References:
- [experiments/results/exp18/exp18_cadp_validation.txt](experiments/results/exp18/exp18_cadp_validation.txt)
- [experiments/results/exp18/exp18_cadp_validation.json](experiments/results/exp18/exp18_cadp_validation.json)

### Summary of Key Metrics

| Aspect | EXP16 (Synthetic) | EXP17 (Real-World) |
| --- | --- | --- |
| Dataset | 30-turn scripted 3-era conversation | 100 real conversations (PersonaChat), 2,009 utterances |
| Trajectory accuracy | Turn 10/20/30 max diff = 0.00e+00 | Not applicable (uses same tracker implementation validated in EXP16) |
| Retrieval impact (P@5 delta) | Era1 +0.400, Era2 +0.000, Era3 +0.000, mean +0.133 | P@5 = 1.000 for both NCM and semantic-only baseline (0.000 delta) |
| Persistence | State/score diffs = 0.00e+00; turn/alpha/weights/top-1 all OK | Validated via real .ncm round-trip in EXP16 and reused in EXP17 |
| State stability | Stress era peak followed by convergence to mixed state | Mean spread ~= 0.0150 (range [0.0022, 0.0473]); mean entropy ~= 1.7464 |
| Latency | Not primary focus (scripted sim) | NCM ~0.05 ms vs RAG ~0.02 ms per query |

Together, EXP16 and EXP17 show that auto-state is (1) numerically correct and well-behaved, (2) beneficial in emotionally skewed scenarios without harming performance elsewhere, and (3) robust at realistic scale with negligible latency overhead and stable behavior across diverse conversations.

### Quick Overview Table

| Group | Scope | Outcome |
|---|---|---|
| Exp1–Exp4 | Core behavior (precision, novelty, state shift, speed) | State-aware retrieval validated + practical cached latency |
| Exp5–Exp9 | System-level ranking and external comparisons | NCM remains competitive, especially when state-awareness is considered |
| Exp10–Exp13 | Recall rematch, real-world corpus, robustness, boundary analysis | NCM shows strong state divergence, robust defaults, and regime-dependent gains |
| Exp14 | Real Ollama persona-memory A/B test | Different memory profiles produce measurably different response style under identical prompts |
| Exp15 | Large synthetic persona-memory stress test | Memory conditioning signal remains strong at scale (5k prompts, 5k memories/persona) |
| Exp16 | Auto-state integration validation | Exact trajectory match + persistence round-trip on locked synthetic design |
| Exp17 | Real-world auto-state scale test | Stable auto-state behavior on 100 real conversations / 2,009 turns |
| Exp18 | Contradiction-aware retrieval validation | Corrected facts deterministically outrank contradicted facts while preserving history |

### Headline Metrics

| Signal | Snapshot |
|---|---|
| Canonical Exp1 protocol | Uses `exp1_redesigned.py` (stored event texts as queries, 1200-memory table) |
| State-conditioned recall shift | Exp3 mean Jaccard ≈ 0.714 for NCM vs ~0 for state-blind baseline |
| Large-scale novelty | Exp2 (AG News): semantic novelty trends to ~0 at 100k while full-manifold remains ~0.119 |
| Real-world corpus | Exp11 (bounded): NCM keeps strongest divergence (JaccardΔ≈0.374) with competitive NDCG/MRR |
| Speed snapshot | Exp4 (100k): semantic 49.346ms, full 10.131ms, cached 7.860ms/query |
| Weight robustness | Exp12: default remains near top-performing preset |
| Honest rematch | Exp13: NCM stronger in low/high shift buckets |
| Real-model persona effect | Exp14 (qwen2:7B): Persona-B warm markers +3.833 and +63 words vs Persona-A under same prompts |
| Large-scale persona effect | Exp15 (synthetic 5k×5k): persona separation L2=0.713, memory-gain positive-rate=1.000 |
| Synthetic validation lock | Exp16: Turn10/20/30 exact match, mean P@5 gain +13.3%, persistence PASS |
| Real-world scale proof | Exp17: 100 real conversations, 2,009 turns, stable mean spread 0.0150 |
| Contradiction resolution | Exp18: single-correction `0.08 -> 1.00`, chain latest-top1 `0.00 -> 1.00`, non-contradiction unchanged `1.00` |

### A few visuals

![State Conditioned Jaccard](experiments/results/exp3/exp3_state_conditioned.png)

State-aware behavior: same query, different internal state → different recalled memories in NCM.

![Real-World Corpus Benchmark](experiments/results/exp11/exp11_real_world_corpus_benchmark.png)

Real-data validation: NCM preserves strongest state-conditioned divergence.

![Baseline Rematch](experiments/results/exp13/exp13_baseline_rematch.png)

Boundary behavior: NCM is stronger at low/high shift regimes; middle regime is closer.

![Persona Memory Summary (Real Ollama)](experiments/results/exp14/exp14_persona_memory_ollama_summary.png)

Real-model behavior: same prompt set with different memory profiles yields measurable style shift.

![Synthetic Persona Memory Scale](experiments/results/exp15/exp15_synthetic_persona_memory_effect_scale.png)

Scale behavior: memory-conditioned persona effect remains strong in large synthetic runs.

![EXP16 State Trajectory](experiments/results/exp16/exp16_state_trajectory.png)

EXP16: locked synthetic trajectory and state evolution validation.

![EXP16 Retrieval Trend](experiments/results/exp16/exp16_retrieval_trend.png)

EXP16: retrieval trend delta across the three eras.

![EXP16 Persistence Validation](experiments/results/exp16/exp16_persistence_validation.png)

EXP16: `.ncm` persistence round-trip checks.

![EXP17 Retrieval Precision](experiments/results/exp17/exp17_scale_retrieval_precision.png)

EXP17: real-world retrieval precision on 100 conversations.

![EXP17 Performance Metrics](experiments/results/exp17/exp17_scale_performance_metrics.png)

EXP17: latency comparison between NCM and semantic baseline.

![EXP17 State Accuracy](experiments/results/exp17/exp17_scale_state_accuracy.png)

EXP17: state stability across diverse real conversations.

For full per-experiment explanations, result tables, and all plots, use [experiments/EXPERIMENT_RESULTS.md](experiments/EXPERIMENT_RESULTS.md).

---

## Experimentation and Hardware

### Experimentation setup

- Synthetic benchmark dataset with ~1,200 memories spanning multiple semantic categories and internal state archetypes.
- Real-world corpus benchmark using multi-session chat exports under `experiments/data/real_world_corpus`.
- Query sets include direct and paraphrase-style prompts.
- Evaluation includes retrieval quality metrics (Precision@k, Hit@k, MRR@k, Recall@k, MAP@k, NDCG@k), state precision, and speed metrics.

### Computer hardware used

All tests were run locally on your laptop:

- Device: **Lenovo IdeaPad Gaming 3**
- Processor (CPU): **AMD Ryzen 7 6800H**
- Graphics (GPU): **NVIDIA GeForce RTX 3050 (4GB VRAM)**
- RAM: **16GB**
- Storage: **512GB SSD**
- OS: **Windows 11**

### Runtime note on "GPU everywhere"

- Encoding-heavy stages are GPU-accelerated through Torch/SentenceTransformer.
- Core geometric retrieval math currently runs in vectorized NumPy on CPU.
- This mixed design keeps correctness stable while giving the largest practical speed gains on the expensive encoder path.

---

## Project Structure

```
NCM/
├── ncm/                          # Core library
│   ├── __init__.py
│   ├── auto_state.py             # 5D auto-state tracker (sigma + EMA + adaptive weights)
│   ├── encoder.py                # Semantic + emotional + state encoding
│   ├── memory.py                 # MemoryEntry + MemoryStore
│   ├── retrieval.py              # Vectorized manifold retrieval
│   ├── profile.py                # Retrieval weights + personalization
│   ├── persistence.py            # Binary .ncm file format
│   └── exceptions.py             # Custom exception hierarchy
├── experiments/
│   ├── data/
│   │   └── real_world_corpus/    # Real-world multi-session chat corpus (jsonl)
│   ├── python/                   # All experiment/runners Python files
│   │   ├── exp1_redesigned.py
│   │   ├── exp5_memory_systems_comparison.py
│   │   ├── exp6_current_memory_systems_vs_ncm.py
│   │   ├── exp7_standard_ranking_and_viz.py
│   │   ├── exp8_external_systems_vs_ncm.py
│   │   ├── exp9_external_systems_speed.py
│   │   ├── exp10_retrieval_recall_benchmark.py
│   │   ├── exp11_real_world_corpus_benchmark.py
│   │   ├── exp12_weight_sensitivity.py
│   │   ├── exp13_baseline_rematch.py
│   │   ├── exp14_persona_memory_ollama.py
│   │   ├── exp15_synthetic_persona_memory_effect.py
│   │   ├── exp16_auto_state_integration.py
│   │   ├── exp17_real_world_autostate_scale.py
│   │   ├── run_fast.py
│   │   └── run_all_experiments.py
│   └── results/                  # Per-experiment outputs (organized)
│       ├── exp1/
│       ├── exp2/
│       ├── exp3/
│       ├── exp4/
│       ├── exp5/
│       ├── exp6/
│       ├── exp7/
│       ├── exp8/
│       ├── exp9/
│       ├── exp10/
│       ├── exp11/
│       ├── exp12/
│       ├── exp13/
│       ├── exp14/
│       ├── exp15/
│       ├── exp16/
│       └── exp17/
├── models/
│   └── all-MiniLM-L6-v2/         # Pre-trained sentence transformer
└── README.md
```

---

## Quick Start

```python
from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.retrieval import retrieve_top_k
import numpy as np

# Initialize
encoder = SentenceEncoder()
store = MemoryStore()

# Optionally set current 5D auto-state (valence, arousal, dominance, curiosity, stress)
store.auto_state.state = np.array([0.7, 0.8, 0.2, 0.8, 0.3], dtype=np.float32)

# For encoder interfaces that expect 7D, pad 5D auto-state with neutral tail values
state5 = store.auto_state.get_current_state()
state7 = np.pad(state5, (0, 2), mode="constant", constant_values=0.5)

# Encode and store a memory
e_sem = encoder.encode("colleague took credit for my work")
e_emo = encoder.encode_emotional(state7)
s_snap = encoder.encode_state(state7)

memory = MemoryEntry(
    e_semantic=e_sem, e_emotional=e_emo, s_snapshot=s_snap,
    timestamp=0, text="colleague took credit for my work"
)
store.add(memory, update_auto_state=True)  # writes auto_state_snapshot automatically

# Retrieve with state-conditioned query
store.auto_state.state = np.array([0.9, 0.1, 0.9, 0.2, 0.8], dtype=np.float32)
query_state7 = np.pad(store.auto_state.get_current_state(), (0, 2), mode="constant", constant_values=0.5)
q_sem = encoder.encode("someone betrayed my trust")
q_emo = encoder.encode_emotional(query_state7)
q_state = encoder.encode_state(query_state7)  # kept for API compatibility

results = retrieve_top_k(q_sem, q_emo, store, q_state, current_step=100, k=3)
for distance, probability, mem in results:
    print(f"  d={distance:.3f}  p={probability:.3f}  {mem.text}")
```

---


### Advanced Capabilities

Below are examples of advanced NCM capabilities listed above:

```python
from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryStore
from ncm.persistence import NCMFile
from ncm.retrieval import retrieval_entropy

# 1. Device Policy & GPU-Required Mode
# Ensures the encoder runs on GPU, raising an error if unavailable
encoder = SentenceEncoder(device="cuda", require_gpu=True)

# 2. Tag-Aware Memory Views & Explicit Removal
store = MemoryStore()
# ... store memories with tags ...
# Filter memories by tag
work_memories = store.filter_by_tag("work")
# Remove a specific memory by ID
if work_memories:
    store.remove(work_memories[0].id)

# 3. Custom Profile Metadata
store.profile.set_custom("user_id", "user_12345")
store.profile.set_custom("mode", "analytical")
user = store.profile.get_custom("user_id")

# 4. Entropy-Style Recall Confidence
# Higher entropy means diffuse retrieval (uncertainty). Lower entropy means focused retrieval.
import numpy as np
distances = np.array([0.1, 0.8, 0.9]) # example distances from retrieve_top_k
entropy = retrieval_entropy(distances)

# 5. Compressed & FP16 Persistence
# Saves memory store with compression and FP16 vector precision for disk efficiency
NCMFile.save(store, "my_memory.ncm", compress=True, fp16=True)
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

This is **Invention 1 of 3** in the TES project stack. NCM is architecturally independent and constitutes a standalone research contribution. Three core claims are independently testable:

1. **State-conditioned retrieval** produces measurably different behavioral trajectories than semantic-only retrieval ✅ (Experiment 3: **Jaccard ≈ 0.714**)
2. **Four-component retrieval** maintains competitive category precision while enabling state-conditioned retrieval ✅ (Experiment 1)
3. **Full manifold novelty detection** remains non-zero at large scale where semantic novelty collapses ✅ (Experiment 2 AG News: semantic ≈ 0 at 100k, full ≈ 0.119)

**New in Experiment 10**: Recall@k across multiple internal states (LongMemEval-style benchmark). Tests whether NCM achieves state-dependent recall patterns while maintaining competitive recall scores vs semantic-only and SBERT baselines.

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

## Live Local Memory Proof (Ollama + NCM)

A real local run in the Ollama integration showed:

- A persisted `.ncm` memory file was written and then loaded again in a later chat session.
- A follow-up user question retrieved relevant earlier context from memory and used it in the response.

This confirms live behavior is working as intended: responses are influenced by relevant persisted memory, not only the immediate turn.

## Future Features

- Automated memory curation pipeline.
    - Retain high-signal memories to sustain retrieval quality over long-running sessions.
    - Advanced options: weighted/additive joint score or a learned gate for selective write decisions.
- Evaluate how each retrieval dimension influences chat behavior while developing stable persona patterns from conversation history.
- Learnable/auto-tuned retrieval weights per user or domain.
- ANN indexing (e.g., FAISS/HNSW) for faster large-scale manifold retrieval.
- Better calibration of strength dynamics (decay/reinforcement scheduling).
- Hybrid routing: fast semantic pre-filter + state-aware manifold rerank.
- Online adaptation from feedback signals (implicit relevance and correction loops).
- Expanded benchmark suite with larger real-world corpora and multilingual memory tests.

---

## License

Private repository. All rights reserved.
