# Experiment Results (Detailed Assessment)

This is the full experiment report for NCM.

- Scripts: [experiments/python](python)
- Outputs (organized per test): [experiments/results](results)
- Short version: [README.md](../README.md)

---

## Quick Overview Table

| Experiment | What it tests | Why needed | Key result |
|---|---|---|---|
| Exp1 | Category vs state precision | Show baseline semantic strength vs NCM state strength | Semantic wins category; NCM wins state precision |
| Exp2 | Novelty sensitivity at scale | Check if novelty collapses as memory grows | Semantic novelty collapses by 100k; full-manifold remains non-zero |
| Exp3 | State-conditioned retrieval | Validate core claim (`s_snapshot`) | Same query returns different sets in NCM, not in baseline |
| Exp4 | Speed scaling | State the latency cost of composite retrieval | Composite costs 2.2x-4.9x a fair semantic baseline; under 13 ms at 100k |
| Exp5 | Internal memory comparison | Compare NCM modes and simple baselines | NCM cached gives best quality-latency tradeoff |
| Exp6 | Current system rematch | Head-to-head with stronger semantic-emotional baseline | NCM remains competitive with stronger state behavior |
| Exp7 | Standardized ranking | Multi-metric ranking under common scoring | NCM variants stay top-tier across balanced scoring |
| Exp8 | External systems quality | Compare BM25/TF-IDF/dense/RAG style baselines | NCM ranks strongly when state-awareness matters |
| Exp9 | External systems speed | Pure latency/QPS benchmark | NCM cached slower than trivial baselines but practical |
| Exp10 | WITHDRAWN | Was a hand-authored mock-up, not a measurement | No result; numbers were typed by hand, see Experiment 10 section |
| Exp11 | Real-world corpus benchmark | Validate beyond synthetic data | NCM keeps strongest state-divergence on real data |
| Exp12 | Weight sensitivity | Test robustness of default weights | Defaults stay near top; no fragile tuning point |
| Exp13 | Honest baseline rematch | Find boundary conditions | NCM better at low/high-shift regimes |
| Exp14 | Real Ollama persona-memory A/B | Test real-model style shift from memory context | Different memory profiles produce measurable response-style deltas |
| Exp15 | Synthetic persona-memory stress test | Validate memory-conditioning effect at scale | Strong persona separation persists on 5k prompts / 5k memories/persona |
| Exp16 | Auto-state integration validation | Verify locked design constants and persistence on integrated code | Exact trajectory match and exact persistence round-trip; no retrieval gain over semantic-only (+0.0000 P@5) |
| Exp17 | Same-session episodic retrieval on Multi-Session Chat | Test the composite manifold against a relevance label supplied by the corpus | Inferred-state NCM 0.4561 P@5 vs semantic-only 0.4640, so `-0.0079`; both clear the recency control by `+0.1710` |
| Exp18 | Contradiction-aware retrieval validation | Verify corrected facts outrank contradicted facts without deleting history | Single correction 0.08→1.00, chain latest@1 0.00→1.00, non-contradiction unchanged |

---

## Headline Metrics

| Signal | Snapshot |
|---|---|
| State-conditioned retrieval | Exp3 mean Jaccard ≈ 0.714 for NCM vs ~0 semantic baseline |
| Novelty scaling | Exp2 (AG News): semantic novelty collapses toward ~0 at 100k while full-manifold remains ~0.119 |
| Real-data behavior | Exp11 (bounded run): strongest divergence remains with NCM (JaccardΔ≈0.374) |
| Weight robustness | Exp12 default near top-performing settings |
| Boundary behavior | Exp13: NCM stronger at low/high state-shift buckets |
| Practical runtime | Exp4 cached path supports real-time-friendly latency |
| Real-model persona shift | Exp14 (qwen2:7B): Persona-B warm markers +3.833 and +63 words under identical prompts |
| Synthetic scale check | Exp15 (5k prompts, 5k memories/persona): separation L2≈0.713, memory-gain positive-rate=1.000 |
| Synthetic validation lock | Exp16: Turn10/20/30 exact match and exact `.ncm` round-trip. Leave-one-out P@5 0.7200 for both semantic-only and inferred-state NCM, so `+0.0000` gain; label-leaking oracle 0.7733 |
| Real-world episodic retrieval | Exp17 (65 conversations, 228 queries, 2,628 turns): P@5 semantic-only 0.4640, inferred-state NCM 0.4561, recency 0.2851, random guess 0.2851; label-leaking oracle 0.4886 |
| Contradiction handling proof | Exp18: corrected-fact dominance enabled via contradiction penalty with persistence-safe links |

---

## Experiment-by-Experiment Detail

## Experiment 1: Retrieval Precision

### What is this experiment?
Evaluates precision on category matching and state matching using controlled synthetic memories.

### Why is it needed?
To separate “semantic recall quality” from “state-conditioned recall quality” in a clean setting.

### Results
Source: [experiments/results/exp1/exp1_redesigned.json](results/exp1/exp1_redesigned.json)

Canonical note: this section tracks `exp1_redesigned.py` (stored-event query protocol), not the legacy `run_all_experiments` Exp1 helper.

| Metric | k | Semantic Only | Sem + Emotional | NCM Full |
|--------|---|:---:|:---:|:---:|
| Category P@k | 1 | 0.925 | 0.625 | 0.625 |
| Category P@k | 3 | 0.933 | 0.692 | 0.692 |
| Category P@k | 5 | 0.950 | 0.800 | 0.800 |
| Category P@k | 10 | 0.955 | 0.900 | 0.890 |
| State P@k | 1 | 0.075 | 0.625 | 0.625 |
| State P@k | 3 | 0.083 | 0.683 | 0.692 |
| State P@k | 5 | 0.105 | 0.435 | 0.435 |
| State P@k | 10 | 0.095 | 0.217 | 0.217 |

### What does it say?
At the canonical 1200-memory setting (stored-event queries), semantic-only dominates category precision, while NCM variants carry much stronger state precision.

---

## Experiment 2: Novelty Sensitivity at Scale

### What is this experiment?
Measures novelty score behavior as memory size increases.

### Why is it needed?
To test saturation risk in large stores.

### Results
| Store Size | Semantic Novelty | NCM Novelty | Advantage |
|:---:|:---:|:---:|:---:|
| 100 | 0.607 | 0.377 | Semantic higher |
| 1,000 | 0.503 | 0.356 | Semantic higher |
| 10,000 | 0.377 | 0.311 | Semantic higher |
| 50,000 | 0.171 | 0.219 | NCM higher |
| 100,000 | 8.94e-09 | 0.119 | NCM higher |

### What does it say?
With AG News online data, semantic novelty decreases rapidly with scale and
collapses by 100k, while full-manifold novelty remains non-zero. At 100,000
memories semantic novelty is 8.94e-09, which is below float32 machine epsilon
(1.19e-07): the nearest-neighbour cosine similarity has saturated to 1.0 within
the precision of the representation, so semantic novelty is not merely small but
numerically indistinguishable from zero.

Two scope limits on this result. First, the composite manifold is *less*
novelty-sensitive than semantic-only at every scale up to 10,000 (ratios 0.62,
0.64, 0.71, 0.78, 0.83); it overtakes semantic only between 10,000 and 50,000
stored memories. The advantage is a large-store effect, not a general one.
Second, no ratio is reported at 100,000 because the denominator is below machine
epsilon there; a ratio computed from it (1.3e7) would be a floating-point
artifact, not a measured advantage.

---

## Experiment 3: State-Conditioned Retrieval

### What is this experiment?
Same semantic query is evaluated across different internal states.

### Why is it needed?
This is the direct proof test for the `s_snapshot` contribution.

### Results
| State Pair | Semantic Jaccard | NCM Jaccard |
|:---|:---:|:---:|
| Calm-Happy vs Stressed-Angry | 0.000 | 0.792 |
| Excited-Curious vs Sad-Withdrawn | 0.000 | 0.769 |
| Confident vs Fearful | 0.000 | 0.832 |
| Neutral vs Exhausted | 0.000 | 0.333 |

Mean Jaccard (NCM) ≈ 0.714.

### What does it say?
Baseline is state-blind; NCM retrieval changes with state in a measurable way.

---

## Experiment 4: Speed Benchmarks

### What is this experiment?
Measures write throughput, retrieval latency, cache construction cost, and
persistence cost as a function of stored memory count.

### Why is it needed?
To state the actual latency cost of composite-distance retrieval relative to a
semantic-only baseline doing the same amount of work.

### Correction to the previous version of this section
The table previously published here reported three retrieval columns named
"Semantic", "Full Manifold", and "NCM Cached", with Full Manifold and NCM
Cached given as different numbers (for example 1.006 ms and 0.819 ms at 10,000
memories). Those were not two implementations. `ncm.retrieval.retrieve_top_k`
returns `retrieve_top_k_fast(...)` verbatim whenever `tag_filter` is None,
which was the case in this benchmark, so both columns ran identical code. The
only difference was that the second loop was preceded by an explicit
`store._rebuild_cache()`, so the first loop additionally paid one cache
rebuild spread over 100 queries. Any speedup ratio taken from those two columns
measured a single cache rebuild, not an algorithmic difference. The rebuild is
now timed once, on its own line.

The previous store-throughput figure (~21.4k to ~24.4k memories/sec) and
storage figure (~560 bytes/memory) also do not reproduce on the machine used
for the current run, which measures 7.3k to 8.5k writes/sec and 297
bytes/memory. The earlier run recorded no hardware or library versions, so the
discrepancy cannot be attributed. The current results file records both.

### Setup
- Source: [experiments/results/run_all_experiments/exp4_speed_benchmarks.json](results/run_all_experiments/exp4_speed_benchmarks.json)
- Scales: 100, 500, 1,000, 5,000, 10,000, 50,000, 100,000 memories
- 100 timed queries per arm after 10 warmup queries, k=5, `time.perf_counter`
- Machine: Windows 11, AMD64 Family 25 Model 68, 16 logical CPUs, Python 3.12.0,
  NumPy 1.26.4, encoder backend `sentence-transformers` on CUDA
- Memory content is synthetic (`"test memory number {i} about various topics"`)
  with state vectors drawn from the eight archetypes plus uniform noise. This
  experiment measures latency only; no retrieval quality is claimed from it.

### Retrieval latency (median ms per query)
Three arms, with the cache state each ran under stated explicitly:

| Memories | Semantic, shipped `retrieve_semantic_only` (rebuilds its own matrix per call) | Semantic, prebuilt matrix (fair-work baseline) | Composite manifold, warm cache | Composite / prebuilt semantic | One-time cache rebuild (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 0.3845 | 0.3083 | 0.6684 | 2.17x | 1.42 |
| 500 | 0.6017 | 0.2784 | 0.6777 | 2.43x | 6.49 |
| 1,000 | 0.9209 | 0.3113 | 0.7921 | 2.54x | 14.96 |
| 5,000 | 5.2166 | 0.3677 | 1.7545 | 4.77x | 68.35 |
| 10,000 | 10.2140 | 0.4158 | 1.8765 | 4.51x | 137.08 |
| 50,000 | 46.5076 | 1.5523 | 7.4176 | 4.78x | 570.80 |
| 100,000 | 95.9612 | 2.6312 | 12.9269 | 4.91x | 1131.88 |

### Write and storage cost
- Store throughput, `update_auto_state=False`: 8,467/sec at 100 memories
  falling to 7,332/sec at 100,000.
- Auto-state write cost, measured separately over 1,000 memories: 13.16 ms per
  memory (76/sec). `MemoryStore.add(update_auto_state=True)`, which is the
  default, calls `AutoStateTracker.update(text)`, and that runs one uncached
  sentence-transformer forward pass per memory. End-to-end write cost with
  state tracking enabled is therefore dominated by this encode, not by the
  store operation: 76/sec against 7,332/sec is roughly two orders of magnitude.
  This figure varied from 18.41 to 13.16 ms per memory across two runs on the
  same machine, so treat it as an order-of-magnitude measurement, not a precise
  one.
- Storage: 297 bytes/memory, flat from 500 memories upward (302 bytes at 100).
- Persistence at 100,000 memories: 5.4 s save, 3.2 s load, 29.0 MB compressed.

### What does it say?
Composite-distance retrieval costs 2.2x to 4.9x a semantic-only search doing
the same amount of work, with the ratio growing with store size because the
composite reads four cached arrays where semantic search reads one. Absolute
latency stays under 13 ms per query at 100,000 memories.

The composite path appears *faster* than the shipped `retrieve_semantic_only`
above 1,000 memories (0.135x at 100,000). That is not an architectural
advantage. `retrieve_semantic_only` reconstructs its (n, 128) candidate matrix
on every call while the composite path reads a cache built once, so the
comparison measures array allocation. The prebuilt-matrix column is the
meaningful baseline. No speedup of the composite path over semantic retrieval
is claimed.

---

## Experiment 5: Memory Systems Comparison

### What is this experiment?
Compares NCM variants and baseline retrieval in the same setup.

### Why is it needed?
To show whether NCM quality gains are still practical on latency.

### Results
Source: [experiments/results/exp5/exp5_memory_systems_comparison.txt](results/exp5/exp5_memory_systems_comparison.txt)

- ncm_cached_full: state_avg=0.7350, category_avg=0.9973, latency_ms=0.3903
- ncm_full: state_avg=0.7350, category_avg=0.9973, latency_ms=0.3989
- semantic_only: state_avg=0.1239, category_avg=1.0000, latency_ms=0.6467

### What does it say?
NCM cached preserves most quality benefits with much lower latency than full mode.

---

## Experiment 6: Current Memory Systems vs NCM

### What is this experiment?
Compares against a stronger semantic-emotional baseline.

### Why is it needed?
To test NCM beyond weak baselines.

### Results
Source: [experiments/results/exp6/exp6_current_memory_systems_vs_ncm.txt](results/exp6/exp6_current_memory_systems_vs_ncm.txt)

- semantic_emotional: state_avg=0.7672, category_avg=0.8764, latency_ms=1.1338
- ncm_cached_full: state_avg=0.5835, category_avg=0.6050, latency_ms=0.3605
- rag_semantic_only: state_avg=0.1200, category_avg=0.9847, latency_ms=0.2094

### What does it say?
Different systems optimize different objectives; NCM stays competitive while preserving state-conditioning.

---

## Experiment 7: Standardized Ranking

### What is this experiment?
Runs a weighted multi-metric ranking across quality and efficiency.

### Why is it needed?
Single metrics can be misleading; this provides balanced evaluation.

### Results
Source: [experiments/results/exp7/exp7_standard_ranking.txt](results/exp7/exp7_standard_ranking.txt)

- Ranking metrics include NDCG@10, Recall@10, MRR@10, MAP@10, state precision@10, latency, throughput, memory footprint.
- Top in the recorded run:
  1. semantic_emotional
  2. ncm_cached_full
  3. ncm_full

### What does it say?
NCM remains in top group under balanced scoring.

---

## Experiment 8: External Systems vs NCM

### What is this experiment?
Quality comparison with BM25/TF-IDF/dense/RAG-style baselines.

### Why is it needed?
To benchmark against common retrieval families.

### Results
Source: [experiments/results/exp8/exp8_external_systems_vs_ncm.txt](results/exp8/exp8_external_systems_vs_ncm.txt)

- Baselines: bm25_text, tfidf_cosine, dense_sbert_cosine, rag_semantic_only, rag_semantic_recency, recency_only
- Top in the recorded run:
  1. ncm_cached_full
  2. ncm_full
  3. rag_semantic_only

### What does it say?
NCM performs strongly when state-aware behavior is part of the objective.

---

## Experiment 9: External Systems Speed

### What is this experiment?
Latency and throughput comparison with external baselines.

### Why is it needed?
To quantify speed tradeoffs transparently.

### Results
Source: [experiments/results/exp9/exp9_external_systems_speed.txt](results/exp9/exp9_external_systems_speed.txt)

- recency_only: avg=0.0107ms
- dense_sbert_cosine: avg=0.1380ms
- ncm_cached_full: avg=0.2946ms
- ncm_full: avg=0.3075ms

### What does it say?
NCM cached is slower than trivial baselines but still practical for interactive usage.

---

## Experiment 10: WITHDRAWN (hand-authored mock-up, not a measurement)

### Status
Withdrawn. The numbers previously published in this section are not
measurements and have been removed. Do not cite them.

### What was previously published here
Four rows of Recall@5, Recall@10, NDCG@10 and Jaccard-divergence figures for
`semantic_only`, `semantic_emotional`, `ncm_full` and `ncm_cached`, presented in
the same format as the live experiments in this document.

### Why it was withdrawn
`experiments/python/exp10_retrieval_recall_benchmark.py` does not run an
experiment. It contains no `MemoryStore`, no encoder call, and no retrieval
call. Its `run_benchmark()` function assigns a literal Python dictionary of
numbers that were typed by hand to illustrate an expected pattern, writes them
to JSON, and plots them. The file's own docstring described this as generating
"synthetic results demonstrating the benchmark structure and expected NCM
advantage", but the published table did not say so, and the summary row for
Exp10 described it only as "synthetic", which understates the problem: these
are not synthetic data fed through the real pipeline, they are the outputs
themselves, invented.

The values therefore cannot be reproduced, because no computation produces
them, and cannot be falsified, because they are not claims about how NCM
behaves. The script additionally printed an interpretation asserting that a
hand-typed 0.127 "proves that s_snapshot genuinely changes what the system
recalls", and compared it against another system's measured 96.6% recall. That
text has been removed from the script.

### What replaces it
The state-conditioning question Exp10 was meant to illustrate is measured for
real, from live retrieval calls, in:
- Experiment 3 (state-conditioned retrieval, `run_all_experiments.py`)
- Experiment 11 (real multi-session chat corpus)
- Experiment 12 (weight sensitivity sweep)

Exp10 is excluded from the suite runner. The script is retained, with a
`provenance: HAND_AUTHORED_LITERALS` field now written into its JSON output and
a warning banner on execution, so the provenance of the previously published
numbers stays auditable.

---

## Experiment 11: Real-World Corpus Benchmark

### What is this experiment?
Evaluation on unseen multi-session real chat corpus.

### Why is it needed?
To address synthetic-only concerns and validate external generalization.

### Results
Source: [experiments/results/exp11/exp11_real_world_corpus_benchmark.txt](results/exp11/exp11_real_world_corpus_benchmark.txt)

- Corpus source: experiments/data/real_world_corpus
- Run config: max_chunks=300, query_stride=20, max_queries=20, k=10
- semantic_only: R@10=0.034, NDCG=0.720, MRR=0.484, JaccardΔ=0.000
- semantic_emotional: R@10=0.034, NDCG=0.721, MRR=0.487, JaccardΔ=0.299
- ncm_full: R@10=0.037, NDCG=0.731, MRR=0.488, JaccardΔ=0.374
- ncm_cached: R@10=0.037, NDCG=0.731, MRR=0.488, JaccardΔ=0.374

### What does it say?
On real data, NCM preserves strongest state-divergence while staying competitive in ranking quality.

---

## Experiment 12: Weight Sensitivity

### What is this experiment?
Sweeps retrieval weights (`alpha`, `beta`, `gamma`, `delta`) on real corpus slice.

### Why is it needed?
To test whether the system is robust or fragile to weight selection.

### Results
Source: [experiments/results/exp12/exp12_weight_sensitivity.txt](results/exp12/exp12_weight_sensitivity.txt)

- semantic_light: NDCG=0.763, R@10=0.011, JaccardΔ=0.529
- emotional_heavy: NDCG=0.763, R@10=0.012, JaccardΔ=0.509
- state_heavy: NDCG=0.762, R@10=0.013, JaccardΔ=0.578
- temporal_heavy: NDCG=0.762, R@10=0.011, JaccardΔ=0.302
- default: NDCG=0.760, R@10=0.012, JaccardΔ=0.369

### What does it say?
Default weights are stable and close to best; no brittle single optimum.

---

## Experiment 13: Honest Head-to-Head Rematch

### What is this experiment?
Controlled rematch of `semantic_emotional` vs `ncm_full`, with shift-bucket analysis.

### Why is it needed?
To identify where each method wins and avoid over-general claims.

### Results
Source: [experiments/results/exp13/exp13_baseline_rematch.txt](results/exp13/exp13_baseline_rematch.txt)

- semantic_emotional: R@10=0.208, NDCG=0.587, MRR=0.463, Divergence=0.252
- ncm_full: R@10=0.220, NDCG=0.605, MRR=0.452, Divergence=0.306
- Buckets:
  - Low shift: NCM +0.033 NDCG
  - Medium shift: semantic_emotional +0.005 NDCG
  - High shift: NCM +0.022 NDCG

### What does it say?
NCM is stronger at low/high shift extremes; middle regime remains competitive for semantic-emotional.

---

## Experiment 14: Persona Memory Effect with Real Ollama

### What is this experiment?
Runs the same prompt set through one real Ollama model (`qwen2:7B`) with two different seeded memory profiles.

### Why is it needed?
To test whether memory context changes response style/persona in real generation, not just synthetic metrics.

### Results
Source: [experiments/results/exp14/exp14_persona_memory_ollama.txt](results/exp14/exp14_persona_memory_ollama.txt)

- Persona B vs Persona A deltas (same prompts):
  - words: +63.167
  - chars: +319.167
  - warm_markers: +3.833
  - exclamations: +0.333

### What does it say?
Memory profile changes measurable style properties in real model responses under identical prompts.

---

## Experiment 15: Synthetic Persona Memory Effect (Large Scale)

### What is this experiment?
Large synthetic stress test of memory-conditioned persona behavior with controlled latent style dimensions.

### Why is it needed?
To verify that persona-conditioning signal remains stable at larger scale, beyond small prompt sets.

### Results
Source: [experiments/results/exp15/exp15_synthetic_persona_memory_effect.txt](results/exp15/exp15_synthetic_persona_memory_effect.txt)

- Config: 5,000 prompts, 5,000 memories per persona bank, top-k=8
- Persona separation L2 (mean): 0.7133
- Persona separation L2 (p90): 0.7946
- Memory-gain positive-rate: 1.000 (both personas)
- Targeted style deltas (B-A):
  - analytical: -0.3430
  - warm: +0.4310
  - expressive: +0.3397
  - direct: -0.2784

### What does it say?
At scale, the memory-conditioned response shift remains strong, separable, and aligned with target persona directions.

---

## Experiment 16: Auto-State Integration Validation

### What is this experiment?
Three independent checks on the integrated auto-state tracker: trajectory determinism against pinned constants, era retrieval precision, and `.ncm` persistence.

### Why is it needed?
To prove the implementation matches the design exactly, survives persistence round-trips, and to measure whether the state channel actually changes retrieval quality.

### Correction to the previous version of this section
The previous version reported "Retrieval trend: mean P@5 gain = +0.133" and concluded that auto-state "keeps retrieval behavior consistent". Both are withdrawn. The retrieval check set the query's 5-dimensional state to `states_at_era_end[era]`, that is, to the exact tracker state produced by the target era's own ten turns. Because the composite distance rewards state proximity, this handed the target era a perfectly aligned state signal derived from the relevance label itself, so the check could not fail. It also ran only three queries, one per era, and scored them with a bespoke 0.5/0.5 scorer rather than the shipped retrieval functions.

### Setup
- Corpus: 30 turns of a hand-authored script, written in three blocks of ten. Turns 1-10 express stress, 11-20 curiosity, 21-30 positive affect. **Era membership is a hand-authored label**, defined by position in a script written for this experiment, not by any corpus annotation.
- Writes go through the shipped `MemoryStore.add(..., update_auto_state=True)`.
- Retrieval goes through `ncm.retrieval.retrieve_semantic_only` and `ncm.retrieval.retrieve_top_k_fast`. No bespoke scorer.
- Scoring: leave-one-out over all 30 turns, plus the 3 original hand-authored era probes. The held-out memory is excluded from its own result list.
- Encoder: `all-MiniLM-L6-v2`, backend `sentence-transformers`. The script aborts rather than reporting numbers if the encoder falls back to the hash encoder.

### Results
Source: [experiments/results/exp16/exp16_auto_state_integration.json](results/exp16/exp16_auto_state_integration.json)

**Check 1, trajectory determinism: PASS.** Max absolute difference against the pinned constants at turns 10, 20 and 30 is `0.00e+00` on all five dimensions, tolerance `1e-05`. This is a regression guard on the update rule. It is not an accuracy claim: the constants were produced by the same code.

**Check 2, era retrieval.** Leave-one-out over all 30 turns, random guess P@5 = `0.3103` (9 same-era peers among 29 candidates):

| Arm | P@5 | P@10 | Era1 P@5 | Era2 P@5 | Era3 P@5 |
|---|---|---|---|---|---|
| `semantic_only` | 0.7200 | 0.6067 | 0.720 | 0.540 | 0.900 |
| `ncm_inferred` (state from query text alone) | 0.7200 | 0.6233 | 0.640 | 0.580 | 0.940 |
| `ncm_oracle` (state from target era, LABEL LEAK) | 0.7733 | 0.6833 | 0.780 | 0.600 | 0.940 |

- `ncm_inferred` minus `semantic_only` P@5: **+0.0000**
- `ncm_oracle` minus `ncm_inferred` P@5: +0.0533

On the three original hand-authored era probes, mean P@5 is `semantic_only` 0.7333, `ncm_inferred` 0.6667, `ncm_oracle` 0.8000, so `ncm_inferred` is **0.0666 worse** than the baseline there.

The `ncm_oracle` arm is retained only as an upper bound and is labelled as leaking in the JSON, the text output and the figure. It sets the query state to the target era's end state, which is produced by the target era's own turns, so it bounds what a perfectly informed state signal could contribute. It is not a system result.

**Check 3, persistence: PASS.** Over a 20-memory store, `max_state_diff` = `0.00e+00` and `max_retrieval_distance_diff` = `0.00e+00`. Turn counter, alpha vector, adaptive weights and memory count all survive, and the full top-10 ranking through `retrieve_top_k_fast` is identical before and after save/load, including top-1.

**Verdict: PASS, scoped to checks 1 and 3.** Era retrieval reports magnitudes and is not a pass/fail gate.

### What does it say?
The auto-state implementation is numerically exact and persists exactly. It does not improve retrieval on this script. With the query state inferred from the query text alone, which is the only deployable condition, the composite manifold scores identically to semantic similarity alone (0.7200 P@5) and loses on the era probes. The previously reported +0.133 gain came from the oracle leak. The oracle ceiling of +0.0533 shows that even a perfectly informed state signal would add little here, which is consistent with a 30-turn corpus in which semantic content and era are almost perfectly correlated by construction.

---

## Experiment 17: Same-Session Episodic Retrieval on Multi-Session Chat

### What is this experiment?
Given a held-out dialogue turn as the query, it measures whether the system retrieves the other turns from the same conversational session, out of a store holding every other turn of that multi-session conversation.

### Why is it needed?
To test the composite manifold against a relevance label that exists in the data rather than one authored for the experiment. Every session in `experiments/data/real_world_corpus/train.jsonl` carries a `session_id`, and turns from one sitting belonging together is exactly the property an episodic memory system should capture. A recency-only arm is included as a control, because sessions are contiguous blocks of turns and session membership is therefore correlated with recency.

### Results
Source: [experiments/results/exp17/exp17_real_world_scale.json](results/exp17/exp17_real_world_scale.json)

- Corpus: 8,939 Multi-Session Chat style records, persona sentences PersonaChat derived
- Sampled 100 conversations, benchmarked 65; 35 skipped for having fewer than 2 sessions
- Queries evaluated: 228, turns stored: 2,628, mean store size: 40.4
- Mean relevant turns per query: 11.53; random-guess P@5: 0.2851
- Seed 20260819, encoder backend sentence-transformers, timer `time.perf_counter`

| arm | P@5 | P@10 | R@10 | NDCG@10 | MRR | med ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `semantic_only` | 0.4640 | 0.4053 | 0.3576 | 0.4518 | 0.7786 | 0.0871 | 0.1387 |
| `recency_only` | 0.2851 | 0.2851 | 0.2625 | 0.2866 | 0.2861 | 0.0110 | 0.0272 |
| `ncm_inferred` | 0.4561 | 0.4088 | 0.3600 | 0.4529 | 0.7714 | 0.2888 | 0.4976 |
| `ncm_oracle` | 0.4886 | 0.4215 | 0.3713 | 0.4690 | 0.7902 | 0.2393 | 0.4094 |

- `ncm_inferred` minus `semantic_only`: P@5 `-0.0079`, NDCG@10 `+0.0011`
- `ncm_inferred` minus `recency_only`: P@5 `+0.1710`
- `ncm_oracle` minus `ncm_inferred`: P@5 `+0.0325`
- Auto-state dispersion, measured on 366 turns from 20 conversations: mean per-turn std-dev `0.0150` (sd `0.0075`), range `[0.0022, 0.0473]`, mean max-min range `0.0423`, mean entropy `1.7464`

`ncm_oracle` sets the query state to the target session's mean stored auto-state. That is derived from the relevance label, so the arm bounds what the state channel could contribute and is not a system result.

### What does it say?
The composite manifold does not improve same-session retrieval over semantic similarity alone when the query state is inferred from the query text, which is the only condition available at deployment. It is `0.0079` P@5 below the baseline. Both NCM arms clear the recency control by `+0.1710` P@5, so the composite distance is doing more than preferring recently written turns, and all three retrieval arms clear random guess. The state channel specifically is what fails to add signal, consistent with EXP16.

### Retraction
This replaces a previously published result of `NCM P@5 / P@10: 1.000 / 1.000` and `RAG P@5 / P@10: 1.000 / 1.000` on 100 conversations and 2,009 turns. Precision in that version was computed as `len(top_5_list) / 5.0`, which is identically `1.0` for any store holding at least five memories, for every arm, so it measured list length and not relevance. The accompanying latency figures of `~0.05 ms` and `~0.02 ms` are withdrawn as well, because they were taken with `time.time()`, whose resolution on Windows is around 15 ms and therefore coarser than the operation being timed.

### Caveats
- The latency column is not a benchmark. Mean store size is 40.4 memories, so fixed per-call overhead dominates, and the two paths differ in cache treatment: `retrieve_top_k_fast` reads a cache warmed once per conversation, while `retrieve_semantic_only` rebuilds its own `(n, 128)` matrix on every call. Experiment 4 is the latency measurement.
- `recency_only` scores identically at k=5 and k=10 because its retrieval window holds `1.026` distinct sessions on average, and `0.2763` of queries have a window drawn entirely from the target session. Such a query scores 1.0 at both k and every other scores 0.0. This is corpus layout, not a scoring error.
- Persistence is not tested here. It is validated in Experiment 16.

---

## Experiment 18: Contradiction-Aware Retrieval Validation (CADP)

### What is this experiment?
Validates contradiction-aware retrieval by creating corrected facts (`old -> new`) and measuring whether retrieval reliably promotes the latest truth.

### Why is it needed?
Temporal decay alone cannot represent local contradiction. This experiment verifies that contradiction links + penalty solve stale-fact dominance while preserving historical memory.

### Results
Source: [experiments/results/exp18/exp18_cadp_validation.json](results/exp18/exp18_cadp_validation.json)

- Single correction (`A -> B`) where `new` beats `old`:
  - baseline: `0.08`
  - CADP: `1.00`
- Chain correction (`A -> B -> C`) latest-top1:
  - baseline: `0.00`
  - CADP: `1.00`
- Conflict trace top-3 retrieval rate: `1.00`
- Non-contradiction regression (top-1 unchanged): `1.00`
- Persistence round-trip for contradiction metadata: PASS
- Verdict: PASS

### What does it say?
CADP fixes the correction-order failure mode without deleting old memories, keeps normal retrieval intact when no contradiction exists, and persists contradiction links/traces safely in `.ncm` files.

---

## Visual Appendix

Figures below are regenerated from the current results files by
`experiments/python/generate_plots.py`. That script had been unrunnable: every
one of its four plot functions opened a results filename that no longer exists
(`exp3_state.json`, `exp2_novelty.json`, `exp4_speed.json`), and two also read
JSON keys that had been renamed (`semantic_jaccard` to
`semantic_jaccard_mean`, and a `ratio` key that was removed entirely). It
therefore raised `FileNotFoundError` on its first call, and the figures
previously shown here were stale copies dated 2026-04-09 that no longer matched
the numbers in the tables above. Those stale copies have been removed; the four
regenerated figures live alongside the JSON they were built from, in
`results/run_all_experiments/`.

![Category Precision](results/exp1/exp1_category_precision.png)
![State Precision](results/exp1/exp1_state_precision.png)
![Precision Bars](results/exp1/exp1_precision_bars.png)

![Novelty Scale](results/run_all_experiments/exp2_novelty_scale.png)
![State Conditioned Jaccard](results/run_all_experiments/exp3_state_conditioned.png)
![Speed Benchmarks](results/run_all_experiments/exp4_speed.png)
![Combined Dashboard](results/run_all_experiments/ncm_dashboard.png)

![Quality Metrics](results/exp7/exp7_quality_metrics.png)
![Efficiency Metrics](results/exp7/exp7_efficiency_metrics.png)
![Overall Ranking](results/exp7/exp7_overall_ranking.png)

![External Quality](results/exp8/exp8_external_quality.png)
![External Ranking](results/exp8/exp8_external_ranking.png)

![Speed Latency](results/exp9/exp9_external_systems_speed_latency.png)
![Speed QPS](results/exp9/exp9_external_systems_speed_qps.png)

![Real-World Corpus Benchmark](results/exp11/exp11_real_world_corpus_benchmark.png)
![Weight Sensitivity](results/exp12/exp12_weight_sensitivity.png)
![Baseline Rematch](results/exp13/exp13_baseline_rematch.png)
![Persona Memory Summary (Real Ollama)](results/exp14/exp14_persona_memory_ollama_summary.png)
![Persona Memory Prompt Deltas (Real Ollama)](results/exp14/exp14_persona_memory_ollama_prompt_deltas.png)
![Synthetic Persona Memory Summary](results/exp15/exp15_synthetic_persona_memory_effect_summary.png)
![Synthetic Persona Style Clusters](results/exp15/exp15_synthetic_persona_memory_effect_clusters.png)
![Synthetic Persona Scale Curve](results/exp15/exp15_synthetic_persona_memory_effect_scale.png)
![EXP16 State Trajectory](results/exp16/exp16_state_trajectory.png)
![EXP16 Era Retrieval](results/exp16/exp16_retrieval_trend.png)
![EXP16 Persistence Validation](results/exp16/exp16_persistence_validation.png)
![EXP17 Retrieval Precision](results/exp17/exp17_scale_retrieval_precision.png)
![EXP17 Performance Metrics](results/exp17/exp17_scale_performance_metrics.png)
![EXP17 State Accuracy](results/exp17/exp17_scale_state_accuracy.png)
![EXP18 Rank Accuracy](results/exp18/exp18_rank_accuracy.png)
![EXP18 Latency & Regression](results/exp18/exp18_latency_regression.png)

![NCM Dashboard](results/run_all_experiments/ncm_dashboard.png)

---

## Setup Notes

- Synthetic benchmark: ~1,200 memories with semantic categories and state archetypes
- Real corpus benchmark: [experiments/data/real_world_corpus](data/real_world_corpus)
- EXP16 validation: 30-turn hand-authored synthetic trajectory with persistence round-trip checks and leave-one-out era retrieval
- EXP17 validation: 100 conversations sampled from the real-world corpus, 65 benchmarked, 228 leave-one-out queries over 2,628 stored turns, relevance label taken from the corpus `session_id` field
- EXP18 validation: contradiction-heavy synthetic tasks with correction chains, conflict traces, and metadata persistence checks
- Metrics used across tracks: Precision@k, Recall@k, MRR, NDCG, MAP, state precision, latency, throughput, footprint
- Hardware context: Ryzen 7 6800H, RTX 3050 (4GB), 16GB RAM, Windows
