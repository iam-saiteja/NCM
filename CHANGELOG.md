# Changelog

## Unreleased

No library changes; new measurements.

- `comparisons/locomo_bedrock.py`: all of LoCoMo with a large model on Amazon Bedrock.
  NCM with gpt-oss-120b answering and gpt-oss-20b judging: 68.6% J, token F1 0.537,
  evidence in the prompt 88.8%, 12 ms retrieval, ~$0.65 a run. README lists the
  published scores of other systems with their sources.
- `bench/mood_effect.py`: does the mood line change the character's replies? With
  qwen2.5:7b, yes: tone fit 3.81 -> 4.09 of 5 (p 0.03), worse with a wrong label; with
  qwen2.5:3b, barely. `mood()` reads 61% of users' moods right, never the opposite.
- `comparisons/memorybench_bridge.py`, `memorybench_summary.py` and
  `comparisons/memorybench/ncm-bedrock.patch`: NCM as a provider in Supermemory's
  MemoryBench, run against its RAG baseline with gpt-oss-120b answering and gpt-oss-20b
  judging. NCM 84.8% vs RAG 82.2% over 5,329 questions (p 2e-6): ConvoMem 95.5 vs 92.3,
  LongMemEval 80.6 vs 79.4, LoCoMo 70.4 vs 68.5; $4.43 vs $35.83.

## 2.1.0 (2026-10-12)

Forgetting, from memory research, measured with `accuracy/bounded.py` (why, and the
ideas that were tested and rejected: [docs/BRAIN.md](docs/BRAIN.md)). Ranking is
unchanged; this only matters with `max_items`.

- `Config.self_weight` (0.5): what people say about themselves is harder to forget.
- Fixed: forgetting measured age from the wall clock, so imported or replayed
  histories looked equally old; it now measures from the newest memory.
- With a store capped at a quarter of each history, the answer reaches the prompt for
  53% of LongMemEval questions (2.0: 42%), and qwen2.5:3b answers 43.1% correctly
  (2.0: 30.9%, p = 0.002). At half the store the two tie.
- `mood()` orders by time even for imported histories; a file with settings from a
  newer version still loads.
- `accuracy/`: how often a character with NCM memory answers correctly about the user
  (LongMemEval, 204 questions, LLM-judged, token P/R/F1 recorded, two-judge check).
- `bench.metrics.paired_bootstrap`: p is capped at 1 (it could exceed 1 when two arms tie).

## 2.0.0 (2026-10-11)

A rebuild. v1's four-channel distance ranked evidence worse than plain cosine
similarity on LoCoMo and LongMemEval, so the core was replaced and every kept
mechanism was measured. Why each piece changed: [docs/HISTORY.md](docs/HISTORY.md).
Numbers: [docs/RESULTS.md](docs/RESULTS.md).

**New**
- `ncm.Memory`: one object for add / recall / context / persona / mood / save.
- Hybrid ranking: dense similarity fused with BM25 (`ncm.lexical`, pure numpy).
- Backward contiguity lift: a strong hit raises the turn before it.
- `context()`: a prompt-ready block with persona, mood, dated memories in order,
  and each hit's neighbouring turns; `self_name` renders the agent's own lines as "you".
- `persona()`: self-describing statements chosen with MMR; questions are skipped.
- Affect from the stored vector (EmoBank probe, no extra model call) used for
  salience and `mood()`.
- Forgetting by ACT-R base-level activation plus salience, batched at capacity.
- `.ncm` format v3: npz with a JSON header, atomic writes, no pickle, BM25 index
  stored; `ncm.legacy.import_v1` imports v1 files.
- `bench/`: LoCoMo and LongMemEval retrieval, LLM QA, persona, speed and affect
  benchmarks, with the v1 baseline replayed on identical embeddings.
- `examples/chat.py`: an Ollama character that remembers you across sessions.
- `tests/`: 20+ tests, no model download needed.

**Removed**
- The 128-d random projection, emotional / state / recency ranking channels,
  `AutoStateTracker`, CADP contradiction links, the write gate, profiles and
  presets, the v1 binary format writer, and `experiments/` (25 scripts, ~88 MB).
  All remain in git history at `9c1e851`.

## 1.x (April - August 2026)

See the v1 history at `git show 9c1e851:CHANGELOG.md` and the summary in
[docs/HISTORY.md](docs/HISTORY.md).
