# Changelog

## Unreleased

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
