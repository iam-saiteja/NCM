# How NCM got here

NCM has had three phases. This file records what each one did, what it got right
and wrong, and what v2 kept. Every number below is reproducible from `bench/`
or from the git history it cites.

## Phase 1: the original design (April to July 2026, Sai Teja)

Commits `9857a33` to `0657956`.

NCM v1 stored each memory as several vectors at once: a 128-d random projection
of a MiniLM sentence embedding, a 3-d "emotional" projection of a 7-d state
vector, the state itself, a timestamp and a strength. Retrieval was a weighted
sum of four distances (semantic 0.4, emotional 0.2, state 0.3, recency 0.1).
Around that core it shipped a binary `.ncm` format, a write gate, eviction,
consolidation, an `AutoStateTracker` that turned text into a 5-d affect state,
contradiction-aware retrieval (CADP), an Ollama chat app and 18 experiments.

**What was right:** the instinct that a memory layer for a character needs more
than a vector store: time, sessions, who said what, emotional weight, a profile
of the person, and forgetting. Clean module boundaries, one-file persistence,
and a working local chat app. All of those ideas survive in v2.

**What did not hold up:** the four-channel distance itself. Measured on two
public long-conversation benchmarks with identical embeddings (`bench/v1_baseline.py`),
v1 ranks the evidence *worse than plain cosine similarity*:

| Recall@5 | v1 | cosine (MiniLM) |
|---|---|---|
| LoCoMo | 0.278 | 0.360 |
| LongMemEval-S | 0.497 | 0.575 |

and gives a local LLM worse answers (LoCoMo QA F1 0.184 vs 0.214). The causes,
in order of size: the 128-d random projection throws away information MiniLM
already had; the affect and state channels are close to random with respect to
relevance (see Phase 2); and the recency channel prefers new memories when the
question is about an old one.

Several published claims were also wrong because of experiment bugs (found in
Phase 2, below): a 50-100x speedup that was really a 2-5x slowdown, a "perfect"
precision that measured list length, an auto-state gain that came from leaking
the test labels.

## Phase 2: the "AI drift" period (19-23 August 2026, Claude, unsupervised)

Commits `b1c6f86` to `9c1e851`, 22 commits, +31,061 / -2,476 lines (mostly
result JSON). Claude was given free rein over the repository.

**What it did well: it audited the science.**

- Found and retracted real errors: the exp4 speedup (reversed: composite
  retrieval was 2.17-4.91x *slower*), exp10 numbers with no source, exp14 persona
  numbers that traced to no file, exp16's oracle label leak (real gain +0.000),
  exp17's precision bug (`len(top_5)/5`), exp15 rounding 4999/5000 up to 1.000.
- Fixed a silent, broken hash fallback in the encoder (it could never run, and
  if it had, it would have produced meaningless numbers with no warning), and a
  `.gitignore` whose every rule had a stray CR.
- Found *why* the affect channels did nothing: a channel's influence on ranking
  is its weight times its spread across candidates, and the emotional channel
  carried 1.26% of the ranking signal at a nominal 20% weight (exp22). Then
  showed that forcing the weights to act makes retrieval worse (exp25, MRR
  -0.084, p = 3.8e-4): the channels contain no retrievable signal.
- Proposed two ideas that v2 measured and partly kept: temporal contiguity
  (the turns around a hit matter) and a persona selector (pick self-disclosing,
  diverse statements).

**What it did badly: it never turned the audit into a better system.**

- Every finding pointed the same way (the composite loses to its own semantic
  channel), yet no change made the shipped default better. Instead it added
  opt-in flags that changed nothing by default (`channel_normalization`,
  `honor_caller_state`, `temporal_anchor`), each guarded by long warnings.
- Code became prose. `ncm/retrieval.py` grew from 585 to 1,058 lines, 527 of
  them comments and docstrings, several hundred words per function defending
  past decisions. The README reached 48 KB of retraction blocks.
- Bugs it found and wrote down were left in place "because CADP is off by
  default", e.g. the correction marker `"now"` matching inside `"know"`.
- No tests. No measurement on a standard benchmark. The goal drifted from
  "make the memory good" to "make every sentence in a future paper defensible".

In short: an excellent reviewer, a poor engineer. The honesty work was
necessary; it just stopped one step short of the fix.

## Phase 3: v2 (11 October 2026)

Rebuilt around measurements on LoCoMo and LongMemEval-S, keeping what measured
well and deleting what did not. Details and every number: [RESULTS.md](RESULTS.md).

| v1 component | v2 | why |
|---|---|---|
| 128-d random projection | full embedding | projection lost recall |
| emotional + state distance channels | removed from ranking | no retrievable signal (exp22, exp25, re-confirmed) |
| recency channel | removed from ranking | ranking by recency alone: R@5 0.002; a recency prior added nothing at weight 0.1 (within +-0.003) and cost recall at 0.3 |
| - | BM25 fused with dense | the single biggest gain: names, numbers, rare words |
| `temporal_anchor` idea | backward contiguity lift | the turn before a hit is usually its question; helps both benchmarks |
| strength + decay + eviction | ACT-R activation, for forgetting only | forgetting should never distort relevance |
| `AutoStateTracker` (5-d EMA, arousal r = 0.10) | EmoBank probe on the stored vector (valence r 0.62, arousal 0.44) | used for salience and `mood()`, not ranking |
| CADP regex contradiction links (precision 0.57) | dated, chronological `context()` | the reader sees which statement is newer; no brittle regex |
| persona payload (437 lines) | `persona()` (~80 lines) | same idea, statements only, questions skipped |
| write gate / consolidation | exact-repeat reinforcement | design choice, not measured: merging near-duplicates can erase genuinely repeated events, exact repeats are safe to fold |
| `.ncm` v2 binary format | `.ncm` v3 (npz, no pickle) + `ncm.legacy.import_v1` | simpler, safe to load, old files still import |
| 25 experiment scripts, ~88 MB results | `bench/` (6 scripts) | standard benchmarks, one command each |

The old experiments, results and paper workspace remain in git history
(`git show 9c1e851:experiments/...`).

### Answer accuracy (11 October 2026, later)

Asked for a character that gets facts about its user right "ninety-nine percent" of the
time, we measured answers rather than retrieval ([accuracy/](../accuracy/README.md)).
The memory layer reached it for the questions that matter (with the reranker, the
evidence was in the prompt for 136 of 136 user-fact and changed-fact questions); the
local 3B/7B readers did not (79-84% overall, 87.5% even when handed the exact
conversations). Three ideas were tried and dropped because they lost accuracy: a
notes-first reader prompt, cutting the turns around each hit, and listing the newest
conversation first; `context()` keeps its oldest-first order, now with a comment saying why.
