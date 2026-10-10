# Results

Everything here is produced by `bench/` and stored in `bench/results/`. All runs
used one laptop: RTX 3050 (4 GB), 16 GB RAM, Windows 11. The embedder is
all-MiniLM-L6-v2 unless stated. Confidence intervals are 95% paired bootstrap
over queries (10,000 resamples).

## 1. Retrieval: does the evidence come back?

Every turn of a conversation is stored as a memory; each question ranks all of
them. A hit is a turn annotated as evidence for the answer.

- **LoCoMo** (Maharana et al. 2024): 10 conversations of ~600 turns over up to
  35 sessions, 1,531 questions with turn-level evidence (adversarial category
  dropped: it has no evidence).
- **LongMemEval-S** (Wu et al. 2025): 500 questions, each over its own ~50-session,
  ~500-turn user/assistant history (246,750 turns in total); 479 questions have
  evidence turns (abstention questions dropped).

| | LoCoMo R@5 | R@10 | nDCG@10 | MRR | LongMemEval R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| NCM v1 (shipped default) | 0.278 | 0.375 | 0.241 | 0.228 | 0.497 | 0.644 | 0.451 | 0.451 |
| cosine (MiniLM) | 0.360 | 0.445 | 0.304 | 0.291 | 0.575 | 0.735 | 0.539 | 0.543 |
| BM25 | 0.506 | 0.578 | 0.442 | 0.429 | 0.683 | 0.794 | 0.650 | 0.657 |
| **NCM v2** | **0.555** | **0.643** | **0.480** | **0.461** | **0.735** | **0.853** | **0.696** | **0.702** |
| **NCM v2 + reranker** (accuracy mode) | **0.657** | **0.711** | **0.611** | **0.618** | **0.812** | **0.903** | **0.759** | **0.752** |

NCM v2 minus cosine, R@5: LoCoMo **+0.195** [0.170, 0.220], LongMemEval **+0.159**
[0.126, 0.192]. Minus v1: **+0.276** [0.251, 0.302] and **+0.238** [0.202, 0.274].
Minus BM25: +0.049 [0.036, 0.062] and +0.052 [0.027, 0.077]. All p < 0.001.

v2 is ahead of v1 and of cosine in **every question category** on both datasets:

| R@5 by category | v1 | cosine | v2 |
|---|---|---|---|
| LoCoMo single-hop (841) | 0.317 | 0.417 | **0.642** |
| LoCoMo temporal (320) | 0.362 | 0.412 | **0.648** |
| LoCoMo multi-hop (281) | 0.126 | 0.184 | **0.275** |
| LoCoMo open-domain (89) | 0.093 | 0.184 | **0.282** |
| LongMemEval single-session-user (64) | 0.672 | 0.820 | **0.953** |
| LongMemEval single-session-assistant (56) | 0.821 | 0.857 | **0.875** |
| LongMemEval single-session-preference (30) | 0.383 | 0.578 | **0.589** |
| LongMemEval knowledge-update (72) | 0.528 | 0.583 | **0.812** |
| LongMemEval multi-session (125) | 0.359 | 0.442 | **0.588** |
| LongMemEval temporal-reasoning (132) | 0.414 | 0.457 | **0.698** |

### Accuracy mode: a cross-encoder over the top 30

`Memory(reranker=True)` re-scores NCM's top 30 with cross-encoder/ms-marco-MiniLM-L-6-v2
(22M parameters) and keeps the reranker's order. R@5: LoCoMo +0.102 [0.085, 0.120],
LongMemEval +0.077 [0.052, 0.103]; MRR +0.157 and +0.049. Fusing the reranker's score
with the first-stage score did worse than the reranker alone (0.626 and 0.802).
Cost: 92 ms (LoCoMo) and 222 ms (LongMemEval, long assistant turns) per query on the
laptop GPU while it was shared with Ollama, against under 1 ms for the default path.

### What each part contributes (ablations, R@5 / nDCG@10)

| | LoCoMo | LongMemEval |
|---|---|---|
| NCM v2 (dense + BM25 0.75 + backward lift) | 0.555 / 0.480 | 0.735 / 0.696 |
| without BM25 | 0.368 / 0.310 | 0.582 / 0.547 |
| without the lift | 0.545 / 0.475 | 0.728 / 0.676 |
| lift both directions (`lift_next=0.75, lift_width=3`) | **0.594 / 0.497** | 0.597 / 0.552 |
| lift forward only | 0.567 / 0.491 | 0.648 / 0.584 |
| BM25 weight 0.5 / 1.0 | 0.557 / 0.550 | 0.725 / 0.728 |

- **BM25 is the big win** (+0.19 / +0.15 R@5). Names, numbers and rare nouns are
  exactly what conversational questions hinge on, and exactly what sentence
  embeddings blur.
- **The backward lift** (a hit raises the turn before it) is small but
  consistent: LongMemEval nDCG +0.020 [0.010, 0.030], MRR +0.031 [0.015, 0.047];
  LoCoMo R@5 +0.010 [0.000, 0.020] (p 0.05).
- **Forward lift is a trade-off, so it is off by default.** It helps a dialogue
  between two people (LoCoMo R@5 +0.039 [0.024, 0.055]) and badly hurts
  user/assistant chat (LongMemEval R@5 -0.138 [-0.164, -0.111]). Turn it on with
  `Config(lift_next=0.75, lift_width=3.0)` for two-person transcripts.

### Ideas tested and rejected

| idea | result |
|---|---|
| rank by recency | R@5 0.002 (LoCoMo), 0.002 (LongMemEval) |
| recency prior on top of hybrid (weight 0.1 / 0.3) | +0.000 / -0.002 (LoCoMo), +0.003 / +0.000 (LongMemEval) |
| reciprocal rank fusion instead of z-score fusion | 0.463 vs 0.547 (LoCoMo), 0.683 vs 0.713 (LongMemEval) |
| session-max boost (a turn inherits its session's best score) | +0.035 LoCoMo, -0.022 LongMemEval |
| spread from every positive hit, symmetric (v2 pre-release; vs hybrid at the same BM25 weight 0.5) | +0.055 LoCoMo, -0.034 LongMemEval |
| spread from the top 3/5/10 hits only | LoCoMo up to +0.069, LongMemEval -0.04 to -0.10 |
| spread into same-speaker turns only (weight 0.75) | +0.039 LoCoMo, -0.047 LongMemEval |
| forward lift gated by turn shortness (short replies need context) | LoCoMo +0.036, LongMemEval -0.016 to -0.076 |
| embed without the "speaker:" prefix | LoCoMo dense R@5 0.360 -> 0.216 |
| 128-d random projection (v1) | part of v1's loss; v2 keeps the full vector |

Tuning protocol, stated plainly: the lexical weight and spreading grid were first
tuned on LoCoMo with LongMemEval held out. The LoCoMo-tuned spreading default
failed on LongMemEval (above), so the final default was chosen with both datasets
in view. It is three numbers and moves both datasets the same way, but no clean
held-out set remains for it.

### Embedding model

On LoCoMo, bge-small-en-v1.5 (33M parameters, 3x MiniLM's encoding cost) raises
dense R@5 from 0.360 to 0.478, but after BM25 fusion the gap almost closes:
hybrid 0.545 (MiniLM) vs 0.563 (bge-small). MiniLM stays the default; use
`Memory(embedder="BAAI/bge-small-en-v1.5")` for the extra ~2 points.

## 2. End-to-end: do the answers get better?

QA_SECTION

## 3. Personalisation: does it know who the person is?

PERSONA_SECTION

## 4. Affect

The valence/arousal probe is a 384x2 ridge regression on MiniLM embeddings. Valence
is fitted on EmoBank (Buechel and Hahn 2017; ratings of news, fiction and letters) plus
GoEmotions (Demszky et al. 2020; Reddit comments, using the dataset's own
positive / negative / neutral grouping), each corpus weighted equally; arousal on
EmoBank alone. The ridge penalty was picked on both dev splits. Test splits:

| estimator | EmoBank valence r | EmoBank arousal r | GoEmotions AUC (positive vs negative) |
|---|---|---|---|
| v1 `AutoStateTracker` anchor words | 0.42 | 0.10 | - |
| v2 anchor sentences (fallback for non-MiniLM embedders) | 0.45 | 0.16 | 0.855 |
| v2 probe, EmoBank only | 0.62 | 0.44 | 0.884 |
| **v2 probe, EmoBank + GoEmotions (shipped)** | **0.62** | **0.44** | **0.905** |

Embedding with the store's "user: " prefix changes EmoBank r by under 0.01.
v1's arousal dimension was essentially noise. The probe reads tone but topic words
can fool it: on the v1 sample files, "I feel overwhelmed trying to memorize dozens of
formulas" scores slightly positive. Treat `mood()` as a soft, relative signal. Affect is not used for ranking:
Phase 2 showed it carries no retrievable signal (see HISTORY.md). It sets
salience, so emotional memories survive capacity eviction longer, and feeds `mood()`.

## 5. Speed and size

SPEED_SECTION
