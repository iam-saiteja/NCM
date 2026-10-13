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
| contextual turn embeddings ("turn (in reply to: previous turn)"), same cost | +0.014 [0.004, 0.023] LoCoMo, -0.008 [-0.019, 0.003] LongMemEval |
| session-diverse top-k (penalise a second hit from the same session) | -0.015 / -0.024 LoCoMo (weight 0.3 / 0.6), +-0.003 LongMemEval |
| embed without the "speaker:" prefix | LoCoMo dense R@5 0.360 -> 0.216 |
| 128-d random projection (v1) | part of v1's loss; v2 keeps the full vector |

Tuning protocol, stated plainly: the lexical weight and spreading grid were first
tuned on LoCoMo with LongMemEval held out. The LoCoMo-tuned spreading default
failed on LongMemEval (above), so the final default was chosen with both datasets
in view. It is three numbers and moves both datasets the same way, but no clean
held-out set remains for it.

### Embedding model

| R@5 | LoCoMo cosine | LoCoMo NCM v2 | LongMemEval cosine | LongMemEval NCM v2 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (default; 22M parameters) | 0.360 | 0.555 | 0.575 | 0.735 |
| bge-small-en-v1.5 (33M; ~1.8x the encoding cost) | 0.478 | 0.565 | 0.706 | 0.759 |

A stronger encoder lifts plain vector search a lot (+0.12 / +0.13), but once BM25 is
fused in the gap nearly closes (+0.010 / +0.024). MiniLM stays the default for its
cost; `Memory(embedder="BAAI/bge-small-en-v1.5")` buys the extra 1-2 points. The
backward lift helps with bge too (LongMemEval R@5 +0.016, MRR +0.027).

## 2. End-to-end: do the answers get better?

> Runtime note, found after these runs: the Ollama install they used had
> `OLLAMA_KV_CACHE_TYPE=q4_0`. A 4-bit KV cache corrupts qwen2.5-3B's output (doubled
> syllables in ~5% of answers, broken JSON), which lowers every arm's F1. The clean,
> larger head-to-head in [comparisons/](../comparisons/README.md) supersedes the absolute
> numbers below; the ordering v1 < plain vector search < NCM v2 holds there as well.

A local model reads each arm's memories and answers; nothing else differs between
arms. Reader: qwen2.5:3b (Q4, Ollama, temperature 0). Metric: SQuAD-style token F1
against the gold answer, the LoCoMo paper's metric. Every arm shows 10 memories,
dated and in time order (`context()` shows 4 hits plus their neighbouring turns, a
similar budget). Raw answers: `bench/results/qa-cache.jsonl`.

| answer F1 | LoCoMo (400 q) | vs v1 | LongMemEval (200 q) | vs v1 |
|---|---|---|---|---|
| no memory | 0.043 | | 0.095 | |
| NCM v1 | 0.184 | | 0.217 | |
| cosine (MiniLM) | 0.214 | +0.030 [-0.000, 0.061] | 0.224 | +0.007 [-0.045, 0.059] |
| NCM v2 `recall()` | 0.273 | **+0.089** [0.055, 0.122] | 0.271 | +0.054 [-0.001, 0.108] |
| **NCM v2 `context()`** | **0.289** | **+0.105** [0.069, 0.140] | **0.293** | **+0.076** [0.021, 0.134] |
| NCM v2 + reranker | 0.273 | +0.089 [0.057, 0.121] | 0.280 | +0.063 [0.010, 0.117] |
| NCM v2, dialogue mode (`lift_next=0.75`) | **0.305** | +0.121 [0.087, 0.154] | | |

- On both datasets **NCM v1 gave the reader worse memories than plain vector search**,
  and v2 gives it better ones. `context()`, the call an integration would use, is the
  best default arm on both: +57% (LoCoMo) and +35% (LongMemEval) relative to v1.
- **The reranker does not change answers at this budget.** It reorders the top of the
  list, but `context()` shows memories in time order and 10 of them, so the reader
  cannot see the better order and the set barely changes. With a tight budget it does help: at 3 memories per question, LoCoMo F1 is
  0.278 with the reranker against 0.241 without (+0.037 [0.005, 0.069]). That is the
  same answer quality as the default with 10 memories (0.273) from a prompt 2.7x
  smaller (736 vs 1,959 characters): the reranker spends ~30 ms of GPU to save LLM
  tokens. At k=3, v1 scores 0.135 and cosine 0.172 (`qa-locomo-qwen2.5_3b-k3.json`).
- **Dialogue mode** (forward lift) is the best arm on LoCoMo, a dialogue between two
  people, and is off by default only because it hurts user/assistant chat retrieval
  (section 1).
- Earlier `context()` runs printed ISO dates (`2023-05-07`); the reader copied them
  into answers and token F1 scored them zero against "7 May 2023" (temporal F1 0.135
  vs 0.254). Dates now read "7 May 2023, 13:56", which fixed it. Those runs remain in
  the cache under `hybrid-context-*-isodate`.
- LongMemEval runs skip single-session-preference (rubric answers) and clip every
  turn to 400 characters in every arm, so a 3B model can read ten of them.
- The local runtime (Ollama, qwen2.5:3b Q4) occasionally stutters inside a word
  ("Joleneene", "Paintinging"): 4-6% of answers in every arm, which lowers F1 for
  all arms alike. Comparisons stand; absolute F1 is a little pessimistic.

## 3. Personalisation: does it know who the person is?

The question a character has to answer: after many conversations, who is this
person? Data: Multi-Session Chat (Xu et al. 2022), test split, 120 conversations of
5 sessions each. Both speakers' human-written persona lines are the ground truth;
no memory arm ever sees them. For each speaker, two true persona lines are paired
with two distractors: persona lines of other people, kept only if no true line is a
near-paraphrase (cosine < 0.6). The reader (qwen2.5:3b) is asked which of the two
statements is true of the speaker, A or B, with positions shuffled. 480 pairs; chance
is 50%.

| context given to the model | accuracy | vs last-10-turns window |
|---|---|---|
| no memory | 0.477 | |
| the last 10 turns (what a sliding-window chatbot keeps) | 0.504 | |
| plain vector search: 6 memories of that speaker by cosine | 0.762 | +0.258 [0.202, 0.317] |
| NCM `persona()` only: 6 self-describing statements, no query | 0.621 | +0.117 [0.056, 0.177] |
| NCM `recall()`, 6 memories of that speaker | 0.785 | +0.281 [0.221, 0.340] |
| **NCM `context()`**: persona + recalled memories in context | **0.790** | **+0.285** [0.229, 0.340] |

A sliding window is at chance on who the person is: what they said three sessions
ago is gone. Long-term memory gets four answers in five right. On this task NCM's
ranking is level with plain vector search (`context()` vs cosine +0.027
[-0.023, 0.079], not significant): the win here is remembering at all, and NCM's
retrieval advantage shows on the conversation benchmarks above. The query-free `persona()` block,
six lines that are always in the prompt, already recovers part of that (0.62).

**Re-run on a clean runtime.** The runs above used an Ollama install with a 4-bit KV
cache that corrupts this model's output. Re-run with a full-precision cache (`--run f16`,
same 480 pairs):

| what the model is given | accuracy | vs last-10-turns window |
|---|---|---|
| no memory | 0.529 | |
| the last 10 turns | 0.588 | |
| NCM `persona()` only | 0.656 | +0.069 [0.021, 0.117] |
| plain vector search | 0.867 | +0.279 [0.233, 0.325] |
| NCM `recall()` | 0.869 | +0.281 [0.235, 0.327] |
| **NCM `context()`** | **0.900** | **+0.312** [0.267, 0.358] |

Every arm scores higher, and one conclusion changes. NCM's retrieval alone still ties
plain vector search (`recall()` +0.002 [-0.023, +0.027], McNemar p 1.0). But `context()`,
which adds the persona block, beats vector search by +0.033 [+0.004, +0.062] (bootstrap
p 0.029, exact McNemar p 0.036: 34 pairs only NCM got right, 18 only vector search).
`context()` also sends more text (2,525 characters against 996), so a control gives
vector search the same budget (15 memories, 2,185 characters): it scores 0.858, no
better than with 6. Against that control `context()` is ahead by +0.042 [+0.013, +0.071]
(p 0.005, exact McNemar p 0.008: 36 pairs against 16), which survives a Holm correction.
The persona block, not the extra text, is what helps here. Results:
`bench/results/persona-choice-f16-msc-qwen2.5_3b.json`.

An earlier yes/no version of this test ("is this statement true of them?") was
uninformative: the 3B model answered "no" to 94-100% of statements in every arm, so
accuracy sat at chance. Its true-positive rates still ordered the arms the same way
(window 0.013, NCM recall 0.102, NCM context 0.117); results are kept in
`bench/results/persona-yesno-msc-qwen2.5_3b.json`.

## 3b. Character consistency: does the AI stay consistent with itself?

Section 3 asks whether the model knows *the user*. This asks whether an AI character
knows *itself*: its own facts, and what it said before. Speech style and "voice" are not
measured; that needs a strong judge model or people.

**Its own facts, asked in character** (`bench/character.py`). Multi-Session Chat, 120
conversations; Speaker 2 plays the character, whose card gives only its name, so
everything it knows about itself comes from memory. Its own persona lines become
in-character questions ("I own a Jeep" becomes "What car do you drive?", gold "Jeep";
drafted by the local LLM with few-shot examples, kept only when the answer is copied
from the fact and absent from the question): 258 questions. The character answers in
its own words; a reply counts as right if it contains the fact.

| what the character has | right about itself | says it does not remember | other (wrong or vague) | prompt |
|---|---|---|---|---|
| no memory | 9.3% | 12.0% | 78.7% | 40 chars |
| the last 10 turns | 19.4% | 3.5% | 77.1% | 1,698 |
| plain vector search, 6 memories | 41.5% | 2.3% | 56.2% | 772 |
| plain vector search, 18 memories (budget control) | 46.5% | | | 2,243 |
| NCM `recall()`, 6 memories | 42.6% | 1.2% | 56.2% | 775 |
| NCM `context()` (self-profile + its own lines as "you") | **48.1%** | 0.8% | 51.2% | 2,214 |

- Memory is what keeps a character consistent about itself: 9% without, 42-48% with.
- `context()` beats plain vector search with 6 memories by +0.066 [+0.016, +0.120]
  (McNemar p 0.024), but **not** vector search given the same budget: +0.016
  [-0.039, +0.070], p 0.64. Here NCM's advantage is the extra context, not the method.
- Even at best a 3B character is right about itself less than half the time, and it
  almost never says it does not remember: it fills gaps by inventing. With no memory,
  79% of its answers are made up.

**What it said before** (LongMemEval, the 56 "single-session-assistant" questions, which
ask about the assistant's own earlier replies; same reader, clean runtime, answer F1):

| memory | F1 | vs vector search |
|---|---|---|
| none | 0.151 | |
| NCM v1 | 0.375 | -0.057 [-0.119, -0.003] |
| plain vector search (10 memories) | 0.432 | |
| NCM v2 `recall()` | 0.404 | -0.028 [-0.091, +0.025] |
| NCM v2 `context()` | 0.415 | -0.017 [-0.079, +0.046] |
| NCM v2 `context(self_name="assistant")` | 0.413 | -0.019 [-0.085, +0.046] |

NCM v2 and vector search tie on remembering the assistant's own statements.

**Summary.** NCM gives a character a much better memory of itself than no memory or a
recent-turns window, and a better grasp of *the user* than plain vector search. It does
not make the character more consistent about *itself* than vector search with the same
prompt size, and a small model still confabulates about half the time.

## 3c. Answer accuracy about the user: how often is the character right?

Full study, code and every graded answer: [accuracy/](../accuracy/README.md).

LongMemEval-S, 204 questions about the user's own past conversations (~500 turns of
history each): facts the user told it (64), facts that later changed (72), what the
assistant itself said (56), and things the user never said (12). Answers graded by a
Gemini Flash judge with LongMemEval's rules (two other models regraded 100 answers each:
97-98% agreement).

| memory | qwen2.5:3b | qwen2.5:7b | evidence in prompt |
|---|---|---|---|
| oracle: the exact conversations (ceiling) | 79.7% | 87.5% | 100% |
| **NCM + reranker** | **79.4%** | **83.8%** | 97.9% |
| NCM default | 76.5% | 82.8% | 94.8% |
| plain vector search (10) | 74.5% | 81.4% | 94.3% |

- With the reranker, the evidence was in the prompt for 100% of user-fact and
  changed-fact questions; NCM + reranker equals the oracle with the 3B reader.
- 7B with NCM + reranker: 90.6% on facts the user told it, 70.8% on facts that changed
  (the oracle gets 76.4%: readers often report the older value).
- NCM + reranker vs vector search: +4.9 (3B) and +2.5 (7B) points, not significant
  (p 0.13, 0.49). A "second look" (re-ask with the two best memories when the reader says
  it was never told) lifts the 3B to 81.4%, +6.9 over vector search (p 0.03).
- Tried and dropped: a notes-first reader (3B 79.4% → 66.7%), listing the newest
  conversation first (changed facts 70.8% → 35%), cutting neighbouring turns (→ 72.5%).

## 3d. When memory is limited: what does forgetting keep?

Full study: [accuracy/README.md](../accuracy/README.md#when-the-character-cannot-keep-everything);
design and rejected ideas: [BRAIN.md](BRAIN.md). Each LongMemEval history (~500 turns) is
replayed into a store capped at 25% or 50% of its size.

| forgetting | answer in prompt, 25% / 50% kept | qwen2.5:3b correct, 25% / 50% kept |
|---|---|---|
| 2.0 | 42.0% / 66.3% | 30.9% / 59.3% |
| 2.1: age from the newest memory + self-reference 0.5 | 52.7% / 70.4% | **43.1%** / 58.8% |

2.1 against 2.0 at 25%: 44 questions better, 19 worse (exact McNemar p = 0.002);
facts that changed 31.9% -> 62.5%. At 50% they tie (30 / 31). Tested and not built:
gist before forgetting (it equalled simply keeping 25% more memories), spill-over from
emotional neighbours (no effect), per-speaker quotas (a trade), change detection at
write time (too noisy without an LLM), suppressing older look-alike statements (worse).

## 4. Affect

The valence/arousal probe is a 384x2 ridge regression on MiniLM embeddings. Valence
is fitted on EmoBank (Buechel and Hahn 2017; ratings of news, fiction and letters) plus
GoEmotions (Demszky et al. 2020; Reddit comments, using the dataset's own
positive / negative / neutral grouping, centred), each corpus weighted equally;
arousal on EmoBank alone. The ridge penalty was picked on both dev splits. Test splits:

| estimator | EmoBank valence r | EmoBank arousal r | GoEmotions AUC (positive vs negative) |
|---|---|---|---|
| v1 `AutoStateTracker` anchor words | 0.42 | 0.10 | - |
| v2 anchor sentences (fallback for non-MiniLM embedders) | 0.45 | 0.16 | 0.855 |
| v2 probe, EmoBank only | 0.62 | 0.44 | 0.884 |
| **v2 probe, EmoBank + GoEmotions (shipped)** | **0.62** | **0.44** | **0.906** |

Embedding with the store's "user: " prefix changes EmoBank r by under 0.01
(0.626 / 0.434). Zero means "emotionally average": the shipped probe puts average
EmoBank text at +0.01 and neutral Reddit comments at -0.17, inside the +-0.4 band
that `mood()` labels neutral.
v1's arousal dimension was essentially noise.

**Does it read the feeling or the topic?** A controlled test (`bench/affect_topic.py`):
320 sentences crossing a feeling with a topic, such as "I was thrilled about the
funeral" or "I was miserable about my wedding". A dedicated sentiment classifier
(DistilBERT fine-tuned on SST-2, 66M parameters) is the reference.

| | NCM probe | DistilBERT SST-2 |
|---|---|---|
| sign right when feeling and topic agree | 100% | 100% |
| sign right when they conflict | 88.8% | 100% |
| standardised weight of the feeling | 0.86 | 1.00 |
| standardised weight of the topic | 0.36 | 0.00 |

The probe follows the feeling, but about a third of its signal comes from topic words,
so roughly one conflicting sentence in nine gets the wrong sign ("thrilled about the
funeral" scores +0.34 against +0.97 for "thrilled about my wedding"). A dedicated
classifier would fix that at the cost of a 66M-parameter model run per memory; the probe
reuses the embedding NCM already has. Treat `mood()` as a soft, relative signal. Affect is not used for ranking:
Phase 2 showed it carries no retrievable signal (see HISTORY.md). It sets
salience, so emotional memories survive capacity eviction longer, and feeds `mood()`.

### Does the mood signal change what the character says?

Mood never enters retrieval; it reaches the model only as text (`context()`'s "Their mood
lately: ..." line and `examples/chat.py`'s "The conversation lately has felt {tone}").
`bench/mood_effect.py` tests whether that text matters. 80 users write five messages of one
clear mood (single-label GoEmotions comments: 40 sad/angry/afraid, 40 joyful/excited/grateful),
then ask something neutral ("What should I cook tonight?"). The character (the shipped
persona and rules) answers three times, same memories, temperature 0: without the mood
lines, with NCM's label, and with the opposite label. One judge (Gemini 2.5 Flash), blind
to the version, rates how well each reply's tone fits the user's recent mood, 1-5.

**NCM's reading:** right for 49 of 80 users (28 of 40 negative, 21 of 40 positive); the rest
it calls neutral. It never read a sad user as happy or the reverse.

| reply tone fit (1-5) | no mood line | NCM's label | opposite label |
|---|---|---|---|
| qwen2.5:7b, all 80 | 3.81 | **4.09** (+0.28, p 0.03) | 3.61 (-0.20) |
| qwen2.5:7b, the 49 NCM read right | 3.84 | **4.18** | 3.45 |
| qwen2.5:7b, sad/angry users | 3.38 | **3.85** | 3.00 |
| qwen2.5:3b, all 80 | 3.35 | 3.45 (+0.10, p 0.32) | 3.26 |

- The line is causal for the 7B model: the right label makes replies fit the user's mood
  better, the wrong label makes them fit worse. The gain is largest for upset users.
- The 3B model barely uses it (differences within noise).
- 94-96% of replies change wording when the line is added, but most changes are small;
  what matters is the judged tone above.

Results: `bench/results/mood-effect-{replies.jsonl,grades.json,summary.json}`.

## 5. Speed and size

All on a laptop CPU (no GPU involved in NCM itself), store of synthetic Zipf-text
memories with random 384-d vectors, so these numbers are NCM's own cost; embedding is
listed separately below. `bench/results/speed.json`.

| memories | v1 recall p50 | **v2 recall p50** | v2 recall p95 | v2 add, per memory | v2 save / load | file |
|---|---|---|---|---|---|---|
| 1,000 | 0.92 ms | **0.56 ms** | 4.1 ms | 0.14 ms | 0.12 s / 0.05 s | 1.1 MB |
| 10,000 | 5.6 ms | **1.3 ms** | 6.0 ms | 0.15 ms | 0.40 s / 0.14 s | 9.4 MB |
| 100,000 | 49 ms | **11.6 ms** | 18.7 ms | 0.16 ms | 2.3 s / 0.54 s | 91 MB |
| 1,000,000 | - | 117 ms | 139 ms | 0.14 ms | 24 s / 6.0 s | 907 MB |

v2 is 1.6-4.3x faster to recall than v1 while also running BM25. A v2 add costs
0.14 ms on top of embedding (indexing, affect, persona evidence); a v1 add embedded
the text twice (once for the memory, once more for its auto-state tracker). Memory
is ~1.5 KB per memory in RAM for the vectors, plus text.

Model costs (single query, p50; batched per item in brackets):

| model | CPU | GPU (RTX 3050 laptop) |
|---|---|---|
| all-MiniLM-L6-v2 (default embedder) | 21 ms (5.9 ms) | 16 ms (0.57 ms) |
| bge-small-en-v1.5 | 38 ms (11.8 ms) | 26 ms (1.1 ms) |
| reranker, 30 pairs (accuracy mode) | 229 ms | 31 ms |

So a recall over 10,000 memories costs ~23 ms on a laptop CPU, embedding included,
and accuracy mode adds ~30 ms on a GPU or ~230 ms on a CPU.
