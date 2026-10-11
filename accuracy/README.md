# Does the character get it right about you?

The question this folder answers: when an AI character with NCM memory talks to someone
over weeks, and that person later asks about something they told it, how often is the
answer right? Not retrieval scores: the actual answer, graded.

Everything here is reproducible with `python -m accuracy.run`; every answer, every
grade and every number is in [`results/`](results/).

## Short version

* **The memory part works.** With the reranker on, NCM put the full evidence into the
  prompt for **100%** of the "facts the user told it" and "facts that changed" questions
  (136/136), and 97.9% of all answerable questions. Missing memories are no longer why
  answers go wrong.
* **The reader is the limit, and it is not 99%.** Given the *exact* conversations that
  hold the answer (the oracle), `qwen2.5:3b` is right 79.7% of the time and
  `qwen2.5:7b` 87.5%. With NCM's context instead they reach **79.4%** (3B, the same as the
  oracle) and **83.8%** (7B).
* **Facts the user told it:** 7B with NCM + reranker is right **90.6%** of the time,
  **93.8%** with the "second look" below. Facts that later changed are the hard part:
  71-76% even with the oracle, because small models often report the older value.
* **Against plain vector search** with the same reader: +4.9 points (3B) and +2.5 points
  (7B) for NCM + reranker, not statistically significant at 204 questions. NCM + reranker
  + second look beats vector search with the 3B by +6.9 points (p = 0.03).
* 99-100% on this test needs a much stronger reader than fits a 4 GB laptop GPU; the
  memory side is ready for it.

## The test

[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025). Each
question comes with its own chat history of ~50 sessions and ~500 turns (about 115k
tokens), spread over weeks, with the answer buried in one or two of them. We use the
question kinds that matter for a personal companion:

| kind | n | example |
|---|---|---|
| facts the user told it (single-session-user) | 64 | "What breed is my dog?" → Golden Retriever |
| facts that changed (knowledge-update) | 72 | "How many bikes do I own?" → 4 (it was 3 a few weeks earlier) |
| what the assistant itself said (single-session-assistant) | 56 | "What did you recommend for my back-end stack?" |
| never told it (abstention) | 12 | "What is my hamster's name?" → right answer: "you never told me" |

204 questions in all. Dates are real: each memory carries the date it was said, and the
question carries "today's date".

## What is compared

The same reader model answers every question, from a context built by one of these:

| memory | what the reader gets |
|---|---|
| oracle | the whole conversation(s) that contain the answer, as LongMemEval's own oracle setting. Not a real system: the ceiling for this reader |
| vector | plain vector search (MiniLM), the 10 most similar turns |
| ncm | `Memory.context(question)`: what you know about the user, then the 4 best memories with one turn either side, dated, oldest first |
| ncm-rerank | the same with `Memory(reranker=True)` (a 22M cross-encoder re-scores NCM's top 30) |
| ncm-top2 | `context(question, k=2, window=0)` with the reranker: only the two best memories |
| ncm-rerank + second look | an app-side cascade: if the reader answers "you never told me" (a fixed regex, `REFUSAL` in `run.py`), ask once more with ncm-top2. Built from the two arms' saved answers |

Readers: `qwen2.5:3b` and `qwen2.5:7b` (Ollama, Q4, temperature 0, 16k context) on a
laptop RTX 3050 with 4 GB.

## How answers are scored

* **Accuracy** (the headline): an LLM judge applies LongMemEval's grading rules (right if
  the reply contains the correct answer or an equivalent; for changed facts, the latest
  value; for never-told questions, saying it was not told). Answers are graded in batches
  of 100 by Gemini Flash models with reasoning on (the free tier allows ~20 calls a day
  per model, so five models shared the work: 2.5, 3.5, 3.7, 3.8 and 3-preview). The model
  that graded each answer is recorded in `judgments.jsonl`.
* **How far to trust the judges:** two other models each regraded a fixed random sample of
  100 answers. Gemini 3.5 Flash agreed with the original grades on 98 of 100 (Cohen's κ
  0.95); Gemini 3 Flash agreed on 97 (κ 0.91), and where they differed it was more
  lenient (79% right against 76%), so the accuracies here are, if anything, slightly low.
  An earlier judge without reasoning marked "a yellow dress and a pair of earrings to
  match" wrong for "a yellow dress"; it was discarded and everything regraded.
* **Token precision / recall / F1** (SQuAD-style, against the gold answer), as requested
  and recorded for every answer. They mostly measure answer *length*: a correct full
  sentence ("You graduated with a degree in Business Administration.") has precision 0.25
  against "Business Administration". Use accuracy for right-or-wrong; F1 is here for
  completeness.
* **Evidence in prompt:** did the context contain the full turn that holds the answer?
  This separates memory failures from reader failures.
* **Says not told:** how often the reader answered "you never told me".

Paired differences (bootstrap 95% CI and exact McNemar test, same questions) are in
`results/summary.json` under `vs_vector`, `vs_oracle` and `vs_ncm-rerank`.

## Results

Accuracy by reader and memory (204 questions; the oracle has no evidence to give for the
12 never-told questions, so it answers 192). 95% intervals are Wilson intervals.

| memory | qwen2.5:3b | qwen2.5:7b | evidence in prompt | prompt size |
|---|---|---|---|---|
| oracle (exact conversations; the ceiling) | 79.7% [73.4, 84.8] | 87.5% [82.1, 91.5] | 100% | 17,229 chars |
| **NCM + reranker** | **79.4%** [73.3, 84.4] | **83.8%** [78.2, 88.2] | 97.9% | 12,190 |
| NCM + reranker + second look | **81.4%** [75.5, 86.1] | 83.3% [77.6, 87.8] | 97.9% | 12,369 |
| NCM (default) | 76.5% [70.2, 81.8] | 82.8% [77.1, 87.4] | 94.8% | 11,573 |
| plain vector search, top 10 | 74.5% [68.1, 80.0] | 81.4% [75.5, 86.1] | 94.3% | 9,435 |
| NCM top 2 only | 73.5% | 78.9% | 87.0% | 1,649 |

By kind of question, NCM + reranker:

| reader | facts the user told it (64) | facts that changed (72) | its own earlier words (56) | never told it (12) |
|---|---|---|---|---|
| qwen2.5:3b | 79.7% | 70.8% | 91.1% | 75.0% |
| qwen2.5:3b, + second look | 85.9% | 72.2% | 91.1% | 66.7% |
| qwen2.5:7b | **90.6%** | 70.8% | 92.9% | 83.3% |
| qwen2.5:7b, + second look | **93.8%** | 72.2% | 92.9% | 50.0% |
| qwen2.5:7b, oracle (ceiling) | 90.6% | 76.4% | 98.2% | - |

Paired differences (same questions; bootstrap 95% CI, exact McNemar p):

| comparison | 3B | 7B |
|---|---|---|
| NCM + reranker vs vector search | +4.9 pts [-0.5, +10.8], p 0.13 | +2.5 [-2.9, +7.8], p 0.49 |
| NCM + reranker vs oracle | +0.0 [-5.7, +5.7], p 1.00 | -3.6 [-8.3, +0.5], p 0.17 |
| second look vs NCM + reranker | +2.0 [+0.0, +4.4], p 0.22 | -0.5 [-2.9, +2.0], p 1.00 |
| second look vs vector search | +6.9 [+1.5, +12.7], **p 0.03** | +2.0 [-3.9, +7.8], p 0.62 |

Token precision / recall / F1 for every row are in [`results/summary.md`](results/summary.md)
(for example 7B with NCM + reranker: P 0.153, R 0.815, F1 0.230). Recall is high because
the answers do contain the gold words; precision and F1 are low because the readers
answer in full sentences.

**Where the remaining errors come from** (NCM + reranker, evidence in the prompt):

1. *The older value of a changed fact.* "How many bikes do I own?" → "three" (the user
   later said four). Both statements are in the prompt, dated, oldest first; the reader
   picks the wrong one. Even the oracle loses 24-29% of these.
2. *"You never told me" when it did.* The 3B model says it was not told in 11% of
   answers, mostly when the fact sits inside a long conversation. The second look fixes
   part of this (3B: 85.9% on user facts) but, asked again with less context, the readers
   also invent answers to questions they were never told (never-told accuracy falls).
3. *A wrong number or detail* copied from a nearby line.

So for this use case: **turn the reranker on** (`Memory(reranker=True)`), use the
largest reader you can run, and keep the default oldest-first order. The second look is
worth it with a small reader if inventing an answer is less costly to you than refusing.

## What did not help

| idea | effect | why |
|---|---|---|
| "notes first" reader (copy the relevant memory lines, then answer; a light Chain-of-Note) | 3B: oracle 79.7% → 59.9%, NCM+rerank 79.4% → 66.7% | the 3B model copies the wrong lines or writes "none", then says it was never told |
| newest conversation first (same lines, reordered) | NCM+rerank, changed facts: 3B 70.8% → 34.7%, 7B 70.8% → 36.1%; overall 3B 79.4% → 66.7%, 7B 83.8% → 69.1% | the readers favour what they read *last*, closest to the question; oldest first puts the latest value there |
| cutting the turns around each hit to 300 characters, 6 hits instead of 4 | 3B, NCM+rerank: 79.4% → 72.5% | the cut turns were often the assistant's own answer, or context the reader needed |
| Gemini 2.5 Flash-Lite as a cloud reader | not run | the free tier allows 20 requests a day per model |

The dropped variants' answers are not in `results/`; the sample in
`judge-check-gemini-3.5-flash.jsonl` includes 19 answers from the cut-turns variant.

## Files

| file | what |
|---|---|
| `run.py` | the whole study: data, memories, contexts, readers, judge, summary |
| `results/answers-<reader>.jsonl` | every reply, with reader time and prompt size |
| `results/judgments.jsonl` | every grade, with the judge model |
| `results/judge-check-<model>.jsonl` | another model's regrade of 100 random answers |
| `results/summary.json`, `results/summary.md` | all numbers, including paired tests (`vs_vector`, `vs_oracle`, `vs_ncm-rerank`) |

```
python -m accuracy.run --readers qwen2.5:3b,qwen2.5:7b     # read, judge, summarise (resumes)
python -m accuracy.run --summarise                          # rebuild the tables only
python -m accuracy.run --summarise --judge-check 100 --check-with gemini-3.5-flash
```

Run time on the laptop: building all contexts ~4 min; reading ~45 min (3B) and ~3 h (7B,
which does not fit in 4 GB of VRAM and spills to the CPU); judging ~30 calls.

Needs the LongMemEval-S file (see `bench/data.py`), Ollama, and a Gemini key in `.env`
(`gemini_api=...`) for the judge.
