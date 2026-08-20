# EXP14 retraction record

Date: 2026-08-20

This file records the retraction of every previously published EXP14 number. It
is kept in the results directory so that anyone reading an old figure or an old
table can find out what happened to it. Nothing in this file is a new result.

## 1. What is retracted

### Tier 1: numbers published in the documentation that match no data file

`README.md` line 490 and `experiments/EXPERIMENT_RESULTS.md` lines 46 and 457
reported:

    Persona-B warm markers +3.833 and +63 words vs Persona-A under same prompts

Those two values do not come from any result file currently in the repository.
The result file that was on disk before this retraction recorded `warm_markers
+3.333` and `words +66.333`. The published `+3.833` and `+63.167` pair traces to
an earlier run whose data was overwritten when the experiment was regenerated on
approximately 2026-04-11. The documentation was never updated to follow.

So the published pair was, at the time of retraction, untraceable. It described a
run whose source data no longer exists.

### Tier 2: numbers in the result files, traceable but measured unsoundly

The pre-retraction `exp14_persona_memory_ollama.json`, `.txt` and `.md` recorded
the following, for `model qwen2:7B`, `prompt_count 6`, `top_k 4`:

| metric              | Persona A mean | Persona B mean | delta B minus A |
| ------------------- | -------------- | -------------- | --------------- |
| chars               | 1304.833       | 1654.500       | +349.667        |
| words               | 199.500        | 265.833        | +66.333         |
| exclamations        | 0.167          | 0.667          | +0.500          |
| questions           | 0.000          | 0.167          | +0.167          |
| analytical_markers  | 1.667          | 2.667          | +1.000          |
| warm_markers        | 1.333          | 4.667          | +3.333          |

These arithmetic means were recomputed from the six per-prompt records and agree
to better than 1e-9, so they are faithful summaries of what was generated. They
are retracted anyway, because the protocol that generated them cannot support
them. The defects are listed in section 2.

### Tier 3: the two figures

`exp14_persona_memory_ollama_summary.png` and
`exp14_persona_memory_ollama_prompt_deltas.png` plotted the Tier 2 means and
per-prompt deltas with no dispersion of any kind. They are deleted rather than
kept, because a bar chart of six unreplicated means reads as an effect size and
there was no basis for reading one. They are replaced by
`exp14_paired_differences.png` and `exp14_effect_sizes.png`, which show every
paired observation and a bootstrap confidence interval respectively.

## 2. Why the protocol could not support the numbers

Six independent defects, each sufficient on its own.

1. **The run is not reproducible.** Generation used `temperature 0.2` and
   `top_p 0.9` with no seed. Sampling at that temperature was measured on this
   machine to produce different output on repeat runs of an identical prompt
   (377 versus 394 words on two runs of one probe). Every reported difference
   therefore had an unknown and unmeasured amount of sampling noise in it.

2. **The model is not installed.** The run recorded `model qwen2:7B`.
   `qwen2:7b` is not present in the local Ollama instance. The installed models
   are `phi4-mini:latest`, `qwen3:1.7b`, `qwen2.5:0.5b`, `nomic-embed-text:latest`,
   `hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M` and
   `hf.co/LiquidAI/LFM2-700M-GGUF:Q4_K_M`. The recorded run cannot be repeated
   here at all, by anyone, with or without a seed.

3. **The analytical marker metric partly counted prompt echo.** The marker list
   contained the word `plan`. Prompt p2 was "Plan my day for study, exercise,
   and rest." A response that repeated the task back scored on the metric
   without exhibiting the style the metric was supposed to detect.

4. **The warm marker metric partly counted context echo.** The marker list
   contained `support` and `care`. The Persona B seed lines included "I
   communicate warmly with empathy and supportive tone" and "I keep responses
   human, caring, and conversational", and those lines were injected into the
   Persona B prompt as retrieved context. Copying an injected line therefore
   scored on the metric. Since `warm_markers +3.333` was the headline result and
   the confound applies asymmetrically to the arm that won, this defect points
   in the direction of the reported effect.

5. **Marker matching used raw substring counting.** `text.lower().count(m)`
   made `care` match inside `career` and `careful`, and counted `step` twice
   inside a single occurrence of `step-by-step`. Counts were inflated by an
   amount that depended on unrelated vocabulary.

6. **There was no statistical analysis.** Six prompts, one sample each, no
   dispersion, no confidence interval, no significance test, and eight reported
   differences with no correction for testing eight things at once. A difference
   of means over n equal to 6 with unmeasured sampling noise is not evidence of
   an effect.

Two further items are not defects in the result but were misleading as
documented. The 7-dimensional state vector `[0.55, 0.45, 0.50, 0.40, 0.60, 0.50,
0.55]` was hand-authored and was never disclosed as such. It is also identical
in both arms, so it could not have contributed to the contrast, and dimensions 6
and 7 are structurally ignored by the state term because
`MemoryStore._rebuild_cache` builds a 5-dimensional auto-state cache. Separately,
`s_current_normalized` was passed to `retrieve_top_k_fast`, which ignores that
argument and reads `store.auto_state.get_current_state()` instead, so that line
was dead code rather than a control.

## 3. What replaces it

`experiments/python/exp14_persona_memory_ollama.py` was rewritten. The
corrections map one to one onto the defects above:

1. Greedy decoding, `temperature 0.0`, `top_p 1.0`, `top_k 1`, fixed seed.
   Reproducibility is evidenced rather than asserted: the result file records a
   SHA-256 hash of every one of the 24 generations, and a rerun compares all 24
   against the previous run. The `validity.generations_reproduced` field carries
   the outcome, and the run exits non-zero if any generation differs.

   An earlier version of this gate instead reissued one request three times and
   required byte-identical output. That gate was measuring the wrong property
   and it failed for a reason unrelated to sampling. On this stack the first
   request for a given prompt differs from repeats of that same request, because
   a repeat reuses the cached prompt prefix and accumulates the prefill in a
   different order, which can resolve a near-tie between two candidate tokens
   the other way. `experiments/python/exp14_determinism_probe.py` separates the
   two properties and records the evidence in
   `exp14_determinism_probe.json`: three identical requests give characters
   `[1834, 2516, 2516]`, so repeat determinism does not hold, but running that
   same sequence twice from a forced model unload reproduces every position
   byte-identically, including reproducing the divergence itself, so history
   determinism does hold. Every measured generation in this experiment uses a
   distinct prompt, so none of them is ever in the repeat regime. History
   determinism is the property a rerun depends on, and it is the property now
   gated.
2. A preflight query against `/api/tags` verifies the model is installed and
   records its digest, parameter size and quantization. An absent model is a hard
   failure. `num_predict` is `-1` and `done_reason` is recorded per generation,
   because the earlier probes had been silently truncated by a 512 token cap and
   any length metric measured under that cap measures the cap.
3 and 4. `check_marker_disjoint` asserts at runtime that no marker word occurs in
   any prompt, in any persona seed line of either arm, or in the system prompt.
   The persona lines are written in paraphrase to satisfy this. Copying an
   injected line now scores exactly zero on both metrics, by construction.
5. Word-boundary regular expression matching. The primary metric per family is
   the number of distinct markers present rather than the total count, because a
   total can be moved by one repeated word.
6. 12 prompts balanced across 6 affective and 6 task register, analyzed as a
   paired design across prompts, reporting per-prompt differences, mean,
   standard deviation, Cohen dz, a seeded bootstrap 95 percent confidence
   interval, a Wilcoxon signed-rank test, and Holm-Bonferroni adjustment across
   the metric family. The two primary metrics are named in the script before the
   run so the analysis is not chosen after seeing the numbers.

The rewritten script also refuses to certify its own output: it writes a
`validity` block and exits non-zero if any generation was truncated or if the
determinism check failed.

## 4. Standing caveat on the replacement

The replacement run is 12 paired prompts, 5 persona lines per arm, one model, one
machine. It can support a statement about that model under that protocol. It
cannot support a general claim about language models, and it does not establish
that the effect scales.
