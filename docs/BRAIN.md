# What NCM takes from the brain, and what it does not

NCM is not a brain model. It borrows a few mechanisms from memory research where they
were measured to help, and this page records which ones, the evidence, and the ones that
were tried and dropped. Numbers come from `accuracy/` (LongMemEval-S, 204 questions about
the user) unless stated otherwise.

## The principle: four questions, four different signals

Every memory system has to decide four things, and the brain uses a different signal
for each. Mixing them up is what failed in NCM v1 (emotion as a ranking channel).

| decision | the brain's signal | NCM |
|---|---|---|
| what comes back for a cue | cue overlap, rare cues weighted most (the fan effect), temporal context | dense + BM25 (BM25's IDF has the shape of ACT-R's fan effect, `S - ln(fan)`), plus a lift for the turn before a hit |
| what is kept | rehearsal, age, emotion, self-relevance | ACT-R activation + emotional salience + self-reference |
| when a memory is out of date | reconsolidation, triggered by prediction error | dated, oldest-first context; no automatic updating (see "not built") |
| how memories are read | recency: the most recent context dominates recall (CMR) | oldest first, so the latest version of a fact is read last |

## Built (v2.1), with evidence

**Self-reference keeps what people say about themselves** (Rogers et al. 1977; Symons &
Johnson 1997). The turns people later ask about are more self-disclosing than other user
turns (AUC 0.68 for NCM's persona evidence score) and more emotional (AUC 0.61). Used for
*ranking* neither helps (recall@5 flat at 0.922 with any boost); used for *keeping* both
do. `Config.self_weight` (0.5) adds the persona evidence score to a memory's activation.

**Age from the newest memory, not the wall clock.** A bug, found while doing this:
for imported or replayed histories every memory looked equally old.

Effect, with a store capped at a quarter / half of each ~500-turn history
(`accuracy/bounded.py`; "answer in context" = the gold answer's words reach the prompt;
169 questions with short answers):

| forgetting | 25% kept | 50% kept | facts the user told it (25% / 50%) |
|---|---|---|---|
| v2.0 | 42.0% | 66.3% | 46.0% / 73.0% |
| age fix | 46.7% | 66.3% | 39.7% / 68.3% |
| **+ self-reference 0.5 (v2.1 default)** | **52.7%** | **70.4%** | 50.8% / 76.2% |
| + self-reference 1 | 55.0% | 69.2% | 55.6% / 81.0% |
| + self-reference 2 | 58.0% | 72.2% | 66.7% / 88.9% |
| no limit | 88.8% | 88.8% | 93.7% |

v2.1 against v2.0, same questions: 32 better / 14 worse at 25% (p = 0.011); 19 / 12 at
50% (p = 0.28, not significant). Answered by qwen2.5:3b from that memory and judged:
30.9% -> **43.1%** at 25% (44 better / 19 worse, p = 0.002; facts that changed 31.9% ->
62.5%), and a tie at 50% (59.3% / 58.8%). Stronger self-reference keeps more of the user's facts
but sacrifices what the assistant itself said (its own earlier words reach the prompt
for 55% of those questions at 50% with the age fix, 45% at 0.5, 23% at 2.0), so 0.5 is
the default.

## Tested and not built

| idea | source | result |
|---|---|---|
| gist before forgetting: a fading turn leaves its self-describing sentences behind, under its own vector, in a separate store of `max_items // 4` | complementary learning systems (McClelland, McNaughton & O'Reilly 1995); fuzzy-trace theory | with self-reference: 57.4% / 76.9% (25% / 50%). But plain memory given the same 25% more room does as well (55.6% / 76.3%; 4 better / 1 worse, 7 / 6): the gain was the extra store, not consolidation. Two earlier versions failed: gists inherited their turn's age and faded at once; then, counted as a replay, they crowded out full memories |
| spill-over: a turn next to an emotional one is kept with it | behavioural tagging (Moncada & Viola 2007) | no effect (+7/-4 at 25%, +3/-4 at 50%) |
| forget the same share of each speaker | source memory | a trade: the assistant's own words recover (23% -> 36% at 50%), the user's facts give back as much |
| stronger self-reference (2.0) | | more user facts (90% at 50%) but the assistant's own words drop from 55% to 23% |
| detect changed facts when they are written (similar sentence, different value) | reconsolidation by prediction error (Nader et al. 2000; Sinclair & Barense 2019) | whole turns: the old statement is the nearest match 39% of the time; sentences: 59%. A threshold catching half the true updates makes ~16 false links per history |
| the same, only against the persona statements (schema) | SLIMM (van Kesteren et al. 2012) | precise (1.7 links per history) but finds only 25% of real updates |
| mark the older of two similar statements "(updated later)" at recall | reconsolidation at retrieval | changed facts 70.8% -> 72.2% (2 better, 1 worse) with qwen2.5:3b: the reader ignores the marks |
| drop the older statement instead | retrieval-induced forgetting (Anderson, Bjork & Bjork 1994) | worse: changed facts -2.7, user facts -6 points; with a weak detector, suppression removes good memories |
| list the newest conversation first | | changed facts halve (70.8% -> 35%): readers trust what they read last (CMR recency) |
| boost emotional or self-disclosing memories in ranking | emotional tagging (McGaugh 2004) | no change: these signals say what matters, not what matches |

Updating facts that changed is the open problem. Frontier models get it wrong too: on
LongMemEval's knowledge-update questions GPT-4.1 and GPT-5.4 reach 91-92% with the full
conversation in the prompt ([Supersede, 2026](https://arxiv.org/html/2606.27472v1)).
Systems that handle it (Zep, Mem0) use an LLM to decide what changed, and rewriting
memories too often corrupts them ([2605.12978](https://www.alphaxiv.org/abs/2605.12978)).

## Not tried yet

- Linking memories formed within the same day across conversations (Cai et al., Nature
  2016). NCM only links neighbouring turns; symmetric spreading within a conversation was
  tested in v2 and helped one benchmark while hurting the other.
- An entity index for multi-hop questions (HippoRAG, Gutiérrez et al. 2024).
- An offline "sleep" pass that writes summaries with an LLM (LightMem, ICLR 2026). Keeping
  sentences did not beat simply keeping more memories (above).

## References

Anderson, Bjork & Bjork 1994 (retrieval-induced forgetting) · Cai et al. 2016, *Nature*
534 (memories linked across time) · Howard & Kahana 2002, *J. Math. Psych.* (temporal
context model) · Polyn, Norman & Kahana 2009 (CMR) · McClelland, McNaughton & O'Reilly
1995 (complementary learning systems) · McGaugh 2004 (emotional modulation of
consolidation) · Moncada & Viola 2007, *J. Neurosci.* (behavioural tagging) · Nader,
Schafe & LeDoux 2000 (reconsolidation) · Rogers, Kuiper & Kirker 1977 (self-reference
effect) · Sinclair & Barense 2019, *TINS* (prediction error and reactivation) · van
Kesteren et al. 2012, *TINS* (SLIMM) · Fountas et al. 2025, ICLR (EM-LLM) · Gutiérrez et
al. 2024, NeurIPS (HippoRAG) · LightMem, ICLR 2026 · Wu et al. 2025, ICLR
(LongMemEval).
