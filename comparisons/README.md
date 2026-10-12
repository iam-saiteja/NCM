# How NCM compares to other memory systems

Every common way of giving an AI long-term memory, run head to head on the same
conversations, the same questions, the same local LLM and the same laptop. The point
is the trade-off: accuracy next to what each approach costs to write, to read, to run
and to adopt. All code is in this folder; all raw outputs are in `results/`.

**Short version.** On these tests NCM v2 gave the most accurate answers and was among
the cheapest to run: no LLM calls to store memories, ~6 ms to store one, ~8 ms to fetch
context, ~500-token prompts. Its lead over a well-chosen vector RAG (bge-base) is small
and not statistically significant here. Its lead over full-context prompting, hybrid
search, Mem0 on a local LLM, and summary memory is significant. A modern learned
keyword model (SPLADE) beats NCM's default retrieval on one of the two datasets.

## 1. Answers: same reader, different memory

LoCoMo conversations conv-26 and conv-30: 38 sessions, 788 turns, 230 questions
(single-hop, multi-hop, temporal, open-domain). Each system stores the conversation,
then supplies context for each question; qwen2.5:3b (Ollama, temperature 0) answers.
Score: token F1 against the gold answer. "vs NCM context()" is the paired difference
with its 95% bootstrap interval (positive = better than NCM context()).

| system | answer F1 | vs NCM context() | write cost per memory | LLM calls to write 788 turns | fetch context (p50) | prompt to reader (≈tokens) | reader time (p50) | storage | needs | effort |
|---|---|---|---|---|---|---|---|---|---|---|
| NCM v2 + reranker (accuracy mode) | **0.436** | +0.030 [-0.010, +0.071] | 5.2 ms | 0 | 187 ms | 507 | 0.30 s | 784 KB | + 22M reranker | low: one flag |
| NCM v2 · recall() | **0.417** | +0.011 [-0.026, +0.048] | 5.5 ms | 0 | 8 ms | 473 | 0.30 s | 784 KB | numpy + embedding model, one file | low: pip install, 3 lines |
| NCM v2 · context() | **0.406** |  | 5.3 ms | 0 | 7 ms | 490 | 0.32 s | 784 KB | numpy + embedding model, one file | low: pip install, 3 lines |
| vector RAG · bge-base (110M) | **0.395** | -0.011 [-0.056, +0.034] | 20.8 ms | 0 | 33 ms | 429 | 0.28 s | 2.3 MB | bigger embedding model + vector DB | low |
| retrieve & rerank · MiniLM → bge-reranker-base | **0.378** | -0.028 [-0.069, +0.013] | 3.6 ms | 0 | 1166 ms | 460 | 0.30 s | 1.2 MB | vector DB + 278M reranker | medium |
| hybrid RRF · BM25 + MiniLM | **0.359** | -0.047 [-0.090, -0.004] | 4.4 ms | 0 | 8 ms | 384 | 0.28 s | 1.2 MB | search engine with hybrid mode (Elasticsearch, Weaviate, Qdrant) | medium |
| full context (whole history in the prompt) | **0.344** | -0.062 [-0.107, -0.019] | 0.0 ms | 0 | 1 ms | 17,675 | 1.22 s | 119 KB | a long-context LLM; cost grows with history | trivial |
| vector RAG · MiniLM-L6 (22M) | **0.337** | -0.069 [-0.116, -0.024] | 3.4 ms | 0 | 7 ms | 392 | 0.28 s | 1.2 MB | embedding model + vector DB | low |
| Mem0 (open-source, local LLM) | **0.291** | -0.115 [-0.170, -0.060] | 4.2 s | 395 (≈3.7M tokens in) | 57 ms | 360 | 0.26 s | 5.7 MB | LLM at every write + Qdrant + SQLite | medium |
| NCM v1 | **0.286** | -0.120 [-0.164, -0.076] | ≈7.5 ms (2 embeddings) | 0 | 0 ms | 403 | 0.29 s | - | numpy (old API) | low |
| summary memory (LLM summary per session) | **0.280** | -0.126 [-0.178, -0.073] | 190.4 ms | 38 (≈31k tokens in) | 0 ms | 2,817 | 0.18 s | 21 KB | an LLM call per session | low |
| sliding window (last 20 turns) | **0.072** | -0.334 [-0.384, -0.285] | 0.0 ms | 0 | 0 ms | 843 | 0.12 s | - | nothing | trivial |
| no memory | **0.046** | -0.360 [-0.411, -0.310] | 0.0 ms | 0 | 0 ms | 18 | 0.11 s | - | nothing | - |

Timings: embedding models and rerankers on a laptop CPU, LLM work on an RTX 3050 (4 GB).
Reader time is the LLM's answer time, re-measured over `127.0.0.1` on 40 sampled
questions per system (`retime.py`): the first recording reached Ollama at `localhost`,
which on Windows tries IPv6 first and waited ~2.1 s per request, inflating every
system's reader time equally without affecting any answer.
"Storage" is what each system keeps for these two conversations: NCM's `.ncm` file (text,
fp16 vectors, BM25 index), fp32 vectors for vector RAG, Mem0's Qdrant + SQLite folders.

What the table says:

- **Memory beats no memory by a mile, and a sliding window is nearly useless** for
  questions about earlier sessions (0.072).
- **Full context is not the ceiling for a small model.** Putting all 13-18k tokens of
  history in the prompt scored 0.344 and cost ~36x the prompt tokens of retrieval and
  4x the reader time (1.2 s against 0.3 s). It
  also stops working once the history outgrows the context window: LongMemEval's
  histories (~115k tokens) cannot be run this way at all.
- **LLM-based memory is expensive to write and was not more accurate here.** Mem0
  spent 395 LLM calls and ≈3.7M prompt tokens (its extraction prompt is ~8.5k tokens per
  write) to store 788 turns, about 4.2 s per turn, and 9 of its writes failed to parse.
  With a 3B local model it answered worse than plain vector RAG. With GPT-4o-mini the
  Mem0 paper reports far better numbers (section 3); that is exactly the trade-off:
  LLM memory buys quality with a strong, paid model at every write.
- **The standard industry upgrades help but cost more.** A 5x larger embedder
  (bge-base) is the strongest non-NCM option at 6-7x the encoding cost; a 278M reranker adds
  1.2 s per query on CPU; Elasticsearch-style hybrid (reciprocal rank fusion) helped
  less than you might expect.
- **NCM v2 is in the top group at the bottom of the cost range.** Its reranker mode is
  the most accurate system tested, for ~0.2 s extra per query.

Run-to-run noise: two copies of this run overlapped for a while and sent concurrent
requests to Ollama; 269 of the answers they both produced differed, and system scores
moved by up to 0.002. Treat differences inside the confidence intervals as ties.

## 2. Retrieval: is the right memory found?

Each method ranks every turn of a conversation for each question; a hit is a turn the
benchmark marks as evidence. LoCoMo: all 1,531 questions. LongMemEval-S: the first 150
histories (139 answerable questions, ~500 turns each). Model times are per memory or
per query on a laptop CPU.

| method | LoCoMo R@5 | LoCoMo MRR | LongMemEval R@5 | LongMemEval MRR | model time to store a memory (CPU) | model time per query (CPU) |
|---|---|---|---|---|---|---|
| NCM v2 + reranker (ms-marco-MiniLM, 22M) | **0.657** | 0.618 | **0.768** | 0.765 | 3.8 ms | 255 ms |
| NCM v2 + bge-reranker-base | **0.643** | 0.593 | **0.798** | 0.803 | 3.8 ms | 1686 ms |
| SPLADE (learned sparse, 110M) | **0.617** | 0.553 | **0.682** | 0.625 | 38.8 ms | 92 ms |
| NCM v2 · SPLADE + BM25 | **0.588** | 0.502 | **0.704** | 0.683 | 42.6 ms | 101 ms |
| NCM v2 · bge-base embedder | **0.567** | 0.475 | **0.731** | 0.714 | 27.2 ms | 40 ms |
| NCM v2 · bge-small embedder | **0.565** | 0.477 | **0.734** | 0.739 | 8.7 ms | 15 ms |
| NCM v2 | **0.555** | 0.461 | **0.696** | 0.700 | 3.8 ms | 8 ms |
| NCM v2 · SPLADE instead of BM25 | **0.553** | 0.468 | **0.715** | 0.687 | 42.6 ms | 101 ms |
| retrieve & rerank · MiniLM top-30 → bge-reranker-base | **0.518** | 0.496 | **0.765** | 0.792 | 3.8 ms | 1686 ms |
| BM25 | **0.506** | 0.429 | **0.622** | 0.616 | none (tokenise only) | none |
| vector RAG · bge-small (33M) | **0.478** | 0.400 | **0.709** | 0.662 | 8.7 ms | 15 ms |
| vector RAG · bge-base (110M) | **0.471** | 0.385 | **0.715** | 0.643 | 27.2 ms | 40 ms |
| hybrid RRF · BM25 + MiniLM | **0.462** | 0.383 | **0.657** | 0.624 | 3.8 ms | 8 ms |
| vector RAG · MiniLM-L6 (22M) | **0.360** | 0.291 | **0.564** | 0.526 | 3.8 ms | 8 ms |
| NCM v1 | **0.278** | 0.226 | **0.426** | 0.408 | 7.5 ms | 8 ms |

What the table says:

- **With the same embedding model, NCM beats plain vector search on both datasets:**
  bge-small +0.087 (LoCoMo) and +0.025 (LongMemEval) R@5; bge-base +0.096 and +0.016.
  NCM adds BM25 fusion and contiguity on top of whatever embedder you choose; its
  default (MiniLM) is the cheapest one, and a bigger one is a constructor argument.
- **"Isn't BM25 old?" Yes, and it still matters.** BM25 alone beats MiniLM vector search
  by +0.15 on LoCoMo. Mem0 2.x, Elasticsearch, Weaviate and Vespa all keep it for the
  same reason: embeddings blur names and numbers.
- **SPLADE, the learned successor to BM25, is genuinely better on LoCoMo** (0.617, the best
  first-stage method there) but not on LongMemEval (0.682), and it costs a 110M-parameter
  pass per memory (~39 ms on CPU) against BM25's nothing. Inside NCM it adds +0.033 /
  +0.008 (as an extra signal) at that cost. Not the default; worth it if write cost is no
  object.
- **Reciprocal rank fusion, the usual hybrid-search default, underperforms.** Fusing the
  same two signals (MiniLM + BM25), it scores below NCM's z-score fusion on both datasets
  (0.462 vs 0.555; 0.657 vs 0.696), and on LoCoMo below BM25 alone.
- **Rerankers are the biggest single accuracy lever** and the most expensive per query.
  NCM's default reranker (22M) gets most of the gain for 15% of bge-reranker-base's cost.

## 3. Published numbers from the vendors (not measured here)

These come from papers and vendor posts. They use GPT-4o-mini or stronger as the reader
and an LLM judge (J score), so they are **not comparable** with the measurements above:
different model, judge, prompts and question filtering. Numbers for the same system
also disagree between sources.

LoCoMo, from the Mem0 paper (Chhikara et al. 2025, Table 2), GPT-4o-mini reader and judge:

| system | J score | memory tokens per query | search p95 | total p95 |
|---|---|---|---|---|
| full context | 72.9% | 26,031 | - | 17.1 s |
| Mem0 with graph (Mem0g) | 68.4% | 3,616 | 0.66 s | 2.6 s |
| Mem0 | 66.9% | 1,764 | 0.20 s | 1.4 s |
| Zep | 66.0% | 3,911 | 0.78 s | 2.9 s |
| best RAG (2 chunks of 256 tokens) | 61.0% | 512 | 0.70 s | 1.9 s |
| LangMem | 58.1% | 127 | 59.8 s | 60.4 s |
| OpenAI ChatGPT memory | 52.9% | 4,437 | - | 0.9 s |
| A-Mem | 48.4% | 2,520 | 1.49 s | 4.4 s |

Disputes and later claims:
- Zep's own corrected run reports **75.1%** J with a p95 search latency of 0.63 s, and
  argues the Mem0 paper misconfigured Zep (both speakers as one user, timestamps in the
  text, sequential searches). Zep's newer posts claim ~80%.
- Zep also points out that LoCoMo conversations (16-26k tokens) fit in modern context
  windows, which is why full context beats every memory system there with a strong LLM.
  With the 3B reader used here it did not (section 1).
- Vendor pages now claim 90%+ on LoCoMo and LongMemEval (Mem0, Zep, Supermemory, Letta);
  these are self-reported with each vendor's own model stack.

Not run here: Zep/Graphiti (needs Neo4j and an LLM per write), Letta/MemGPT (an agent
runtime that edits its own memory with LLM calls), LangMem, A-Mem and ChatGPT memory.
All of them, like Mem0, spend LLM calls at write time; section 1's Mem0 row shows what
that costs on local hardware.

Sources: [Mem0 paper, arXiv 2504.19413](https://arxiv.org/abs/2504.19413) ·
[Zep: "Lies, damn lies, & statistics"](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) ·
[Memori LoCoMo results](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) ·
[Atlan: agent memory frameworks 2026](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)

## 3b. NCM on all of LoCoMo with a large model (measured here)

`locomo_bedrock.py`: every LoCoMo question of categories 1-4 that has evidence (1,531), all
10 conversations. NCM as in section 1's best setup (`Memory(reranker=True)`, the 10 best
memories, dated, oldest first). One model, Mistral Large 3 (675B, on Amazon Bedrock,
temperature 0), answers in a short phrase and then grades its own answer with the usual
LoCoMo judge rule (generous on wording and date format). Only NCM was run.

| | NCM + Mistral Large 3 |
|---|---|
| judged correct (J), the model grading itself | **76.4%** |
| judged correct, independent judge (Gemini 2.5 Flash) on a random 100 | **65%** (the self-judge gave the same 100: 77%) |
| token F1 | 0.475 |
| evidence turn in the prompt | 78.1% |
| memory tokens sent per question | ~586 |
| retrieval latency (p50, laptop, with reranker) | 43 ms |
| model calls to store memories | 0 |
| cost of the whole run (answering and grading) | ~$0.72 |

By kind (self-judged): single-hop 84.8% (841), multi-hop 71.9% (281), temporal 65.0% (320),
open-domain 51.7% (89).

**Which J to quote.** The two judges agree on 86 of 100 answers (κ 0.67). Where they differ,
the independent one is stricter, and mostly right: "the week before 21 January" answered as
"21 January", three hikes listed when the answer is four, one of two books. So the score is
about **65-76% depending on the judge**; ~65% is the conservative figure. The cross-check is
in `results/locomo-bedrock-mistral-large-3-675b-instruct-judge-check.json`.

**Against section 3's published numbers** (different reader and judge, so only a rough
guide): in the range of Mem0 (66.9%) and Zep (66.0-75.1%) as reported in the Mem0 paper and
by Zep, below full context with GPT-4o-mini (72.9%) and the 88-94% that vendors report for
their own stacks. What stands out is cost: no model calls to store memories, ~586 tokens per
answer against 1,764 (Mem0) and 26,031 (full context), and 43 ms retrieval. The main loss
is retrieval itself: for 22% of questions the evidence turn is not among the 10 memories.

## 4. How it was run, and what to keep in mind

- **One reader for everyone:** qwen2.5:3b through Ollama 0.33, temperature 0, seed 0,
  8k context (24k for full context). Ollama must run with a full-precision KV cache:
  `OLLAMA_KV_CACHE_TYPE=q4_0` corrupts this model's output (we saw "Carolineline"),
  which broke Mem0's JSON extraction and depressed every score. Mem0 also needs
  `OLLAMA_CONTEXT_LENGTH=16384`, because its extraction prompt alone is ~8,500 tokens and
  is silently truncated otherwise.
- **Mem0** is the real `mem0ai` 2.2.1 library with its own extraction prompt, BM25 extras
  and spaCy, a local Qdrant, a fresh history database and telemetry off. Two turns per
  write; the date of each session is written into
  the messages and the metadata (the open-source SDK has no timestamp argument).
- **A small local reader is not GPT-4.** Absolute F1 is low for every system and the
  ranking could differ with a stronger reader, especially for LLM-based memory.
- **Subsets.** End-to-end uses the two shortest LoCoMo conversations so full context fits
  a 3B model and Mem0's ingestion stays affordable; retrieval uses all of LoCoMo and 150
  LongMemEval histories. Each number is one run; see the noise note in section 1.
- **NCM v1** is replayed from rankings saved by `bench/v1_baseline.py`, so its write cost
  is the v1 design's (two embeddings per memory), not a fresh measurement.

## Reproduce

```bash
# Ollama with a full-precision KV cache and a 16k context (PowerShell shown)
$env:OLLAMA_KV_CACHE_TYPE="f16"; $env:OLLAMA_CONTEXT_LENGTH="16384"; ollama serve
ollama pull qwen2.5:3b
pip install -e ".[dev]" "mem0ai[extras,nlp]" qdrant-client ollama
python -m spacy download en_core_web_sm
python -m comparisons.run_all        # retrieval, end-to-end, then the tables (~4 h on a laptop)
```

Files: `retrieval.py` (section 2), `end_to_end.py` (section 1, one class per system),
`locomo_bedrock.py` and `locomo_judge_check.py` (section 3b; a Bedrock API key in `.env`),
`common.py` (metered LLM client, subsets), `report.py` (builds these tables from
`results/`), `run_all.py` (everything in order). Datasets: see `bench/data.py`.
