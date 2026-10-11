"""Build comparisons/README.md tables from comparisons/results/*.json.

  python -m comparisons.report     # prints the tables; paste-ready markdown
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from comparisons.common import RESULTS   # noqa: E402

# what each system needs to run, and how much work it is to adopt (our judgement, stated)
PROFILE = {
    "no memory": ("nothing", "-"),
    "sliding window (last 20 turns)": ("nothing", "trivial"),
    "full context (whole history in the prompt)": ("a long-context LLM; cost grows with history", "trivial"),
    "summary memory (LLM summary per session)": ("an LLM call per session", "low"),
    "vector RAG · MiniLM-L6 (22M)": ("embedding model + vector DB", "low"),
    "vector RAG · bge-base (110M)": ("bigger embedding model + vector DB", "low"),
    "hybrid RRF · BM25 + MiniLM": ("search engine with hybrid mode (Elasticsearch, Weaviate, Qdrant)", "medium"),
    "retrieve & rerank · MiniLM → bge-reranker-base": ("vector DB + 278M reranker", "medium"),
    "Mem0 (open-source, local LLM)": ("LLM at every write + Qdrant + SQLite", "medium"),
    "NCM v1": ("numpy (old API)", "low"),
    "NCM v2 · recall()": ("numpy + embedding model, one file", "low: pip install, 3 lines"),
    "NCM v2 · context()": ("numpy + embedding model, one file", "low: pip install, 3 lines"),
    "NCM v2 + reranker (accuracy mode)": ("+ 22M reranker", "low: one flag"),
}


def fmt_bytes(b):
    return f"{b / 2**20:.1f} MB" if b >= 2**20 else f"{b / 1024:.0f} KB" if b else "-"


def e2e_table():
    s = json.load(open(os.path.join(RESULTS, "e2e-stats.json"), encoding="utf-8"))
    lines = ["| system | answer F1 | vs NCM context() | write cost per memory | LLM calls to write 788 turns "
             "| fetch context (p50) | prompt to reader (≈tokens) | reader time (p50) | storage | needs | effort |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(s.items(), key=lambda kv: -kv[1]["f1"]):
        d = r.get("ncm_context_minus_this")
        vs = "" if d is None else f"{-d['delta']:+.3f} [{-d['ci95'][1]:+.3f}, {-d['ci95'][0]:+.3f}]"
        w = r["write_llm"]
        write = f"{r['ingest_ms_per_turn']:.1f} ms" if r["ingest_ms_per_turn"] < 1000 else f"{r['ingest_ms_per_turn'] / 1000:.1f} s"
        if r.get("note"):            # replayed from saved rankings, so its write cost was not measured here
            write = "≈7.5 ms (2 embeddings)"
        tok = w["llm_prompt_tokens"]
        calls = (f"{w['llm_calls']} (≈{tok / 1e6:.1f}M tokens in)" if tok >= 1e6 else
                 f"{w['llm_calls']} (≈{tok / 1e3:.0f}k tokens in)") if w["llm_calls"] else "0"
        needs, effort = PROFILE.get(name, ("", ""))
        lines.append(f"| {name} | **{r['f1']:.3f}** | {vs} | {write} | {calls} | {r['context_ms_p50']:.0f} ms "
                     f"| {r['prompt_tokens_mean']:,.0f} | {r['answer_seconds_p50']:.2f} s | {fmt_bytes(r['storage_bytes'])} "
                     f"| {needs} | {effort} |")
    return "\n".join(lines)


# models each retrieval method runs: (at write time, per memory), (at query time)
MS, BGE_R = "ms-marco-MiniLM-L-6-v2-30pairs", "bge-reranker-base-30pairs"
MODELS_USED = {
    "vector RAG · MiniLM-L6 (22M)": (["minilm"], ["minilm"]),
    "vector RAG · bge-small (33M)": (["bge-small"], ["bge-small"]),
    "vector RAG · bge-base (110M)": (["bge-base"], ["bge-base"]),
    "BM25": ([], []),
    "SPLADE (learned sparse, 110M)": (["splade"], ["splade"]),
    "hybrid RRF · BM25 + MiniLM": (["minilm"], ["minilm"]),
    "retrieve & rerank · MiniLM top-30 → bge-reranker-base": (["minilm"], ["minilm", BGE_R]),
    "NCM v1": (["minilm", "minilm"], ["minilm"]),   # v1 embedded each text twice (memory + auto-state)
    "NCM v2": (["minilm"], ["minilm"]),
    "NCM v2 · bge-small embedder": (["bge-small"], ["bge-small"]),
    "NCM v2 · bge-base embedder": (["bge-base"], ["bge-base"]),
    "NCM v2 · SPLADE instead of BM25": (["minilm", "splade"], ["minilm", "splade"]),
    "NCM v2 · SPLADE + BM25": (["minilm", "splade"], ["minilm", "splade"]),
    "NCM v2 + reranker (ms-marco-MiniLM, 22M)": (["minilm"], ["minilm", MS]),
    "NCM v2 + bge-reranker-base": (["minilm"], ["minilm", BGE_R]),
}


def retrieval_table():
    r = json.load(open(os.path.join(RESULTS, "retrieval.json"), encoding="utf-8"))
    cost = r.get("model_costs", {})

    def total(keys, field):
        return sum(cost.get(f"{k}@cpu", {}).get(field, 0) for k in keys)
    lines = ["| method | LoCoMo R@5 | LoCoMo MRR | LongMemEval R@5 | LongMemEval MRR | model time to store a memory (CPU) "
             "| model time per query (CPU) |", "|---|---|---|---|---|---|---|"]
    for name, a in sorted(r["locomo"].items(), key=lambda kv: -kv[1]["recall@5"]):
        b = r["longmemeval_150"][name]
        doc, query = MODELS_USED.get(name, ([], []))
        w = total(doc, "ms_per_memory_batched")
        q = total(query, "ms_per_query")
        lines.append(f"| {name} | **{a['recall@5']:.3f}** | {a['mrr']:.3f} | **{b['recall@5']:.3f}** | {b['mrr']:.3f} "
                     f"| {f'{w:.1f} ms' if w else 'none (tokenise only)'} | {f'{q:.0f} ms' if q else 'none'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if os.path.exists(os.path.join(RESULTS, "retrieval.json")):
        print(retrieval_table(), "\n")
    if os.path.exists(os.path.join(RESULTS, "e2e-stats.json")):
        print(e2e_table())
