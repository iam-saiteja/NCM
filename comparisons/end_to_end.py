"""End-to-end head-to-head: same conversations, same questions, same reader LLM.

Each memory system ingests LoCoMo conv-26 and conv-30 (38 sessions, 788 turns) and then
supplies context for all 230 questions about them. A local qwen2.5:3b reads that context
and answers; answers are scored with token F1. Besides accuracy we record what each system
costs: time and LLM calls to write, time to fetch context, prompt size, storage.

  python -m comparisons.end_to_end                 # all systems (resumes from cache)
  python -m comparisons.end_to_end --systems mem0  # one system

Embedding models run on CPU (a laptop deployment); LLM work runs on the GPU through Ollama.
Ollama must run with a full-precision KV cache (OLLAMA_KV_CACHE_TYPE=f16): a 4-bit KV cache
corrupts qwen2.5-3B's output, and Mem0 needs OLLAMA_CONTEXT_LENGTH >= 16384 because its
extraction prompt alone is ~8,500 tokens.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

os.environ.setdefault("MEM0_TELEMETRY", "False")   # before mem0 is imported anywhere: no usage data leaves the machine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.metrics import paired_bootstrap              # noqa: E402
from bench.qa import SYSTEM, f1                          # noqa: E402
from comparisons.common import CACHE, RESULTS, Meter, chat, e2e_data, save  # noqa: E402

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
K = 10


def when(days):
    return EPOCH + timedelta(days=days)


def day(days):
    d = when(days)
    return f"{d.day} {d:%B %Y}"


def render(h, rows):
    lines, last = [], None
    for r in sorted(set(int(x) for x in rows), key=lambda r: h.times[r]):
        if last is not None and h.sessions[r] != h.sessions[last]:
            lines.append("")
        lines.append(f"[{day(h.times[r])}] {h.texts[r]}")
        last = r
    return "Memories (oldest first):\n" + "\n".join(lines)


_st = {}


def st(name):
    """A sentence-transformers model on CPU."""
    if name not in _st:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        if "reranker" in name or "cross-encoder" in name:
            _st[name] = CrossEncoder(name, device="cpu", max_length=256)
        else:
            _st[name] = SentenceTransformer(name, device="cpu")
    return _st[name]


def dense_index(name, h, prefix_q=""):
    return st(name).encode(h.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------------- systems
class System:
    name = ""
    kind = ""            # "baseline", "retrieval", "LLM memory", "NCM"
    infra = ""
    num_ctx = 8192

    def __init__(self):
        self.write = Meter()

    def prepare(self):
        """Load models; not timed (a one-off cost per process, not per memory)."""

    def ingest(self, hays):
        pass

    def context(self, h, q):
        return ""

    def storage_bytes(self):
        return 0


class NoMemory(System):
    name, kind, infra = "no memory", "baseline", "none"


class Window(System):
    name, kind, infra = "sliding window (last 20 turns)", "baseline", "none (LangChain buffer-window style)"

    def context(self, h, q):
        return render(h, range(len(h.texts) - 20, len(h.texts)))

    def storage_bytes(self):
        return 0


class FullContext(System):
    name, kind, infra = "full context (whole history in the prompt)", "baseline", "a long-context LLM"
    num_ctx = 24576

    def ingest(self, hays):
        self.chars = sum(len(t) for h in hays for t in h.texts)

    def context(self, h, q):
        return render(h, range(len(h.texts)))

    def storage_bytes(self):
        return self.chars


class Summaries(System):
    name, kind, infra = "summary memory (LLM summary per session)", "LLM memory", "LLM at write time"

    def ingest(self, hays):
        path = os.path.join(CACHE, "e2e-summaries.json")
        cached = json.load(open(path)) if os.path.exists(path) else {}
        self.summaries = {}
        for h in hays:
            for s in sorted(set(h.sessions)):
                rows = [i for i, x in enumerate(h.sessions) if x == s]
                key = f"{h.id}|{s}"
                if key not in cached:
                    transcript = "\n".join(h.texts[i] for i in rows)
                    cached[key] = chat([{"role": "system", "content": (
                        "Summarise this conversation session for long-term memory in at most 120 words. "
                        "Keep every name, date, event, plan, preference and fact; drop small talk.")},
                        {"role": "user", "content": f"Session date: {day(h.times[rows[0]])}\n\n{transcript}"}],
                        meter=self.write, num_ctx=8192, num_predict=220)
                    os.makedirs(CACHE, exist_ok=True)
                    json.dump(cached, open(path, "w"))
                self.summaries.setdefault(h.id, []).append((day(h.times[rows[0]]), cached[key]))

    def context(self, h, q):
        return "Session summaries (oldest first):\n" + "\n\n".join(f"[{d}] {s}" for d, s in self.summaries[h.id])

    def storage_bytes(self):
        return sum(len(s) for v in self.summaries.values() for _, s in v)


class Dense(System):
    kind = "retrieval"

    def __init__(self, model, label, prefix=""):
        super().__init__()
        self.model, self.prefix = model, prefix
        self.name = f"vector RAG · {label}"
        self.infra = "embedding model + vector DB (Chroma / pgvector / FAISS)"

    def prepare(self):
        st(self.model).encode(["warm up"])

    def ingest(self, hays):
        self.idx = {h.id: dense_index(self.model, h) for h in hays}

    def first(self, h, q):
        v = st(self.model).encode([self.prefix + q.question], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.idx[h.id] @ v

    def context(self, h, q):
        return render(h, np.argsort(-self.first(h, q), kind="stable")[:K])

    def storage_bytes(self):
        return sum(v.size * 4 for v in self.idx.values())


class Hybrid(Dense):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2", "")
        self.name = "hybrid RRF · BM25 + MiniLM"
        self.infra = "Elasticsearch / OpenSearch / Weaviate / Qdrant hybrid search"

    def ingest(self, hays):
        from ncm.lexical import BM25
        super().ingest(hays)
        self.bm = {}
        for h in hays:
            self.bm[h.id] = BM25()
            for t in h.texts:
                self.bm[h.id].add(t)

    def context(self, h, q):
        from comparisons.retrieval import rrf
        s = rrf(self.first(h, q), self.bm[h.id].scores(q.question))
        return render(h, np.argsort(-s, kind="stable")[:K])


class Rerank(Dense):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2", "")
        self.name = "retrieve & rerank · MiniLM → bge-reranker-base"
        self.infra = "vector DB + reranker model (Cohere-rerank pattern)"

    def prepare(self):
        super().prepare()
        st("BAAI/bge-reranker-base").predict([("warm", "up")])

    def context(self, h, q):
        head = np.argsort(-self.first(h, q), kind="stable")[:30]
        r = st("BAAI/bge-reranker-base").predict([(q.question, h.texts[i][:1000]) for i in head], show_progress_bar=False)
        return render(h, head[np.argsort(-np.asarray(r), kind="stable")][:K])


class Mem0(System):
    name, kind = "Mem0 (open-source, local LLM)", "LLM memory"
    infra = "LLM at every write + vector DB (Qdrant) + SQLite history"

    def ingest(self, hays):
        os.environ["MEM0_TELEMETRY"] = "False"
        from mem0 import Memory
        from mem0.llms import ollama as O
        self.dir = os.path.join(CACHE, "mem0-e2e")
        done = os.path.join(self.dir, "done.json")
        if os.path.exists(done):
            self.write.__dict__.update(json.load(open(done))["meter"])
        cfg = {"llm": {"provider": "ollama", "config": {"model": "qwen2.5:3b", "temperature": 0, "max_tokens": 2000}},
               "embedder": {"provider": "huggingface",
                            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384,
                                       "model_kwargs": {"device": "cpu"}}},
               "vector_store": {"provider": "qdrant", "config": {"collection_name": "e2e", "embedding_model_dims": 384,
                                                                 "path": os.path.join(self.dir, "qdrant")}},
               "history_db_path": os.path.join(self.dir, "history.db")}
        if not os.path.exists(done):
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.dir)
            meter, orig = self.write, O.OllamaLLM.generate_response

            def metered(llm, *a, **k):                      # count Mem0's own LLM work
                t0 = time.perf_counter()
                out = orig(llm, *a, **k)
                meter.calls += 1
                meter.seconds += time.perf_counter() - t0
                msgs = k.get("messages") or (a[0] if a else [])
                meter.prompt_tokens += sum(len(m.get("content", "")) for m in msgs) // 4
                meter.output_tokens += len(out or "") // 4
                return out
            O.OllamaLLM.generate_response = metered
        self.mem = Memory.from_config(cfg)
        if os.path.exists(done):
            return
        t0 = time.perf_counter()
        for h in hays:                                       # 2 turns per add
            for i in range(0, len(h.texts), 2):
                rows = [r for r in (i, i + 1) if r < len(h.texts) and h.sessions[r] == h.sessions[i]]
                rows = rows or [i]
                msgs = [{"role": "user", "content": f"[{day(h.times[r])}] {h.texts[r]}"} for r in rows]
                # the OSS SDK has no timestamp argument (cloud only): the date rides in the text and metadata
                self.mem.add(msgs, user_id=h.id, metadata={"date": day(h.times[rows[0]])})
                if i % 50 == 0:
                    print(f"  mem0 {h.id} {i}/{len(h.texts)} {time.perf_counter() - t0:.0f}s", flush=True)
        json.dump({"meter": self.write.__dict__, "seconds": time.perf_counter() - t0}, open(done, "w"))

    def context(self, h, q):
        res = self.mem.search(q.question, filters={"user_id": h.id}, top_k=K)["results"]
        lines = [f"- ({(r.get('metadata') or {}).get('date', '')}) {r['memory']}" for r in res]
        return "Memories:\n" + "\n".join(lines) if lines else ""

    def storage_bytes(self):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(self.dir) for f in fs)


class NCMv1(System):
    name, kind, infra = "NCM v1", "NCM", "numpy (v1 code)"

    def ingest(self, hays):
        self.rank = json.load(open(os.path.join(os.path.dirname(__file__), "..", "bench", "results",
                                                "rankings-v1-locomo-minilm.json")))

    def context(self, h, q):
        return render(h, self.rank[f"{h.id}|{q.question}"][:K])


class NCMv2(System):
    kind, infra = "NCM", "numpy only (+ embedding model)"

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.name = {"recall": "NCM v2 · recall()", "context": "NCM v2 · context()",
                     "rerank": "NCM v2 + reranker (accuracy mode)"}[mode]

    def prepare(self):
        from ncm import Embedder
        from ncm.embed import Reranker
        self.emb = Embedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.emb(["warm up"])
        self.rr = None
        if self.mode == "rerank":
            self.rr = Reranker(device="cpu")
            self.rr("warm", ["up"])

    def ingest(self, hays):
        from ncm import Memory
        self.mems = {}
        for h in hays:
            m = Memory(embedder=self.emb, reranker=self.rr)
            for s in sorted(set(h.sessions)):
                rows = [i for i, x in enumerate(h.sessions) if x == s]
                m.new_session()
                m.add_many([h.texts[i].split(": ", 1)[1] for i in rows], [h.speakers[i] for i in rows],
                           [when(h.times[i]) for i in rows])
            self.mems[h.id] = m

    def context(self, h, q):
        m = self.mems[h.id]
        if self.mode == "context":
            return m.context(q.question, persona_of=None)
        rows = []
        for hit in m.recall(q.question, k=K):     # map each hit back to its turn for identical rendering
            t_days = (hit.time - EPOCH.timestamp()) / 86400
            rows.append(min((i for i, t in enumerate(h.texts) if t == f"{hit.speaker}: {hit.text}"),
                            key=lambda i: abs(h.times[i] - t_days)))
        return render(h, rows)

    def storage_bytes(self):
        tmp = os.path.join(CACHE, "ncm-size.ncm")
        total = 0
        for m in self.mems.values():
            m.save(tmp)
            total += os.path.getsize(tmp)
        return total


def systems():
    return [NoMemory(), Window(), FullContext(), Summaries(),
            Dense("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (22M)"),
            Dense("BAAI/bge-base-en-v1.5", "bge-base (110M)",
                  prefix="Represent this sentence for searching relevant passages: "),
            Hybrid(), Rerank(), Mem0(), NCMv1(), NCMv2("recall"), NCMv2("context"), NCMv2("rerank")]


# ----------------------------------------------------------------------------------- runner
def main():
    sys.stdout.reconfigure(encoding="utf-8")   # names contain arrows; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="", help="comma-separated substrings of system names")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test; stats not saved)")
    a = ap.parse_args()
    hays, queries = e2e_data()
    hmap = {h.id: h for h in hays}
    queries = sorted(queries, key=lambda q: (q.haystack, q.question))   # conversation order: prompt caching
    if a.limit:
        queries = queries[:a.limit]
    picked = [s for s in systems() if not a.systems or any(x in s.name.lower() for x in a.systems.split(","))]
    cache_path = os.path.join(RESULTS, "e2e-answers.jsonl")
    os.makedirs(RESULTS, exist_ok=True)
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[(r["system"], r["qid"])] = r
    stats_path = os.path.join(RESULTS, "e2e-stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    out = open(cache_path, "a", encoding="utf-8")
    for sysm in picked:
        print(f"== {sysm.name}", flush=True)
        sysm.prepare()
        t0 = time.perf_counter()
        sysm.ingest(hays)
        ingest_s = time.perf_counter() - t0
        if isinstance(sysm, Mem0) and os.path.exists(os.path.join(sysm.dir, "done.json")):
            ingest_s = json.load(open(os.path.join(sysm.dir, "done.json")))["seconds"]
        ctx_ms, chars, answer_s, scores = [], [], [], []
        for i, q in enumerate(queries):
            h = hmap[q.haystack]
            qid = f"{q.haystack}|{q.question}"
            t1 = time.perf_counter()
            ctx = sysm.context(h, q)
            if i:
                ctx_ms.append(1000 * (time.perf_counter() - t1))
            prompt = (f"{ctx}\n\n" if ctx else "") + f"Question: {q.question}\nShort answer:"
            key = (sysm.name, qid)
            if key not in cache:
                t2 = time.perf_counter()
                ans = chat([{"role": "system", "content": SYSTEM["locomo"]}, {"role": "user", "content": prompt}],
                           num_ctx=sysm.num_ctx)
                cache[key] = {"system": sysm.name, "qid": qid, "answer": ans, "gold": q.answer,
                              "category": q.category, "prompt_chars": len(prompt),
                              "answer_seconds": time.perf_counter() - t2}
                out.write(json.dumps(cache[key]) + "\n")
                out.flush()
            r = cache[key]
            chars.append(r["prompt_chars"])
            answer_s.append(r["answer_seconds"])
            scores.append(f1(r["answer"], q.answer))
        n_turns = sum(len(h.texts) for h in hays)
        stats[sysm.name] = {
            "kind": sysm.kind, "infra": sysm.infra, "n_questions": len(queries), "f1": float(np.mean(scores)),
            "f1_by_category": {c: float(np.mean([s for s, q in zip(scores, queries) if q.category == c]))
                               for c in sorted({q.category for q in queries})},
            "per_question_f1": scores,
            "ingest_seconds_total": ingest_s, "ingest_ms_per_turn": 1000 * ingest_s / n_turns,
            "write_llm": sysm.write.as_dict(),
            "context_ms_p50": float(np.median(ctx_ms)) if ctx_ms else 0.0,
            "prompt_tokens_mean": float(np.mean(chars)) / 4, "answer_seconds_p50": float(np.median(answer_s)),
            "storage_bytes": sysm.storage_bytes()}
        s = stats[sysm.name]
        print(f"   F1 {s['f1']:.3f} | ingest {s['ingest_ms_per_turn']:.1f} ms/turn, {s['write_llm']['llm_calls']} LLM calls"
              f" | context {s['context_ms_p50']:.1f} ms | prompt ~{s['prompt_tokens_mean']:.0f} tok"
              f" | answer {s['answer_seconds_p50']:.2f} s", flush=True)
        if not a.limit:
            save("e2e-stats.json", stats)
    if a.limit:
        return
    if "NCM v2 · context()" in stats:          # paired CIs against NCM v2 context()
        base = stats["NCM v2 · context()"]["per_question_f1"]
        for name, s in stats.items():
            if name != "NCM v2 · context()" and len(s["per_question_f1"]) == len(base):
                s["ncm_context_minus_this"] = paired_bootstrap(s["per_question_f1"], base)
        save("e2e-stats.json", stats)


if __name__ == "__main__":
    main()
