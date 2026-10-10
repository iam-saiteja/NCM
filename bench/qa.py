"""End-to-end QA: a local LLM answers questions from each arm's memories.

  python -m bench.qa                                   # LoCoMo, 400 questions
  python -m bench.qa --dataset longmemeval --n 200

Arms differ only in which memories reach the prompt. Every arm renders memories the
same way (dated, oldest first). Scored with SQuAD-style token F1, the LoCoMo metric.
LongMemEval skips single-session-preference (its gold answers are rubrics, not
answers) and clips each turn to 400 characters for every arm, so a 3B model can read
ten of them. Answers are cached in bench/results/qa-cache.jsonl; interrupted runs resume.
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import load_locomo, load_longmemeval     # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import paired_bootstrap               # noqa: E402
from ncm import Config, Memory                           # noqa: E402
from ncm.lexical import BM25                             # noqa: E402
from ncm.memory import score                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
SYSTEM = {
    "locomo": ("You answer questions about a long conversation between two people, using only the "
               "memories provided. Reply with a short answer of a few words, no explanation. When a "
               "memory says 'yesterday', 'last week' and so on, work out the date from the date shown "
               "in brackets."),
    "longmemeval": ("You are an assistant answering a question about your past conversations with the "
                    "user, using only the memories provided. If the user changed their mind or updated "
                    "a fact, the most recent memory is right. Reply with a short answer of a few words, "
                    "no explanation."),
}
CLIP = {"locomo": None, "longmemeval": 400}


def ask_raw(system, prompt, model, num_predict=40, url="http://localhost:11434/api/chat"):
    body = {"model": model, "stream": False, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": num_predict}}
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"].strip()
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("ollama unreachable")


def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


CONFIGS = {"hybrid-nolift": Config(lift_previous=0.0),
           "ncm-v2": Config(),
           "ncm-v2-bidirectional": Config(lift_previous=0.75, lift_next=0.75, lift_width=3.0)}


def neighbour_max(s, sess, width):
    """Strongest same-session neighbour of each turn, either side, decayed by distance."""
    out, lift = np.zeros_like(s), np.maximum(s, 0)
    for u in np.unique(sess):
        idx = np.flatnonzero(sess == u)
        k = np.exp(-np.abs(idx[:, None] - idx[None, :]) / width)
        np.fill_diagonal(k, 0)
        out[idx] = (k * lift[idx][None, :]).max(axis=1)
    return out


def when(days):
    return EPOCH + timedelta(days=days)


def flat(text, clip):
    text = " ".join(text.split())
    return text if clip is None or len(text) <= clip else text[:clip] + "..."


def render(h, rows, clip=None):
    lines, last = [], None
    for r in sorted(set(rows), key=lambda r: h.times[r]):
        if last is not None and h.sessions[r] != h.sessions[last]:
            lines.append("")
        text = h.texts[r] if clip is None else flat(h.texts[r], clip)
        lines.append(f"[{when(h.times[r]):%d %B %Y}] {text}")
        last = r
    return "Memories (oldest first):\n" + "\n".join(lines)


class Cached:
    """Library-compatible embedder serving the shared benchmark cache from memory."""
    name = "minilm"

    def __init__(self, table, tag):
        self.table, self.tag = table, tag

    def __call__(self, texts, query=False):
        missing = [t for t in texts if (query, t) not in self.table]
        if missing:
            self.table.update(zip([(query, t) for t in missing],
                                  embed(missing, "minilm", "query" if query else "doc", tag=self.tag)))
        return np.stack([self.table[(query, t)] for t in texts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=("locomo", "longmemeval"))
    ap.add_argument("--n", type=int, default=None, help="questions (default 400 LoCoMo, 200 LongMemEval)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    ds, clip = a.dataset, CLIP[a.dataset]
    if ds == "locomo":
        hays, queries = load_locomo()
        arms = (a.arms or "none,dense,ncm-v1,hybrid-nolift,hybrid-symmetric-spread,ncm-v2,"
                          "ncm-v2-bidirectional,ncm-v2-context").split(",")
    else:
        hays, queries = load_longmemeval()
        queries = [q for q in queries if q.category != "single-session-preference"]
        arms = (a.arms or "none,dense,ncm-v1,ncm-v2,ncm-v2-context").split(",")
    random.Random(0).shuffle(queries)
    queries = queries[:a.n or (400 if ds == "locomo" else 200)]
    keep = {q.haystack for q in queries}
    hays = [h for h in hays if h.id in keep]
    hmap = {h.id: h for h in hays}
    v1 = json.load(open(os.path.join(RES, f"rankings-v1-{ds}-minilm.json")))
    docs = {h.id: embed(h.texts, "minilm", "doc", tag=ds) for h in hays}
    bms = {}
    for h in hays:
        bms[h.id] = BM25()
        for t in h.texts:
            bms[h.id].add(t)
    mems = {}
    for h in hays:
        # the Memory stores flattened (and, for LongMemEval, clipped) text so context() renders
        # exactly what the other arms render; its vectors are those of the original full turns
        table = {}
        m = Memory(embedder=Cached(table, ds))
        for s in sorted(set(h.sessions)):
            rows = [i for i, x in enumerate(h.sessions) if x == s]
            texts = [flat(h.texts[i].split(": ", 1)[1], clip) for i in rows]
            for i, t in zip(rows, texts):
                table[(False, f"{h.speakers[i]}: {t}")] = docs[h.id][i]
            m.new_session()
            m.add_many(texts, [h.speakers[i] for i in rows], [when(h.times[i]) for i in rows])
        mems[h.id] = m

    cache_path = os.path.join(RES, "qa-cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[(r["arm"], r["model"], r["k"], r["qid"])] = r
    out = open(cache_path, "a", encoding="utf-8")

    results = {arm: [] for arm in arms}
    t0 = time.time()
    for qi, q in enumerate(queries):
        h = hmap[q.haystack]
        qv = embed([q.question], "minilm", "query", tag=ds)[0]
        qid = f"{q.haystack}|{q.question}"
        for arm in arms:
            key = (arm, a.model, a.k, qid)
            if key not in cache:
                if arm == "none":
                    ctx = ""
                elif arm == "dense":
                    ctx = render(h, np.argsort(-(docs[h.id] @ qv), kind="stable")[:a.k], clip)
                elif arm == "ncm-v1":
                    ctx = render(h, v1[qid][:a.k], clip)
                elif arm in CONFIGS:
                    s = score(docs[h.id] @ qv, bms[h.id].scores(q.question), np.asarray(h.sessions), CONFIGS[arm])
                    ctx = render(h, np.argsort(-s, kind="stable")[:a.k], clip)
                elif arm == "hybrid-symmetric-spread":   # pre-release prototype: max-neighbour, both directions
                    s = score(docs[h.id] @ qv, bms[h.id].scores(q.question), np.asarray(h.sessions),
                              Config(lift_previous=0))
                    ctx = render(h, np.argsort(-(s + 0.75 * neighbour_max(s, np.asarray(h.sessions), 3.0)),
                                               kind="stable")[:a.k], clip)
                elif arm == "ncm-v2-context":            # the shipped Memory.context(), ~same turn budget
                    ctx = mems[h.id].context(q.question, k=max(1, a.k * 2 // 5), persona_of=None)
                else:
                    raise ValueError(f"unknown arm {arm}")
                head = f"Today's date: {when(q.time):%d %B %Y}\n\n" if ds == "longmemeval" else ""
                prompt = head + (f"{ctx}\n\n" if ctx else "") + f"Question: {q.question}\nShort answer:"
                ans = ask_raw(SYSTEM[ds], prompt, a.model)
                cache[key] = {"arm": arm, "model": a.model, "k": a.k, "qid": qid, "answer": ans,
                              "gold": q.answer, "category": q.category, "prompt_chars": len(prompt)}
                out.write(json.dumps(cache[key]) + "\n")
                out.flush()
            r = cache[key]
            results[arm].append({"f1": f1(r["answer"], q.answer), "category": q.category,
                                 "chars": r["prompt_chars"]})
        if qi % 25 == 0:
            print(f"{qi}/{len(queries)} {time.time()-t0:.0f}s " + " ".join(
                f"{arm}={np.mean([x['f1'] for x in results[arm]]):.3f}" for arm in arms), flush=True)

    summary = {"dataset": ds, "model": a.model, "k": a.k, "n": len(queries), "arms": {}}
    for arm in arms:
        rows = results[arm]
        cats = sorted({r["category"] for r in rows})
        summary["arms"][arm] = {
            "f1": float(np.mean([r["f1"] for r in rows])),
            "prompt_chars": float(np.mean([r["chars"] for r in rows])),
            "by_category": {c: float(np.mean([r["f1"] for r in rows if r["category"] == c])) for c in cats}}
        if arm != "dense" and "dense" in results:
            summary["arms"][arm]["vs_dense"] = paired_bootstrap(
                [r["f1"] for r in results["dense"]], [r["f1"] for r in rows])
        if arm != "ncm-v1" and "ncm-v1" in results:
            summary["arms"][arm]["vs_v1"] = paired_bootstrap(
                [r["f1"] for r in results["ncm-v1"]], [r["f1"] for r in rows])
    json.dump(summary, open(os.path.join(RES, f"qa-{ds}-{a.model.replace(':', '_')}.json"), "w"), indent=1)
    print(json.dumps({k: (v["f1"], v["prompt_chars"]) for k, v in summary["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
