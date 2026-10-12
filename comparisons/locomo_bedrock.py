"""LoCoMo with NCM and one large model on Amazon Bedrock (answers and grades).

All 10 LoCoMo conversations; the questions of categories 1-4 (multi-hop, temporal,
open-domain, single-hop) that have evidence, the set memory papers report on. NCM runs
as in comparisons/end_to_end.py's best setup: Memory(reranker=True), the 10 best memories,
dated, oldest first. The model answers in a short phrase; the same model then grades the
answer against the gold one with the LLM-judge rule used for LoCoMo's "J" score (generous
on wording and date format, strict on the fact). Token F1 is recorded too, as a check that
does not depend on the model. Every call's token usage is kept.

Needs bedrock_api_key (a Bedrock API key) and bedrock_region in .env.

  python -m comparisons.locomo_bedrock --limit 3           # pilot
  python -m comparisons.locomo_bedrock                     # all questions (resumes)
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import load_locomo                       # noqa: E402
from bench.qa import normalize                           # noqa: E402
from comparisons.end_to_end import K, render, when       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = "mistral.mistral-large-3-675b-instruct"
ANSWER_SYSTEM = ("You answer questions about a long conversation between two people, using memories of "
                 "it. Each memory starts with the date it was said. Answer with a short phrase (a few "
                 "words), only from the memories. Turn relative times such as 'last year' or 'yesterday' "
                 "into the actual date or period, using the memory's date.")
JUDGE_SYSTEM = "You grade answers to questions about a conversation. Reply with JSON only."
JUDGE_PROMPT = """Label the generated answer CORRECT or WRONG, given the gold answer.
Be generous: if the generated answer refers to the same fact, person, thing or time as the
gold answer, it is CORRECT, even if it is longer or worded differently. For dates and times,
a different format (for example "May 7th" and "7 May") or a reference to the same period is
CORRECT. If it gives a different fact, or says it does not know, it is WRONG.

Question: {q}
Gold answer: {gold}
Generated answer: {answer}

Reply with JSON: {{"reasoning": "one sentence", "label": "CORRECT" or "WRONG"}}"""


def env():
    p = os.path.join(HERE, "..", ".env")
    return {l.split("=", 1)[0].strip(): l.split("=", 1)[1].strip().strip('"').strip("'")
            for l in open(p, encoding="utf-8") if "=" in l}


def converse(system, prompt, max_tokens, _cfg={}):
    if not _cfg:
        e = env()
        _cfg.update(key=e["bedrock_api_key"], region=e.get("bedrock_region", "us-east-1"))
    url = f"https://bedrock-runtime.{_cfg['region']}.amazonaws.com/model/{MODEL}/converse"
    body = {"system": [{"text": system}], "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0}}
    for attempt in range(6):
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Authorization": f"Bearer {_cfg['key']}",
                                                                       "Content-Type": "application/json"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            text = "".join(c.get("text", "") for c in out["output"]["message"]["content"]).strip()
            u = out.get("usage", {})
            return text, u.get("inputTokens", 0), u.get("outputTokens", 0), time.time() - t0
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"bedrock {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("bedrock unreachable")


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="a random sample of N questions (pilot)")
    a = ap.parse_args()
    hays, queries = load_locomo()
    if a.limit:
        queries = random.Random(0).sample(queries, a.limit)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"locomo-bedrock-{MODEL.split('.', 1)[1]}{'-pilot' if a.limit else ''}.jsonl")
    done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()

    from ncm import Embedder, Memory
    from ncm.embed import Reranker
    emb, rr = Embedder("sentence-transformers/all-MiniLM-L6-v2"), Reranker()
    need = {q.haystack for q in queries}
    mems = {}
    for h in hays:
        if h.id not in need:
            continue
        m = Memory(embedder=emb, reranker=rr)
        for s in sorted(set(h.sessions)):
            rows = [i for i, x in enumerate(h.sessions) if x == s]
            m.new_session()
            m.add_many([h.texts[i].split(": ", 1)[1] for i in rows], [h.speakers[i] for i in rows],
                       [when(h.times[i]) for i in rows])
        mems[h.id] = (h, m)
    print(f"{len(queries)} questions, {len(mems)} conversations, model {MODEL}", flush=True)

    with open(path, "a", encoding="utf-8") as out:
        for n, q in enumerate(queries):
            key = f"{q.haystack}|{q.question}"
            if key in done:
                continue
            h, m = mems[q.haystack]
            t0 = time.perf_counter()
            rows = []
            for hit in m.recall(q.question, k=K):          # the same rendering as comparisons/end_to_end.py
                t_days = (hit.time - when(0).timestamp()) / 86400
                rows.append(min((i for i, t in enumerate(h.texts) if t == f"{hit.speaker}: {hit.text}"),
                                key=lambda i: abs(h.times[i] - t_days)))
            ctx = render(h, rows)
            recall_ms = (time.perf_counter() - t0) * 1000
            answer, ai, ao, asec = converse(ANSWER_SYSTEM, f"{ctx}\n\nQuestion: {q.question}\nAnswer:", 60)
            verdict, ji, jo, _ = converse(JUDGE_SYSTEM, JUDGE_PROMPT.format(q=q.question, gold=q.answer, answer=answer), 200)
            try:
                label = json.loads(verdict[verdict.find("{"):verdict.rfind("}") + 1])["label"].upper()
            except (ValueError, KeyError):
                label = "CORRECT" if "CORRECT" in verdict.upper() and "WRONG" not in verdict.upper() else "WRONG"
            rec = {"key": key, "category": q.category, "question": q.question, "gold": q.answer, "answer": answer,
                   "label": label, "judge": verdict, "f1": f1(answer, q.answer),
                   "evidence_in_context": any(r in rows for r in q.gold), "recall_ms": recall_ms,
                   "answer_seconds": asec, "tokens": {"answer_in": ai, "answer_out": ao, "judge_in": ji, "judge_out": jo}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if a.limit or n % 50 == 0:
                print(f"  {n}: [{q.category}] {q.question} | gold: {q.answer} | answer: {answer} | {label}", flush=True)
    summarise(path)


def summarise(path):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    j = np.mean([r["label"] == "CORRECT" for r in rs])
    tok = {k: sum(r["tokens"][k] for r in rs) for k in ("answer_in", "answer_out", "judge_in", "judge_out")}
    print(f"\n{len(rs)} questions: J {j:.3f}, F1 {np.mean([r['f1'] for r in rs]):.3f}, "
          f"evidence in context {np.mean([r['evidence_in_context'] for r in rs]):.3f}, "
          f"recall {np.median([r['recall_ms'] for r in rs]):.0f} ms, answer {np.median([r['answer_seconds'] for r in rs]):.1f} s")
    for c in sorted({r["category"] for r in rs}):
        cr = [r for r in rs if r["category"] == c]
        print(f"  {c:12s} n {len(cr):4d}  J {np.mean([r['label'] == 'CORRECT' for r in cr]):.3f}  F1 {np.mean([r['f1'] for r in cr]):.3f}")
    print(f"tokens: {tok} (per question: answer {tok['answer_in'] / len(rs):.0f} in / {tok['answer_out'] / len(rs):.0f} out)")


if __name__ == "__main__":
    main()
