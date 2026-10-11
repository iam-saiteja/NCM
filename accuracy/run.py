"""How often does an assistant with NCM memory answer correctly about the person it talks to?

LongMemEval-S questions of the kinds that matter for a personal character:
  single-session-user (64)       facts the user told it
  knowledge-update (72)          facts that changed later (the newest one is right)
  single-session-assistant (56)  what the assistant itself said before
  + the 12 abstention questions of the first two kinds: the user never said it, and the
    right answer is "you didn't tell me" rather than an invented one.
Each question has its own ~50-session, ~500-turn history.

Every answer is scored two ways: an LLM judge (LongMemEval's grading rules) gives the
accuracy, and SQuAD-style token precision / recall / F1 against the gold answer are
recorded too. The share of questions whose context contained the evidence turn is
recorded separately, so memory errors and reader errors can be told apart.

  python -m accuracy.run --readers qwen2.5:3b,qwen2.5:7b     # read, judge, summarise (resumes)
  python -m accuracy.run --readers qwen2.5:3b+notes --arms oracle,ncm-rerank
Results and their reading: accuracy/README.md.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import DATA_DIR                          # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import paired_bootstrap               # noqa: E402
from bench.qa import normalize                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CATS = ("single-session-user", "knowledge-update", "single-session-assistant")
OLLAMA = "http://127.0.0.1:11434/api/chat"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
# Judges, in order: the free tier allows ~20 calls a day per model, so each call grades a
# batch of answers and the next model takes over when one runs out for the day.
JUDGES = ("gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
          "gemini-3-flash-preview")
JUDGE_BATCH = 100
READER_SYSTEM = ("You are a personal assistant with a long-term memory of your past conversations with "
                 "this user. Answer the user's question from those memories. If facts changed over "
                 "time, the most recent one is true. If the memories do not contain the answer, say "
                 "that the user has not told you, and do not guess. Answer in one or two sentences.")
# "+notes" readers: quote the memory lines that bear on the question first, then answer.
# A light Chain-of-Note (LongMemEval, Wu et al. 2025); small models otherwise often say
# "you never told me" about a fact that is right there in a long context. Measured: it
# hurt the 3B reader (oracle 79.7% -> 59.9%); kept so the result can be reproduced.
NOTES_SYSTEM = ("You are a personal assistant with a long-term memory of your past conversations with "
                "this user. First write 'Notes:' and copy, word for word, every memory line that helps "
                "answer the user's question, or write 'none' if no line does. Then write 'Answer:' and "
                "answer in one or two sentences, using only the notes. If facts changed over time, the "
                "most recent one is true. If the notes are 'none', say that the user has not told you.")


# ------------------------------------------------------------------------------- data
def load():
    with open(os.path.join(DATA_DIR, "longmemeval", "longmemeval_s.json"), "rb") as f:
        # read in chunks: one 277 MB read() fails with EINVAL on Windows when memory is tight
        raw = json.loads(b"".join(iter(lambda: f.read(1 << 24), b"")).decode("utf-8"))
    fmt = "%Y/%m/%d (%a) %H:%M"
    items = []
    for x in raw:
        if x["question_type"] not in CATS:
            continue
        abstain = x["question_id"].endswith("_abs")
        if abstain and x["question_type"] == "single-session-assistant":
            continue
        turns = []
        for s, (sess, date) in enumerate(zip(x["haystack_sessions"], x["haystack_dates"])):
            t0 = datetime.strptime(date, fmt).replace(tzinfo=timezone.utc).timestamp()
            for i, t in enumerate(sess):
                if t["content"].strip():
                    turns.append({"role": t["role"], "text": t["content"], "session": s, "time": t0 + 60 * i,
                                  "gold": bool(t.get("has_answer"))})
        items.append({"id": x["question_id"], "category": x["question_type"], "abstain": abstain,
                      "question": x["question"], "answer": str(x["answer"]), "date": x["question_date"],
                      "turns": turns})
    return items


def day(ts):
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{d.day} {d:%B %Y}, {d:%H:%M}"


def render(item, rows, title="Memories of past conversations (oldest first)"):
    lines, last = [], None
    for r in sorted(set(rows), key=lambda r: item["turns"][r]["time"]):
        t = item["turns"][r]
        if last is not None and t["session"] != last:
            lines.append("")
        lines.append(f"[{day(t['time'])}] {'you' if t['role'] == 'assistant' else 'user'}: {t['text']}")
        last = t["session"]
    return f"{title}:\n" + "\n".join(lines)


# ------------------------------------------------------------------------------ memory
class Cached:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, table):
        self.table = table

    def __call__(self, texts, query=False):
        miss = [t for t in texts if (query, t) not in self.table]
        if miss:
            self.table.update(zip([(query, t) for t in miss],
                                  embed(miss, "minilm", "query" if query else "doc", tag="longmemeval")))
        return np.stack([self.table[(query, t)] for t in texts])


def build_memories(items):
    from ncm import Memory
    texts = [f"{t['role']}: {t['text']}" for it in items for t in it["turns"]]
    vecs = embed(texts, "minilm", "doc", tag="longmemeval")
    table = dict(zip([(False, t) for t in texts], vecs))
    emb = Cached(table)
    mems, docs, i = {}, {}, 0
    for it in items:
        n = len(it["turns"])
        docs[it["id"]], i = vecs[i:i + n], i + n
        m = Memory(embedder=emb)
        for s in sorted({t["session"] for t in it["turns"]}):
            rows = [t for t in it["turns"] if t["session"] == s]
            m.new_session()
            m.add_many([t["text"] for t in rows], [t["role"] for t in rows], [t["time"] for t in rows])
        mems[it["id"]] = m
    return mems, docs, emb


def contexts(arm, items, mems, docs, emb):
    out = {}
    for it in items:
        if arm == "oracle":                                # the whole evidence sessions, as LongMemEval's oracle
            sess = {t["session"] for t in it["turns"] if t["gold"]}
            if not sess:
                continue                                   # abstention: there is no evidence to hand over
            out[it["id"]] = render(it, [i for i, t in enumerate(it["turns"]) if t["session"] in sess])
        elif arm == "vector":                              # plain vector search, 10 turns
            qv = emb([it["question"]], query=True)[0]
            out[it["id"]] = render(it, np.argsort(-(docs[it["id"]] @ qv), kind="stable")[:10])
        elif arm in ("ncm", "ncm-rerank"):                 # the shipped Memory.context()
            out[it["id"]] = mems[it["id"]].context(it["question"], persona_of="user", self_name="assistant")
        elif arm == "ncm-top2":                            # reranker, just the two best memories: the second look
            out[it["id"]] = mems[it["id"]].context(it["question"], k=2, window=0, persona_of=None,
                                                   self_name="assistant")
    return out


def evidence_in(ctx, item):
    gold = [t["text"] for t in item["turns"] if t["gold"]]
    return None if not gold else any(g in ctx for g in gold)


# ------------------------------------------------------------------------------ models
def gemini_key():
    raw = [l.split("=", 1)[1] for l in open(os.path.join(HERE, "..", ".env"), encoding="utf-8")
           if l.strip().startswith("gemini_api")][0]
    return raw.strip().strip('"').strip("'").strip()


_last = {"t": 0.0}


class DailyQuota(RuntimeError):
    pass


def gemini(model, system, prompt, json_mode=False, min_interval=4.5, thinking=0):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 32768 if thinking else 8192,
                                 "thinkingConfig": {"thinkingBudget": thinking}}}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    for attempt in range(8):
        wait = min_interval - (time.time() - _last["t"])
        if wait > 0:
            time.sleep(wait)
        _last["t"] = time.time()
        try:
            req = urllib.request.Request(GEMINI.format(model), json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "x-goog-api-key": gemini_key()})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 429 and "PerDay" in msg:
                raise DailyQuota(model) from None
            if e.code in (429, 500, 503):
                print(f"gemini {model} {e.code}, retry {attempt + 1}", file=sys.stderr, flush=True)
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"gemini {e.code}: {msg[:300]}") from None
        except (urllib.error.URLError, TimeoutError, KeyError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("gemini unavailable")


def ollama(model, system, prompt, num_predict=120):
    body = {"model": model, "stream": False, "messages": [{"role": "system", "content": system},
                                                          {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": num_predict}}
    for attempt in range(5):
        try:
            req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("ollama unreachable")


def read(reader, prompt):
    """(answer, full output). A reader is an Ollama or Gemini model name, optionally "+notes"."""
    model, notes = reader.removesuffix("+notes"), reader.endswith("+notes")
    system = NOTES_SYSTEM if notes else READER_SYSTEM
    if model.startswith("gemini"):
        raw = gemini(model, system, prompt, min_interval=4.5)
    else:
        raw = ollama(model, system, prompt, num_predict=500 if notes else 120)
    answer = raw.rsplit("Answer:", 1)[-1].strip() if notes else raw
    return answer or raw, raw


# ------------------------------------------------------------------------------- judge
# LongMemEval's grading rules (Wu et al. 2025), one per question kind.
RULES = {
    "standard": "correct if the response contains the correct answer or something equivalent to it, or all "
                "the steps to get it; wrong if it gives only part of the required information, a different "
                "answer, or says it does not know.",
    "update": "the fact changed over time. Correct if the response gives the correct (most recent) answer; "
              "mentioning older information as well is fine as long as the updated answer is the one given.",
    "unanswerable": "the user never gave this information. Correct if the response says it was not provided "
                    "(it may mention related things that were); wrong if it invents or asserts an answer.",
}


def kind(b):
    return "unanswerable" if b["abstain"] else "update" if b["category"] == "knowledge-update" else "standard"


def judge_batch(batch, judges=JUDGES):
    """[(correct, judge model)] for a batch of answers, graded in one call."""
    lines = [f"### Item {i} ({kind(b)})\nQuestion: {b['question']}\n"
             f"{'Note' if b['abstain'] else 'Correct answer'}: {b['gold']}\nResponse: {b['reply']}"
             for i, b in enumerate(batch)]
    prompt = ("Grade each response to a question about the user's own past conversations. Grade each item "
              "on its own. Extra details that do not contradict the correct answer are fine. Rules by kind:\n"
              + "\n".join(f"- {k}: {v}" for k, v in RULES.items())
              + '\n\nReturn JSON: {"grades": [{"item": 0, "correct": true}, ...]} with one entry per item.\n\n'
              + "\n\n".join(lines))
    for model in judges:
        if model in _spent:
            continue
        try:
            out = json.loads(gemini(model, "You are a strict, fair grader of question answering.", prompt,
                                    json_mode=True, min_interval=7, thinking=12000))
        except DailyQuota:
            _spent.add(model)
            print(f"judge {model}: daily quota used up", file=sys.stderr, flush=True)
            continue
        except (json.JSONDecodeError, RuntimeError) as e:   # cut off, or overloaded: the next judge grades it
            print(f"judge {model}: {type(e).__name__}, trying the next judge", file=sys.stderr, flush=True)
            continue
        grades = {int(g["item"]): bool(g["correct"]) for g in out.get("grades", [])}
        if set(grades) != set(range(len(batch))):
            print(f"judge {model}: incomplete batch, trying the next judge", file=sys.stderr, flush=True)
            continue
        return [(grades[i], model) for i in range(len(batch))]
    raise DailyQuota("every judge model is out of quota for today; rerun tomorrow to finish")


_spent = set()


def prf(pred, gold):
    p, g = normalize(pred), normalize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0, 0.0, 0.0
    pr, rc = common / len(p), common / len(g)
    return pr, rc, 2 * pr * rc / (pr + rc)


# -------------------------------------------------------------------------------- main
def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", default="qwen2.5:3b,qwen2.5:7b")
    ap.add_argument("--arms", default="oracle,vector,ncm,ncm-rerank,ncm-top2")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--no-judge", action="store_true", help="read only")
    ap.add_argument("--summarise", action="store_true", help="only rebuild summary.json from saved judgments")
    ap.add_argument("--judge-check", type=int, default=0, metavar="N",
                    help="regrade N random judged answers with --check-with and report agreement")
    ap.add_argument("--check-with", default="gemini-3-flash-preview")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    items = load()
    byid = {it["id"]: it for it in items}
    print(f"{len(items)} questions: {dict(Counter((it['category'], it['abstain']) for it in items))}", flush=True)

    arms = a.arms.split(",")
    ctx_path = os.path.join(RES, "contexts.json")
    ctx = json.load(open(ctx_path, encoding="utf-8")) if os.path.exists(ctx_path) else {}
    if any(arm not in ctx for arm in arms):
        mems, docs, emb = build_memories(items)
        for arm in arms:
            if arm in ctx:
                continue
            if arm in ("ncm-rerank", "ncm-top2"):
                from ncm.embed import Reranker
                rr = Reranker()
                for m in mems.values():
                    m.reranker = rr
            ctx[arm] = contexts(arm, items, mems, docs, emb)
            for m in mems.values():
                m.reranker = None
            print(f"context ready: {arm}", flush=True)
        json.dump(ctx, open(ctx_path, "w", encoding="utf-8"))
    if a.limit:
        keep = {it["id"] for it in items[:a.limit]}
        ctx = {arm: {k: v for k, v in c.items() if k in keep} for arm, c in ctx.items()}

    for reader in [r for r in a.readers.split(",") if r and not a.summarise]:
        ans_path = os.path.join(RES, f"answers-{reader.replace(':', '_').replace('+', '-')}.jsonl")
        done = {(r["arm"], r["id"]) for r in jsonl(ans_path)}
        with open(ans_path, "a", encoding="utf-8") as out:
            for arm in arms:
                t0 = time.time()
                print(f"reading: {reader} {arm}", flush=True)
                for n, (qid, c) in enumerate(ctx[arm].items()):
                    if (arm, qid) in done:
                        continue
                    it = byid[qid]
                    prompt = f"Today's date: {it['date']}\n\n{c}\n\nUser: {it['question']}"
                    t1 = time.time()
                    reply, raw = read(reader, prompt)
                    out.write(json.dumps({"reader": reader, "arm": arm, "id": qid, "reply": reply,
                                          **({"raw": raw} if raw != reply else {}),
                                          "seconds": time.time() - t1, "prompt_chars": len(prompt)}) + "\n")
                    out.flush()
                    if n % 50 == 0:
                        print(f"  {reader} {arm} {n}/{len(ctx[arm])} {time.time() - t0:.0f}s", flush=True)

    if a.no_judge:
        return
    judged_path = os.path.join(RES, "judgments.jsonl")
    judged = {(r["reader"], r["arm"], r["id"]): r for r in jsonl(judged_path)}
    answers = [r for f in sorted(os.listdir(RES)) if f.startswith("answers-") for r in jsonl(os.path.join(RES, f))]
    todo = [] if a.summarise else [r for r in answers if (r["reader"], r["arm"], r["id"]) not in judged]
    with open(judged_path, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), JUDGE_BATCH):
            batch = [{**r, **{k: byid[r["id"]][k] for k in ("question", "category", "abstain")},
                      "gold": byid[r["id"]]["answer"]} for r in todo[i:i + JUDGE_BATCH]]
            for b, (ok, judge) in zip(batch, judge_batch(batch)):
                rec = {"reader": b["reader"], "arm": b["arm"], "id": b["id"], "correct": ok, "judge": judge}
                judged[(b["reader"], b["arm"], b["id"])] = rec
                out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"  judged {min(i + JUDGE_BATCH, len(todo))}/{len(todo)}", flush=True)
    summarise(items, byid, answers, judged, ctx)
    if a.judge_check:
        judge_check(byid, answers, judged, a.judge_check, a.check_with)


def judge_check(byid, answers, judged, n, model):
    """How far to trust the judges: `model` regrades a fixed random sample of answers that
    other models graded (results/judge-check-<model>.jsonl)."""
    path = os.path.join(RES, f"judge-check-{model}.jsonl")
    done = jsonl(path)
    if len(done) < n:
        pool = [r for r in answers if judged.get((r["reader"], r["arm"], r["id"]), {}).get("judge") not in (None, model)]
        sample = [pool[i] for i in np.random.default_rng(0).permutation(len(pool))[:n]]
        batch = [{**r, **{k: byid[r["id"]][k] for k in ("question", "category", "abstain")},
                  "gold": byid[r["id"]]["answer"]} for r in sample]
        with open(path, "w", encoding="utf-8") as out:
            for b, (ok, second) in zip(batch, judge_batch(batch, judges=(model,))):
                first = judged[(b["reader"], b["arm"], b["id"])]
                out.write(json.dumps({"reader": b["reader"], "arm": b["arm"], "id": b["id"],
                                      "first": first["correct"], "first_judge": first.get("judge"),
                                      "second": ok, "second_judge": second}) + "\n")
        done = jsonl(path)
    for who in sorted({d["first_judge"] for d in done}) + [None]:
        ds = [d for d in done if who is None or d["first_judge"] == who]
        a_, b_ = np.array([d["first"] for d in ds], float), np.array([d["second"] for d in ds], float)
        po = float((a_ == b_).mean())
        pe = float(a_.mean() * b_.mean() + (1 - a_.mean()) * (1 - b_.mean()))
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        print(f"judge check {who or 'all'} vs {model}: {len(ds)} answers, agreement {po:.3f}, "
              f"kappa {kappa:.3f}, right per first {a_.mean():.3f}, per second {b_.mean():.3f}")


# The reader saying it was never told (checked by eye on the 3B answers).
_SAY = r"(told|tell|mention|mentioned|provide|provided|specify|specified|share|shared|say|said)"
REFUSAL = re.compile(rf"(not|never|n't) {_SAY}\b|(don't|do not|doesn't|does not) (have|contain|include|specify|mention)"
                     r"|no (specific )?(information|details|record|mention)|isn't a specific"
                     r"|not (available|in (our|your|the))", re.I)


def second_look(answers, judged, ctx):
    """The cascade an app can run: answer from context(); only if the reader says it was
    never told, ask again with just the two best memories. Built from the "ncm-rerank" and
    "ncm-top2" answers, so it costs no extra reads here; its time is the sum of both."""
    by = {(r["reader"], r["arm"], r["id"]): r for r in answers}
    out, verdicts = [], {}
    for (reader, arm, qid), r in by.items():
        second = by.get((reader, "ncm-top2", qid))
        if arm != "ncm-rerank" or second is None or (reader, "ncm-top2", qid) not in judged:
            continue
        refused = bool(REFUSAL.search(r["reply"]))
        pick = second if refused else r
        out.append({**pick, "arm": "ncm-rerank+second-look", "refused_first": refused,
                    "seconds": r["seconds"] + (second["seconds"] if refused else 0),
                    "prompt_chars": r["prompt_chars"] + (second["prompt_chars"] if refused else 0)})
        verdicts[(reader, "ncm-rerank+second-look", qid)] = judged[(pick["reader"], pick["arm"], qid)]
    ctx["ncm-rerank+second-look"] = ctx.get("ncm-rerank", {})
    return out, verdicts


def summarise(items, byid, answers, judged, ctx):
    extra, verdicts = second_look(answers, judged, ctx)
    answers, judged = answers + extra, {**judged, **verdicts}
    rows = {}
    for r in answers:
        k = (r["reader"], r["arm"])
        j = judged.get((r["reader"], r["arm"], r["id"]))
        if j is None:
            continue
        it = byid[r["id"]]
        p, rc, f = prf(r["reply"], it["answer"])
        rows.setdefault(k, []).append({"id": r["id"], "category": "abstention" if it["abstain"] else it["category"],
                                       "correct": float(j["correct"]), "p": p, "r": rc, "f1": f,
                                       "evidence": evidence_in(ctx[r["arm"]][r["id"]], it), "seconds": r["seconds"],
                                       "chars": r["prompt_chars"], "refused": bool(REFUSAL.search(r["reply"]))})
    summary = {}
    for (reader, arm), rs in sorted(rows.items()):
        ev = [x["evidence"] for x in rs if x["evidence"] is not None]
        cats = sorted({x["category"] for x in rs})
        summary[f"{reader} | {arm}"] = s = {
            "n": len(rs), "accuracy": float(np.mean([x["correct"] for x in rs])),
            "accuracy_answerable": float(np.mean([x["correct"] for x in rs if x["category"] != "abstention"])),
            "token_precision": float(np.mean([x["p"] for x in rs])), "token_recall": float(np.mean([x["r"] for x in rs])),
            "token_f1": float(np.mean([x["f1"] for x in rs])),
            "evidence_in_context": float(np.mean(ev)) if ev else None,
            "accuracy_by_category": {c: float(np.mean([x["correct"] for x in rs if x["category"] == c])) for c in cats},
            "n_by_category": {c: sum(1 for x in rs if x["category"] == c) for c in cats},
            "reader_seconds_p50": float(np.median([x["seconds"] for x in rs])),
            "prompt_chars_mean": float(np.mean([x["chars"] for x in rs])),
            "says_not_told": float(np.mean([x["refused"] for x in rs])),
            "says_not_told_but_evidence_given": float(np.mean([x["refused"] and bool(x["evidence"]) for x in rs]))}
        for base in ("vector", "oracle", "ncm-rerank"):   # paired, on the questions both arms answered
            other = {x["id"]: x["correct"] for x in rows.get((reader, base), [])}
            pairs = [(other[x["id"]], x["correct"]) for x in rs if x["id"] in other]
            if base == arm or len(pairs) < 20:
                continue
            a_, b_ = zip(*pairs)
            wins, losses = sum(y > x for x, y in pairs), sum(x > y for x, y in pairs)
            s[f"vs_{base}"] = paired_bootstrap(a_, b_) | {
                "n": len(pairs), "only_this_right": wins, f"only_{base}_right": losses,
                "mcnemar_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0}
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write(table(summary))
    for k, s in summary.items():
        print(f"{k:42s} acc {s['accuracy']:.3f} | P {s['token_precision']:.3f} R {s['token_recall']:.3f} "
              f"F1 {s['token_f1']:.3f} | evidence {s['evidence_in_context']} | {s['accuracy_by_category']}")


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def table(summary):
    """Every number in summary.json as one markdown table (written to results/summary.md)."""
    cats = ("single-session-user", "knowledge-update", "single-session-assistant", "abstention")
    head = ["reader", "memory", "n", "accuracy [95% CI]", "user facts", "changed facts", "its own words",
            "abstain", "token P", "token R", "token F1", "evidence in prompt", "says not told",
            "prompt chars", "reader s (p50)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, s in summary.items():
        reader, arm = key.split(" | ")
        lo, hi = wilson(s["accuracy"], s["n"])
        by = s["accuracy_by_category"]
        out.append("| " + " | ".join([
            reader, arm, str(s["n"]), f"{s['accuracy']:.1%} [{lo:.1%}, {hi:.1%}]",
            *[f"{by[c]:.1%}" if c in by else "-" for c in cats],
            f"{s['token_precision']:.3f}", f"{s['token_recall']:.3f}", f"{s['token_f1']:.3f}",
            f"{s['evidence_in_context']:.1%}" if s["evidence_in_context"] is not None else "-",
            f"{s['says_not_told']:.1%}", f"{s['prompt_chars_mean']:,.0f}", f"{s['reader_seconds_p50']:.1f}"]) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
