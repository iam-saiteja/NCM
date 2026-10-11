"""When the character cannot keep everything: which forgetting rule keeps what people ask about?

The 204 LongMemEval-S questions of accuracy/run.py, but each history (~500 turns) is
replayed into a Memory capped at a fraction of its size (Config.max_items), so the
least active memories fade as it goes. Then context(question) is built as an app would
(with the reranker). Measured, per forgetting setup:

  answer in context  the gold answer's words appear in the prompt (questions whose
                     answer is at most 6 words; a gist counts)
  evidence turn      the full original evidence turn is in the prompt
  kept               memories in the store at the end, and their characters

Setups are Config settings (see ncm/memory.py): ACT-R activation with emotional
salience as in v2.0 (whose ages were measured from the wall clock), the age fix, and
self-reference at three weights. Ideas that were measured and dropped: docs/BRAIN.md.

  python -m accuracy.bounded                  # writes accuracy/results/bounded.json
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accuracy.run import RES, Cached, evidence_in, load       # noqa: E402
from bench.embed import embed                                 # noqa: E402
from bench.qa import normalize                                # noqa: E402
from ncm import Memory                                        # noqa: E402
from ncm.memory import Config                                 # noqa: E402
from ncm.embed import Reranker                                # noqa: E402

FRACTIONS = (0.25, 0.5)
SETUPS = {
    "wall-clock age (v2.0)": {"self_weight": 0.0, "wall_clock": True},
    "age fix": {"self_weight": 0.0},
    "+ self 0.5 (default)": {"self_weight": 0.5},
    "+ self 1": {"self_weight": 1.0},
    "+ self 2": {"self_weight": 2.0},
}


class WallClock(Memory):
    """NCM 2.0's forgetting: ages measured from the wall clock, not the newest memory."""
    def activation(self, now=None):
        return super().activation(now=None)


def answer_in(ctx, answer):
    a = normalize(answer)
    return " ".join(a) in " ".join(normalize(ctx)) if 0 < len(a) <= 6 else None


def run(items, emb, rr, setup, frac):
    opts = dict(SETUPS[setup])
    cls = WallClock if opts.pop("wall_clock", False) else Memory
    rows, ctxs = [], {}
    for it in items:
        n = len(it["turns"])
        m = cls(embedder=emb, reranker=rr, config=Config(max_items=max(20, int(frac * n)), **opts))
        for s in sorted({t["session"] for t in it["turns"]}):
            ts = [t for t in it["turns"] if t["session"] == s]
            m.new_session()
            m.add_many([t["text"] for t in ts], [t["role"] for t in ts], [t["time"] for t in ts])
        c = m.context(it["question"], persona_of="user", self_name="assistant")
        ctxs[it["id"]] = c
        rows.append({"id": it["id"], "category": it["category"], "abstain": it["abstain"],
                     "answer_in": answer_in(c, it["answer"]), "evidence": evidence_in(c, it),
                     "kept": len(m), "kept_chars": sum(map(len, m.texts)),
                     "gists": sum(1 for x in m.metas if x.get("gist")), "chars": len(c)})
    return rows, ctxs


def summary(rows):
    ans = [r for r in rows if not r["abstain"]]

    def mean(key, rs):
        v = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(v)) if v else None

    out = {"answer_in_context": mean("answer_in", ans), "evidence_turn_in_context": mean("evidence", ans),
           "n_answer_scored": sum(r["answer_in"] is not None for r in ans),
           "kept": mean("kept", rows), "kept_chars": mean("kept_chars", rows), "gists": mean("gists", rows),
           "prompt_chars": mean("chars", rows)}
    for c in ("single-session-user", "knowledge-update", "single-session-assistant"):
        rs = [r for r in ans if r["category"] == c]
        out[c] = {"answer_in_context": mean("answer_in", rs), "evidence_turn_in_context": mean("evidence", rs)}
    return out


def read_and_judge(reader, setups):
    """Answer from the saved bounded contexts with `reader`, grade with the study's judge,
    and compare setups pairwise on the same questions."""
    from scipy.stats import binomtest
    from accuracy.run import JUDGE_BATCH, judge_batch, jsonl, read
    items = {it["id"]: it for it in load()}
    ctxs = json.load(open(os.path.join(RES, "bounded-contexts.json"), encoding="utf-8"))
    keys = [f"{s} @ {f:.0%}" for f in FRACTIONS for s in setups]
    ans_path, jud_path = os.path.join(RES, "bounded-answers.jsonl"), os.path.join(RES, "bounded-judgments.jsonl")
    done = {(r["reader"], r["setup"], r["id"]) for r in jsonl(ans_path)}
    with open(ans_path, "a", encoding="utf-8") as out:
        for key in keys:
            for n, (q, c) in enumerate(ctxs[key].items()):
                if (reader, key, q) in done:
                    continue
                it = items[q]
                t0 = time.time()
                reply = read(reader, f"Today's date: {it['date']}\n\n{c}\n\nUser: {it['question']}")[0]
                out.write(json.dumps({"reader": reader, "setup": key, "id": q, "reply": reply,
                                      "seconds": time.time() - t0}) + "\n")
                out.flush()
                if n % 50 == 0:
                    print(f"  {reader} {key} {n}/{len(ctxs[key])}", flush=True)
    judged = {(r["reader"], r["setup"], r["id"]): r for r in jsonl(jud_path)}
    todo = [r for r in jsonl(ans_path) if (r["reader"], r["setup"], r["id"]) not in judged]
    with open(jud_path, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), JUDGE_BATCH):
            batch = [{**r, **{k: items[r["id"]][k] for k in ("question", "category", "abstain")},
                      "gold": items[r["id"]]["answer"]} for r in todo[i:i + JUDGE_BATCH]]
            for b, (ok, judge) in zip(batch, judge_batch(batch)):
                judged[(b["reader"], b["setup"], b["id"])] = rec = {
                    "reader": b["reader"], "setup": b["setup"], "id": b["id"], "correct": ok, "judge": judge}
                out.write(json.dumps(rec) + "\n")
            out.flush()
    res = {}
    for key in keys:
        g = {q: judged[(reader, key, q)]["correct"] for q in items if (reader, key, q) in judged}
        cats = {}
        for q, ok in g.items():
            c = "abstention" if items[q]["abstain"] else items[q]["category"]
            cats.setdefault(c, []).append(ok)
        res[key] = {"n": len(g), "accuracy": float(np.mean(list(g.values()))),
                    "by_category": {c: float(np.mean(v)) for c, v in sorted(cats.items())}, "grades": g}
    for f in FRACTIONS:
        base = res[f"{setups[0]} @ {f:.0%}"]["grades"]
        for s in setups[1:]:
            other = res[f"{s} @ {f:.0%}"]["grades"]
            ids = [q for q in base if q in other]
            w, l = sum(other[q] > base[q] for q in ids), sum(other[q] < base[q] for q in ids)
            res[f"{s} @ {f:.0%}"][f"vs {setups[0]}"] = {
                "wins": w, "losses": l, "mcnemar_p": float(binomtest(w, w + l).pvalue) if w + l else 1.0}
    path = os.path.join(RES, f"bounded-accuracy-{reader.replace(':', '_')}.json")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "grades"} for k, v in res.items()},
              open(path, "w", encoding="utf-8"), indent=1)
    for k, v in res.items():
        print(f"{reader} | {k:34s} accuracy {v['accuracy']:.3f} {v['by_category']} "
              f"{ {kk: vv for kk, vv in v.items() if kk.startswith('vs')} }")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if os.environ.get("READ_SETUPS"):                  # e.g. READ_SETUPS="wall-clock age (v2.0)|age fix"
        return read_and_judge(os.environ.get("READER", "qwen2.5:3b"), os.environ["READ_SETUPS"].split("|"))
    items = load()
    texts = [f"{t['role']}: {t['text']}" for it in items for t in it["turns"]]
    emb = Cached(dict(zip([(False, t) for t in texts], embed(texts, "minilm", "doc", tag="longmemeval"))))
    rr = Reranker()
    only = sys.argv[1:]                                # optional: re-run setups whose name contains these
    path = os.path.join(RES, "bounded.json")
    res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    rows_path = os.path.join(RES, "bounded-rows.json")
    all_rows = json.load(open(rows_path, encoding="utf-8")) if os.path.exists(rows_path) else {}
    ctx_path = os.path.join(RES, "bounded-contexts.json")
    all_ctx = json.load(open(ctx_path, encoding="utf-8")) if os.path.exists(ctx_path) else {}
    for setup in SETUPS:
        for frac in FRACTIONS:
            key = f"{setup} @ {frac:.0%}"
            if key in res and not any(o in setup for o in only):
                continue
            t0 = time.time()
            rows, ctxs = run(items, emb, rr, setup, frac)
            res[key], all_rows[key], all_ctx[key] = summary(rows), rows, ctxs
            s = res[key]
            print(f"{key:38s} answer {s['answer_in_context']:.3f} evidence {s['evidence_turn_in_context']:.3f} "
                  f"| user {s['single-session-user']['answer_in_context']:.3f} "
                  f"changed {s['knowledge-update']['answer_in_context']:.3f} "
                  f"own {s['single-session-assistant']['evidence_turn_in_context']:.3f} "
                  f"| kept {s['kept']:.0f} ({s['gists']:.0f} gists) {time.time() - t0:.0f}s", flush=True)
            for p, obj in ((path, res), (rows_path, all_rows), (ctx_path, all_ctx)):
                json.dump(obj, open(p, "w", encoding="utf-8"), indent=1 if p == path else None)


if __name__ == "__main__":
    main()
