"""Personalisation: after a multi-session chat, does the model know who each speaker is?

MSC test split (Xu et al. 2022). Each conversation's 5 sessions are stored in a Memory.
For each speaker we sample persona statements that are true of them (their own
human-written persona lines) and distractors (persona lines of the same speaker slot in
other conversations, kept only if no true line is a near-paraphrase). A local LLM then
judges them from each arm's context.

  --mode choice  (default) one true + one distractor, "which is true, A or B?";
                 chance is 50% and a model's yes/no bias cannot hide the signal
  --mode yesno   judge each statement alone; small models answer "no" to almost everything

  python -m bench.persona --convs 120

Get the data:  git show 9c1e851:experiments/data/real_world_corpus/test.jsonl > ../ncm-data/msc/test.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import DATA_DIR                          # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import paired_bootstrap               # noqa: E402
from bench.qa import ask_raw                             # noqa: E402
from ncm import Memory                                   # noqa: E402
from ncm.persona import persona                          # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SYSTEM = {
    "yesno": ("You decide whether a statement is true of a person, using only the conversation "
              "memories provided. If the memories do not support it, answer no. Reply with one "
              "word: yes or no."),
    "choice": ("You decide which of two statements is true of a person, using the conversation "
               "memories provided. Exactly one is true. Reply with one letter: A or B."),
}


class Cached:
    """Serves MiniLM vectors from one up-front batch; embeds anything new on demand.
    Named like the real model so the store uses the shipped MiniLM affect probe."""
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, table):
        self.table = table

    def __call__(self, texts, query=False):
        miss = [t for t in texts if (query, t) not in self.table]
        if miss:
            self.table.update(zip([(query, t) for t in miss],
                                  embed(miss, "minilm", "query" if query else "doc", tag="msc")))
        return np.stack([self.table[(query, t)] for t in texts])


def persona_lines(conv, speaker):
    lines = []
    for block in [conv["init_personas"]] + [s["personas"] for s in conv["sessions"]]:
        for p in block:
            if p["speaker"] == speaker:
                lines += [t.strip() for t in p["text"] if t.strip()]
    return list(dict.fromkeys(lines))


def sample_items(convs, n_convs, per, rng):
    """(conv, speaker, true statements, distractor statements) per speaker."""
    pool = {sp: [(i, t) for i, c in enumerate(convs) for t in persona_lines(c, sp)]
            for sp in ("Speaker 1", "Speaker 2")}
    pool_vec = {sp: embed([t for _, t in pool[sp]], "minilm", "doc", tag="msc-persona") for sp in pool}
    out = []
    for ci in rng.sample(range(len(convs)), n_convs):
        for sp in ("Speaker 1", "Speaker 2"):
            true = persona_lines(convs[ci], sp)
            if len(true) < per:
                continue
            tv = embed(true, "minilm", "doc", tag="msc-persona")
            cand = [j for j, (cj, _) in enumerate(pool[sp]) if cj != ci]
            sims = (pool_vec[sp][cand] @ tv.T).max(axis=1)
            safe = [j for j, s in zip(cand, sims) if s < 0.6]
            out.append((ci, sp, rng.sample(true, per), [pool[sp][j][1] for j in rng.sample(safe, per)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=120)
    ap.add_argument("--per", type=int, default=2, help="true (and as many false) statements per speaker")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--arms", default="none,recent,ncm-recall,ncm-persona,ncm-context")
    ap.add_argument("--mode", default="choice", choices=("choice", "yesno"))
    a = ap.parse_args()

    convs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "msc", "test.jsonl"), encoding="utf-8")]
    rng = random.Random(0)
    sampled = sample_items(convs, a.convs, a.per, rng)
    items = []   # (conv, speaker, statement or (A, B), label)
    for ci, sp, trues, falses in sampled:
        if a.mode == "yesno":
            items += [(ci, sp, t, True) for t in trues] + [(ci, sp, f, False) for f in falses]
        else:
            for t, f in zip(trues, falses):
                items.append((ci, sp, (t, f), "A") if rng.random() < 0.5 else (ci, sp, (f, t), "B"))

    used = sorted({x[0] for x in items})
    turns = [f"{d['speaker']}: {d['text']}" for ci in used for s in convs[ci]["sessions"] for d in s["dialogue"]]
    emb = Cached(dict(zip([(False, t) for t in turns], embed(turns, "minilm", "doc", tag="msc"))))
    mems = {}
    for ci in used:
        m = Memory(embedder=emb)
        for s, sess in enumerate(convs[ci]["sessions"]):
            m.new_session()
            m.add_many([d["text"] for d in sess["dialogue"]], [d["speaker"] for d in sess["dialogue"]],
                       [1.7e9 + s * 7 * 86400 + i * 60 for i in range(len(sess["dialogue"]))])
        mems[ci] = m

    def lines(m, rows, title):
        return f"{title}:\n" + "\n".join(f"- {m.speakers[r]}: {m.texts[r]}" for r in sorted(rows))

    cache_path = os.path.join(RES, "persona-cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[(r["arm"], r["model"], r["item"])] = r
    out = open(cache_path, "a", encoding="utf-8")
    arms = a.arms.split(",")
    res = {arm: [] for arm in arms}
    t0 = time.time()
    for n, (ci, sp, stmt, label) in enumerate(items):
        m = mems[ci]
        query = " ".join(stmt) if a.mode == "choice" else stmt
        item = f"choice|{ci}|{sp}|{stmt[0]}|{stmt[1]}" if a.mode == "choice" else f"{ci}|{sp}|{stmt}"
        for arm in arms:
            key = (arm, a.model, item)
            if key not in cache:
                if arm == "none":
                    ctx = ""
                elif arm == "recent":   # sliding window: the last 10 turns, what most bots keep
                    ctx = lines(m, range(max(0, len(m) - 10), len(m)), "Recent conversation")
                elif arm == "ncm-recall":
                    ctx = lines(m, [m._row(h.id) for h in m.recall(query, k=6, speaker=sp)], "Memories")
                elif arm == "ncm-persona":
                    ctx = f"What you know about {sp}:\n" + "\n".join(f"- {t}" for t in persona(m, sp, k=6))
                elif arm == "ncm-context":
                    ctx = m.context(query, k=4, persona_of=sp, persona_k=6)
                if a.mode == "choice":
                    ask = (f'Which statement is true of {sp}?\nA: "{stmt[0]}"\nB: "{stmt[1]}"\n'
                           "Answer with A or B.")
                else:
                    ask = f'Statement about {sp}: "{stmt}"\nIs this true of {sp}? Answer yes or no.'
                prompt = (f"{ctx}\n\n" if ctx else "") + ask
                ans = ask_raw(SYSTEM[a.mode], prompt, a.model, num_predict=3)
                cache[key] = {"arm": arm, "model": a.model, "item": item, "label": label,
                              "answer": ans, "chars": len(prompt)}
                out.write(json.dumps(cache[key]) + "\n")
                out.flush()
            r = cache[key]
            if a.mode == "choice":   # first standalone A or B: "Answer: B", "**B**", "B." all parse
                letter = re.search(r"\b([AB])\b", r["answer"].upper())
                got = letter.group(1) if letter else None
            else:
                got = r["answer"].strip().lower().startswith("yes")
            res[arm].append({"correct": float(got == label), "label": label, "chars": r["chars"]})
        if n % 50 == 0:
            print(f"{n}/{len(items)} {time.time()-t0:.0f}s " + " ".join(
                f"{arm}={np.mean([x['correct'] for x in res[arm]]):.3f}" for arm in arms), flush=True)

    summary = {"model": a.model, "mode": a.mode, "items": len(items), "arms": {}}
    for arm in arms:
        rows = res[arm]
        summary["arms"][arm] = {"accuracy": float(np.mean([r["correct"] for r in rows])),
                                "prompt_chars": float(np.mean([r["chars"] for r in rows]))}
        if a.mode == "yesno":
            summary["arms"][arm]["true_positive_rate"] = float(np.mean([r["correct"] for r in rows if r["label"]]))
            summary["arms"][arm]["true_negative_rate"] = float(
                np.mean([r["correct"] for r in rows if not r["label"]]))
        if arm != "recent" and "recent" in res:
            summary["arms"][arm]["vs_recent"] = paired_bootstrap(
                [r["correct"] for r in res["recent"]], [r["correct"] for r in rows])
    path = os.path.join(RES, f"persona-{a.mode}-msc-{a.model.replace(':', '_')}.json")
    json.dump(summary, open(path, "w"), indent=1)
    print(json.dumps({k: v["accuracy"] for k, v in summary["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
