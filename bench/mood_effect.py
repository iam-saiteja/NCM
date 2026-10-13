"""Does NCM's mood signal change what the character says?

Retrieval in NCM ignores mood; the mood reaches the model only as text: context() adds
"Their mood lately: ..." and examples/chat.py tells the model "The conversation lately has
felt {tone}; let that colour your tone naturally". This tests whether those lines matter.

Each scenario: a user writes five messages of one clear mood (real Reddit comments from
GoEmotions with a single sad/angry/afraid label, or a single joyful/excited/grateful one),
then asks something neutral ("What should I cook tonight?"). The character answers three
times, with the same memories and model, temperature 0:

  none      the two mood sentences removed (control)
  ncm       as shipped: NCM's mood label
  opposite  the opposite label (positive <-> negative): does the model follow the line?

Measured: does NCM's mood() read the user's mood right; how many replies change; and one
judge (Gemini 2.5 Flash) rates, blind to the arm, how well each reply's tone fits the
user's recent mood (1-5) and whether it acknowledges their feelings.

  python -m bench.mood_effect --models qwen2.5:3b,qwen2.5:7b --n 40
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accuracy.run import gemini                                    # noqa: E402
from bench.data import DATA_DIR                                    # noqa: E402
from bench.qa import ask_raw                                       # noqa: E402
from ncm import Embedder, Memory                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
JUDGE = "gemini-2.5-flash"
NEG = {"sadness", "grief", "disappointment", "anger", "fear", "nervousness", "remorse"}
POS = {"joy", "excitement", "gratitude", "love", "amusement", "optimism", "pride"}
QUESTIONS = ["What should I cook tonight?", "Can you suggest a movie for this evening?",
             "Any ideas for what to do this weekend?", "How should I organise my desk?",
             "What's a good book to start next?", "Should I go for a walk or a run today?",
             "What music would you put on right now?", "Any tips for planning my week?",
             "What should I have for breakfast tomorrow?", "Can you recommend a hobby to try?"]

spec = importlib.util.spec_from_file_location("chat", os.path.join(HERE, "..", "examples", "chat.py"))
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)                                      # the shipped persona and rules


def messages(rng, n):
    d = os.path.join(DATA_DIR, "goemotions")
    names = open(os.path.join(d, "emotions.txt"), encoding="utf-8").read().split()
    pools = {"negative": [], "positive": []}
    for row in csv.reader(open(os.path.join(d, "test.tsv"), encoding="utf-8"), delimiter="\t"):
        labels = [names[int(i)] for i in row[1].split(",")]
        text = row[0].strip()
        if len(labels) != 1 or len(text.split()) < 6 or "[NAME]" in text:
            continue
        if labels[0] in NEG:
            pools["negative"].append(text)
        elif labels[0] in POS:
            pools["positive"].append(text)
    return [{"mood": mood, "history": rng.sample(pools[mood], 5), "question": QUESTIONS[i % len(QUESTIONS)]}
            for mood in ("negative", "positive") for i in range(n)]


def system_prompt(card, memory, tone):
    parts = [f"You are {card['name']}, {card['persona']}. Style: {card['style']}."]
    if tone is not None:
        parts.append(f"The conversation lately has felt {tone}; let that colour your tone naturally.")
    return "\n\n".join(parts + [chat.RULES, memory])


def flip(label):
    return label.replace("negative", "§").replace("positive", "negative").replace("§", "positive")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5:3b,qwen2.5:7b")
    ap.add_argument("--n", type=int, default=40, help="scenarios per mood")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    scen = messages(random.Random(0), a.n)
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    card = chat.DEFAULT_CARD
    path = os.path.join(RES, "mood-effect-replies.jsonl")
    done = {(r["model"], r["i"], r["arm"]) for r in map(json.loads, open(path, encoding="utf-8"))} if os.path.exists(path) else set()
    with open(path, "a", encoding="utf-8") as out:
        for i, s in enumerate(scen):
            m = Memory(embedder=emb)
            m.add_many(s["history"], ["user"] * 5, [1.7e9 + 60 * j for j in range(5)])
            label = m.mood()["label"]
            mem_block = m.context(s["question"], persona_of="user", self_name=card["name"])
            arms = {"none": (None, mem_block.replace(f"\nTheir mood lately: {label}", "")),
                    "ncm": (label, mem_block),
                    "opposite": (flip(label), mem_block.replace(f"Their mood lately: {label}",
                                                               f"Their mood lately: {flip(label)}"))}
            for model in a.models.split(","):
                for arm, (tone, block) in arms.items():
                    if (model, i, arm) in done:
                        continue
                    reply = ask_raw(system_prompt(card, block, tone), s["question"], model, num_predict=120)
                    out.write(json.dumps({"model": model, "i": i, "arm": arm, "mood": s["mood"], "label": label,
                                          "history": s["history"], "question": s["question"], "reply": reply}) + "\n")
                    out.flush()
            if i % 10 == 0:
                print(f"  scenario {i}/{len(scen)} ({s['mood']}, NCM read: {label})", flush=True)
    judge(path)


def judge(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    gpath = path.replace("replies.jsonl", "grades.json")
    grades = json.load(open(gpath, encoding="utf-8")) if os.path.exists(gpath) else {}
    todo = [r for r in rows if f"{r['model']}|{r['i']}|{r['arm']}" not in grades]
    order = random.Random(1).sample(todo, len(todo))                # arms shuffled: the judge cannot tell them apart
    for b in range(0, len(order), 60):
        batch = order[b:b + 60]
        items = "\n\n".join(f"### Item {j}\nUser's recent messages:\n" + "\n".join(f"- {h}" for h in r["history"])
                            + f"\nUser now asks: {r['question']}\nCharacter's reply: {r['reply']}"
                            for j, r in enumerate(batch))
        prompt = ("For each item, rate how well the character's reply fits the user's recent emotional state, "
                  "judged from their recent messages: 5 = the tone fits it well (for example gentle and "
                  "supportive if they seem upset, warm and upbeat if they seem happy), 3 = neutral, 1 = "
                  "clearly mismatched (for example cheerful while they are grieving). Also say whether the reply "
                  "acknowledges their feelings at all.\nReturn JSON: {\"grades\": [{\"item\": 0, \"fit\": 1-5, "
                  "\"acknowledges\": true}, ...]}\n\n" + items)
        out = json.loads(gemini(JUDGE, "You rate the emotional fit of chat replies.", prompt, json_mode=True,
                                min_interval=7, thinking=8000))
        for g in out["grades"]:
            r = batch[int(g["item"])]
            grades[f"{r['model']}|{r['i']}|{r['arm']}"] = {"fit": int(g["fit"]), "ack": bool(g["acknowledges"])}
        json.dump(grades, open(gpath, "w", encoding="utf-8"), indent=0)
        print(f"  judged {min(b + 60, len(order))}/{len(order)}", flush=True)
    summarise(rows, grades, path.replace("replies.jsonl", "summary.json"))


def summarise(rows, grades, path):
    out = {}
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["i"]), {})[r["arm"]] = r
    scen = {}
    for (model, i), arms in by.items():
        scen.setdefault(model, []).append(arms)
    for model, ss in scen.items():
        res = {"scenarios": len(ss)}
        first = [s["ncm"] for s in ss]
        res["ncm_reads_mood_right"] = float(np.mean([(r["mood"] in r["label"]) for r in first]))
        res["ncm_says_neutral"] = float(np.mean([r["label"] == "neutral" for r in first]))
        res["replies_changed_by_mood_line"] = float(np.mean([s["ncm"]["reply"] != s["none"]["reply"] for s in ss]))
        g = {arm: [grades.get(f"{model}|{s[arm]['i']}|{arm}") for s in ss] for arm in ("none", "ncm", "opposite")}
        for arm in g:
            res[f"fit_{arm}"] = float(np.mean([x["fit"] for x in g[arm] if x]))
            res[f"acknowledges_{arm}"] = float(np.mean([x["ack"] for x in g[arm] if x]))
            for mood in ("negative", "positive"):
                res[f"fit_{arm}_{mood}"] = float(np.mean([x["fit"] for x, s in zip(g[arm], ss) if x and s["ncm"]["mood"] == mood]))
        for arm in ("ncm", "opposite"):
            a_ = [x["fit"] for x in g["none"]]
            b_ = [x["fit"] for x in g[arm]]
            d = np.array(b_) - np.array(a_)
            res[f"{arm}_vs_none_fit"] = {"delta": float(d.mean()), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                                         "wilcoxon_p": float(wilcoxon(a_, b_).pvalue) if d.any() else 1.0}
            w = sum(x["ack"] and not y["ack"] for x, y in zip(g[arm], g["none"]))
            l = sum(y["ack"] and not x["ack"] for x, y in zip(g[arm], g["none"]))
            res[f"{arm}_vs_none_acknowledges"] = {"more": w, "fewer": l, "p": float(binomtest(w, w + l).pvalue) if w + l else 1.0}
        out[model] = res
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
