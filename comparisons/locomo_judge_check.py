"""How far to trust the self-graded J score of comparisons/locomo_bedrock.py: an independent
judge (Gemini Flash, Gemma as fallback; validated in accuracy/) regrades a fixed random
sample with the same rule, and the two are compared.

  python -m comparisons.locomo_judge_check --n 100
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accuracy.run import JUDGES, gemini                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = ("Label each generated answer CORRECT or WRONG, given the gold answer. Be generous: if it refers to "
        "the same fact, person, thing or time as the gold answer, it is CORRECT, even if longer or worded "
        "differently; a different date format or the same period is CORRECT. A different fact, a wrong count "
        "or number, or 'I don't know' is WRONG.")


def grade(batch):
    items = "\n\n".join(f"### Item {i}\nQuestion: {r['question']}\nGold answer: {r['gold']}\nGenerated answer: {r['answer']}"
                        for i, r in enumerate(batch))
    prompt = f"{RULE}\n\nReturn JSON: {{\"grades\": [{{\"item\": 0, \"label\": \"CORRECT\"}}, ...]}}\n\n{items}"
    for model in JUDGES:
        try:
            reply = gemini(model, "You grade answers to questions about a conversation.", prompt,
                           json_mode=True, min_interval=7, thinking=8000)
            out = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
            g = {int(x["item"]): x["label"].upper() == "CORRECT" for x in out["grades"]}
            if set(g) == set(range(len(batch))):
                return [g[i] for i in range(len(batch))], model
        except Exception as e:                                # quota, overload, bad JSON: next judge
            print(f"  {model}: {type(e).__name__}", flush=True)
    raise RuntimeError("no judge available")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--run", default=os.path.join(HERE, "results", "locomo-bedrock-mistral-large-3-675b-instruct.jsonl"))
    a = ap.parse_args()
    rs = [json.loads(l) for l in open(a.run, encoding="utf-8")]
    sample = [rs[i] for i in np.random.default_rng(0).permutation(len(rs))[:a.n]]
    second, model = grade(sample)
    first = np.array([r["label"] == "CORRECT" for r in sample])
    second = np.array(second)
    po = float((first == second).mean())
    pe = float(first.mean() * second.mean() + (1 - first.mean()) * (1 - second.mean()))
    out = {"n": len(sample), "second_judge": model, "agreement": po, "kappa": (po - pe) / (1 - pe),
           "self_judge_correct": float(first.mean()), "independent_judge_correct": float(second.mean()),
           "disagreements": [{"question": r["question"], "gold": r["gold"], "answer": r["answer"],
                              "self": bool(f), "independent": bool(s)}
                             for r, f, s in zip(sample, first, second) if f != s]}
    path = a.run.replace(".jsonl", "-judge-check.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"{len(sample)} answers regraded by {model}: agreement {po:.3f}, kappa {out['kappa']:.3f}; "
          f"correct per self-judge {first.mean():.3f}, per independent judge {second.mean():.3f}")
    for d in out["disagreements"]:
        print(f"  self {'C' if d['self'] else 'W'} / indep {'C' if d['independent'] else 'W'} | {d['question']} | "
              f"gold: {d['gold']} | answer: {d['answer']}")


if __name__ == "__main__":
    main()
