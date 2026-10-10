"""Fit the valence/arousal probe shipped in ncm/affect_minilm.npy, and score every estimator.

EmoBank (Buechel & Hahn 2017, CC-BY-SA 4.0): 10k sentences with human valence and
arousal ratings on a 1-5 scale. Ridge on MiniLM embeddings: lambda picked on the dev
split, Pearson r reported on the untouched test split.

  python -m bench.affect_fit
Data: curl -L -o ../ncm-data/emobank/emobank.csv https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import DATA_DIR                          # noqa: E402
from bench.embed import embed                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "emobank", "emobank.csv"), encoding="utf-8")))
    split = {s: [x for x in rows if x["split"] == s] for s in ("train", "dev", "test")}
    X = {s: embed([x["text"] for x in v], "minilm", "doc", tag="emobank") for s, v in split.items()}
    Xpre = embed(["user: " + x["text"] for x in split["test"]], "minilm", "doc", tag="emobank")
    Y = {s: np.array([[float(x["V"]), float(x["A"])] for x in v]) for s, v in split.items()}
    mu, sd = Y["train"].mean(0), Y["train"].std(0)
    out = {"n": {s: len(v) for s, v in split.items()}}

    best = None
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        A = X["train"].T @ X["train"] + lam * np.eye(X["train"].shape[1])
        W = np.linalg.solve(A, X["train"].T @ ((Y["train"] - mu) / sd))
        dev = np.mean([r(X["dev"] @ W[:, i], Y["dev"][:, i]) for i in range(2)])
        out.setdefault("dev_mean_r", {})[lam] = dev
        if best is None or dev > best[0]:
            best = (dev, lam, W)
    _, lam, W = best
    pred = X["test"] @ W
    out["probe"] = {"lambda": lam,
                    "test_r_valence": r(pred[:, 0], Y["test"][:, 0]),
                    "test_r_arousal": r(pred[:, 1], Y["test"][:, 1]),
                    "test_r_valence_with_speaker_prefix": r((Xpre @ W)[:, 0], Y["test"][:, 0]),
                    "test_r_arousal_with_speaker_prefix": r((Xpre @ W)[:, 1], Y["test"][:, 1])}

    from ncm.affect import ANCHORS
    for dim, i in (("valence", 0), ("arousal", 1)):
        pos = embed(ANCHORS[dim][0], "minilm", "doc", tag="anchors").mean(0)
        neg = embed(ANCHORS[dim][1], "minilm", "doc", tag="anchors").mean(0)
        out.setdefault("anchors", {})[f"test_r_{dim}"] = r(X["test"] @ (pos - neg), Y["test"][:, i])

    np.save(os.path.join(HERE, "..", "ncm", "affect_minilm.npy"), W.astype(np.float32))
    json.dump(out, open(os.path.join(HERE, "results", "affect-emobank.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
