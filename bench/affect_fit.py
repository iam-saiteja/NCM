"""Fit the valence/arousal probe shipped in ncm/affect_minilm.npy, and score every estimator.

Two human-labelled corpora, MiniLM embeddings, ridge regression:
  EmoBank (Buechel & Hahn 2017, CC-BY-SA 4.0): 10k sentences (news, fiction, letters)
    rated for valence and arousal on 1-5.
  GoEmotions (Demszky et al. 2020, CC-BY 4.0): 58k Reddit comments with emotion labels;
    its official sentiment grouping gives conversational valence: positive +1,
    negative -1, neutral 0 (ambiguous or mixed comments are skipped).
Valence is fitted on both (each corpus weighted equally), arousal on EmoBank only.
The ridge penalty is picked on the two dev splits; test numbers are reported for both
the EmoBank-only probe and the combined one, and the better one on the dev splits ships.

  python -m bench.affect_fit
Data (into ../ncm-data):
  emobank/emobank.csv  https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv
  goemotions/{train,dev,test}.tsv, emotions.txt, sentiment_mapping.json from
  https://github.com/google-research/google-research/tree/master/goemotions/data
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
LAMBDAS = (0.3, 1.0, 3.0, 10.0)


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half (average ranks)."""
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ridge(X, y, lam, w=None):
    w = np.ones(len(X)) if w is None else w
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ y)


def emobank():
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "emobank", "emobank.csv"), encoding="utf-8")))
    out = {}
    for s in ("train", "dev", "test"):
        v = [x for x in rows if x["split"] == s]
        out[s] = (embed([x["text"] for x in v], "minilm", "doc", tag="emobank"),
                  np.array([[float(x["V"]), float(x["A"])] for x in v]))
    return out


def goemotions():
    d = os.path.join(DATA_DIR, "goemotions")
    names = open(os.path.join(d, "emotions.txt"), encoding="utf-8").read().split()
    groups = json.load(open(os.path.join(d, "sentiment_mapping.json")))
    sign = {e: 1.0 for e in groups["positive"]} | {e: -1.0 for e in groups["negative"]}
    sign["neutral"] = 0.0
    out = {}
    for s in ("train", "dev", "test"):
        texts, ys = [], []
        for line in open(os.path.join(d, f"{s}.tsv"), encoding="utf-8"):
            text, labels, _ = line.rstrip("\n").split("\t")
            vals = {sign.get(names[int(i)]) for i in labels.split(",")}
            if len(vals) == 1 and None not in vals:          # one clear polarity
                texts.append(text)
                ys.append(vals.pop())
        out[s] = (embed(texts, "minilm", "doc", tag="goemotions"), np.array(ys))
    return out


def scores(Wv, Wa, eb, ge, split):
    X, Y = eb[split]
    G, g = ge[split]
    pv = G @ Wv
    return {"emobank_r_valence": r(X @ Wv, Y[:, 0]), "emobank_r_arousal": r(X @ Wa, Y[:, 1]),
            "goemotions_auc_pos_vs_neg": auc(pv[g > 0], pv[g < 0]),
            "goemotions_r_valence": r(pv, g),
            # where 0 sits: the probe's mean on neutral chat and on average EmoBank text
            "goemotions_neutral_mean_valence": float(pv[g == 0].mean()),
            "emobank_mean_valence": float((X @ Wv).mean())}


def main():
    eb, ge = emobank(), goemotions()
    mu, sd = eb["train"][1].mean(0), eb["train"][1].std(0)
    zv = (eb["train"][1][:, 0] - mu[0]) / sd[0]
    za = (eb["train"][1][:, 1] - mu[1]) / sd[1]
    Xe, Xg, yg = eb["train"][0], ge["train"][0], ge["train"][1]
    # no intercept in the ridge, so centre each corpus's targets: 0 stays "emotionally average"
    yg = yg - yg.mean()
    out = {"n": {k: {s: len(v[s][0]) for s in v} for k, v in (("emobank", eb), ("goemotions", ge))},
           "candidates": {}, "test": {}}
    best = None
    arousal = {lam: ridge(Xe, za, lam) for lam in LAMBDAS}
    for name in ("emobank-only", "emobank+goemotions"):
        for lam in LAMBDAS:
            Wa = arousal[lam]
            if name == "emobank-only":
                Wv = ridge(Xe, zv, lam)
            else:
                w = np.concatenate([np.ones(len(Xe)), np.full(len(Xg), len(Xe) / len(Xg))])
                Wv = ridge(np.vstack([Xe, Xg]), np.concatenate([zv, yg]), lam, w)
            dev = scores(Wv, Wa, eb, ge, "dev")
            sel = (dev["emobank_r_valence"] + dev["emobank_r_arousal"] + 2 * (dev["goemotions_auc_pos_vs_neg"] - 0.5)) / 3
            out["candidates"][f"{name}@{lam}"] = {"dev": dev, "selection": sel}
            if best is None or sel > best[0]:
                best = (sel, name, lam, Wv, Wa)
            if lam == 1.0:        # the previously shipped setting, for comparison on test
                out["test"][name] = scores(Wv, Wa, eb, ge, "test")
    _, name, lam, Wv, Wa = best
    out["shipped"] = {"probe": name, "lambda": lam, "test": scores(Wv, Wa, eb, ge, "test")}
    # the store embeds "speaker: text", so score the shipped probe on that input too
    rows = [x for x in csv.DictReader(open(os.path.join(DATA_DIR, "emobank", "emobank.csv"), encoding="utf-8"))
            if x["split"] == "test"]
    Xp = embed(["user: " + x["text"] for x in rows], "minilm", "doc", tag="emobank")
    out["shipped"]["test_with_speaker_prefix"] = {"emobank_r_valence": r(Xp @ Wv, eb["test"][1][:, 0]),
                                                  "emobank_r_arousal": r(Xp @ Wa, eb["test"][1][:, 1])}

    from ncm.affect import ANCHORS
    axes = []
    for pos, neg in ANCHORS.values():
        axes.append(embed(pos, "minilm", "doc", tag="anchors").mean(0) - embed(neg, "minilm", "doc", tag="anchors").mean(0))
    out["test"]["anchor-sentences"] = scores(axes[0], axes[1], eb, ge, "test")

    np.save(os.path.join(HERE, "..", "ncm", "affect_minilm.npy"), np.stack([Wv, Wa], axis=1).astype(np.float32))
    json.dump(out, open(os.path.join(HERE, "results", "affect.json"), "w"), indent=1)
    print(json.dumps({"test": out["test"], "shipped": out["shipped"]}, indent=1))


if __name__ == "__main__":
    main()
