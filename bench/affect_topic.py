"""Does the affect probe read the feeling, or the topic? A controlled 2x2 test.

Sentences cross a feeling (positive / negative) with a topic (pleasant / unpleasant):
  "I was thrilled about my wedding"       feeling +, topic +   (agree)
  "I was miserable about my wedding"      feeling -, topic +   (conflict)
  "I was thrilled about the funeral"      feeling +, topic -   (conflict)
  "I was miserable about the funeral"     feeling -, topic -   (agree)
The probe should follow the feeling. If topic words fool it, its score tracks the topic
and its sign is wrong more often when feeling and topic conflict. A dedicated sentiment
classifier (DistilBERT fine-tuned on SST-2) is scored on the same sentences as a reference.

  python -m bench.affect_topic        # writes bench/results/affect-topic.json

Also re-scores the shipped probe file on the held-out EmoBank and GoEmotions test splits,
so the published numbers are checked against what ships, not against the fitting run.
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.affect_fit import auc, emobank, goemotions, r   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POS_FEEL = ["I was thrilled about", "I felt so happy about", "I loved", "I was grateful for",
            "I was excited about", "I felt proud of", "I really enjoyed", "I felt at peace with"]
NEG_FEEL = ["I was miserable about", "I felt so sad about", "I hated", "I was furious about",
            "I was dreading", "I felt ashamed of", "I really resented", "I felt sick about"]
POS_TOPIC = ["my wedding", "my birthday party", "the holiday trip", "my promotion", "the concert",
             "graduation day", "the new puppy", "the beach vacation", "the surprise gift", "the family reunion"]
NEG_TOPIC = ["the funeral", "the hospital stay", "the court hearing", "the layoffs", "the car crash",
             "the divorce papers", "the root canal", "the eviction notice", "the final exam", "the flood damage"]


def sentences():
    out = []
    for feel, fsign in ((POS_FEEL, 1), (NEG_FEEL, -1)):
        for topic, tsign in ((POS_TOPIC, 1), (NEG_TOPIC, -1)):
            for f, t in itertools.product(feel, topic):
                out.append({"text": f"{f} {t}.", "feeling": fsign, "topic": tsign})
    return out


def probe_scores(texts):
    from ncm.affect import Affect
    from ncm.embed import Embedder
    emb = Embedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return Affect(emb)(emb([f"user: {t}" for t in texts]))[:, 0]   # embedded as the store embeds


def sst2_scores(texts):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                   device=-1)
    return np.array([p["score"] if p["label"] == "POSITIVE" else -p["score"] for p in clf(texts, batch_size=32)])


def cells(scores, data):
    out = {}
    for f, t in itertools.product((1, -1), (1, -1)):
        idx = [i for i, d in enumerate(data) if d["feeling"] == f and d["topic"] == t]
        s = scores[idx]
        out[f"feeling {'+' if f > 0 else '-'} / topic {'+' if t > 0 else '-'}"] = {
            "n": len(idx), "mean_score": float(s.mean()), "sign_matches_feeling": float(np.mean(np.sign(s) == f))}
    feel = np.array([d["feeling"] for d in data], float)
    topic = np.array([d["topic"] for d in data], float)
    z = (scores - scores.mean()) / scores.std()
    beta = np.linalg.lstsq(np.stack([np.ones_like(feel), feel, topic], 1), z, rcond=None)[0]
    agree = feel == topic
    return {"cells": out,
            "sign_accuracy_when_feeling_and_topic_agree": float(np.mean(np.sign(scores[agree]) == feel[agree])),
            "sign_accuracy_when_they_conflict": float(np.mean(np.sign(scores[~agree]) == feel[~agree])),
            # standardised effect of each factor on the score (both coded +-1)
            "effect_of_feeling": float(beta[1]), "effect_of_topic": float(beta[2])}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = sentences()
    texts = [d["text"] for d in data]
    res = {"n_sentences": len(data), "examples": texts[:2] + texts[-2:],
           "probe": cells(probe_scores(texts), data), "sst2_distilbert": cells(sst2_scores(texts), data)}

    W = np.load(os.path.join(HERE, "..", "ncm", "affect_minilm.npy"))
    eb, ge = emobank(), goemotions()
    X, Y = eb["test"]
    G, g = ge["test"]
    pv = G @ W[:, 0]
    res["shipped_probe_on_test_splits"] = {
        "emobank_r_valence": r(X @ W[:, 0], Y[:, 0]), "emobank_r_arousal": r(X @ W[:, 1], Y[:, 1]),
        "goemotions_auc_pos_vs_neg": auc(pv[g > 0], pv[g < 0])}
    json.dump(res, open(os.path.join(HERE, "results", "affect-topic.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
