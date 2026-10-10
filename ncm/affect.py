"""Valence and arousal read off the embedding the store already has: no extra model call.

For the default MiniLM model a linear probe fitted on EmoBank human ratings is used
(test Pearson r: valence 0.62, arousal 0.44; see bench/affect_fit.py). Any other
embedder falls back to anchor sentences (valence r 0.45, arousal r 0.16 with MiniLM).
Outputs are roughly z-scores: 0 is an emotionally average sentence, +-1 is strong.
"""

import os

import numpy as np

ANCHORS = {
    "valence": (["I'm so happy, this is wonderful news!", "I love this, thank you so much.",
                 "What a great day, everything went well.", "I'm proud and excited about it."],
                ["I'm so sad, this is terrible news.", "I hate this, it's awful.",
                 "What a horrible day, everything went wrong.", "I feel hurt, lonely and miserable."]),
    "arousal": (["I'm furious and shaking, I can't believe it!", "This is so exciting, I can't wait!!",
                 "Help, it's an emergency, hurry!", "I'm terrified and panicking right now."],
                ["I'm calm and relaxed, just resting.", "It's a quiet, sleepy afternoon.",
                 "Nothing much happening, feeling mellow.", "I'm tired and winding down for bed."]),
}
_PROBES = {"sentence-transformers/all-MiniLM-L6-v2": "affect_minilm.npy",
           "all-MiniLM-L6-v2": "affect_minilm.npy"}


class Affect:
    def __init__(self, embedder):
        probe = _PROBES.get(embedder.name)
        if probe:
            self.W = np.load(os.path.join(os.path.dirname(__file__), probe))
        else:
            cols = []
            for pos, neg in ANCHORS.values():
                p, n = embedder(pos).mean(0), embedder(neg).mean(0)
                d = p - n
                cols.append(2 * d / max(float(d @ d), 1e-9))   # pos anchors -> ~+1, neg -> ~-1
            self.W = np.stack(cols, axis=1).astype(np.float32)

    def __call__(self, vecs: np.ndarray) -> np.ndarray:
        """(n, dim) unit vectors -> (n, 2) [valence, arousal]."""
        return vecs @ self.W


def salience(va: np.ndarray) -> np.ndarray:
    """Emotional weight of a memory, 0..1.5 (added to its activation, in log units)."""
    return np.clip(0.5 * (np.abs(va[:, 0]) + np.maximum(va[:, 1], 0)), 0, 1.5)


def label(valence: float, arousal: float) -> str:
    v = "positive" if valence > 0.4 else "negative" if valence < -0.4 else "neutral"
    a = "energetic" if arousal > 0.4 else "calm" if arousal < -0.4 else ""
    return f"{v}, {a}" if a else v
