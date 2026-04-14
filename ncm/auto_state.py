"""Auto-state tracker for NCM.

Standalone utility module: no imports from memory.py or retrieval.py.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_DIMS = ["valence", "arousal", "dominance", "curiosity", "stress"]
DEFAULT_ALPHA = np.array([0.15, 0.15, 0.15, 0.20, 0.25], dtype=np.float32)

ANCHORS_RAW: Dict[str, Dict[str, str]] = {
    "valence": {
        "pos": "happy joyful wonderful pleased delighted content",
        "neg": "sad depressed terrible awful miserable unhappy",
    },
    "arousal": {
        "pos": "hyper restless racing pumped fired up buzzing wired",
        "neg": "calm relaxed tired sleepy sluggish passive",
    },
    "dominance": {
        "pos": "confident in control powerful capable assertive decisive",
        "neg": "helpless powerless weak submissive",
    },
    "curiosity": {
        "pos": "interesting wonder exploring question curious fascinating",
        "neg": "boring obvious routine dull uninteresting predictable",
    },
    "stress": {
        "pos": "worried anxious deadline pressure overwhelmed tense panicking",
        "neg": "relaxed comfortable safe peaceful serene at ease",
    },
}


_MODEL = None
_ANCHOR_VECS = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        local_path = os.path.join("models", "all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(local_path if os.path.exists(local_path) else "all-MiniLM-L6-v2")
    return _MODEL


def _encode(text: str) -> np.ndarray:
    vec = _get_model().encode(text, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def _get_anchor_vectors() -> Dict[str, Dict[str, np.ndarray]]:
    global _ANCHOR_VECS
    if _ANCHOR_VECS is None:
        _ANCHOR_VECS = {
            dim: {"pos": _encode(v["pos"]), "neg": _encode(v["neg"])}
            for dim, v in ANCHORS_RAW.items()
        }
    return _ANCHOR_VECS


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class AutoStateTracker:
    """Maintains NCM's 5D auto-state trajectory and adaptive blending weights."""

    def __init__(self, alpha: Optional[Iterable[float]] = None):
        alpha_arr = np.array(list(alpha) if alpha is not None else DEFAULT_ALPHA, dtype=np.float32)
        if alpha_arr.shape != (5,):
            raise ValueError(f"alpha must have shape (5,), got {alpha_arr.shape}")

        self.alpha = alpha_arr
        self.state = np.full(5, 0.5, dtype=np.float32)
        self.turn = 0
        self._dims = DEFAULT_DIMS
        self._anchors = _get_anchor_vectors()

    def _sigma(self, e_input: np.ndarray, dim: str) -> float:
        pos = self._anchors[dim]["pos"]
        neg = self._anchors[dim]["neg"]
        return float(np.clip((1.0 + _cosine(e_input, pos) - _cosine(e_input, neg)) / 2.0, 0.0, 1.0))

    def update(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        e = _encode(text)
        signal = np.array([self._sigma(e, d) for d in self._dims], dtype=np.float32)
        self.state = (1.0 - self.alpha) * self.state + self.alpha * signal
        self.turn += 1
        return self.state.copy()

    def get_current_state(self) -> np.ndarray:
        return self.state.copy()

    def get_adaptive_weights(self) -> Tuple[float, float]:
        spread = float(np.max(self.state) - np.min(self.state))
        w_state = float(np.clip(0.3 + 0.4 * spread, 0.3, 0.7))
        w_sem = 1.0 - w_state
        return w_state, float(w_sem)

    def to_dict(self) -> dict:
        return {
            "state": self.state.tolist(),
            "alpha": self.alpha.tolist(),
            "turn": int(self.turn),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoStateTracker":
        tracker = cls(alpha=d.get("alpha", DEFAULT_ALPHA.tolist()))
        tracker.state = np.array(d.get("state", [0.5] * 5), dtype=np.float32)
        if tracker.state.shape != (5,):
            raise ValueError(f"state must have shape (5,), got {tracker.state.shape}")
        tracker.turn = int(d.get("turn", 0))
        return tracker
