"""NCM - Personalization profile with learnable retrieval weights."""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from ncm.exceptions import ProfileError


@dataclass
class RetrievalWeights:
    """
    Learnable weights for four-component retrieval.
    Must sum to 1.0. Supports Dirichlet regularization.
    
    Math justification:
      The weight vector w = [alpha, beta, gamma, delta] lives on the 3-simplex.
      Dirichlet(a0, a0, a0, a0) is the conjugate prior for categorical distributions.
      KL(w || Dir(a0)) penalizes deviation from uniform, preventing manifold collapse.
    """
    alpha: float = 0.4   # semantic
    beta: float = 0.2    # emotional
    gamma: float = 0.3   # state
    delta: float = 0.1   # temporal

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ProfileError(f"Retrieval weights must sum to 1.0, got {total:.4f}")
        for name, val in [("alpha", self.alpha), ("beta", self.beta),
                          ("gamma", self.gamma), ("delta", self.delta)]:
            if val < 0:
                raise ProfileError(f"Weight {name} must be >= 0, got {val}")

    def as_tuple(self) -> tuple:
        return (self.alpha, self.beta, self.gamma, self.delta)

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma, self.delta], dtype=np.float64)

    def dirichlet_kl(self, a0: float = 1.0) -> float:
        """
        KL divergence from weights to symmetric Dirichlet(a0).
        
        Math:
          KL(w || Dir(a0)) = sum_i [ (w_i - a0/sum(a0)) * (psi(w_i*K) - psi(a0)) ]
          
        Simplified for our use: we treat w as the mean of a Dirichlet and compute
        KL between Dir(w*K) and Dir(a0, a0, a0, a0).
        
        For regularization, we use the simpler proxy:
          L_balance = sum_i (w_i - 0.25)^2
          
        This is the L2 penalty toward uniform, which is the gradient of KL 
        near the uniform point and computationally stable.
        """
        uniform = 0.25
        w = self.as_array()
        return float(np.sum((w - uniform) ** 2))

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "delta": self.delta}

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalWeights":
        return cls(**d)


@dataclass
class MemoryProfile:
    """
    Profile that travels with every .ncm file.
    Defines identity and retrieval behavior.
    """
    name: str = "default"
    retrieval_weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    semantic_dim: int = 128
    emotional_dim: int = 3
    state_dim: int = 7
    decay_rate: float = 0.001
    write_threshold: float = 0.15
    max_size: int = 10000
    temperature: float = 0.1  # softmax retrieval temperature
    version: str = "2.0"
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "retrieval_weights": self.retrieval_weights.to_dict(),
            "semantic_dim": self.semantic_dim,
            "emotional_dim": self.emotional_dim,
            "state_dim": self.state_dim,
            "decay_rate": self.decay_rate,
            "write_threshold": self.write_threshold,
            "max_size": self.max_size,
            "temperature": self.temperature,
            "version": self.version,
            "custom": self.custom,
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryProfile":
        d = d.copy()
        if "retrieval_weights" in d:
            d["retrieval_weights"] = RetrievalWeights.from_dict(d["retrieval_weights"])
        return cls(**d)

    @classmethod
    def from_json(cls, data: bytes) -> "MemoryProfile":
        try:
            d = json.loads(data.decode("utf-8"))
            return cls.from_dict(d)
        except Exception as e:
            raise ProfileError(f"Failed to parse profile: {e}")

    def set_custom(self, key: str, value: Any) -> None:
        self.custom[key] = value

    def get_custom(self, key: str, default: Any = None) -> Any:
        return self.custom.get(key, default)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "MemoryProfile":
        """Build a profile from a named, measured configuration.

        Keyword overrides are applied after the preset, so
        from_preset("temporal_contiguity", max_size=50000) keeps the measured
        retrieval behaviour and changes only capacity. Overriding
        retrieval_weights or custom replaces the preset's value outright rather
        than merging, because a partially overridden weight vector would silently
        stop summing to one and a partially overridden custom dict would leave
        the anchor set with no width to go with it.

        Read describe_preset(preset) first. Every preset beats the shipped
        default on the task it was measured on, and some are worse elsewhere.
        """
        try:
            spec = PRESETS[preset]
        except KeyError:
            raise ProfileError(
                f"Unknown preset {preset!r}. Available: "
                f"{', '.join(sorted(PRESETS))}"
            )
        kwargs = {
            "name": preset,
            "retrieval_weights": RetrievalWeights(*spec["weights"]),
            # Copied, so a caller mutating profile.custom cannot edit the preset
            # for every profile built afterwards in the same process.
            "custom": dict(spec["custom"]),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Named profiles measured on held-out data.
#
# A preset exists so a configuration that beats the shipped default on a stated
# task can be reached in one line without that default moving underneath callers
# who never asked for it. Every entry carries the measurement that justifies it,
# the split it was measured on, and the reason it is not the default. A preset
# with no stated evidence is only a set of magic numbers with a friendly name.
# ---------------------------------------------------------------------------

PRESETS = {
    "temporal_contiguity": {
        "weights": (0.5, 0.0, 0.0, 0.5),
        "custom": {
            "temporal_anchor": "semantic_rank1",
            "temporal_kernel_width": 4.0,
        },
        "use_when": (
            "Retrieving the neighbourhood of a past episode: the memories written "
            "around the same time as the one the query most resembles. Session "
            "retrieval, conversation resumption, and 'what were we doing when we "
            "talked about X' are this shape."
        ),
        "measured": (
            "Multi-Session Chat, held-out test split, 501 conversations, 2505 "
            "queries, mean store 55.5 memories, random guessing P@5 0.2000. "
            "P@5 0.5436 against 0.4242 for the shipped default, a gain of "
            "+0.1194 over 2136 changed queries, Wilcoxon exact p 9.92e-45. "
            "nDCG@10 0.5106, MRR 0.6114. Attribution is clean: the same weights "
            "with the default store_end anchor score 0.4255, a null result at "
            "p 0.82, so the whole gain belongs to the anchor and none of it to "
            "the reweighting."
        ),
        "caveats": (
            "Two honest limits. First, the benchmark label is session membership "
            "and a session is a contiguous run of timestamps, so a contiguity "
            "kernel is favoured by the construction of the label: an arm using "
            "the kernel alone and ignoring content entirely scores 0.5366, close "
            "behind the full configuration. Second, MRR falls about 0.10 against "
            "the default because the rank-1 semantic anchor lands in the correct "
            "session only 57 percent of the time, and a missed anchor centres the "
            "kernel on the wrong episode. Prefer the default when the query has "
            "no episode to sit next to, such as single-fact lookup."
        ),
        "notes": (
            "beta and gamma are zero because both channels measured near-inert "
            "on this task, the emotional channel at AUC 0.514 with candidate "
            "spread 0.013 and the state channel at 0.525 with spread 0.040. "
            "decay_rate is unread under this anchor: the rate is "
            "1/temporal_kernel_width, so the width in turns is the only temporal "
            "knob that acts."
        ),
    },
}


def list_presets() -> dict:
    """Preset names paired with the task each was measured on."""
    return {name: spec["use_when"] for name, spec in PRESETS.items()}


def describe_preset(name: str) -> str:
    """The full record for one preset: what it is for, what it measured, and
    where it is known to be worse. Meant to be printed before adopting one."""
    try:
        spec = PRESETS[name]
    except KeyError:
        raise ProfileError(
            f"Unknown preset {name!r}. Available: {', '.join(sorted(PRESETS))}"
        )
    alpha, beta, gamma, delta = spec["weights"]
    return "\n".join([
        f"preset: {name}",
        f"weights: alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} "
        f"delta={delta:.2f}",
        f"custom: {spec['custom']}",
        f"use when: {spec['use_when']}",
        f"measured: {spec['measured']}",
        f"caveats: {spec['caveats']}",
        f"notes: {spec['notes']}",
    ])
