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
