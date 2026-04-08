"""
NCM - Memory storage for episodic experiences.

Each memory is a geometric object in the four-dimensional retrieval space.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from ncm.exceptions import (
    DimensionMismatchError,
    MemoryNotFoundError,
    EmptyStoreError,
)
from ncm.profile import MemoryProfile


@dataclass
class MemoryEntry:
    """
    A single episodic memory as a geometric object.
    
    Fields define position in the four-dimensional retrieval space:
      e_semantic:  semantic embedding (what happened)
      e_emotional: emotional projection via W_emo (how it felt)
      s_snapshot:  L2-normalized state copy (who you were)
      timestamp:   step number (when it happened)
      strength:    reinforcement accumulator with bounded growth
    """
    e_semantic: np.ndarray
    e_emotional: np.ndarray
    s_snapshot: np.ndarray
    timestamp: int
    strength: float = 1.0
    text: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "e_semantic": self.e_semantic.tolist(),
            "e_emotional": self.e_emotional.tolist(),
            "s_snapshot": self.s_snapshot.tolist(),
            "timestamp": self.timestamp,
            "strength": float(self.strength),
            "text": self.text,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d["id"],
            e_semantic=np.array(d["e_semantic"], dtype=np.float32),
            e_emotional=np.array(d["e_emotional"], dtype=np.float32),
            s_snapshot=np.array(d["s_snapshot"], dtype=np.float32),
            timestamp=int(d["timestamp"]),
            strength=float(d["strength"]),
            text=d.get("text", ""),
            tags=d.get("tags", []),
        )


class MemoryStore:
    """
    Storage and lifecycle management for episodic memories.
    
    Strength bounds:
      Max strength = 2.0 (prevents unbounded reinforcement growth)
      Math justification: In Hebbian learning, synaptic weights are bounded
      to prevent runaway excitation. Analogously, memory strength must be
      bounded to keep the reinforcement signal meaningful.
      
      Decay rate = 0.999 per tick (half-life ≈ 693 steps)
      Math: 0.999^693 ≈ 0.500
    """

    def __init__(self, profile: Optional[MemoryProfile] = None):
        self.profile = profile or MemoryProfile()
        self._memories: dict = {}
        self.step: int = 0
        # Vectorized cache for fast retrieval
        self._sem_cache = None
        self._emo_cache = None
        self._state_cache = None
        self._ts_cache = None
        self._str_cache = None
        self._id_order = []
        self._cache_dirty = True

    def _invalidate_cache(self):
        self._cache_dirty = True

    def _rebuild_cache(self):
        """Build numpy arrays for vectorized retrieval."""
        if not self._cache_dirty:
            return
        if not self._memories:
            self._sem_cache = np.zeros((0, self.profile.semantic_dim), dtype=np.float32)
            self._emo_cache = np.zeros((0, self.profile.emotional_dim), dtype=np.float32)
            self._state_cache = np.zeros((0, self.profile.state_dim), dtype=np.float32)
            self._ts_cache = np.zeros(0, dtype=np.int64)
            self._str_cache = np.zeros(0, dtype=np.float32)
            self._id_order = []
            self._cache_dirty = False
            return

        self._id_order = list(self._memories.keys())
        mems = [self._memories[mid] for mid in self._id_order]
        self._sem_cache = np.array([m.e_semantic for m in mems], dtype=np.float32)
        self._emo_cache = np.array([m.e_emotional for m in mems], dtype=np.float32)
        self._state_cache = np.array([m.s_snapshot for m in mems], dtype=np.float32)
        self._ts_cache = np.array([m.timestamp for m in mems], dtype=np.int64)
        self._str_cache = np.array([m.strength for m in mems], dtype=np.float32)
        self._cache_dirty = False

    def add(self, memory: MemoryEntry) -> MemoryEntry:
        expected_sem = self.profile.semantic_dim
        if memory.e_semantic.shape != (expected_sem,):
            raise DimensionMismatchError(
                expected=(expected_sem,), got=memory.e_semantic.shape, context="e_semantic"
            )

        if len(self._memories) >= self.profile.max_size:
            self._evict_weakest()

        self._memories[memory.id] = memory
        self._invalidate_cache()
        return memory

    def _evict_weakest(self) -> None:
        if not self._memories:
            return
        weakest_id = min(self._memories, key=lambda mid: self._memories[mid].strength)
        del self._memories[weakest_id]
        self._invalidate_cache()

    def get(self, memory_id: str) -> MemoryEntry:
        if memory_id not in self._memories:
            raise MemoryNotFoundError(memory_id)
        return self._memories[memory_id]

    def get_all(self) -> List[MemoryEntry]:
        if not self._memories:
            raise EmptyStoreError("Cannot retrieve from empty MemoryStore")
        return list(self._memories.values())

    def get_all_safe(self) -> List[MemoryEntry]:
        return list(self._memories.values())

    def reinforce(self, memory_id: str, amount: float = 0.1) -> None:
        memory = self.get(memory_id)
        memory.strength = min(memory.strength + amount, 2.0)
        self._invalidate_cache()

    def decay_all(self) -> None:
        for memory in self._memories.values():
            memory.strength *= 0.999
        self._invalidate_cache()

    def filter_by_tag(self, tag: str) -> List[MemoryEntry]:
        return [m for m in self._memories.values() if tag in m.tags]

    def tick(self) -> None:
        self.step += 1
        self.decay_all()

    def summary(self) -> dict:
        if not self._memories:
            return {"count": 0, "step": self.step, "avg_strength": 0.0,
                    "min_strength": 0.0, "max_strength": 0.0, "profile_name": self.profile.name}
        strengths = [m.strength for m in self._memories.values()]
        return {
            "count": len(self._memories), "step": self.step,
            "avg_strength": float(np.mean(strengths)),
            "min_strength": float(np.min(strengths)),
            "max_strength": float(np.max(strengths)),
            "profile_name": self.profile.name,
        }

    def get_semantic_matrix(self) -> np.ndarray:
        """Return (N, semantic_dim) matrix of all semantic vectors. For fast retrieval."""
        self._rebuild_cache()
        return self._sem_cache

    def __len__(self) -> int:
        return len(self._memories)

    def __repr__(self) -> str:
        return f"MemoryStore(name='{self.profile.name}', size={len(self)}, step={self.step})"
