"""
NCM (Native Cognitive Memory)
Tensor-based episodic memory with state-conditioned retrieval for AI systems.

Quick start:
    from ncm import SentenceEncoder, MemoryStore, MemoryEntry, retrieve_top_k

    encoder = SentenceEncoder()
    store = MemoryStore()

    state = [0.7, 0.8, 0.2, 0.8, 0.3, 0.7, 0.2]
    mem = MemoryEntry(
        e_semantic=encoder.encode("something happened"),
        e_emotional=encoder.encode_emotional(state),
        s_snapshot=encoder.encode_state(state),
        timestamp=0,
    )
    store.add(mem)
"""

__version__ = "1.1.0"

from ncm.encoder import SentenceEncoder
from ncm.auto_state import AutoStateTracker
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile, RetrievalWeights
from ncm.retrieval import (
    retrieve_top_k,
    retrieve_top_k_fast,
    retrieve_semantic_only,
    retrieve_semantic_emotional,
    retrieval_entropy,
    vectorized_manifold_distance,
)
from ncm.persistence import NCMFile
from ncm.exceptions import (
    NCMError,
    DimensionMismatchError,
    MemoryNotFoundError,
    EmptyStoreError,
    EncoderNotInitializedError,
    PersistenceError,
    InvalidStateVectorError,
    CorruptFileError,
    ProfileError,
)

__all__ = [
    # Core
    "SentenceEncoder",
    "AutoStateTracker",
    "MemoryEntry",
    "MemoryStore",
    "MemoryProfile",
    "RetrievalWeights",
    # Retrieval
    "retrieve_top_k",
    "retrieve_top_k_fast",
    "retrieve_semantic_only",
    "retrieve_semantic_emotional",
    "retrieval_entropy",
    "vectorized_manifold_distance",
    # Persistence
    "NCMFile",
    # Exceptions
    "NCMError",
    "DimensionMismatchError",
    "MemoryNotFoundError",
    "EmptyStoreError",
    "EncoderNotInitializedError",
    "PersistenceError",
    "InvalidStateVectorError",
    "CorruptFileError",
    "ProfileError",
]
