"""
NCM - Text and state encoding system.

FIXES from v1:
  1. Semantic projection is now documented as Johnson-Lindenstrauss random projection
     (not "trained projector"). The construction is the standard JL one, but the JL
     distance-preservation bound is NOT satisfied at the dimensions shipped here.
     `verify_math()` computes the bound and records jl_satisfied=False for the
     shipped configuration (semantic_dim=128). See the note at the projection
     construction below for the arithmetic. Treat the projection as an empirically
     adequate compression with uncharacterized distortion, not as a guarantee.
     
  2. Emotional encoding now returns BOTH the projected vector AND exposes
     encode_emotional() so retrieval can compare projected-to-projected.
     
  3. Added encode_state() for proper state normalization.

ENCODING GATE:
  The write gate is a nearest-neighbour novelty score, not an information
  measure:
    novelty(x, M) = clip(1 - max_j cos(x, m_j), 0, 1)
  over the stored memory vectors m_j. If novelty is below a threshold, the
  experience is treated as too predictable to store.

  An earlier version of this docstring described the gate as
  "gate(x, S) = H(x|S) / H_max", a conditional entropy normalised by the
  maximum entropy. That description was wrong. `encoding_gate` estimates no
  probability distribution, computes no entropy, and has no H_max term, so the
  gate is not information-theoretic and should not be described as such.

  Selective encoding here is a design choice motivated by bounded
  storage, not a result taken from a cited source. Ebbinghaus (1885)
  concerns retention and forgetting over time, not what is written at
  encoding time, so it is not a warrant for this gate.
"""

import os
import warnings

import numpy as np

from ncm.exceptions import (
    InvalidStateVectorError,
)


class SentenceEncoder:
    """
    Encodes text and state vectors into geometric form for NCM.
    
    Encoding methods:
      encode(text)              -> semantic_dim vector (L2-normalized)
      encode_emotional(state)   -> emotional_dim vector (L2-normalized)  
      encode_state(state)       -> state_dim vector (L2-normalized to unit ball)
      encode_batch(texts)       -> (N, semantic_dim) matrix
      encoding_gate(text, memories) -> float in [0,1], novelty signal
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_dir: str = "models/",
        semantic_dim: int = 128,
        emotional_dim: int = 3,
        state_dim: int = 7,
        seed: int = 42,
        device: str = "auto",
        require_gpu: bool = False,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.semantic_dim = semantic_dim
        self.emotional_dim = emotional_dim
        self.state_dim = state_dim
        self.seed = seed
        self.device = device
        self.require_gpu = require_gpu
        self._model = None
        self._projection = None
        self._w_emo = None
        self._initialized = False
        self._backend_error = None

    @property
    def backend(self) -> str:
        """
        Which encoder actually produced the vectors: "sentence-transformers"
        or "hash-fallback".

        Experiment scripts should serialize this into their result files so a
        reader can tell whether reported numbers came from a real semantic
        encoder or from the meaningless hash fallback. Accessing this property
        initializes the encoder if it is not already initialized.
        """
        self._ensure_initialized()
        return "hash-fallback" if self._model is None else "sentence-transformers"

    @property
    def backend_error(self) -> str:
        """The exception that forced the hash fallback, or None if not used."""
        self._ensure_initialized()
        return self._backend_error

    def _resolve_device(self) -> str:
        """Resolve runtime device for SentenceTransformer."""
        if self.device and self.device.lower() not in {"", "auto"}:
            return self.device.lower()

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        os.makedirs(self.model_dir, exist_ok=True)
        local_path = os.path.join(self.model_dir, self.model_name)

        try:
            from sentence_transformers import SentenceTransformer
            resolved_device = self._resolve_device()
            if self.require_gpu and not resolved_device.startswith("cuda"):
                raise RuntimeError("GPU is required (require_gpu=True), but CUDA is not available.")
            if os.path.exists(local_path):
                self._model = SentenceTransformer(local_path, device=resolved_device)
            else:
                self._model = SentenceTransformer(self.model_name, device=resolved_device)
                self._model.save(local_path)
        except Exception as exc:
            if self.require_gpu:
                raise
            # Fallback: deterministic hash-based encoder.
            #
            # This path carries NO semantic information: it is a SHA-512 digest
            # reinterpreted as floats. Retrieval still runs and still returns
            # results, so a silent fallback would make meaningless numbers
            # indistinguishable from real ones. Any experiment that lands here
            # must be discarded, so we warn loudly and record the reason in
            # `backend_error` for result files to serialize.
            self._model = None
            self._backend_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "SentenceEncoder could not load the sentence-transformers model "
                f"'{self.model_name}' ({self._backend_error}). Falling back to a "
                "deterministic SHA-512 hash encoder. This fallback preserves NO "
                "semantic structure; any retrieval quality measured in this state "
                "is meaningless and must not be reported. Pass require_gpu=True or "
                "install sentence-transformers to make this a hard error.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Semantic projection: Johnson-Lindenstrauss random projection.
        #
        # NOTE ON THE JL BOUND: the classical bound is
        #   k >= 4*ln(n) / (epsilon^2/2 - epsilon^3/3).
        # For n=1e5 and epsilon=0.1 this requires k >= 9868, and even as
        # epsilon -> 1 it requires k >= 277. The 128 dimensions used here
        # therefore DO NOT satisfy the JL bound at the scales benchmarked, and
        # `verify_math()` records this as jl_satisfied=False. The projection is
        # an empirically adequate compression whose distortion is
        # uncharacterized; it is not a distance-preserving guarantee.
        rng = np.random.RandomState(self.seed)
        # Scale factor 1/sqrt(k), as in the standard JL construction.
        self._projection = rng.randn(384, self.semantic_dim).astype(np.float32)
        self._projection /= np.sqrt(self.semantic_dim)

        # Emotional projection: W_emo ∈ R^(emotional_dim × state_dim)
        # Orthonormal via QR decomposition (constructed from a random matrix).
        # Math: W_emo · W_emo^T ≈ I_k numerically. This preserves geometric
        # independence of emotional dimensions and prevents information collapse
        # in retrieval. Note: numeric orthonormality is subject to floating-point
        # precision. The experiment suite records a small orthonormality error on the
        # order of 1e-7 (see experiments outputs). Treat this as an empirical
        # observation rather than an unconditional mathematical equality.
        # for typical runs (see `experiments/results/run_all_experiments/math_verification.json`).
        rng2 = np.random.RandomState(7)
        raw2 = rng2.randn(self.state_dim, self.emotional_dim).astype(np.float32)
        Q, _ = np.linalg.qr(raw2)
        self._w_emo = Q[:, :self.emotional_dim].T.astype(np.float32)  # (emotional_dim, state_dim)

        self._initialized = True

    def encode(self, text: str) -> np.ndarray:
        """Encode text -> semantic_dim L2-normalized vector."""
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        self._ensure_initialized()

        if self._model is not None:
            raw = self._model.encode(
                text, convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
        else:
            # Deterministic fallback for testing without sentence-transformers
            raw = self._deterministic_encode(text)

        projected = raw @ self._projection
        norm = np.linalg.norm(projected)
        if norm < 1e-8:
            return projected
        return (projected / norm).astype(np.float32)

    def _deterministic_encode(self, text: str) -> np.ndarray:
        """
        Hash-based deterministic encoder, used only when the sentence-transformer
        model cannot be loaded (see the RuntimeWarning in _ensure_initialized).

        Returns a 384-dimensional L2-normalized vector so that it is shape-
        compatible with the semantic projection. It carries NO semantic
        structure: cosine similarity between two such vectors is meaningless.
        Callers must consult `backend` before reporting any retrieval quality.

        BUGFIX: this previously built the buffer from 6 copies of a SHA-512
        digest. A digest is 64 bytes, so 6 copies is 384 *bytes*, which is only
        96 float32 values, not 384. The subsequent matmul against the (384, k)
        projection therefore raised a dimension-mismatch ValueError on every
        call, meaning this fallback could never actually execute end to end.
        We now derive the full 384 floats from a hash-seeded PRNG, which is
        both correctly shaped and better distributed than reinterpreted digest
        bytes (raw digest bytes decode to wildly varying float magnitudes,
        including denormals and NaN/Inf bit patterns).
        """
        import hashlib

        digest = hashlib.sha512(text.encode("utf-8")).digest()
        # Seed a PRNG from the digest so the mapping stays deterministic.
        seed = int.from_bytes(digest[:8], "little", signed=False) % (2 ** 32)
        rng = np.random.RandomState(seed)
        arr = rng.randn(384).astype(np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 1e-8 else arr

    def encode_emotional(self, state: np.ndarray) -> np.ndarray:
        """
        Project state vector into emotional subspace via W_emo.
        
        FIX from v1: This is now the ONLY way to get emotional vectors.
        Both memory encoding AND retrieval query must use this function,
        ensuring projected-to-projected comparison in the same space.
        
        Math: e_emotional = W_emo · s_padded, then L2-normalize
        Constraint: W_emo · W_emo^T = I_k (orthonormal)
        """
        self._ensure_initialized()
        state = np.asarray(state, dtype=np.float32).copy()

        if np.any(state < -0.01) or np.any(state > 1.01):
            raise InvalidStateVectorError(
                f"Values must be in [0, 1], got min={state.min():.3f} max={state.max():.3f}"
            )
        state = np.clip(state, 0.0, 1.0)

        # Pad or truncate to state_dim
        if len(state) < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(state)] = state
            state = padded
        else:
            state = state[:self.state_dim]

        projected = self._w_emo @ state  # (emotional_dim,)
        norm = np.linalg.norm(projected)
        if norm < 1e-8:
            return projected
        return (projected / norm).astype(np.float32)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state vector for s_snapshot storage.
        
        Maps to unit ball: values in [0,1]^n, then L2-normalize.
        This ensures Euclidean distance in state space is bounded
        and comparable across different state dimensionalities.
        
        Max Euclidean distance between two L2-normalized vectors = 2.0
        (when vectors point in opposite directions).
        For vectors in positive orthant (all values >= 0): max = sqrt(2).
        """
        state = np.asarray(state, dtype=np.float32).copy()
        state = np.clip(state, 0.0, 1.0)

        if len(state) < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(state)] = state
            state = padded
        else:
            state = state[:self.state_dim]

        norm = np.linalg.norm(state)
        if norm < 1e-8:
            return state
        return (state / norm).astype(np.float32)

    def encode_batch(self, texts: list, batch_size: int = 128) -> np.ndarray:
        """
        Encode multiple texts efficiently with adaptive batching.
        
        OPTIMIZATION: Process texts in batches to maintain L2 cache efficiency
        and reduce memory fragmentation. Large batch_size (128) amortizes Python
        loop overhead while staying within typical GPU memory budgets.
        """
        if not texts:
            return np.zeros((0, self.semantic_dim), dtype=np.float32)

        self._ensure_initialized()

        if self._model is not None:
            raw = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size,
            ).astype(np.float32)
        else:
            raw = np.array([self._deterministic_encode(t) for t in texts])

        # OPTIMIZATION: Use in-place operations and vectorized norm computation
        # Avoid creating intermediate arrays where possible
        projected = raw @ self._projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # OPTIMIZATION: Safe division with branch prediction-friendly comparison
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (projected / norms).astype(np.float32)

    def encoding_gate(self, query_vec: np.ndarray, memory_vecs: np.ndarray) -> float:
        """
        Nearest-neighbour novelty gate for writes.

        This is not an information-theoretic quantity. It estimates no
        distribution and computes no entropy.

          novelty(q, M) = clip(1 - max_similarity(q, M), 0, 1)
          where similarity is cosine similarity to all stored memories.

          If novelty < threshold, the input is too predictable to store.
          Discarding predictable input is a design choice in NCM, made to
          keep storage bounded. It is not a reproduction of a published
          empirical result.

        For empty memory, returns 1.0 (everything is novel).
        """
        if memory_vecs is None or len(memory_vecs) == 0:
            return 1.0
        
        # Cosine similarities (query and memories are L2-normalized)
        sims = memory_vecs @ query_vec
        max_sim = float(np.max(sims))
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property 
    def w_emo(self) -> np.ndarray:
        """Expose W_emo for verification of orthonormality."""
        self._ensure_initialized()
        return self._w_emo.copy()

    def __repr__(self) -> str:
        status = "ready" if self._initialized else "not loaded"
        return f"SentenceEncoder(model='{self.model_name}', semantic_dim={self.semantic_dim}, status={status})"
