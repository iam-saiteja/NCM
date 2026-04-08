"""
NCM - Vectorized manifold retrieval.

FIXES from v1:
  1. Emotional distance now compares projected-to-projected vectors
     (both through W_emo), not projected vs raw state.
     
  2. Normalization constants are DERIVED, not arbitrary:
     - Semantic: cosine distance already in [0, 1]. No normalization needed.
     - Emotional: For L2-normalized vectors in R^k, max Euclidean distance = 2.0
       (opposite directions). Divide by 2.0. DERIVED from ||a-b||² = 2 - 2cos(θ).
     - State: For L2-normalized vectors in positive orthant R^n+, 
       max distance = sqrt(2) ≈ 1.414. Divide by sqrt(2). DERIVED.
     - Temporal: exp decay already in [0, 1]. No normalization needed.
     
  3. ALL retrieval is VECTORIZED with numpy broadcasting.
     No Python loops over memories. O(N) with BLAS-accelerated matmul.

NEW MATH:
  Adaptive temperature:
    T(t) = T_base * (1 + eta * novelty(t))
    where novelty(t) = 1 - max(cosine_sims) over retrieved memories.
    High novelty -> higher T -> more exploratory recall.
    Low novelty -> lower T -> deterministic recall of best match.
    This makes retrieval personality DYNAMIC, not static.
"""

import numpy as np
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import RetrievalWeights


# ───────────────────────────────────────────
# DERIVED NORMALIZATION CONSTANTS
# ───────────────────────────────────────────
# For L2-normalized vectors a, b:
#   ||a - b||² = ||a||² + ||b||² - 2·a·b = 2 - 2·cos(θ)
#   max ||a - b|| = sqrt(2 - 2·(-1)) = 2.0 (when cos(θ) = -1)
#
# For vectors in positive orthant (all components >= 0):
#   cos(θ) >= 0 always, so max ||a - b|| = sqrt(2 - 0) = sqrt(2)
#
EMO_NORM = 2.0        # General L2-normalized vectors
STATE_NORM = np.sqrt(2.0)  # Positive orthant L2-normalized vectors


def vectorized_manifold_distance(
    sem_matrix: np.ndarray,      # (N, d) semantic vectors
    emo_matrix: np.ndarray,      # (N, k) emotional vectors  
    state_matrix: np.ndarray,    # (N, n) state snapshots
    ts_array: np.ndarray,        # (N,) timestamps
    query_semantic: np.ndarray,  # (d,) query semantic vector
    query_emotional: np.ndarray, # (k,) query emotional vector (PROJECTED via W_emo)
    s_current: np.ndarray,       # (n,) current state (L2-normalized)
    current_step: int,
    weights: RetrievalWeights,
    decay_rate: float = 0.001,
    strength_array: np.ndarray = None,  # (N,) memory strengths for strength-weighted retrieval
    strength_boost: float = 0.1,        # how much strength reduces distance
) -> np.ndarray:
    """
    Compute manifold distance for ALL memories at once via vectorized numpy.
    
    Returns (N,) array of distances in [0, 1].
    
    Math:
      d_raw(m, q) = α·d_sem + β·d_emo + γ·d_state + δ·d_time
      d(m, q)     = d_raw · (1 - strength_boost · (strength - 1))
      
      Strength modulation:
        strength=1.0 (default) -> no change
        strength=2.0 (max, heavily reinforced) -> distance reduced by strength_boost
        strength=0.5 (decayed) -> distance increased by 0.5·strength_boost
        
        This implements the spacing effect: frequently retrieved memories
        are easier to recall (Ebbinghaus, 1885; Cepeda et al., 2006).
      
      d_sem   = 1 - cos(e_sem_m, e_sem_q)           ∈ [0, 1]
      d_emo   = ||e_emo_m - e_emo_q|| / 2.0          ∈ [0, 1]  (projected vs projected)
      d_state = ||s_snap_m - s_current|| / sqrt(2)    ∈ [0, 1]  (positive orthant bound)
      d_time  = 1 - exp(-λ · Δt)                      ∈ [0, 1]
    """
    N = sem_matrix.shape[0]
    if N == 0:
        return np.array([], dtype=np.float32)

    alpha, beta, gamma, delta = weights.as_tuple()

    # Semantic: cosine distance via dot product (vectors are L2-normalized)
    sem_sims = sem_matrix @ query_semantic  # (N,)
    d_sem = np.clip(1.0 - sem_sims, 0.0, 1.0)

    # Emotional: Euclidean between PROJECTED vectors
    emo_diff = emo_matrix - query_emotional[np.newaxis, :]  # (N, k)
    d_emo = np.clip(np.linalg.norm(emo_diff, axis=1) / EMO_NORM, 0.0, 1.0)

    # State: Euclidean between L2-normalized state vectors
    state_diff = state_matrix - s_current[np.newaxis, :]  # (N, n)
    d_state = np.clip(np.linalg.norm(state_diff, axis=1) / STATE_NORM, 0.0, 1.0)

    # Temporal: exponential decay
    delta_t = np.maximum(0, current_step - ts_array).astype(np.float64)
    d_time = np.clip(1.0 - np.exp(-decay_rate * delta_t), 0.0, 1.0).astype(np.float32)

    # Weighted sum
    total = alpha * d_sem + beta * d_emo + gamma * d_state + delta * d_time

    # Strength modulation: reinforced memories are easier to recall
    if strength_array is not None and strength_boost > 0:
        # strength ranges [0, 2], centered at 1.0
        # modulator = 1 - boost * (strength - 1) -> range [1+boost, 1-boost]
        modulator = 1.0 - strength_boost * (strength_array - 1.0)
        modulator = np.clip(modulator, 0.5, 1.5)  # safety clamp
        total = total * modulator

    return np.clip(total, 0.0, 1.0).astype(np.float32)


def softmax_retrieval(
    distances: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Convert distances to retrieval probabilities via softmax.
    
    Math:
      P(m_i | q) = exp(-d_i / T) / Σ_j exp(-d_j / T)
      
    Low T (→0): deterministic, picks lowest distance.
    High T (→∞): uniform random across all memories.
    
    Adaptive temperature (NEW):
      T(t) = T_base * (1 + η * novelty(t))
    """
    if len(distances) == 0:
        return np.array([], dtype=np.float32)
    
    logits = -distances / max(temperature, 1e-8)
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / (exp_logits.sum() + 1e-8)
    return probs.astype(np.float32)


def adaptive_temperature(
    distances: np.ndarray,
    t_base: float = 0.1,
    eta: float = 0.5,
) -> float:
    """
    NEW MATH: Compute adaptive retrieval temperature.
    
    T(t) = T_base * (1 + η * novelty)
    novelty = min_distance (closest memory's distance = how novel the query is)
    
    When min_distance is high (nothing matches well) -> T increases -> exploratory
    When min_distance is low (strong match exists) -> T stays low -> deterministic
    
    η controls exploration sensitivity. Default 0.5.
    """
    if len(distances) == 0:
        return t_base * (1 + eta)  # maximum exploration
    
    novelty = float(np.min(distances))
    return t_base * (1.0 + eta * novelty)


def retrieve_top_k(
    query_semantic: np.ndarray,
    query_emotional: np.ndarray,  # must be pre-projected via encode_emotional
    store: MemoryStore,
    s_current_normalized: np.ndarray,  # must be pre-normalized via encode_state
    current_step: int,
    k: int = 3,
    tag_filter: str = None,
    use_adaptive_temp: bool = True,
    use_strength: bool = True,
) -> list:
    """
    Retrieve k most relevant memories using vectorized manifold distance.
    
    Uses strength-weighted retrieval by default: reinforced memories are
    easier to recall, decayed memories are harder (spacing effect).
    
    Returns list of (distance, probability, MemoryEntry) tuples.
    """
    candidates = store.get_all_safe()
    if not candidates:
        return []

    if tag_filter:
        candidates = [m for m in candidates if tag_filter in m.tags]
        if not candidates:
            return []

    # Build matrices
    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)
    emo_matrix = np.array([m.e_emotional for m in candidates], dtype=np.float32)
    state_matrix = np.array([m.s_snapshot for m in candidates], dtype=np.float32)
    ts_array = np.array([m.timestamp for m in candidates], dtype=np.int64)
    str_array = np.array([m.strength for m in candidates], dtype=np.float32) if use_strength else None

    weights = store.profile.retrieval_weights
    decay_rate = store.profile.decay_rate

    distances = vectorized_manifold_distance(
        sem_matrix, emo_matrix, state_matrix, ts_array,
        query_semantic, query_emotional, s_current_normalized,
        current_step, weights, decay_rate,
        strength_array=str_array,
    )

    # Adaptive temperature
    if use_adaptive_temp:
        temp = adaptive_temperature(distances, store.profile.temperature)
    else:
        temp = store.profile.temperature

    probs = softmax_retrieval(distances, temp)

    # Sort by distance (ascending = most relevant first)
    indices = np.argsort(distances)[:k]
    
    results = []
    for idx in indices:
        results.append((
            float(distances[idx]),
            float(probs[idx]),
            candidates[idx],
        ))
    
    return results


def retrieve_top_k_fast(
    query_semantic: np.ndarray,
    query_emotional: np.ndarray,
    store: MemoryStore,
    s_current_normalized: np.ndarray,
    current_step: int,
    k: int = 3,
    use_strength: bool = True,
) -> list:
    """
    Ultra-fast retrieval using pre-cached matrices from MemoryStore.
    Avoids rebuilding numpy arrays on every call.
    
    Includes strength-weighted retrieval by default.
    """
    store._rebuild_cache()
    
    if store._sem_cache.shape[0] == 0:
        return []

    weights = store.profile.retrieval_weights
    decay_rate = store.profile.decay_rate
    str_array = store._str_cache if use_strength else None

    distances = vectorized_manifold_distance(
        store._sem_cache, store._emo_cache, store._state_cache, store._ts_cache,
        query_semantic, query_emotional, s_current_normalized,
        current_step, weights, decay_rate,
        strength_array=str_array,
    )

    temp = adaptive_temperature(distances, store.profile.temperature)
    probs = softmax_retrieval(distances, temp)

    indices = np.argsort(distances)[:k]
    
    results = []
    for idx in indices:
        mid = store._id_order[idx]
        results.append((
            float(distances[idx]),
            float(probs[idx]),
            store._memories[mid],
        ))
    
    return results


def retrieval_entropy(distances: np.ndarray) -> float:
    """
    Shannon entropy of retrieval distribution.
    
    H = -Σ P(i) · log(P(i))
    
    High H (>1.5): diffuse retrieval, unfamiliar territory
    Low H (<0.5): focused retrieval, recognized pattern
    """
    if len(distances) == 0:
        return 2.0
    
    d = np.array(distances, dtype=np.float32)
    weights = np.exp(-d)
    total = weights.sum()
    if total < 1e-8:
        return 2.0
    probs = weights / total
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    return float(entropy)


# ───────────────────────────────────────────
# SEMANTIC-ONLY BASELINE (for experiments)
# ───────────────────────────────────────────
def retrieve_semantic_only(
    query_semantic: np.ndarray,
    store: MemoryStore,
    k: int = 3,
) -> list:
    """Baseline: retrieve by cosine similarity only (standard RAG approach)."""
    candidates = store.get_all_safe()
    if not candidates:
        return []

    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)
    sims = sem_matrix @ query_semantic
    distances = 1.0 - sims

    indices = np.argsort(distances)[:k]
    return [(float(distances[idx]), candidates[idx]) for idx in indices]


def retrieve_semantic_emotional(
    query_semantic: np.ndarray,
    query_emotional: np.ndarray,
    store: MemoryStore,
    k: int = 3,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list:
    """Ablation: semantic + emotional only (no state, no temporal)."""
    candidates = store.get_all_safe()
    if not candidates:
        return []

    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)
    emo_matrix = np.array([m.e_emotional for m in candidates], dtype=np.float32)

    d_sem = np.clip(1.0 - sem_matrix @ query_semantic, 0.0, 1.0)
    emo_diff = emo_matrix - query_emotional[np.newaxis, :]
    d_emo = np.clip(np.linalg.norm(emo_diff, axis=1) / EMO_NORM, 0.0, 1.0)

    distances = alpha * d_sem + beta * d_emo

    indices = np.argsort(distances)[:k]
    return [(float(distances[idx]), candidates[idx]) for idx in indices]
