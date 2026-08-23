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

import warnings

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

# Channel rescaling modes for `vectorized_manifold_distance`.
#
# WHY THIS EXISTS. The two constants above are worst-case bounds, and the
# distances they divide never approach them. Measured over 9,401 (query,
# memory) pairs in experiments/results/exp22/exp22_emo_ablation.json, under the
# shipped weights (0.4, 0.2, 0.3, 0.1):
#
#   channel    mean      sd       weight x sd    effective influence
#   d_sem      0.82217   0.15513  0.062050       91.67%
#   d_emo      0.00698   0.00427  0.000853        1.26%
#   d_state    0.02524   0.01184  0.003551        5.25%
#   d_time     0.02123   0.01231  0.001231        1.82%
#
# Ranking depends only on how much a channel varies between candidates, so a
# channel's real influence is its weight times its spread, not its weight. The
# emotional term is nominally a fifth of the decision and is in fact 1.26% of
# it. That is why removing it changes almost nothing: exp22 measured a null on
# a channel that was already inert.
#
# "minmax" and "robust" rescale each channel across the candidate set so every
# channel spans the unit interval and the profile weights become effective.
# Because the weights sum to 1 and each rescaled channel lies in [0, 1], the
# composite stays in [0, 1] without relying on the final clip.
#
# HONEST LIMITS. Rescaling is per query and per candidate set, so the resulting
# composite is a ranking score and is NOT comparable across queries or across
# stores of different composition. Absolute-threshold logic must not use it.
# "none" is the default and preserves the shipped arithmetic exactly.
CHANNEL_NORMALIZATION_MODES = ("none", "minmax", "robust")

# Spread below which a channel is treated as carrying no ranking information.
# Dividing by a smaller span would amplify floating-point noise into a signal.
CHANNEL_SPAN_EPS = 1e-9

# Percentile bounds for the "robust" mode, which resists a single outlying
# candidate compressing every other candidate into a narrow band.
CHANNEL_ROBUST_LOW = 5.0
CHANNEL_ROBUST_HIGH = 95.0


# ---------------------------------------------------------------------------
# TEMPORAL ANCHOR
#
# The temporal channel's weakness is not its weight, it is where it is anchored.
# In "store_end", the shipped behaviour, delta_t is max(0, current_step - ts).
# current_step is the store's write counter, so within one conversation it is the
# same number for every query, and the channel therefore emits an identical
# vector for all of them. It is a static per-memory recency prior with no
# query-specific content. exp17's recency_only arm scores P@5 0.2851 on the
# real-world corpus, and the random-guess precision on that same benchmark is
# 0.2851, so the channel performs at chance. That arm is defined as a sort by
# MemoryEntry.timestamp descending rather than as the d_time channel itself; the
# two induce the same ranking because max(0, step - ts) is monotone in ts, so this
# is an inference from a timestamp sort and not a direct measurement of d_time.
#
# The functional form is the second half of the problem. max(0, step - ts) is
# one-sided and monotone, so the only proposition it can express is "older is
# further away". Locating a memory near a point in time needs a two-sided kernel
# peaked at an anchor, which is what abs(anchor - ts) gives. These are different
# functions and no reweighting turns the first into the second.
#
# "semantic_rank1" anchors on the timestamp of the best semantic match and
# measures abs(anchor - ts), which is the temporal contiguity effect: recalling
# an item makes items encoded near it in time more accessible. Contiguity is the
# most reproduced finding in the free-recall literature and is the mechanism
# behind Howard and Kahana's Temporal Context Model (2002). It uses only the
# semantic channel and the timestamps already in the store, and reads no label.
#
# HONEST LIMITS, to be stated wherever a gain from this is reported. On a
# benchmark whose relevance label is session membership, and whose sessions are
# stored as contiguous runs of timestamps, a contiguity kernel is close to
# optimal by construction. A gain therefore demonstrates that the system can
# exploit conversational structure; it does not demonstrate better semantics.
# Attribution requires a contiguity-only arm carrying no semantic weight and an
# anchor-only arm, so the gain can be split between the kernel and mere block
# detection. The anchor is also only as good as the rank-1 hit, so this mode
# inherits and can amplify a semantic error. And because the anchor is chosen from
# the candidate set, a tag filter that excludes the true rank-1 hit relocates the
# anchor and so changes every d_time: the temporal rule is the same in both entry
# points, but the anchor it resolves to is a function of which candidates survive
# filtering. Under "store_end" a filter cannot move the anchor, since there is
# nothing to move, so the two modes are not directly comparable under a filter.
#
# Profile customs are read by retrieve_top_k and retrieve_top_k_fast only.
# vectorized_manifold_distance is public API and honours its own keyword
# arguments, whose defaults are the shipped behaviour, so a caller that invokes it
# directly gets "store_end" whatever the store's profile says. The novelty probes
# at experiments/python/run_all_experiments.py:437 and run_fast.py:322 do exactly
# that, deliberately, measuring against fixed distance parameters.
TEMPORAL_ANCHOR_MODES = ("store_end", "semantic_rank1")

# Width in turns of the contiguity kernel, used as the rate 1/width. Ignored
# entirely under "store_end", and required to be positive under "semantic_rank1",
# where 0.0 therefore means "unset" rather than a usable value. The memory
# decay_rate cannot serve as a default: it is 0.001, which over a whole 40-turn
# store spans d_time about 0.039 and cannot separate memories inside an 11-turn
# session, so defaulting to it would turn a forgotten setting into a measured
# null. The right width is a property of the conversation length being searched,
# so the caller states it.
TEMPORAL_KERNEL_WIDTH_DEFAULT = 0.0


def _rescale_channel(d: np.ndarray, mode: str) -> np.ndarray:
    """Rescale one channel's distances so its spread matches the other channels.

    Returns a channel in [0, 1] that never inverts a pair: if d[i] < d[j] then
    the output at i is less than or equal to the output at j. The map is order
    preserving in that weak sense only, because it can lose a distinction:

      * "minmax" is affine with positive slope, so it is strictly increasing in
        exact arithmetic and can only tie two candidates when float32 runs out
        of significant bits.
      * "robust" clips at the CHANNEL_ROBUST_LOW and CHANNEL_ROBUST_HIGH
        percentiles, so every candidate in a tail is mapped to exactly 0.0 or
        exactly 1.0 and ties with the rest of that tail by design. That is the
        point of the mode, but it does discard ordering information inside the
        tails that "minmax" keeps.

    Measured over 1,824 (query, channel, mode) checks in
    experiments/results/exp25/exp25_channel_normalization.json: 0 inversions,
    0 bounds violations, and 0 of 228 queries where "minmax" changed the
    semantic ranking. Under "robust" the clip newly tied 2,546 adjacent pairs
    and 3,470 pairs counted over all (i, j), and every one of the adjacent ties
    sat exactly on a clip bound, so the ties are the clip rather than float32.
    Because the low tail of a distance channel is the head of the ranking, that
    lost ordering shows up in the rank-one metric: the sem_pure_robust arm
    scores MRR 0.7179 against sem_pure_none's 0.7786 on identical weights.

    A channel that is constant across the candidate set returns all zeros,
    because it cannot separate candidates and must not be amplified. A candidate
    set with fewer than two members is returned unchanged, since there is no
    ordering to preserve and no span to divide by.
    """
    if mode == "none" or d is None or d.size < 2:
        return d
    if mode == "minmax":
        lo = float(d.min())
        hi = float(d.max())
    else:  # "robust"
        lo = float(np.percentile(d, CHANNEL_ROBUST_LOW))
        hi = float(np.percentile(d, CHANNEL_ROBUST_HIGH))
    span = hi - lo
    if span <= CHANNEL_SPAN_EPS:
        return np.zeros_like(d)
    out = ((d - lo) / span).astype(d.dtype, copy=False)
    return np.clip(out, 0.0, 1.0, out=out)


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
    contradiction_array: np.ndarray = None,  # (N,) 1 if contradicted else 0
    contradiction_weight: float = 0.0,       # lambda for contradiction penalty
    contradiction_gate: float = 1.0,         # query-intent gate in [0, 1]
    use_fast_temporal: bool = False,    # opt-in approximation for temporal term
    channel_normalization: str = "none",  # "none" | "minmax" | "robust", see above
    temporal_anchor: str = "store_end",  # "store_end" | "semantic_rank1", see above
    temporal_kernel_width: float = TEMPORAL_KERNEL_WIDTH_DEFAULT,  # turns; must be >0 under the anchor
) -> np.ndarray:
    """
    Compute manifold distance for ALL memories at once via optimized vectorized numpy.
    
    OPTIMIZATIONS:
    1. Pre-allocate output array to avoid intermediate allocations
    2. Use in-place operations where possible (np.clip with out parameter)
    3. Combine weight multiplication into single operation
    4. Optional fast approximation for temporal decay (opt-in only)
    
    Returns (N,) array of distances in [0, 1].
    
        Math:
            d_raw(m, q) = α·d_sem + β·d_emo + γ·d_state + δ·d_time
            d_contra    = λ·I[contradicted]·gate
            d(m, q)     = (1-λ)·d_raw + d_contra
            d_final     = d(m, q) · (1 - strength_boost · (strength - 1))
      
      Strength modulation:
        strength=1.0 (default) -> no change
        strength=2.0 (max, heavily reinforced) -> distance reduced by strength_boost
        strength=0.5 (decayed) -> distance increased by 0.5·strength_boost
        
        Strength modulation is an NCM design choice, not a modelled
        psychological effect. It makes reinforced memories easier to
        retrieve and decayed memories harder. Note that the mechanism
        carries no interstudy-interval term, so it does not implement
        the spacing effect. The temporal term d_time below is an
        exponential forgetting curve of the general form reported by
        Ebbinghaus (1885).
      
      d_sem   = 1 - cos(e_sem_m, e_sem_q)           ∈ [0, 1]
      d_emo   = ||e_emo_m - e_emo_q|| / 2.0          ∈ [0, 1]  (projected vs projected)
      d_state = ||s_snap_m - s_current|| / sqrt(2)    ∈ [0, 1]  (positive orthant bound)
      d_time  = 1 - exp(-r · Δt)                      ∈ [0, 1]

      The temporal term's Δt and rate r depend on temporal_anchor:

        "store_end"      Δt = max(0, current_step - ts),  r = decay_rate
                         The shipped behaviour. One-sided and monotone in age,
                         and anchored on a number that does not depend on the
                         query, so it emits the same vector for every query
                         against one store. See TEMPORAL_ANCHOR_MODES.

        "semantic_rank1" Δt = |ts[argmax(sem_matrix @ query_semantic)] - ts|,
                         r  = 1/temporal_kernel_width, which must be positive.
                         Two-sided and peaked at the best semantic match, so it
                         says "near that memory in time", which is the temporal
                         contiguity effect. Requires no label and no new input.
                         Its accuracy is bounded by the rank-1 hit's accuracy.
                         The anchor comes from the raw similarities rather than
                         from argmin(d_sem), which is not the same thing once
                         d_sem has been clipped into [0, 1]; see the branch.
    """
    N = sem_matrix.shape[0]
    alpha, beta, gamma, delta = weights.as_tuple()
    lambda_contra = float(np.clip(contradiction_weight, 0.0, 1.0))
    gate = float(np.clip(contradiction_gate, 0.0, 1.0))
    base_scale = 1.0 - lambda_contra
    # Arguments are validated before the empty-candidate-set shortcut below, so a
    # typo cannot survive a smoke test that happens to run against an empty store.
    rescale = "none" if channel_normalization is None else str(channel_normalization)
    if rescale not in CHANNEL_NORMALIZATION_MODES:
        raise ValueError(
            "channel_normalization must be one of %r, got %r"
            % (list(CHANNEL_NORMALIZATION_MODES), channel_normalization)
        )
    anchor_mode = "store_end" if temporal_anchor is None else str(temporal_anchor)
    if anchor_mode not in TEMPORAL_ANCHOR_MODES:
        raise ValueError(
            "temporal_anchor must be one of %r, got %r"
            % (list(TEMPORAL_ANCHOR_MODES), temporal_anchor)
        )
    # The width is validated whatever the mode and whatever delta is, because the
    # values that would otherwise pass silently are the dangerous ones. A nan width
    # fails every "> 0" test and would look like a deliberate choice. An inf width
    # gives rate 0.0, so d_time becomes exactly 0 for every candidate and the
    # channel is dead while still drawing its weight.
    kernel_width = TEMPORAL_KERNEL_WIDTH_DEFAULT if temporal_kernel_width is None \
        else float(temporal_kernel_width)
    if not np.isfinite(kernel_width):
        raise ValueError(
            "temporal_kernel_width must be finite, got %r. nan and inf are "
            "rejected because they would silently disable the temporal channel "
            "rather than fail." % (temporal_kernel_width,)
        )
    # A non-positive width is refused under "semantic_rank1" rather than silently
    # falling back to decay_rate. Falling back is what the first draft did, and it
    # is a trap: at decay_rate the contiguity kernel spans about 0.02 over a
    # 40-turn store, which is roughly 0.6 percent of the semantic channel's
    # influence, so enabling the anchor and forgetting the width would measure a
    # near-inert channel and report a false null. There is no sensible default
    # here, because the right width is a property of the conversation length being
    # searched, so the caller has to state it. To anchor at the memory decay rate
    # deliberately, pass the width that expresses it, 1.0/decay_rate.
    if anchor_mode == "semantic_rank1" and kernel_width <= 0.0:
        raise ValueError(
            "temporal_anchor='semantic_rank1' requires a positive "
            "temporal_kernel_width in turns, got %r. The width sets the rate as "
            "1/width; at the default decay_rate of 0.001 the kernel is too flat "
            "to separate memories inside one session, so there is no safe "
            "default. Try a width near the session length, for example 4.0, or "
            "pass 1.0/decay_rate to anchor at the memory decay rate on purpose."
            % (temporal_kernel_width,)
        )

    if N == 0:
        return np.array([], dtype=np.float32)

    # OPTIMIZATION: Pre-allocate output
    total = np.zeros(N, dtype=np.float32)

    # Each channel is computed first and accumulated below, so that the optional
    # rescaling in between sees the whole candidate set. A channel whose weight
    # is negligible is left as None and never computed, as before.
    d_sem = d_emo = d_state = d_time = None
    sem_sims = None  # kept so the temporal anchor can reuse it, see below

    # Semantic: cosine distance via dot product (vectors are L2-normalized)
    if alpha > 1e-8:
        sem_sims = sem_matrix @ query_semantic  # (N,)
        d_sem = np.clip(1.0 - sem_sims, 0.0, 1.0)

    # Emotional: Euclidean between PROJECTED vectors
    if beta > 1e-8:
        emo_diff = emo_matrix - query_emotional[np.newaxis, :]  # (N, k)
        d_emo = np.clip(np.linalg.norm(emo_diff, axis=1) / EMO_NORM, 0.0, 1.0)

    # State: Euclidean between L2-normalized state vectors
    if gamma > 1e-8:
        state_diff = state_matrix - s_current[np.newaxis, :]  # (N, n)
        d_state = np.clip(np.linalg.norm(state_diff, axis=1) / STATE_NORM, 0.0, 1.0)

    # Temporal: exact exponential by default; optional fast approximation is opt-in.
    if delta > 1e-8:
        if anchor_mode == "semantic_rank1":
            # The anchor is taken from the raw similarities, not from d_sem, and
            # not from any rescaled copy. Two reasons. d_sem is clipped into
            # [0, 1], so a float32 cosine slightly above 1.0 saturates to exactly
            # 0.0 and can tie with a genuinely worse candidate, and argmin would
            # then answer on tie order rather than on similarity. And this branch
            # has to work when alpha is 0, which is the contiguity-only control
            # arm, where d_sem is never computed at all. Reusing sem_sims when it
            # exists keeps the common path at one matrix-vector product.
            if sem_sims is None:
                sem_sims = sem_matrix @ query_semantic
            # Non-finite similarities are excluded from the choice of anchor. A
            # plain argmax returns the index of the first nan, which would centre
            # the whole channel on one corrupt row: under "store_end" a nan
            # polluted only its own d_sem, so this mode would otherwise widen the
            # blast radius of bad input from one memory to all N. An inf would
            # likewise win the argmax unconditionally. If nothing is finite there
            # is no anchor to choose and no way to guess one, so that raises.
            finite_sims = np.isfinite(sem_sims)
            if bool(finite_sims.all()):
                anchor_idx = int(np.argmax(sem_sims))
            elif bool(finite_sims.any()):
                anchor_idx = int(np.argmax(np.where(finite_sims, sem_sims, -np.inf)))
            else:
                raise ValueError(
                    "temporal_anchor='semantic_rank1' cannot pick an anchor: "
                    "every one of the %d candidate similarities is nan or inf, "
                    "which means sem_matrix or query_semantic is corrupt." % N
                )
            # Two-sided: distance from the anchor's position in time, in either
            # direction. This is the whole point of the mode. max(0, step - ts)
            # can only say "older is worse"; abs(anchor - ts) can say "near this".
            #
            # The difference is taken in float64 rather than in ts_array's own
            # dtype, which matters for two reasons. np.abs is a no-op on an
            # unsigned dtype, so on a uint array the wraparound in the subtraction
            # survives it and every memory older than the anchor saturates at
            # maximum distance, silently restoring the one-sidedness this mode
            # exists to remove. And a signed subtraction can overflow, scoring the
            # two most distant memories as adjacent. float64 is exact for integer
            # timestamps up to 2**53, far beyond any store, and d_time is cast to
            # float32 afterwards anyway.
            ts_f = np.asarray(ts_array, dtype=np.float64)
            delta_t = np.abs(ts_f - ts_f[anchor_idx]).astype(np.float32)
            # A contiguity kernel needs its own width, which is why a positive one
            # is required above. At width 4 and on the exact branch, d is 0.0 on
            # the anchor, 0.6321 four turns away, 0.8647 at eight and 0.9502 at
            # twelve. Under use_fast_temporal the same width gives 0.0, 0.5,
            # 0.6667 and 0.75, so the approximation flattens the kernel as well as
            # cheapening it.
            #
            # Degenerate case, stated because the contiguity-only control arm is
            # where it would hide: if every similarity is equal, argmax returns
            # index 0 and the kernel becomes a plain age ramp from the oldest
            # candidate, carrying no query information while still looking like a
            # working channel.
            rate = 1.0 / kernel_width
        else:
            # Shipped behaviour, bit for bit: one-sided age against the store's
            # write counter, at the memory decay rate.
            delta_t = np.maximum(0, current_step - ts_array).astype(np.float32)
            rate = decay_rate
        if use_fast_temporal:
            # Opt-in approximation: exp(-x) ≈ 1 / (1 + x)
            # Faster but introduces approximation error in the temporal component.
            # Use when large-scale speed is critical and small temporal error is acceptable.
            # This is intentionally opt-in (default = False) so callers must opt into the
            # approximation when they accept the tradeoff.
            d_time = np.clip(1.0 - 1.0 / (1.0 + rate * delta_t), 0.0, 1.0)
        else:
            # Exact temporal term (default): preserves baseline math behavior.
            d_time = np.clip(1.0 - np.exp(-rate * delta_t), 0.0, 1.0)

    # Optional: equalize the channels' spreads so the profile weights are the
    # weights that actually act. Off by default, in which case the accumulation
    # below is arithmetically identical to the shipped version.
    if rescale != "none":
        d_sem = _rescale_channel(d_sem, rescale)
        d_emo = _rescale_channel(d_emo, rescale)
        d_state = _rescale_channel(d_state, rescale)
        d_time = _rescale_channel(d_time, rescale)

    # Accumulated in the original order so that "none" reproduces the shipped
    # floating-point result exactly, addition not being associative.
    if d_sem is not None:
        total += base_scale * alpha * d_sem
    if d_emo is not None:
        total += base_scale * beta * d_emo
    if d_state is not None:
        total += base_scale * gamma * d_state
    if d_time is not None:
        total += base_scale * delta * d_time

    # Contradiction penalty: push contradicted memories down unless query gate disables it
    if contradiction_array is not None and lambda_contra > 1e-8 and gate > 1e-8:
        d_contra = np.clip(contradiction_array, 0.0, 1.0) * (lambda_contra * gate)
        total += d_contra

    # Strength modulation: reinforced memories are easier to recall
    if strength_array is not None and strength_boost > 1e-8:
        # strength ranges [0, 2], centered at 1.0
        # modulator = 1 - boost * (strength - 1) -> range [1+boost, 1-boost]
        modulator = 1.0 - strength_boost * (strength_array - 1.0)
        # Bound the modulator so an outlier strength cannot dominate the
        # manifold distance.
        #
        # WHAT THESE BOUNDS ACTUALLY DO AT THE SHIPPED DEFAULT: nothing. The
        # modulator spans [1-strength_boost, 1+strength_boost], and
        # strength_boost defaults to 0.1 with no caller in this repository
        # overriding it, so the reachable span is [0.9, 1.1] and a memory at
        # maximum strength is 10% easier to recall, not 50%. ncm/memory.py caps
        # strength at 2.0, so the clip below binds only when strength_boost
        # exceeds 0.5. It is retained as a guard for tuned configurations, not
        # because it is active by default. An earlier version of this comment
        # read the bounds as the effect size and claimed 50%, which overstated
        # the default by a factor of five and contradicted the arithmetic given
        # in this function's own docstring.
        np.clip(modulator, 0.5, 1.5, out=modulator)
        total *= modulator

    # OPTIMIZATION: Final clip in-place
    np.clip(total, 0.0, 1.0, out=total)
    return total.astype(np.float32)


def softmax_retrieval(
    distances: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Convert distances to retrieval probabilities via numerically stable softmax.
    
    OPTIMIZATIONS:
    1. Use log-sum-exp trick for stability (always done)
    2. Pre-allocate output array
    3. Use in-place operations where safe
    4. Temperature clipping prevents division by very small numbers
    
    Math:
      P(m_i | q) = exp(-d_i / T) / Σ_j exp(-d_j / T)
      
    Low T (→0): deterministic, picks lowest distance.
    High T (→∞): uniform random across all memories.
    
    Adaptive temperature (NEW):
      T(t) = T_base * (1 + η * novelty(t))
    """
    if len(distances) == 0:
        return np.array([], dtype=np.float32)
    
    # OPTIMIZATION: Ensure temperature is safe from overflow/underflow
    T_safe = np.clip(temperature, 1e-8, 100.0)
    
    # OPTIMIZATION: Pre-scale to avoid recalculation
    logits = -distances / T_safe
    # Log-sum-exp trick: subtract max for numerical stability
    logits -= logits.max()  
    # OPTIMIZATION: In-place exponential
    # NOTE: after this line, `logits` no longer stores logits; it stores exp(logits).
    # Keep this aliasing behavior explicit to avoid future debugging confusion.
    np.exp(logits, out=logits)
    
    exp_sum = logits.sum() + 1e-8
    probs = logits / exp_sum
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


def _resolve_current_state(store: MemoryStore,
                           s_current_normalized: np.ndarray,
                           channel_width: int) -> np.ndarray:
    """Pick the state vector the state channel compares memories against.

    Both retrieval entry points accept an `s_current_normalized` argument that
    they historically discarded. The cause was a width mismatch, not an
    oversight that could be flipped: the state channel is 5 wide, built from
    MemoryEntry.auto_state_snapshot, while every in-tree caller passes
    SentenceEncoder.encode_state(...), which pads to state_dim and is 7 wide by
    default. A 7-vector cannot feed a 5-wide channel, so both paths fell back to
    store.auto_state and said nothing about it. `retrieve_top_k` even labels the
    parameter "retained for backward compatibility".

    Consequences worth knowing, because they differ per experiment: exp22
    documented the no-op and worked around it by assigning store.auto_state.state
    before each query, so its state channel really was query-dependent. exp11 and
    exp12 passed a query state and did not, so their d_state compared every
    memory against whatever state the last add() happened to leave behind, which
    is a per-memory bias rather than a query-relevance signal.

    This resolves the argument instead of ignoring it, but only behind the
    profile custom "honor_caller_state", so the default is byte-for-byte the
    previous behaviour and no committed result moves. A refused vector now warns
    rather than vanishing. Warning text is constant per branch so Python's
    once-per-location filter collapses a whole benchmark loop into one message.
    """
    s_current_auto = store.auto_state.get_current_state()
    if s_current_normalized is None:
        return s_current_auto

    supplied = np.asarray(s_current_normalized, dtype=np.float32).reshape(-1)
    if supplied.shape[0] != channel_width:
        warnings.warn(
            "s_current_normalized was ignored: the state channel is "
            f"{channel_width} wide and the supplied vector is "
            f"{supplied.shape[0]} wide. Building it with "
            "SentenceEncoder.encode_state produces exactly this mismatch, "
            "because encode_state pads to state_dim while the channel reads "
            "the 5-wide auto-state. The store's own auto_state was used, so "
            "the state channel did not see your vector. To drive it, assign "
            "store.auto_state.state, or pass a vector of the channel's width "
            "with the profile custom 'honor_caller_state' set true.",
            RuntimeWarning, stacklevel=3)
        return s_current_auto

    if not bool(store.profile.get_custom("honor_caller_state", False)):
        warnings.warn(
            "s_current_normalized was ignored because the profile custom "
            "'honor_caller_state' is not set, so the store's own auto_state "
            "was used. This preserves the behaviour every committed result was "
            "scored under. Set that custom true to drive the state channel "
            "from the caller.",
            RuntimeWarning, stacklevel=3)
        return s_current_auto

    return supplied


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize, leaving a near-zero vector alone rather than dividing.

    The norm is deliberately left as the numpy float64 scalar that
    np.linalg.norm returns, so the division promotes to float64 and casts back
    exactly as the inlined code it replaces did. Converting it to a Python float
    first would keep the division in float32 under numpy's weak scalar rules and
    could move the last bit of every state distance.
    """
    n = np.linalg.norm(v)
    if n > 1e-8:
        return (v / n).astype(np.float32)
    return np.asarray(v).astype(np.float32)


def _resolve_temporal_options(store: MemoryStore) -> tuple:
    """Read the temporal anchor and kernel width from the store's profile.

    Both entry points call this, so a store carries its own temporal rule rather
    than depending on which call site it reached. Absent from a profile, the answer
    is ("store_end", 0.0), which is the shipped behaviour and computes the same
    floating-point d_time as the version before the anchor existed.

    Note that the same rule does not mean the same d_time across the two entry
    points. Under "semantic_rank1" the anchor is chosen from the candidate set, so
    a tag filter that excludes the true rank-1 hit relocates it, and the
    tag-filtered path in retrieve_top_k will produce a different temporal channel
    from the unfiltered fast path on the same query. That is intended, since
    anchoring on an excluded memory would be worse, but it means the two are not
    comparable under a filter. See TEMPORAL_ANCHOR_MODES.

    Neither value is validated here; vectorized_manifold_distance raises on an
    unknown mode and on a width that is non-finite or non-positive under the
    anchor, so a profile mistake fails loudly at the one place that can name the
    supported values. The width is converted here rather than passed through, so
    a profile carrying a non-numeric value names the offending key in the error
    instead of raising a bare "float() argument must be a string or a real number"
    from inside the distance function.
    """
    anchor = str(store.profile.get_custom("temporal_anchor", "store_end"))
    raw_width = store.profile.get_custom(
        "temporal_kernel_width", TEMPORAL_KERNEL_WIDTH_DEFAULT)
    if raw_width is None:
        return anchor, TEMPORAL_KERNEL_WIDTH_DEFAULT
    try:
        width = float(raw_width)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "profile custom 'temporal_kernel_width' must be a number of turns, "
            "got %r" % (raw_width,)
        ) from exc
    return anchor, width


def retrieve_top_k(
    query_semantic: np.ndarray,
    query_emotional: np.ndarray,  # must be pre-projected via encode_emotional
    store: MemoryStore,
    s_current_normalized: np.ndarray,  # retained for backward compatibility
    current_step: int,
    k: int = 3,
    tag_filter: str = None,
    use_adaptive_temp: bool = True,
    use_strength: bool = True,
    use_fast_temporal: bool = False,
) -> list:
    """
    Retrieve k most relevant memories using vectorized manifold distance.
    
    OPTIMIZATIONS:
    1. Use retrieve_top_k_fast for most cases (pre-cached matrices)
    2. Only build matrices when tag filtering required
    3. Avoid redundant adaptive temperature computation
    
    Uses strength-weighted retrieval by default: reinforced memories are
    easier to recall, decayed memories are harder. This is a design choice;
    it is not an implementation of the spacing effect.
    
    Returns list of (distance, probability, MemoryEntry) tuples.
    """
    candidates = store.get_all_safe()
    if not candidates:
        return []

    # OPTIMIZATION: If no tag filter, use fast path with pre-cached matrices.
    # The current-state vector is resolved further down, after the candidate
    # matrices exist, because only the tag-filtered branch below consumes it and
    # its required width is a property of those matrices. Resolving it here
    # instead meant doing the work on every delegated call and throwing it away.
    if not tag_filter:
        return retrieve_top_k_fast(
            query_semantic, query_emotional, store, s_current_normalized,
            current_step, k=k, use_strength=use_strength,
            use_fast_temporal=use_fast_temporal,
        )

    # Slower path: build matrices for filtered candidates
    if tag_filter:
        candidates = [m for m in candidates if tag_filter in m.tags]
        if not candidates:
            return []

    # Build matrices only for filtered candidates
    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)
    emo_matrix = np.array([m.e_emotional for m in candidates], dtype=np.float32)
    def _memory_auto_state(m: MemoryEntry) -> np.ndarray:
        if m.auto_state_snapshot is not None:
            raw = np.asarray(m.auto_state_snapshot, dtype=np.float32)
        else:
            raw = np.asarray(m.s_snapshot[:5], dtype=np.float32)
        n = float(np.linalg.norm(raw))
        return (raw / n).astype(np.float32) if n > 1e-8 else raw

    state_matrix = np.array([_memory_auto_state(m) for m in candidates], dtype=np.float32)
    s_current_for_distance = _unit(_resolve_current_state(
        store, s_current_normalized, int(state_matrix.shape[1])))
    ts_array = np.array([m.timestamp for m in candidates], dtype=np.int64)
    str_array = np.array([m.strength for m in candidates], dtype=np.float32) if use_strength else None

    weights = store.profile.retrieval_weights
    decay_rate = store.profile.decay_rate
    contra_enabled = bool(store.profile.get_custom("enable_contradiction_awareness", False))
    contra_lambda = float(store.profile.get_custom("contradiction_penalty", 0.0)) if contra_enabled else 0.0
    contra_gate = float(store.profile.get_custom("contradiction_query_gate", 1.0)) if contra_enabled else 1.0
    contra_array = np.array([
        1.0 if m.contradicted_by is not None else 0.0 for m in candidates
    ], dtype=np.float32) if contra_enabled else None
    # Same opt-in rescaling as the fast path, so tag-filtered retrieval and
    # unfiltered retrieval rank by the same rule. Note the candidate set here is
    # the filtered one, which is the correct set to rescale against.
    chan_norm = str(store.profile.get_custom("channel_normalization", "none"))
    temp_anchor, temp_width = _resolve_temporal_options(store)

    distances = vectorized_manifold_distance(
        sem_matrix, emo_matrix, state_matrix, ts_array,
        query_semantic, query_emotional, s_current_for_distance,
        current_step, weights, decay_rate,
        strength_array=str_array,
        contradiction_array=contra_array,
        contradiction_weight=contra_lambda,
        contradiction_gate=contra_gate,
        use_fast_temporal=use_fast_temporal,
        channel_normalization=chan_norm,
        temporal_anchor=temp_anchor,
        temporal_kernel_width=temp_width,
    )

    # Adaptive temperature
    if use_adaptive_temp:
        temp = adaptive_temperature(distances, store.profile.temperature)
    else:
        temp = store.profile.temperature

    probs = softmax_retrieval(distances, temp)

    # OPTIMIZATION: Use partition instead of full sort for top-k
    # np.partition is O(N) vs O(N log N) for argsort
    if k < len(distances) // 2:
        # Partition for efficiency when k << N
        indices = np.argpartition(distances, min(k, len(distances) - 1))[:k]
        # Re-sort within the top-k
        indices = indices[np.argsort(distances[indices])]
    else:
        # Full sort is faster for large k
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
    use_fast_temporal: bool = False,
) -> list:
    """
    Ultra-fast retrieval using pre-cached matrices from MemoryStore.
    Avoids rebuilding numpy arrays on every call.
    
    OPTIMIZATIONS:
    1. Skip cache rebuild if it's already current (check _cache_dirty flag)
    2. Use partition + partial sort for top-k instead of full sort
    3. Compute adaptive temperature from minimal work
    
    Includes strength-weighted retrieval by default.

    s_current_normalized: for most of this function's history this argument was
    accepted and then silently discarded, because the state channel reads the
    5-wide auto-state cache while every in-tree caller passes
    SentenceEncoder.encode_state(...), which is state_dim wide (7 by default).
    A 7-vector cannot feed a 5-wide channel, so the argument was unusable as
    declared and the function read store.auto_state instead. exp22 documented
    the no-op and worked around it by assigning store.auto_state.state; exp11
    and exp12 did not, and so scored d_state against whatever state the last
    add() had left behind rather than against their query.

    The argument is now real, but only behind an opt-in, so no committed result
    changes. With the profile custom "honor_caller_state" absent or false the
    behaviour is exactly as before. Set it true and pass a vector whose width
    matches the auto-state cache to drive the state channel from outside, which
    is the only supported way to feed the channel from a real affect model
    rather than from the internal keyword-anchor probe. A caller that passes a
    vector of the wrong width, or passes one while the opt-in is off, now gets
    a RuntimeWarning instead of silence.
    """
    store._rebuild_cache()
    
    if store._sem_cache.shape[0] == 0:
        return []

    s_current_for_distance = _unit(_resolve_current_state(
        store, s_current_normalized, int(store._auto_state_cache.shape[1])))

    weights = store.profile.retrieval_weights
    decay_rate = store.profile.decay_rate
    str_array = store._str_cache if use_strength else None
    contra_enabled = bool(store.profile.get_custom("enable_contradiction_awareness", False))
    contra_lambda = float(store.profile.get_custom("contradiction_penalty", 0.0)) if contra_enabled else 0.0
    contra_gate = float(store.profile.get_custom("contradiction_query_gate", 1.0)) if contra_enabled else 1.0
    contra_array = store._contra_cache if contra_enabled else None
    # Opt-in channel rescaling, read from the profile so a store carries its own
    # retrieval semantics. Absent from a profile, this is "none" and the
    # composite is byte-for-byte the shipped one.
    chan_norm = str(store.profile.get_custom("channel_normalization", "none"))
    temp_anchor, temp_width = _resolve_temporal_options(store)

    distances = vectorized_manifold_distance(
        store._sem_cache, store._emo_cache, store._auto_state_cache, store._ts_cache,
        query_semantic, query_emotional, s_current_for_distance,
        current_step, weights, decay_rate,
        strength_array=str_array,
        contradiction_array=contra_array,
        contradiction_weight=contra_lambda,
        contradiction_gate=contra_gate,
        use_fast_temporal=use_fast_temporal,
        channel_normalization=chan_norm,
        temporal_anchor=temp_anchor,
        temporal_kernel_width=temp_width,
    )

    temp = adaptive_temperature(distances, store.profile.temperature)
    probs = softmax_retrieval(distances, temp)

    # OPTIMIZATION: Use partition for efficient top-k
    N = len(distances)
    k_safe = min(k, N)
    
    if k_safe < N // 2:
        # Partition is faster for small k relative to N
        indices = np.argpartition(distances, min(k_safe, N - 1))[:k_safe]
        # Re-sort within the top-k
        indices = indices[np.argsort(distances[indices])]
    else:
        # Full sort for larger k
        indices = np.argsort(distances)[:k_safe]
    
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
    """Baseline: retrieve by cosine similarity only (standard RAG approach).

    Reads store._sem_cache rather than rebuilding the (N, dim) matrix on every
    call. The old body ran np.array([m.e_semantic for m in candidates]) per query,
    which at 10000 memories is 10000 Python attribute reads plus a fresh 5 MB
    allocation, and measured 8.362 ms p50 against 1.446 ms for the full composite
    path that reads the cache. A semantic-only baseline being 5.8x slower than the
    four-channel scorer it is meant to be the cheap comparison for is backwards,
    and every experiment that used it as a latency reference was reading that
    overhead as the cost of cosine retrieval.

    The output is unchanged, deliberately. _rebuild_cache builds from
    list(self._memories.keys()) while get_all_safe returns
    list(self._memories.values()), which for a dict is the same insertion order, so
    index i denotes the same entry either way. np.argsort is kept rather than
    replaced with the faster argpartition because argsort's tie order is part of
    what published ablations measured, and the rebuild was the cost worth removing.
    """
    store._rebuild_cache()
    if store._sem_cache.shape[0] == 0:
        return []

    distances = 1.0 - store._sem_cache @ query_semantic

    indices = np.argsort(distances)[:k]
    return [(float(distances[idx]), store._memories[store._id_order[idx]])
            for idx in indices]


def retrieve_semantic_emotional(
    query_semantic: np.ndarray,
    query_emotional: np.ndarray,
    store: MemoryStore,
    k: int = 3,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list:
    """Ablation: semantic + emotional only (no state, no temporal).

    Reads the caches for the same reason retrieve_semantic_only does: this rebuilt
    two (N, dim) matrices per call. Output is unchanged, including argsort's tie
    order.
    """
    store._rebuild_cache()
    if store._sem_cache.shape[0] == 0:
        return []

    d_sem = np.clip(1.0 - store._sem_cache @ query_semantic, 0.0, 1.0)
    emo_diff = store._emo_cache - query_emotional[np.newaxis, :]
    d_emo = np.clip(np.linalg.norm(emo_diff, axis=1) / EMO_NORM, 0.0, 1.0)

    distances = alpha * d_sem + beta * d_emo

    indices = np.argsort(distances)[:k]
    return [(float(distances[idx]), store._memories[store._id_order[idx]])
            for idx in indices]

# ───────────────────────────────────────────
# MULTI-HOP SPREADING ACTIVATION RETRIEVAL
# ───────────────────────────────────────────

def retrieve_multi_hop(
    query_semantic: np.ndarray,
    store: MemoryStore,
    k: int = 3,
    max_hops: int = 2,
    gamma: float = 0.8,
    similarity_threshold: float = 0.5,
) -> list:
    """Retrieve memories using multi‑hop spreading activation.

    The method starts from the direct semantic similarity of the query to all
    memories, then repeatedly spreads activation through a learned transition
    matrix built from pairwise semantic similarities. This enables reasoning
    over chains such as "A is B, B is C, therefore A is C" without constructing an
    explicit graph.
    """
    # Gather all candidates
    candidates = store.get_all_safe()
    if not candidates:
        return []

    # Semantic matrix (N x d)
    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)

    # Initial activation (similarity scores)
    init_sim = sem_matrix @ query_semantic  # (N,)
    activation = init_sim.copy()
    total_activation = activation.copy()

    # Build transition matrix from pairwise semantic similarity (cosine similarity)
    T = sem_matrix @ sem_matrix.T  # (N, N)
    mask = T > similarity_threshold
    T = T * mask
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums

    # Propagate activation for a limited number of hops
    for _ in range(max_hops):
        activation = T @ activation
        total_activation += gamma * activation

    # Convert activation to distance‑like score (higher activation = lower distance)
    distances = 1.0 - total_activation / np.max(total_activation)

    # Return top‑k memories
    indices = np.argsort(distances)[:k]
    return [(float(distances[idx]), candidates[idx]) for idx in indices]


def retrieve_multi_hop_auto(
        query_semantic: np.ndarray,
        store: MemoryStore,
        k: int = 3,
        base_max_hops: int = 2,
        base_gamma: float = 0.8,
        similarity_threshold: float = 0.5,
    ) -> list:
    """Automatic multi‑hop retrieval.

    Computes the retrieval entropy of the initial semantic similarity scores and
    adjusts the number of activation hops and the decay factor (`gamma`) based on
    this entropy. Higher entropy (more uncertain query) results in more hops and
    a lower decay factor, encouraging broader spreading activation.
    """
    # Gather semantic matrix for all candidates
    candidates = store.get_all_safe()
    if not candidates:
        return []
    sem_matrix = np.array([m.e_semantic for m in candidates], dtype=np.float32)
    # Initial distances (1 - cosine similarity)
    init_sim = sem_matrix @ query_semantic
    distances = 1.0 - init_sim
    # Compute entropy
    entropy = retrieval_entropy(distances)
    # Heuristic adjustment
    max_hops = int(np.clip(base_max_hops + entropy, 1, 10))
    # Reduce gamma as entropy grows (more diffusion)
    gamma = float(np.clip(base_gamma * (1.0 - entropy / 5.0), 0.1, 0.9))
    # Delegate to core multi‑hop function
    return retrieve_multi_hop(
        query_semantic,
        store,
        k=k,
        max_hops=max_hops,
        gamma=gamma,
        similarity_threshold=similarity_threshold,
    )
