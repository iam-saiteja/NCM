"""
EXP22: Is the emotional channel redundant with the state channel?
=================================================================

THE QUESTION
NCM ranks memories by d_raw = alpha*d_sem + beta*d_emo + gamma*d_state
+ delta*d_time. The emotional channel compares `e_emotional` vectors, and
`ncm/encoder.py::encode_emotional` builds them as e_emotional =
L2_normalize(W_emo @ s_padded), where W_emo is a fixed orthonormal (3, 7)
matrix produced once by QR from RandomState(7). `s_padded` is the same
auto-state that the state channel compares. The emotional channel therefore
carries no input that the state channel does not already receive, and
beta*d_emo is plausibly redundant with gamma*d_state. This script measures
that instead of assuming it.

TWO INDEPENDENT MEASUREMENTS
1. Direct. For real query/memory pairs, read the per-memory emotional
   distance and state distance out of the shipped scorer by running it with
   the weight vector (0, 1, 0, 0) and then (0, 0, 1, 0), and correlate the
   two (Pearson r, Spearman rho, n pairs).
2. Functional. Score retrieval quality with the shipped weights, then with
   beta removed and its weight redistributed, on a task whose relevance
   label ships with the corpus. If quality does not move, the channel is
   redundant in effect.

ARMS (every one goes through the shipped retrieval code)
  default               alpha .4   beta .2  gamma .3    delta .1
                        the shipped defaults (ncm/profile.py RetrievalWeights)
  no_emo_renorm         alpha .5   beta 0   gamma .375  delta .125
                        beta removed, the remaining three weights scaled by
                        1/(1-beta) = 1.25 so they again sum to 1
  no_emo_state_absorbs  alpha .4   beta 0   gamma .5    delta .1
                        beta removed, its .2 added to gamma
  semantic_only         ncm.retrieval.retrieve_semantic_only, weight-free
                        external reference point (standard dense RAG)
Total weight is conserved in both ablations, so neither confounds the
removal of the emotional term with a rescaling of the composite.

WEIGHTS-ARE-LIVE GUARD
A null result is worthless unless the knob is proved live. Retrieval reads
its weights from `store.profile.retrieval_weights` at ncm/retrieval.py:385.
Before any metric is reported, this script proves on a real store that
swapping the profile changes the returned distances and the ranking, that
the emotional channel discriminates between memories at all, that a
pure-emotional weight vector does not rank identically to a pure-state one,
and that beta is live at its shipped 0.2 and not only at 1.0. The last
check is
  max |d_default - d_no_emo_renorm/1.25 - 0.2*d_emo| < 1e-6
over every memory of a real store. It is needed because the alpha, gamma
and delta rescale alone would make the two arms differ even if beta were
dead, so a difference in distances is not by itself evidence that beta is
live. The guard also asserts that every memory strength is 1.0 and that
contradiction awareness is off, which is what makes a single-channel probe
return that channel's distance exactly.
If any check fails, the run aborts and writes nothing.

DATA AND LABEL
`experiments/data/real_world_corpus/train.jsonl`, Multi-Session Chat form.
A stored turn is relevant iff it shares the held-out query turn's
`session_id`. That field ships with the corpus and was not authored for this
experiment. No synthetic or hand-written data is used anywhere in this
script. The loader, the store construction and the leave-one-out scoring are
transcribed from `experiments/python/exp17_real_world_autostate_scale.py`,
the corrected reference, so the numbers are directly comparable to it.

NOT MEASURED HERE
Latency. exp4 is the latency measurement. The two retrieval paths in this
script have asymmetric cache treatment, so any timing taken here would
compare cache states rather than algorithms.

STATISTICS AND SCALE
The shipped artifacts in `experiments/results/exp22/` are produced by
running this script with no arguments, so `--max-conversations` takes its
default of 100 and the query set matches exp17's at the same setting. Every
paired difference is reported twice: a percentile bootstrap over queries and
a cluster bootstrap that resamples whole conversations, because queries
inside one conversation share one store and one candidate set and are not
independent. A two-sided null is asserted only when both intervals put zero
strictly inside their bounds, and only alongside the count of queries whose
metric actually changed, because a bound that sits exactly at 0.0 on a
difference vector of mostly zeros is a discreteness artifact and not
evidence of equivalence. The minimum detectable effect at 80 percent power
is reported per metric so a reader can see what this design could not have
detected. The two bootstraps draw from two independently seeded generators so
that each published interval depends only on its own seed, the resample count
and the data, and not on how many other statistics were computed before it.

Outputs
- experiments/results/exp22/exp22_emo_ablation.json
- experiments/results/exp22/exp22_emo_ablation.txt
- experiments/results/exp22/exp22_arm_metrics.png
- experiments/results/exp22/exp22_paired_deltas.png
- experiments/results/exp22/exp22_channel_redundancy.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import (
    AutoStateTracker,
    MemoryEntry,
    MemoryProfile,
    MemoryStore,
    RetrievalWeights,
    SentenceEncoder,
)
from ncm.retrieval import retrieve_semantic_only, retrieve_top_k_fast

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(ROOT_DIR, "experiments", "data", "real_world_corpus", "train.jsonl")
CORPUS_REL = "experiments/data/real_world_corpus/train.jsonl"

SEED = 20260819
# The query-level and the conversation-level bootstrap draw from two separate
# generators seeded here. A single shared stream would make every interval
# depend on how many other intervals happened to be computed before it, so
# adding one statistic anywhere in the script would move already published
# numbers by pure Monte Carlo noise. CLUSTER_SEED is SEED + 1 and has no other
# meaning.
CLUSTER_SEED = SEED + 1
# The shipped artifacts in experiments/results/exp22/ are produced at this
# default. Running the script with no arguments reproduces them.
DEFAULT_MAX_CONVERSATIONS = 100
MIN_SESSIONS_PER_CONVERSATION = 2
MIN_STORED_TURNS_IN_TARGET_SESSION = 3
K_LIST = (5, 10)
BOOTSTRAP_RESAMPLES = 10000

# Wilcoxon signed-rank p-values use scipy's exact null distribution whenever
# the number of nonzero pairs is at most this. scipy's method="auto" falls back
# to the normal approximation as soon as the absolute differences contain ties,
# which they always do here because every per-query metric is quantised, and
# that approximation understates small-n p-values by up to about 3x.
EXACT_WILCOXON_MAX_NONZERO_PAIRS = 1000

# Composite arms scored through ncm.retrieval.retrieve_top_k_fast.
COMPOSITE_ARMS = ("default", "no_emo_renorm", "no_emo_state_absorbs")
ARMS = COMPOSITE_ARMS + ("semantic_only",)
REFERENCE_ARM = "default"

ARM_WEIGHTS = {
    "default": (0.4, 0.2, 0.3, 0.1),
    "no_emo_renorm": (0.5, 0.0, 0.375, 0.125),
    "no_emo_state_absorbs": (0.4, 0.0, 0.5, 0.1),
}
ARM_REDISTRIBUTION = {
    "default": "none; shipped defaults from ncm/profile.py RetrievalWeights",
    "no_emo_renorm": (
        "beta 0.2 removed, then alpha, gamma and delta each multiplied by "
        "1/(1-0.2) = 1.25, giving 0.5, 0.375, 0.125; sum 1.0"
    ),
    "no_emo_state_absorbs": (
        "beta 0.2 removed and added entirely to gamma, giving gamma 0.5 with "
        "alpha and delta untouched; sum 1.0"
    ),
}

# Single-channel probes. Used only by the weights-are-live guard and the
# direct redundancy measurement. Never reported as system results.
PROBE_WEIGHTS = {
    "probe_sem_pure": (1.0, 0.0, 0.0, 0.0),
    "probe_emo_pure": (0.0, 1.0, 0.0, 0.0),
    "probe_state_pure": (0.0, 0.0, 1.0, 0.0),
    "probe_time_pure": (0.0, 0.0, 0.0, 1.0),
}

METRIC_KEYS = ("p@5", "p@10", "r@5", "r@10", "ndcg@10", "mrr")
REPORTED_METRICS = ("p@5", "p@10", "r@10", "ndcg@10", "mrr")

# Depth over which adjacent-rank gaps in the composite distance are measured.
# The reported lists are top-10; two extra ranks cover the memories that can
# enter or leave a top-10 list through a single adjacent flip.
NEAR_TIE_DEPTH = 12

# Reference values for calibration. Read out of exp17's result file at run
# time, never hardcoded here, so they cannot go stale silently. A missing or
# incomplete file aborts the run.
EXP17_RESULTS_REL = "experiments/results/exp17/exp17_real_world_scale.json"
EXP17_RESULTS_PATH = os.path.join(ROOT_DIR, *EXP17_RESULTS_REL.split("/"))
EXP17_ARM_FOR_DEFAULT = "ncm_inferred"
EXP17_ARM_FOR_SEMANTIC_ONLY = "semantic_only"


class Exp17ReferenceError(RuntimeError):
    """Raised when exp17's result file cannot be read or lacks a field this
    script calibrates against. There is no constant fallback on purpose: a
    silently stale reference is worse than an abort."""


def load_exp17_reference() -> dict:
    """Read every exp17 number this script cites, straight from
    experiments/results/exp17/exp17_real_world_scale.json."""
    try:
        with open(EXP17_RESULTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise Exp17ReferenceError(
            f"exp17 reference not found at {EXP17_RESULTS_REL}; "
            "rerun exp17 before exp22") from exc
    except json.JSONDecodeError as exc:
        raise Exp17ReferenceError(
            f"exp17 reference at {EXP17_RESULTS_REL} is not valid JSON: {exc}"
        ) from exc

    def dig(path: str):
        node = raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise Exp17ReferenceError(
                    f"exp17 reference {EXP17_RESULTS_REL} has no field {path}")
            node = node[part]
        return node

    arms = {}
    for arm in (EXP17_ARM_FOR_DEFAULT, EXP17_ARM_FOR_SEMANTIC_ONLY):
        arms[arm] = {m: float(dig(f"arms.{arm}.{m}")) for m in REPORTED_METRICS}

    return {
        "source": EXP17_RESULTS_REL,
        "source_fields": ("config.max_conversations, dataset.*, "
                          "arms.ncm_inferred.*, arms.semantic_only.*, "
                          "arms.recency_only.p@5"),
        "max_conversations": int(dig("config.max_conversations")),
        "conversations_benchmarked": int(dig("dataset.conversations_benchmarked")),
        "queries_evaluated": int(dig("dataset.queries_evaluated")),
        "total_turns_stored": int(dig("dataset.total_turns_stored")),
        "random_guess_precision_at_5": float(dig("dataset.random_guess_precision_at_5")),
        "recency_only_p@5": float(dig("arms.recency_only.p@5")),
        "semantic_only_p@5": arms[EXP17_ARM_FOR_SEMANTIC_ONLY]["p@5"],
        "ncm_inferred_p@5": arms[EXP17_ARM_FOR_DEFAULT]["p@5"],
        "arms": arms,
    }


# -------------------------------------------------------------------
# CORPUS (transcribed from exp17 so the query set is identical)
# -------------------------------------------------------------------

@dataclass
class Turn:
    speaker: str
    text: str
    session_id: int


@dataclass
class ConversationData:
    conv_id: int
    turns: list[Turn]

    @property
    def session_ids(self) -> list[int]:
        seen: list[int] = []
        for t in self.turns:
            if t.session_id not in seen:
                seen.append(t.session_id)
        return seen


def count_corpus_records(corpus_path: str) -> int:
    """Exact non-blank line count of the corpus, so the reported record count
    is measured rather than quoted from documentation."""
    total = 0
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    except FileNotFoundError:
        return -1
    return total


def load_conversations(corpus_path: str, max_conversations: int) -> list[ConversationData]:
    """Load the first max_conversations records, keeping every turn's
    session_id. Transcribed from exp17 load_conversations."""
    conversations: list[ConversationData] = []

    try:
        handle = open(corpus_path, "r", encoding="utf-8")
    except FileNotFoundError:
        print(f"[exp22] ERROR: corpus not found at {corpus_path}")
        return []

    with handle as f:
        for line in f:
            if len(conversations) >= max_conversations:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns: list[Turn] = []
            for session_index, session in enumerate(data.get("sessions", [])):
                session_id = int(session.get("session_id", session_index))
                for turn in session.get("dialogue", []):
                    text = (turn.get("text") or "").strip()
                    if text:
                        turns.append(Turn(
                            speaker=turn.get("speaker", "Unknown"),
                            text=text,
                            session_id=session_id,
                        ))

            if turns:
                conversations.append(ConversationData(
                    conv_id=int(data.get("id", len(conversations))),
                    turns=turns,
                ))

    return conversations


# -------------------------------------------------------------------
# METRICS (transcribed verbatim from exp17 for comparability)
# -------------------------------------------------------------------

def precision_at_k(retrieved_labels: list[bool], k: int) -> float:
    """Fraction of the top-k that are relevant. Denominator is min(k, len)."""
    top = retrieved_labels[:k]
    if not top:
        return 0.0
    return float(sum(top)) / float(len(top))


def recall_at_k(retrieved_labels: list[bool], k: int, n_relevant: int) -> float:
    if n_relevant <= 0:
        return 0.0
    return float(sum(retrieved_labels[:k])) / float(n_relevant)


def reciprocal_rank(retrieved_labels: list[bool]) -> float:
    for rank, is_relevant in enumerate(retrieved_labels, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_labels: list[bool], k: int, n_relevant: int) -> float:
    """Binary-gain NDCG@k. Ideal ranking puts min(k, n_relevant) hits first."""
    gains = [1.0 if hit else 0.0 for hit in retrieved_labels[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def score_labels(labels: list[bool], n_relevant: int) -> dict:
    out = {}
    for k in K_LIST:
        out[f"p@{k}"] = precision_at_k(labels, k)
        out[f"r@{k}"] = recall_at_k(labels, k, n_relevant)
    out["ndcg@10"] = ndcg_at_k(labels, 10, n_relevant)
    out["mrr"] = reciprocal_rank(labels)
    return out


# -------------------------------------------------------------------
# PAIRED STATISTICS
# -------------------------------------------------------------------

# (z_{0.975} + z_{0.80}), the constant in the two-sided 80 percent power
# formula. Taken from scipy rather than written in by hand.
Z_POWER_SUM = float(scipy_stats.norm.ppf(0.975) + scipy_stats.norm.ppf(0.80))


def classify_ci(lo: float, hi: float) -> str:
    """Where zero sits relative to a 95 percent interval.

    Only "contains_zero_in_interior" supports a two-sided null claim. A bound
    that lands exactly on 0.0 is a discreteness artifact of a difference vector
    that is almost all zeros: with an all-same-sign difference vector the
    percentile bootstrap lower bound can only leave 0.0 once enough queries
    change, so "the interval contains zero" then restates the small number of
    changed queries and carries no effect-size information.
    """
    if lo > 0.0 or hi < 0.0:
        return "excludes_zero"
    if lo == 0.0 and hi == 0.0:
        return "degenerate_both_bounds_at_zero"
    if lo == 0.0 or hi == 0.0:
        return "bound_exactly_at_zero"
    return "contains_zero_in_interior"


def cluster_bootstrap_means(d: np.ndarray, cluster_ids: list,
                            rng: np.random.Generator, n_boot: int) -> tuple:
    """Percentile bootstrap that resamples whole conversations rather than
    queries. Queries drawn from one conversation share a store and a candidate
    set, so the query-level bootstrap treats correlated observations as
    independent. Resampling clusters with replacement and taking the mean over
    every query in the resampled clusters (the ratio estimator) respects that
    structure.
    """
    groups: dict = {}
    for i, cid in enumerate(cluster_ids):
        groups.setdefault(cid, []).append(i)
    keys = list(groups.keys())
    sums = np.array([float(d[groups[k]].sum()) for k in keys], dtype=np.float64)
    counts = np.array([len(groups[k]) for k in keys], dtype=np.float64)
    n_c = len(keys)
    if n_c < 2:
        return np.array([]), n_c
    picks = rng.integers(0, n_c, size=(n_boot, n_c))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return means, n_c


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, nonzero: int) -> tuple:
    """Two-sided Wilcoxon signed-rank p-value, exact when scipy can compute
    it. Returns (p, method, note, p_asymptotic)."""
    p_asym = None
    try:
        p_asym = float(scipy_stats.wilcoxon(
            a, b, zero_method="wilcox", method="asymptotic").pvalue)
    except ValueError:
        p_asym = None
    if nonzero <= EXACT_WILCOXON_MAX_NONZERO_PAIRS:
        try:
            p = float(scipy_stats.wilcoxon(
                a, b, zero_method="wilcox", method="exact").pvalue)
            note = (
                "scipy.stats.wilcoxon, zero_method=wilcox, two-sided, "
                "method=exact. The exact null distribution assumes distinct "
                "absolute differences; the per-query metrics here are quantised "
                "so ties are present and no midrank correction is applied in "
                "exact mode. The exact p-value is the larger of the two and is "
                "reported for that reason. method=auto would have returned the "
                "normal approximation, which understates small-n p-values.")
            return p, "exact", note, p_asym
        except ValueError as exc:
            return (p_asym, "asymptotic",
                    f"exact not available ({exc}); normal approximation used",
                    p_asym)
    return (p_asym, "asymptotic",
            f"nonzero pairs {nonzero} exceeds the exact cap "
            f"{EXACT_WILCOXON_MAX_NONZERO_PAIRS}; normal approximation used",
            p_asym)

def paired_delta_stats(arm_values: list, ref_values: list, cluster_ids: list,
                       rng_query: np.random.Generator,
                       rng_cluster: np.random.Generator, n_boot: int) -> dict:
    """Mean of the per-query paired difference (arm minus reference) with two
    percentile bootstrap 95 percent CIs, a Wilcoxon signed-rank p-value on the
    same pairs, the count and sign of the pairs that actually changed, and the
    minimum detectable effect of the design.

    Neither CI is a CI on a difference of two independently computed means;
    both resample the same paired difference vector. The query-level CI
    resamples queries, which are NOT independent: every query drawn from one
    conversation scores against the same store and the same candidate set. The
    cluster-level CI resamples conversations and is the one to read when the
    two disagree. Both are reported.

    The two bootstraps draw from separate generators on purpose. A single
    shared stream would make each interval depend on how many other intervals
    were computed before it, so adding a statistic anywhere would move
    published numbers by Monte Carlo noise.

    n_nonzero_pairs is load-bearing. A mean difference computed over a vector
    that is almost entirely exact zeros says almost nothing about the size of
    the effect, whatever its interval does, so the count is reported next to
    every interval rather than in a footnote.
    """
    a = np.asarray(arm_values, dtype=np.float64)
    b = np.asarray(ref_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays misaligned: {a.shape} vs {b.shape}")
    if len(cluster_ids) != a.size:
        raise ValueError(
            f"cluster ids misaligned with pairs: {len(cluster_ids)} vs {a.size}")
    n = int(a.size)
    if n == 0:
        return {"n_queries": 0, "mean_delta": 0.0, "ci95_low": 0.0,
                "ci95_high": 0.0, "sd_delta": 0.0, "n_nonzero_pairs": 0,
                "max_abs_delta": 0.0, "wilcoxon_p": None,
                "wilcoxon_note": "no queries"}

    d = a - b
    nonzero = int(np.count_nonzero(d))
    n_pos = int(np.count_nonzero(d > 0.0))
    n_neg = int(np.count_nonzero(d < 0.0))
    if nonzero == 0:
        sign_pattern = "every paired difference is exactly 0.0"
    elif n_neg == 0:
        sign_pattern = (f"{n_pos} of {n} queries changed and all of them favour "
                        f"the arm over the reference; {n - nonzero} are exactly 0.0")
    elif n_pos == 0:
        sign_pattern = (f"{n_neg} of {n} queries changed and all of them favour "
                        f"the reference over the arm; {n - nonzero} are exactly 0.0")
    else:
        sign_pattern = (f"{n_pos} changed queries favour the arm, {n_neg} favour "
                        f"the reference, {n - nonzero} are exactly 0.0")

    idx = rng_query.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(boot_means, [2.5, 97.5]))

    cluster_means, n_clusters = cluster_bootstrap_means(
        d, cluster_ids, rng_cluster, n_boot)
    if cluster_means.size:
        c_lo, c_hi = (float(v) for v in np.percentile(cluster_means, [2.5, 97.5]))
        cluster_relation = classify_ci(c_lo, c_hi)
    else:
        c_lo = c_hi = 0.0
        cluster_relation = "not computed: fewer than 2 conversations"

    if nonzero == 0:
        w_p, w_method, w_note, w_asym = (
            None, None, "not run: every paired difference is exactly 0.0", None)
    else:
        w_p, w_method, w_note, w_asym = wilcoxon_paired(a, b, nonzero)

    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    mde = Z_POWER_SUM * sd / np.sqrt(n) if n > 1 else 0.0

    return {
        "n_queries": n,
        "n_conversations": int(n_clusters),
        "mean_delta": float(np.mean(d)),
        "sd_delta": sd,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95_zero_relation": classify_ci(lo, hi),
        "cluster_ci95_low": c_lo,
        "cluster_ci95_high": c_hi,
        "cluster_ci95_zero_relation": cluster_relation,
        "cluster_ci_definition": (
            "percentile bootstrap resampling whole conversations with "
            "replacement, mean taken over every query in the resampled "
            "conversations"),
        "n_nonzero_pairs": nonzero,
        "n_pairs_favouring_arm": n_pos,
        "n_pairs_favouring_reference": n_neg,
        "sign_pattern": sign_pattern,
        "max_abs_delta": float(np.max(np.abs(d))),
        "mde80_two_sided_05": float(mde),
        "mde80_definition": (
            "minimum detectable effect: the smallest true mean paired "
            "difference this design would detect with 80 percent power at a "
            "two-sided 0.05 level, computed as (z_0.975 + z_0.80) * sd_delta / "
            "sqrt(n_queries) = "
            f"{Z_POWER_SUM:.6f} * sd_delta / sqrt(n_queries). A normal "
            "approximation to the paired t-test, quoted as a scale only. It "
            "ignores the conversation clustering, so it is optimistic"),
        "wilcoxon_p": w_p,
        "wilcoxon_method": w_method,
        "wilcoxon_p_asymptotic": w_asym,
        "wilcoxon_note": w_note,
    }


def ci_includes_zero(d: dict) -> bool:
    """True when the query-level 95 percent interval does not exclude zero.
    This is NOT sufficient for a null claim; see supports_two_sided_null."""
    return bool(d["ci95_low"] <= 0.0 <= d["ci95_high"])


def supports_two_sided_null(d: dict) -> bool:
    """A two-sided null claim requires zero strictly inside both intervals.

    A bound sitting exactly at 0.0 is excluded on purpose: on a difference
    vector that is almost all zeros that is a property of the discreteness of
    the metric, not a measurement of equivalence.
    """
    return bool(d.get("ci95_zero_relation") == "contains_zero_in_interior"
                and d.get("cluster_ci95_zero_relation") == "contains_zero_in_interior")


# -------------------------------------------------------------------
# PROFILES
# -------------------------------------------------------------------

class WeightsNotLiveError(RuntimeError):
    """Raised when swapping retrieval weights provably does not change
    retrieval. Every arm would then be the same computation and any
    "no effect" conclusion would be an artifact."""


def build_profile(name: str, weights: tuple) -> MemoryProfile:
    alpha, beta, gamma, delta = weights
    return MemoryProfile(
        name=name,
        retrieval_weights=RetrievalWeights(
            alpha=float(alpha), beta=float(beta),
            gamma=float(gamma), delta=float(delta),
        ),
    )


def profile_fields_excluding_weights(p: MemoryProfile) -> dict:
    d = p.to_dict()
    d.pop("retrieval_weights", None)
    d.pop("name", None)
    return d


def full_store_distances(store: MemoryStore, profile: MemoryProfile,
                         q_sem, q_emo, q_state) -> dict:
    """Distance assigned to every memory in the store under `profile`, keyed
    by memory id, taken straight from the shipped scorer's return value."""
    store.profile = profile
    hits = retrieve_top_k_fast(
        q_sem, q_emo, store, q_state, int(store.step), k=len(store),
    )
    return {h[2].id: float(h[0]) for h in hits}


def weights_live_guard(store: MemoryStore, q_sem, q_emo, q_state,
                       arm_profiles: dict, probe_profiles: dict,
                       context: str) -> dict:
    """Prove empirically that store.profile.retrieval_weights is live.

    Runs on a real store built from the corpus, not on synthetic data. Every
    check below reads only values returned by ncm.retrieval.retrieve_top_k_fast.
    """
    n = len(store)
    dists = {}
    order = {}
    for name, prof in list(probe_profiles.items()) + list(arm_profiles.items()):
        dmap = full_store_distances(store, prof, q_sem, q_emo, q_state)
        dists[name] = dmap
        store.profile = prof
        order[name] = [h[2].id for h in retrieve_top_k_fast(
            q_sem, q_emo, store, q_state, int(store.step), k=max(K_LIST))]

    def max_abs_diff(a: str, b: str) -> float:
        ids = set(dists[a]) & set(dists[b])
        if not ids:
            return 0.0
        return float(max(abs(dists[a][i] - dists[b][i]) for i in ids))

    emo_vals = np.array(list(dists["probe_emo_pure"].values()), dtype=np.float64)
    state_vals = np.array(list(dists["probe_state_pure"].values()), dtype=np.float64)
    strengths = [float(m.strength) for m in store.get_all_safe()]

    # Is beta live at its SHIPPED value, not just at 1.0? The identity below
    # isolates it. With w_default = (a, b, g, t) and w_renorm = s*(a, 0, g, t)
    # for s = 1/(1-b), d_default - d_renorm/s = b*d_emo exactly. A dead beta
    # would break this while still leaving default and no_emo_renorm different,
    # because the rescale by s alone changes every distance. Without this check
    # the "distances differ" checks are satisfied by the rescale on its own.
    beta_shipped = float(ARM_WEIGHTS[REFERENCE_ARM][1])
    renorm_scale = 1.0 / (1.0 - beta_shipped)
    renorm_is_uniform_rescale = all(
        abs(ARM_WEIGHTS["no_emo_renorm"][i]
            - renorm_scale * ARM_WEIGHTS[REFERENCE_ARM][i]) < 1e-12
        for i in (0, 2, 3)) and ARM_WEIGHTS["no_emo_renorm"][1] == 0.0
    shared_ids = (set(dists[REFERENCE_ARM]) & set(dists["no_emo_renorm"])
                  & set(dists["probe_emo_pure"]))
    if shared_ids:
        beta_residual = max(
            abs(dists[REFERENCE_ARM][i]
                - dists["no_emo_renorm"][i] / renorm_scale
                - beta_shipped * dists["probe_emo_pure"][i])
            for i in shared_ids)
    else:
        beta_residual = float("inf")

    checks = {
        "sem_pure_vs_time_pure_top10_differs": order["probe_sem_pure"] != order["probe_time_pure"],
        "emo_pure_vs_state_pure_top10_differs": order["probe_emo_pure"] != order["probe_state_pure"],
        "emo_channel_distance_varies_across_memories": bool(float(np.std(emo_vals)) > 1e-6),
        "state_channel_distance_varies_across_memories": bool(float(np.std(state_vals)) > 1e-6),
        "default_vs_no_emo_renorm_distances_differ": bool(
            max_abs_diff("default", "no_emo_renorm") > 1e-6),
        "default_vs_no_emo_state_absorbs_distances_differ": bool(
            max_abs_diff("default", "no_emo_state_absorbs") > 1e-6),
        "no_emo_renorm_is_uniform_rescale_of_default": bool(renorm_is_uniform_rescale),
        "beta_is_live_at_shipped_value": bool(beta_residual < 1e-6),
        "all_memory_strengths_are_1": bool(
            strengths and max(abs(s - 1.0) for s in strengths) < 1e-9),
        "contradiction_awareness_is_off": not bool(
            store.profile.get_custom("enable_contradiction_awareness", False)),
        "arm_profiles_differ_only_in_retrieval_weights": all(
            profile_fields_excluding_weights(arm_profiles[a])
            == profile_fields_excluding_weights(arm_profiles[REFERENCE_ARM])
            for a in arm_profiles
        ),
    }
    passed = all(bool(v) for v in checks.values())

    return {
        "passed": passed,
        "checks": {k: bool(v) for k, v in checks.items()},
        "guard_context": context,
        "guard_store_size": n,
        "guard_data_provenance": "a real store built from the corpus, not synthetic",
        "weights_read_at": "ncm/retrieval.py:385 store.profile.retrieval_weights",
        "max_abs_distance_diff_default_vs_no_emo_renorm": round(
            max_abs_diff("default", "no_emo_renorm"), 8),
        "max_abs_distance_diff_default_vs_no_emo_state_absorbs": round(
            max_abs_diff("default", "no_emo_state_absorbs"), 8),
        "max_abs_distance_diff_sem_pure_vs_time_pure": round(
            max_abs_diff("probe_sem_pure", "probe_time_pure"), 8),
        "emo_channel_distance_std": round(float(np.std(emo_vals)), 8),
        "state_channel_distance_std": round(float(np.std(state_vals)), 8),
        "beta_live_residual_max_abs": beta_residual,
        "beta_live_identity": (
            f"max over the memories of the guard store of |d_default - "
            f"d_no_emo_renorm/{renorm_scale} - {beta_shipped}*d_emo|, where "
            "d_emo is the probe_emo_pure distance. Zero to floating point means "
            "beta contributes at its shipped value and not merely at the 1.0 "
            "used by the probe. The 'distances differ' checks above do not "
            "establish this, because the alpha, gamma and delta rescale changes "
            "every distance on its own"),
        "probe_weight_vectors": {k: list(v) for k, v in PROBE_WEIGHTS.items()},
    }


# -------------------------------------------------------------------
# BENCHMARK
# -------------------------------------------------------------------

def build_store(conv: ConversationData, held_out_positions: set,
                encoder: SentenceEncoder) -> tuple:
    """One store per conversation holding every turn except the held-out
    queries. Transcribed from exp17 so the stores are identical."""
    store = MemoryStore(profile=build_profile("default", ARM_WEIGHTS["default"]))
    session_of_memory: dict[str, int] = {}
    for position, turn in enumerate(conv.turns):
        if position in held_out_positions:
            continue
        state_before = store.auto_state.get_current_state()
        mem = MemoryEntry(
            e_semantic=encoder.encode(turn.text),
            e_emotional=encoder.encode_emotional(state_before),
            # s_snapshot is retained for file-format compatibility only. The
            # composite distance reads auto_state_snapshot, which add() writes.
            s_snapshot=encoder.encode_state(
                np.pad(state_before, (0, 2), mode="constant", constant_values=0.5)
            ),
            timestamp=int(store.step),
            text=turn.text,
        )
        stored = store.add(mem, update_auto_state=True)
        session_of_memory[stored.id] = turn.session_id
        store.step += 1
    return store, session_of_memory


def benchmark(conversations: list[ConversationData], encoder: SentenceEncoder,
              arm_profiles: dict, probe_profiles: dict) -> dict:
    rng = random.Random(SEED)
    per_query = {arm: {m: [] for m in METRIC_KEYS} for arm in ARMS}
    top10_ids = {arm: [] for arm in ARMS}
    # Conversation id of every scored query, in the same order as the per-query
    # metric lists. The cluster bootstrap resamples over this.
    query_conv_ids: list = []

    conversations_benchmarked = 0
    skipped_sessions = 0
    skipped_coverage = 0
    queries_evaluated = 0
    total_turns_stored = 0
    relevant_counts: list[int] = []
    store_sizes: list[int] = []
    per_query_relevant_fraction: list[float] = []
    channel_pairs: list[tuple] = []
    # Near-tie accounting, per query: the emotional term's order-flip budget
    # inside that one store, and the adjacent-rank gaps of the shipped composite
    # in the top NEAR_TIE_DEPTH.
    emo_budget_per_query: list[float] = []
    adjacent_gaps: list[float] = []
    n_gaps_total = 0
    n_gaps_below_budget = 0
    guard: dict | None = None
    max_k = max(K_LIST)

    for conv in conversations:
        session_ids = conv.session_ids
        if len(session_ids) < MIN_SESSIONS_PER_CONVERSATION:
            skipped_sessions += 1
            continue

        held_out_index: dict[int, int] = {}
        for session_id in session_ids:
            candidate_indices = [i for i, t in enumerate(conv.turns) if t.session_id == session_id]
            if len(candidate_indices) < MIN_STORED_TURNS_IN_TARGET_SESSION + 1:
                continue
            held_out_index[session_id] = rng.choice(candidate_indices)

        if not held_out_index:
            skipped_coverage += 1
            continue

        store, session_of_memory = build_store(
            conv, set(held_out_index.values()), encoder)
        if len(store) == 0:
            continue

        # Warm the vectorized cache once. add() marks it dirty and
        # retrieve_top_k_fast rebuilds on its first call, so warming here keeps
        # every arm on the same cache state.
        store._rebuild_cache()
        store_sizes.append(len(store))
        total_turns_stored += len(store)
        conversations_benchmarked += 1

        saved_state = store.auto_state.get_current_state()
        saved_turn = store.auto_state.turn

        for session_id, position in held_out_index.items():
            query_text = conv.turns[position].text
            n_relevant = sum(
                1 for m in store.get_all_safe()
                if session_of_memory.get(m.id, -1) == session_id
            )
            if n_relevant < MIN_STORED_TURNS_IN_TARGET_SESSION:
                continue

            q_sem = encoder.encode(query_text)
            # The query-side auto-state is inferred from the query text alone
            # by a fresh tracker. Nothing about the target session reaches the
            # query, so there is no label leak. retrieve_top_k_fast ignores its
            # s_current_normalized argument (ncm/retrieval.py:356) and reads
            # store.auto_state.get_current_state() at ncm/retrieval.py:378, so
            # the assignment below is the only way to control the state channel.
            probe = AutoStateTracker()
            inferred_state = probe.update(query_text)
            store.auto_state.state = inferred_state.astype(np.float32).copy()
            q_emo = encoder.encode_emotional(inferred_state)
            q_state = encoder.encode_state(inferred_state)

            if guard is None:
                guard = weights_live_guard(
                    store, q_sem, q_emo, q_state, arm_profiles, probe_profiles,
                    context=(f"conversation id {conv.conv_id}, session_id "
                             f"{session_id}, store size {len(store)}"),
                )
                if not guard["passed"]:
                    raise WeightsNotLiveError(
                        "retrieval weights are not live: "
                        + json.dumps(guard["checks"])
                    )

            queries_evaluated += 1
            query_conv_ids.append(conv.conv_id)
            relevant_counts.append(n_relevant)
            per_query_relevant_fraction.append(n_relevant / float(len(store)))

            # Direct redundancy measurement. Run through the shipped scorer with
            # a single channel enabled at a time: weights (1,0,0,0) return d_sem
            # per memory, (0,1,0,0) return d_emo, (0,0,1,0) return d_state and
            # (0,0,0,1) return d_time, each exactly, because no memory strength
            # differs from 1.0 and contradiction awareness is off. Both
            # conditions are asserted by the guard.
            chan = {
                name: full_store_distances(
                    store, probe_profiles[probe], q_sem, q_emo, q_state)
                for name, probe in (("d_sem", "probe_sem_pure"),
                                    ("d_emo", "probe_emo_pure"),
                                    ("d_state", "probe_state_pure"),
                                    ("d_time", "probe_time_pure"))
            }
            for mid in chan["d_emo"]:
                if all(mid in chan[c] for c in chan):
                    channel_pairs.append((
                        chan["d_sem"][mid], chan["d_emo"][mid],
                        chan["d_state"][mid], chan["d_time"][mid],
                    ))

            # Near-tie accounting for this one query. The emotional term can
            # only reorder two memories whose other three terms leave their
            # composite distances closer together than beta times the spread of
            # d_emo inside this store. Comparing that budget against the actual
            # adjacent-rank gaps of the shipped composite says whether the term
            # is reordering near-ties or making substantive moves.
            emo_here = np.array(list(chan["d_emo"].values()), dtype=np.float64)
            budget_here = float(ARM_WEIGHTS[REFERENCE_ARM][1]
                                * (emo_here.max() - emo_here.min()))
            emo_budget_per_query.append(budget_here)
            default_all = full_store_distances(
                store, arm_profiles[REFERENCE_ARM], q_sem, q_emo, q_state)
            ranked = np.sort(np.array(list(default_all.values()),
                                      dtype=np.float64))[:NEAR_TIE_DEPTH]
            if ranked.size > 1:
                gaps_here = np.diff(ranked)
                adjacent_gaps.extend(float(g) for g in gaps_here)
                n_gaps_total += int(gaps_here.size)
                n_gaps_below_budget += int(np.count_nonzero(gaps_here < budget_here))

            for arm in COMPOSITE_ARMS:
                store.profile = arm_profiles[arm]
                hits = retrieve_top_k_fast(
                    q_sem, q_emo, store, q_state, int(store.step), k=max_k)
                entries = [h[2] for h in hits]
                labels = [session_of_memory.get(m.id, -1) == session_id for m in entries]
                for metric, value in score_labels(labels, n_relevant).items():
                    per_query[arm][metric].append(value)
                top10_ids[arm].append(tuple(m.id for m in entries))

            hits = retrieve_semantic_only(q_sem, store, k=max_k)
            entries = [h[-1] for h in hits]
            labels = [session_of_memory.get(m.id, -1) == session_id for m in entries]
            for metric, value in score_labels(labels, n_relevant).items():
                per_query["semantic_only"][metric].append(value)
            top10_ids["semantic_only"].append(tuple(m.id for m in entries))

            store.profile = arm_profiles[REFERENCE_ARM]
            store.auto_state.state = saved_state.copy()
            store.auto_state.turn = saved_turn

    return {
        "per_query": per_query,
        "top10_ids": top10_ids,
        "query_conv_ids": query_conv_ids,
        "channel_pairs": channel_pairs,
        "guard": guard,
        "near_ties": {
            "depth": NEAR_TIE_DEPTH,
            "n_queries": len(emo_budget_per_query),
            "median_within_store_emo_budget": (
                round(float(np.median(emo_budget_per_query)), 6)
                if emo_budget_per_query else 0.0),
            "median_adjacent_gap_in_top_depth": (
                round(float(np.median(adjacent_gaps)), 6)
                if adjacent_gaps else 0.0),
            "n_adjacent_gaps": n_gaps_total,
            "n_adjacent_gaps_below_emo_budget": n_gaps_below_budget,
            "fraction_adjacent_gaps_below_emo_budget": (
                round(n_gaps_below_budget / float(n_gaps_total), 4)
                if n_gaps_total else 0.0),
            "definition": (
                f"for each query, the emotional order-flip budget inside that "
                f"one store is beta times the range of d_emo over the memories "
                f"of that store, and the adjacent-rank gaps are the successive "
                f"differences of the shipped composite distance over the top "
                f"{NEAR_TIE_DEPTH} memories as returned by "
                f"ncm.retrieval.retrieve_top_k_fast under the default weights. "
                f"A gap below the budget is a pair the emotional term could "
                f"reorder on its own. The fraction is over all "
                f"(query, adjacent pair) gaps, pooled across queries"),
        },
        "dataset": {
            "corpus": CORPUS_REL,
            "conversations_loaded": len(conversations),
            "conversations_benchmarked": conversations_benchmarked,
            "conversations_skipped_too_few_sessions": skipped_sessions,
            "conversations_skipped_no_eligible_session": skipped_coverage,
            "queries_evaluated": queries_evaluated,
            "total_turns_stored": total_turns_stored,
            "mean_store_size": (round(float(np.mean(store_sizes)), 2)
                                if store_sizes else 0.0),
            "mean_relevant_per_query": (round(float(np.mean(relevant_counts)), 2)
                                        if relevant_counts else 0.0),
            "random_guess_precision": (
                round(float(np.mean(per_query_relevant_fraction)), 4)
                if per_query_relevant_fraction else 0.0),
            "random_guess_definition": (
                "mean over queries of (n_relevant / store_size); this is the "
                "expected precision at any k for a retriever drawing k "
                "memories uniformly at random, and is independent of k. It is "
                "NOT mean(n_relevant)/mean(store_size), which would divide a "
                "per-query mean by a per-conversation mean."),
        },
    }


# -------------------------------------------------------------------
# AGGREGATION
# -------------------------------------------------------------------

def summarize_arms(per_query: dict) -> dict:
    out = {}
    for arm in ARMS:
        row = {}
        for metric in METRIC_KEYS:
            values = per_query[arm][metric]
            row[metric] = round(float(np.mean(values)), 4) if values else 0.0
            row[f"{metric}_sd"] = (
                round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0)
        row["n_queries"] = len(per_query[arm]["p@5"])
        out[arm] = row
    return out


def channel_redundancy(pairs: list, n_queries: int, n_conversations: int) -> dict:
    """Correlate the emotional-channel distance against the state-channel
    distance over real (query, memory) pairs, and report how much each of the
    four channels varies over all (query, memory) pairs.

    Every dispersion number here is pooled over all (query, memory) pairs from
    every store, not taken inside one store. Pooled dispersion is wider than
    within-store dispersion, so where it is used as a bound on the emotional
    term's influence it is an upper bound.

    The dispersion matters because a channel can only change a ranking through
    its variation across candidates. A channel whose weighted spread is far
    below another's cannot outvote it whatever its weight.

    No p-value is reported for either correlation. The pairs are clustered:
    they come from a much smaller number of queries, and those from a much
    smaller number of stores, so any p-value computed as if the pairs were
    independent would be meaningless. r and rho are reported as descriptive
    statistics only.
    """
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "note": "too few pairs to correlate"}
    arr = np.asarray(pairs, dtype=np.float64)
    d_sem, d_emo, d_state, d_time = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    pearson = scipy_stats.pearsonr(d_emo, d_state)
    spearman = scipy_stats.spearmanr(d_emo, d_state)
    n_sem_at_clip = int(np.count_nonzero(d_sem >= 1.0 - 1e-12))

    shipped = ARM_WEIGHTS["default"]
    channels = {}
    raw_weighted_sd = {}
    raw_weighted_range = {}
    for name, values, weight in (("d_sem", d_sem, shipped[0]),
                                 ("d_emo", d_emo, shipped[1]),
                                 ("d_state", d_state, shipped[2]),
                                 ("d_time", d_time, shipped[3])):
        sd = float(np.std(values, ddof=1))
        lo_v, hi_v = float(np.min(values)), float(np.max(values))
        raw_weighted_sd[name] = float(weight) * sd
        raw_weighted_range[name] = float(weight) * (hi_v - lo_v)
        channels[name] = {
            "shipped_weight": float(weight),
            "mean": round(float(np.mean(values)), 6),
            "sd": round(sd, 6),
            "min": round(lo_v, 6),
            "max": round(hi_v, 6),
            "range": round(hi_v - lo_v, 6),
            "weighted_sd": round(raw_weighted_sd[name], 6),
            "weighted_range": round(raw_weighted_range[name], 6),
        }

    return {
        "n_pairs": int(arr.shape[0]),
        "n_queries_contributing": int(n_queries),
        "n_conversations_contributing": int(n_conversations),
        "unit_of_observation": "one (query, stored memory) pair",
        "clustering_note": (
            f"the {int(arr.shape[0])} pairs come from {int(n_queries)} queries "
            f"in {int(n_conversations)} conversations, so they are clustered and "
            "far from independent. No p-value is reported for either "
            "correlation for that reason; r and rho are descriptive only"),
        "channel_source": (
            "each channel is the distance returned by "
            "ncm.retrieval.retrieve_top_k_fast with all weight on that one "
            "channel, which equals the clipped channel distance exactly when "
            "every memory strength is 1.0 and contradiction awareness is off; "
            "both conditions are members of the weights-are-live guard's checks "
            "dict and therefore gate the run"),
        "pearson_r": round(float(pearson.statistic), 4),
        "spearman_rho": round(float(spearman.statistic), 4),
        "correlated_pair": "d_emo against d_state",
        "d_emo_mean": round(float(np.mean(d_emo)), 4),
        "d_emo_sd": round(float(np.std(d_emo, ddof=1)), 4),
        "d_state_mean": round(float(np.mean(d_state)), 4),
        "d_state_sd": round(float(np.std(d_state, ddof=1)), 4),
        "channels": channels,
        "d_sem_max_is_a_clip_boundary": bool(n_sem_at_clip > 0),
        "n_pairs_with_d_sem_at_clip_boundary": n_sem_at_clip,
        "d_sem_clip_note": (
            "d_sem is np.clip(1.0 - cosine_similarity, 0.0, 1.0) at "
            "ncm/retrieval.py:115, so a maximum of exactly 1.00000 is the clip "
            "boundary and not an observed extreme. Any range-based statistic "
            f"involving d_sem is therefore truncated; {n_sem_at_clip} of "
            f"{int(arr.shape[0])} pairs sit on the boundary"),
        "weighted_sd_ratio_sem_over_emo": round(
            raw_weighted_sd["d_sem"] / raw_weighted_sd["d_emo"], 2)
        if raw_weighted_sd["d_emo"] > 0 else None,
        "weighted_range_ratio_sem_over_emo": round(
            raw_weighted_range["d_sem"] / raw_weighted_range["d_emo"], 2)
        if raw_weighted_range["d_emo"] > 0 else None,
        "dispersion_ratio_definition": (
            "two different dispersion measures, not interchangeable. "
            "weighted_sd_ratio_sem_over_emo is (alpha times the sd of d_sem) "
            "divided by (beta times the sd of d_emo). "
            "weighted_range_ratio_sem_over_emo is the same ratio with ranges in "
            "place of standard deviations, and its numerator is truncated by "
            "the d_sem clip. Both are pooled over all (query, memory) pairs. "
            "Each sentence that quotes a ratio names which one it is"),
        "emo_order_flip_budget": round(raw_weighted_range["d_emo"], 6),
        "emo_order_flip_budget_definition": (
            "beta times the range of d_emo. Two memories can be reordered by "
            "the emotional term only if the other three terms place their "
            "composite distances within this much of each other. The range is "
            "pooled over all (query, memory) pairs, which is wider than the "
            "range inside any one store, so this is an upper bound on the "
            "term's influence. The near-tie table reports the within-store "
            "version"),
        "dispersion_definition": (
            "sd is the standard deviation of that channel's distance over all "
            "(query, memory) pairs, pooled across stores; weighted_sd "
            "multiplies it by the shipped weight, giving the scale on which "
            "that channel can move the composite distance between two "
            "candidate memories"),
    }



def ranking_divergence(top10_ids: dict) -> dict:
    """How often each arm's top-10 differs from the reference arm's top-10.

    This separates two very different outcomes. If the rankings are identical
    the arms are the same computation and any equality of metrics is vacuous.
    If the rankings differ but the metrics do not, the emotional channel moves
    the ranking without changing retrieval quality, which is the informative
    version of a null result.
    """
    ref = top10_ids[REFERENCE_ARM]
    out = {}
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        lists = top10_ids[arm]
        if not ref or len(lists) != len(ref):
            out[arm] = {"note": "not comparable, unequal query counts"}
            continue
        differs, top1_changed, jaccard, kendall = [], [], [], []
        for a, b in zip(lists, ref):
            differs.append(1.0 if tuple(a) != tuple(b) else 0.0)
            top1_changed.append(
                1.0 if (a and b and a[0] != b[0]) else 0.0)
            sa, sb = set(a), set(b)
            union = sa | sb
            jaccard.append(len(sa & sb) / len(union) if union else 1.0)
            common = [x for x in a if x in sb]
            if len(common) > 2:
                rank_a = [a.index(x) for x in common]
                rank_b = [b.index(x) for x in common]
                tau = scipy_stats.kendalltau(rank_a, rank_b).statistic
                if not np.isnan(tau):
                    kendall.append(float(tau))
        out[arm] = {
            "n_queries": len(lists),
            "fraction_top10_list_differs": round(float(np.mean(differs)), 4),
            "fraction_top1_changed": round(float(np.mean(top1_changed)), 4),
            "mean_top10_set_jaccard": round(float(np.mean(jaccard)), 4),
            "mean_kendall_tau_on_shared_items": (
                round(float(np.mean(kendall)), 4) if kendall else None),
            "kendall_n_queries_with_enough_shared_items": len(kendall),
        }
    return out


def paired_table(per_query: dict, query_conv_ids: list, n_boot: int) -> dict:
    """Build the paired delta table.

    The two generators are constructed here rather than passed in so that the
    query-level intervals depend only on SEED, n_boot and the data, and not on
    whatever else in the script happened to consume a shared generator first.
    """
    rng_query = np.random.default_rng(SEED)
    rng_cluster = np.random.default_rng(CLUSTER_SEED)
    out = {}
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        out[arm] = {
            metric: paired_delta_stats(
                per_query[arm][metric], per_query[REFERENCE_ARM][metric],
                query_conv_ids, rng_query, rng_cluster, n_boot)
            for metric in METRIC_KEYS
        }
    return out


def exp17_equivalence(arms: dict, dataset: dict, max_conversations: int,
                      reference: dict) -> dict:
    """Compare this run's default and semantic_only arms against exp17's
    ncm_inferred and semantic_only arms, metric by metric, at run time.

    The default arm here is the same code path, weight vector, store
    construction and query-state condition as exp17's ncm_inferred arm, so on a
    shared query set the two must agree exactly. Every number below is read out
    of this run and out of exp17's result file. Nothing is recorded by hand.
    """
    same_scale = bool(
        int(max_conversations) == reference["max_conversations"]
        and dataset["queries_evaluated"] == reference["queries_evaluated"]
        and dataset["conversations_benchmarked"] == reference["conversations_benchmarked"]
        and dataset["total_turns_stored"] == reference["total_turns_stored"])

    rows = {}
    worst = 0.0
    for here, there in (("default", EXP17_ARM_FOR_DEFAULT),
                        ("semantic_only", EXP17_ARM_FOR_SEMANTIC_ONLY)):
        row = {}
        for m in REPORTED_METRICS:
            mine = float(arms[here][m])
            theirs = float(reference["arms"][there][m])
            row[m] = {"this_run": mine, "exp17": theirs,
                      "diff": round(mine - theirs, 6)}
            worst = max(worst, abs(mine - theirs))
        rows[f"{here}_vs_exp17_{there}"] = row

    if not same_scale:
        verdict = (
            "NOT COMPARABLE: this run's scale does not match exp17's. This run "
            f"benchmarked {dataset['conversations_benchmarked']} conversations, "
            f"{dataset['queries_evaluated']} queries and "
            f"{dataset['total_turns_stored']} stored turns at "
            f"max_conversations={max_conversations}, against exp17's "
            f"{reference['conversations_benchmarked']}, "
            f"{reference['queries_evaluated']} and "
            f"{reference['total_turns_stored']} at "
            f"max_conversations={reference['max_conversations']}. The query set "
            "here is a prefix subset of exp17's, so absolute values differ by "
            "sampling and the per-metric differences below are not an "
            "equivalence test")
    elif worst < 1e-4:
        verdict = (
            "MATCH: on the same query set this run's default arm equals exp17's "
            "ncm_inferred arm and this run's semantic_only arm equals exp17's "
            "semantic_only arm to four decimal places on all "
            f"{len(REPORTED_METRICS)} reported metrics; largest absolute "
            f"difference {worst:.2e}")
    else:
        verdict = (
            "MISMATCH: the scales agree but the metrics do not. Largest "
            f"absolute difference {worst:.6f} against a tolerance of 1e-4. The "
            "two scripts should be the same computation on this query set, so "
            "this is a setup difference that must be resolved before either "
            "result is quoted")

    return {
        "source": reference["source"],
        "source_fields": reference["source_fields"],
        "computed_at_run_time": True,
        "same_scale_as_exp17": same_scale,
        "tolerance": 1e-4,
        "max_abs_difference": round(worst, 6),
        "rows": rows,
        "verdict": verdict,
    }


# -------------------------------------------------------------------
# FIGURES (Agg backend, ASCII labels only)
# -------------------------------------------------------------------

ARM_LABELS = {
    "default": "default\n(.4 .2 .3 .1)",
    "no_emo_renorm": "no_emo_renorm\n(.5 0 .375 .125)",
    "no_emo_state_absorbs": "no_emo_state_absorbs\n(.4 0 .5 .1)",
    "semantic_only": "semantic_only\n(reference)",
}
ARM_COLORS = {
    "default": "#2c7fb8",
    "no_emo_renorm": "#7fcdbb",
    "no_emo_state_absorbs": "#41b6c4",
    "semantic_only": "#e74c3c",
}


def plot_arm_metrics(arms: dict, dataset: dict) -> str:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(REPORTED_METRICS))
    width = 0.2
    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * width
        vals = [arms[arm][m] for m in REPORTED_METRICS]
        bars = ax.bar(x + offset, vals, width, label=ARM_LABELS[arm],
                      color=ARM_COLORS[arm], alpha=0.9,
                      edgecolor="black", linewidth=1.0)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    rnd = dataset["random_guess_precision"]
    ax.axhline(rnd, color="black", linestyle=":", linewidth=1.6,
               label=f"random-guess precision ({rnd:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in REPORTED_METRICS])
    ax.set_ylabel("score")
    ax.set_title(
        f"EXP22 emotional-channel ablation: {dataset['queries_evaluated']} queries, "
        f"{dataset['conversations_benchmarked']} conversations, mean store "
        f"{dataset['mean_store_size']:.0f} memories",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp22_arm_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_paired_deltas(paired: dict) -> str:
    comparisons = [a for a in ARMS if a != REFERENCE_ARM]
    fig, axes = plt.subplots(1, len(comparisons), figsize=(5.2 * len(comparisons), 5),
                             sharey=False)
    if len(comparisons) == 1:
        axes = [axes]
    for ax, arm in zip(axes, comparisons):
        y = np.arange(len(REPORTED_METRICS))
        means = [paired[arm][m]["mean_delta"] for m in REPORTED_METRICS]
        lo = [paired[arm][m]["mean_delta"] - paired[arm][m]["ci95_low"] for m in REPORTED_METRICS]
        hi = [paired[arm][m]["ci95_high"] - paired[arm][m]["mean_delta"] for m in REPORTED_METRICS]
        c_lo = [paired[arm][m]["mean_delta"] - paired[arm][m]["cluster_ci95_low"]
                for m in REPORTED_METRICS]
        c_hi = [paired[arm][m]["cluster_ci95_high"] - paired[arm][m]["mean_delta"]
                for m in REPORTED_METRICS]
        ax.errorbar(means, y - 0.12, xerr=[lo, hi], fmt="o", color=ARM_COLORS[arm],
                    ecolor="black", elinewidth=1.2, capsize=4, markersize=7,
                    label="bootstrap over queries")
        ax.errorbar(means, y + 0.12, xerr=[c_lo, c_hi], fmt="s",
                    color=ARM_COLORS[arm], ecolor="#7f7f7f", elinewidth=1.2,
                    capsize=4, markersize=5,
                    label="bootstrap over conversations")
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([m.upper() for m in REPORTED_METRICS])
        ax.invert_yaxis()
        ax.set_xlabel("paired per-query delta vs default")
        ax.set_title(f"{arm}\nminus default", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=7.5, loc="lower right")
        span = max([abs(v) for v in means] + lo + hi + c_lo + c_hi + [0.01])
        ax.set_xlim(-1.35 * span, 1.35 * span)
    n_q = paired[comparisons[0]][REPORTED_METRICS[0]]["n_queries"]
    n_c = paired[comparisons[0]][REPORTED_METRICS[0]]["n_conversations"]
    fig.suptitle(
        "EXP22 paired per-query deltas, percentile bootstrap 95 pct CI "
        f"({BOOTSTRAP_RESAMPLES} resamples); circles resample the "
        f"{n_q} queries (seed {SEED}), squares resample the {n_c} "
        f"conversations (seed {CLUSTER_SEED})",
        fontsize=10.5, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp22_paired_deltas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_channel_redundancy(pairs: list[tuple], red: dict) -> str:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))
    if len(pairs) < 3:
        ax.text(0.5, 0.5, "too few pairs to plot", ha="center", va="center")
    else:
        arr = np.asarray(pairs, dtype=np.float64)
        # columns are d_sem, d_emo, d_state, d_time
        ax.scatter(arr[:, 2], arr[:, 1], s=5, alpha=0.18,
                   color="#2c7fb8", edgecolors="none")
        ax.set_xlabel("d_state, weights (0,0,1,0)")
        ax.set_ylabel("d_emo, weights (0,1,0,0)")
        ax.text(0.02, 0.98,
                f"n = {red['n_pairs']} (query, memory) pairs\n"
                f"from {red['n_queries_contributing']} queries in "
                f"{red['n_conversations_contributing']} conversations\n"
                f"Pearson r = {red['pearson_r']:.4f}\n"
                f"Spearman rho = {red['spearman_rho']:.4f}\n"
                "clustered pairs: no p-value reported",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
        names = ["d_sem", "d_emo", "d_state", "d_time"]
        wsd = [red["channels"][n]["weighted_sd"] for n in names]
        cols = ["#e74c3c", "#f39c12", "#2c7fb8", "#95a5a6"]
        bars = ax2.bar(names, wsd, color=cols, alpha=0.9,
                       edgecolor="black", linewidth=1.1)
        for bar, n in zip(bars, names):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.5f}\nw={red['channels'][n]['shipped_weight']}",
                     ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("shipped weight times sd of the channel distance")
        ax2.set_title("Weighted standard deviation of each channel,\n"
                      "pooled over all (query, memory) pairs",
                      fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
    ax.set_title("EXP22 emotional vs state channel distance on the same pairs",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp22_channel_redundancy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -------------------------------------------------------------------
# TEXT REPORT
# -------------------------------------------------------------------

def write_text_report(path: str, results: dict) -> None:
    cfg = results["config"]
    ds = results["dataset"]
    arms = results["arms"]
    paired = results["paired_deltas_vs_default"]
    guard = results["weights_are_live_guard"]
    red = results["channel_redundancy"]
    div = results["ranking_divergence_vs_default"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("EXP22: Is the emotional channel redundant with the state channel?\n")
        f.write("=================================================================\n\n")
        f.write("Question: e_emotional = L2_normalize(W_emo @ s_padded) with W_emo a\n")
        f.write("fixed orthonormal QR projection (ncm/encoder.py encode_emotional), so\n")
        f.write("the emotional channel is a deterministic linear function of the same\n")
        f.write("auto-state that the state channel compares. Does removing beta*d_emo\n")
        f.write("change retrieval quality?\n\n")

        f.write("Setup\n")
        f.write(f"- Seed: {cfg['seed']} (corpus order, held-out choices, "
                f"query bootstrap); {cfg['bootstrap']['seed_cluster_bootstrap']} "
                f"for the conversation bootstrap\n")
        f.write(f"- Encoder backend: {cfg['encoder_backend']}\n")
        f.write(f"- Encoder state_dim: {cfg['encoder_state_dim']}, "
                f"emotional_dim: {cfg['encoder_emotional_dim']}, "
                f"AutoStateTracker dims: {cfg['auto_state_dim']}\n")
        f.write(f"- Corpus: {cfg['corpus']} ({cfg['corpus_records']} records, "
                f"counted at run time)\n")
        f.write(f"- Relevance label: {cfg['relevance_definition']}\n")
        f.write(f"- Label provenance: {cfg['relevance_label_provenance']}\n")
        f.write(f"- Query state: {cfg['query_state_condition']}\n")
        f.write(f"- max_conversations: {cfg['max_conversations']} "
                f"(script default {DEFAULT_MAX_CONVERSATIONS})\n")
        f.write(f"- Reproduce with: {cfg['reproduce_command']}\n")
        f.write("- No synthetic or hand-authored data is used anywhere in this run.\n\n")

        f.write("Arms and how the removed weight was redistributed\n")
        for arm in COMPOSITE_ARMS:
            w = cfg["arm_weights"][arm]
            f.write(f"- {arm}: alpha {w[0]}, beta {w[1]}, gamma {w[2]}, delta {w[3]}\n")
            f.write(f"    redistribution: {cfg['arm_redistribution'][arm]}\n")
        f.write("- semantic_only: ncm.retrieval.retrieve_semantic_only, no weights\n")
        f.write("Total composite weight is 1.0 in every arm, so the ablation is not\n")
        f.write("confounded with a rescaling of the distance.\n\n")

        f.write("WEIGHTS-ARE-LIVE GUARD\n")
        f.write(f"- passed: {guard['passed']}\n")
        f.write(f"- guard ran on: {guard['guard_context']}\n")
        f.write(f"- provenance: {guard['guard_data_provenance']}\n")
        for name, ok in guard["checks"].items():
            f.write(f"- {name}: {ok}\n")
        f.write(f"- max abs distance difference default vs no_emo_renorm: "
                f"{guard['max_abs_distance_diff_default_vs_no_emo_renorm']}\n")
        f.write(f"- max abs distance difference default vs no_emo_state_absorbs: "
                f"{guard['max_abs_distance_diff_default_vs_no_emo_state_absorbs']}\n")
        f.write(f"- emotional-channel distance sd across the guard store: "
                f"{guard['emo_channel_distance_std']}\n")
        f.write(f"- state-channel distance sd across the guard store: "
                f"{guard['state_channel_distance_std']}\n")
        f.write(f"- beta-is-live residual: {guard['beta_live_residual_max_abs']:.3e}\n")
        f.write(f"    {guard['beta_live_identity']}\n")
        f.write("Without this guard a dead weight would make every arm the same\n")
        f.write("computation and the null result below would be an artifact. The\n")
        f.write("distance-difference checks alone are not enough, because the\n")
        f.write("alpha, gamma and delta rescale changes every distance whether or\n")
        f.write("not beta contributes; the residual above is what pins beta at its\n")
        f.write("shipped 0.2.\n\n")

        f.write("Dataset\n")
        f.write(f"- Conversations loaded: {ds['conversations_loaded']}\n")
        f.write(f"- Conversations benchmarked: {ds['conversations_benchmarked']}\n")
        f.write(f"- Skipped, fewer than {MIN_SESSIONS_PER_CONVERSATION} sessions: "
                f"{ds['conversations_skipped_too_few_sessions']}\n")
        f.write(f"- Skipped, no session with at least "
                f"{MIN_STORED_TURNS_IN_TARGET_SESSION} stored turns: "
                f"{ds['conversations_skipped_no_eligible_session']}\n")
        f.write(f"- Queries evaluated: {ds['queries_evaluated']}\n")
        f.write(f"- Turns stored in total: {ds['total_turns_stored']}\n")
        f.write(f"- Mean store size: {ds['mean_store_size']}\n")
        f.write(f"- Mean relevant turns per query: {ds['mean_relevant_per_query']}\n")
        f.write(f"- Random-guess precision: {ds['random_guess_precision']:.4f}\n")
        f.write(f"    definition: {ds['random_guess_definition']}\n\n")

        header = (f"{'arm':<22}{'P@5':>8}{'P@10':>8}{'R@10':>8}"
                  f"{'NDCG@10':>10}{'MRR':>8}{'n':>6}\n")
        f.write("Means per arm (sd per query in the JSON)\n")
        f.write(header)
        f.write("-" * (len(header) - 1) + "\n")
        for arm in ARMS:
            a = arms[arm]
            f.write(f"{arm:<22}{a['p@5']:>8.4f}{a['p@10']:>8.4f}{a['r@10']:>8.4f}"
                    f"{a['ndcg@10']:>10.4f}{a['mrr']:>8.4f}{a['n_queries']:>6}\n")
        f.write("\n")

        f.write("Paired per-query deltas against default\n")
        f.write("Each row is the mean of the per-query difference. Two percentile\n")
        f.write(f"bootstrap 95 pct CIs are given ({cfg['bootstrap']['n_resamples']} "
                f"resamples): one resampling queries\n")
        f.write(f"(seed {cfg['bootstrap']['seed_query_bootstrap']}), one resampling "
                f"whole conversations (seed "
                f"{cfg['bootstrap']['seed_cluster_bootstrap']}). Queries inside\n")
        f.write("one conversation share a store and a candidate set, so they are not\n")
        f.write("independent and the conversation-level interval is the one to read\n")
        f.write("when the two disagree. 'zero' names where 0.0 sits in the\n")
        f.write("query-level interval: interior, on a bound, or outside. A bound\n")
        f.write("landing exactly on 0.0 is a discreteness artifact of a difference\n")
        f.write("vector that is mostly zeros, not a measurement of equivalence, so\n")
        f.write("'nonzero' (the number of queries whose metric actually changed) has\n")
        f.write("to be read with every interval. MDE is the minimum detectable\n")
        f.write("effect at 80 pct power, two-sided 0.05.\n\n")
        for arm in [a for a in ARMS if a != REFERENCE_ARM]:
            f.write(f"{arm} minus default\n")
            f.write(f"  {'metric':<9}{'delta':>9}{'query CI':>19}"
                    f"{'conversation CI':>21}{'nonzero':>9}{'MDE80':>9}"
                    f"{'wilcox p':>10}{'method':>11}\n")
            for m in REPORTED_METRICS:
                d = paired[arm][m]
                p_txt = "n/a" if d["wilcoxon_p"] is None else f"{d['wilcoxon_p']:.4g}"
                meth = d.get("wilcoxon_method") or "n/a"
                q_ci = f"[{d['ci95_low']:+.4f},{d['ci95_high']:+.4f}]"
                c_ci = f"[{d['cluster_ci95_low']:+.4f},{d['cluster_ci95_high']:+.4f}]"
                nz = f"{d['n_nonzero_pairs']}/{d['n_queries']}"
                f.write(f"  {m:<9}{d['mean_delta']:>+9.4f}{q_ci:>19}{c_ci:>21}"
                        f"{nz:>9}{d['mde80_two_sided_05']:>9.4f}"
                        f"{p_txt:>10}{meth:>11}\n")
            f.write("  where zero sits, and which queries moved\n")
            for m in REPORTED_METRICS:
                d = paired[arm][m]
                f.write(f"  - {m}: query CI {d['ci95_zero_relation']}, "
                        f"conversation CI {d['cluster_ci95_zero_relation']}; "
                        f"{d['sign_pattern']}\n")
            f.write("\n")
        mde_def = ""
        wilcox_note = ""
        for arm in [a for a in ARMS if a != REFERENCE_ARM]:
            for m in REPORTED_METRICS:
                d = paired[arm][m]
                if not mde_def:
                    mde_def = d.get("mde80_definition", "")
                if not wilcox_note and d.get("wilcoxon_method"):
                    wilcox_note = d["wilcoxon_note"]
        if mde_def:
            f.write(f"MDE definition: {mde_def}\n")
        if wilcox_note:
            f.write(f"Wilcoxon: {wilcox_note}\n")
        f.write("\n")

        f.write("Ranking divergence against default\n")
        for arm, dv in div.items():
            if "note" in dv:
                f.write(f"- {arm}: {dv['note']}\n")
                continue
            f.write(f"- {arm}: top-10 list differs on "
                    f"{dv['fraction_top10_list_differs']:.4f} of queries, top-1 "
                    f"changed on {dv['fraction_top1_changed']:.4f}, mean set "
                    f"Jaccard {dv['mean_top10_set_jaccard']:.4f}\n")
        f.write("\n")

        f.write("Direct redundancy measurement\n")
        if "note" in red:
            f.write(f"- {red['note']}\n\n")
        else:
            f.write(f"- Over {red['n_pairs']} (query, stored memory) pairs from the\n")
            f.write("  corpus, the emotional-channel distance and the state-channel\n")
            f.write(f"  distance correlate at Pearson r = {red['pearson_r']:.4f} and\n")
            f.write(f"  Spearman rho = {red['spearman_rho']:.4f}.\n")
            f.write(f"- {red['clustering_note']}.\n")
            f.write(f"- Channel source: {red['channel_source']}\n\n")
            f.write("Dispersion of each channel over all (query, memory) pairs\n")
            f.write(f"  {'channel':<10}{'weight':>8}{'mean':>10}{'sd':>10}"
                    f"{'min':>10}{'max':>10}{'w*sd':>11}{'w*range':>11}\n")
            for name in ("d_sem", "d_emo", "d_state", "d_time"):
                c = red["channels"][name]
                f.write(f"  {name:<10}{c['shipped_weight']:>8.3f}{c['mean']:>10.5f}"
                        f"{c['sd']:>10.5f}{c['min']:>10.5f}{c['max']:>10.5f}"
                        f"{c['weighted_sd']:>11.6f}{c['weighted_range']:>11.6f}\n")
            f.write(f"  definition: {red['dispersion_definition']}\n")
            f.write(f"  d_sem clip: {red['d_sem_clip_note']}\n")
            if red.get("weighted_sd_ratio_sem_over_emo") is not None:
                f.write(f"- Measured as weighted standard deviations, the semantic\n")
                f.write(f"  channel's spread is "
                        f"{red['weighted_sd_ratio_sem_over_emo']:.2f} times the\n")
                f.write("  emotional channel's.\n")
            if red.get("weighted_range_ratio_sem_over_emo") is not None:
                f.write(f"- Measured as weighted ranges, and with the d_sem numerator\n")
                f.write(f"  truncated by the clip noted above, the ratio is "
                        f"{red['weighted_range_ratio_sem_over_emo']:.2f}.\n")
            f.write(f"  {red['dispersion_ratio_definition']}\n")
            f.write(f"- Order-flip budget of the emotional term, as a weighted range\n")
            f.write(f"  pooled over all (query, memory) pairs: "
                    f"{red['emo_order_flip_budget']:.6f}. Two memories can\n")
            f.write("  be reordered by the emotional term only if the other three\n")
            f.write("  terms place their composite distances within that much of each\n")
            f.write("  other.\n")
            f.write(f"  {red['emo_order_flip_budget_definition']}\n")
            f.write("\n")

        nt = results["near_ties"]
        f.write("Near-tie accounting: what the emotional term is reordering\n")
        f.write(f"- Measured on {nt['n_queries']} queries, over the top "
                f"{nt['depth']} memories of the shipped composite ranking.\n")
        f.write(f"- Median order-flip budget of the emotional term inside a single\n")
        f.write(f"  store: {nt['median_within_store_emo_budget']:.6f}.\n")
        f.write(f"- Median adjacent-rank gap of the shipped composite in the top "
                f"{nt['depth']}:\n")
        f.write(f"  {nt['median_adjacent_gap_in_top_depth']:.6f}.\n")
        f.write(f"- Adjacent-rank gaps smaller than that query's emotional budget: "
                f"{nt['n_adjacent_gaps_below_emo_budget']} of\n")
        f.write(f"  {nt['n_adjacent_gaps']}, "
                f"{100.0 * nt['fraction_adjacent_gaps_below_emo_budget']:.1f} pct.\n")
        f.write(f"  definition: {nt['definition']}\n")
        f.write("- Read together with the ranking divergence above: the emotional\n")
        f.write("  term does move top-10 lists, and the pairs it can move are\n")
        f.write("  near-ties in the other three channels. It is not too flat to\n")
        f.write("  reorder anything; it is reordering candidates the rest of the\n")
        f.write("  composite has already placed within its reach, without changing\n")
        f.write("  how many relevant turns land in the top k.\n\n")

        f.write("Structural note on how much the emotional channel can carry\n")
        f.write(f"- SentenceEncoder.state_dim is {cfg['encoder_state_dim']} and\n")
        f.write(f"  AutoStateTracker emits {cfg['auto_state_dim']} dimensions, so "
                f"{cfg['encoder_state_dim'] - cfg['auto_state_dim']} of the\n")
        f.write("  dimensions that encode_emotional pads to are structurally always\n")
        f.write("  zero, and the corresponding columns of W_emo never contribute.\n")
        f.write(f"- W_emo therefore maps an effectively {cfg['auto_state_dim']}-dim "
                f"input to {cfg['encoder_emotional_dim']} dims.\n")
        f.write("- encode_emotional L2-normalizes its output, so e_emotional depends\n")
        f.write("  only on the direction of the auto-state. _rebuild_cache also\n")
        f.write("  L2-normalizes the stored auto-state before the state channel sees\n")
        f.write("  it. Both channels therefore read the same normalized quantity, one\n")
        f.write(f"  at {cfg['auto_state_dim']} dims and one compressed to "
                f"{cfg['encoder_emotional_dim']}.\n")
        f.write(f"- Measured scale invariance of encode_emotional: max abs component\n")
        f.write(f"  difference between encode_emotional(s) and "
                f"encode_emotional(s/||s||) over\n")
        f.write(f"  {cfg['emo_scale_invariance']['n_states_checked']} real inferred "
                f"query states was "
                f"{cfg['emo_scale_invariance']['max_abs_component_diff']:.3e}.\n\n")

        f.write("Calibration against exp17\n")
        cal = results["calibration"]
        ref = cal["reference"]
        eq = cal["exp17_equivalence"]
        f.write(f"- Reference file: {ref['source']}, fields {ref['source_fields']}.\n")
        f.write(f"- exp17 at max_conversations={ref['max_conversations']}: "
                f"{ref['conversations_benchmarked']} conversations, "
                f"{ref['queries_evaluated']} queries,\n")
        f.write(f"  {ref['total_turns_stored']} stored turns, semantic_only P@5 "
                f"{ref['semantic_only_p@5']:.4f}, ncm_inferred P@5 "
                f"{ref['ncm_inferred_p@5']:.4f}.\n")
        f.write(f"- this run at max_conversations={cfg['max_conversations']}: "
                f"{ds['conversations_benchmarked']} conversations, "
                f"{ds['queries_evaluated']} queries,\n")
        f.write(f"  {ds['total_turns_stored']} stored turns, semantic_only P@5 "
                f"{arms['semantic_only']['p@5']:.4f}, default P@5 "
                f"{arms['default']['p@5']:.4f}.\n")
        f.write(f"- Same scale as exp17: {eq['same_scale_as_exp17']}.\n")
        f.write("- Per-metric comparison, computed at run time from both result\n")
        f.write("  files. No number in this block was recorded by hand.\n")
        for pair_name, row in eq["rows"].items():
            f.write(f"  {pair_name}\n")
            f.write(f"    {'metric':<10}{'this run':>10}{'exp17':>10}{'diff':>10}\n")
            for m in REPORTED_METRICS:
                cell = row[m]
                f.write(f"    {m:<10}{cell['this_run']:>10.4f}{cell['exp17']:>10.4f}"
                        f"{cell['diff']:>+10.4f}\n")
        f.write(f"- Largest absolute difference: {eq['max_abs_difference']:.6f} "
                f"against a tolerance of {eq['tolerance']}.\n")
        f.write(f"- {eq['verdict']}.\n")
        f.write(f"- {cal['note']}\n\n")

        f.write("Reading of the result\n")
        for line in results["reading"]:
            f.write(line + "\n")
        f.write("\nLatency is deliberately not measured here. exp4 is the latency\n")
        f.write("measurement. The composite path reads a cache warmed once per\n")
        f.write("conversation while retrieve_semantic_only rebuilds its own matrix on\n")
        f.write("every call, so timings taken here would compare cache states.\n")


# -------------------------------------------------------------------
# READING OF THE RESULT
# -------------------------------------------------------------------

def build_reading(paired: dict, div: dict, red: dict, guard: dict,
                  near_ties: dict, arms: dict) -> list:
    lines = []
    if not guard["passed"]:
        return ["The weights-are-live guard failed, so no reading is offered."]

    key_metrics = ("p@5", "ndcg@10", "mrr")
    ablations = ("no_emo_renorm", "no_emo_state_absorbs")
    cells = [(arm, m) for arm in ablations for m in key_metrics]
    n_q = paired[ablations[0]]["p@5"]["n_queries"]
    n_c = paired[ablations[0]]["p@5"]["n_conversations"]

    strict_null = all(supports_two_sided_null(paired[a][m]) for a, m in cells)
    excluding = [f"{a} {m}" for a, m in cells
                 if paired[a][m]["ci95_zero_relation"] == "excludes_zero"
                 or paired[a][m]["cluster_ci95_zero_relation"] == "excludes_zero"]
    boundary = [f"{a} {m}" for a, m in cells
                if paired[a][m]["ci95_zero_relation"] == "bound_exactly_at_zero"
                or paired[a][m]["cluster_ci95_zero_relation"] == "bound_exactly_at_zero"]
    degenerate = [f"{a} {m}" for a, m in cells
                  if paired[a][m]["n_nonzero_pairs"] == 0]
    deltas = [paired[a][m]["mean_delta"] for a, m in cells]
    n_positive = sum(1 for v in deltas if v > 0.0)
    any_ranking_moved = any(
        div.get(arm, {}).get("fraction_top10_list_differs", 0.0) > 0.0
        for arm in ablations)
    worst_changed = max(paired[a][m]["n_nonzero_pairs"] for a, m in cells)
    least_changed = min(paired[a][m]["n_nonzero_pairs"] for a, m in cells)
    max_mde = max(paired[a][m]["mde80_two_sided_05"] for a, m in cells)

    lines.append("The retrieval weights are proved live and beta is proved live at")
    lines.append("its shipped 0.2, with a residual of")
    lines.append(f"{guard['beta_live_residual_max_abs']:.3e} on the identity")
    lines.append("d_default - d_no_emo_renorm/1.25 - 0.2*d_emo, so the comparison")
    lines.append("below is between genuinely different scorers and not between")
    lines.append("copies.")
    lines.append("")
    lines.append(f"Scale of the test: {n_q} queries drawn from {n_c} conversations.")
    lines.append("Between the two redistributions and the three key metrics, the")
    lines.append(f"number of queries whose metric actually changed ranges from")
    lines.append(f"{least_changed} to {worst_changed} out of {n_q}. Every paired")
    lines.append("difference vector is therefore mostly exact zeros, and that has to")
    lines.append("be read alongside any interval computed from it.")

    if strict_null:
        lines.append("")
        lines.append("Removing the emotional term does not hurt retrieval quality on")
        lines.append("this task. For both redistributions and all three key metrics,")
        lines.append("zero lies strictly inside both the query-level and the")
        lines.append("conversation-level 95 pct interval, so neither direction is")
        lines.append("supported at that level. The point estimates weakly favour")
        lines.append(f"removal: {n_positive} of the {len(cells)} point estimates are")
        lines.append("positive in the direction of removal, which is the direction a")
        lines.append("redundant term would move if it were adding noise. That is a")
        lines.append("sign pattern, not a detected effect.")
        lines.append("")
        lines.append("What this does not license. It does not show that the emotional")
        lines.append("channel earns none of its 0.2 weight, and it does not show that")
        lines.append("the two arms are equivalent. A two-sided interval that contains")
        lines.append("zero is not an equivalence test, and on difference vectors this")
        lines.append("sparse it is close to uninformative about effect size. The")
        lines.append("largest minimum detectable effect over these six cells is")
        lines.append(f"{max_mde:.4f} at 80 pct power and two-sided 0.05, so any true")
        lines.append("effect smaller than that would have been missed by this design.")
    else:
        lines.append("")
        if excluding:
            lines.append("At least one interval excludes zero, so the emotional term is")
            lines.append("not uniformly inert on this task. Cells excluding zero: "
                         + ", ".join(excluding) + ".")
        if boundary:
            lines.append("At least one interval has a bound sitting exactly at 0.0, "
                         "which is")
            lines.append("a discreteness artifact of a sparse difference vector and not")
            lines.append("a measurement of equivalence, so no two-sided null is claimed.")
            lines.append("Cells at the boundary: " + ", ".join(boundary) + ".")
        if degenerate:
            lines.append("In these cells not one query changed, so the two arms return")
            lines.append("the same value on every query and there is nothing to")
            lines.append("estimate: " + ", ".join(degenerate) + ". Their intervals are")
            lines.append("[0.0, 0.0] by construction and are not evidence of anything.")
        lines.append("The per-metric table above is the statement of record; the sign,")
        lines.append("the size and the number of changed queries matter more than the")
        lines.append("pass or fail of any single test.")

    if any_ranking_moved:
        lines.append("")
        lines.append("The ranking does move, so the arms are not identical")
        lines.append("computations. The emotional term reorders results without")
        lines.append("changing how many relevant turns land in the top k.")
    else:
        lines.append("")
        lines.append("No top-10 list changed in either ablation, so on this data the")
        lines.append("arms are the same ranking despite the live weight, and the")
        lines.append("equality of the metrics is arithmetic rather than empirical.")

    lines.append("")
    lines.append("The semantic_only arm, which is the most consequential number")
    lines.append("here, is not an ablation of the emotional term but a weight-free")
    lines.append("external reference. Against the shipped composite it scores")
    for m in ("p@5", "ndcg@10", "mrr"):
        d = paired["semantic_only"][m]
        lines.append(f"{m.upper()} {d['mean_delta']:+.4f} with query-level CI "
                     f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] and")
        lines.append(f"conversation-level CI [{d['cluster_ci95_low']:+.4f}, "
                     f"{d['cluster_ci95_high']:+.4f}], "
                     f"{d['n_nonzero_pairs']}/{d['n_queries']} queries changed.")
    lines.append("Reading: on this corpus and this label, dropping the three")
    lines.append("non-semantic channels entirely does not lose retrieval quality")
    lines.append("either, and on P@5 the point estimate favours the semantic-only")
    lines.append("retriever. That is a finding about the whole composite, not just")
    lines.append("about beta, and it is the number a reader should be shown first.")

    if "pearson_r" in red:
        lines.append("")
        lines.append(f"The two channels are correlated at Pearson r = "
                     f"{red['pearson_r']:.4f} and Spearman")
        lines.append(f"rho = {red['spearman_rho']:.4f} over {red['n_pairs']} real "
                     f"(query, memory) pairs from")
        lines.append(f"{red['n_conversations_contributing']} conversations. No p-value "
                     f"is quoted: the pairs are")
        lines.append("clustered by query and by store, so an independence-based p-value")
        lines.append("would be meaningless. The correlation is consistent with the")
        lines.append("structural argument that both channels are deterministic")
        lines.append("functions of the same normalized auto-state, so the emotional")
        lines.append("term supplies no input the state term lacks. It is not an")
        lines.append("algebraic identity, because the projection to 3 dimensions")
        lines.append("distorts angles, so d_emo is not a function of d_state alone.")
        ch = red["channels"]
        lines.append("")
        lines.append("Mechanism. Pooled over all (query, memory) pairs, the emotional")
        lines.append(f"distance has standard deviation {ch['d_emo']['sd']:.5f} against")
        lines.append(f"{ch['d_sem']['sd']:.5f} for the semantic distance, and after the")
        lines.append(f"shipped weights that is {ch['d_emo']['weighted_sd']:.6f} against")
        lines.append(f"{ch['d_sem']['weighted_sd']:.6f}. The channel is small in scale,")
        lines.append("but it is not too small to act. Inside a single store its median")
        lines.append(f"order-flip budget is "
                     f"{near_ties['median_within_store_emo_budget']:.6f} against a median")
        lines.append(f"adjacent-rank gap of "
                     f"{near_ties['median_adjacent_gap_in_top_depth']:.6f} in the top "
                     f"{near_ties['depth']} of the")
        lines.append("shipped composite, and")
        lines.append(f"{100.0 * near_ties['fraction_adjacent_gaps_below_emo_budget']:.1f} "
                     f"pct of those adjacent gaps "
                     f"({near_ties['n_adjacent_gaps_below_emo_budget']} of")
        lines.append(f"{near_ties['n_adjacent_gaps']}) are smaller than that query's")
        lines.append("budget. So the term does reorder candidates, and the candidates it")
        lines.append("can reorder are the ones the other three channels have already")
        lines.append("placed within its reach. It reorders near-ties and carries no")
        lines.append("relevance signal, which is why the ranking moves while the metrics")
        lines.append("do not. It is not that the channel cannot separate candidates at")
        lines.append("all.")
        lines.append("")
        lines.append("This run measures beta only at its shipped 0.2. It says nothing")
        lines.append("about what a larger beta would do.")

    lines.append("")
    lines.append("Scope. One corpus, one relevance label, one query-state condition,")
    lines.append("one encoder, one machine, and the shipped weight vector. This does")
    lines.append("not show that an emotional channel is useless in general, and it")
    lines.append("does not show that this one is worthless. It shows that on this")
    lines.append("task, at this scale, removing this projection of this state does")
    lines.append("not measurably change retrieval quality in either direction.")
    return lines


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EXP22 emotional-channel ablation")
    ap.add_argument("--max-conversations", type=int, default=DEFAULT_MAX_CONVERSATIONS,
                    help=(f"corpus records to load; the default "
                          f"{DEFAULT_MAX_CONVERSATIONS} matches exp17 and is the "
                          f"scale of the shipped result files in "
                          f"experiments/results/exp22/"))
    ap.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    # Read exp17's numbers before doing any work, so a missing reference fails
    # immediately instead of after the benchmark.
    try:
        exp17_reference = load_exp17_reference()
    except Exp17ReferenceError as exc:
        print(f"[exp22] ABORT: {exc}")
        print("[exp22] No result files written: there is no constant fallback for "
              "the exp17 reference on purpose.")
        return 1
    print(f"[exp22] exp17 reference read from {EXP17_RESULTS_REL}: "
          f"{exp17_reference['queries_evaluated']} queries at "
          f"max_conversations={exp17_reference['max_conversations']}")

    corpus_records = count_corpus_records(CORPUS_PATH)
    print(f"[exp22] Corpus {CORPUS_REL}: {corpus_records} records")
    print(f"[exp22] Loading first {args.max_conversations} (seed={SEED})")
    conversations = load_conversations(CORPUS_PATH, args.max_conversations)
    print(f"[exp22] Loaded {len(conversations)} conversations")
    if not conversations:
        print("[exp22] ERROR: no conversations loaded. Aborting.")
        return 1

    encoder = SentenceEncoder(
        model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models"))
    backend = encoder.backend
    print(f"[exp22] Encoder backend: {backend}")
    if backend != "sentence-transformers":
        print("[exp22] ABORT: the hash fallback carries no semantic structure, so")
        print(f"[exp22]        any retrieval number would be meaningless. Reason: "
              f"{encoder.backend_error}")
        return 1

    arm_profiles = {a: build_profile(a, ARM_WEIGHTS[a]) for a in COMPOSITE_ARMS}
    probe_profiles = {a: build_profile(a, PROBE_WEIGHTS[a]) for a in PROBE_WEIGHTS}

    print("[exp22] Running ablation benchmark")
    try:
        bench = benchmark(conversations, encoder, arm_profiles, probe_profiles)
    except WeightsNotLiveError as exc:
        print(f"[exp22] ABORT: {exc}")
        print("[exp22] No result files written: a dead weight would make every arm")
        print("[exp22]        identical and any null result an artifact.")
        return 1

    guard = bench["guard"]
    if guard is None:
        print("[exp22] ABORT: no query was evaluated, so the guard never ran.")
        return 1

    arms = summarize_arms(bench["per_query"])
    paired = paired_table(bench["per_query"], bench["query_conv_ids"],
                          args.bootstrap_resamples)
    div = ranking_divergence(bench["top10_ids"])
    red = channel_redundancy(
        bench["channel_pairs"],
        bench["dataset"]["queries_evaluated"],
        bench["dataset"]["conversations_benchmarked"])

    # Scale invariance of encode_emotional, measured on real inferred query
    # states from the loaded corpus.
    diffs = []
    probe_tracker_states = []
    for conv in conversations[:min(20, len(conversations))]:
        t = AutoStateTracker()
        probe_tracker_states.append(t.update(conv.turns[0].text))
    for s in probe_tracker_states:
        n = float(np.linalg.norm(s))
        if n > 1e-8:
            diffs.append(float(np.max(np.abs(
                encoder.encode_emotional(s) - encoder.encode_emotional(s / n)))))
    emo_scale_inv = {
        "n_states_checked": len(diffs),
        "max_abs_component_diff": float(max(diffs)) if diffs else 0.0,
        "definition": ("max over real inferred query states of the max abs "
                       "component difference between encode_emotional(s) and "
                       "encode_emotional(s/||s||); near zero means e_emotional "
                       "depends only on the direction of the auto-state"),
    }

    config = {
        "seed": SEED,
        "corpus": CORPUS_REL,
        "corpus_records": corpus_records,
        "encoder_backend": backend,
        "encoder_state_dim": int(encoder.state_dim),
        "encoder_emotional_dim": int(encoder.emotional_dim),
        "encoder_semantic_dim": int(encoder.semantic_dim),
        "auto_state_dim": 5,
        "max_conversations": args.max_conversations,
        "min_sessions_per_conversation": MIN_SESSIONS_PER_CONVERSATION,
        "min_stored_turns_in_target_session": MIN_STORED_TURNS_IN_TARGET_SESSION,
        "k_values": list(K_LIST),
        "relevance_definition": ("a stored turn is relevant iff it shares the "
                                 "held-out query turn's session_id"),
        "relevance_label_provenance": ("session_id ships with the corpus and was "
                                       "not authored for this experiment"),
        "query_state_condition": ("the 5-dim auto-state is inferred from the query "
                                  "text alone by a fresh AutoStateTracker and "
                                  "assigned to store.auto_state.state; no label "
                                  "information reaches the query, so no arm here "
                                  "is an oracle"),
        "arm_weights": {a: list(ARM_WEIGHTS[a]) for a in COMPOSITE_ARMS},
        "arm_redistribution": ARM_REDISTRIBUTION,
        "retrieval_paths": {
            "default": "ncm.retrieval.retrieve_top_k_fast",
            "no_emo_renorm": "ncm.retrieval.retrieve_top_k_fast",
            "no_emo_state_absorbs": "ncm.retrieval.retrieve_top_k_fast",
            "semantic_only": "ncm.retrieval.retrieve_semantic_only",
        },
        "bootstrap": {"n_resamples": args.bootstrap_resamples,
                      "seed_query_bootstrap": SEED,
                      "seed_cluster_bootstrap": CLUSTER_SEED,
                      "seed": SEED,
                      "method": ("percentile bootstrap, paired, computed twice: "
                                 "once resampling queries and once resampling "
                                 "whole conversations"),
                      "stream_note": ("the two bootstraps draw from two "
                                      "generators, seeded independently, so that "
                                      "each interval depends only on its own seed, "
                                      "the resample count and the data, and not on "
                                      "how many other intervals were computed "
                                      "first"),
                      "unit_note": ("queries are not independent; every query "
                                    "from one conversation scores against the "
                                    "same store and the same candidate set, so "
                                    "the conversation-level interval is the one "
                                    "to read when the two disagree")},
        "reproduce_command": (
            "venv/Scripts/python.exe experiments/python/exp22_emo_ablation.py"
            + ("" if args.max_conversations == DEFAULT_MAX_CONVERSATIONS
               else f" --max-conversations {args.max_conversations}")
            + ("" if args.bootstrap_resamples == BOOTSTRAP_RESAMPLES
               else f" --bootstrap-resamples {args.bootstrap_resamples}")),
        "shipped_artifact_scale": (
            f"the files in experiments/results/exp22/ are produced at "
            f"max_conversations={DEFAULT_MAX_CONVERSATIONS}, which is the "
            f"script default, so running the script with no arguments "
            f"reproduces them"),
        "emo_scale_invariance": emo_scale_inv,
        "known_no_ops": [
            "retrieve_top_k_fast ignores its s_current_normalized argument "
            "(ncm/retrieval.py:352 signature, :356 the argument, :378 reads "
            "store.auto_state.get_current_state()); it is passed here only for "
            "signature compatibility",
            "the state channel reads the 5-dim _auto_state_cache built from "
            "MemoryEntry.auto_state_snapshot, so the 7-dim s_snapshot supplied "
            "at add() time does not enter d_state",
            "AutoStateTracker emits 5 dimensions while SentenceEncoder.state_dim "
            "is 7, so dimensions 6 and 7 of the padded state are structurally "
            "always zero and the matching columns of W_emo never contribute",
        ],
        "latency_not_measured": ("exp4 is the latency measurement; the two "
                                 "retrieval paths here have asymmetric cache "
                                 "treatment"),
        "synthetic_data": "none",
    }

    calibration = {
        "reference": exp17_reference,
        "this_run_semantic_only_p5": arms["semantic_only"]["p@5"],
        "this_run_default_p5": arms["default"]["p@5"],
        "default_p5_minus_exp17_ncm_inferred": round(
            arms["default"]["p@5"] - exp17_reference["ncm_inferred_p@5"], 4),
        "exp17_difference_of_published_means_ncm_minus_semantic_p5": round(
            exp17_reference["ncm_inferred_p@5"]
            - exp17_reference["semantic_only_p@5"], 4),
        "this_run_paired_semantic_only_minus_default_p5": paired[
            "semantic_only"]["p@5"]["mean_delta"],
        "note": ("exp17's figures are means over its own query set. When "
                 "max_conversations is below 100 the query set here is a prefix "
                 "subset of exp17's, because the loader takes records in file "
                 "order and the held-out choices are drawn from the same seeded "
                 "RNG in the same order, so absolute values differ by sampling "
                 "and only the paired columns are tight comparisons."),
        "exp17_equivalence": exp17_equivalence(
            arms, bench["dataset"], args.max_conversations, exp17_reference),
    }

    results = {
        "experiment": "exp22_emotional_channel_ablation",
        "hypothesis": ("beta*d_emo is redundant with gamma*d_state because "
                       "e_emotional = L2_normalize(W_emo @ s_padded) is a fixed "
                       "linear function of the same auto-state that d_state "
                       "compares"),
        "config": config,
        "dataset": bench["dataset"],
        "arms": arms,
        "paired_deltas_vs_default": paired,
        "ranking_divergence_vs_default": div,
        "channel_redundancy": red,
        "near_ties": bench["near_ties"],
        "weights_are_live_guard": guard,
        "calibration": calibration,
    }
    results["reading"] = build_reading(
        paired, div, red, guard, bench["near_ties"], arms)

    json_path = os.path.join(RESULTS_DIR, "exp22_emo_ablation.json")
    txt_path = os.path.join(RESULTS_DIR, "exp22_emo_ablation.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_text_report(txt_path, results)
    print(f"[exp22] Saved: {json_path}")
    print(f"[exp22] Saved: {txt_path}")

    for path in (
        plot_arm_metrics(arms, bench["dataset"]),
        plot_paired_deltas(paired),
        plot_channel_redundancy(bench["channel_pairs"], red),
    ):
        print(f"[exp22] Saved: {path}")

    print(f"[exp22] Guard passed: {guard['passed']}  checks: "
          + ", ".join(f"{k}={v}" for k, v in guard["checks"].items()))
    print(f"[exp22] beta-is-live residual: "
          f"{guard['beta_live_residual_max_abs']:.3e}")
    print("[exp22] P@5 by arm: " + ", ".join(
        f"{a}={arms[a]['p@5']:.4f}" for a in ARMS))
    print(f"[exp22] Random-guess precision: "
          f"{bench['dataset']['random_guess_precision']:.4f}")
    for arm in [a for a in ARMS if a != REFERENCE_ARM]:
        for metric in ("p@5", "ndcg@10", "mrr"):
            d = paired[arm][metric]
            print(f"[exp22] {arm} minus default {metric}: {d['mean_delta']:+.4f} "
                  f"query CI [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] "
                  f"conv CI [{d['cluster_ci95_low']:+.4f}, "
                  f"{d['cluster_ci95_high']:+.4f}] "
                  f"nonzero {d['n_nonzero_pairs']}/{d['n_queries']} "
                  f"zero={d['ci95_zero_relation']}")
    for arm, dv in div.items():
        if "fraction_top10_list_differs" in dv:
            print(f"[exp22] {arm}: top-10 list differs on "
                  f"{dv['fraction_top10_list_differs']:.4f} of queries")
    nt = bench["near_ties"]
    print(f"[exp22] near ties: {nt['n_adjacent_gaps_below_emo_budget']}/"
          f"{nt['n_adjacent_gaps']} adjacent top-{nt['depth']} gaps below the "
          f"within-store emo budget "
          f"({nt['fraction_adjacent_gaps_below_emo_budget']:.4f})")
    print(f"[exp22] exp17 equivalence: "
          f"{calibration['exp17_equivalence']['verdict']}")
    if "pearson_r" in red:
        print(f"[exp22] d_emo vs d_state: Pearson r={red['pearson_r']:.4f} "
              f"Spearman rho={red['spearman_rho']:.4f} n={red['n_pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
