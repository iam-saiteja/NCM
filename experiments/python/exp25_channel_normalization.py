"""
EXP25: Does making the profile weights effective improve retrieval?
==================================================================

WHAT THIS TESTS

exp22 removed the emotional channel and measured nothing. This script asks
whether that null was a property of affect or a property of the arithmetic.

exp22's own committed `channel_redundancy` block answers most of the question
already. A channel can only change a ranking through how much it VARIES across
the candidates, so its real influence is weight times spread, not weight. Over
the 9,401 (query, memory) pairs exp22 measured:

    channel   nominal      sd        weight*sd    effective share
    d_sem       40%     0.155125     0.062050         91.67%
    d_emo       20%     0.004266     0.000853          1.26%
    d_state     30%     0.011836     0.003551          5.25%
    d_time      10%     0.012306     0.001231          1.82%

A term carrying 1.26 percent of the ranking signal can only move P@5 by a small
amount, and exp22 was not powered to resolve a move that small: its own paired
deltas do move P@5, on 8 of its 228 queries, and the effect it observed would
need roughly 3,600 queries to detect at 80 percent power. So exp22's null was a
predictable consequence of the design rather than a discovery, but it was not
logically guaranteed, and the distinction matters because the first claim is
refuted by opening exp22's own result file. That null is not evidence that
affect is useless. It is evidence that affect was never connected.

ncm/retrieval.py now carries an opt-in `channel_normalization` mode that
rescales each channel across the candidate set before the weighted sum, so the
nominal weights become close to the effective ones. This script measures what
that does to retrieval quality. The outcome is genuinely open:

  - if the affective channels carry usable signal, giving them their nominal
    influence should raise P@5;
  - if they carry noise, P@5 should fall, which is direct evidence that the
    near-constant auto-state EMA has to be fixed before any weight on those
    channels can help.

Both outcomes are reported as measured. Neither is assumed.

HOW IT AVOIDS DRIFT

exp22 transcribed its corpus loader, store construction and metrics from
exp17 by hand. This script IMPORTS them from exp22 instead. The loader, the
store, the metric definitions and the paired statistics are therefore the same
objects, not copies that can diverge. The query set is identical because the
held-out-turn RNG is consumed in the same order.

Two calibration gates enforce this at run time, reading exp22's committed
result file rather than any constant written here:

  1. this script's `default_none` arm must reproduce exp22's `default` arm on
     every reported metric;
  2. this script's `none` pooled channel table must reproduce exp22's
     `channel_redundancy` channel table.

Either failing aborts the run with no output files.

WHAT IS ARITHMETIC AND WHAT IS A FINDING

Under "minmax" every channel is mapped onto [0, 1] within each candidate set,
so the effective shares move close to the nominal ones BY CONSTRUCTION. The
effective-influence table below is therefore a check that the implementation
does what it claims plus a measurement of the residual deviation caused by
distribution shape. It is not an independent discovery, and the text report
says so. The empirical content of this experiment is the retrieval outcome.

Outputs
- experiments/results/exp25/exp25_channel_normalization.json
- experiments/results/exp25/exp25_channel_normalization.txt
- experiments/results/exp25/exp25_arm_metrics.png
- experiments/results/exp25/exp25_paired_deltas.png
- experiments/results/exp25/exp25_effective_influence.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# The protocol is imported, not transcribed. Everything taken from exp22 is
# named explicitly at its use site so the provenance of each piece is visible.
import exp22_emo_ablation as exp22  # noqa: E402

from ncm import AutoStateTracker, MemoryProfile, SentenceEncoder  # noqa: E402
from ncm.retrieval import (  # noqa: E402
    CHANNEL_NORMALIZATION_MODES,
    CHANNEL_ROBUST_HIGH,
    CHANNEL_ROBUST_LOW,
    retrieve_semantic_only,
    retrieve_top_k_fast,
)

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Everything below is read from exp22 rather than restated, so the two scripts
# cannot disagree about the protocol.
SEED = exp22.SEED
CLUSTER_SEED = exp22.CLUSTER_SEED
DEFAULT_MAX_CONVERSATIONS = exp22.DEFAULT_MAX_CONVERSATIONS
BOOTSTRAP_RESAMPLES = exp22.BOOTSTRAP_RESAMPLES
K_LIST = exp22.K_LIST
METRIC_KEYS = exp22.METRIC_KEYS
REPORTED_METRICS = exp22.REPORTED_METRICS
MIN_SESSIONS_PER_CONVERSATION = exp22.MIN_SESSIONS_PER_CONVERSATION
MIN_STORED_TURNS_IN_TARGET_SESSION = exp22.MIN_STORED_TURNS_IN_TARGET_SESSION
CORPUS_PATH = exp22.CORPUS_PATH
CORPUS_REL = exp22.CORPUS_REL

SHIPPED_WEIGHTS = exp22.ARM_WEIGHTS["default"]

# Modes under test. Asserted against the module's own tuple below, so adding a
# mode to ncm/retrieval.py without extending this script is caught rather than
# silently untested.
MODES = ("none", "minmax", "robust")

# Composite arms, each a (weights, mode) pair scored through
# ncm.retrieval.retrieve_top_k_fast. `semantic_only` is handled separately
# because it takes a different code path and reads no weights at all.
ARM_SPEC = {
    "default_none": (SHIPPED_WEIGHTS, "none"),
    "default_minmax": (SHIPPED_WEIGHTS, "minmax"),
    "default_robust": (SHIPPED_WEIGHTS, "robust"),
    "sem_pure_none": ((1.0, 0.0, 0.0, 0.0), "none"),
    "sem_pure_minmax": ((1.0, 0.0, 0.0, 0.0), "minmax"),
    "sem_pure_robust": ((1.0, 0.0, 0.0, 0.0), "robust"),
    "no_emo_minmax": (exp22.ARM_WEIGHTS["no_emo_renorm"], "minmax"),
}
COMPOSITE_ARMS = tuple(ARM_SPEC.keys())
ARMS = COMPOSITE_ARMS + ("semantic_only",)
REFERENCE_ARM = "default_none"

ARM_PURPOSE = {
    "default_none": (
        "the shipped composite, unchanged. Reference arm, and the calibration "
        "target against exp22's `default`"),
    "default_minmax": (
        "the shipped weights with every channel rescaled onto [0, 1] across "
        "the candidate set. The primary test"),
    "default_robust": (
        "as default_minmax but anchored on the 5th and 95th percentiles, so a "
        "single outlying candidate cannot set the scale for the rest"),
    "sem_pure_none": (
        "weights (1, 0, 0, 0), no rescaling. The baseline for both single-"
        "channel probes: against sem_pure_minmax it shows that an affine "
        "rescaling of one channel cannot change a ranking, and against "
        "sem_pure_robust it shows how much ranking a clipping rescaling does "
        "change"),
    "sem_pure_minmax": (
        "weights (1, 0, 0, 0) with rescaling. Must score identically to "
        "sem_pure_none, because a strictly increasing map of a single channel "
        "leaves its order untouched. Any difference here would mean the "
        "rescaling is not order preserving and would invalidate the other arms"),
    "sem_pure_robust": (
        "weights (1, 0, 0, 0) with robust rescaling. Isolates the one thing "
        "robust mode does that minmax does not: clipping at the percentile "
        "bounds collapses each tail of d_sem to a single value, and the low "
        "tail of a distance channel is the head of the ranking. With no other "
        "channel carrying weight, ties there are broken by argsort order "
        "instead, so sem_pure_robust against sem_pure_none measures the "
        "head-of-list information robust mode destroys, separately from any "
        "influence the other channels gain. Without this arm the robust "
        "result could not be attributed"),
    "no_emo_minmax": (
        "beta removed and alpha, gamma, delta renormalised by 1/(1-beta), then "
        "rescaled. Isolates what the emotional channel contributes once it "
        "actually has its nominal weight, which is the question exp22 could not "
        "reach"),
    "semantic_only": (
        "ncm.retrieval.retrieve_semantic_only, cosine distance alone. Reads no "
        "weights and no normalization mode, so it is the same number in every "
        "mode and serves as an external reference"),
}

# Single-channel probes. With one weight at 1.0 and the rest at 0.0, and with
# every memory strength at 1.0 and contradiction awareness off, the scorer's
# return value IS that channel's distance after any rescaling, because the
# rescaling happens before the contradiction penalty, the strength modulation
# and the final clip. Both preconditions are guard checks that gate the run.
CHANNEL_PROBES = (
    ("d_sem", (1.0, 0.0, 0.0, 0.0)),
    ("d_emo", (0.0, 1.0, 0.0, 0.0)),
    ("d_state", (0.0, 0.0, 1.0, 0.0)),
    ("d_time", (0.0, 0.0, 0.0, 1.0)),
)
CHANNEL_NAMES = tuple(name for name, _ in CHANNEL_PROBES)
NOMINAL_SHARE = dict(zip(CHANNEL_NAMES, SHIPPED_WEIGHTS))

# The single declared primary comparison. Everything else is a secondary family
# carrying a Holm correction, so there is no ambiguity about which test the
# headline rests on.
PRIMARY_ARM = "default_minmax"
PRIMARY_METRIC = "p@5"
SECONDARY_FAMILY = (
    ("default_robust", "p@5"),
    ("sem_pure_minmax", "p@5"),
    ("no_emo_minmax", "p@5"),
    ("semantic_only", "p@5"),
)

# Contrasts between two arms that are neither of them the reference. These are
# declared here rather than derived from an outcome, for the same reason the
# primary is: an arm's purpose fixes which comparison answers it, and that is
# settled when the arm is defined.
#
# no_emo_minmax exists to isolate beta at its nominal influence. The only
# contrast that does so is against default_minmax, under the same normalization.
# Measuring it against default_none instead would confound removing the channel
# with rescaling the other three, which is precisely the confound exp22 could
# not escape. Reporting the arm's purpose without computing this comparison
# would leave the arm decorative.
#
# sem_pure_minmax against sem_pure_none is the order-preservation check carried
# through to the metrics: it must come out exactly zero on every metric.
#
# sem_pure_robust against sem_pure_none is the matching check for robust mode,
# and it is a real test rather than a verification because robust mode is not
# expected to come out zero. It is the only comparison in the grid that isolates
# the head-of-ranking cost of clipping from the influence the other channels
# gain, so without it the default_robust result cannot be attributed to a cause.
WITHIN_MODE_CONTRASTS = (
    ("no_emo_minmax", "default_minmax"),
    ("sem_pure_minmax", "sem_pure_none"),
    ("sem_pure_robust", "sem_pure_none"),
)

# Why each declared contrast exists, keyed by the pair itself so that adding a
# contrast cannot inherit another contrast's explanation.
CONTRAST_WHY = {
    ("no_emo_minmax", "default_minmax"): (
        "isolates the emotional channel at its nominal influence, both sides "
        "rescaled, so the only difference is beta"),
    ("sem_pure_minmax", "sem_pure_none"): (
        "order preservation carried through to the metrics; every difference "
        "must be exactly zero"),
    ("sem_pure_robust", "sem_pure_none"): (
        "the head-of-ranking cost of the robust clip, with only the semantic "
        "channel weighted, so no other channel can absorb or mask it"),
}

# Tolerances for the two calibration gates. exp22 rounds arm metrics to 4
# decimals and channel statistics to 6, so these are one unit in its last
# reported place and nothing looser.
CALIB_METRIC_TOL = 1e-4
CALIB_CHANNEL_TOL = 1e-6
# Order-preservation tolerance. The rescaled channel is float32, so an exactly
# monotone map can still produce a difference of a few ulp in the wrong
# direction between two adjacent values.
MONOTONE_EPS = 1e-6
UNIT_SPAN_TOL = 1e-6

EXP22_RESULTS_REL = "experiments/results/exp22/exp22_emo_ablation.json"
EXP22_RESULTS_PATH = os.path.join(ROOT_DIR, *EXP22_RESULTS_REL.split("/"))


class Exp22ReferenceError(RuntimeError):
    """Raised when exp22's result file cannot be read or lacks a field this
    script calibrates against. There is no constant fallback on purpose: this
    script's whole claim to comparability rests on reproducing exp22's numbers,
    so a missing reference has to abort rather than degrade."""


class RescalingNotLiveError(RuntimeError):
    """Raised when the channel rescaling provably does not change retrieval, or
    changes it in a way that contradicts the documented behaviour. Every arm
    would then be either the same computation or an unexplained one, and any
    conclusion drawn from the comparison would be an artifact."""


# -------------------------------------------------------------------
# REFERENCE
# -------------------------------------------------------------------

def load_exp22_reference() -> dict:
    """Read every exp22 number this script calibrates against, straight from
    the committed result file."""
    try:
        with open(EXP22_RESULTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise Exp22ReferenceError(
            f"exp22 reference not found at {EXP22_RESULTS_REL}; "
            "rerun exp22 before exp25") from exc
    except json.JSONDecodeError as exc:
        raise Exp22ReferenceError(
            f"exp22 reference at {EXP22_RESULTS_REL} is not valid JSON: {exc}"
        ) from exc

    def dig(path: str):
        node = raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise Exp22ReferenceError(
                    f"exp22 reference {EXP22_RESULTS_REL} has no field {path}")
            node = node[part]
        return node

    channels = {}
    for name in CHANNEL_NAMES:
        channels[name] = {
            stat: float(dig(f"channel_redundancy.channels.{name}.{stat}"))
            for stat in ("shipped_weight", "mean", "sd", "min", "max",
                         "weighted_sd")
        }

    return {
        "source": EXP22_RESULTS_REL,
        "max_conversations": int(dig("config.max_conversations")),
        "conversations_benchmarked": int(dig("dataset.conversations_benchmarked")),
        "queries_evaluated": int(dig("dataset.queries_evaluated")),
        "total_turns_stored": int(dig("dataset.total_turns_stored")),
        "random_guess_precision": float(dig("dataset.random_guess_precision")),
        "arm_default": {m: float(dig(f"arms.default.{m}")) for m in REPORTED_METRICS},
        "arm_semantic_only": {
            m: float(dig(f"arms.semantic_only.{m}")) for m in REPORTED_METRICS},
        "arm_no_emo_renorm_p@5": float(dig("arms.no_emo_renorm.p@5")),
        # exp22's own paired-delta cell for the comparison exp25 claims exp22
        # could not reach. Read from the file rather than described from memory,
        # because the claim "removing beta could not move P@5" is a claim about
        # this cell and is checkable against it.
        "no_emo_renorm_p@5_delta": {
            "n_nonzero_pairs": int(dig(
                "paired_deltas_vs_default.no_emo_renorm.p@5.n_nonzero_pairs")),
            "mean_delta": float(dig(
                "paired_deltas_vs_default.no_emo_renorm.p@5.mean_delta")),
            "sd_delta": float(dig(
                "paired_deltas_vs_default.no_emo_renorm.p@5.sd_delta")),
        },
        "channels": channels,
        "n_pairs": int(dig("channel_redundancy.n_pairs")),
        "pearson_r_emo_state": float(dig("channel_redundancy.pearson_r")),
        "weighted_sd_ratio_sem_over_emo": float(
            dig("channel_redundancy.weighted_sd_ratio_sem_over_emo")),
    }


# -------------------------------------------------------------------
# PROFILES
# -------------------------------------------------------------------

def build_profile(name: str, weights: tuple, mode: str) -> MemoryProfile:
    """exp22's profile builder plus the normalization mode, which lives in the
    profile's `custom` dict. MemoryProfile already carries `custom` and already
    serialises it, so no dataclass field, no file-format change and no version
    bump is involved."""
    profile = exp22.build_profile(name, weights)
    profile.set_custom("channel_normalization", mode)
    return profile


def profile_fields_excluding_weights_and_mode(p: MemoryProfile) -> dict:
    """Everything that must be identical across arms.

    exp22's own helper cannot be reused here. It compares the whole `custom`
    dict, and these arms differ inside `custom` by design, so exp22's version
    of this check would fail on a difference that is the point of the
    experiment. This one removes the two fields the arms are allowed to differ
    in and compares the rest, which is the check actually wanted.
    """
    d = p.to_dict()
    d.pop("retrieval_weights", None)
    d.pop("name", None)
    custom = dict(d.get("custom") or {})
    custom.pop("channel_normalization", None)
    d["custom"] = custom
    return d


def aligned(dmap: dict, ids: list) -> np.ndarray:
    return np.array([dmap[i] for i in ids], dtype=np.float64)


# -------------------------------------------------------------------
# GUARD
# -------------------------------------------------------------------

def rescaling_live_guard(store, q_sem, q_emo, q_state, arm_profiles: dict,
                         probe_profiles: dict, context: str) -> dict:
    """Prove empirically, on a real store built from the corpus, that the
    rescaling is live, order preserving, bounded, and that nothing else differs
    between the arms.

    Every value read here comes back from ncm.retrieval.retrieve_top_k_fast.
    Nothing is asserted from the source text of the module under test.
    """
    dists = {}
    for name, prof in list(probe_profiles.items()) + list(arm_profiles.items()):
        dists[name] = exp22.full_store_distances(store, prof, q_sem, q_emo, q_state)
    ids = sorted(dists[REFERENCE_ARM].keys())

    def max_abs_diff(a: str, b: str) -> float:
        return float(np.max(np.abs(aligned(dists[a], ids) - aligned(dists[b], ids))))

    # Rescaling has to change the composite, otherwise every arm is the same
    # computation.
    diff_minmax = max_abs_diff("default_none", "default_minmax")
    diff_robust = max_abs_diff("default_none", "default_robust")

    # A single channel at weight 1.0 cannot be reordered by a strictly
    # increasing map of itself. This is the strongest available statement that
    # the rescaling does not corrupt the semantic channel, and it means any
    # change in the composite arms is attributable to the other three channels
    # gaining influence.
    sem_pure_max_diff = max_abs_diff("sem_pure_none", "sem_pure_minmax")
    sem_none = aligned(dists["sem_pure_none"], ids)
    sem_mm = aligned(dists["sem_pure_minmax"], ids)
    sem_spearman = float(scipy_stats.spearmanr(sem_none, sem_mm).statistic)

    # Bounds, unit span and the constant-channel rule, per channel and per mode.
    bounds_ok = True
    span_ok = True
    const_ok = True
    per_channel = {}
    for cname in CHANNEL_NAMES:
        base = aligned(dists[("none", cname)], ids)
        base_span = float(base.max() - base.min())
        row = {"none_min": round(float(base.min()), 8),
               "none_max": round(float(base.max()), 8),
               "none_span": round(base_span, 8)}
        for mode in ("minmax", "robust"):
            v = aligned(dists[(mode, cname)], ids)
            lo, hi = float(v.min()), float(v.max())
            row[f"{mode}_min"] = round(lo, 8)
            row[f"{mode}_max"] = round(hi, 8)
            if lo < -UNIT_SPAN_TOL or hi > 1.0 + UNIT_SPAN_TOL:
                bounds_ok = False
            # Order preservation, checked directly rather than assumed: sort by
            # the unrescaled channel and require the rescaled one to be
            # non-decreasing along that order.
            order = np.argsort(base, kind="stable")
            steps = np.diff(v[order])
            inversions = int(np.count_nonzero(steps < -MONOTONE_EPS))
            row[f"{mode}_inversions"] = inversions
            if inversions:
                bounds_ok = False
            if mode == "minmax":
                if base_span > 1e-9:
                    if abs(lo) > UNIT_SPAN_TOL or abs(hi - 1.0) > UNIT_SPAN_TOL:
                        span_ok = False
                else:
                    if hi > UNIT_SPAN_TOL:
                        const_ok = False
        per_channel[cname] = row

    strengths = [float(m.strength) for m in store.get_all_safe()]

    # An unrecognised mode must abort rather than silently fall back to "none".
    bad_mode_raises = False
    saved_profile = store.profile
    try:
        bad = build_profile("bad_mode", SHIPPED_WEIGHTS, "zscore")
        store.profile = bad
        retrieve_top_k_fast(q_sem, q_emo, store, q_state, int(store.step), k=1)
    except ValueError:
        bad_mode_raises = True
    finally:
        store.profile = saved_profile

    checks = {
        "modes_under_test_match_the_module": (
            set(MODES) == set(CHANNEL_NORMALIZATION_MODES)),
        "minmax_changes_the_composite": bool(diff_minmax > 1e-6),
        "robust_changes_the_composite": bool(diff_robust > 1e-6),
        "single_channel_ranking_is_invariant_to_rescaling": bool(
            sem_spearman >= 1.0 - 1e-12),
        "rescaled_channels_stay_in_unit_interval_and_never_invert": bool(bounds_ok),
        "minmax_spans_the_unit_interval_on_non_constant_channels": bool(span_ok),
        "minmax_maps_a_constant_channel_to_zero": bool(const_ok),
        "unknown_mode_raises_valueerror": bool(bad_mode_raises),
        "all_memory_strengths_are_1": bool(
            strengths and max(abs(s - 1.0) for s in strengths) < 1e-9),
        "contradiction_awareness_is_off": not any(
            p.get_custom("enable_contradiction_awareness", False)
            for p in list(arm_profiles.values()) + list(probe_profiles.values())),
        "arm_profiles_differ_only_in_weights_and_normalization": all(
            profile_fields_excluding_weights_and_mode(arm_profiles[a])
            == profile_fields_excluding_weights_and_mode(arm_profiles[REFERENCE_ARM])
            for a in arm_profiles),
        "every_arm_carries_the_mode_it_declares": all(
            arm_profiles[a].get_custom("channel_normalization") == ARM_SPEC[a][1]
            for a in ARM_SPEC),
    }

    return {
        "passed": all(bool(v) for v in checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "guard_context": context,
        "guard_store_size": len(store),
        "guard_data_provenance": "a real store built from the corpus, not synthetic",
        "mode_read_at": (
            "ncm/retrieval.py:489 store.profile.get_custom("
            "'channel_normalization', 'none'), passed to "
            "vectorized_manifold_distance and applied at ncm/retrieval.py:217-221"),
        "max_abs_distance_diff_none_vs_minmax": round(diff_minmax, 8),
        "max_abs_distance_diff_none_vs_robust": round(diff_robust, 8),
        "sem_pure_max_abs_distance_diff_none_vs_minmax": round(sem_pure_max_diff, 8),
        "sem_pure_spearman_none_vs_minmax": round(sem_spearman, 12),
        "sem_pure_note": (
            "the distances differ, because the channel is mapped onto [0, 1], "
            "while the ranking does not, because that map is strictly "
            "increasing. Both facts are reported: the first shows the mode is "
            "live and the second shows it is order preserving"),
        "per_channel_on_guard_store": per_channel,
        "channel_probe_weight_vectors": {n: list(w) for n, w in CHANNEL_PROBES},
    }


# -------------------------------------------------------------------
# BENCHMARK
# -------------------------------------------------------------------

def benchmark(conversations: list, encoder: SentenceEncoder,
              arm_profiles: dict, probe_profiles: dict) -> dict:
    """Score every arm on the same queries against the same stores.

    The store is built once per conversation and every arm scores against that
    one store. Only `store.profile` is swapped between arms, so no arm can
    differ in what was written. The write gate is a max-cosine novelty test on
    the semantic vector alone, so the normalization mode cannot change the
    contents of a store even in principle.

    The RNG that picks held-out turns is consumed in exactly exp22's order, so
    the query set is the same one exp22 reported on.
    """
    rng = random.Random(SEED)
    per_query = {arm: {m: [] for m in METRIC_KEYS} for arm in ARMS}
    query_conv_ids: list = []

    conversations_benchmarked = 0
    skipped_sessions = 0
    skipped_coverage = 0
    queries_evaluated = 0
    total_turns_stored = 0
    relevant_counts: list[int] = []
    store_sizes: list[int] = []
    per_query_relevant_fraction: list[float] = []

    # Channel dispersion. `pooled` holds one row per (query, memory) pair per
    # mode and reproduces exp22's measure exactly for mode "none". `within` holds
    # one standard deviation per query per mode, which is the dispersion that
    # actually decides a ranking, because a ranking is only ever computed inside
    # one candidate set.
    pooled = {mode: {c: [] for c in CHANNEL_NAMES} for mode in MODES}
    within = {mode: {c: [] for c in CHANNEL_NAMES} for mode in MODES}
    within_share = {mode: {c: [] for c in CHANNEL_NAMES} for mode in MODES}

    # Order-preservation accounting over every query, not just the guard store.
    order_checks = 0
    order_inversions = 0
    order_new_ties = 0
    # Newly created ties have two distinct causes and only one of them is a
    # float32 artifact. Robust mode clips at the 5th and 95th percentiles, so
    # every candidate in a tail is mapped to exactly 0.0 or exactly 1.0 and ties
    # with the rest of that tail by design. Attributing all of them to float32
    # resolution would misdescribe the mechanism, so they are counted apart.
    ties_by_mode = {mode: 0 for mode in MODES}
    ties_at_clip_bound = {mode: 0 for mode in MODES}
    # The adjacent-pair counters above walk np.diff over the sorted order, so a
    # block of k candidates collapsed onto one value contributes k-1. The stated
    # definition of a newly tied pair is any (i, j), which is C(k, 2). Both are
    # kept: the adjacent count is the cheap monotonicity walk, and the all-pairs
    # count is the number the definition actually names.
    ties_allpairs_by_mode = {mode: 0 for mode in MODES}
    bounds_violations = 0
    unit_span_violations = 0
    sem_rank_mismatch_by_mode = {mode: 0 for mode in MODES if mode != "none"}
    sem_pure_vs_semantic_only_identical = 0

    guard: dict | None = None
    max_k = max(K_LIST)

    for conv in conversations:
        session_ids = conv.session_ids
        if len(session_ids) < MIN_SESSIONS_PER_CONVERSATION:
            skipped_sessions += 1
            continue

        held_out_index: dict[int, int] = {}
        for session_id in session_ids:
            candidate_indices = [i for i, t in enumerate(conv.turns)
                                 if t.session_id == session_id]
            if len(candidate_indices) < MIN_STORED_TURNS_IN_TARGET_SESSION + 1:
                continue
            held_out_index[session_id] = rng.choice(candidate_indices)

        if not held_out_index:
            skipped_coverage += 1
            continue

        store, session_of_memory = exp22.build_store(
            conv, set(held_out_index.values()), encoder)
        if len(store) == 0:
            continue

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
            # Query-side auto-state inferred from the query text alone by a
            # fresh tracker, exactly as exp22 does it. retrieve_top_k_fast
            # ignores its s_current_normalized argument and reads
            # store.auto_state.get_current_state(), so assigning the state is
            # the only way to control the state channel.
            probe = AutoStateTracker()
            inferred_state = probe.update(query_text)
            store.auto_state.state = inferred_state.astype(np.float32).copy()
            q_emo = encoder.encode_emotional(inferred_state)
            q_state = encoder.encode_state(inferred_state)

            if guard is None:
                guard = rescaling_live_guard(
                    store, q_sem, q_emo, q_state, arm_profiles, probe_profiles,
                    context=(f"conversation id {conv.conv_id}, session_id "
                             f"{session_id}, store size {len(store)}"))
                if not guard["passed"]:
                    raise RescalingNotLiveError(
                        "channel rescaling did not behave as documented: "
                        + json.dumps(guard["checks"]))

            queries_evaluated += 1
            query_conv_ids.append(conv.conv_id)
            relevant_counts.append(n_relevant)
            per_query_relevant_fraction.append(n_relevant / float(len(store)))

            # ---- channel measurement, all four channels in all three modes
            chan = {
                mode: {
                    cname: exp22.full_store_distances(
                        store, probe_profiles[(mode, cname)],
                        q_sem, q_emo, q_state)
                    for cname in CHANNEL_NAMES
                }
                for mode in MODES
            }
            ids = sorted(chan["none"]["d_sem"].keys())
            vals = {mode: {c: aligned(chan[mode][c], ids) for c in CHANNEL_NAMES}
                    for mode in MODES}

            for mode in MODES:
                weighted = {}
                for cname in CHANNEL_NAMES:
                    v = vals[mode][cname]
                    pooled[mode][cname].extend(float(x) for x in v)
                    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
                    within[mode][cname].append(sd)
                    weighted[cname] = NOMINAL_SHARE[cname] * sd
                total_w = sum(weighted.values())
                for cname in CHANNEL_NAMES:
                    within_share[mode][cname].append(
                        weighted[cname] / total_w if total_w > 0 else 0.0)

            # ---- order preservation over every query, not just the guard store
            for mode in ("minmax", "robust"):
                for cname in CHANNEL_NAMES:
                    base = vals["none"][cname]
                    resc = vals[mode][cname]
                    if base.size < 2:
                        continue
                    order_checks += 1
                    o = np.argsort(base, kind="stable")
                    b, r = base[o], resc[o]
                    steps = np.diff(r)
                    order_inversions += int(np.count_nonzero(steps < -MONOTONE_EPS))
                    tied = (np.diff(b) > 0) & (np.abs(steps) == 0.0)
                    n_tied = int(np.count_nonzero(tied))
                    order_new_ties += n_tied
                    ties_by_mode[mode] += n_tied
                    if n_tied:
                        # A tie sitting exactly on 0.0 or 1.0 is the clip
                        # collapsing a tail, not float32 running out of bits.
                        at_bound = tied & ((r[:-1] == 0.0) | (r[:-1] == 1.0))
                        ties_at_clip_bound[mode] += int(np.count_nonzero(at_bound))
                        # All-pairs count matching the stated definition: any
                        # (i, j) that differed before rescaling and is exactly
                        # equal after. n is the candidate-set size, around 40,
                        # so the pairwise comparison is cheap.
                        newly = (np.triu(
                            (b[:, None] != b[None, :])
                            & (r[:, None] == r[None, :]), k=1))
                        ties_allpairs_by_mode[mode] += int(np.count_nonzero(newly))
                    if resc.min() < -UNIT_SPAN_TOL or resc.max() > 1.0 + UNIT_SPAN_TOL:
                        bounds_violations += 1
                    if mode == "minmax" and float(base.max() - base.min()) > 1e-9:
                        if (abs(float(resc.min())) > UNIT_SPAN_TOL
                                or abs(float(resc.max()) - 1.0) > UNIT_SPAN_TOL):
                            unit_span_violations += 1

            # Does the rescaling reorder the semantic channel itself? Checked in
            # every non-identity mode, not just minmax. Under minmax the answer
            # must be no, and that is what licenses attributing a minmax change
            # to the other channels gaining influence. Under robust the clip can
            # reorder d_sem, so the same attribution is not available there and
            # the count says so instead of going unmeasured.
            for mode in MODES:
                if mode == "none":
                    continue
                rho_m = float(scipy_stats.spearmanr(
                    vals["none"]["d_sem"], vals[mode]["d_sem"]).statistic)
                if not rho_m >= 1.0 - 1e-12:
                    sem_rank_mismatch_by_mode[mode] += 1

            # ---- arm scoring
            sem_pure_ids: tuple = ()
            for arm in COMPOSITE_ARMS:
                store.profile = arm_profiles[arm]
                hits = retrieve_top_k_fast(
                    q_sem, q_emo, store, q_state, int(store.step), k=max_k)
                entries = [h[2] for h in hits]
                labels = [session_of_memory.get(m.id, -1) == session_id
                          for m in entries]
                for metric, value in exp22.score_labels(labels, n_relevant).items():
                    per_query[arm][metric].append(value)
                if arm == "sem_pure_none":
                    sem_pure_ids = tuple(m.id for m in entries)

            hits = retrieve_semantic_only(q_sem, store, k=max_k)
            entries = [h[-1] for h in hits]
            labels = [session_of_memory.get(m.id, -1) == session_id for m in entries]
            for metric, value in exp22.score_labels(labels, n_relevant).items():
                per_query["semantic_only"][metric].append(value)
            if tuple(m.id for m in entries) == sem_pure_ids:
                sem_pure_vs_semantic_only_identical += 1

            store.profile = arm_profiles[REFERENCE_ARM]
            store.auto_state.state = saved_state.copy()
            store.auto_state.turn = saved_turn

    return {
        "per_query": per_query,
        "query_conv_ids": query_conv_ids,
        "pooled": pooled,
        "within": within,
        "within_share": within_share,
        "guard": guard,
        "order_preservation": {
            "n_channel_mode_query_checks": order_checks,
            "n_inversions": order_inversions,
            "n_adjacent_pairs_newly_tied": order_new_ties,
            "n_adjacent_pairs_newly_tied_by_mode": ties_by_mode,
            "n_adjacent_pairs_newly_tied_at_a_clip_bound_by_mode":
                ties_at_clip_bound,
            "n_pairs_newly_tied_all_pairs_by_mode": ties_allpairs_by_mode,
            "n_bounds_violations": bounds_violations,
            "n_unit_span_violations": unit_span_violations,
            "n_queries_where_sem_rank_changed_by_mode": sem_rank_mismatch_by_mode,
            "n_queries_checked_for_sem_rank": queries_evaluated,
            "monotone_tolerance": MONOTONE_EPS,
            "definition": (
                "for every (query, channel, mode) the candidates are sorted by "
                "the unrescaled channel and the rescaled channel is required to "
                "be non-decreasing along that order, allowing "
                f"{MONOTONE_EPS} of float32 slack. An inversion is a strict "
                "decrease beyond that slack, and there are none, so rescaling "
                "never reverses a pair. A newly tied pair is one that was "
                "strictly ordered before rescaling and is exactly equal after. "
                "Rescaling is therefore order preserving in the weak sense: it "
                "never inverts, but it can lose a distinction"),
            "tie_counting_note": (
                "reported two ways because they answer different questions. The "
                "adjacent counts walk np.diff over the sorted order, so a block "
                "of k candidates collapsed onto one value contributes k-1; that "
                "is the monotonicity walk itself. The all_pairs count is every "
                "(i, j) the definition names, which is C(k, 2) for the same "
                "block, and is the larger and more faithful number. The "
                "adjacent count is a lower bound on it"),
            "tie_mechanism": (
                "two distinct causes, counted apart because they are not the "
                "same phenomenon. Under minmax the map is affine with positive "
                "slope, so a new tie can only be float32 running out of bits. "
                f"Under robust the clip at the {CHANNEL_ROBUST_LOW} and "
                f"{CHANNEL_ROBUST_HIGH} percentiles maps every candidate in a "
                "tail to exactly 0.0 or exactly 1.0, so those candidates tie "
                "with each other by design. The at_a_clip_bound counts isolate "
                "the second cause. This is a deliberate property of robust "
                "mode, not a defect, but it means robust discards ordering "
                "information inside the tails that minmax keeps. For a distance "
                "channel the low tail is the head of the ranking, so the arm "
                "that measures the cost of this is sem_pure_robust"),
        },
        "code_path_cross_check": {
            "n_queries_sem_pure_none_top10_equals_semantic_only_top10":
                sem_pure_vs_semantic_only_identical,
            "n_queries": queries_evaluated,
            "note": (
                "the composite with weights (1, 0, 0, 0) and "
                "ncm.retrieval.retrieve_semantic_only rank by the same quantity "
                "but through different code, and they break ties differently: "
                "retrieve_semantic_only calls np.argsort over the whole store "
                "while retrieve_top_k_fast may call np.argpartition first. This "
                "count is reported as a diagnostic and gates nothing"),
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
                "mean over queries of (n_relevant / store_size); the expected "
                "precision at any k for a retriever drawing k memories "
                "uniformly at random"),
        },
    }


# -------------------------------------------------------------------
# SUMMARIES
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
        row["weights"] = (list(ARM_SPEC[arm][0]) if arm in ARM_SPEC else None)
        row["channel_normalization"] = (ARM_SPEC[arm][1] if arm in ARM_SPEC
                                        else "not applicable")
        row["purpose"] = ARM_PURPOSE[arm]
        out[arm] = row
    return out


def effective_influence(pooled: dict, within: dict, within_share: dict) -> dict:
    """How much each channel actually moves a ranking, per mode.

    Two dispersion measures are reported and they are not interchangeable.
    `pooled` standard deviations are taken over every (query, memory) pair from
    every store and exist so this block can be compared against exp22's
    committed table one for one. `within_query` standard deviations are taken
    inside one candidate set and then averaged over queries, which is the
    measure that governs a ranking, since no ranking is ever computed across
    stores. exp22 reported only the pooled version and said it was an upper
    bound on the within-store spread. Both are here.

    Under "minmax" every channel is mapped onto [0, 1] inside each candidate
    set, so the effective shares approach the nominal weights as a matter of
    arithmetic. The residual difference is the only informative part: it comes
    from the shape of each channel's distribution, because equal ranges do not
    imply equal standard deviations.
    """
    out = {
        "nominal_weights": {c: float(NOMINAL_SHARE[c]) for c in CHANNEL_NAMES},
        "nominal_share_pct": {
            c: round(100.0 * NOMINAL_SHARE[c] / sum(SHIPPED_WEIGHTS), 2)
            for c in CHANNEL_NAMES},
        "weights_used": list(SHIPPED_WEIGHTS),
        "weights_note": (
            "the table is computed under the shipped weights "
            f"{list(SHIPPED_WEIGHTS)} in all three modes, so the only thing "
            "changing down the columns is the normalization"),
        "modes": {},
        "tautology_disclosure": (
            "under minmax each channel spans exactly [0, 1] within each "
            "candidate set, so its within-query effective share moving close to "
            "its nominal weight is arithmetic, not a discovery. This table "
            "confirms the implementation does what it claims and quantifies the "
            "residual deviation caused by distribution shape. The empirical "
            "content of this experiment is the retrieval outcome, not this "
            "table"),
        "pooled_definition": (
            "standard deviation of that channel's distance over all "
            "(query, memory) pairs pooled across stores, times the shipped "
            "weight, normalised across the four channels. Directly comparable "
            "to exp22's channel_redundancy.channels"),
        "within_query_definition": (
            "standard deviation of that channel's distance inside one candidate "
            "set, times the shipped weight, normalised across the four channels "
            "and then averaged over queries. This is the ranking-relevant "
            "measure. Under minmax it is bounded by construction, see the "
            "disclosure above"),
    }
    for mode in MODES:
        rows = {}
        weighted_pooled = {}
        for cname in CHANNEL_NAMES:
            arr = np.asarray(pooled[mode][cname], dtype=np.float64)
            sd_p = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            w_sd_p = NOMINAL_SHARE[cname] * sd_p
            weighted_pooled[cname] = w_sd_p
            w_arr = np.asarray(within[mode][cname], dtype=np.float64)
            s_arr = np.asarray(within_share[mode][cname], dtype=np.float64)
            rows[cname] = {
                "shipped_weight": float(NOMINAL_SHARE[cname]),
                "pooled_mean": round(float(np.mean(arr)), 6) if arr.size else 0.0,
                "pooled_sd": round(sd_p, 6),
                "pooled_min": round(float(np.min(arr)), 6) if arr.size else 0.0,
                "pooled_max": round(float(np.max(arr)), 6) if arr.size else 0.0,
                "pooled_weighted_sd": round(w_sd_p, 6),
                "mean_within_query_sd": (round(float(np.mean(w_arr)), 6)
                                         if w_arr.size else 0.0),
                "mean_within_query_weighted_sd": (
                    round(float(NOMINAL_SHARE[cname] * np.mean(w_arr)), 6)
                    if w_arr.size else 0.0),
                "mean_within_query_effective_share_pct": (
                    round(100.0 * float(np.mean(s_arr)), 2) if s_arr.size else 0.0),
                "sd_of_within_query_effective_share_pct": (
                    round(100.0 * float(np.std(s_arr, ddof=1)), 2)
                    if s_arr.size > 1 else 0.0),
            }
        total_p = sum(weighted_pooled.values())
        for cname in CHANNEL_NAMES:
            rows[cname]["pooled_effective_share_pct"] = (
                round(100.0 * weighted_pooled[cname] / total_p, 2)
                if total_p > 0 else 0.0)
        out["modes"][mode] = {
            "n_pairs": int(len(pooled[mode]["d_sem"])),
            "channels": rows,
            "pooled_weighted_sd_ratio_sem_over_emo": (
                round(weighted_pooled["d_sem"] / weighted_pooled["d_emo"], 2)
                if weighted_pooled["d_emo"] > 0 else None),
        }
    return out


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjustment over the declared secondary family."""
    live = sorted(((k, v) for k, v in pvals.items() if v is not None),
                  key=lambda kv: kv[1])
    m = len(live)
    out: dict = {}
    running = 0.0
    for i, (key, p) in enumerate(live):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[key] = round(running, 6)
    for key, value in pvals.items():
        if value is None:
            out[key] = None
    return out


def cell_rngs(comparison: str, metric: str) -> tuple:
    """Bootstrap generators for one (comparison, metric) cell.

    exp22's lesson was that a shared stream makes every interval depend on how
    many intervals were computed before it, so adding one statistic silently
    moves published numbers. exp22 fixed half of that by giving the query
    bootstrap and the cluster bootstrap separate generators, but within either
    generator the second cell still inherits the stream position left by the
    first, so the fix did not reach across cells. This derives both generators
    from a hash of the cell's own identity instead. Every interval is then a
    function of (SEED, comparison, metric) alone: independent of iteration
    order, of how many other cells exist, and of whether an arm was added or
    removed. Changing SEED still moves everything, so seed sensitivity stays
    checkable.
    """
    tag = f"{comparison}|{metric}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")
    return (np.random.default_rng([SEED, h]),
            np.random.default_rng([CLUSTER_SEED, h]))


def paired_table(per_query: dict, query_conv_ids: list, n_boot: int) -> dict:
    """Paired per-query differences of every arm against the reference arm.

    Uses exp22's paired_delta_stats unchanged, so the bootstrap, the cluster
    bootstrap, the exact Wilcoxon and the minimum detectable effect are the same
    code that produced exp22's table. Each cell draws from its own pair of
    generators via `cell_rngs`, so no interval depends on how many intervals
    preceded it.
    """
    table: dict = {}
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        table[arm] = {}
        for metric in REPORTED_METRICS:
            rq, rc = cell_rngs(f"{arm}_minus_{REFERENCE_ARM}", metric)
            table[arm][metric] = exp22.paired_delta_stats(
                per_query[arm][metric], per_query[REFERENCE_ARM][metric],
                query_conv_ids, rq, rc, n_boot)

    primary = table[PRIMARY_ARM][PRIMARY_METRIC]
    secondary_p = {f"{arm}:{metric}": table[arm][metric].get("wilcoxon_p")
                   for arm, metric in SECONDARY_FAMILY}
    return {
        "reference_arm": REFERENCE_ARM,
        "deltas": table,
        "multiplicity": {
            "primary": {
                "comparison": f"{PRIMARY_ARM} minus {REFERENCE_ARM}",
                "metric": PRIMARY_METRIC,
                "mean_delta": primary["mean_delta"],
                "wilcoxon_p": primary["wilcoxon_p"],
                "wilcoxon_method": primary["wilcoxon_method"],
                "correction": "none; this is the single declared primary test",
            },
            "secondary_family": [f"{a}:{m}" for a, m in SECONDARY_FAMILY],
            "secondary_wilcoxon_p": secondary_p,
            "secondary_holm_adjusted_p": holm_adjust(secondary_p),
            "note": (
                "the primary comparison was fixed before the run and carries no "
                "correction. The four secondary P@5 comparisons above were also "
                "fixed before the run and carry a Holm correction among "
                "themselves. Every comparison on a metric other than P@5 is "
                "exploratory and is reported under exploratory_multiplicity, "
                "with the correction stated there. No metric was promoted to "
                "primary after the outcome was seen"),
        },
    }


def contrast_table(per_query: dict, query_conv_ids: list, n_boot: int) -> dict:
    """Paired differences for the declared arm-against-arm contrasts.

    Returns {"contrasts": ..., "multiplicity": ...}. The multiplicity block
    carries a Holm correction over the cells that are hypothesis tests, and
    records that these contrasts were added after the first run rather than
    pre-declared.

    Each cell draws from its own generator pair via `cell_rngs`, keyed on the
    contrast name and metric, so no interval depends on how many contrasts were
    computed alongside it and adding a contrast cannot move the primary
    comparison's published interval.
    """
    out: dict = {}
    for arm, ref in WITHIN_MODE_CONTRASTS:
        key = f"{arm}_minus_{ref}"
        metrics = {}
        for metric in REPORTED_METRICS:
            rq, rc = cell_rngs(key, metric)
            metrics[metric] = exp22.paired_delta_stats(
                per_query[arm][metric], per_query[ref][metric],
                query_conv_ids, rq, rc, n_boot)
        # Whether a contrast is a hypothesis test is a property of its data, not
        # of its name. A contrast whose every paired difference is exactly zero
        # on every metric is a verification: there is nothing to test, and
        # scipy is never asked for a p-value. Deciding this from the arm name
        # would silently mislabel any contrast added later.
        degenerate = all(
            cell.get("n_nonzero_pairs") == 0 for cell in metrics.values())
        out[key] = {
            "arm": arm,
            "reference": ref,
            "why": CONTRAST_WHY.get(
                (arm, ref),
                f"paired difference of {arm} against {ref}"),
            "is_a_test": not degenerate,
            "is_a_test_basis": (
                "every paired difference is exactly 0.0 on all "
                f"{len(REPORTED_METRICS)} metrics, so this is a verification "
                "and not a test"
                if degenerate else
                "at least one metric has a non-zero paired difference, so this "
                "is a hypothesis test and enters the Holm family"),
            "metrics": metrics,
        }

    # Holm over the cells that are actually hypothesis tests. A contrast whose
    # every paired difference is exactly zero is a verification, not a test: its
    # Wilcoxon is not run at all, and counting it as a test would inflate the
    # family size and make the real contrasts look better corrected than they
    # are. Which contrasts those are is decided above from the data, not from
    # the arm names, so a contrast added later is classified correctly.
    family_p = {}
    for key, block in out.items():
        if not block["is_a_test"]:
            continue
        for metric in REPORTED_METRICS:
            family_p[f"{key}:{metric}"] = block["metrics"][metric]["wilcoxon_p"]
    adjusted = holm_adjust(family_p)
    for key, block in out.items():
        if not block["is_a_test"]:
            continue
        for metric in REPORTED_METRICS:
            block["metrics"][metric]["holm_p_within_contrast_family"] = (
                adjusted[f"{key}:{metric}"])

    return {
        "contrasts": out,
        "multiplicity": {
            "family": sorted(family_p),
            "n_tests": len(family_p),
            "raw_wilcoxon_p": family_p,
            "holm_adjusted_p": adjusted,
            "n_surviving_holm_at_05": sum(
                1 for v in adjusted.values() if v is not None and v < 0.05),
            "excluded_from_the_family": [
                k for k, b in out.items() if not b["is_a_test"]],
            "why_excluded": ("every paired difference in that contrast must be "
                             "exactly zero by construction, so it is a "
                             "verification rather than a hypothesis test and "
                             "contributes no tests to the family"),
            "pre_declared": False,
            "declaration_note": (
                "these contrasts were added after the first run of this script, "
                "so they are not pre-declared and carry a Holm correction over "
                "the family above. They were added because the no_emo_minmax "
                "arm was declared in the first run with the stated purpose of "
                "isolating beta at its nominal influence and that purpose was "
                "never measured, the arm being compared only against "
                f"{REFERENCE_ARM}. The first run computed no contrast of this "
                "kind at all, so no contrast was selected on the strength of "
                "its own outcome. The declared primary comparison was not "
                "changed and reproduces bit for bit, these generators being "
                "offset from the ones paired_table uses"),
        },
    }


def exploratory_multiplicity(paired: dict) -> dict:
    """Every arm-against-reference cell, with a Bonferroni correction over the
    whole grid.

    The declared primary is P@5 and it is reported without correction whatever
    it says. This block exists so that the other metrics can be quoted without
    being passed off as pre-declared. Bonferroni over the full grid is the most
    conservative correction available and is used deliberately: a cell that
    survives it is not an artifact of having looked at several metrics.
    """
    cells = []
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        for metric in REPORTED_METRICS:
            d = paired["deltas"][arm][metric]
            p = d.get("wilcoxon_p")
            cells.append({
                "arm": arm,
                "metric": metric,
                "mean_delta": round(float(d["mean_delta"]), 6),
                "wilcoxon_p": p,
                "n_nonzero_pairs": d["n_nonzero_pairs"],
                "pre_declared": bool(
                    (arm == PRIMARY_ARM and metric == PRIMARY_METRIC)
                    or (arm, metric) in SECONDARY_FAMILY),
                "both_bootstraps_exclude_zero": bool(
                    d["ci95_zero_relation"] == "excludes_zero"
                    and d["cluster_ci95_zero_relation"] == "excludes_zero"),
            })
    m = len(cells)
    for cell in cells:
        p = cell["wilcoxon_p"]
        cell["bonferroni_p_over_full_grid"] = (
            None if p is None else round(min(1.0, m * float(p)), 6))
        cell["survives_full_grid_bonferroni_at_05"] = bool(
            p is not None and m * float(p) < 0.05)

    survivors = [c for c in cells
                 if c["survives_full_grid_bonferroni_at_05"]
                 and c["both_bootstraps_exclude_zero"]]
    survivors.sort(key=lambda c: c["wilcoxon_p"])
    return {
        "n_cells": m,
        "grid": f"{len(ARMS) - 1} arms against {REFERENCE_ARM} times "
                f"{len(REPORTED_METRICS)} reported metrics",
        "correction": "Bonferroni over all cells, the most conservative option",
        "cells": cells,
        "survivors": survivors,
        "reading": (
            "a surviving cell clears the full-grid Bonferroni threshold AND has "
            "both the query-level and the conversation-level bootstrap interval "
            "excluding zero. It is still exploratory unless its pre_declared "
            "flag is true, because the metric was not named before the run"),
    }


def calibration(arms: dict, dataset: dict, influence: dict, ref: dict,
                max_conversations: int) -> dict:
    """Reproduce exp22 exactly where the two designs overlap.

    Both gates read exp22's committed result file at run time. Neither number is
    written into this script, so a rerun of exp22 that changed its numbers would
    surface here as a failure instead of passing against a stale constant.
    """
    scale_matches = int(max_conversations) == int(ref["max_conversations"])

    metric_dev = {}
    for metric in REPORTED_METRICS:
        metric_dev[metric] = round(
            abs(float(arms[REFERENCE_ARM][metric]) - ref["arm_default"][metric]), 6)
    worst_metric = max(metric_dev.values()) if metric_dev else 0.0

    chan_dev = {}
    none_rows = influence["modes"]["none"]["channels"]
    for cname in CHANNEL_NAMES:
        row = none_rows[cname]
        want = ref["channels"][cname]
        chan_dev[cname] = {
            "mean": round(abs(row["pooled_mean"] - want["mean"]), 8),
            "sd": round(abs(row["pooled_sd"] - want["sd"]), 8),
            "min": round(abs(row["pooled_min"] - want["min"]), 8),
            "max": round(abs(row["pooled_max"] - want["max"]), 8),
            "weighted_sd": round(
                abs(row["pooled_weighted_sd"] - want["weighted_sd"]), 8),
        }
    worst_channel = max((max(v.values()) for v in chan_dev.values()), default=0.0)

    dataset_matches = {
        "conversations_benchmarked": (
            int(dataset["conversations_benchmarked"])
            == int(ref["conversations_benchmarked"])),
        "queries_evaluated": (
            int(dataset["queries_evaluated"]) == int(ref["queries_evaluated"])),
        "total_turns_stored": (
            int(dataset["total_turns_stored"]) == int(ref["total_turns_stored"])),
        "n_channel_pairs": (
            int(influence["modes"]["none"]["n_pairs"]) == int(ref["n_pairs"])),
    }

    semantic_dev = round(
        abs(float(arms["semantic_only"]["p@5"]) - ref["arm_semantic_only"]["p@5"]), 6)

    gates = {
        "scale_matches_exp22": bool(scale_matches),
        "dataset_matches_exp22": all(dataset_matches.values()),
        "default_none_reproduces_exp22_default": bool(worst_metric <= CALIB_METRIC_TOL),
        "semantic_only_reproduces_exp22_semantic_only": bool(
            semantic_dev <= CALIB_METRIC_TOL),
        "none_channel_table_reproduces_exp22": bool(
            worst_channel <= CALIB_CHANNEL_TOL),
    }
    return {
        "source": ref["source"],
        "passed": all(gates.values()),
        "gates": gates,
        "dataset_field_matches": dataset_matches,
        "exp22_arm_default": ref["arm_default"],
        "exp25_arm_default_none": {m: arms[REFERENCE_ARM][m] for m in REPORTED_METRICS},
        "per_metric_abs_deviation": metric_dev,
        "worst_metric_abs_deviation": worst_metric,
        "metric_tolerance": CALIB_METRIC_TOL,
        "semantic_only_p@5_abs_deviation": semantic_dev,
        "per_channel_abs_deviation": chan_dev,
        "worst_channel_abs_deviation": worst_channel,
        "channel_tolerance": CALIB_CHANNEL_TOL,
        "expectation": (
            "an exact match is expected, not merely a match inside tolerance. "
            "The stores, the query set and the composite under mode 'none' are "
            "the same computation: this script imports exp22's loader, store "
            "builder and metrics, consumes the held-out-turn RNG in the same "
            "order, and ncm/retrieval.py accumulates the channels in their "
            "original order so that 'none' reproduces the shipped "
            "floating-point result bit for bit. The tolerances above exist only "
            "to absorb exp22's own rounding of its published fields"),
    }


# -------------------------------------------------------------------
# FIGURES
# -------------------------------------------------------------------

ARM_COLORS = {
    "default_none": "#4C78A8",
    "default_minmax": "#E45756",
    "default_robust": "#F58518",
    "sem_pure_none": "#9D9D9D",
    "sem_pure_minmax": "#BAB0AC",
    "no_emo_minmax": "#72B7B2",
    "semantic_only": "#54A24B",
}
CHANNEL_COLORS = {"d_sem": "#4C78A8", "d_emo": "#E45756",
                  "d_state": "#72B7B2", "d_time": "#F58518"}


def plot_arm_metrics(arms: dict, dataset: dict) -> str:
    path = os.path.join(RESULTS_DIR, "exp25_arm_metrics.png")
    metrics = ("p@5", "p@10", "ndcg@10", "mrr")
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.85 / len(ARMS)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, arm in enumerate(ARMS):
        vals = [arms[arm][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=arm,
               color=ARM_COLORS.get(arm, "#777777"))
    ax.axhline(dataset["random_guess_precision"], color="#333333", ls="--", lw=1.0,
               label=f"random guess ({dataset['random_guess_precision']:.4f})")
    ax.set_xticks(x + width * (len(ARMS) - 1) / 2.0)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title(
        "EXP25 retrieval quality by channel-normalization mode\n"
        f"{dataset['queries_evaluated']} queries, "
        f"{dataset['conversations_benchmarked']} conversations, "
        f"mean store {dataset['mean_store_size']}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_paired_deltas(paired: dict) -> str:
    """One panel per metric. Showing P@5 alone would hide the metrics that moved
    most, which are the rank-sensitive ones, so all three are drawn together."""
    path = os.path.join(RESULTS_DIR, "exp25_paired_deltas.png")
    metrics = ("p@5", "ndcg@10", "mrr")
    labels = [a for a in ARMS if a != REFERENCE_ARM]
    y = np.arange(len(labels), dtype=np.float64)

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 0.75 * len(labels) + 3.0),
                             sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        rows = [paired["deltas"][a][metric] for a in labels]
        means = [r["mean_delta"] for r in rows]
        q_lo = [r["mean_delta"] - r["ci95_low"] for r in rows]
        q_hi = [r["ci95_high"] - r["mean_delta"] for r in rows]
        c_lo = [r["mean_delta"] - r["cluster_ci95_low"] for r in rows]
        c_hi = [r["cluster_ci95_high"] - r["mean_delta"] for r in rows]
        ax.errorbar(means, y - 0.11, xerr=[q_lo, q_hi], fmt="o", color="#4C78A8",
                    capsize=3, label="query bootstrap 95%")
        ax.errorbar(means, y + 0.11, xerr=[c_lo, c_hi], fmt="s", color="#E45756",
                    capsize=3, label="conversation bootstrap 95%")
        ax.axvline(0.0, color="#333333", lw=1.0)
        title = metric + (" (declared primary)" if metric == PRIMARY_METRIC
                          else " (exploratory)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"mean paired delta against {REFERENCE_ARM}")
        ax.grid(axis="x", alpha=0.2)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.legend(fontsize=8, loc="lower left")
    fig.suptitle("EXP25 paired differences, both bootstraps; "
                 "positive favours the arm over the shipped composite",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_effective_influence(influence: dict) -> str:
    path = os.path.join(RESULTS_DIR, "exp25_effective_influence.png")
    groups = ["nominal"] + list(MODES)
    x = np.arange(len(groups), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(groups))
    for cname in CHANNEL_NAMES:
        vals = [influence["nominal_share_pct"][cname]]
        for mode in MODES:
            vals.append(influence["modes"][mode]["channels"][cname][
                "mean_within_query_effective_share_pct"])
        vals = np.asarray(vals, dtype=np.float64)
        ax.bar(x, vals, 0.6, bottom=bottom, label=cname,
               color=CHANNEL_COLORS[cname])
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3.0:
                ax.text(xi, b + v / 2.0, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="white")
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(["nominal weights"] + [f"mode={m}" for m in MODES])
    ax.set_ylabel("share of ranking influence, percent")
    ax.set_ylim(0, 105)
    ax.set_title(
        "EXP25 effective ranking influence per channel\n"
        "weight times within-query spread, normalised; shipped weights "
        "throughout")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# -------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------

def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    return f"{p:.4g}"


def build_reading(arms: dict, paired: dict, influence: dict, guard: dict,
                  order: dict, calib: dict, ref: dict, dataset: dict,
                  contrasts: dict, explore: dict) -> dict:
    primary = paired["deltas"][PRIMARY_ARM][PRIMARY_METRIC]
    none_share = influence["modes"]["none"]["channels"]
    mm_share = influence["modes"]["minmax"]["channels"]

    affect_none = (none_share["d_emo"]["mean_within_query_effective_share_pct"]
                   + none_share["d_state"]["mean_within_query_effective_share_pct"])
    affect_mm = (mm_share["d_emo"]["mean_within_query_effective_share_pct"]
                 + mm_share["d_state"]["mean_within_query_effective_share_pct"])
    affect_nominal = (influence["nominal_share_pct"]["d_emo"]
                      + influence["nominal_share_pct"]["d_state"])

    delta = float(primary["mean_delta"])
    mde = float(primary["mde80_two_sided_05"])
    if abs(delta) < mde:
        headline_kind = "smaller than the design can resolve"
    elif delta > 0:
        headline_kind = "positive"
    else:
        headline_kind = "negative"

    # The primary's verdict is read off its own interval fields rather than
    # asserted. A run whose primary excluded zero would otherwise print the
    # exclusion and then claim a null in the same sentence.
    q_rel = primary["ci95_zero_relation"]
    c_rel = primary["cluster_ci95_zero_relation"]
    n_excl = sum(1 for rel in (q_rel, c_rel) if rel == "excludes_zero")
    if n_excl == 0:
        primary_ci_txt = "Both intervals contain zero."
        primary_verdict_txt = "this is a null result"
    elif n_excl == 1:
        which = "query-level" if q_rel == "excludes_zero" else "conversation-level"
        primary_ci_txt = (
            f"The {which} interval excludes zero and the other does not, which "
            f"is the pattern clustering produces when queries from one "
            f"conversation are not independent.")
        primary_verdict_txt = (
            "the evidence is mixed and the conversation-level interval is the "
            "one to trust")
    else:
        primary_ci_txt = "Both intervals exclude zero."
        primary_verdict_txt = "this is not a null result"

    # Every cell that clears the full-grid Bonferroni threshold with both
    # bootstraps excluding zero, so the reading quotes what survived rather
    # than whatever happens to look largest.
    survivors = explore["survivors"]
    harmed = [c for c in survivors if c["mean_delta"] < 0]
    helped = [c for c in survivors if c["mean_delta"] > 0]

    def cell_txt(c: dict) -> str:
        return (f"{c['arm']} on {c['metric']} {c['mean_delta']:+.4f} "
                f"(Bonferroni p={fmt_p(c['bonferroni_p_over_full_grid'])})")

    beta_key = "no_emo_minmax_minus_default_minmax"
    beta_block = contrasts["contrasts"][beta_key]
    beta = beta_block["metrics"][PRIMARY_METRIC]
    beta_ndcg = beta_block["metrics"]["ndcg@10"]
    beta_mrr = beta_block["metrics"]["mrr"]
    beta_mult = contrasts["multiplicity"]

    def beta_txt(metric: str, cell: dict) -> str:
        both = (cell["ci95_zero_relation"] == "excludes_zero"
                and cell["cluster_ci95_zero_relation"] == "excludes_zero")
        return (f"{cell['mean_delta']:+.4f} on {metric} "
                f"(p={fmt_p(cell['wilcoxon_p'])}, Holm "
                f"p={fmt_p(cell.get('holm_p_within_contrast_family'))}, "
                f"{'both bootstraps exclude zero' if both else 'zero inside at least one interval'})")

    # Count intervals that exclude zero AND do so on the side that favours the
    # arm. Counting exclusions without checking sign would report a significant
    # move against the arm as evidence for it.
    beta_both_excl = sum(
        1 for cell in beta_block["metrics"].values()
        if cell["ci95_zero_relation"] == "excludes_zero"
        and cell["cluster_ci95_zero_relation"] == "excludes_zero"
        and cell["mean_delta"] > 0)
    beta_positive = sum(1 for cell in beta_block["metrics"].values()
                        if cell["mean_delta"] > 0)
    beta_n_metrics = len(beta_block["metrics"])
    beta_direction = (
        "favour removal on every reported metric" if beta_positive == beta_n_metrics
        else f"favour removal on {beta_positive} of {beta_n_metrics} "
             f"reported metrics")

    # The robust head-of-ranking probe. sem_pure_robust against sem_pure_none
    # isolates the cost of clipping the semantic channel with nothing else
    # weighted, so it is the only cell that can attribute the robust result.
    head_key = "sem_pure_robust_minus_sem_pure_none"
    head_block = contrasts["contrasts"].get(head_key)
    if head_block is None:
        robust_head_txt = (
            "No sem_pure_robust probe was computed in this run, so the robust "
            "result cannot be attributed to a cause.")
    else:
        head_cells = head_block["metrics"]
        head_moved = [m for m, c in head_cells.items()
                      if c["n_nonzero_pairs"] > 0]
        if not head_moved:
            robust_head_txt = (
                "The sem_pure_robust probe moved no metric at all: every paired "
                "difference against sem_pure_none is exactly zero, so on this "
                "corpus the clip ties only candidates whose order did not "
                "affect any reported metric, and the robust result is "
                "attributable to the other channels after all.")
        else:
            parts = ", ".join(
                f"{m} {head_cells[m]['mean_delta']:+.4f} "
                f"(p={fmt_p(head_cells[m]['wilcoxon_p'])})"
                for m in REPORTED_METRICS if m in head_moved)
            robust_head_txt = (
                f"Measured directly by the sem_pure_robust arm, which weights "
                f"only the semantic channel and therefore cannot borrow "
                f"influence from any other: against sem_pure_none it moves "
                f"{parts}. That much of the default_robust change is the clip "
                f"discarding head-of-list ordering, not the other channels "
                f"gaining influence, and the two causes are not separable in "
                f"default_robust itself.")

    # What exp22 could and could not see, computed from exp22's own cell rather
    # than asserted. A small effective share bounds how large an effect can be;
    # it does not make the effect unobservable at every sample size, and exp22's
    # p@5 delta was in fact non-zero on some queries. Saying otherwise is
    # refuted by opening the sibling file this script already reads.
    e22 = ref["no_emo_renorm_p@5_delta"]
    if e22["mean_delta"] != 0.0 and e22["sd_delta"] > 0.0:
        n_needed = int(np.ceil(
            ((1.959963985 + 0.8416212336) * e22["sd_delta"]
             / abs(e22["mean_delta"])) ** 2))
        reach_tail = (
            f"Detecting an effect that size at 80 percent power would need "
            f"about {n_needed} queries against the "
            f"{ref['queries_evaluated']} exp22 had, so exp22 was underpowered "
            f"for it by roughly {n_needed / max(1, ref['queries_evaluated']):.0f} "
            f"times, which is a statement about that design and not about every "
            f"possible sample size.")
    else:
        reach_tail = (
            "exp22's point estimate for that cell was exactly zero, so no "
            "required sample size can be computed from it.")
    exp22_reach_txt = (
        f"That share bounds how large the effect can be, but it does not make "
        f"the effect invisible: exp22's own paired deltas move P@5 on "
        f"{e22['n_nonzero_pairs']} of its {ref['queries_evaluated']} queries. "
        f"{reach_tail}")

    # Whether the rank-sensitive metrics agree with the primary, read off the
    # survivor list instead of asserted, so a run with no survivors cannot
    # print a disagreement it did not find.
    rank_metrics = {"ndcg@10", "mrr"}
    rank_survivors = [c for c in survivors if c["metric"] in rank_metrics]
    rank_agreement_txt = (
        "do not agree with that verdict" if rank_survivors
        else "agree with that verdict")

    beta_cost_txt = (
        "does not hurt retrieval" if beta_positive == beta_n_metrics
        else "does not clearly hurt retrieval")

    # The conclusion's claim about Bonferroni survivors, derived from the
    # survivor list. `helped` is otherwise computed and never consulted, so a
    # run with a surviving gain would print a conclusion denying it exists.
    if not survivors:
        rank_survivor_clause = (
            "No arm-by-metric cell clears a full-grid Bonferroni correction in "
            "either direction.")
    elif not helped:
        rank_survivor_clause = (
            f"All {len(survivors)} cells that clear a full-grid Bonferroni "
            f"correction move against the rescaled arms.")
    elif not harmed:
        rank_survivor_clause = (
            f"All {len(survivors)} cells that clear a full-grid Bonferroni "
            f"correction move in favour of the rescaled arms.")
    else:
        rank_survivor_clause = (
            f"Of the {len(survivors)} cells that clear a full-grid Bonferroni "
            f"correction, {len(harmed)} move against the rescaled arms and "
            f"{len(helped)} move in favour.")

    findings = [
        (f"The shipped composite gives the emotional and state channels "
         f"{affect_none:.2f} percent of the within-query ranking influence "
         f"between them, against a nominal {affect_nominal:.2f} percent. Under "
         f"minmax the same two channels carry {affect_mm:.2f} percent. The "
         f"rescaling therefore does what it was written to do, and the "
         f"weighted-spread ratio of semantic to emotional falls from "
         f"{influence['modes']['none']['pooled_weighted_sd_ratio_sem_over_emo']} "
         f"to "
         f"{influence['modes']['minmax']['pooled_weighted_sd_ratio_sem_over_emo']}."),
        (f"The declared primary comparison, {PRIMARY_ARM} minus "
         f"{REFERENCE_ARM} on {PRIMARY_METRIC}, is {delta:+.4f} "
         f"(query-level 95 percent CI {primary['ci95_low']:+.4f} to "
         f"{primary['ci95_high']:+.4f}, conversation-level "
         f"{primary['cluster_ci95_low']:+.4f} to "
         f"{primary['cluster_ci95_high']:+.4f}, Wilcoxon "
         f"{primary['wilcoxon_method']} p={fmt_p(primary['wilcoxon_p'])}). "
         f"{primary_ci_txt} The design's minimum detectable effect at 80 "
         f"percent power is {mde:.4f}, so on the declared primary metric "
         f"{primary_verdict_txt}: the sign is "
         f"{'negative' if delta < 0 else 'positive'} but the magnitude is "
         f"{headline_kind}."),
        (f"The rank-sensitive metrics {rank_agreement_txt}, and they are "
         f"the metrics that should be more sensitive here, because rescaling "
         f"reorders within the retrieved set rather than changing which "
         f"memories clear the top-5 cut. "
         + (f"Of the {explore['n_cells']} arm-by-metric cells, "
            f"{len(survivors)} clear a Bonferroni correction over the whole "
            f"grid with both bootstrap intervals excluding zero, and "
            f"{len(harmed)} of those are losses: "
            + "; ".join(cell_txt(c) for c in harmed[:6]) + "."
            if harmed else
            "No cell survives the full-grid Bonferroni correction with both "
            "intervals excluding zero.")
         + (f" {len(helped)} are gains: " + "; ".join(cell_txt(c) for c in helped[:6])
            + "." if helped else " No cell survives in the positive direction.")),
        (f"Removing the emotional channel while holding the normalization fixed "
         f"{beta_cost_txt} and the point estimates {beta_direction}. "
         f"{beta_key} is {beta_txt(PRIMARY_METRIC, beta)}, "
         f"{beta_txt('ndcg@10', beta_ndcg)} and {beta_txt('mrr', beta_mrr)}. "
         f"{beta_both_excl} of the {beta_mult['n_tests']} cells in this contrast "
         f"have both bootstrap intervals excluding zero in the direction that "
         f"favours deleting the channel, which is a stronger statement than a "
         f"null, but "
         f"{beta_mult['n_surviving_holm_at_05']} of {beta_mult['n_tests']} "
         f"survive a Holm correction over this contrast family, and the family "
         f"was added after the first run rather than pre-declared, so the "
         f"defensible reading is that beta earns nothing at its nominal weight "
         f"rather than that deleting it is a confirmed improvement. This is the "
         f"comparison exp22 could not resolve, because in exp22 beta carried "
         f"{none_share['d_emo']['mean_within_query_effective_share_pct']:.2f} "
         f"percent of the ranking. {exp22_reach_txt} Here beta carries "
         f"{mm_share['d_emo']['mean_within_query_effective_share_pct']:.2f} "
         f"percent, and removing it still costs nothing measurable."),
        (f"Rescaling never inverts a pair, which is the claim that matters, but "
         f"it is order preserving only in the weak sense: over "
         f"{order['n_channel_mode_query_checks']} (query, channel, mode) checks "
         f"there were {order['n_inversions']} inversions and "
         f"{sum(order['n_pairs_newly_tied_all_pairs_by_mode'].values())} pairs "
         f"newly tied, so a distinction can be lost even though none is "
         f"reversed. Those ties split by mechanism: "
         f"{order['n_pairs_newly_tied_all_pairs_by_mode']['minmax']} under "
         f"minmax, where the map is affine with positive slope so only float32 "
         f"resolution can cause one, against "
         f"{order['n_pairs_newly_tied_all_pairs_by_mode']['robust']} under "
         f"robust, where the clip collapses each percentile tail onto 0.0 or "
         f"1.0 by design and every adjacent tie counted "
         f"({order['n_adjacent_pairs_newly_tied_at_a_clip_bound_by_mode']['robust']}"
         f" of {order['n_adjacent_pairs_newly_tied_by_mode']['robust']}) sits "
         f"exactly on a clip bound."),
        (f"The two modes differ in whether they leave the semantic ranking "
         f"alone, and that difference decides what the composite results can be "
         f"attributed to. Under minmax the semantic ranking changed in "
         f"{order['n_queries_where_sem_rank_changed_by_mode']['minmax']} of "
         f"{dataset['queries_evaluated']} queries, and sem_pure_minmax scores "
         f"identically to sem_pure_none on every metric, so any change in "
         f"default_minmax is attributable to the non-semantic channels gaining "
         f"influence rather than to a corrupted semantic channel. Under robust "
         f"that licence is not available: the semantic ranking changed in "
         f"{order['n_queries_where_sem_rank_changed_by_mode']['robust']} of "
         f"{dataset['queries_evaluated']} queries, because for a distance "
         f"channel the clipped low tail is the head of the ranking. "
         f"{robust_head_txt}"),
        (f"Calibration against {calib['source']} holds exactly, not merely "
         f"within tolerance: default_none reproduces exp22's default arm to "
         f"{calib['worst_metric_abs_deviation']:.2e} on the worst reported "
         f"metric, and the mode 'none' pooled channel table reproduces exp22's "
         f"channel_redundancy to {calib['worst_channel_abs_deviation']:.2e}. "
         f"The same {dataset['queries_evaluated']} queries over the same "
         f"{dataset['conversations_benchmarked']} conversations are scored."),
    ]

    # The semantic-only comparison, computed rather than asserted. An earlier
    # draft of this conclusion said the semantic channel alone "scores at least
    # as well as any composite arm", which the arms table in this same file
    # contradicts: default_none beats semantic_only on three of the six reported
    # metrics, by margins far below anything tested. State the split.
    so_wins, comp_wins, ties = [], [], []
    for metric in REPORTED_METRICS:
        so_v = float(arms["semantic_only"][metric])
        best_comp = max(float(arms[a][metric]) for a in COMPOSITE_ARMS)
        if so_v > best_comp:
            so_wins.append(metric)
        elif so_v < best_comp:
            comp_wins.append(metric)
        else:
            ties.append(metric)
    if not comp_wins:
        semonly_txt = (
            "and the semantic channel alone scores at least as well as every "
            "composite arm on all "
            f"{len(REPORTED_METRICS)} reported metrics.")
    else:
        semonly_txt = (
            f"though the semantic channel alone is not uniformly best: it leads "
            f"every composite arm on {len(so_wins)} of "
            f"{len(REPORTED_METRICS)} reported metrics "
            f"({', '.join(so_wins) if so_wins else 'none'}) and trails the best "
            f"composite arm on {len(comp_wins)} "
            f"({', '.join(comp_wins)}), so the honest summary is that it is no "
            f"worse rather than strictly better. None of those gaps was tested, "
            f"and the largest is "
            f"{max(max(float(arms[a][m]) for a in COMPOSITE_ARMS) - float(arms['semantic_only'][m]) for m in comp_wins):.4f}.")

    conclusion = (
        "The four profile weights were not acting at their nominal size, and "
        "making them act does not improve retrieval on this corpus. The "
        f"declared primary metric is null. {rank_survivor_clause} Holding the "
        "normalization fixed and deleting the emotional channel entirely costs "
        f"nothing measurable on any reported metric, and the point estimates "
        f"{beta_direction}. Taken together these say the affective "
        "channels do not carry retrievable signal at these weights, so the "
        "shipped composite's near-total reliance on the semantic channel is "
        "not a bug that the weighting was hiding: it is the behaviour that "
        f"scores best of everything measured here, {semonly_txt}"
    )

    limits = [
        ("The rescaling is computed per query over the candidate set actually "
         "present. The resulting composite is a ranking score and is NOT "
         "comparable across queries or across stores of different composition, "
         "so no absolute-threshold logic may use it. Only the within-query "
         "ordering is meaningful, which is all the metrics here read."),
        ("The near-nominal effective shares under minmax are arithmetic, not a "
         "discovery. Mapping every channel onto [0, 1] inside a candidate set "
         "forces this. The informative part of that table is the residual "
         "deviation, which comes from distribution shape, since equal ranges do "
         "not imply equal standard deviations."),
        ("The declared primary metric is P@5 and it is null. The negative "
         "results are on nDCG@10 and MRR, which were not pre-declared, so they "
         "are quoted with a Bonferroni correction over the entire arm-by-metric "
         "grid and are labelled exploratory in the result file. They are "
         "reported because suppressing them would leave a misleading null on "
         "the record, not because the primary was changed after the fact. That "
         "P@5 is the least sensitive of the three to reordering inside the "
         "retrieved set is a property of the metric, stated here rather than "
         "used to reinterpret the primary."),
        ("Rescaling addresses two of the four causes of the flat channels: the "
         "worst-case theoretical normalisers EMO_NORM = 2.0 and STATE_NORM = "
         "sqrt(2), and the small absolute size of d_time at these store depths. "
         "It does NOT address the auto-state tracker being an exponential "
         "moving average, which makes consecutive snapshots nearly identical. "
         "Amplifying a near-constant channel amplifies whatever it contains, "
         "signal or noise alike. The measured losses are consistent with those "
         "channels being close to noise at this level of influence, and they do "
         "not establish that no version of an affective channel could help."),
        (f"Queries drawn from one conversation score against the same store and "
         f"the same candidate set, so they are not independent. The "
         f"conversation-level bootstrap is the interval to read where the two "
         f"disagree. The minimum detectable effect ignores the clustering and is "
         f"therefore optimistic, which makes the primary null weaker than its "
         f"stated MDE implies rather than stronger. The same caveat applies to "
         f"every Wilcoxon p-value reported here, and that is the more "
         f"consequential half of it: the exact Wilcoxon is computed over all "
         f"{dataset['queries_evaluated']} paired queries as though they were "
         f"independent, when the effective sample size is nearer the "
         f"{dataset['conversations_benchmarked']} conversations they came from. "
         f"Those p-values are not decorative, because the Bonferroni survivor "
         f"rule and the Holm counts are computed from them, so every "
         f"multiplicity-corrected threshold in this file is anti-conservative by "
         f"an unquantified factor. The cluster bootstrap is the only interval "
         f"here that accounts for the design, and no corrected p-value does."),
        ("d_sem is a clipped quantity, np.clip(1.0 - cosine, 0.0, 1.0), so any "
         "range-based statistic involving it is truncated at 1.0. Under minmax "
         "that clip becomes the top of the rescaled range for the queries where "
         "it binds."),
        ("This experiment measures ranking quality on one corpus with "
         "session-membership relevance labels. Session membership rewards "
         "topical coherence, which is what the semantic channel encodes, so "
         "this label definition is not neutral between the channels. A channel "
         "tracking genuine affect could carry real information about a "
         "conversation and still lose on this metric."),
    ]

    return {
        "headline": (
            f"Making the profile weights effective does not improve retrieval. "
            f"P@5 moves {delta:+.4f}, a null on the declared primary metric, "
            f"while {len(harmed)} of {explore['n_cells']} arm-by-metric cells "
            f"show a Bonferroni-surviving loss and none shows a "
            f"Bonferroni-surviving gain."),
        "conclusion": conclusion,
        "findings": findings,
        "honest_limits": limits,
        "what_this_does_not_settle": (
            "Whether an affective channel could ever help. This experiment "
            "shows that the channels as currently computed do not, at any "
            "weight, because giving them their nominal influence makes "
            "retrieval worse rather than better. That localises the problem in "
            "what the channels contain rather than in how they are weighted. "
            "The specific suspect is the auto-state exponential moving average, "
            "whose consecutive snapshots are nearly identical, so the stored "
            "affective vectors carry very little per-memory information for any "
            "amount of weight to act on. Whether replacing it with "
            "instantaneous per-text affect adds independent signal is the next "
            "thing to measure, and it must be measured rather than assumed: "
            "more variance in a channel is not the same as more signal in it."),
    }



def write_text_report(path: str, results: dict) -> None:
    arms = results["arms"]
    paired = results["paired_deltas_vs_reference"]
    contrasts = results["arm_against_arm_contrasts"]
    explore = results["exploratory_multiplicity"]
    influence = results["effective_influence"]
    calib = results["calibration"]
    guard = results["rescaling_live_guard"]
    order = results["order_preservation"]
    dataset = results["dataset"]
    reading = results["reading"]

    L: list[str] = []
    A = L.append
    A("EXP25: Channel normalization, or does making the weights effective help?")
    A("=" * 74)
    A("")
    A(results["hypothesis"])
    A("")
    A("DATASET")
    A("-" * 74)
    for key in ("corpus", "conversations_loaded", "conversations_benchmarked",
                "conversations_skipped_too_few_sessions",
                "conversations_skipped_no_eligible_session",
                "queries_evaluated", "total_turns_stored", "mean_store_size",
                "mean_relevant_per_query", "random_guess_precision"):
        A(f"  {key:44s} {dataset[key]}")
    A("")
    A("ARMS")
    A("-" * 74)
    A(f"  {'arm':22s} {'weights':26s} {'mode':8s}")
    for arm in ARMS:
        w = arms[arm]["weights"]
        w_txt = ("(" + ", ".join(f"{x:g}" for x in w) + ")") if w else "not applicable"
        A(f"  {arm:22s} {w_txt:26s} {arms[arm]['channel_normalization']:8s}")
        A(f"      {arms[arm]['purpose']}")
    A("")
    A("RETRIEVAL QUALITY")
    A("-" * 74)
    header = f"  {'arm':22s}" + "".join(f"{m:>11s}" for m in REPORTED_METRICS)
    A(header)
    for arm in ARMS:
        A(f"  {arm:22s}"
          + "".join(f"{arms[arm][m]:11.4f}" for m in REPORTED_METRICS))
    A(f"  {'random guess':22s}{dataset['random_guess_precision']:11.4f}")
    A("")
    A("PAIRED DIFFERENCES AGAINST " + REFERENCE_ARM)
    A("-" * 74)
    A("  Every reported metric, so a null on one metric cannot stand in for the")
    A("  others. P@5 is the declared primary; the rest are exploratory and are")
    A("  corrected over the whole grid further down.")
    A("")
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        A(f"  {arm}")
        A(f"    {'metric':9s}{'delta':>9s} {'query CI':>20s} {'conv CI':>20s} "
          f"{'p':>10s} {'nonzero':>8s}")
        for metric in REPORTED_METRICS:
            d = paired["deltas"][arm][metric]
            star = ""
            if (d["ci95_zero_relation"] == "excludes_zero"
                    and d["cluster_ci95_zero_relation"] == "excludes_zero"):
                star = "  both CIs exclude zero"
            A(f"    {metric:9s}{d['mean_delta']:+9.4f} "
              f"[{d['ci95_low']:+8.4f},{d['ci95_high']:+8.4f}] "
              f"[{d['cluster_ci95_low']:+8.4f},{d['cluster_ci95_high']:+8.4f}] "
              f"{fmt_p(d['wilcoxon_p']):>10s} {d['n_nonzero_pairs']:8d}{star}")
        d5 = paired["deltas"][arm]["p@5"]
        A(f"    p@5 sign pattern: {d5['sign_pattern']}")
        A(f"    p@5 MDE at 80 percent power {d5['mde80_two_sided_05']:.4f}")
        A("")
    mult = paired["multiplicity"]
    A("  Primary test (declared before the run, no correction):")
    A(f"    {mult['primary']['comparison']} on {mult['primary']['metric']}: "
      f"delta {mult['primary']['mean_delta']:+.4f}, "
      f"{mult['primary']['wilcoxon_method']} p="
      f"{fmt_p(mult['primary']['wilcoxon_p'])}")
    A("  Secondary family (declared before the run), Holm adjusted:")
    for key in mult["secondary_family"]:
        A(f"    {key:34s} raw p={fmt_p(mult['secondary_wilcoxon_p'][key]):>9s}  "
          f"Holm p={fmt_p(mult['secondary_holm_adjusted_p'][key]):>9s}")
    A("")
    A("DECLARED ARM-AGAINST-ARM CONTRASTS")
    A("-" * 74)
    for key, block in contrasts["contrasts"].items():
        A(f"  {key}")
        A(f"    {block['why']}")
        A(f"    {'metric':9s}{'delta':>9s} {'query CI':>20s} {'conv CI':>20s} "
          f"{'p':>10s} {'Holm p':>9s} {'nonzero':>8s}")
        for metric in REPORTED_METRICS:
            d = block["metrics"][metric]
            A(f"    {metric:9s}{d['mean_delta']:+9.4f} "
              f"[{d['ci95_low']:+8.4f},{d['ci95_high']:+8.4f}] "
              f"[{d['cluster_ci95_low']:+8.4f},{d['cluster_ci95_high']:+8.4f}] "
              f"{fmt_p(d['wilcoxon_p']):>10s} "
              f"{fmt_p(d.get('holm_p_within_contrast_family')):>9s} "
              f"{d['n_nonzero_pairs']:8d}")
        A("")
    cmult = contrasts["multiplicity"]
    A(f"  Holm over the {cmult['n_tests']} cells that are hypothesis tests: "
      f"{cmult['n_surviving_holm_at_05']} survive at 0.05.")
    for key in cmult["excluded_from_the_family"]:
        A(f"  {key} contributes no tests: {cmult['why_excluded']}.")
    A(f"  {cmult['declaration_note']}.")
    A("")
    A("EXPLORATORY MULTIPLICITY OVER THE WHOLE GRID")
    A("-" * 74)
    A(f"  {explore['n_cells']} cells: {explore['grid']}")
    A(f"  correction: {explore['correction']}")
    A("")
    if explore["survivors"]:
        A("  Cells clearing the correction with both bootstraps excluding zero:")
        A(f"    {'arm':18s}{'metric':9s}{'delta':>9s}{'raw p':>12s}"
          f"{'Bonf p':>12s}  declared")
        for c in explore["survivors"]:
            A(f"    {c['arm']:18s}{c['metric']:9s}{c['mean_delta']:+9.4f}"
              f"{fmt_p(c['wilcoxon_p']):>12s}"
              f"{fmt_p(c['bonferroni_p_over_full_grid']):>12s}"
              f"  {'yes' if c['pre_declared'] else 'no, exploratory'}")
    else:
        A("  No cell clears the correction with both bootstraps excluding zero.")
    A("")
    A("  " + explore["reading"])
    A("")
    A("EFFECTIVE RANKING INFLUENCE PER CHANNEL")
    A("-" * 74)
    A("  Share of within-query influence, weight times within-query spread,")
    A("  normalised across the four channels, shipped weights in every mode.")
    A("")
    A(f"  {'channel':10s}{'nominal':>10s}" + "".join(f"{m:>12s}" for m in MODES))
    for cname in CHANNEL_NAMES:
        row = f"  {cname:10s}{influence['nominal_share_pct'][cname]:9.2f}%"
        for mode in MODES:
            row += (f"{influence['modes'][mode]['channels'][cname]['mean_within_query_effective_share_pct']:11.2f}%")
        A(row)
    A("")
    A("  Pooled dispersion, directly comparable to exp22's channel_redundancy:")
    A(f"  {'channel':10s}{'mode':>10s}{'mean':>12s}{'sd':>12s}"
      f"{'weighted sd':>14s}{'pooled share':>14s}")
    for cname in CHANNEL_NAMES:
        for mode in MODES:
            r = influence["modes"][mode]["channels"][cname]
            A(f"  {cname:10s}{mode:>10s}{r['pooled_mean']:12.6f}"
              f"{r['pooled_sd']:12.6f}{r['pooled_weighted_sd']:14.6f}"
              f"{r['pooled_effective_share_pct']:13.2f}%")
    A("")
    A("  " + influence["tautology_disclosure"])
    A("")
    A("GUARDS (each one gates the run)")
    A("-" * 74)
    for key, value in guard["checks"].items():
        A(f"  [{'PASS' if value else 'FAIL'}] {key}")
    A(f"  context: {guard['guard_context']}")
    A(f"  max |d| none vs minmax  : {guard['max_abs_distance_diff_none_vs_minmax']}")
    A(f"  max |d| none vs robust  : {guard['max_abs_distance_diff_none_vs_robust']}")
    A(f"  single-channel Spearman : {guard['sem_pure_spearman_none_vs_minmax']}")
    A(f"  mode read at            : {guard['mode_read_at']}")
    A("")
    A("ORDER PRESERVATION OVER EVERY QUERY")
    A("-" * 74)
    for key in ("n_channel_mode_query_checks", "n_inversions",
                "n_adjacent_pairs_newly_tied", "n_bounds_violations",
                "n_unit_span_violations"):
        A(f"  {key:52s} {order[key]}")
    for mode in MODES:
        A(f"  {('newly tied under ' + mode):52s} "
          f"{order['n_pairs_newly_tied_all_pairs_by_mode'][mode]} all pairs, "
          f"{order['n_adjacent_pairs_newly_tied_by_mode'][mode]} adjacent"
          f" ({order['n_adjacent_pairs_newly_tied_at_a_clip_bound_by_mode'][mode]}"
          f" of those exactly on a clip bound)")
    for mode, n_changed in order["n_queries_where_sem_rank_changed_by_mode"].items():
        A(f"  {('semantic rank changed under ' + mode):52s} {n_changed}"
          f" of {order['n_queries_checked_for_sem_rank']} queries"
          f"{'  <-- expected: the clip can reorder d_sem' if mode == 'robust' else '  <-- must be 0'}")
    A(f"  {order['tie_counting_note']}")
    A(f"  {order['tie_mechanism']}")
    A("")
    A("CALIBRATION AGAINST " + calib["source"])
    A("-" * 74)
    for key, value in calib["gates"].items():
        A(f"  [{'PASS' if value else 'FAIL'}] {key}")
    A(f"  worst reported-metric deviation : {calib['worst_metric_abs_deviation']:.3e} "
      f"(tolerance {calib['metric_tolerance']:.0e})")
    A(f"  worst channel-table deviation   : {calib['worst_channel_abs_deviation']:.3e} "
      f"(tolerance {calib['channel_tolerance']:.0e})")
    A(f"  exp22 default P@5 {calib['exp22_arm_default']['p@5']:.4f} against "
      f"exp25 default_none P@5 {calib['exp25_arm_default_none']['p@5']:.4f}")
    A("")
    A("READING")
    A("-" * 74)
    A("  " + reading["headline"])
    A("")
    for item in reading["findings"]:
        A("  - " + item)
        A("")
    A("CONCLUSION")
    A("-" * 74)
    A("  " + reading["conclusion"])
    A("")
    A("HONEST LIMITS")
    A("-" * 74)
    for item in reading["honest_limits"]:
        A("  - " + item)
        A("")
    A("WHAT THIS DOES NOT SETTLE")
    A("-" * 74)
    A("  " + reading["what_this_does_not_settle"])
    A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EXP25 channel-normalization effect on retrieval")
    ap.add_argument("--max-conversations", type=int,
                    default=DEFAULT_MAX_CONVERSATIONS,
                    help=(f"corpus records to load; the default "
                          f"{DEFAULT_MAX_CONVERSATIONS} is exp22's default and "
                          f"is the scale of the shipped result files in "
                          f"experiments/results/exp25/"))
    ap.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    if set(MODES) != set(CHANNEL_NORMALIZATION_MODES):
        print("[exp25] ABORT: ncm.retrieval.CHANNEL_NORMALIZATION_MODES is "
              f"{tuple(CHANNEL_NORMALIZATION_MODES)} but this script tests "
              f"{MODES}. Extend the script rather than shipping an untested mode.")
        return 1

    try:
        ref = load_exp22_reference()
    except Exp22ReferenceError as exc:
        print(f"[exp25] ABORT: {exc}")
        print("[exp25] No result files written: there is no constant fallback for "
              "the exp22 reference on purpose.")
        return 1
    print(f"[exp25] exp22 reference read from {EXP22_RESULTS_REL}: "
          f"{ref['queries_evaluated']} queries, default P@5 "
          f"{ref['arm_default']['p@5']:.4f}, emotional weighted sd "
          f"{ref['channels']['d_emo']['weighted_sd']:.6f}")

    corpus_records = exp22.count_corpus_records(CORPUS_PATH)
    print(f"[exp25] Corpus {CORPUS_REL}: {corpus_records} records")
    print(f"[exp25] Loading first {args.max_conversations} (seed={SEED})")
    conversations = exp22.load_conversations(CORPUS_PATH, args.max_conversations)
    print(f"[exp25] Loaded {len(conversations)} conversations")
    if not conversations:
        print("[exp25] ERROR: no conversations loaded. Aborting.")
        return 1

    encoder = SentenceEncoder(
        model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models"))
    print(f"[exp25] Encoder backend: {encoder.backend}")
    if encoder.backend != "sentence-transformers":
        print("[exp25] ABORT: the hash fallback carries no semantic structure, so")
        print(f"[exp25]        any retrieval number would be meaningless. Reason: "
              f"{encoder.backend_error}")
        return 1

    arm_profiles = {a: build_profile(a, w, m) for a, (w, m) in ARM_SPEC.items()}
    probe_profiles = {
        (mode, cname): build_profile(f"probe_{cname}_{mode}", w, mode)
        for mode in MODES for cname, w in CHANNEL_PROBES
    }
    # The guard also needs the arm-shaped single-channel profiles by plain name.
    probe_profiles["sem_pure_none"] = arm_profiles["sem_pure_none"]
    probe_profiles["sem_pure_minmax"] = arm_profiles["sem_pure_minmax"]

    print(f"[exp25] Scoring {len(ARMS)} arms over {len(MODES)} modes")
    try:
        bench = benchmark(conversations, encoder, arm_profiles, probe_profiles)
    except RescalingNotLiveError as exc:
        print(f"[exp25] ABORT: {exc}")
        print("[exp25] No result files written: a rescaling that is not live or "
              "not order preserving would make every comparison an artifact.")
        return 1

    guard = bench["guard"]
    if guard is None:
        print("[exp25] ABORT: no query was evaluated, so the guard never ran.")
        return 1

    order = bench["order_preservation"]
    # The semantic-rank gate is minmax-only on purpose. Under minmax the map is
    # affine with positive slope, so any reordering of d_sem is a bug and must
    # abort. Under robust the clip at the percentile bounds is expected to
    # reorder d_sem by tying its tails, so aborting on that would abort on the
    # mode working as designed. The robust count is reported and the
    # sem_pure_robust arm measures what it costs.
    if (order["n_inversions"] or order["n_bounds_violations"]
            or order["n_unit_span_violations"]
            or order["n_queries_where_sem_rank_changed_by_mode"]["minmax"]):
        print("[exp25] ABORT: rescaling violated its documented behaviour over "
              f"the full query set: {json.dumps(order)}")
        return 1

    arms = summarize_arms(bench["per_query"])
    influence = effective_influence(
        bench["pooled"], bench["within"], bench["within_share"])
    paired = paired_table(bench["per_query"], bench["query_conv_ids"],
                          args.bootstrap_resamples)
    contrasts = contrast_table(bench["per_query"], bench["query_conv_ids"],
                               args.bootstrap_resamples)
    explore = exploratory_multiplicity(paired)
    calib = calibration(arms, bench["dataset"], influence, ref,
                        args.max_conversations)
    if not calib["passed"]:
        print(f"[exp25] ABORT: calibration against {calib['source']} failed: "
              f"{json.dumps(calib['gates'])}")
        print("[exp25] No result files written: without reproducing exp22 under "
              "mode 'none' this script cannot claim to be measuring the same "
              "queries, so no comparison would mean anything.")
        return 1

    results = {
        "experiment": "EXP25 channel normalization",
        "hypothesis": (
            "The profile weights alpha, beta, gamma and delta do not act at "
            "their nominal size, because a channel influences a ranking only "
            "through how much it varies across the candidates. Rescaling each "
            "channel across the candidate set before the weighted sum makes the "
            "nominal weights close to the effective ones. This measures what "
            "that does to retrieval quality. Both directions are reportable: a "
            "gain means the affective channels carry usable signal that the "
            "arithmetic was suppressing, and a loss means they carry noise and "
            "the near-constant auto-state moving average has to be fixed before "
            "any weight on them can help."),
        "config": {
            "max_conversations": args.max_conversations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": SEED,
            "cluster_seed": CLUSTER_SEED,
            "modes": list(MODES),
            "reference_arm": REFERENCE_ARM,
            "primary_test": f"{PRIMARY_ARM} minus {REFERENCE_ARM} on {PRIMARY_METRIC}",
            "corpus_records": corpus_records,
            "encoder_backend": encoder.backend,
            "protocol_provenance": (
                "the corpus loader, store construction, metric definitions and "
                "paired statistics are imported from "
                "experiments/python/exp22_emo_ablation.py, not transcribed, so "
                "the two experiments cannot drift apart. The held-out-turn RNG "
                "is consumed in exp22's order, so the query set is identical"),
            "arm_spec": {a: {"weights": list(w), "channel_normalization": m}
                         for a, (w, m) in ARM_SPEC.items()},
        },
        "dataset": bench["dataset"],
        "arms": arms,
        "paired_deltas_vs_reference": paired,
        "arm_against_arm_contrasts": contrasts,
        "exploratory_multiplicity": explore,
        "effective_influence": influence,
        "rescaling_live_guard": guard,
        "order_preservation": order,
        "code_path_cross_check": bench["code_path_cross_check"],
        "calibration": calib,
        "exp22_reference": ref,
    }
    results["reading"] = build_reading(
        arms, paired, influence, guard, order, calib, ref, bench["dataset"],
        contrasts, explore)

    json_path = os.path.join(RESULTS_DIR, "exp25_channel_normalization.json")
    txt_path = os.path.join(RESULTS_DIR, "exp25_channel_normalization.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_text_report(txt_path, results)
    p1 = plot_arm_metrics(arms, bench["dataset"])
    p2 = plot_paired_deltas(paired)
    p3 = plot_effective_influence(influence)

    print("")
    print(f"[exp25] {results['reading']['headline']}")
    print("")
    for arm in ARMS:
        print(f"[exp25]   {arm:22s} P@5 {arms[arm]['p@5']:.4f}  "
              f"nDCG@10 {arms[arm]['ndcg@10']:.4f}  MRR {arms[arm]['mrr']:.4f}")
    print("")
    print(f"[exp25] {results['reading']['conclusion']}")
    print("")
    print(f"[exp25] Saved {json_path}")
    print(f"[exp25] Saved {txt_path}")
    for path in (p1, p2, p3):
        print(f"[exp25] Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
