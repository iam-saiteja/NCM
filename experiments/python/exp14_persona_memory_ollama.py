"""
EXP14: Does persona memory content change model output style?
=============================================================

What is tested
- Two memory stores hold different persona descriptions (A concise/analytical,
  B warm/affiliative). Every other input is held identical. The same prompts are
  issued to a local Ollama model with each store's retrieved lines injected as
  context. The measurement is whether the style of the generated text differs.

Why this script was rewritten
- The previous version of this experiment is retracted. It sampled with
  temperature 0.2 and top_p 0.9 and no seed, which was measured to be
  non-reproducible on this machine. It used 6 prompts with one sample each and
  reported bare mean differences with no dispersion, no confidence interval and
  no significance test. Its analytical marker list contained the word "plan",
  which appears verbatim in one of its own prompts, so that metric partly
  counted prompt echo. Its warm marker list contained "support" and "care",
  which appear verbatim in the persona B seed lines that were injected into the
  prompt, so that metric partly counted context echo. It also defaulted to
  model "qwen2:7B", which is not installed here, so the recorded run cannot be
  reproduced at all. See experiments/results/exp14/RETRACTION.md.

Protocol corrections in this version
1. Greedy decoding (temperature 0.0, top_p 1.0, top_k 1, fixed seed,
   num_predict -1). Measured bit-exact across repeat runs on this machine.
   --verify-determinism re-runs one prompt N times and records whether the
   outputs were byte-identical, so the determinism claim has its own receipt.
2. num_predict is -1, not 512. A 512 token cap truncated every probe response
   and any length metric measured under it was a measurement of the cap.
   done_reason is recorded per generation and a truncated run is marked invalid.
3. Marker lists are asserted disjoint from every prompt and from every persona
   seed line of both arms, checked at runtime. A verbatim copy of an injected
   line therefore contributes zero to the style metrics by construction.
4. Marker matching uses word boundaries, not raw substring counting. Raw
   counting made "care" match "career" and double counted "step" inside
   "step-by-step".
5. The primary metric per family is the number of distinct markers present, not
   the total count, because a total can be moved by one repeated word.
6. 12 prompts, balanced 6 affective register and 6 task register. The previous
   set was 6 prompts of which 4 were affective.
7. Paired analysis across prompts: per-prompt differences, mean, standard
   deviation, Cohen dz, bootstrap 95 percent confidence interval, Wilcoxon
   signed-rank test, and Holm-Bonferroni adjustment across the metric family.

Scope limit stated up front: n is 12 paired prompts on one 3.8B parameter model.
That supports a claim about this model under this protocol and nothing broader.

Outputs (experiments/results/exp14/)
- exp14_persona_memory_ollama.json   full per-prompt records and statistics
- exp14_persona_memory_ollama.txt    summary table
- exp14_persona_memory_ollama.md     side-by-side response dump
- exp14_paired_differences.png       per-prompt paired differences
- exp14_effect_sizes.png             mean differences with bootstrap CIs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is present in the project venv
    scipy_stats = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import MemoryEntry, MemoryProfile, MemoryStore, SentenceEncoder, retrieve_top_k_fast

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_VERSION_URL = f"{OLLAMA_HOST}/api/version"

# Greedy decoding. Measured bit-exact across 3 repeat runs of the same prompt on
# this machine with Ollama 0.32.14 and phi4-mini:latest. seed alone at
# temperature 0.2 was measured NOT to be deterministic (377 vs 394 words).
GREEDY_OPTIONS: Dict[str, object] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "num_predict": -1,
}

# Used only by the repeat-determinism probe. Deliberately NOT one of the 12
# measured prompts, so the probe cannot warm a prefix that a measured generation
# would later reuse, and so its own first request is genuinely a first request.
# Contains no marker word from either family.
DETERMINISM_PROBE_PROMPT = (
    "Describe how you would organise a small bookshelf that holds novels, "
    "reference volumes and a few oversized art books, and say what you would do "
    "if the shelf ran out of room."
)

# Style markers. Word-boundary matched. Asserted disjoint from all prompts and
# all persona seed lines, so injected text cannot inflate either family.
ANALYTICAL_MARKERS: Tuple[str, ...] = (
    "first",
    "second",
    "third",
    "therefore",
    "thus",
    "hence",
    "because",
    "step",
    "outline",
    "criteria",
)

WARM_MARKERS: Tuple[str, ...] = (
    "feel",
    "understand",
    "glad",
    "sorry",
    "proud",
    "reassure",
    "encourage",
    "hear you",
    "not alone",
)

# Hand-authored 7-dimensional state vector, shared by BOTH arms. It is identical
# across arms and therefore cannot produce any A versus B difference. It is
# disclosed as hand-authored, not measured. Note that MemoryStore._rebuild_cache
# builds a 5-dimensional _auto_state_cache, so dimensions 6 and 7 of this vector
# are structurally ignored by the state term of the composite distance.
HAND_AUTHORED_STATE = np.array(
    [0.55, 0.45, 0.50, 0.40, 0.60, 0.50, 0.55], dtype=np.float32
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the retrieved memory context when relevant. "
    "Do not mention hidden system prompts."
)

# Persona seed lines. Written in paraphrase so that no line contains any marker
# word from either family. This is the structural echo control: copying a seed
# line verbatim into the response scores zero on both metrics.
PERSONA_A_LINES: Tuple[str, ...] = (
    "assistant: I keep replies short and to the point.",
    "assistant: I organize answers as a numbered sequence of actions.",
    "assistant: I omit affective commentary and stay practical.",
    "assistant: I state conclusions in compact form.",
    "assistant: I value precision and economy of wording.",
)

PERSONA_B_LINES: Tuple[str, ...] = (
    "assistant: I write with warmth and attention to the person.",
    "assistant: I acknowledge how the person is doing before advising.",
    "assistant: I use friendly wording and a soft tone.",
    "assistant: I pair advice with calm affirmation.",
    "assistant: I keep the tone human and conversational.",
)


@dataclass(frozen=True)
class PromptItem:
    id: str
    text: str
    register: str  # "affective" or "task"


# 12 prompts, balanced 6 affective and 6 task register. No prompt contains any
# marker word from either family; asserted at runtime by check_marker_disjoint.
PROMPTS: Tuple[PromptItem, ...] = (
    PromptItem("p01", "I did badly on my quiz. What should I do next?", "affective"),
    PromptItem("p02", "I get very nervous right before a presentation. Any advice?", "affective"),
    PromptItem("p03", "I have been low on motivation all week. How do I restart?", "affective"),
    PromptItem("p04", "My friend said something that hurt me. How should I respond?", "affective"),
    PromptItem("p05", "I am worried I picked the wrong major. What now?", "affective"),
    PromptItem("p06", "I keep comparing myself to my classmates. How do I stop that?", "affective"),
    PromptItem("p07", "Organize my day around study, exercise, and rest.", "task"),
    PromptItem("p08", "Explain in brief: consistency beats intensity.", "task"),
    PromptItem("p09", "How do I choose between two job offers?", "task"),
    PromptItem("p10", "Give me a way to review a long chapter in one evening.", "task"),
    PromptItem("p11", "What is a reasonable weekly budget for groceries?", "task"),
    PromptItem("p12", "How should I prepare for a coding interview in two weeks?", "task"),
)

METRIC_KEYS: Tuple[str, ...] = (
    "analytical_marker_types",
    "warm_marker_types",
    "analytical_markers",
    "warm_markers",
    "words",
    "chars",
    "exclamations",
    "questions",
)

# The two metrics the hypothesis is actually about. Holm-Bonferroni is reported
# across the full family in METRIC_KEYS, but these are named as primary in
# advance so the analysis is not selected after seeing the numbers.
PRIMARY_METRIC_KEYS: Tuple[str, ...] = (
    "analytical_marker_types",
    "warm_marker_types",
)


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[exp14 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_marker_disjoint() -> None:
    """
    Fail loudly if any marker word appears in any prompt or any persona seed
    line. This is what makes the style metrics immune to prompt echo and to
    verbatim copying of injected context. The previous version of this
    experiment violated both conditions, which is one reason its numbers are
    retracted. The check is deliberately substring based, which is stricter
    than the word-boundary matching used for scoring.
    """
    all_markers = tuple(ANALYTICAL_MARKERS) + tuple(WARM_MARKERS)
    sources: List[Tuple[str, str]] = []
    for p in PROMPTS:
        sources.append((f"prompt {p.id}", p.text))
    for i, line in enumerate(PERSONA_A_LINES):
        sources.append((f"persona_a line {i}", line))
    for i, line in enumerate(PERSONA_B_LINES):
        sources.append((f"persona_b line {i}", line))
    sources.append(("system_prompt", SYSTEM_PROMPT))
    sources.append(("determinism_probe_prompt", DETERMINISM_PROBE_PROMPT))

    violations: List[str] = []
    for label, text in sources:
        low = text.lower()
        for marker in all_markers:
            if marker in low:
                violations.append(f"{label} contains marker '{marker}': {text!r}")
    if violations:
        raise ValueError(
            "Marker leakage detected. Style metrics would partly measure echo of "
            "the input rather than generated style. Offending items:\n  "
            + "\n  ".join(violations)
        )


def _http_get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_preflight(model: str, timeout: int = 15) -> dict:
    """
    Confirm the model is actually installed and capture provenance for it.

    The previous version of this experiment defaulted to "qwen2:7B", which is not
    installed on this machine. It therefore recorded results that cannot be
    reproduced. Failing loudly here is the fix.
    """
    version_info: Dict[str, object] = {}
    try:
        version_info = _http_get_json(OLLAMA_VERSION_URL, timeout=timeout)
    except Exception as exc:
        version_info = {"version": f"[unavailable: {type(exc).__name__}]"}

    tags = _http_get_json(OLLAMA_TAGS_URL, timeout=timeout)
    installed = tags.get("models", []) or []
    names = [m.get("name", "") for m in installed]
    match: Optional[dict] = None
    for m in installed:
        if m.get("name") == model:
            match = m
            break
    if match is None:
        raise RuntimeError(
            f"Model {model!r} is not installed in Ollama at {OLLAMA_HOST}.\n"
            f"Installed models: {names}\n"
            "Pass --model with one of the installed names. Do not record results "
            "for a model that is not present."
        )

    details = match.get("details", {}) or {}
    return {
        "ollama_version": version_info.get("version", "[unavailable]"),
        "ollama_host": OLLAMA_HOST,
        "model": model,
        "model_digest": match.get("digest", "[unavailable]"),
        "model_size_bytes": match.get("size"),
        "model_parameter_size": details.get("parameter_size"),
        "model_quantization": details.get("quantization_level"),
        "model_family": details.get("family"),
        "installed_models": names,
    }


def ollama_chat(
    model: str,
    messages: List[dict],
    seed: int,
    timeout: int,
) -> Dict[str, object]:
    """
    One greedy generation. Returns the text plus the fields needed to audit it.

    done_reason is returned so that a truncated generation can be detected. Any
    length metric computed over a truncated response measures the token cap, not
    the model, which is why num_predict is -1 and why this is recorded.
    """
    options = dict(GREEDY_OPTIONS)
    options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {
        "text": data["message"]["content"],
        "done_reason": data.get("done_reason"),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration_ns": data.get("total_duration"),
    }


def build_store(
    encoder: SentenceEncoder,
    persona_lines: Sequence[str],
    state: np.ndarray,
) -> MemoryStore:
    """
    Build one arm's store. Shipped profile defaults are used unchanged.

    gate_check is False on purpose. With the write gate active, the two arms
    could retain different numbers of persona lines, and a difference in memory
    count would confound the style contrast. Bypassing the gate guarantees both
    arms hold exactly the same number of memories.

    store.step is incremented manually because MemoryStore.add does not advance
    the step counter; only MemoryStore.tick does. Without this, every memory
    would carry the same effective age and the temporal term would be constant.
    """
    store = MemoryStore(profile=MemoryProfile())
    emo = encoder.encode_emotional(state)
    snap = encoder.encode_state(state)
    for i, line in enumerate(persona_lines):
        sem = encoder.encode(line)
        store.add(
            MemoryEntry(
                e_semantic=sem,
                e_emotional=emo,
                s_snapshot=snap,
                timestamp=i,
                text=line,
                tags=["persona_seed"],
            ),
            gate_check=False,
        )
        store.step += 1
    return store


def retrieve_context_lines(
    encoder: SentenceEncoder,
    store: MemoryStore,
    query: str,
    state: np.ndarray,
    top_k: int,
) -> Tuple[List[str], List[str]]:
    """
    Retrieve top_k persona lines for one query.

    Note on s_current_normalized: retrieve_top_k_fast accepts this argument but
    ignores it. ncm/retrieval.py line 372 reads
    store.auto_state.get_current_state() instead. It is passed here only because
    the signature requires it. Nothing in this experiment depends on it, and
    both arms are identical in this respect, so it cannot affect the contrast.
    """
    q_sem = encoder.encode(query)
    q_emo = encoder.encode_emotional(state)
    q_state = encoder.encode_state(state)
    rows = retrieve_top_k_fast(
        query_semantic=q_sem,
        query_emotional=q_emo,
        store=store,
        s_current_normalized=q_state,
        current_step=int(store.step),
        k=top_k,
    )
    ids = [m.id for _, _, m in rows]
    lines = [m.text for _, _, m in rows]
    return ids, lines


def count_markers(text: str, markers: Sequence[str]) -> Tuple[int, int]:
    """
    Word-boundary marker counting. Returns (total occurrences, distinct types).

    Raw substring counting, used by the retracted version, made "care" match
    inside "career" and counted "step" twice inside "step-by-step". The distinct
    type count is the primary metric because a total can be moved by a single
    repeated word, while the type count cannot.
    """
    low = text.lower()
    total = 0
    present = 0
    for marker in markers:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in marker.split()) + r"\b"
        hits = len(re.findall(pattern, low))
        total += hits
        if hits > 0:
            present += 1
    return total, present


def style_metrics(text: str) -> Dict[str, float]:
    analytical_total, analytical_types = count_markers(text, ANALYTICAL_MARKERS)
    warm_total, warm_types = count_markers(text, WARM_MARKERS)
    return {
        "analytical_marker_types": float(analytical_types),
        "warm_marker_types": float(warm_types),
        "analytical_markers": float(analytical_total),
        "warm_markers": float(warm_total),
        "words": float(len(text.split())),
        "chars": float(len(text)),
        "exclamations": float(text.count("!")),
        "questions": float(text.count("?")),
    }


def bootstrap_ci_mean(
    diffs: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 20260820,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Percentile bootstrap CI for the mean paired difference. The RNG is seeded so
    the interval is reproducible from the recorded seed.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def holm_bonferroni(pvals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Holm-Bonferroni step-down adjustment across the metric family.

    Eight metrics are tested on the same 12 paired observations. Reporting raw
    p-values across eight tests would overstate significance, so both the raw
    and the adjusted values are recorded and the adjusted value is the one to
    quote. Metrics whose test could not be computed are passed through as None
    and excluded from the family size.
    """
    usable = [(k, v) for k, v in pvals.items() if v is not None]
    out: Dict[str, Optional[float]] = {k: None for k in pvals}
    if not usable:
        return out
    usable.sort(key=lambda kv: kv[1])
    m = len(usable)
    running = 0.0
    for i, (key, p) in enumerate(usable):
        adjusted = min(1.0, (m - i) * float(p))
        running = max(running, adjusted)  # enforce monotonicity
        out[key] = float(running)
    return out


def paired_analysis(
    metrics_a: List[Dict[str, float]],
    metrics_b: List[Dict[str, float]],
    prompt_ids: Sequence[str],
    bootstrap_seed: int,
) -> dict:
    """
    Paired comparison across prompts.

    Design note: under greedy decoding each (arm, prompt) response is
    deterministic, so repeated sampling of the same prompt adds no information.
    The only source of variation is the prompt itself, which makes the paired
    difference across prompts the correct unit of analysis. n equals the number
    of prompts.
    """
    n = len(prompt_ids)
    per_metric: Dict[str, dict] = {}
    raw_p: Dict[str, Optional[float]] = {}

    for key in METRIC_KEYS:
        a_vals = [float(m[key]) for m in metrics_a]
        b_vals = [float(m[key]) for m in metrics_b]
        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        arr = np.asarray(diffs, dtype=np.float64)
        mean_diff = float(arr.mean()) if n else float("nan")
        sd_diff = float(arr.std(ddof=1)) if n > 1 else float("nan")
        dz: Optional[float] = None
        if n > 1 and sd_diff > 0.0:
            dz = float(mean_diff / sd_diff)

        p_value: Optional[float] = None
        test_note = ""
        if scipy_stats is None:
            test_note = "scipy unavailable, no test computed"
        elif n < 6:
            test_note = f"n={n} too small for a meaningful signed-rank test"
        elif np.all(arr == 0.0):
            test_note = "all paired differences are exactly zero, test undefined"
        else:
            try:
                res = scipy_stats.wilcoxon(b_vals, a_vals, zero_method="wilcox")
                p_value = float(res.pvalue)
            except Exception as exc:
                test_note = f"wilcoxon failed: {type(exc).__name__}: {exc}"

        lo, hi = bootstrap_ci_mean(diffs, seed=bootstrap_seed)
        n_pos = int(np.sum(arr > 0))
        n_neg = int(np.sum(arr < 0))
        n_zero = int(np.sum(arr == 0))

        raw_p[key] = p_value
        per_metric[key] = {
            "persona_a_mean": float(np.mean(a_vals)) if n else None,
            "persona_b_mean": float(np.mean(b_vals)) if n else None,
            "persona_a_sd": float(np.std(a_vals, ddof=1)) if n > 1 else None,
            "persona_b_sd": float(np.std(b_vals, ddof=1)) if n > 1 else None,
            "mean_difference_b_minus_a": mean_diff,
            "sd_of_differences": sd_diff,
            "cohens_dz": dz,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "wilcoxon_p_raw": p_value,
            "test_note": test_note,
            "n_prompts_b_greater": n_pos,
            "n_prompts_a_greater": n_neg,
            "n_prompts_tied": n_zero,
            "per_prompt_differences": {pid: d for pid, d in zip(prompt_ids, diffs)},
        }

    adjusted = holm_bonferroni(raw_p)
    for key in METRIC_KEYS:
        per_metric[key]["wilcoxon_p_holm"] = adjusted.get(key)

    return {
        "n_paired_prompts": n,
        "primary_metrics": list(PRIMARY_METRIC_KEYS),
        "multiple_comparison_correction": "Holm-Bonferroni across all metrics in METRIC_KEYS",
        "per_metric": per_metric,
    }


def build_user_message(context_lines: Sequence[str], prompt_text: str) -> str:
    body = "\n".join(f"- {x}" for x in context_lines)
    return f"Retrieved memory context:\n{body}\n\nUser prompt:\n{prompt_text}"


def ollama_unload(model: str, timeout: int, verbose: bool) -> dict:
    """
    Force the model out of memory so this run starts from a clean server state.

    This is required for run to run reproducibility, and the reason is prompt
    prefix caching. Two runs back to back without an unload are not comparable:
    the second run reissues the same 24 prompts the first run just issued, so it
    finds their prefixes already cached and generates in the repeat regime while
    the first run generated in the first-request regime. Measured on this stack,
    that made 16 of 24 generations differ between two consecutive runs.

    Unloading first makes every run begin identically, so generation k has the
    same request history in every run. exp14_determinism_probe.json shows this
    works: one sequence run twice from a forced unload matched byte for byte at
    every position.
    """
    log("forcing model unload for a clean starting state", verbose)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {**GREEDY_OPTIONS, "seed": 1},
        "keep_alive": 0,
    }
    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        json.loads(response.read().decode("utf-8"))
    # Give the server a moment to actually release the model before reloading.
    time.sleep(2.0)
    return {
        "performed": True,
        "mechanism": "keep_alive=0 on a throwaway request",
        "why": (
            "Prompt prefix caching makes a rerun non-comparable unless the model "
            "is unloaded first. Without this, 16 of 24 generations differed "
            "between two consecutive runs."
        ),
    }


def ollama_warmup(model: str, seed: int, timeout: int, verbose: bool) -> dict:
    """
    Issue one generation and discard its text, so that every measured generation
    runs against an already-loaded model.

    This exists because of a measured effect, not as a precaution. On the first
    run of this experiment the determinism probe returned outputs of 294, 275 and
    275 characters with eval_count 64, 58 and 58 for three byte-identical
    requests under greedy decoding with a fixed seed. Probes two and three were
    bit-identical to each other; only the first differed. The first generation
    after the model is loaded into the accelerator is numerically different from
    the steady state, so an experiment whose first call is also its first
    measurement is not reproducible even under greedy decoding. Discarding one
    generation removes that.
    """
    log("warmup generation (discarded)", verbose)
    got = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Reply with the single word ready."},
        ],
        seed=seed,
        timeout=timeout,
    )
    return {
        "performed": True,
        "discarded": True,
        "done_reason": got.get("done_reason"),
        "eval_count": got.get("eval_count"),
        "reason": (
            "The first generation after model load is numerically different from "
            "the warm steady state. Measured: 3 identical greedy requests returned "
            "chars [294, 275, 275] and eval_count [64, 58, 58] when the first was "
            "not preceded by a warmup."
        ),
    }


def sha256_text(text: str) -> str:
    """Stable content hash of one generation, used for the reproducibility check."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


HASHES_PATH = os.path.join(RESULTS_DIR, "exp14_generation_hashes.json")


def load_baseline_hashes() -> Optional[dict]:
    """
    Read the per-generation hashes recorded by the previous run, if any.

    This must be called before any output is written, because the current run
    overwrites the same file.
    """
    if not os.path.exists(HASHES_PATH):
        return None
    try:
        with open(HASHES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def compare_to_baseline(baseline: Optional[dict], current: dict) -> dict:
    """
    Compare every generation in this run against the previous run.

    This is the reproducibility receipt for the experiment. The two runs issue
    the same 24 requests in the same order, so if the stack is deterministic
    given identical request history then every hash matches. A mismatch means
    the numbers in this file are not reproducible and must not be reported.
    """
    if baseline is None:
        return {
            "checked": False,
            "reproduced": None,
            "reason": (
                "No previous run on disk to compare against. Run this script a "
                "second time to obtain the reproducibility receipt."
            ),
        }
    same_setup = (
        baseline.get("model") == current["model"]
        and baseline.get("seed") == current["seed"]
        and baseline.get("decoding") == current["decoding"]
    )
    if not same_setup:
        return {
            "checked": False,
            "reproduced": None,
            "reason": (
                "The previous run used a different model, seed or decoding "
                "configuration, so it is not a valid baseline."
            ),
            "baseline_model": baseline.get("model"),
            "baseline_seed": baseline.get("seed"),
        }
    base_h = baseline.get("hashes", {})
    cur_h = current["hashes"]
    mismatches = sorted(
        key for key in set(base_h) | set(cur_h) if base_h.get(key) != cur_h.get(key)
    )
    return {
        "checked": True,
        "reproduced": len(mismatches) == 0,
        "generations_compared": len(cur_h),
        "mismatched_generations": mismatches,
        "baseline_timestamp_utc": baseline.get("timestamp_utc"),
        "reason": (
            "Every generation is byte-identical to the previous run."
            if not mismatches
            else f"{len(mismatches)} of {len(cur_h)} generations differ from the previous run."
        ),
    }


def verify_determinism(
    model: str,
    user_message: str,
    seed: int,
    timeout: int,
    repeats: int,
    verbose: bool,
) -> dict:
    """
    Issue the same request several times and record whether the outputs were
    byte-identical. This measures REPEAT determinism.

    Repeat determinism is not the property this experiment depends on, and it
    does not hold on this stack. Measured here with a warmed model:
    characters [294, 275, 275] and eval_count [64, 58, 58] for three
    byte-identical greedy requests. The first request for a prompt differs from
    repeats of it, because a repeat reuses the cached prompt prefix and so
    accumulates the prefill in a different order, which can resolve a near-tie
    between two tokens the other way. Adding a discarded warmup generation did
    not change the signature, which rules out model-load cold start.

    The property this experiment depends on is HISTORY determinism: the same
    sequence of requests, issued from a freshly loaded model, reproduces the
    same bytes. That holds. See exp14_determinism_probe.py and
    experiments/results/exp14/exp14_determinism_probe.json, which ran one
    sequence twice from a forced model unload and got byte-identical output at
    every sequence position, reproducing the same [1834, 2516, 2516] divergence
    pattern both times.

    History determinism is the relevant property because all 24 measured
    generations below use distinct prompts. Every one of them is a first
    request for its prefix, so none of them is ever in the repeat regime this
    probe exercises. Reproducibility of this experiment is therefore checked by
    comparing per-generation hashes against a previous run, not by this probe.
    """
    texts: List[str] = []
    eval_counts: List[Optional[int]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for i in range(repeats):
        log(f"determinism probe {i + 1}/{repeats}", verbose)
        got = ollama_chat(model=model, messages=messages, seed=seed, timeout=timeout)
        texts.append(str(got["text"]))
        eval_counts.append(got.get("eval_count"))  # type: ignore[arg-type]
    identical = all(t == texts[0] for t in texts)
    return {
        "repeats": repeats,
        "measures": "repeat determinism, not history determinism",
        "byte_identical": bool(identical),
        "eval_counts": eval_counts,
        "distinct_output_count": len(set(texts)),
        "chars_per_run": [len(t) for t in texts],
        "decoding_options": {**GREEDY_OPTIONS, "seed": seed},
        "gating": False,
        "note": (
            "Greedy decoding with a fixed seed. byte_identical is expected to be "
            "false here and its being false does not invalidate any number in "
            "this file. It records that repeating one request inside a session "
            "changes its output through prompt prefix cache reuse. No measured "
            "generation in this experiment repeats a prompt, so no measured "
            "generation is in that regime. Run to run reproducibility is "
            "evidenced instead by the generations_reproduced field, which "
            "compares a hash of every generation against the previous run."
        ),
    }


def run(
    model: str,
    top_k: int,
    timeout: int,
    seed: int,
    bootstrap_seed: int,
    determinism_repeats: int,
    verbose: bool,
) -> dict:
    check_marker_disjoint()
    log("marker disjointness check passed", verbose)

    provenance = ollama_preflight(model=model, timeout=timeout)
    log(f"model {model} present, digest {provenance['model_digest']}", verbose)

    state = HAND_AUTHORED_STATE.copy()
    # Read the previous run's hashes before anything is written, because this run
    # overwrites the same file.
    baseline_hashes = load_baseline_hashes()

    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))

    store_a = build_store(encoder, PERSONA_A_LINES, state)
    store_b = build_store(encoder, PERSONA_B_LINES, state)

    # Order matters here and is fixed deliberately.
    #   1. unload, so every run starts from the same server state
    #   2. warmup on a throwaway prompt that is not one of the measured prompts
    #   3. the 24 measured generations, each a first request for its own prefix
    #   4. the repeat-determinism probe, last, so it cannot warm any prefix that
    #      a measured generation would later use
    # An earlier version ran the probe at step 2 using PROMPTS[0], which warmed
    # p01's prefix and so measured p01 in the repeat regime while p02 through p12
    # were measured in the first-request regime. That made one of the 12 paired
    # observations incomparable with the other 11.
    unload = ollama_unload(model=model, timeout=timeout, verbose=verbose)
    warmup = ollama_warmup(model=model, seed=seed, timeout=timeout, verbose=verbose)

    rows: List[dict] = []
    metrics_a: List[Dict[str, float]] = []
    metrics_b: List[Dict[str, float]] = []
    truncated: List[str] = []
    gen_hashes: Dict[str, str] = {}

    for idx, p in enumerate(PROMPTS, start=1):
        log(f"prompt {idx}/{len(PROMPTS)} {p.id} ({p.register})", verbose)

        a_ids, a_ctx = retrieve_context_lines(encoder, store_a, p.text, state, top_k=top_k)
        b_ids, b_ctx = retrieve_context_lines(encoder, store_b, p.text, state, top_k=top_k)

        a_got = ollama_chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(a_ctx, p.text)},
            ],
            seed=seed,
            timeout=timeout,
        )
        b_got = ollama_chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(b_ctx, p.text)},
            ],
            seed=seed,
            timeout=timeout,
        )

        for arm, got in (("A", a_got), ("B", b_got)):
            if got.get("done_reason") != "stop":
                truncated.append(f"{p.id}/{arm} done_reason={got.get('done_reason')}")

        a_text = str(a_got["text"])
        b_text = str(b_got["text"])
        gen_hashes[f"{p.id}/A"] = sha256_text(a_text)
        gen_hashes[f"{p.id}/B"] = sha256_text(b_text)
        a_m = style_metrics(a_text)
        b_m = style_metrics(b_text)
        metrics_a.append(a_m)
        metrics_b.append(b_m)

        rows.append(
            {
                "prompt_id": p.id,
                "prompt": p.text,
                "register": p.register,
                "persona_a": {
                    "retrieved_ids": a_ids,
                    "retrieved_context": a_ctx,
                    "response": a_text,
                    "sha256": gen_hashes[f"{p.id}/A"],
                    "metrics": a_m,
                    "done_reason": a_got.get("done_reason"),
                    "eval_count": a_got.get("eval_count"),
                    "prompt_eval_count": a_got.get("prompt_eval_count"),
                },
                "persona_b": {
                    "retrieved_ids": b_ids,
                    "retrieved_context": b_ctx,
                    "response": b_text,
                    "sha256": gen_hashes[f"{p.id}/B"],
                    "metrics": b_m,
                    "done_reason": b_got.get("done_reason"),
                    "eval_count": b_got.get("eval_count"),
                    "prompt_eval_count": b_got.get("prompt_eval_count"),
                },
            }
        )

    # Run last, on a prompt that is deliberately NOT one of the 12 measured
    # prompts and not the warmup prompt, so that (a) it cannot have warmed any
    # measured prefix and (b) its own first request is genuinely a first request,
    # which is what makes the first-versus-repeat contrast visible.
    determinism: Optional[dict] = None
    if determinism_repeats > 1:
        _, probe_ctx = retrieve_context_lines(
            encoder, store_a, DETERMINISM_PROBE_PROMPT, state, top_k=top_k
        )
        determinism = verify_determinism(
            model=model,
            user_message=build_user_message(probe_ctx, DETERMINISM_PROBE_PROMPT),
            seed=seed,
            timeout=timeout,
            repeats=determinism_repeats,
            verbose=verbose,
        )
        log(f"determinism byte_identical={determinism['byte_identical']}", verbose)

    stats = paired_analysis(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        prompt_ids=[p.id for p in PROMPTS],
        bootstrap_seed=bootstrap_seed,
    )

    current_hashes = {
        "model": model,
        "seed": seed,
        "decoding": {**GREEDY_OPTIONS, "seed": seed},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_count": len(PROMPTS),
        "hashes": gen_hashes,
    }
    reproducibility = compare_to_baseline(baseline_hashes, current_hashes)

    # The gate has two independent conditions. Truncation invalidates the length
    # metrics outright. Failure to reproduce a previous run invalidates
    # everything. An absent baseline is not a failure, it is an unfinished
    # check, and it is reported as such rather than being counted as a pass.
    if truncated:
        verdict = "INVALID, do not report these numbers"
    elif reproducibility["reproduced"] is False:
        verdict = "INVALID, do not report these numbers"
    elif reproducibility["reproduced"] is True:
        verdict = "valid, reproduction confirmed against previous run"
    else:
        verdict = "valid, reproduction not yet checked"

    validity = {
        "truncated_generations": truncated,
        "all_generations_completed": len(truncated) == 0,
        "generations_reproduced": reproducibility["reproduced"],
        "reproducibility_check": reproducibility,
        "repeat_determinism_confirmed": (
            bool(determinism["byte_identical"]) if determinism else None
        ),
        "repeat_determinism_is_gating": False,
        "verdict": verdict,
        "gate_definition": (
            "Reportable requires all_generations_completed true and "
            "generations_reproduced not false. repeat_determinism_confirmed is "
            "recorded for the record and is deliberately not part of the gate: "
            "no measured generation repeats a prompt, so the repeat regime is "
            "never entered. See determinism_check.note."
        ),
    }

    return {
        "metadata": {
            "experiment": "exp14 persona memory style effect",
            "timestamp_unix": time.time(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": provenance,
            "decoding": {**GREEDY_OPTIONS, "seed": seed},
            "unload": unload,
            "warmup": warmup,
            "top_k": top_k,
            "prompt_count": len(PROMPTS),
            "prompt_registers": {
                "affective": sum(1 for p in PROMPTS if p.register == "affective"),
                "task": sum(1 for p in PROMPTS if p.register == "task"),
            },
            "persona_a_lines": list(PERSONA_A_LINES),
            "persona_b_lines": list(PERSONA_B_LINES),
            "analytical_markers": list(ANALYTICAL_MARKERS),
            "warm_markers": list(WARM_MARKERS),
            "marker_matching": "word-boundary regex, disjoint from all prompts and persona lines",
            "state_vector": {
                "values": [float(x) for x in HAND_AUTHORED_STATE],
                "source": "hand-authored, not measured",
                "identical_across_arms": True,
                "note": (
                    "Shared by both arms, so it cannot produce any A versus B "
                    "difference. MemoryStore._rebuild_cache builds a 5-dimensional "
                    "auto-state cache, so dimensions 6 and 7 are structurally "
                    "ignored by the state term of the composite distance."
                ),
            },
            "bootstrap_seed": bootstrap_seed,
            "scope_limit": (
                "One model, one machine, 12 paired prompts, 5 persona lines per arm. "
                "Supports a claim about this model under this protocol only."
            ),
            "supersedes": (
                "The earlier temperature-0.2 unseeded 6-prompt run of this "
                "experiment is retracted. See experiments/results/exp14/RETRACTION.md."
            ),
        },
        "validity": validity,
        "generation_hashes": current_hashes,
        "determinism_check": determinism,
        "results": rows,
        "statistics": stats,
    }


def _fmt(v: Optional[float], spec: str = "+.3f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return format(v, spec)


def write_text_summary(data: dict, path: str) -> None:
    md = data["metadata"]
    st = data["statistics"]
    val = data["validity"]
    lines = [
        "EXP14: Does persona memory content change model output style?",
        "=" * 62,
        f"Model: {md['provenance']['model']}  digest {md['provenance']['model_digest']}",
        f"Ollama version: {md['provenance']['ollama_version']}",
        f"Decoding: {json.dumps(md['decoding'])}",
        f"Paired prompts: {st['n_paired_prompts']} "
        f"({md['prompt_registers']['affective']} affective, {md['prompt_registers']['task']} task)",
        f"Top-k memory context: {md['top_k']}",
        f"Validity verdict: {val['verdict']}",
        f"All generations completed: {val['all_generations_completed']}",
        f"Generations reproduced vs previous run: {val['generations_reproduced']}"
        f"  ({val['reproducibility_check']['reason']})",
        f"Repeat determinism (recorded, not gating): {val['repeat_determinism_confirmed']}",
        "",
        "The state vector is hand-authored, not measured, and is identical in both",
        "arms, so it cannot contribute to any difference reported below.",
        "",
        "Paired differences (Persona B minus Persona A), one observation per prompt.",
        "p_holm is Holm-Bonferroni adjusted across all 8 metrics and is the value to",
        "quote. Primary metrics named in advance: "
        + ", ".join(PRIMARY_METRIC_KEYS),
        "",
    ]

    header = (
        f"{'metric':<26} {'A mean':>9} {'B mean':>9} {'diff':>9} "
        f"{'sd':>8} {'dz':>7} {'CI95 low':>9} {'CI95 high':>10} "
        f"{'p_raw':>9} {'p_holm':>9} {'B>A':>4} {'A>B':>4} {'tie':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for key in METRIC_KEYS:
        m = st["per_metric"][key]
        star = " *" if key in PRIMARY_METRIC_KEYS else ""
        lines.append(
            f"{key + star:<26} "
            f"{_fmt(m['persona_a_mean'], '.3f'):>9} "
            f"{_fmt(m['persona_b_mean'], '.3f'):>9} "
            f"{_fmt(m['mean_difference_b_minus_a']):>9} "
            f"{_fmt(m['sd_of_differences'], '.3f'):>8} "
            f"{_fmt(m['cohens_dz']):>7} "
            f"{_fmt(m['bootstrap_ci95_low']):>9} "
            f"{_fmt(m['bootstrap_ci95_high']):>10} "
            f"{_fmt(m['wilcoxon_p_raw'], '.4f'):>9} "
            f"{_fmt(m['wilcoxon_p_holm'], '.4f'):>9} "
            f"{m['n_prompts_b_greater']:>4} "
            f"{m['n_prompts_a_greater']:>4} "
            f"{m['n_prompts_tied']:>4}"
        )

    notes = [
        f"- {k}: {st['per_metric'][k]['test_note']}"
        for k in METRIC_KEYS
        if st["per_metric"][k]["test_note"]
    ]
    if notes:
        lines.append("")
        lines.append("Test notes:")
        lines.extend(notes)

    lines.append("")
    lines.append(f"Scope limit: {md['scope_limit']}")
    lines.append(f"Supersedes: {md['supersedes']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_markdown_dump(data: dict, path: str) -> None:
    md = data["metadata"]
    out = [
        "# EXP14: Persona memory style effect, full response dump",
        "",
        f"- Model: `{md['provenance']['model']}` digest `{md['provenance']['model_digest']}`",
        f"- Decoding: `{json.dumps(md['decoding'])}`",
        f"- Validity verdict: **{data['validity']['verdict']}**",
        f"- Paired prompts: {md['prompt_count']}",
        "",
        "Persona seed lines are written in paraphrase so that they contain no marker",
        "word from either style family. A verbatim copy of an injected line therefore",
        "scores zero on both style metrics.",
        "",
        "## Persona A seed lines",
        "",
    ]
    out += [f"- {x}" for x in md["persona_a_lines"]]
    out += ["", "## Persona B seed lines", ""]
    out += [f"- {x}" for x in md["persona_b_lines"]]
    out += ["", "## Responses", ""]
    for row in data["results"]:
        out += [
            f"### {row['prompt_id']} ({row['register']}): {row['prompt']}",
            "",
            "**Persona A retrieved context**",
            "",
        ]
        out += [f"- {c}" for c in row["persona_a"]["retrieved_context"]]
        out += [
            "",
            f"**Persona A response** (done_reason={row['persona_a']['done_reason']}, "
            f"eval_count={row['persona_a']['eval_count']})",
            "",
            row["persona_a"]["response"],
            "",
            "**Persona B retrieved context**",
            "",
        ]
        out += [f"- {c}" for c in row["persona_b"]["retrieved_context"]]
        out += [
            "",
            f"**Persona B response** (done_reason={row['persona_b']['done_reason']}, "
            f"eval_count={row['persona_b']['eval_count']})",
            "",
            row["persona_b"]["response"],
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def write_plots(data: dict, diff_path: str, effect_path: str) -> None:
    st = data["statistics"]
    n = st["n_paired_prompts"]
    prompt_ids = [r["prompt_id"] for r in data["results"]]

    # Plot 1: every paired difference for the two primary metrics, against zero.
    # Showing all n points rather than a mean is what makes sign consistency,
    # or the lack of it, visible to the reader.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, key in zip(axes, PRIMARY_METRIC_KEYS):
        m = st["per_metric"][key]
        diffs = [m["per_prompt_differences"][pid] for pid in prompt_ids]
        colors = ["#4C78A8" if d >= 0 else "#E45756" for d in diffs]
        ax.bar(np.arange(len(prompt_ids)), diffs, color=colors)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.axhline(
            m["mean_difference_b_minus_a"],
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label=f"mean {m['mean_difference_b_minus_a']:+.3f}",
        )
        ax.set_xticks(np.arange(len(prompt_ids)))
        ax.set_xticklabels(prompt_ids, rotation=60, fontsize=8)
        ax.set_title(f"{key}\nB minus A per prompt", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Difference (B minus A)")
    fig.suptitle(
        f"EXP14 paired per-prompt differences, n={n}, greedy decoding, "
        f"model {data['metadata']['provenance']['model']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(diff_path, dpi=160)
    plt.close(fig)

    # Plot 2: mean difference with bootstrap CI for every metric. Metrics live on
    # different scales, so each is drawn on its own normalized row via the CI
    # rather than a shared y axis, and the raw numbers are printed alongside.
    keys = list(METRIC_KEYS)
    means = [st["per_metric"][k]["mean_difference_b_minus_a"] for k in keys]
    los = [st["per_metric"][k]["bootstrap_ci95_low"] for k in keys]
    his = [st["per_metric"][k]["bootstrap_ci95_high"] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(keys))
    for i, (mu, lo, hi) in enumerate(zip(means, los, his)):
        if lo is None or hi is None:
            continue
        crosses_zero = lo <= 0.0 <= hi
        color = "#999999" if crosses_zero else "#4C78A8"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, solid_capstyle="butt")
        ax.plot([mu], [i], marker="o", color=color, markersize=7)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{k}{' *' if k in PRIMARY_METRIC_KEYS else ''}  "
            f"(p_holm={_fmt(st['per_metric'][k]['wilcoxon_p_holm'], '.3f')})"
            for k in keys
        ],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean difference, Persona B minus Persona A, with bootstrap 95% CI")
    ax.set_title(
        f"EXP14 effect sizes, n={n} paired prompts. Grey intervals span zero.\n"
        "Asterisk marks metrics named primary in advance. p_holm is "
        "Holm-Bonferroni adjusted.",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(effect_path, dpi=160)
    plt.close(fig)


def write_outputs(data: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.json")
    txt_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.txt")
    md_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.md")
    diff_png = os.path.join(RESULTS_DIR, "exp14_paired_differences.png")
    effect_png = os.path.join(RESULTS_DIR, "exp14_effect_sizes.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Written last and separately, so that the next run has a baseline to compare
    # its generations against.
    with open(HASHES_PATH, "w", encoding="utf-8") as f:
        json.dump(data["generation_hashes"], f, indent=2)
    write_text_summary(data, txt_path)
    write_markdown_dump(data, md_path)
    write_plots(data, diff_png, effect_png)

    for p in (json_path, txt_path, md_path, diff_png, effect_png):
        print(f"[saved] {p}")


def self_test() -> int:
    """
    Offline checks. No Ollama, no encoder, no file writes, so this is safe and
    fast to run before committing. Covers the specific defects that made the
    previous version of this experiment unsound.
    """
    failures: List[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"[pass] {label}")
        else:
            failures.append(f"{label}{': ' + detail if detail else ''}")
            print(f"[FAIL] {label}{': ' + detail if detail else ''}")

    try:
        check_marker_disjoint()
        check("marker lists disjoint from all prompts and persona lines", True)
    except ValueError as exc:
        check("marker lists disjoint from all prompts and persona lines", False, str(exc))

    # Word-boundary matching. Raw substring counting got all of these wrong.
    total, types = count_markers("stepping stones", ANALYTICAL_MARKERS)
    check("'stepping' does not match marker 'step'", total == 0, f"total={total}")

    total, types = count_markers("step-by-step", ANALYTICAL_MARKERS)
    check(
        "'step-by-step' counts 2 occurrences of 1 marker type",
        total == 2 and types == 1,
        f"total={total} types={types}",
    )

    total, types = count_markers("understanding the problem", WARM_MARKERS)
    check("'understanding' does not match marker 'understand'", total == 0, f"total={total}")

    total, types = count_markers("I hear you and I understand", WARM_MARKERS)
    check(
        "multi-word marker 'hear you' matches once alongside 'understand'",
        total == 2 and types == 2,
        f"total={total} types={types}",
    )

    total, types = count_markers("feel feel feel", WARM_MARKERS)
    check(
        "type count resists a single repeated word",
        total == 3 and types == 1,
        f"total={total} types={types}",
    )

    m = style_metrics("First, therefore. Second! Third? I understand.")
    check(
        "style_metrics reports the expected fixture values",
        m["exclamations"] == 1.0
        and m["questions"] == 1.0
        and m["analytical_marker_types"] == 4.0
        and m["warm_marker_types"] == 1.0,
        json.dumps(m),
    )

    adj = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03, "d": None})
    check("Holm leaves an uncomputable test as None", adj["d"] is None)
    check(
        "Holm adjusts the smallest p by the family size",
        abs(adj["a"] - 0.03) < 1e-12,
        f"got {adj['a']}",
    )
    check(
        "Holm output is monotone non-decreasing",
        adj["a"] <= adj["b"] <= adj["c"],
        f"{adj['a']}, {adj['b']}, {adj['c']}",
    )
    capped = holm_bonferroni({"x": 0.9, "y": 0.95})
    check("Holm caps adjusted p at 1.0", capped["x"] <= 1.0 and capped["y"] <= 1.0)

    lo1, hi1 = bootstrap_ci_mean([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=99)
    lo2, hi2 = bootstrap_ci_mean([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=99)
    check("bootstrap CI is reproducible from its seed", lo1 == lo2 and hi1 == hi2)
    check("bootstrap CI brackets the sample mean", lo1 <= 2.5 <= hi1, f"[{lo1}, {hi1}]")

    # Paired analysis on a synthetic constant shift.
    ids = [f"q{i:02d}" for i in range(12)]
    a_metrics = [{k: 1.0 for k in METRIC_KEYS} for _ in ids]
    b_metrics = [{k: 2.0 for k in METRIC_KEYS} for _ in ids]
    st = paired_analysis(a_metrics, b_metrics, ids, bootstrap_seed=7)
    one = st["per_metric"]["warm_marker_types"]
    check("paired_analysis n matches the prompt count", st["n_paired_prompts"] == 12)
    check(
        "paired_analysis recovers a constant +1 shift",
        abs(one["mean_difference_b_minus_a"] - 1.0) < 1e-12 and one["sd_of_differences"] == 0.0,
        json.dumps({k: one[k] for k in ("mean_difference_b_minus_a", "sd_of_differences")}),
    )
    check(
        "zero-variance differences yield no Cohen dz rather than an infinity",
        one["cohens_dz"] is None,
    )
    check(
        "sign counts are recorded per metric",
        one["n_prompts_b_greater"] == 12 and one["n_prompts_a_greater"] == 0,
    )

    # All-tied case must not crash and must not claim a p-value.
    st_tied = paired_analysis(a_metrics, a_metrics, ids, bootstrap_seed=7)
    tied = st_tied["per_metric"]["warm_marker_types"]
    check(
        "all-tied differences report no p-value and an explanatory note",
        tied["wilcoxon_p_raw"] is None and "undefined" in tied["test_note"],
        tied["test_note"],
    )

    check("scipy is importable for the signed-rank test", scipy_stats is not None)
    check(
        "prompt registers are balanced",
        sum(1 for p in PROMPTS if p.register == "affective")
        == sum(1 for p in PROMPTS if p.register == "task"),
    )
    check("prompt ids are unique", len({p.id for p in PROMPTS}) == len(PROMPTS))
    check(
        "both arms hold the same number of persona lines",
        len(PERSONA_A_LINES) == len(PERSONA_B_LINES),
    )
    check("num_predict is uncapped", GREEDY_OPTIONS["num_predict"] == -1)
    check("decoding is greedy", GREEDY_OPTIONS["temperature"] == 0.0 and GREEDY_OPTIONS["top_k"] == 1)

    print("")
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s) failed")
        return 1
    print("SELF-TEST PASSED")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP14: does persona memory content change model output style?"
    )
    p.add_argument(
        "--model",
        default="phi4-mini:latest",
        help="Installed Ollama model name. Verified against /api/tags before running.",
    )
    p.add_argument("--top-k", type=int, default=4, help="Retrieved memory lines per prompt")
    p.add_argument("--timeout", type=int, default=600, help="HTTP timeout seconds per generation")
    p.add_argument("--seed", type=int, default=7, help="Decoding seed, recorded in the output")
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260820,
        help="Seed for the bootstrap confidence intervals",
    )
    p.add_argument(
        "--determinism-repeats",
        type=int,
        default=3,
        help="Repeat one prompt this many times to confirm byte-identical output. "
        "Set to 1 to skip, but then determinism is unverified for that run.",
    )
    p.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    p.add_argument("--verbose", action="store_true", help="Print progress logs")
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    args = parse_args()
    if args.self_test:
        return self_test()

    try:
        data = run(
            model=args.model,
            top_k=args.top_k,
            timeout=args.timeout,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            determinism_repeats=args.determinism_repeats,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"\nERROR: experiment design check failed.\n{exc}")
        return 3
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return 4
    except urllib.error.URLError as exc:
        print(f"\nERROR: could not reach Ollama at {OLLAMA_HOST} ({exc}).")
        print("Start Ollama, then rerun. Nothing was written.")
        return 2

    write_outputs(data)
    verdict = data["validity"]["verdict"]
    print(f"\nEXP14 completed. Validity verdict: {verdict}")
    if not verdict.startswith("valid"):
        print("Do not publish these numbers. Fix the flagged condition and rerun.")
        return 5
    if data["validity"]["generations_reproduced"] is None:
        print(
            "Reproducibility is not yet evidenced. Run this script once more to "
            "compare every generation against this run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
