"""
EXP14 supporting diagnostic: what kind of determinism does local greedy
decoding actually have?

EXP14's validity gate issues one request three times and requires the three
outputs to be byte-identical. That gate fails on this machine with a stable
signature: characters [294, 275, 275] and eval_count [64, 58, 58]. The first
request differs, the second and third are bit-identical to each other. Adding a
discarded warmup generation before the probes did not change the signature at
all, which rules out model-load cold start as the cause.

The remaining candidate is prompt prefix caching. The first request for a novel
prompt is prefilled in one large batch. A repeat of the same prompt reuses the
cached key/value tensors and prefills little or nothing, so the floating point
reduction order differs, and a near-tie between two tokens can resolve the other
way.

If that is the cause, the divergence is not noise. It is a deterministic
function of request history. This probe tests that claim the only way it can be
tested: it runs the same sequence of requests twice, from a forced-unloaded
model both times, and asks whether sequence position i in run one is
byte-identical to sequence position i in run two.

The distinction matters for what EXP14 is allowed to claim:

  repeat determinism    same request twice in one session gives the same bytes
  history determinism   the same sequence of requests gives the same bytes

EXP14 needs history determinism, because rerunning the experiment reissues the
same 24 requests in the same order. It does not need repeat determinism, which
is a property no measured EXP14 generation ever relies on: all 24 prompts are
distinct, so every measured generation is a first-request-for-this-prefix.

Usage:
    venv/Scripts/python.exe experiments/python/exp14_determinism_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

GREEDY_OPTIONS = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "num_predict": -1}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user directly and concisely."
)

# A prompt long enough that the prefill stage does real work, so that a cached
# prefix and an uncached one take measurably different code paths.
PROBE_PROMPT = (
    "I have been trying to keep a consistent routine for the last few weeks, "
    "but between work deadlines and family commitments I keep falling behind "
    "on the parts that matter to me. Tell me how you would think about "
    "rebuilding a routine from scratch when the previous one has already "
    "failed several times."
)

REPEATS = 3
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "exp14"


def log(message: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {message}", flush=True)


def chat(model: str, prompt: str, seed: int, timeout: int, keep_alive=None) -> dict:
    """One greedy chat completion. keep_alive=0 unloads the model afterwards."""
    options = dict(GREEDY_OPTIONS)
    options["seed"] = seed
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": options,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body.get("message", {}).get("content", "")
    return {
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "eval_count": body.get("eval_count"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "done_reason": body.get("done_reason"),
    }


def unload(model: str, timeout: int) -> bool:
    """Force the model out of memory so the next request loads it fresh."""
    try:
        chat(model, "hi", seed=1, timeout=timeout, keep_alive=0)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log(f"unload request failed: {exc}")
        return False


def run_sequence(model: str, seed: int, timeout: int, label: str) -> list:
    """Unload, then issue the identical probe request REPEATS times in order."""
    log(f"{label}: forcing model unload")
    unloaded = unload(model, timeout)
    if not unloaded:
        raise RuntimeError("could not unload model; probe would not be clean")
    time.sleep(2.0)
    results = []
    for index in range(REPEATS):
        log(f"{label}: request {index + 1}/{REPEATS}")
        results.append(chat(model, PROBE_PROMPT, seed=seed, timeout=timeout))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phi4-mini:latest")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    run_a = run_sequence(args.model, args.seed, args.timeout, "run A")
    run_b = run_sequence(args.model, args.seed, args.timeout, "run B")

    # Repeat determinism: are the REPEATS outputs within one run all the same?
    repeat_ok_a = len({r["sha256"] for r in run_a}) == 1
    repeat_ok_b = len({r["sha256"] for r in run_b}) == 1

    # History determinism: does position i of run A match position i of run B?
    position_match = [a["sha256"] == b["sha256"] for a, b in zip(run_a, run_b)]
    history_ok = all(position_match)

    report = {
        "experiment": "exp14_determinism_probe",
        "purpose": (
            "Distinguish repeat determinism (same request twice in one session) "
            "from history determinism (same request sequence twice, from a "
            "forced-unloaded model). EXP14 reproducibility depends on the "
            "second, not the first."
        ),
        "model": args.model,
        "seed": args.seed,
        "decoding": {**GREEDY_OPTIONS, "seed": args.seed},
        "repeats_per_run": REPEATS,
        "run_a": [{k: v for k, v in r.items() if k != "text"} for r in run_a],
        "run_b": [{k: v for k, v in r.items() if k != "text"} for r in run_b],
        "repeat_determinism_run_a": repeat_ok_a,
        "repeat_determinism_run_b": repeat_ok_b,
        "position_match_a_vs_b": position_match,
        "history_determinism": history_ok,
        "chars_run_a": [r["chars"] for r in run_a],
        "chars_run_b": [r["chars"] for r in run_b],
        "eval_counts_run_a": [r["eval_count"] for r in run_a],
        "eval_counts_run_b": [r["eval_count"] for r in run_b],
        "prompt_eval_counts_run_a": [r["prompt_eval_count"] for r in run_a],
        "prompt_eval_counts_run_b": [r["prompt_eval_count"] for r in run_b],
    }

    if history_ok and not repeat_ok_a:
        report["interpretation"] = (
            "Greedy decoding here is deterministic given identical request "
            "history, but not invariant to request history. Repeating one "
            "request inside a session changes its output because the repeat "
            "reuses a cached prompt prefix. Reissuing the same sequence from a "
            "clean model reproduces every output exactly. EXP14 is therefore "
            "reproducible run to run, and its repeat-based validity gate was "
            "testing a property the experiment does not use."
        )
    elif history_ok and repeat_ok_a:
        report["interpretation"] = (
            "Both repeat determinism and history determinism hold in this "
            "probe. The EXP14 gate failure is not reproduced here and needs a "
            "different explanation."
        )
    else:
        report["interpretation"] = (
            "History determinism does NOT hold: the same request sequence from "
            "a clean model produced different output. EXP14 numbers are not "
            "reproducible run to run and no style metric from it can be "
            "reported."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp14_determinism_probe.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")

    print("\n" + "=" * 68)
    print(f"chars run A         : {report['chars_run_a']}")
    print(f"chars run B         : {report['chars_run_b']}")
    print(f"eval_count run A    : {report['eval_counts_run_a']}")
    print(f"eval_count run B    : {report['eval_counts_run_b']}")
    print(f"prompt_eval run A   : {report['prompt_eval_counts_run_a']}")
    print(f"prompt_eval run B   : {report['prompt_eval_counts_run_b']}")
    print(f"repeat determinism  : A={repeat_ok_a}  B={repeat_ok_b}")
    print(f"position match A/B  : {position_match}")
    print(f"HISTORY DETERMINISM : {history_ok}")
    print("=" * 68)
    print(report["interpretation"])
    return 0 if history_ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
