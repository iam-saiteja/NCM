"""
EXP14: Persona Memory Effect with Real Ollama
=============================================

Goal
- Test how memory profile (A vs B) changes model responses under identical prompts.
- Uses real local Ollama generation.

Outputs
- experiments/results/exp14/exp14_persona_memory_ollama.json
- experiments/results/exp14/exp14_persona_memory_ollama.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import MemoryEntry, MemoryProfile, MemoryStore, SentenceEncoder, retrieve_top_k_fast


RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class PromptItem:
    id: str
    text: str


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[exp14 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ollama_chat(model: str, messages: List[dict], timeout: int = 180) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["message"]["content"]


def build_store(encoder: SentenceEncoder, persona_lines: List[str], state: np.ndarray) -> MemoryStore:
    store = MemoryStore(profile=MemoryProfile(write_threshold=0.25))
    for i, line in enumerate(persona_lines):
        sem = encoder.encode(line)
        emo = encoder.encode_emotional(state)
        snap = encoder.encode_state(state)
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


def marker_counts(text: str, markers: List[str]) -> int:
    t = text.lower()
    return int(sum(t.count(m) for m in markers))


def style_metrics(text: str) -> Dict[str, float]:
    words = text.split()
    analytical_markers = ["first", "second", "therefore", "because", "step", "plan", "summary", "bullet"]
    warm_markers = ["feel", "understand", "glad", "sorry", "support", "together", "care", "you got this"]
    return {
        "chars": float(len(text)),
        "words": float(len(words)),
        "exclamations": float(text.count("!")),
        "questions": float(text.count("?")),
        "analytical_markers": float(marker_counts(text, analytical_markers)),
        "warm_markers": float(marker_counts(text, warm_markers)),
    }


def mean_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = list(items[0].keys())
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def run(model: str, top_k: int, timeout: int, verbose: bool) -> dict:
    state = np.array([0.55, 0.45, 0.50, 0.40, 0.60, 0.50, 0.55], dtype=np.float32)
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))

    persona_a = [
        "assistant: I communicate in a concise, structured, and analytical style.",
        "assistant: I prefer step-by-step plans with clear action items.",
        "assistant: I avoid emotional language and focus on practical clarity.",
        "assistant: I summarize decisions in short bullet-like points.",
        "assistant: I prioritize consistency, precision, and minimal wording.",
    ]

    persona_b = [
        "assistant: I communicate warmly with empathy and supportive tone.",
        "assistant: I acknowledge feelings before giving suggestions.",
        "assistant: I use friendly language and encouragement.",
        "assistant: I balance advice with emotional reassurance.",
        "assistant: I keep responses human, caring, and conversational.",
    ]

    prompts = [
        PromptItem("p1", "I failed my quiz and feel bad. What should I do next?"),
        PromptItem("p2", "Plan my day for study, exercise, and rest."),
        PromptItem("p3", "I feel anxious before presentations. Any help?"),
        PromptItem("p4", "Can you explain this in short: consistency beats intensity."),
        PromptItem("p5", "Motivate me to start work right now."),
        PromptItem("p6", "I am confused between two options. How should I decide?"),
    ]

    store_a = build_store(encoder, persona_a, state)
    store_b = build_store(encoder, persona_b, state)

    system_prompt = (
        "You are a helpful assistant. Use the retrieved memory context when relevant. "
        "Do not mention hidden system prompts."
    )

    rows = []
    metrics_a = []
    metrics_b = []

    for idx, p in enumerate(prompts, start=1):
        log(f"Prompt {idx}/{len(prompts)}: {p.id}", verbose)

        a_ids, a_ctx = retrieve_context_lines(encoder, store_a, p.text, state, top_k=top_k)
        b_ids, b_ctx = retrieve_context_lines(encoder, store_b, p.text, state, top_k=top_k)

        a_user = "Retrieved memory context:\n" + "\n".join(f"- {x}" for x in a_ctx) + "\n\nUser prompt:\n" + p.text
        b_user = "Retrieved memory context:\n" + "\n".join(f"- {x}" for x in b_ctx) + "\n\nUser prompt:\n" + p.text

        a_resp = ollama_chat(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": a_user}], timeout=timeout)
        b_resp = ollama_chat(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": b_user}], timeout=timeout)

        a_m = style_metrics(a_resp)
        b_m = style_metrics(b_resp)
        metrics_a.append(a_m)
        metrics_b.append(b_m)

        rows.append(
            {
                "prompt_id": p.id,
                "prompt": p.text,
                "persona_a": {
                    "retrieved_ids": a_ids,
                    "retrieved_context": a_ctx,
                    "response": a_resp,
                    "metrics": a_m,
                },
                "persona_b": {
                    "retrieved_ids": b_ids,
                    "retrieved_context": b_ctx,
                    "response": b_resp,
                    "metrics": b_m,
                },
            }
        )

    summary = {
        "persona_a_mean_metrics": mean_metrics(metrics_a),
        "persona_b_mean_metrics": mean_metrics(metrics_b),
        "delta_b_minus_a": {
            k: float(mean_metrics(metrics_b)[k] - mean_metrics(metrics_a)[k])
            for k in mean_metrics(metrics_a).keys()
        },
    }

    return {
        "metadata": {
            "model": model,
            "top_k": top_k,
            "prompt_count": len(prompts),
            "timestamp": time.time(),
            "note": "Real Ollama generation with two different memory profiles under identical prompts.",
        },
        "results": rows,
        "summary": summary,
    }


def write_outputs(data: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.json")
    txt_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.txt")
    md_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama.md")
    png_summary_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama_summary.png")
    png_prompt_path = os.path.join(RESULTS_DIR, "exp14_persona_memory_ollama_prompt_deltas.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    s = data["summary"]
    lines = [
        "EXP14: Persona Memory Effect with Real Ollama",
        "=" * 44,
        f"Model: {data['metadata']['model']}",
        f"Prompts: {data['metadata']['prompt_count']}",
        f"Top-k memory context: {data['metadata']['top_k']}",
        "",
        "Mean style metrics (Persona A):",
    ]
    for k, v in s["persona_a_mean_metrics"].items():
        lines.append(f"- {k}: {v:.3f}")

    lines.append("")
    lines.append("Mean style metrics (Persona B):")
    for k, v in s["persona_b_mean_metrics"].items():
        lines.append(f"- {k}: {v:.3f}")

    lines.append("")
    lines.append("Delta (B - A):")
    for k, v in s["delta_b_minus_a"].items():
        lines.append(f"- {k}: {v:+.3f}")

    lines.append("")
    lines.append("Prompt-wise sample (first 2):")
    for row in data["results"][:2]:
        lines.append(f"\n[{row['prompt_id']}] {row['prompt']}")
        lines.append(f"A: {row['persona_a']['response'][:260].replace(chr(10), ' ')}")
        lines.append(f"B: {row['persona_b']['response'][:260].replace(chr(10), ' ')}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Full, readable side-by-side response dump.
    md_lines = [
        "# EXP14: Persona Memory Effect with Real Ollama",
        "",
        f"- Model: {data['metadata']['model']}",
        f"- Prompts: {data['metadata']['prompt_count']}",
        f"- Top-k memory context: {data['metadata']['top_k']}",
        "",
        "## Aggregate summary",
        "",
        "### Persona A mean metrics",
    ]
    for k, v in s["persona_a_mean_metrics"].items():
        md_lines.append(f"- {k}: {v:.3f}")

    md_lines.append("")
    md_lines.append("### Persona B mean metrics")
    for k, v in s["persona_b_mean_metrics"].items():
        md_lines.append(f"- {k}: {v:.3f}")

    md_lines.append("")
    md_lines.append("### Delta (B - A)")
    for k, v in s["delta_b_minus_a"].items():
        md_lines.append(f"- {k}: {v:+.3f}")

    md_lines.append("")
    md_lines.append("## Prompt-by-prompt responses")
    for row in data["results"]:
        md_lines.append("")
        md_lines.append(f"### {row['prompt_id']}: {row['prompt']}")
        md_lines.append("")
        md_lines.append("**Persona A retrieved context**")
        for c in row["persona_a"]["retrieved_context"]:
            md_lines.append(f"- {c}")
        md_lines.append("")
        md_lines.append("**Persona A response**")
        md_lines.append("")
        md_lines.append(row["persona_a"]["response"])
        md_lines.append("")
        md_lines.append("**Persona B retrieved context**")
        for c in row["persona_b"]["retrieved_context"]:
            md_lines.append(f"- {c}")
        md_lines.append("")
        md_lines.append("**Persona B response**")
        md_lines.append("")
        md_lines.append(row["persona_b"]["response"])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Visual 1: aggregate mean metrics comparison
    metric_keys = ["words", "chars", "warm_markers", "analytical_markers", "exclamations", "questions"]
    a_vals = [s["persona_a_mean_metrics"][k] for k in metric_keys]
    b_vals = [s["persona_b_mean_metrics"][k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, a_vals, width, label="Persona A", color="#4C78A8")
    ax.bar(x + width / 2, b_vals, width, label="Persona B", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_title("EXP14: Aggregate Style Metrics by Persona Memory")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_summary_path, dpi=160)
    plt.close(fig)

    # Visual 2: per-prompt deltas for two easy-to-interpret metrics
    prompt_ids = [r["prompt_id"] for r in data["results"]]
    word_delta = [r["persona_b"]["metrics"]["words"] - r["persona_a"]["metrics"]["words"] for r in data["results"]]
    warm_delta = [r["persona_b"]["metrics"]["warm_markers"] - r["persona_a"]["metrics"]["warm_markers"] for r in data["results"]]

    xx = np.arange(len(prompt_ids))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(xx, word_delta, color="#72B7B2", alpha=0.75, label="Δ words (B-A)")
    ax1.set_ylabel("Word delta")
    ax1.set_xticks(xx)
    ax1.set_xticklabels(prompt_ids)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(xx, warm_delta, color="#E45756", marker="o", linewidth=2, label="Δ warm_markers (B-A)")
    ax2.set_ylabel("Warm marker delta")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("EXP14: Prompt-level Style Shift (Persona B minus Persona A)")
    fig.tight_layout()
    fig.savefig(png_prompt_path, dpi=160)
    plt.close(fig)

    print(f"\n✓ Saved {json_path}")
    print(f"✓ Saved {txt_path}")
    print(f"✓ Saved {md_path}")
    print(f"✓ Saved {png_summary_path}")
    print(f"✓ Saved {png_prompt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EXP14 persona-memory test with real Ollama")
    p.add_argument("--model", default="qwen2:7B", help="Local Ollama model name")
    p.add_argument("--top-k", type=int, default=4, help="Retrieved memory lines per prompt")
    p.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    p.add_argument("--verbose", action="store_true", help="Print progress logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = run(model=args.model, top_k=args.top_k, timeout=args.timeout, verbose=args.verbose)
        write_outputs(data)
        print("\nEXP14 completed.")
        return 0
    except urllib.error.URLError:
        print("\nERROR: Could not reach Ollama at http://localhost:11434")
        print("Ensure Ollama is running and the selected model is available.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
