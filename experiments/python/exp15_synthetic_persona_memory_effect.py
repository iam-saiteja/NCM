"""
EXP15: Synthetic Persona Memory Effect (Large Scale)
===================================================

Goal
- Quantify how memory profile changes generated response style at scale.
- Use large synthetic data to isolate memory-conditioning signal.

Outputs
- experiments/results/exp15/exp15_synthetic_persona_memory_effect.json
- experiments/results/exp15/exp15_synthetic_persona_memory_effect.txt
- experiments/results/exp15/exp15_synthetic_persona_memory_effect_summary.png
- experiments/results/exp15/exp15_synthetic_persona_memory_effect_clusters.png
- experiments/results/exp15/exp15_synthetic_persona_memory_effect_scale.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class PersonaBank:
    name: str
    memory_sem: np.ndarray      # (M, d_sem)
    memory_style: np.ndarray    # (M, d_style)
    centroid_style: np.ndarray  # (d_style,)


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8
    return x / n


def build_persona_banks(rng: np.random.Generator, memory_size: int, d_sem: int, d_style: int) -> tuple[PersonaBank, PersonaBank]:
    # Distinct style centroids for two personalities
    # A: structured/analytical/compact
    centroid_a = np.array([0.80, 0.15, 0.25, 0.75], dtype=np.float32)
    # B: warm/expressive/conversational
    centroid_b = np.array([0.25, 0.85, 0.80, 0.30], dtype=np.float32)

    # Semantic banks: both broad, with slight center shift to simulate different memory history domains
    sem_center_a = l2_normalize(rng.normal(loc=0.0, scale=1.0, size=(1, d_sem))).reshape(-1)
    sem_center_b = l2_normalize(rng.normal(loc=0.0, scale=1.0, size=(1, d_sem))).reshape(-1)

    mem_a_sem = l2_normalize(rng.normal(loc=0.0, scale=1.0, size=(memory_size, d_sem)) + 0.40 * sem_center_a, axis=1).astype(np.float32)
    mem_b_sem = l2_normalize(rng.normal(loc=0.0, scale=1.0, size=(memory_size, d_sem)) + 0.40 * sem_center_b, axis=1).astype(np.float32)

    # Style memories clustered around persona centroids
    mem_a_style = np.clip(rng.normal(loc=centroid_a, scale=0.10, size=(memory_size, d_style)), 0.0, 1.0).astype(np.float32)
    mem_b_style = np.clip(rng.normal(loc=centroid_b, scale=0.10, size=(memory_size, d_style)), 0.0, 1.0).astype(np.float32)

    bank_a = PersonaBank("persona_a", mem_a_sem, mem_a_style, centroid_a)
    bank_b = PersonaBank("persona_b", mem_b_sem, mem_b_style, centroid_b)
    return bank_a, bank_b


def retrieve_style(prompt_sem: np.ndarray, bank: PersonaBank, top_k: int) -> np.ndarray:
    sims = bank.memory_sem @ prompt_sem
    idx = np.argpartition(-sims, kth=min(top_k - 1, len(sims) - 1))[:top_k]
    # sort selected by similarity desc
    idx = idx[np.argsort(-sims[idx])]
    return np.mean(bank.memory_style[idx], axis=0)


def simulate_responses(
    rng: np.random.Generator,
    prompts_sem: np.ndarray,
    prompts_base_style: np.ndarray,
    bank_a: PersonaBank,
    bank_b: PersonaBank,
    top_k: int,
    memory_influence: float,
    noise_sigma: float,
) -> dict:
    n = prompts_sem.shape[0]
    d_style = prompts_base_style.shape[1]

    out_no_mem = np.zeros((n, d_style), dtype=np.float32)
    out_a = np.zeros((n, d_style), dtype=np.float32)
    out_b = np.zeros((n, d_style), dtype=np.float32)

    for i in range(n):
        base = prompts_base_style[i]
        ra = retrieve_style(prompts_sem[i], bank_a, top_k)
        rb = retrieve_style(prompts_sem[i], bank_b, top_k)

        eps0 = rng.normal(0.0, noise_sigma, size=d_style).astype(np.float32)
        epsa = rng.normal(0.0, noise_sigma, size=d_style).astype(np.float32)
        epsb = rng.normal(0.0, noise_sigma, size=d_style).astype(np.float32)

        out_no_mem[i] = np.clip(base + eps0, 0.0, 1.0)
        out_a[i] = np.clip((1.0 - memory_influence) * base + memory_influence * ra + epsa, 0.0, 1.0)
        out_b[i] = np.clip((1.0 - memory_influence) * base + memory_influence * rb + epsb, 0.0, 1.0)

    return {"no_mem": out_no_mem, "a": out_a, "b": out_b}


def persona_class_acc(outputs: np.ndarray, c_pos: np.ndarray, c_neg: np.ndarray) -> float:
    # correct if output is closer to positive centroid
    d_pos = np.linalg.norm(outputs - c_pos[None, :], axis=1)
    d_neg = np.linalg.norm(outputs - c_neg[None, :], axis=1)
    return float(np.mean(d_pos < d_neg))


def run_experiment(
    num_prompts: int,
    memory_size: int,
    d_sem: int,
    d_style: int,
    top_k: int,
    seed: int,
    verbose: bool,
) -> dict:
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[exp15] Building synthetic banks (memory_size={memory_size})", flush=True)

    bank_a, bank_b = build_persona_banks(rng, memory_size, d_sem, d_style)

    prompts_sem = l2_normalize(rng.normal(0.0, 1.0, size=(num_prompts, d_sem)), axis=1).astype(np.float32)
    prompts_base_style = np.clip(rng.normal(loc=np.array([0.5, 0.5, 0.5, 0.5]), scale=0.18, size=(num_prompts, d_style)), 0.0, 1.0).astype(np.float32)

    if verbose:
        print(f"[exp15] Simulating {num_prompts} prompts", flush=True)

    sim = simulate_responses(
        rng,
        prompts_sem,
        prompts_base_style,
        bank_a,
        bank_b,
        top_k=top_k,
        memory_influence=0.62,
        noise_sigma=0.04,
    )

    out0 = sim["no_mem"]
    outa = sim["a"]
    outb = sim["b"]

    # Core metrics
    # Persona separation under same prompt
    sep_l2 = np.linalg.norm(outa - outb, axis=1)

    # Memory effect gain vs no-memory baseline
    d0a = np.linalg.norm(out0 - bank_a.centroid_style[None, :], axis=1)
    dAa = np.linalg.norm(outa - bank_a.centroid_style[None, :], axis=1)
    d0b = np.linalg.norm(out0 - bank_b.centroid_style[None, :], axis=1)
    dBb = np.linalg.norm(outb - bank_b.centroid_style[None, :], axis=1)

    gain_a = d0a - dAa
    gain_b = d0b - dBb

    acc_a = persona_class_acc(outa, bank_a.centroid_style, bank_b.centroid_style)
    acc_b = persona_class_acc(outb, bank_b.centroid_style, bank_a.centroid_style)

    # Prompt-level targeted dimensions
    # style dims: [analytical, warm, expressive, direct]
    delta = outb - outa
    target_summary = {
        "delta_analytical_mean": float(np.mean(delta[:, 0])),
        "delta_warm_mean": float(np.mean(delta[:, 1])),
        "delta_expressive_mean": float(np.mean(delta[:, 2])),
        "delta_direct_mean": float(np.mean(delta[:, 3])),
    }

    # Scale study (same prompt pool, varying memory size)
    scale_sizes = [50, 200, 1000, 5000]
    scale_points = []
    for msz in scale_sizes:
        bsa, bsb = build_persona_banks(rng, msz, d_sem, d_style)
        sim_s = simulate_responses(
            rng,
            prompts_sem,
            prompts_base_style,
            bsa,
            bsb,
            top_k=min(top_k, msz),
            memory_influence=0.62,
            noise_sigma=0.04,
        )
        sa, sb = sim_s["a"], sim_s["b"]
        sep = np.linalg.norm(sa - sb, axis=1)
        scale_points.append(
            {
                "memory_size": msz,
                "persona_separation_l2_mean": float(np.mean(sep)),
                "persona_class_acc_a": float(persona_class_acc(sa, bsa.centroid_style, bsb.centroid_style)),
                "persona_class_acc_b": float(persona_class_acc(sb, bsb.centroid_style, bsa.centroid_style)),
            }
        )

    return {
        "metadata": {
            "num_prompts": num_prompts,
            "memory_size": memory_size,
            "d_sem": d_sem,
            "d_style": d_style,
            "top_k": top_k,
            "seed": seed,
            "timestamp": time.time(),
            "style_dims": ["analytical", "warm", "expressive", "direct"],
        },
        "metrics": {
            "persona_separation_l2_mean": float(np.mean(sep_l2)),
            "persona_separation_l2_p90": float(np.quantile(sep_l2, 0.90)),
            "memory_gain_a_mean": float(np.mean(gain_a)),
            "memory_gain_b_mean": float(np.mean(gain_b)),
            "memory_gain_a_positive_rate": float(np.mean(gain_a > 0.0)),
            "memory_gain_b_positive_rate": float(np.mean(gain_b > 0.0)),
            "persona_class_acc_a": float(acc_a),
            "persona_class_acc_b": float(acc_b),
        },
        "targeted_deltas": target_summary,
        "scale_study": scale_points,
        "artifacts": {
            "sep_l2": sep_l2.tolist(),
            "gain_a": gain_a.tolist(),
            "gain_b": gain_b.tolist(),
            "out_a": outa.tolist(),
            "out_b": outb.tolist(),
        },
    }


def write_outputs(data: dict) -> None:
    json_path = os.path.join(RESULTS_DIR, "exp15_synthetic_persona_memory_effect.json")
    txt_path = os.path.join(RESULTS_DIR, "exp15_synthetic_persona_memory_effect.txt")
    png_summary = os.path.join(RESULTS_DIR, "exp15_synthetic_persona_memory_effect_summary.png")
    png_clusters = os.path.join(RESULTS_DIR, "exp15_synthetic_persona_memory_effect_clusters.png")
    png_scale = os.path.join(RESULTS_DIR, "exp15_synthetic_persona_memory_effect_scale.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    m = data["metrics"]
    d = data["targeted_deltas"]
    lines = [
        "EXP15: Synthetic Persona Memory Effect (Large Scale)",
        "=" * 50,
        f"Prompts: {data['metadata']['num_prompts']}",
        f"Memory size per persona bank: {data['metadata']['memory_size']}",
        f"Top-k: {data['metadata']['top_k']}",
        "",
        "Core metrics:",
        f"- Persona separation L2 (mean): {m['persona_separation_l2_mean']:.4f}",
        f"- Persona separation L2 (p90): {m['persona_separation_l2_p90']:.4f}",
        f"- Memory gain A (mean): {m['memory_gain_a_mean']:.4f}",
        f"- Memory gain B (mean): {m['memory_gain_b_mean']:.4f}",
        f"- Memory gain A positive-rate: {m['memory_gain_a_positive_rate']:.3f}",
        f"- Memory gain B positive-rate: {m['memory_gain_b_positive_rate']:.3f}",
        f"- Persona class acc A: {m['persona_class_acc_a']:.3f}",
        f"- Persona class acc B: {m['persona_class_acc_b']:.3f}",
        "",
        "Targeted style deltas (B - A):",
        f"- analytical: {d['delta_analytical_mean']:+.4f}",
        f"- warm: {d['delta_warm_mean']:+.4f}",
        f"- expressive: {d['delta_expressive_mean']:+.4f}",
        f"- direct: {d['delta_direct_mean']:+.4f}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Visual 1: summary bars
    keys = [
        "persona_separation_l2_mean",
        "memory_gain_a_mean",
        "memory_gain_b_mean",
        "persona_class_acc_a",
        "persona_class_acc_b",
    ]
    vals = [m[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(keys)), vals, color=["#4C78A8", "#72B7B2", "#72B7B2", "#F58518", "#F58518"])
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_title("EXP15 Summary Metrics")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_summary, dpi=160)
    plt.close(fig)

    # Visual 2: cluster-like scatter in style space (warm vs analytical)
    out_a = np.array(data["artifacts"]["out_a"], dtype=np.float32)
    out_b = np.array(data["artifacts"]["out_b"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(out_a[:, 0], out_a[:, 1], s=5, alpha=0.20, label="A responses", color="#4C78A8")
    ax.scatter(out_b[:, 0], out_b[:, 1], s=5, alpha=0.20, label="B responses", color="#E45756")
    ax.set_xlabel("analytical")
    ax.set_ylabel("warm")
    ax.set_title("EXP15 Response Style Space (analytical vs warm)")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(png_clusters, dpi=160)
    plt.close(fig)

    # Visual 3: memory-size scaling
    scale = data["scale_study"]
    ms = [x["memory_size"] for x in scale]
    sep = [x["persona_separation_l2_mean"] for x in scale]
    acca = [x["persona_class_acc_a"] for x in scale]
    accb = [x["persona_class_acc_b"] for x in scale]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ms, sep, marker="o", color="#4C78A8", label="Persona separation (L2 mean)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Memory size per persona bank (log scale)")
    ax1.set_ylabel("Separation (L2)")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(ms, acca, marker="s", color="#F58518", label="Class acc A")
    ax2.plot(ms, accb, marker="^", color="#54A24B", label="Class acc B")
    ax2.set_ylabel("Persona classification accuracy")
    ax2.set_ylim(0.5, 1.01)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="lower right")
    ax1.set_title("EXP15 Scaling: Persona Effect vs Memory Size")
    fig.tight_layout()
    fig.savefig(png_scale, dpi=160)
    plt.close(fig)

    print(f"\n✓ Saved {json_path}")
    print(f"✓ Saved {txt_path}")
    print(f"✓ Saved {png_summary}")
    print(f"✓ Saved {png_clusters}")
    print(f"✓ Saved {png_scale}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EXP15 synthetic persona-memory effect")
    p.add_argument("--num-prompts", type=int, default=5000, help="Number of synthetic prompts")
    p.add_argument("--memory-size", type=int, default=5000, help="Memory entries per persona bank")
    p.add_argument("--top-k", type=int, default=8, help="Top-k retrieved memories")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = run_experiment(
        num_prompts=args.num_prompts,
        memory_size=args.memory_size,
        d_sem=128,
        d_style=4,
        top_k=args.top_k,
        seed=args.seed,
        verbose=args.verbose,
    )
    write_outputs(data)
    print("\nEXP15 completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
