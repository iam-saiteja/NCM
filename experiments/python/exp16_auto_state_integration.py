"""
EXP16: Auto-State Integration Validation
=======================================

Purpose
- Consolidate legacy sim checks (Sim 2/3/4) into one experiment script.
- Validate trajectory, retrieval trend, and persistence on integrated NCM code.

Outputs
- experiments/results/exp16/exp16_auto_state_integration.json
- experiments/results/exp16/exp16_auto_state_integration.txt
- experiments/results/exp16/exp16_state_trajectory.png
- experiments/results/exp16/exp16_retrieval_trend.png
- experiments/results/exp16/exp16_persistence_validation.png
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import AutoStateTracker, MemoryEntry, MemoryStore, NCMFile, SentenceEncoder

RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", "exp16")
os.makedirs(RESULTS_DIR, exist_ok=True)

TOL = 1e-5
DIMS = ["valence", "arousal", "dominance", "curiosity", "stress"]

CONVERSATION_30 = [
    "I have so many deadlines and I am completely overwhelmed",
    "I failed my exam and I feel terrible about it",
    "There is too much pressure and I cannot handle it",
    "I am worried I will not finish this project in time",
    "Everything is going wrong and I feel helpless",
    "The deadline is tomorrow and I have barely started",
    "I am so anxious about the presentation next week",
    "I cannot sleep because I keep thinking about all my tasks",
    "My boss is angry and I feel like I am failing at everything",
    "I feel tense and stressed about everything in my life right now",
    "How does the human brain store long-term memories?",
    "I wonder what causes northern lights, the physics is fascinating",
    "What if we could simulate consciousness in a machine?",
    "Tell me about the history of the Roman empire",
    "I am curious about how black holes actually work",
    "What are the most interesting unsolved problems in mathematics?",
    "How do plants convert sunlight into energy through photosynthesis?",
    "I want to explore the theory of relativity in more depth",
    "What is the current state of quantum computing research?",
    "Can you explain how transformers work in machine learning?",
    "I finished my project and I feel amazing and relieved",
    "Today was a wonderful day and I am so happy",
    "I got great feedback and I feel confident and pleased",
    "Everything is going well and I feel content and at peace",
    "I had a delightful conversation with my friend today",
    "I completed all my tasks and I feel joyful and free",
    "Life feels wonderful right now, I am so grateful",
    "I feel calm, happy, and completely at ease with myself",
    "Today I received wonderful news and I feel delighted",
    "I am in a great mood, feeling cheerful and content",
]

EXPECTED_PROPOSED = {
    "turn10": np.array([0.4068923, 0.45871156, 0.49421537, 0.41032666, 0.63592273], dtype=np.float32),
    "turn20": np.array([0.46631497, 0.48791257, 0.50877178, 0.51426035, 0.5078938], dtype=np.float32),
    "turn30": np.array([0.52644837, 0.45842755, 0.4931564, 0.48029253, 0.4312407], dtype=np.float32),
}

QUERIES = {
    "Era1": "I am having a really hard time with everything right now",
    "Era2": "I want to learn and understand something new and interesting",
    "Era3": "I feel great and everything is going really well",
}


@dataclass
class Mem:
    turn: int
    text: str
    era: str
    emb: np.ndarray
    state: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def trajectory_check() -> dict:
    tracker = AutoStateTracker()
    states = [tracker.get_current_state()]
    state_history = [{"turn": 0, "state": states[0].tolist()}]
    
    for t in CONVERSATION_30:
        states.append(tracker.update(t))
        state_history.append({"turn": len(states) - 1, "state": states[-1].tolist()})

    d10 = float(np.max(np.abs(states[10] - EXPECTED_PROPOSED["turn10"])))
    d20 = float(np.max(np.abs(states[20] - EXPECTED_PROPOSED["turn20"])))
    d30 = float(np.max(np.abs(states[30] - EXPECTED_PROPOSED["turn30"])))

    return {
        "turn10_max_abs_diff": d10,
        "turn20_max_abs_diff": d20,
        "turn30_max_abs_diff": d30,
        "pass": bool(d10 < TOL and d20 < TOL and d30 < TOL),
        "turn10": states[10].tolist(),
        "turn20": states[20].tolist(),
        "turn30": states[30].tolist(),
        "state_history": state_history,
    }


def build_memories(encoder: SentenceEncoder) -> tuple[list[Mem], dict[str, np.ndarray]]:
    tracker = AutoStateTracker()
    memories: list[Mem] = []

    for i, text in enumerate(CONVERSATION_30):
        s = tracker.update(text)
        era = "Era1" if i < 10 else ("Era2" if i < 20 else "Era3")
        memories.append(Mem(i + 1, text, era, encoder.encode(text), s.copy()))

    states_at_era_end = {
        "Era1": memories[9].state.copy(),
        "Era2": memories[19].state.copy(),
        "Era3": memories[29].state.copy(),
    }
    return memories, states_at_era_end


def score_combined(q_emb: np.ndarray, q_state: np.ndarray, m: Mem, w_sem: float = 0.5, w_state: float = 0.5) -> float:
    sem = float(np.dot(q_emb, m.emb))
    d = float(np.linalg.norm(normalize(q_state) - normalize(m.state))) / math.sqrt(2.0)
    state_sim = 1.0 - float(np.clip(d, 0.0, 1.0))
    return w_sem * sem + w_state * state_sim


def retrieval_trend_check(encoder: SentenceEncoder) -> dict:
    memories, states_at_era_end = build_memories(encoder)

    combined = {}
    baseline = {}

    for era, query in QUERIES.items():
        q_emb = encoder.encode(query)
        q_state = states_at_era_end[era]

        scored_comb = sorted(
            [(score_combined(q_emb, q_state, m), m.era) for m in memories],
            key=lambda x: x[0],
            reverse=True,
        )
        top5_c = scored_comb[:5]
        top10_c = scored_comb[:10]
        p5_c = sum(1 for _, e in top5_c if e == era) / 5.0
        p10_c = sum(1 for _, e in top10_c if e == era) / 10.0

        scored_sem = sorted(
            [(float(np.dot(q_emb, m.emb)), m.era) for m in memories],
            key=lambda x: x[0],
            reverse=True,
        )
        top5_b = scored_sem[:5]
        top10_b = scored_sem[:10]
        p5_b = sum(1 for _, e in top5_b if e == era) / 5.0
        p10_b = sum(1 for _, e in top10_b if e == era) / 10.0

        combined[era] = {"p5": round(p5_c, 3), "p10": round(p10_c, 3)}
        baseline[era] = {"p5": round(p5_b, 3), "p10": round(p10_b, 3)}

    deltas = {era: round(combined[era]["p5"] - baseline[era]["p5"], 3) for era in ["Era1", "Era2", "Era3"]}
    trend_mean_gain = round(float(np.mean(list(deltas.values()))), 3)

    return {
        "combined": combined,
        "semantic_baseline": baseline,
        "p5_delta_vs_baseline": deltas,
        "trend_mean_p5_gain": trend_mean_gain,
    }


def persistence_check(encoder: SentenceEncoder) -> dict:
    store_pre = MemoryStore()

    for text in CONVERSATION_30[:20]:
        s = store_pre.auto_state.get_current_state()
        mem = MemoryEntry(
            e_semantic=encoder.encode(text),
            e_emotional=encoder.encode_emotional(s),
            s_snapshot=encoder.encode_state(np.pad(s, (0, 2), mode="constant", constant_values=0.5)),
            timestamp=int(store_pre.step),
            text=text,
        )
        store_pre.add(mem, update_auto_state=True)
        store_pre.step += 1

    s_pre = store_pre.auto_state.get_current_state()
    w_state_pre, w_sem_pre = store_pre.auto_state.get_adaptive_weights()

    fd, ncm_path = tempfile.mkstemp(prefix="exp16_", suffix=".ncm")
    os.close(fd)

    NCMFile.save(store_pre, ncm_path, compress=True, fp16=False)
    store_post = NCMFile.load(ncm_path)

    s_post = store_post.auto_state.get_current_state()
    w_state_post, w_sem_post = store_post.auto_state.get_adaptive_weights()

    max_state_diff = float(np.max(np.abs(s_pre - s_post)))
    turn_ok = bool(store_pre.auto_state.turn == store_post.auto_state.turn)
    alpha_ok = bool(np.max(np.abs(store_pre.auto_state.alpha - store_post.auto_state.alpha)) < TOL)
    weight_ok = bool(abs(w_state_pre - w_state_post) < TOL and abs(w_sem_pre - w_sem_post) < TOL)

    q = encoder.encode("I feel overwhelmed and anxious about my work")
    score_pre = [score_combined(q, s_pre, Mem(i + 1, m.text, "Era", m.e_semantic, m.auto_state_snapshot), w_sem_pre, w_state_pre)
                 for i, m in enumerate(store_pre.get_all_safe())]
    score_post = [score_combined(q, s_post, Mem(i + 1, m.text, "Era", m.e_semantic, m.auto_state_snapshot), w_sem_post, w_state_post)
                  for i, m in enumerate(store_post.get_all_safe())]

    max_score_diff = float(max(abs(a - b) for a, b in zip(score_pre, score_post))) if score_pre else 0.0
    top1_ok = bool(int(np.argmax(score_pre)) == int(np.argmax(score_post))) if score_pre else True

    return {
        "max_state_diff": max_state_diff,
        "max_score_diff": max_score_diff,
        "turn_ok": turn_ok,
        "alpha_ok": alpha_ok,
        "weights_ok": weight_ok,
        "top1_ok": top1_ok,
        "pass": bool(max_state_diff < TOL and max_score_diff < TOL and turn_ok and alpha_ok and weight_ok and top1_ok),
        "s_pre": [float(x) for x in s_pre],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUALIZATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_state_trajectory(trajectory_history):
    """
    Plot 5D state evolution across 30 turns.
    trajectory_history: list of state dicts with turn numbers
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    turns = [s["turn"] for s in trajectory_history]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
    
    for i, dim in enumerate(DIMS):
        values = [s["state"][i] for s in trajectory_history]
        ax.plot(turns, values, marker="o", linewidth=2.5, markersize=5, 
                label=dim.capitalize(), color=colors[i], alpha=0.85)
    
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(10, 1.05, "Turn 10", ha="center", fontsize=9, color="gray")
    ax.text(20, 1.05, "Turn 20", ha="center", fontsize=9, color="gray")
    
    # Shade stress era (turns 1-10) and curiosity era (11-20)
    ax.axvspan(0.5, 10.5, alpha=0.08, color="#e74c3c", label="Stress Era")
    ax.axvspan(10.5, 20.5, alpha=0.08, color="#3498db", label="Curiosity Era")
    
    ax.set_xlabel("Turn", fontsize=11)
    ax.set_ylabel("State Value", fontsize=11)
    ax.set_title("EXP16 Auto-State Trajectory: 5D Emotional State Evolution", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_state_trajectory.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def plot_retrieval_trend(trend_data):
    """
    Plot P@5 retrieval delta by era: combined vs semantic baseline.
    trend_data: dict with p5_delta_vs_baseline
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    eras = ["Era1\n(Stress)", "Era2\n(Curiosity)", "Era3\n(Mixed)"]
    deltas = [
        trend_data["p5_delta_vs_baseline"]["Era1"],
        trend_data["p5_delta_vs_baseline"]["Era2"],
        trend_data["p5_delta_vs_baseline"]["Era3"],
    ]
    
    # Color bars based on positive/negative
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
    bars = ax.bar(eras, deltas, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
    
    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + (0.01 if height > 0 else -0.02),
                f"{delta:+.3f}", ha="center", va="bottom" if height > 0 else "top",
                fontsize=11, fontweight="bold")
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(y=trend_data["trend_mean_p5_gain"], color="blue", linestyle="--", 
               linewidth=2, alpha=0.7, label=f"Mean Gain: {trend_data['trend_mean_p5_gain']:+.3f}")
    
    ax.set_ylabel("P@5 Delta (Combined - Semantic)", fontsize=11)
    ax.set_title("EXP16 Retrieval Improvement: Auto-State Integration Effect", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 0.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_retrieval_trend.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def plot_persistence_validation(persist_data):
    """
    Plot persistence validation metrics: state/score diffs and boolean checks.
    persist_data: dict with max_state_diff, max_score_diff, and boolean flags
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Difference magnitudes
    metrics_left = ["State\nMax Diff", "Score\nMax Diff"]
    values_left = [persist_data["max_state_diff"], persist_data["max_score_diff"]]
    colors_left = ["#2ecc71" if v < 1e-4 else "#f39c12" for v in values_left]
    
    bars1 = ax1.bar(metrics_left, values_left, color=colors_left, alpha=0.85, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Max Absolute Difference", fontsize=11)
    ax1.set_title("Round-Trip Diff Magnitudes", fontsize=11, fontweight="bold")
    
    # Only use log scale if all values are > 0
    if all(v > 0 for v in values_left):
        ax1.set_yscale("log")
    else:
        ax1.set_ylim(-0.1, max(values_left) + 0.1 if values_left else 1.0)
    
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars1, values_left):
        y_pos = val * 1.5 if val > 0 else 0.05
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.2e}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Right: Boolean checks
    checks = ["Turn\nOK", "Alpha\nOK", "Weights\nOK", "Top1\nOK"]
    values_right = [
        1.0 if persist_data["turn_ok"] else 0.0,
        1.0 if persist_data["alpha_ok"] else 0.0,
        1.0 if persist_data["weights_ok"] else 0.0,
        1.0 if persist_data["top1_ok"] else 0.0,
    ]
    colors_right = ["#2ecc71" if v == 1.0 else "#e74c3c" for v in values_right]
    
    bars2 = ax2.bar(checks, values_right, color=colors_right, alpha=0.85, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Pass Status", fontsize=11)
    ax2.set_title("Integrity Checks", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["FAIL", "PASS"])
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add checkmarks/X marks
    for bar, val in zip(bars2, values_right):
        mark = "✓" if val == 1.0 else "✗"
        color = "green" if val == 1.0 else "red"
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                mark, ha="center", va="bottom", fontsize=16, fontweight="bold", color=color)
    
    fig.suptitle("EXP16 Persistence Validation: .NCM Round-Trip", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp16_persistence_validation.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    return png_path


def main() -> None:
    print("[exp16] Loading encoder")
    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models"))

    print("[exp16] Running trajectory check")
    traj = trajectory_check()

    print("[exp16] Running retrieval trend check")
    trend = retrieval_trend_check(encoder)

    print("[exp16] Running persistence check")
    persist = persistence_check(encoder)

    verdict = bool(traj["pass"] and persist["pass"])

    results = {
        "experiment": "exp16_auto_state_integration",
        "design_constants": {
            "dimensions": DIMS,
            "alpha": [0.15, 0.15, 0.15, 0.20, 0.25],
            "sigma": "(1 + cos(e,pos) - cos(e,neg)) / 2 clipped to [0,1]",
            "initial_state": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        "trajectory_check": traj,
        "retrieval_trend": trend,
        "persistence_check": persist,
        "verdict": "PASS" if verdict else "FAIL",
    }

    json_path = os.path.join(RESULTS_DIR, "exp16_auto_state_integration.json")
    txt_path = os.path.join(RESULTS_DIR, "exp16_auto_state_integration.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EXP16: Auto-State Integration Validation\n")
        f.write("========================================\n\n")
        f.write(f"Verdict: {results['verdict']}\n\n")
        f.write("Trajectory checkpoint max-abs-diff\n")
        f.write(f"- Turn10: {traj['turn10_max_abs_diff']:.2e}\n")
        f.write(f"- Turn20: {traj['turn20_max_abs_diff']:.2e}\n")
        f.write(f"- Turn30: {traj['turn30_max_abs_diff']:.2e}\n\n")
        f.write("Retrieval P@5 trend delta (combined - semantic baseline)\n")
        for era in ["Era1", "Era2", "Era3"]:
            f.write(f"- {era}: {trend['p5_delta_vs_baseline'][era]:+.3f}\n")
        f.write(f"- Mean gain: {trend['trend_mean_p5_gain']:+.3f}\n\n")
        f.write("Persistence\n")
        f.write(f"- max_state_diff: {persist['max_state_diff']:.2e}\n")
        f.write(f"- max_score_diff: {persist['max_score_diff']:.2e}\n")
        f.write(f"- turn_ok/alpha_ok/weights_ok/top1_ok: {persist['turn_ok']}/{persist['alpha_ok']}/{persist['weights_ok']}/{persist['top1_ok']}\n")

    print(f"[exp16] Saved: {json_path}")
    print(f"[exp16] Saved: {txt_path}")

    # Generate visualizations
    print("[exp16] Generating visualizations")
    state_history = traj.get("state_history", [])
    if state_history:
        png1 = plot_state_trajectory(state_history)
        print(f"[exp16] Saved: {png1}")
    
    png2 = plot_retrieval_trend(trend)
    print(f"[exp16] Saved: {png2}")
    
    png3 = plot_persistence_validation(persist)
    print(f"[exp16] Saved: {png3}")


if __name__ == "__main__":
    main()
