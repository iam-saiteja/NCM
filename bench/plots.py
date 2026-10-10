"""Figures for docs/RESULTS.md, drawn from bench/results/*.json.

  python -m bench.plots
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "docs", "img")

SURFACE, INK, MUTED_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASE, V1, V2 = "#b9b8b2", "#eb6834", "#2a78d6"     # baselines gray, v1 orange, v2 blue
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED_INK,
                     "xtick.color": MUTED_INK, "ytick.color": MUTED_INK, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def load(name):
    return json.load(open(os.path.join(RES, name)))


def hbars(ax, labels, values, colors, title, fmt="{:.3f}"):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def retrieval():
    arms = [("NCM v1", "ncm-v1", V1), ("cosine (MiniLM)", "dense", BASE), ("BM25", "bm25", BASE),
            ("NCM v2", "ncm-v2", V2)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    for ax, (ds, title) in zip(axes, (("locomo", "LoCoMo"), ("longmemeval", "LongMemEval-S"))):
        m = load(f"retrieval-{ds}-minilm.json")["methods"]
        hbars(ax, [a for a, _, _ in arms], [m[k]["all"]["recall@5"] for _, k, _ in arms],
              [c for _, _, c in arms], f"{title}: evidence recall@5")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "retrieval.png"), dpi=160)


def qa():
    d = load("qa-locomo-qwen2.5_3b.json")["arms"]
    arms = [("no memory", "none", BASE), ("NCM v1", "ncm-v1", V1), ("cosine (MiniLM)", "dense", BASE),
            ("NCM v2 recall", "ncm-v2", V2), ("NCM v2 context()", "ncm-v2-context", V2)]
    arms = [a for a in arms if a[1] in d]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    hbars(ax, [a for a, _, _ in arms], [d[k]["f1"] for _, k, _ in arms], [c for _, _, c in arms],
          "LoCoMo QA, qwen2.5:3b reading 10 memories: answer F1")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qa.png"), dpi=160)


def speed():
    d = load("speed.json")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for name, key, color in (("NCM v1", "v1", V1), ("NCM v2", "v2", V2)):
        ns = sorted(d[key], key=int)
        xs, ys = [int(n) for n in ns], [d[key][n]["recall"]["p50_ms"] for n in ns]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=name)
        ax.annotate(f"{ys[-1]:.0f} ms", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3),
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("memories in the store")
    ax.set_ylabel("recall latency, p50 (ms)")
    ax.set_title("Recall latency (CPU, excluding the query embedding)", loc="left", fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed.png"), dpi=160)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    retrieval()
    qa()
    speed()
    print("wrote", os.listdir(OUT))
