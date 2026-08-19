"""
EXP10: Retrieval Recall Mock-up (NOT AN EXPERIMENT, NOT A MEASUREMENT)
=====================================================================

WARNING. This file does not measure anything. It contains no MemoryStore, no
encoder call, and no retrieval call. `run_benchmark()` assigns a literal Python
dictionary of numbers that were typed by hand to illustrate an expected
pattern, writes them to JSON, and plots them. Nothing here is derived from
running NCM.

Every number this file emits is a HAND_AUTHORED_LITERAL. The numbers cannot be
reproduced because no computation produces them, and they cannot be falsified
because they are not claims about a system's behaviour. They must never be
cited as evidence, compared against baselines, or reported as results.

The illustration it was written to convey (baseline retrieval is invariant to
internal state, composite retrieval is not) IS tested for real elsewhere, from
live retrieval calls:
  - exp3 in run_all_experiments.py (state-conditioned retrieval)
  - exp11_real_world_corpus_benchmark.py (real multi-session chat corpus)
  - exp12_weight_sensitivity.py (weight sweep)
Cite those instead.

This file is retained only so that the provenance of the previously published
exp10 numbers stays auditable. It is excluded from the experiment suite runner.
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default to cp1252, which cannot encode the check mark this
# script prints, so the run died with UnicodeEncodeError after writing the JSON
# but before writing the plot. Force UTF-8 on the streams when reconfigure is
# available.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split('_')[0]
RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'results', RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

PROVENANCE = "HAND_AUTHORED_LITERALS"
PROVENANCE_NOTE = (
    "Every value in this file was typed by hand to illustrate an expected "
    "pattern. No MemoryStore, encoder, or retrieval call was executed to "
    "produce any of them. These are not measurements and must not be cited "
    "as results. For a real state-conditioning measurement see exp3 in "
    "run_all_experiments.py, exp11_real_world_corpus_benchmark.py, or "
    "exp12_weight_sensitivity.py."
)


def run_benchmark():
    """Emit the hand-authored illustration. Computes nothing."""

    print("=" * 80)
    print("EXP10: HAND-AUTHORED MOCK-UP -- NOT A MEASUREMENT")
    print("=" * 80)
    print(PROVENANCE_NOTE)
    print("=" * 80)

    # Hand-authored literals. See PROVENANCE_NOTE. No code produces these.
    results = {
        "provenance": PROVENANCE,
        "provenance_note": PROVENANCE_NOTE,
        "is_measurement": False,
        "metadata": {
            "num_memories": 1200,
            "num_queries": 12,
            "num_states": 3,
            "note": (
                "HAND_AUTHORED_LITERALS. These three counts describe a benchmark "
                "that was never run; no code in this file consumes them."
            ),
        },
        "systems": {
            "semantic_only": {
                "by_state": {
                    "calm_happy": {
                        "recall@5": 0.428,
                        "recall@10": 0.615,
                        "ndcg@10": 0.548,
                        "mrr": 0.582,
                    },
                    "neutral": {
                        "recall@5": 0.428,
                        "recall@10": 0.615,
                        "ndcg@10": 0.548,
                        "mrr": 0.582,
                    },
                    "stressed_angry": {
                        "recall@5": 0.428,
                        "recall@10": 0.615,
                        "ndcg@10": 0.548,
                        "mrr": 0.582,
                    },
                },
                "state_delta": {
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "ndcg@10": 0.0,
                },
                "overall": {
                    "avg_recall@5": 0.428,
                    "avg_recall@10": 0.615,
                    "avg_ndcg@10": 0.548,
                    "avg_mrr": 0.582,
                    "state_delta_recall@5": 0.0,
                    "state_delta_ndcg": 0.0,
                },
            },
            "semantic_emotional": {
                "by_state": {
                    "calm_happy": {
                        "recall@5": 0.391,
                        "recall@10": 0.573,
                        "ndcg@10": 0.512,
                        "mrr": 0.548,
                    },
                    "neutral": {
                        "recall@5": 0.391,
                        "recall@10": 0.573,
                        "ndcg@10": 0.512,
                        "mrr": 0.548,
                    },
                    "stressed_angry": {
                        "recall@5": 0.391,
                        "recall@10": 0.573,
                        "ndcg@10": 0.512,
                        "mrr": 0.548,
                    },
                },
                "state_delta": {
                    "recall@5": 0.001,
                    "recall@10": 0.002,
                    "ndcg@10": 0.001,
                },
                "overall": {
                    "avg_recall@5": 0.391,
                    "avg_recall@10": 0.573,
                    "avg_ndcg@10": 0.512,
                    "avg_mrr": 0.548,
                    "state_delta_recall@5": 0.001,
                    "state_delta_ndcg": 0.001,
                },
            },
            "ncm_full": {
                "by_state": {
                    "calm_happy": {
                        "recall@5": 0.455,
                        "recall@10": 0.658,
                        "ndcg@10": 0.582,
                        "mrr": 0.611,
                    },
                    "neutral": {
                        "recall@5": 0.382,
                        "recall@10": 0.527,
                        "ndcg@10": 0.468,
                        "mrr": 0.501,
                    },
                    "stressed_angry": {
                        "recall@5": 0.328,
                        "recall@10": 0.441,
                        "ndcg@10": 0.392,
                        "mrr": 0.421,
                    },
                },
                "state_delta": {
                    "recall@5": 0.127,
                    "recall@10": 0.217,
                    "ndcg@10": 0.190,
                },
                "overall": {
                    "avg_recall@5": 0.388,
                    "avg_recall@10": 0.542,
                    "avg_ndcg@10": 0.481,
                    "avg_mrr": 0.511,
                    "state_delta_recall@5": 0.127,
                    "state_delta_ndcg": 0.190,
                },
            },
            "ncm_cached": {
                "by_state": {
                    "calm_happy": {
                        "recall@5": 0.445,
                        "recall@10": 0.642,
                        "ndcg@10": 0.571,
                        "mrr": 0.599,
                    },
                    "neutral": {
                        "recall@5": 0.377,
                        "recall@10": 0.515,
                        "ndcg@10": 0.457,
                        "mrr": 0.491,
                    },
                    "stressed_angry": {
                        "recall@5": 0.324,
                        "recall@10": 0.432,
                        "ndcg@10": 0.384,
                        "mrr": 0.412,
                    },
                },
                "state_delta": {
                    "recall@5": 0.121,
                    "recall@10": 0.210,
                    "ndcg@10": 0.187,
                },
                "overall": {
                    "avg_recall@5": 0.382,
                    "avg_recall@10": 0.530,
                    "avg_ndcg@10": 0.471,
                    "avg_mrr": 0.501,
                    "state_delta_recall@5": 0.121,
                    "state_delta_ndcg": 0.187,
                },
            },
        },
    }
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, "exp10_retrieval_recall.json")
    png_path = os.path.join(RESULTS_DIR, "exp10_retrieval_recall.png")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}\n")

    # Visualization
    systems = ["semantic_only", "semantic_emotional", "ncm_full", "ncm_cached"]
    labels = [s.replace("_", " ") for s in systems]
    avg_recall = [results["systems"][s]["overall"]["avg_recall@5"] for s in systems]
    state_delta = [results["systems"][s]["overall"]["state_delta_recall@5"] for s in systems]
    x = np.arange(len(systems))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, avg_recall, width=width, label="Avg Recall@5", color="#4E79A7")
    ax.bar(x + width / 2, state_delta, width=width, label="State Δ Recall@5", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Score")
    ax.set_title("EXP10 HAND-AUTHORED MOCK-UP -- NOT A MEASUREMENT")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"✓ Plot saved to {png_path}\n")
    
    # Echo the literals back, labelled as literals.
    print("=" * 80)
    print("HAND-AUTHORED VALUES (no computation produced these)")
    print("=" * 80)
    for system_name in ["semantic_only", "semantic_emotional", "ncm_full", "ncm_cached"]:
        delta_recall = results["systems"][system_name]["overall"]["state_delta_recall@5"]
        delta_ndcg = results["systems"][system_name]["overall"]["state_delta_ndcg"]
        avg_recall = results["systems"][system_name]["overall"]["avg_recall@5"]
        print(f"{system_name:20} d_R@5={delta_recall:.3f}  d_NDCG={delta_ndcg:.3f}  Avg_R@5={avg_recall:.3f}")

    print("\n" + "=" * 80)
    print("WHY THERE IS NO INTERPRETATION SECTION")
    print("=" * 80)
    print("""
An earlier version of this file printed an interpretation of the numbers above,
including the sentences "This proves that s_snapshot genuinely changes what the
system recalls based on its internal state" and a side-by-side comparison
against a reported 96.6% recall figure for another system.

That text has been removed. The numbers above were typed by hand. They are not
observations of NCM's behaviour, so they cannot support a conclusion about
NCM's behaviour, and they cannot be compared against another system's measured
result. Nothing can be concluded from this file.

The state-conditioning question is tested for real, from live retrieval calls,
in exp3 (run_all_experiments.py), exp11_real_world_corpus_benchmark.py, and
exp12_weight_sensitivity.py. Read those.
""")


if __name__ == "__main__":
    run_benchmark()
