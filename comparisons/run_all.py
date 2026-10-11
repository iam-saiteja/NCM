"""Run every comparison in order; each step resumes from its own saved results.

  python -m comparisons.run_all

Needs Ollama with qwen2.5:3b, started with OLLAMA_KV_CACHE_TYPE=f16 and
OLLAMA_CONTEXT_LENGTH=16384 (see comparisons/README.md), and the datasets from bench/data.py.
"""

import subprocess
import sys

STEPS = [
    ["-m", "comparisons.retrieval"],
    ["-m", "comparisons.retrieval", "NCM v2 · SPLADE instead of BM25", "NCM v2 · SPLADE + BM25"],
    ["-m", "comparisons.end_to_end"],
    ["-m", "comparisons.report"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">>", " ".join(step), flush=True)
        code = subprocess.call([sys.executable] + step)
        print(f"<< exit {code}", flush=True)
