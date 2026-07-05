import sys
import os
import warnings
# Suppress noisy ResourceTracker warnings from multiprocess
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocess.resource_tracker")

# Add project root to PYTHONPATH so local `ncm` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import numpy as np
from ncm.memory import MemoryStore, MemoryEntry
from ncm.retrieval import retrieve_multi_hop_auto

# Simple deterministic semantic encoder (hash → random unit vector)
def encode_semantic(text: str, dim: int = 128) -> np.ndarray:
    rnd = np.random.RandomState(abs(hash(text)) % (2**32))
    vec = rnd.rand(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


def main(args):
    # Initialize store
    store = MemoryStore()

    # Add a chain of facts (A is B, B is C, C is D)
    facts = ["A is B", "B is C", "C is D"]
    for f in facts:
        entry = MemoryEntry(
            e_semantic=encode_semantic(f),
            e_emotional=np.zeros(1, dtype=np.float32),  # placeholder
            s_snapshot=np.zeros(1, dtype=np.float32),
            timestamp=store.step,
            text=f,
        )
        store.add(entry)
        store.step += 1

    # Query about "A" (or provided query)
    query_vec = encode_semantic(args.query)

    # Run multi‑hop retrieval with automatic parameter selection
    results = retrieve_multi_hop_auto(
        query_vec,
        store,
        k=args.k,
        base_max_hops=args.hops,
        base_gamma=args.gamma,
        similarity_threshold=args.threshold,
    )
    
    # Preserve the original factual order for readability
    fact_order = {text: idx for idx, text in enumerate(facts)}
    results.sort(key=lambda tup: fact_order.get(tup[1].text, 999))
    
    print("--- Multi-hop retrieval results ---")
    for dist, mem in results:
        print(f"Dist: {dist:.4f}, Text: {mem.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo multi‑hop spreading‑activation retrieval.")
    parser.add_argument("--query", type=str, default="A", help="Query string (default: 'A')")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--hops", type=int, default=3, help="Maximum activation hops")
    parser.add_argument("--gamma", type=float, default=0.8, help="Decay factor for each hop")
    parser.add_argument("--threshold", type=float, default=0.3, help="Similarity threshold for transition edges")
    args = parser.parse_args()
    main(args)
