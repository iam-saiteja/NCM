"""
Experiment 20: MemoryAgentBench Rigorous Baseline Comparison
===========================================================
This script compares NCM against a Plain Vector-Similarity RAG Baseline
on significantly harder synthetic test cases designed to mimic the core
challenges of MemoryAgentBench:
1. Semantically similar distractors (not just random noise).
2. Implicit contradictions without explicit "Correction:" markers.
3. Long-range context windows (1000+ steps).
"""

import sys, os, time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile
from ncm.retrieval import retrieve_top_k

encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, 'models'))

# ==========================================
# 1. Plain RAG Baseline Implementation
# ==========================================
class PlainRAGBaseline:
    """
    A standard cosine-similarity vector database baseline.
    Has no concept of episodic time, emotional vectors, or auto-state.
    """
    def __init__(self):
        self.embeddings = []
        self.texts = []

    def add(self, text: str):
        # Encode and store just like standard RAG
        e_sem = encoder.encode(text)
        self.embeddings.append(e_sem)
        self.texts.append(text)

    def retrieve_top_k(self, query: str, k: int = 3):
        if not self.embeddings:
            return []
        
        q_sem = encoder.encode(query)
        matrix = np.array(self.embeddings, dtype=np.float32)
        
        # Cosine similarity (vectors are L2 normalized by the encoder)
        sims = matrix @ q_sem
        
        # Sort descending
        indices = np.argsort(-sims)[:k]
        
        results = []
        for idx in indices:
            results.append((1.0 - sims[idx], self.texts[idx]))
        return results


# ==========================================
# 2. NCM Wrapper
# ==========================================
class NCMSystem:
    def __init__(self, enable_contradiction=False):
        profile = MemoryProfile()
        if enable_contradiction:
            profile.set_custom("enable_contradiction_awareness", True)
            profile.set_custom("contradiction_penalty", 1.0)
            # HARD MODE: disable the requirement for explicit marker words
            profile.set_custom("contradiction_requires_marker", False)
            profile.set_custom("write_conflict_trace", True)
            profile.set_custom("contradiction_similarity_threshold", 0.70)
        self.store = MemoryStore(profile=profile)
        
    def add(self, text: str):
        sem = encoder.encode(text)
        state = self.store.auto_state.get_current_state()
        emo = encoder.encode_emotional(state)
        s_snap = encoder.encode_state(state)
        
        entry = MemoryEntry(
            e_semantic=sem, e_emotional=emo, s_snapshot=s_snap,
            timestamp=self.store.step, text=text,
            strength=1.5
        )
        self.store.add(entry, update_auto_state=True)
        self.store.tick()
        
    def retrieve_top_k(self, query: str, k: int = 3):
        q_sem = encoder.encode(query)
        state = self.store.auto_state.get_current_state()
        q_emo = encoder.encode_emotional(state)
        s_norm = encoder.encode_state(state)
        
        raw_results = retrieve_top_k(q_sem, q_emo, self.store, s_norm, self.store.step, k=k)
        # Format identical to RAG baseline for easy comparison
        return [(dist, m.text) for dist, prob, m in raw_results]


# ==========================================
# 3. Hard Benchmark Tasks
# ==========================================

def run_hard_lru_task():
    """
    Task: Long-Range Understanding (LRU)
    Test: Semantic extraction among highly similar distractors over time.
    """
    print("\n" + "="*50)
    print("TASK: Hard Long-Range Understanding (LRU)")
    print("="*50)
    
    rag = PlainRAGBaseline()
    ncm = NCMSystem()
    
    core_fact = "The security password for the main terminal is DELTA-9."
    rag.add(core_fact)
    ncm.add(core_fact)
    
    # Insert 1000 *semantically similar* distractors to confuse standard RAG
    # In the previous toy test, distractors were completely unrelated.
    for i in range(1000):
        if i % 3 == 0:
            noise = f"The security password for the backup system is ALPHA-{i}."
        elif i % 3 == 1:
            noise = f"The username for the main terminal is user-{i}."
        else:
            noise = f"The access code for the side door is BETA-{i}."
            
        rag.add(noise)
        ncm.add(noise)
        
    # Consolidate NCM
    ncm.store.consolidate(0.95)
    
    query = "What is the security password for the main terminal?"
    print(f"Query: '{query}'")
    
    rag_res = rag.retrieve_top_k(query, k=1)
    ncm_res = ncm.retrieve_top_k(query, k=1)
    
    rag_top = rag_res[0][1] if rag_res else "None"
    ncm_top = ncm_res[0][1] if ncm_res else "None"
    
    print(f"\n[RAG Baseline Top-1]: {rag_top}")
    print(f"[NCM System Top-1]  : {ncm_top}")
    
    rag_pass = "DELTA-9" in rag_top
    ncm_pass = "DELTA-9" in ncm_top
    
    print(f"\nRAG PASS: {rag_pass} | NCM PASS: {ncm_pass}")
    return rag_pass, ncm_pass


def run_hard_selective_forgetting_task():
    """
    Task: Conflict Resolution / Selective Forgetting (CR)
    Test: Implicit contradiction spanning hundreds of steps. No marker words.
    """
    print("\n" + "="*50)
    print("TASK: Hard Selective Forgetting (Implicit CR)")
    print("="*50)
    
    rag = PlainRAGBaseline()
    ncm = NCMSystem(enable_contradiction=True)
    
    fact_v1 = "The location of the secret key is hidden under the doormat."
    rag.add(fact_v1)
    ncm.add(fact_v1)
    
    # 500 steps of somewhat related actions
    for i in range(500):
        noise = f"I walked around the house and checked area {i}."
        rag.add(noise)
        ncm.add(noise)
        
    # Implicit update (NO "Correction:" or "Update:" markers!)
    fact_v2 = "The location of the secret key is inside the flowerpot."
    rag.add(fact_v2)
    ncm.add(fact_v2)
    
    query = "Where is the location of the secret key?"
    print(f"Query: '{query}'")
    
    rag_res = rag.retrieve_top_k(query, k=1)
    ncm_res = ncm.retrieve_top_k(query, k=1)
    
    rag_top = rag_res[0][1] if rag_res else "None"
    ncm_top = ncm_res[0][1] if ncm_res else "None"
    
    print(f"\n[RAG Baseline Top-1]: {rag_top}")
    print(f"[NCM System Top-1]  : {ncm_top}")
    
    # To pass, it MUST retrieve the updated fact (v2) and NOT the outdated one (v1).
    rag_pass = "flowerpot" in rag_top and "doormat" not in rag_top
    ncm_pass = "flowerpot" in ncm_top and "doormat" not in ncm_top
    
    print(f"\nRAG PASS: {rag_pass} | NCM PASS: {ncm_pass}")
    
    # Print traces to prove NCM logic
    traces = [m.text for m in ncm.store.get_all_safe() if getattr(m, 'is_conflict_trace', False)]
    if traces:
        print("\nNCM detected implicit conflicts:")
        for t in traces:
            print(f"  -> {t}")
            
    return rag_pass, ncm_pass

if __name__ == "__main__":
    t0 = time.perf_counter()
    lru_rag, lru_ncm = run_hard_lru_task()
    cr_rag, cr_ncm = run_hard_selective_forgetting_task()
    
    print("\n" + "="*50)
    print("FINAL HEAD-TO-HEAD RESULTS")
    print("="*50)
    print(f"Long-Range Understanding : RAG [{'PASS' if lru_rag else 'FAIL'}] vs NCM [{'PASS' if lru_ncm else 'FAIL'}]")
    print(f"Selective Forgetting     : RAG [{'PASS' if cr_rag else 'FAIL'}] vs NCM [{'PASS' if cr_ncm else 'FAIL'}]")
    print(f"Total test time: {time.perf_counter() - t0:.1f}s")
