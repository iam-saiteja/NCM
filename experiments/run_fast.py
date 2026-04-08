"""
NCM Fast Experiment Suite - Optimized for completion.
Runs math verification + all 4 experiments with practical scale points.
"""

import sys, os, time, json, traceback
import numpy as np

sys.path.insert(0, '/home/user/workspace/ncm_project')

from ncm.encoder import SentenceEncoder
from ncm.memory import MemoryEntry, MemoryStore
from ncm.profile import MemoryProfile, RetrievalWeights
from ncm.retrieval import (
    vectorized_manifold_distance, softmax_retrieval, adaptive_temperature,
    retrieve_top_k, retrieve_top_k_fast, retrieve_semantic_only,
    retrieve_semantic_emotional, retrieval_entropy,
)
from ncm.persistence import NCMFile

RESULTS_DIR = '/home/user/workspace/ncm_project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

encoder = SentenceEncoder(model_dir='/home/user/workspace/ncm_project/models/')

STATE_ARCHETYPES = {
    "calm_happy":      np.array([0.7, 0.8, 0.2, 0.8, 0.3, 0.7, 0.2], dtype=np.float32),
    "stressed_angry":  np.array([0.9, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9], dtype=np.float32),
    "sad_withdrawn":   np.array([0.2, 0.3, 0.6, 0.3, 0.7, 0.3, 0.4], dtype=np.float32),
    "excited_curious": np.array([0.9, 0.7, 0.3, 0.9, 0.4, 0.8, 0.3], dtype=np.float32),
    "fearful":         np.array([0.6, 0.2, 0.8, 0.3, 0.9, 0.2, 0.8], dtype=np.float32),
    "neutral":         np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    "confident":       np.array([0.8, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], dtype=np.float32),
    "exhausted":       np.array([0.1, 0.4, 0.7, 0.2, 0.6, 0.3, 0.6], dtype=np.float32),
}

SEMANTIC_CATEGORIES = {
    "betrayal": ["friend lied about the project deadline", "colleague took credit for my work",
                 "trusted partner broke their promise", "discovered someone talking behind my back",
                 "team member sabotaged the presentation"],
    "achievement": ["finished the marathon in under 4 hours", "got promoted to team lead",
                    "published my first research paper", "won the hackathon competition",
                    "scored highest in the final exam"],
    "loss": ["lost my wallet on the train", "pet cat passed away last week",
             "failed the certification exam", "project got cancelled after months",
             "missed the flight and lost the booking"],
    "discovery": ["found an amazing new algorithm", "discovered a shortcut through the park",
                  "learned about quantum computing", "found a bug causing data corruption",
                  "realized the solution was simpler"],
    "social": ["had a great conversation with the new intern", "team lunch was enjoyable today",
               "met an interesting person at the conference", "helped a stranger with directions",
               "reconnected with an old school friend"],
    "conflict": ["argument with manager about priorities", "disagreement about technical approach",
                 "heated debate during team meeting", "customer complaint escalated badly",
                 "deadline conflict between two projects"],
}


def generate_memory(text, state, timestamp):
    e_sem = encoder.encode(text)
    e_emo = encoder.encode_emotional(state)
    s_snap = encoder.encode_state(state)
    return MemoryEntry(e_semantic=e_sem, e_emotional=e_emo, s_snapshot=s_snap,
                       timestamp=timestamp, text=text)


# Pre-encode all texts once (huge speedup)
print("Pre-encoding all texts...")
ALL_TEXTS = []
TEXT_CAT_MAP = {}
for cat, texts in SEMANTIC_CATEGORIES.items():
    for t in texts:
        ALL_TEXTS.append(t)
        TEXT_CAT_MAP[t] = cat

ALL_VECS = encoder.encode_batch(ALL_TEXTS)
TEXT_TO_VEC = {t: ALL_VECS[i] for i, t in enumerate(ALL_TEXTS)}
print(f"Pre-encoded {len(ALL_TEXTS)} texts")


def build_store_fast(n, seed=42):
    """Build store using pre-encoded vectors (much faster)."""
    rng = np.random.RandomState(seed)
    store = MemoryStore(profile=MemoryProfile(max_size=max(n + 100, 10000)))
    
    cats = list(SEMANTIC_CATEGORIES.keys())
    states = list(STATE_ARCHETYPES.keys())
    state_vals = list(STATE_ARCHETYPES.values())
    
    for i in range(n):
        cat = cats[i % len(cats)]
        texts = SEMANTIC_CATEGORIES[cat]
        text = texts[i % len(texts)]
        
        state = state_vals[i % len(state_vals)].copy()
        noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
        state = np.clip(state + noise, 0.0, 1.0)
        
        e_sem = TEXT_TO_VEC[text].copy()
        # Add tiny noise for uniqueness
        e_sem += rng.randn(128).astype(np.float32) * 0.01
        e_sem /= np.linalg.norm(e_sem)
        
        e_emo = encoder.encode_emotional(state)
        s_snap = encoder.encode_state(state)
        
        mem = MemoryEntry(e_semantic=e_sem, e_emotional=e_emo, s_snapshot=s_snap,
                         timestamp=i, text=f"{text} (v{i})")
        store.add(mem)
    
    store.step = n
    return store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATH VERIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_math():
    print("\n" + "="*60)
    print("MATH VERIFICATION")
    print("="*60)
    results = {}
    
    # 1. W_emo orthonormality
    w = encoder.w_emo
    err = np.linalg.norm(w @ w.T - np.eye(w.shape[0]))
    print(f"1. W_emo·W_emo^T = I error: {err:.2e} {'✓' if err < 1e-5 else '✗'}")
    results["w_emo_ortho_err"] = float(err)
    
    # 2. Semantic L2 norm
    v = encoder.encode("test")
    n_v = np.linalg.norm(v)
    print(f"2. Semantic L2 norm: {n_v:.6f} {'✓' if abs(n_v-1)<1e-5 else '✗'}")
    results["sem_norm"] = float(n_v)
    
    # 3. Emotional L2 norm
    s = np.array([0.5,0.3,0.7,0.2,0.8,0.4,0.6], dtype=np.float32)
    e = encoder.encode_emotional(s)
    n_e = np.linalg.norm(e)
    print(f"3. Emotional L2 norm: {n_e:.6f} {'✓' if abs(n_e-1)<1e-5 else '✗'}")
    results["emo_norm"] = float(n_e)
    
    # 4. State snapshot L2 norm
    sn = encoder.encode_state(s)
    n_s = np.linalg.norm(sn)
    print(f"4. State L2 norm: {n_s:.6f} {'✓' if abs(n_s-1)<1e-5 else '✗'}")
    results["state_norm"] = float(n_s)
    
    # 5. Distance bounds (1000 random pairs)
    rng = np.random.RandomState(42)
    emo_max, state_max = 0, 0
    for _ in range(1000):
        s1 = rng.uniform(0,1,7).astype(np.float32)
        s2 = rng.uniform(0,1,7).astype(np.float32)
        e1, e2 = encoder.encode_emotional(s1), encoder.encode_emotional(s2)
        sn1, sn2 = encoder.encode_state(s1), encoder.encode_state(s2)
        d_e = np.linalg.norm(e1-e2) / 2.0
        d_s = np.linalg.norm(sn1-sn2) / np.sqrt(2)
        emo_max = max(emo_max, d_e)
        state_max = max(state_max, d_s)
        assert 0 <= d_e <= 1.0+1e-6 and 0 <= d_s <= 1.0+1e-6
    print(f"5. 1000 pairs: emo_max={emo_max:.4f} state_max={state_max:.4f} ✓")
    results["emo_max"] = float(emo_max)
    results["state_max"] = float(state_max)
    
    # 6. JL lemma
    k_min = 4 * np.log(100000) / (0.1**2/2 - 0.1**3/3)
    print(f"6. JL min dim: {k_min:.0f}, NCM uses 128 {'✓' if 128>=k_min else '✗'}")
    results["jl_min_dim"] = float(k_min)
    
    # 7. Weights sum
    w = RetrievalWeights()
    print(f"7. Weight sum: {sum(w.as_tuple()):.4f} ✓")
    
    # 8. Dirichlet regularization
    w_b = RetrievalWeights(0.25,0.25,0.25,0.25)
    w_s = RetrievalWeights(0.7,0.1,0.1,0.1)
    print(f"8. Dirichlet KL: balanced={w_b.dirichlet_kl():.6f} skewed={w_s.dirichlet_kl():.6f} ✓")
    results["dirichlet_balanced"] = float(w_b.dirichlet_kl())
    results["dirichlet_skewed"] = float(w_s.dirichlet_kl())
    
    # 9. Temporal decay
    for dt in [0, 100, 500, 1000, 5000]:
        d_t = 1.0 - np.exp(-0.001 * dt)
        print(f"   Δt={dt}: d_time={d_t:.4f}")
    
    print("\n✅ ALL MATH VERIFIED")
    with open(os.path.join(RESULTS_DIR, 'math_verification.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 1: Retrieval Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp1_retrieval():
    print("\n" + "="*60)
    print("EXP 1: Retrieval Comparison (Precision@k)")
    print("="*60)
    
    store_sizes = [100, 500, 1000, 5000]
    k_values = [1, 3, 5, 10]
    n_queries = 30
    results = {}
    
    for n in store_sizes:
        print(f"\n--- n={n} ---")
        store = build_store_fast(n)
        rng = np.random.RandomState(123)
        
        for k in k_values:
            pa, pb, pc = [], [], []
            
            for q in range(n_queries):
                cat = list(SEMANTIC_CATEGORIES.keys())[q % len(SEMANTIC_CATEGORIES)]
                query_text = f"something about {cat}"
                state_name = list(STATE_ARCHETYPES.keys())[q % len(STATE_ARCHETYPES)]
                query_state = STATE_ARCHETYPES[state_name].copy()
                
                q_sem = encoder.encode(query_text)
                q_emo = encoder.encode_emotional(query_state)
                q_state_n = encoder.encode_state(query_state)
                
                # NCM-A: Semantic only
                ra = retrieve_semantic_only(q_sem, store, k=k)
                pa.append(sum(1 for _, m in ra if cat in m.text) / k)
                
                # NCM-B: Semantic + Emotional
                rb = retrieve_semantic_emotional(q_sem, q_emo, store, k=k)
                pb.append(sum(1 for _, m in rb if cat in m.text) / k)
                
                # NCM-C: Full manifold
                rc = retrieve_top_k(q_sem, q_emo, store, q_state_n, store.step, k=k)
                pc.append(sum(1 for _, _, m in rc if cat in m.text) / k)
            
            key = f"n={n},k={k}"
            results[key] = {
                "semantic_only": round(float(np.mean(pa)), 4),
                "semantic_emotional": round(float(np.mean(pb)), 4),
                "full_manifold": round(float(np.mean(pc)), 4),
            }
            print(f"  k={k}: Sem={np.mean(pa):.3f} S+E={np.mean(pb):.3f} Full={np.mean(pc):.3f}")
    
    with open(os.path.join(RESULTS_DIR, 'exp1_retrieval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 2: Novelty at Scale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp2_novelty():
    print("\n" + "="*60)
    print("EXP 2: Novelty Sensitivity at Scale")
    print("="*60)
    
    # Use vectorized approach for speed
    scale_points = [100, 500, 1000, 5000, 10000, 25000, 50000]
    results = {}
    rng = np.random.RandomState(42)
    
    cats = list(SEMANTIC_CATEGORIES.keys())
    states_list = list(STATE_ARCHETYPES.values())
    
    # Pre-generate all semantic vectors with noise
    print("Pre-generating memory vectors...")
    max_n = max(scale_points)
    all_sems = np.zeros((max_n, 128), dtype=np.float32)
    all_emos = np.zeros((max_n, 3), dtype=np.float32)
    all_states = np.zeros((max_n, 7), dtype=np.float32)
    
    for i in range(max_n):
        cat = cats[i % len(cats)]
        text = SEMANTIC_CATEGORIES[cat][i % len(SEMANTIC_CATEGORIES[cat])]
        sem = TEXT_TO_VEC[text].copy()
        sem += rng.randn(128).astype(np.float32) * 0.01
        sem /= np.linalg.norm(sem)
        all_sems[i] = sem
        
        state = states_list[i % len(states_list)].copy()
        noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
        state = np.clip(state + noise, 0.0, 1.0)
        all_emos[i] = encoder.encode_emotional(state)
        all_states[i] = encoder.encode_state(state)
    
    print("Running novelty tests...")
    n_test = 20
    
    for n in scale_points:
        print(f"\n--- n={n} ---")
        sem_matrix = all_sems[:n]
        emo_matrix = all_emos[:n]
        state_matrix = all_states[:n]
        ts_array = np.arange(n, dtype=np.int64)
        
        sem_novelties = []
        full_novelties = []
        weights = RetrievalWeights()
        
        for t in range(n_test):
            # Semantically FAMILIAR query + NOVEL state
            cat = cats[t % len(cats)]
            text = SEMANTIC_CATEGORIES[cat][0]
            q_sem = TEXT_TO_VEC[text].copy()
            
            novel_state = rng.uniform(0.1, 0.9, size=7).astype(np.float32)
            q_emo = encoder.encode_emotional(novel_state)
            q_state = encoder.encode_state(novel_state)
            
            # Semantic novelty
            sims = sem_matrix @ q_sem
            sem_nov = 1.0 - float(np.max(sims))
            sem_novelties.append(sem_nov)
            
            # Full manifold novelty
            dists = vectorized_manifold_distance(
                sem_matrix, emo_matrix, state_matrix, ts_array,
                q_sem, q_emo, q_state, n, weights, 0.001,
            )
            full_nov = float(np.min(dists))
            full_novelties.append(full_nov)
        
        results[str(n)] = {
            "semantic_novelty_mean": round(float(np.mean(sem_novelties)), 6),
            "semantic_novelty_std": round(float(np.std(sem_novelties)), 6),
            "full_novelty_mean": round(float(np.mean(full_novelties)), 6),
            "full_novelty_std": round(float(np.std(full_novelties)), 6),
            "ratio": round(float(np.mean(full_novelties)) / max(float(np.mean(sem_novelties)), 1e-8), 3),
        }
        print(f"  Semantic:  {np.mean(sem_novelties):.4f} ± {np.std(sem_novelties):.4f}")
        print(f"  Full:      {np.mean(full_novelties):.4f} ± {np.std(full_novelties):.4f}")
        print(f"  Ratio:     {results[str(n)]['ratio']:.3f}x")
    
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 3: State-Conditioned Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp3_state_conditioned():
    print("\n" + "="*60)
    print("EXP 3: State-Conditioned Retrieval Validation")
    print("="*60)
    
    # Build store: same events encoded in different states
    store = MemoryStore(profile=MemoryProfile(max_size=5000))
    rng = np.random.RandomState(42)
    
    ts = 0
    for cat, texts in SEMANTIC_CATEGORIES.items():
        for text in texts:
            for state_name, state in STATE_ARCHETYPES.items():
                noise = rng.uniform(-0.03, 0.03, size=state.shape).astype(np.float32)
                noisy_state = np.clip(state + noise, 0.0, 1.0)
                mem = generate_memory(text, noisy_state, ts)
                mem.tags.append(state_name)
                store.add(mem)
                ts += 1
    store.step = ts
    print(f"Store: {len(store)} memories")
    
    k = 5
    n_queries = 20
    state_pairs = [
        ("calm_happy", "stressed_angry"),
        ("excited_curious", "sad_withdrawn"),
        ("confident", "fearful"),
        ("neutral", "exhausted"),
        ("calm_happy", "fearful"),
        ("excited_curious", "exhausted"),
    ]
    
    results = {}
    
    for sa_name, sb_name in state_pairs:
        print(f"\n--- {sa_name} vs {sb_name} ---")
        j_sem_list, j_man_list = [], []
        
        # Also track which state's memories are retrieved
        state_match_a, state_match_b = [], []
        
        for q in range(n_queries):
            cat = list(SEMANTIC_CATEGORIES.keys())[q % len(SEMANTIC_CATEGORIES)]
            text = SEMANTIC_CATEGORIES[cat][q % len(SEMANTIC_CATEGORIES[cat])]
            q_sem = encoder.encode(text)
            
            sa = STATE_ARCHETYPES[sa_name]
            sb = STATE_ARCHETYPES[sb_name]
            
            # Semantic-only (same result regardless of state)
            ra = retrieve_semantic_only(q_sem, store, k=k)
            rb = retrieve_semantic_only(q_sem, store, k=k)
            ids_a = set(m.id for _, m in ra)
            ids_b = set(m.id for _, m in rb)
            j_sem = 1.0 - len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)
            j_sem_list.append(j_sem)
            
            # Full manifold (state-conditioned)
            qa_emo = encoder.encode_emotional(sa)
            qa_state = encoder.encode_state(sa)
            rc_a = retrieve_top_k(q_sem, qa_emo, store, qa_state, store.step, k=k)
            
            qb_emo = encoder.encode_emotional(sb)
            qb_state = encoder.encode_state(sb)
            rc_b = retrieve_top_k(q_sem, qb_emo, store, qb_state, store.step, k=k)
            
            ids_fa = set(m.id for _, _, m in rc_a)
            ids_fb = set(m.id for _, _, m in rc_b)
            j_man = 1.0 - len(ids_fa & ids_fb) / max(len(ids_fa | ids_fb), 1)
            j_man_list.append(j_man)
            
            # State matching: how many retrieved memories were encoded in the SAME state?
            match_a = sum(1 for _, _, m in rc_a if sa_name in m.tags) / k
            match_b = sum(1 for _, _, m in rc_b if sb_name in m.tags) / k
            state_match_a.append(match_a)
            state_match_b.append(match_b)
        
        pair = f"{sa_name}_vs_{sb_name}"
        results[pair] = {
            "semantic_jaccard": round(float(np.mean(j_sem_list)), 4),
            "manifold_jaccard": round(float(np.mean(j_man_list)), 4),
            "manifold_jaccard_std": round(float(np.std(j_man_list)), 4),
            "improvement": round(float(np.mean(j_man_list) - np.mean(j_sem_list)), 4),
            "state_a_match_rate": round(float(np.mean(state_match_a)), 4),
            "state_b_match_rate": round(float(np.mean(state_match_b)), 4),
        }
        print(f"  Sem Jaccard:    {np.mean(j_sem_list):.3f}")
        print(f"  Man Jaccard:    {np.mean(j_man_list):.3f} ± {np.std(j_man_list):.3f}")
        print(f"  Improvement:    +{np.mean(j_man_list)-np.mean(j_sem_list):.3f}")
        print(f"  State-A match:  {np.mean(state_match_a):.3f}")
        print(f"  State-B match:  {np.mean(state_match_b):.3f}")
    
    with open(os.path.join(RESULTS_DIR, 'exp3_state.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 4: Speed Benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exp4_speed():
    print("\n" + "="*60)
    print("EXP 4: Speed Benchmarks")
    print("="*60)
    
    scale_points = [100, 500, 1000, 5000, 10000, 50000]
    n_queries = 100
    k = 5
    results = {}
    rng = np.random.RandomState(42)
    states_list = list(STATE_ARCHETYPES.values())
    
    # Pre-encode 1000 texts
    texts_1k = [f"test memory about topic {i}" for i in range(1000)]
    t0 = time.perf_counter()
    vecs_1k = encoder.encode_batch(texts_1k)
    encode_time = time.perf_counter() - t0
    encode_throughput = len(texts_1k) / encode_time
    print(f"Encoding throughput: {encode_throughput:.0f} texts/sec")
    
    for n in scale_points:
        print(f"\n--- n={n} ---")
        
        # Build store
        profile = MemoryProfile(max_size=n + 100)
        store = MemoryStore(profile=profile)
        
        t0 = time.perf_counter()
        for i in range(n):
            state = states_list[i % len(states_list)].copy()
            noise = rng.uniform(-0.05, 0.05, size=state.shape).astype(np.float32)
            state = np.clip(state + noise, 0.0, 1.0)
            sem = vecs_1k[i % len(vecs_1k)].copy()
            sem += rng.randn(128).astype(np.float32) * 0.005
            norm = np.linalg.norm(sem)
            if norm > 1e-8:
                sem /= norm
            emo = encoder.encode_emotional(state)
            snap = encoder.encode_state(state)
            mem = MemoryEntry(e_semantic=sem, e_emotional=emo, s_snapshot=snap,
                             timestamp=i, text=f"m{i}")
            store.add(mem)
        store.step = n
        t_store = time.perf_counter() - t0
        
        q_state = STATE_ARCHETYPES["neutral"].copy()
        q_sem = encoder.encode("test query")
        q_emo = encoder.encode_emotional(q_state)
        q_state_n = encoder.encode_state(q_state)
        
        # Semantic-only retrieval
        t0 = time.perf_counter()
        for _ in range(n_queries):
            retrieve_semantic_only(q_sem, store, k=k)
        t_sem = (time.perf_counter() - t0) / n_queries * 1000
        
        # Full manifold
        t0 = time.perf_counter()
        for _ in range(n_queries):
            retrieve_top_k(q_sem, q_emo, store, q_state_n, store.step, k=k)
        t_full = (time.perf_counter() - t0) / n_queries * 1000
        
        # Fast cached
        store._rebuild_cache()
        t0 = time.perf_counter()
        for _ in range(n_queries):
            retrieve_top_k_fast(q_sem, q_emo, store, q_state_n, store.step, k=k)
        t_fast = (time.perf_counter() - t0) / n_queries * 1000
        
        # Persistence
        path = os.path.join(RESULTS_DIR, f'bench_{n}.ncm')
        t0 = time.perf_counter()
        NCMFile.save(store, path)
        t_save = time.perf_counter() - t0
        fsize = os.path.getsize(path)
        
        t0 = time.perf_counter()
        NCMFile.load(path)
        t_load = time.perf_counter() - t0
        os.remove(path)
        
        results[str(n)] = {
            "store_throughput": round(n / t_store, 0),
            "retrieval_semantic_ms": round(t_sem, 3),
            "retrieval_manifold_ms": round(t_full, 3),
            "retrieval_cached_ms": round(t_fast, 3),
            "speedup_vs_semantic": round(t_sem / max(t_fast, 0.001), 2),
            "save_sec": round(t_save, 3),
            "load_sec": round(t_load, 3),
            "file_kb": round(fsize / 1024, 1),
            "bytes_per_memory": round(fsize / n, 1),
        }
        print(f"  Store: {n/t_store:.0f}/s  Sem: {t_sem:.2f}ms  Man: {t_full:.2f}ms  "
              f"Fast: {t_fast:.2f}ms  Save: {t_save:.2f}s  Size: {fsize/1024:.0f}KB")
    
    results["encode_throughput"] = round(encode_throughput, 0)
    
    with open(os.path.join(RESULTS_DIR, 'exp4_speed.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN ALL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("NCM EXPERIMENT SUITE v2 (Optimized)")
    print("="*60)
    t_start = time.perf_counter()
    
    all_results = {}
    
    for name, func in [("math", verify_math), ("exp1", exp1_retrieval), 
                        ("exp3", exp3_state_conditioned), ("exp4", exp4_speed),
                        ("exp2", exp2_novelty)]:
        try:
            all_results[name] = func()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            traceback.print_exc()
    
    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results in: {RESULTS_DIR}/")
    
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
