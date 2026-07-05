"""
MemoryAgentBench Competencies Benchmark for NCM
===============================================
Evaluates NCM on the four core competencies identified in MemoryAgentBench:
1. Accurate Retrieval
2. Test-Time Learning
3. Long-Range Understanding
4. Selective Forgetting
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

def get_store(enable_contradiction=False):
    profile = MemoryProfile()
    if enable_contradiction:
        profile.set_custom("enable_contradiction_awareness", True)
        profile.set_custom("contradiction_penalty", 1.0)
        profile.set_custom("contradiction_requires_marker", True)
        profile.set_custom("write_conflict_trace", True)
    return MemoryStore(profile=profile)

def create_entry(store, text, tags=None):
    sem = encoder.encode(text)
    state = store.auto_state.get_current_state()
    emo = encoder.encode_emotional(state)
    s_snap = encoder.encode_state(state)
    entry = MemoryEntry(
        e_semantic=sem, e_emotional=emo, s_snapshot=s_snap,
        timestamp=store.step, text=text, tags=tags or [],
        strength=1.5
    )
    return entry

def query_system(store, text):
    q_sem = encoder.encode(text)
    state = store.auto_state.get_current_state()
    q_emo = encoder.encode_emotional(state)
    s_norm = encoder.encode_state(state)
    return retrieve_top_k(q_sem, q_emo, store, s_norm, store.step, k=3)

def test_accurate_retrieval():
    print("\n--- Task 1: Accurate Retrieval ---")
    store = get_store()
    
    facts = [
        "The project deadline is next Friday.",
        "The server IP address is 192.168.1.100.",
        "The access code to the vault is 7890.",
        "John likes to drink green tea in the morning.",
        "The meeting is scheduled in room 4B."
    ]
    for f in facts:
        store.add(create_entry(store, f), update_auto_state=True)
        store.tick()
        
    # Query something related to the access code
    results = query_system(store, "What is the vault access code?")
    top_text = results[0][2].text if results else "None"
    
    success = "7890" in top_text
    print(f"Query: 'What is the vault access code?'")
    print(f"Top result: '{top_text}'")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success

def test_test_time_learning():
    print("\n--- Task 2: Test-Time Learning ---")
    store = get_store()
    
    # Check what happens when query before learning
    results_before = query_system(store, "What is the capital of Mars?")
    
    # Learn fact at test time
    print("Learning new fact...")
    store.add(create_entry(store, "The newly established capital of Mars is Olympus City."), update_auto_state=True)
    store.tick()
    
    # Query again
    results_after = query_system(store, "What is the capital of Mars?")
    top_text_after = results_after[0][2].text if results_after else "None"
    
    success = "Olympus City" in top_text_after
    print(f"Query after learning: 'What is the capital of Mars?'")
    print(f"Top result: '{top_text_after}'")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success

def test_long_range_understanding():
    print("\n--- Task 3: Long-Range Understanding ---")
    store = get_store()
    
    # Insert core fact at step 0
    core_fact = "The secret password to override the system is 'DELTA-9'."
    store.add(create_entry(store, core_fact), update_auto_state=True)
    store.tick()
    
    # Insert 500 distractors and age the system
    print("Simulating passage of time and distractor tasks (500 steps)...")
    for i in range(500):
        store.add(create_entry(store, f"Distractor log entry {i}: standard operations nominal."), update_auto_state=True)
        store.tick()
        
    # Consolidate
    merged = store.consolidate(0.95)
    print(f"System age: {store.step} ticks. Consolidations: {merged}")
    
    # Query core fact
    results = query_system(store, "What is the override password?")
    top_text = results[0][2].text if results else "None"
    
    success = "DELTA-9" in top_text
    print(f"Top result: '{top_text}'")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success

def test_selective_forgetting():
    print("\n--- Task 4: Selective Forgetting ---")
    store = get_store(enable_contradiction=True)
    
    # Insert initial fact
    fact1 = "The current project manager is Alice."
    m1 = store.add(create_entry(store, fact1), update_auto_state=True)
    store.tick()
    
    # Insert some noise
    for i in range(5):
        store.add(create_entry(store, f"Noise {i}"), update_auto_state=True)
        store.tick()
        
    # Insert correction
    fact2 = "Correction: The current project manager is Bob."
    m2 = store.add(create_entry(store, fact2), update_auto_state=True)
    store.tick()
    
    results = query_system(store, "Who is the project manager?")
    
    found_bob = False
    found_alice = False
    
    for _, _, m in results:
        if "Bob" in m.text:
            found_bob = True
            print(f"Top result: '{m.text}'")
            break
        elif "Alice" in m.text:
            found_alice = True
            
    # Check conflict trace was created
    traces = [m for m in store.get_all_safe() if m.is_conflict_trace]
    
    success = found_bob and not found_alice and len(traces) > 0
    print(f"Result: {'PASS' if success else 'FAIL'}")
    if not success:
        print(f"Found Bob: {found_bob}, Found Alice: {found_alice}, Conflict traces created: {len(traces)}")
        for _, _, m in results:
            print(f"  - returned: {m.text}")
    else:
        print(f"Conflict traces successfully managed obsolete memory.")
    return success

if __name__ == "__main__":
    t0 = time.perf_counter()
    r1 = test_accurate_retrieval()
    r2 = test_test_time_learning()
    r3 = test_long_range_understanding()
    r4 = test_selective_forgetting()
    
    print("\n==================================")
    print("MemoryAgentBench NCM Results:")
    print(f"1. Accurate Retrieval:       {'PASS' if r1 else 'FAIL'}")
    print(f"2. Test-Time Learning:       {'PASS' if r2 else 'FAIL'}")
    print(f"3. Long-Range Understanding: {'PASS' if r3 else 'FAIL'}")
    print(f"4. Selective Forgetting:     {'PASS' if r4 else 'FAIL'}")
    print("==================================")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")
