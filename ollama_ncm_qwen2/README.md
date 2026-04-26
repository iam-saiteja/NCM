# Ollama + NCM local chat

This folder provides a small local chat app that:

- calls your local Ollama model (default `qwen2:7B`, configurable)
- retrieves memory context from NCM before each response
- stores the memory file (`memory_store.ncm`) in this same folder
- enables contradiction-aware retrieval (CADP) by default for real-time correction testing

## Run

From the repository root:

python ollama_ncm_qwen2/chat_with_ncm.py

Use a different local model if needed:

python ollama_ncm_qwen2/chat_with_ncm.py --model qwen2:0.5b
python ollama_ncm_qwen2/chat_with_ncm.py --model llama3.2

Or set model once via environment variable:

set OLLAMA_MODEL=llama3.2
python ollama_ncm_qwen2/chat_with_ncm.py

You can also set an initial state at startup:

python ollama_ncm_qwen2/chat_with_ncm.py --state 0.6,0.4,0.7,0.3,0.5

Then chat in the terminal.

## Commands

- `/exit` or `/quit` to stop
- `/state val,aro,dom,cur,str` to set the 5D auto-state used for retrieval/write
- `/showstate` to print current state and adaptive weights (`w_state`, `w_sem`)

## State vector guide (5D auto-state)

The auto-state vector is a normalized control signal used by NCM for state-conditioned retrieval.
Values are in `[0, 1]` for 5 fixed dimensions:

1. valence
2. arousal
3. dominance
4. curiosity
5. stress

Why this matters:
- Auto-state is normally updated automatically from each message by the tracker.
- `/state ...` is a real-time override for testing controlled state scenarios.
- If all values stay near `0.5`, early memories are encoded under similar state conditions.
- Changing state over time increases separation in state space, making state-conditioned retrieval more informative.

## Notes

- Ollama must be running locally (`http://localhost:11434`).
- If `memory_store.ncm` does not exist yet, it is created automatically after first interaction.
- Write-time novelty gating is active in chat storage (`gate_check=True`) to reduce near-duplicate memory writes.
- New stores are initialized with a tighter write threshold (`write_threshold=0.25`) for chat usage.
- Existing `.ncm` files keep their saved profile values (including thresholds), enabling stable behavior across restarts.
- Retrieval state comes from `store.auto_state.get_current_state()` and per-memory `auto_state_snapshot` fields.
- Contradiction-aware retrieval is enabled internally for correction handling; no extra runtime command is required.
- Contradiction metadata (`contradicted_by`, `is_conflict_trace`) is persisted in `.ncm` and reused across restarts.
