# Ollama + NCM local chat

This folder provides a small local chat app that:

- calls your local Ollama model (default `qwen2:7B`, configurable)
- retrieves memory context from NCM before each response
- stores the memory file (`memory_store.ncm`) in this same folder

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

python ollama_ncm_qwen2/chat_with_ncm.py --state 0.6,0.4,0.7,0.3,0.5,0.2,0.8

Then chat in the terminal.

## Commands

- `/exit` or `/quit` to stop
- `/state a,b,c,d,e,f,g` to set the 7D state vector used for retrieval/write
- `/showstate` to print current state

## State vector guide (7D)

The state vector is a normalized control signal used by NCM for state-conditioned retrieval.
Values are in `[0, 1]` for 7 dimensions.

The base system treats these as abstract latent dimensions. For practical use, map them to your own app semantics (example):

1. formality
2. emotional intensity
3. confidence
4. urgency
5. empathy
6. curiosity
7. task focus

Why this matters:
- If all values stay at `0.5`, early memories are encoded under nearly identical state conditions.
- Changing state over time increases separation in state space, making state-conditioned retrieval much more informative.

## Notes

- Ollama must be running locally (`http://localhost:11434`).
- If `memory_store.ncm` does not exist yet, it is created automatically after first interaction.
- Write-time novelty gating is active in chat storage (`gate_check=True`) to reduce near-duplicate memory writes.
- New stores are initialized with a tighter write threshold (`write_threshold=0.25`) for chat usage.
- Existing `.ncm` files keep their saved profile values (including thresholds), enabling stable behavior across restarts.
