# Ollama + NCM local chat

This folder provides a small local chat app that:

- calls your local Ollama model `qwen2:7B`
- retrieves memory context from NCM before each response
- stores the memory file (`memory_store.ncm`) in this same folder

## Run

From the repository root:

python ollama_ncm_qwen2/chat_with_ncm.py

Then chat in the terminal.

## Commands

- `/exit` or `/quit` to stop
- `/state a,b,c,d,e,f,g` to set the 7D state vector used for retrieval/write
- `/showstate` to print current state

## Notes

- Ollama must be running locally (`http://localhost:11434`).
- If `memory_store.ncm` does not exist yet, it is created automatically after first interaction.
