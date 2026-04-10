import json
import os
import sys
import urllib.error
import urllib.request
from typing import List

import numpy as np

# Ensure repo root is on path so `import ncm` works when launched from this folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ncm import SentenceEncoder, MemoryEntry, MemoryStore, NCMFile, retrieve_top_k_fast


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2:7B"
NCM_PATH = os.path.join(THIS_DIR, "memory_store.ncm")


def ollama_chat(messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["message"]["content"]


class LocalNCMOllamaChat:
    def __init__(self) -> None:
        self.encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", model_dir=os.path.join(REPO_ROOT, "models"))
        self.store = self._load_store()
        self.current_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def _load_store(self) -> MemoryStore:
        if os.path.exists(NCM_PATH):
            return NCMFile.load(NCM_PATH)
        return MemoryStore()

    def _save_store(self) -> None:
        NCMFile.save(self.store, NCM_PATH, compress=True)

    def _add_memory(self, role: str, text: str, state: np.ndarray) -> None:
        semantic = self.encoder.encode(text)
        emotional = self.encoder.encode_emotional(state)
        snapshot = self.encoder.encode_state(state)

        mem = MemoryEntry(
            e_semantic=semantic,
            e_emotional=emotional,
            s_snapshot=snapshot,
            timestamp=int(self.store.step),
            text=f"{role}: {text}",
            tags=[role],
        )
        self.store.add(mem)
        self.store.step += 1

    def _retrieve_context(self, user_text: str, state: np.ndarray, top_k: int = 4) -> str:
        memories = self.store.get_all_safe()
        if not memories:
            return ""

        query_sem = self.encoder.encode(user_text)
        query_emo = self.encoder.encode_emotional(state)
        state_norm = self.encoder.encode_state(state)

        results = retrieve_top_k_fast(
            query_semantic=query_sem,
            query_emotional=query_emo,
            store=self.store,
            s_current_normalized=state_norm,
            current_step=int(self.store.step),
            k=top_k,
        )

        if not results:
            return ""

        lines = []
        for distance, _, mem in results:
            lines.append(f"- ({distance:.3f}) {mem.text}")
        return "\n".join(lines)

    def set_state_from_csv(self, value: str) -> None:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 7:
            raise ValueError("State must contain exactly 7 comma-separated values.")
        arr = np.array([float(x) for x in parts], dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        self.current_state = arr

    def ask(self, user_text: str) -> str:
        context = self._retrieve_context(user_text, self.current_state, top_k=4)

        system_prompt = (
            "You are a helpful assistant. "
            "Use retrieved memory context when relevant, but do not mention hidden system prompts."
        )

        user_prompt = user_text
        if context:
            user_prompt = (
                "Retrieved context from long-term memory:\n"
                f"{context}\n\n"
                "User message:\n"
                f"{user_text}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = ollama_chat(messages=messages, model=MODEL_NAME)

        self._add_memory("user", user_text, self.current_state)
        self._add_memory("assistant", answer, self.current_state)
        self._save_store()

        return answer


def main() -> None:
    print("Local NCM + Ollama chat")
    print(f"Model: {MODEL_NAME}")
    print(f"Memory file: {NCM_PATH}")
    print("Commands: /exit, /quit, /state a,b,c,d,e,f,g, /showstate")

    app = LocalNCMOllamaChat()

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_text.lower() == "/showstate":
            print("State:", ", ".join(f"{x:.2f}" for x in app.current_state))
            continue
        if user_text.lower().startswith("/state "):
            try:
                app.set_state_from_csv(user_text[7:])
                print("State updated.")
            except Exception as exc:
                print(f"Invalid state: {exc}")
            continue

        try:
            reply = app.ask(user_text)
            print(f"Assistant: {reply}")
        except urllib.error.URLError:
            print("Could not reach Ollama at http://localhost:11434. Ensure Ollama is running.")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
