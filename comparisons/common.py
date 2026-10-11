"""Shared pieces for the comparisons: a metered LLM client and the evaluation subsets."""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import load_locomo, load_longmemeval   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "..", "bench", ".cache", "comparisons")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
READER = os.environ.get("NCM_READER", "qwen2.5:3b")

# End-to-end subset: the two shortest LoCoMo conversations, so full-context fits a 3B
# model's window and LLM-based memory ingestion stays affordable. Every system answers
# every question in them (categories 1-4: 230 questions over 38 sessions, 788 turns).
E2E_CONVERSATIONS = ("conv-26", "conv-30")
# Retrieval subset of LongMemEval-S for the expensive encoders (SPLADE, bge-base, rerankers):
# the first 150 questions; every retrieval method is scored on the same ones.
LME_LIMIT = 150


class Meter:
    """Counts what an LLM-backed component spends."""

    def __init__(self):
        self.calls = self.prompt_tokens = self.output_tokens = 0
        self.seconds = 0.0

    def as_dict(self):
        return {"llm_calls": self.calls, "llm_prompt_tokens": self.prompt_tokens,
                "llm_output_tokens": self.output_tokens, "llm_seconds": round(self.seconds, 2)}


def chat(messages, meter=None, num_ctx=8192, num_predict=40, fmt=None, model=None):
    """One Ollama chat call, temperature 0. Returns the text and records cost on `meter`."""
    body = {"model": model or READER, "stream": False, "messages": messages,
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": num_predict}}
    if fmt:
        body["format"] = fmt
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.loads(r.read())
            if meter is not None:
                meter.calls += 1
                meter.prompt_tokens += out.get("prompt_eval_count", 0)
                meter.output_tokens += out.get("eval_count", 0)
                meter.seconds += time.perf_counter() - t0
            return out["message"]["content"].strip()
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("ollama unreachable")


def e2e_data():
    hays, queries = load_locomo()
    hays = [h for h in hays if h.id in E2E_CONVERSATIONS]
    queries = [q for q in queries if q.haystack in E2E_CONVERSATIONS]
    return hays, queries


def lme_data():
    return load_longmemeval(limit=LME_LIMIT)


def save(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
