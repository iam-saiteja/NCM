"""Embed every unique text once per (dataset, model) and cache it on disk."""

import hashlib
import os

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), ".cache")

# name -> (hf id, query prefix, document prefix)
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: ", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5",
                 "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
}

_loaded = {}
_tables = {}   # path -> {sha1: vector}, read from disk once per process


def model(name):
    if name not in _loaded:
        from sentence_transformers import SentenceTransformer
        _loaded[name] = SentenceTransformer(MODELS[name][0], device="cuda")
    return _loaded[name]


def _key(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def embed(texts, name, kind="doc", tag="misc", batch_size=128):
    """Unit vectors (len(texts), dim) float32. kind is "doc" or "query"."""
    prefix = MODELS[name][1] if kind == "query" else MODELS[name][2]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}-{kind}-{name}.npz")
    if path not in _tables:
        _tables[path] = {}
        if os.path.exists(path):
            z = np.load(path)
            _tables[path] = dict(zip(z["keys"].tolist(), z["emb"]))
    table = _tables[path]
    keys = [_key(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in table}.items())
    if missing:
        vecs = model(name).encode([prefix + t for _, t in missing], batch_size=batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=len(missing) > 5000)
        table.update({k: v.astype(np.float16) for (k, _), v in zip(missing, vecs)})
        np.savez(path, keys=np.array(list(table.keys())), emb=np.stack(list(table.values())))
    return np.stack([table[k] for k in keys]).astype(np.float32)
