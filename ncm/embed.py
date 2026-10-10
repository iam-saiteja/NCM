"""Text -> unit vectors. Wraps sentence-transformers; any callable works too."""

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval-tuned models expect an instruction on the query side.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
}
DOC_PREFIX = {"intfloat/e5-small-v2": "passage: "}


class Embedder:
    """Lazy sentence-transformers model. Loads on first use, on GPU when present."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = None):
        self.name = model
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def __call__(self, texts, query: bool = False) -> np.ndarray:
        prefix = (QUERY_PREFIX if query else DOC_PREFIX).get(self.name, "")
        vecs = self._load().encode([prefix + t for t in texts], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class FunctionEmbedder:
    """Adapts any `fn(list[str]) -> array (n, d)`, e.g. an API client. Rows are L2-normalised."""

    def __init__(self, fn, name: str = "custom"):
        self.fn, self.name = fn, name

    def __call__(self, texts, query: bool = False) -> np.ndarray:
        v = np.asarray(self.fn(list(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def as_embedder(e):
    if e is None:
        return Embedder()
    if isinstance(e, str):
        return Embedder(e)
    if isinstance(e, (Embedder, FunctionEmbedder)) or (callable(e) and hasattr(e, "name")):
        return e  # anything with .name and __call__(texts, query=False) -> (n, d) unit rows
    if callable(e):
        return FunctionEmbedder(e, getattr(e, "__name__", "custom"))
    raise TypeError(f"embedder must be a model name, an Embedder or a callable, got {type(e)}")
