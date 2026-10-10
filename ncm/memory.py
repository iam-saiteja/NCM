"""NCM Memory: episodic memory for conversational agents.

Ranking is relevance only: dense similarity fused with BM25, then spread to the
turns around each strong hit in the same session (temporal contiguity). What is
*kept* is decided separately, by an ACT-R style activation (use count, age,
emotional salience), so forgetting never distorts ranking.
"""

import io
import json
import math
import os
import re
import threading
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from ncm.embed import as_embedder
from ncm.lexical import BM25

FORMAT_VERSION = 3


@dataclass
class Config:
    lexical_weight: float = 0.75     # BM25 weight next to dense, both z-scored per query
    # Contiguity spreading: a strong hit lifts the turns around it in the ranking. Helps when
    # evidence spans adjacent dialogue turns (LoCoMo R@5 +0.055 at 0.75/3.0) and hurts when it
    # sits in single turns between long replies (LongMemEval -0.050), so it is off by default;
    # context() shows each hit's neighbouring turns instead.
    neighbour_weight: float = 0.0
    neighbour_width: float = 3.0     # turns; kernel exp(-distance / width)
    session_gap: float = 1800.0      # seconds of silence that start a new session
    max_items: int = 0               # 0 = unlimited; above it the least active memory fades
    decay: float = 0.5               # ACT-R base-level decay d


@dataclass
class Recalled:
    id: int
    text: str
    speaker: str
    time: float
    session: int
    score: float
    meta: dict = field(default_factory=dict)

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.time, tz=timezone.utc)


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def score(dense, lexical, sessions, cfg: Config) -> np.ndarray:
    """Relevance of every memory to one query. Higher is better.

    dense, lexical: (N,) similarities; sessions: (N,) session id per row, rows in time order.
    """
    s = _z(dense.astype(np.float32))
    if cfg.lexical_weight and lexical is not None:
        s = s + cfg.lexical_weight * _z(lexical.astype(np.float32))
    if cfg.neighbour_weight and len(s) > 1:
        lift = np.maximum(s, 0)
        nb = np.zeros_like(s)
        for o in range(1, min(len(s), int(math.ceil(3 * cfg.neighbour_width))) + 1):
            same = sessions[o:] == sessions[:-o]
            w = math.exp(-o / cfg.neighbour_width)
            np.maximum(nb[:-o], np.where(same, lift[o:] * w, 0), out=nb[:-o])
            np.maximum(nb[o:], np.where(same, lift[:-o] * w, 0), out=nb[o:])
        s = s + cfg.neighbour_weight * nb
    return s


def _to_ts(t) -> float:
    if t is None:
        return _time.time()
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()
    return float(t)


def _norm_text(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


class Memory:
    """Long-term memory for one agent (or one user, or one character).

        mem = Memory("alice.ncm")
        mem.add("I adopted a beagle called Max", speaker="user")
        mem.recall("what is my dog's name?")
        prompt_block = mem.context("what is my dog's name?")
        mem.save()
    """

    def __init__(self, path: str = None, embedder=None, config: Config = None):
        self.path = path
        self.embedder = as_embedder(embedder)
        self.config = config or Config()
        self._lock = threading.RLock()
        self._reset()
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.ids, self.texts, self.speakers, self.metas = [], [], [], []
        self._emb = None                       # (capacity, dim) buffer
        self._times = np.zeros(0)
        self._sessions = np.zeros(0, dtype=np.int64)
        self._uses = np.zeros(0, dtype=np.int32)
        self._salience = np.zeros(0, dtype=np.float32)
        self._bm25 = BM25()
        self._index = {}                       # (speaker, normalised text) -> id
        self._rows = {}                        # id -> row
        self._next_id = 0
        self._session = 0
        self._new_session = False

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"Memory({len(self)} memories, {self._session + 1} sessions, embedder={self.embedder.name!r})"

    @property
    def embeddings(self) -> np.ndarray:
        return self._emb[:len(self)] if self._emb is not None else np.zeros((0, 0), np.float32)

    def _row(self, mid: int) -> int:
        try:
            return self._rows[mid]
        except KeyError:
            raise KeyError(f"no memory with id {mid}") from None

    # ------------------------------------------------------------------ write
    def new_session(self):
        """Mark the next memory as the start of a new conversation."""
        self._new_session = True

    def add(self, text: str, speaker: str = "user", time=None, meta: dict = None) -> int:
        """Store one utterance. Returns its id. Exact repeats by the same speaker
        reinforce the existing memory instead of storing a copy."""
        return self.add_many([text], speakers=[speaker], times=[time], metas=[meta])[0]

    def add_many(self, texts, speakers=None, times=None, metas=None) -> list:
        """Store many utterances with one batched embedding call (much faster)."""
        texts = list(texts)
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("every memory needs non-empty text")
        n = len(texts)
        speakers = list(speakers) if speakers is not None else ["user"] * n
        times = list(times) if times is not None else [None] * n
        metas = list(metas) if metas is not None else [None] * n
        if not (len(speakers) == len(times) == len(metas) == n):
            raise ValueError("texts, speakers, times and metas must have the same length")
        with self._lock:
            out = [None] * n
            pending, fresh = {}, []
            for i, (t, sp, tm, me) in enumerate(zip(texts, speakers, times, metas)):
                key = (sp, _norm_text(t))
                if key in self._index:
                    out[i] = self._index[key]
                    self._uses[self._rows[out[i]]] += 1
                elif key in pending:
                    pending[key].append(i)
                else:
                    pending[key] = [i]
                    fresh.append((key, (t, sp, _to_ts(tm), me or {})))
            if fresh:
                vecs = self.embedder([f"{sp}: {t}" for _, (t, sp, _, _) in fresh])
                for (key, item), v in zip(fresh, vecs):
                    mid = self._index[key] = self._append(item, v)
                    self._uses[-1] += len(pending[key]) - 1
                    for i in pending[key]:
                        out[i] = mid
            self._forget_overflow()
            return out

    def _append(self, item, vec) -> int:
        text, speaker, ts, meta = item
        n = len(self)
        if n and (self._new_session or ts - self._times[-1] > self.config.session_gap):
            self._session += 1
        self._new_session = False
        if self._emb is None:
            self._emb = np.zeros((64, len(vec)), dtype=np.float32)
        elif len(vec) != self._emb.shape[1]:
            raise ValueError(f"embedding size {len(vec)} != store size {self._emb.shape[1]}")
        if n == len(self._emb):
            self._emb = np.concatenate([self._emb, np.zeros_like(self._emb)])
        self._emb[n] = vec
        self._times = np.append(self._times, ts)
        self._sessions = np.append(self._sessions, self._session)
        self._uses = np.append(self._uses, 0).astype(np.int32)
        self._salience = np.append(self._salience, np.float32(meta.get("salience", 0.0)))
        mid = self._next_id
        self._next_id += 1
        self._rows[mid] = n
        self.ids.append(mid)
        self.texts.append(text)
        self.speakers.append(speaker)
        self.metas.append(meta)
        self._bm25.add(f"{speaker}: {text}")
        return mid

    def add_messages(self, messages, time=None) -> list:
        """Store OpenAI-style [{"role": ..., "content": ...}] messages (system ones skipped)."""
        msgs = [m for m in messages if m.get("role") != "system" and str(m.get("content", "")).strip()]
        return self.add_many([m["content"] for m in msgs], speakers=[m["role"] for m in msgs],
                             times=[time] * len(msgs))

    # ------------------------------------------------------------ forgetting
    def activation(self, now: float = None) -> np.ndarray:
        """ACT-R base-level activation per memory: ln((uses+1)/(1-d)) - d*ln(age), plus
        emotional salience. High = easy to keep. Decides forgetting, not ranking."""
        now = _time.time() if now is None else now
        d = self.config.decay
        age_days = np.maximum((now - self._times) / 86400.0, 1.0 / 1440)
        return np.log((self._uses + 1) / (1 - d)) - d * np.log(age_days) + self._salience

    def _forget_overflow(self):
        cap = self.config.max_items
        while cap and len(self) > cap:
            self.forget(self.ids[int(np.argmin(self.activation()))])

    def forget(self, mid: int) -> None:
        """Delete one memory permanently."""
        with self._lock:
            row = self._row(mid)
            keep = np.arange(len(self)) != row
            emb = self.embeddings[keep]
            self._emb[:len(emb)] = emb
            self._times, self._sessions = self._times[keep], self._sessions[keep]
            self._uses, self._salience = self._uses[keep], self._salience[keep]
            for lst in (self.ids, self.texts, self.speakers, self.metas):
                del lst[row]
            self._index = {k: v for k, v in self._index.items() if v != mid}
            self._rows = {m: i for i, m in enumerate(self.ids)}
            self._bm25 = BM25()
            for t, sp in zip(self.texts, self.speakers):
                self._bm25.add(f"{sp}: {t}")  # ponytail: O(N) rebuild per delete; fine below ~1e5 deletes

    # ------------------------------------------------------------------- read
    def scores(self, query: str, speaker: str = None) -> np.ndarray:
        """Relevance of every memory to `query` (higher is better), -inf for filtered rows."""
        with self._lock:
            if not len(self):
                return np.zeros(0, dtype=np.float32)
            q = self.embedder([query], query=True)[0]
            s = score(self.embeddings @ q, self._bm25.scores(query), self._sessions, self.config)
            if speaker is not None:
                s = np.where(np.asarray(self.speakers) == speaker, s, -np.inf)
            return s

    def recall(self, query: str, k: int = 5, speaker: str = None, reinforce: bool = False) -> list:
        """The k memories most relevant to `query`, best first."""
        with self._lock:
            s = self.scores(query, speaker)
            if not len(s):
                return []
            k = min(k, int(np.isfinite(s).sum()))
            top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
            top = top[np.argsort(-s[top], kind="stable")]
            if reinforce:
                self._uses[top] += 1
            return [self._recalled(int(r), float(s[r])) for r in top]

    def _recalled(self, row: int, sc: float = 0.0) -> Recalled:
        return Recalled(self.ids[row], self.texts[row], self.speakers[row], float(self._times[row]),
                        int(self._sessions[row]), sc, self.metas[row])

    def get(self, mid: int) -> Recalled:
        return self._recalled(self._row(mid))

    def all(self) -> list:
        return [self._recalled(r) for r in range(len(self))]

    def context(self, query: str, k: int = 8, persona_of: str = "user", persona_k: int = 6,
                window: int = 1) -> str:
        """A prompt-ready block: who the person is, then the memories relevant to `query`
        (each with `window` turns of surrounding context), oldest first and dated so a
        model can tell which of two conflicting statements is the latest."""
        from ncm.persona import persona
        with self._lock:
            hits = self.recall(query, k=k, reinforce=True)
            rows = set()
            for h in hits:
                r = self._row(h.id)
                for o in range(-window, window + 1):
                    if 0 <= r + o < len(self) and self._sessions[r + o] == self._sessions[r]:
                        rows.add(r + o)
            parts = []
            facts = persona(self, speaker=persona_of, k=persona_k) if persona_of else []
            if facts:
                parts.append(f"What you know about {persona_of}:\n" + "\n".join(f"- {f}" for f in facts))
            if rows:
                lines, last = [], None
                for r in sorted(rows, key=lambda r: (self._times[r], r)):
                    if last is not None and self._sessions[r] != self._sessions[last]:
                        lines.append("")
                    day = datetime.fromtimestamp(self._times[r], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"[{day}] {self.speakers[r]}: {self.texts[r]}")
                    last = r
                parts.append("Relevant memories (oldest first):\n" + "\n".join(lines))
            return "\n\n".join(parts)

    # ------------------------------------------------------------ persistence
    def save(self, path: str = None) -> str:
        """Write everything to one file, atomically. No pickle: safe to load untrusted files."""
        path = path or self.path
        if not path:
            raise ValueError("no path given")
        with self._lock:
            meta = {"format": "ncm", "version": FORMAT_VERSION, "embedder": self.embedder.name,
                    "config": asdict(self.config), "next_id": self._next_id,
                    "session": self._session, "ids": self.ids, "texts": self.texts,
                    "speakers": self.speakers, "metas": self.metas}
            buf = io.BytesIO()
            np.savez_compressed(
                buf, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
                emb=self.embeddings.astype(np.float16), times=self._times,
                sessions=self._sessions, uses=self._uses, salience=self._salience)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                f.write(buf.getvalue())
            os.replace(tmp, path)
            self.path = path
            return path

    @classmethod
    def load(cls, path: str, embedder=None, config: Config = None) -> "Memory":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return cls(path, embedder=embedder, config=config)

    def _load(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
            if meta.get("format") != "ncm" or meta.get("version") != FORMAT_VERSION:
                raise ValueError(f"{path} is not an NCM v{FORMAT_VERSION} file "
                                 "(use ncm.legacy.import_v1 for old .ncm files)")
            if meta["embedder"] != self.embedder.name:
                raise ValueError(f"{path} was embedded with {meta['embedder']!r}, "
                                 f"not {self.embedder.name!r}; pass embedder={meta['embedder']!r}")
            emb = z["emb"].astype(np.float32)
            self._times, self._sessions = z["times"], z["sessions"]
            self._uses, self._salience = z["uses"], z["salience"]
        if self.config == Config():
            self.config = Config(**meta["config"])
        self.ids, self.texts = meta["ids"], meta["texts"]
        self.speakers, self.metas = meta["speakers"], meta["metas"]
        self._next_id, self._session = meta["next_id"], meta["session"]
        self._emb = emb if len(emb) else None
        self._rows = {m: i for i, m in enumerate(self.ids)}
        for i, t, sp in zip(self.ids, self.texts, self.speakers):
            self._bm25.add(f"{sp}: {t}")
            self._index[(sp, _norm_text(t))] = i
