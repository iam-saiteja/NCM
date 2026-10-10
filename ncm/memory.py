"""NCM Memory: episodic memory for conversational agents.

Ranking is relevance only: dense similarity fused with BM25. What is *kept* is
decided separately, by an ACT-R style activation (use count, age, emotional
salience), so forgetting never distorts ranking. context() turns a query into a
prompt block: who the person is, plus the relevant memories with the turns around
them, dated and in order.
"""

import io
import json
import math
import os
import re
import threading
import zlib
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from ncm.affect import Affect, label, salience
from ncm.embed import as_embedder, as_reranker
from ncm.lexical import BM25
from ncm.persona import evidence, statements

FORMAT_VERSION = 3
LIFT_ANCHORS = 1024
_COLS = {"times": np.float64, "sessions": np.int64, "uses": np.int32, "salience": np.float32,
         "valence": np.float32, "arousal": np.float32, "evidence": np.float32, "stmt_words": np.int32,
         "doc": np.int64, "speaker": np.int32}
_DERIVED = ("doc", "speaker")              # rebuilt on load, not saved


@dataclass
class Config:
    lexical_weight: float = 0.75     # BM25 weight next to dense, both z-scored per query
    # Contiguity: a strong hit lifts the turns just before it in the same session (the
    # question that led to a matching answer). Measured on LoCoMo and LongMemEval: lifting
    # the previous turn helps both; also lifting following turns (0.75, width 3) helps LoCoMo
    # (+0.04 R@5) but costs LongMemEval -0.14, so lift_next is off by default.
    lift_previous: float = 0.75
    lift_next: float = 0.0
    lift_width: float = 1.0          # turns; kernel exp(-distance / width)
    session_gap: float = 1800.0      # seconds of silence that start a new session
    max_items: int = 0               # 0 = unlimited; when full, the least active 5% fade
    rerank_top: int = 30             # with a reranker: how many first-stage hits it re-scores
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
    if (cfg.lift_previous or cfg.lift_next) and len(s) > 1:
        # Only the strongest hits lift their neighbours (a weak hit's lift could never reach the
        # top); with <= LIFT_ANCHORS memories this is every positive score, i.e. exact.
        n = len(s)
        a = np.flatnonzero(s > 0) if n <= LIFT_ANCHORS else np.argpartition(-s, LIFT_ANCHORS)[:LIFT_ANCHORS]
        a = a[s[a] > 0]
        prev, nxt = np.zeros_like(s), np.zeros_like(s)
        for o in range(1, int(math.ceil(3 * cfg.lift_width)) + 1):
            w = math.exp(-o / cfg.lift_width)
            for out, tgt in ((prev, a - o), (nxt, a + o)):           # i+o lifts i / i lifts i+o
                ok = (tgt >= 0) & (tgt < n)
                src, tgt = a[ok], tgt[ok]
                ok = sessions[tgt] == sessions[src]
                np.maximum.at(out, tgt[ok], s[src[ok]] * w)
        s = s + cfg.lift_previous * prev + cfg.lift_next * nxt
    return s


def _to_ts(t) -> float:
    if t is None:
        return _time.time()
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()
    return float(t)


def _key(speaker: str, t: str) -> tuple:
    """Dedupe key: case and punctuation do not matter, unless that is all there is ("!!!", emoji)."""
    return speaker, re.sub(r"\W+", " ", t.lower()).strip() or t.strip()


class Memory:
    """Long-term memory for one agent (or one user, or one character).

        mem = Memory("alice.ncm")
        mem.add("I adopted a beagle called Max", speaker="user")
        mem.recall("what is my dog's name?")
        prompt_block = mem.context("what is my dog's name?")
        mem.save()
    """

    def __init__(self, path: str = None, embedder=None, config: Config = None, reranker=None):
        self.path = path
        self.embedder = as_embedder(embedder)
        # optional second stage, e.g. reranker=True or a cross-encoder name: sharper, slower
        self.reranker = as_reranker(reranker)
        self.config = config or Config()
        self._lock = threading.RLock()
        self._reset()
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.ids, self.texts, self.speakers, self.metas = [], [], [], []
        self._buf = {}                         # column -> (capacity, ...) array
        self._bm25 = BM25()
        self._index = {}                       # (speaker, normalised text) -> id
        self._rows = {}                        # id -> row
        self._next_id = 0
        self._session = 0
        self._new_session = False
        self._affect = None
        self._speaker_ids = {}                 # speaker name -> small int, for fast filtering

    def _col(self, name):
        return self._buf[name][:len(self)] if self._buf else np.zeros(0, _COLS[name])

    _times = property(lambda self: self._col("times"))
    _sessions = property(lambda self: self._col("sessions"))
    _uses = property(lambda self: self._col("uses"))
    _salience = property(lambda self: self._col("salience"))
    _doc = property(lambda self: self._col("doc"))
    _valence = property(lambda self: self._col("valence"))
    _arousal = property(lambda self: self._col("arousal"))
    _evidence = property(lambda self: self._col("evidence"))

    def rows_of(self, speaker: str = None) -> np.ndarray:
        """Row indices of one speaker's memories (all rows for None), oldest first."""
        if speaker is None:
            return np.arange(len(self))
        return np.flatnonzero(self._col("speaker") == self._speaker_ids.get(speaker, -1))

    @property
    def embeddings(self) -> np.ndarray:
        return self._buf["emb"][:len(self)] if self._buf else np.zeros((0, 0), np.float32)

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"Memory({len(self)} memories, {self._session + 1} sessions, embedder={self.embedder.name!r})"

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
        keys = [_key(sp, t) for t, sp in zip(texts, speakers)]
        with self._lock:                       # which texts are new? (cheap, under the lock)
            if self._index is None:
                self._index = {_key(sp, t): i for i, t, sp in zip(self.ids, self.texts, self.speakers)}
            new = list(dict.fromkeys(k for k in keys if k not in self._index))
        first = {k: keys.index(k) for k in new}
        vecs = va = None
        if new:                                # the model runs without blocking readers or writers
            vecs = self.embedder([f"{speakers[first[k]]}: {texts[first[k]]}" for k in new])
            if self._affect is None:
                self._affect = Affect(self.embedder)
            va = self._affect(vecs)
        row_of_new = {k: j for j, k in enumerate(new)}
        out = []
        with self._lock:
            sal = salience(va) if new else None
            for i, k in enumerate(keys):
                if k in self._index:           # stored before, or earlier in this batch, or by another thread
                    mid = self._index[k]
                    if mid in self._rows:
                        self._uses[self._rows[mid]] += 1
                else:
                    j = row_of_new[k]
                    item = (texts[i], speakers[i], _to_ts(times[i]), dict(metas[i] or {}))
                    mid = self._index[k] = self._append(item, vecs[j], va[j],
                                                        float(item[3].get("salience", sal[j])))
                out.append(mid)
            self._forget_overflow()
            return [o if o in self._rows else None for o in out]   # None: forgotten at once, store full

    def _append(self, item, vec, va, sal) -> int:
        text, speaker, ts, meta = item
        n = len(self)
        if n and (self._new_session or ts - self._times[-1] > self.config.session_gap):
            self._session += 1
        self._new_session = False
        if not self._buf:
            self._buf = {k: np.zeros(64, dt) for k, dt in _COLS.items()}
            self._buf["emb"] = np.zeros((64, len(vec)), np.float32)
        elif len(vec) != self._buf["emb"].shape[1]:
            raise ValueError(f"embedding size {len(vec)} != store size {self._buf['emb'].shape[1]}")
        if n == len(self._buf["emb"]):
            self._buf = {k: np.concatenate([v, np.zeros_like(v)]) for k, v in self._buf.items()}
        b = self._buf
        b["emb"][n], b["times"][n], b["sessions"][n] = vec, ts, self._session
        b["uses"][n], b["salience"][n] = 0, sal
        b["valence"][n], b["arousal"][n] = va
        b["evidence"][n], b["stmt_words"][n] = evidence(text), len(" ".join(statements(text)).split())
        b["speaker"][n] = self._speaker_ids.setdefault(speaker, len(self._speaker_ids))
        b["doc"][n] = self._bm25.add(f"{speaker}: {text}")
        mid = self._next_id
        self._next_id += 1
        self._rows[mid] = n
        self.ids.append(mid)
        self.texts.append(text)
        self.speakers.append(speaker)
        self.metas.append(meta)
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

    def mood(self, speaker: str = None, last: int = 20, half_life: float = 5.0) -> dict:
        """Emotional tone of the most recent statements (optionally of one speaker),
        recency-weighted: {"valence": v, "arousal": a, "label": "positive, calm"}.
        Questions and one-word turns are skipped: asking about feelings is not feeling."""
        with self._lock:
            rows = self.rows_of(speaker)
            rows = rows[self._col("stmt_words")[rows] >= 3][-last:]   # skip questions and "hi"
            if not len(rows):
                return {"valence": 0.0, "arousal": 0.0, "label": "neutral"}
            w = 0.5 ** (np.arange(len(rows))[::-1] / half_life)
            v = float(w @ self._valence[rows] / w.sum())
            a = float(w @ self._arousal[rows] / w.sum())
            return {"valence": v, "arousal": a, "label": label(v, a)}

    def _forget_overflow(self):
        cap = self.config.max_items
        if cap and len(self) > cap:
            drop = len(self) - (cap - cap // 20)   # batch it: one compaction per 5% of capacity
            order = np.argsort(self.activation(), kind="stable")[:drop]
            self.forget([self.ids[r] for r in order])

    def forget(self, ids) -> None:
        """Delete one memory id, or a list of them, permanently."""
        ids = [ids] if isinstance(ids, (int, np.integer)) else list(ids)
        with self._lock:
            rows = {self._row(m) for m in ids}
            for r in rows:
                self._bm25.remove(int(self._doc[r]))
            keep = np.array([r not in rows for r in range(len(self))])
            n_new = int(keep.sum())
            for k, v in self._buf.items():
                v[:n_new] = v[:len(self)][keep]
            for name in ("ids", "texts", "speakers", "metas"):
                setattr(self, name, [x for x, k in zip(getattr(self, name), keep) if k])
            gone = set(ids)
            if self._index is not None:
                self._index = {k: v for k, v in self._index.items() if v not in gone}
            self._rows = {m: i for i, m in enumerate(self.ids)}
            if self._bm25.n_docs > 2 * max(len(self), 32):   # mostly tombstones: rebuild
                self._bm25 = BM25()
                self._doc[:] = [self._bm25.add(f"{sp}: {t}") for t, sp in zip(self.texts, self.speakers)]

    # ------------------------------------------------------------------- read
    def scores(self, query: str, speaker: str = None) -> np.ndarray:
        """Relevance of every memory to `query` (higher is better), -inf for filtered rows."""
        if not len(self):
            return np.zeros(0, dtype=np.float32)
        q = self.embedder([query], query=True)[0]       # outside the lock: the slow part
        with self._lock:
            if not len(self):
                return np.zeros(0, dtype=np.float32)
            return self._score_vector(q, query, speaker)

    def _score_vector(self, q, query, speaker):
        """scores() for an already-embedded query; call with the lock held."""
        bm = self._bm25
        lex = bm.scores(query)
        if not (bm.n_docs == bm.n_alive == len(self)):          # tombstones: map docs to rows
            lex = lex[self._doc]
        s = score(self.embeddings @ q, lex, self._sessions, self.config)
        if speaker is not None:
            s = np.where(self._col("speaker") == self._speaker_ids.get(speaker, -1), s, -np.inf)
        return s

    def recall(self, query: str, k: int = 5, speaker: str = None, reinforce: bool = False) -> list:
        """The k memories most relevant to `query`, best first. With a reranker, the first
        stage's top `config.rerank_top` are re-scored by it and `score` is the reranker's."""
        q = self.embedder([query], query=True)[0] if len(self) else None
        with self._lock:
            if not len(self) or q is None:
                return []
            s = self._score_vector(q, query, speaker)
            n_ok = int(np.isfinite(s).sum())
            m = min(max(k, self.config.rerank_top) if self.reranker else k, n_ok)
            if min(k, n_ok) <= 0:
                return []
            top = np.argpartition(-s, m - 1)[:m] if m < len(s) else np.arange(len(s))
            top = top[np.argsort(-s[top], kind="stable")]
            hits = [self._recalled(int(r), float(s[r])) for r in top]
        if self.reranker:                      # slow: score a snapshot without holding the lock
            rs = np.asarray(self.reranker(query, [f"{h.speaker}: {h.text}" for h in hits]), np.float32)
            hits = [hits[i] for i in np.argsort(-rs, kind="stable")]
            for h, sc in zip(hits, np.sort(rs)[::-1]):
                h.score = float(sc)
        hits = hits[:k]
        if reinforce:
            with self._lock:
                rows = [self._rows[h.id] for h in hits if h.id in self._rows]
                self._uses[rows] += 1
        return hits

    def _recalled(self, row: int, sc: float = 0.0) -> Recalled:
        return Recalled(self.ids[row], self.texts[row], self.speakers[row], float(self._times[row]),
                        int(self._sessions[row]), sc, self.metas[row])

    def get(self, mid: int) -> Recalled:
        return self._recalled(self._row(mid))

    def all(self) -> list:
        return [self._recalled(r) for r in range(len(self))]

    def context(self, query: str, k: int = 4, persona_of: str = "user", persona_k: int = 6,
                window: int = 1, self_name: str = None) -> str:
        """A prompt-ready block: who the person is, then the memories relevant to `query`
        (each with `window` turns of surrounding context), oldest first and dated so a
        model can tell which of two conflicting statements is the latest. Lines spoken by
        `self_name` (the agent itself) are shown as "you"."""
        from ncm.persona import persona
        hits = self.recall(query, k=k, reinforce=True)
        with self._lock:
            rows = set()
            for h in hits:
                if h.id not in self._rows:     # forgotten while the reranker ran
                    continue
                r = self._rows[h.id]
                for o in range(-window, window + 1):
                    if 0 <= r + o < len(self) and self._sessions[r + o] == self._sessions[r]:
                        rows.add(r + o)
            parts = []
            facts = persona(self, speaker=persona_of, k=persona_k) if persona_of else []
            if facts:
                tone = self.mood(persona_of)["label"]
                parts.append(f"What you know about {persona_of}:\n" + "\n".join(f"- {f}" for f in facts)
                             + (f"\nTheir mood lately: {tone}" if tone != "neutral" else ""))
            if rows:
                lines, last = [], None
                for r in sorted(rows, key=lambda r: (self._times[r], r)):
                    if last is not None and self._sessions[r] != self._sessions[last]:
                        lines.append("")
                    dt = datetime.fromtimestamp(self._times[r], tz=timezone.utc)
                    day = f"{dt.day} {dt:%B %Y}, {dt:%H:%M}"   # "7 May 2023": how people (and LLMs) say dates
                    who = "you" if self.speakers[r] == self_name else self.speakers[r]
                    lines.append(f"[{day}] {who}: {self.texts[r]}")
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
            new_id = np.full(self._bm25.n_docs, -1, np.int64)
            new_id[self._doc] = np.arange(len(self))
            terms, indptr, docs, tfs = self._bm25.export(new_id)
            meta = {"format": "ncm", "version": FORMAT_VERSION, "embedder": self.embedder.name,
                    "config": asdict(self.config), "next_id": self._next_id,
                    "session": self._session, "ids": self.ids, "texts": self.texts,
                    "speakers": self.speakers, "metas": self.metas, "bm25_terms": terms}
            buf = io.BytesIO()
            # vectors do not compress, so only the JSON (texts) is zlib-ed: 10x faster saves
            np.savez(buf, meta=np.frombuffer(zlib.compress(json.dumps(meta).encode("utf-8"), 6), np.uint8),
                     emb=self.embeddings.astype(np.float16), times=self._times,
                     sessions=self._sessions, uses=self._uses, salience=self._salience,
                     valence=self._valence, arousal=self._arousal, evidence=self._evidence,
                     stmt_words=self._col("stmt_words"), bm25_indptr=indptr,
                     bm25_docs=docs, bm25_tfs=tfs, bm25_len=self._bm25._len[self._doc])
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                f.write(buf.getvalue())
            os.replace(tmp, path)
            self.path = path
            return path

    @classmethod
    def load(cls, path: str, embedder=None, config: Config = None, reranker=None) -> "Memory":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return cls(path, embedder=embedder, config=config, reranker=reranker)

    def _load(self, path: str):
        with open(path, "rb") as f:
            head = f.read(4)
        if head == b"NCM\x02":
            raise ValueError(f"{path} is an NCM v1 file: import it with ncm.legacy.import_v1({path!r})")
        if head != b"PK\x03\x04":
            raise ValueError(f"{path} is not an NCM file")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(zlib.decompress(z["meta"].tobytes()).decode("utf-8"))
            if meta.get("format") != "ncm" or meta.get("version") != FORMAT_VERSION:
                raise ValueError(f"{path} is not an NCM v{FORMAT_VERSION} file "
                                 "(use ncm.legacy.import_v1 for old .ncm files)")
            if meta["embedder"] != self.embedder.name:
                raise ValueError(f"{path} was embedded with {meta['embedder']!r}, "
                                 f"not {self.embedder.name!r}; pass embedder={meta['embedder']!r}")
            cols = {k: np.array(z[k], dtype=dt) for k, dt in _COLS.items() if k not in _DERIVED}
            emb = np.array(z["emb"], dtype=np.float32)
            bm = [z[k] for k in ("bm25_indptr", "bm25_docs", "bm25_tfs", "bm25_len")]
        if self.config == Config():
            self.config = Config(**meta["config"])
        self.ids, self.texts = meta["ids"], meta["texts"]
        self.speakers, self.metas = meta["speakers"], meta["metas"]
        self._next_id, self._session = meta["next_id"], meta["session"]
        self._rows = {m: i for i, m in enumerate(self.ids)}
        self._bm25 = BM25.from_arrays(meta["bm25_terms"], *bm)
        self._index = None                    # dedupe index, rebuilt on the first write
        if self.ids:
            self._speaker_ids = {sp: i for i, sp in enumerate(dict.fromkeys(self.speakers))}
            spk = np.array([self._speaker_ids[sp] for sp in self.speakers], dtype=np.int32)
            self._buf = {**cols, "emb": emb, "doc": np.arange(len(self.ids), dtype=np.int64), "speaker": spk}
