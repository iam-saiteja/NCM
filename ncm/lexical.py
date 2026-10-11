"""BM25 keyword index. Pure Python + numpy; incremental adds, cheap deletes.

Dense embeddings blur exact tokens (names, numbers, rare nouns); BM25 keeps them.
"""

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be
been being am i me my mine we us our you your yours he him his she her hers it its
they them their this that these those what which who whom whose when where why how
do does did done doing have has had having not no yes can could would should will
shall may might must just very really also too there here about into over than out
up down off again once all any both each few more most other some such only own same
s t don doesn didn isn wasn aren weren won wouldn couldn shouldn ll ve re d m o y
user assistant oh ok okay yeah hey hi hello lol um uh like get got go going
""".split())


def tokenize(text: str) -> list:
    out = []
    for w in _TOKEN.findall(text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _grow(a: np.ndarray, n: int) -> np.ndarray:
    return a if n < len(a) else np.concatenate([a, np.zeros_like(a)])


class BM25:
    """Documents are numbered 0, 1, 2... in insertion order. remove() tombstones a doc."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings = {}                    # term -> [docs int32 buf, tfs f32 buf, count]
        self._len = np.zeros(64, dtype=np.float32)
        self._alive = np.zeros(64, dtype=bool)
        self.n_docs = 0                        # including removed ones
        self.n_alive = 0
        self._total = 0.0

    def __len__(self):
        return self.n_alive

    def add(self, text: str) -> int:
        doc = self.n_docs
        toks = tokenize(text)
        self._len, self._alive = _grow(self._len, doc), _grow(self._alive, doc)
        self._len[doc], self._alive[doc] = len(toks), True
        self.n_docs += 1
        self.n_alive += 1
        self._total += len(toks)
        for term, tf in Counter(toks).items():
            p = self.postings.get(term)
            if p is None:
                p = self.postings[term] = [np.zeros(4, np.int32), np.zeros(4, np.float32), 0]
            if p[2] == len(p[0]):
                p[0], p[1] = _grow(p[0], p[2]), _grow(p[1], p[2])
            p[0][p[2]], p[1][p[2]] = doc, tf
            p[2] += 1
        return doc

    def remove(self, doc: int) -> None:
        if self._alive[doc]:
            self._alive[doc] = False
            self.n_alive -= 1
            self._total -= self._len[doc]

    def scores(self, query: str) -> np.ndarray:
        """Score per doc number (removed docs score 0). Length n_docs."""
        n = self.n_docs
        out = np.zeros(n, dtype=np.float32)
        if not self.n_alive:
            return out
        norm = self.k1 * (1 - self.b + self.b * self._len[:n] / max(self._total / self.n_alive, 1e-9))
        for term in set(tokenize(query)):
            p = self.postings.get(term)
            if p is None:
                continue
            docs, tfs = p[0][:p[2]], p[1][:p[2]]
            df = len(docs) if self.n_alive == self.n_docs else int(self._alive[docs].sum())
            if not df:
                continue
            idf = math.log(1 + (self.n_alive - df + 0.5) / (df + 0.5))
            out[docs] += idf * tfs * (self.k1 + 1) / (tfs + norm[docs])
        out[:n][~self._alive[:n]] = 0
        return out

    def export(self, new_id: np.ndarray):
        """The index as CSR arrays over live docs, renumbered by new_id[doc] (-1 drops a doc)."""
        terms, indptr, docs, tfs = [], [0], [], []
        for term, (d, tf, c) in self.postings.items():
            nid = new_id[d[:c]]
            keep = nid >= 0
            if keep.any():
                terms.append(term)
                docs.append(nid[keep])
                tfs.append(tf[:c][keep])
                indptr.append(indptr[-1] + int(keep.sum()))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
        return terms, np.array(indptr, np.int64), cat(docs, np.int32), cat(tfs, np.float32)

    @classmethod
    def from_arrays(cls, terms, indptr, docs, tfs, lengths, k1=1.2, b=0.75) -> "BM25":
        bm = cls(k1, b)
        n = len(lengths)
        bm._len = np.zeros(max(64, 2 * n), np.float32)
        bm._alive = np.zeros(len(bm._len), bool)
        bm._len[:n], bm._alive[:n] = lengths, True
        bm.n_docs = bm.n_alive = n
        bm._total = float(np.sum(lengths))
        for i, t in enumerate(terms):
            lo, hi = indptr[i], indptr[i + 1]
            bm.postings[t] = [docs[lo:hi].copy(), tfs[lo:hi].copy(), int(hi - lo)]
        return bm
