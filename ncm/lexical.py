"""BM25 keyword index. Pure Python + numpy, incremental adds.

Dense embeddings blur exact tokens (names, numbers, rare nouns); BM25 keeps them.
"""

import math
import re
from collections import Counter, defaultdict

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


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)   # term -> [(doc, tf), ...]
        self.lengths = []
        self._arrays = {}                    # term -> (docs, tfs) numpy cache

    def __len__(self):
        return len(self.lengths)

    def add(self, text: str) -> None:
        doc = len(self.lengths)
        toks = tokenize(text)
        self.lengths.append(len(toks))
        for term, tf in Counter(toks).items():
            self.postings[term].append((doc, tf))
            self._arrays.pop(term, None)

    def scores(self, query: str) -> np.ndarray:
        n = len(self.lengths)
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        lengths = np.asarray(self.lengths, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * lengths / max(lengths.mean(), 1e-9))
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            arr = self._arrays.get(term)
            if arr is None:
                arr = self._arrays[term] = (np.array([d for d, _ in plist]),
                                            np.array([t for _, t in plist], dtype=np.float32))
            docs, tfs = arr
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            out[docs] += idf * tfs * (self.k1 + 1) / (tfs + norm[docs])
        return out
