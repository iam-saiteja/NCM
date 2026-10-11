"""Benchmark loaders. Every dataset becomes the same two things:

Haystack: one conversation history, turn by turn, in the order it happened.
Query:    a question over one haystack, with the turn indices that hold the answer.

Datasets are read from $NCM_DATA (default ../ncm-data next to the repo):
  locomo/locomo10.json             https://github.com/snap-research/locomo
  longmemeval/longmemeval_s.json   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

DATA_DIR = os.environ.get(
    "NCM_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "ncm-data"))


@dataclass
class Haystack:
    id: str
    texts: list = field(default_factory=list)
    speakers: list = field(default_factory=list)
    sessions: list = field(default_factory=list)   # session index per turn
    times: list = field(default_factory=list)      # days since 2000-01-01 per turn


@dataclass
class Query:
    haystack: str
    question: str
    answer: str
    gold: list          # turn indices holding the evidence
    category: str
    time: float         # when the question is asked, same unit as Haystack.times


_EPOCH = datetime(2000, 1, 1)


def _days(dt: datetime) -> float:
    return (dt - _EPOCH).total_seconds() / 86400.0


def load_locomo(path=None):
    """LoCoMo: 10 long conversations (~600 turns), QA with turn-level evidence.
    Category 5 (adversarial, unanswerable) is dropped: it has no answer to find."""
    path = path or os.path.join(DATA_DIR, "locomo", "locomo10.json")
    cats = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
    hays, queries = [], []
    for conv in json.load(open(path, encoding="utf-8")):
        c = conv["conversation"]
        h = Haystack(id=conv["sample_id"])
        index = {}
        n_sessions = sum(1 for k in c if re.fullmatch(r"session_\d+", k))
        for s in range(1, n_sessions + 1):
            when = datetime.strptime(c[f"session_{s}_date_time"], "%I:%M %p on %d %B, %Y")
            for i, turn in enumerate(c[f"session_{s}"]):
                text = turn["text"]
                if turn.get("blip_caption"):
                    text += f" [shares a photo of {turn['blip_caption']}]"
                index[turn["dia_id"]] = len(h.texts)
                h.texts.append(f"{turn['speaker']}: {text}")
                h.speakers.append(turn["speaker"])
                h.sessions.append(s - 1)
                h.times.append(_days(when) + i / 1440.0)  # one minute per turn
        hays.append(h)
        for q in conv["qa"]:
            if q["category"] not in cats:
                continue
            gold = sorted({index[e] for e in q.get("evidence", []) if e in index})
            if not gold:
                continue
            queries.append(Query(h.id, q["question"], str(q["answer"]), gold,
                                 cats[q["category"]], h.times[-1] + 1.0))
    return hays, queries


def load_longmemeval(path=None, limit=None):
    """LongMemEval-S: 500 questions, each with its own ~50-session haystack.
    Abstention questions (no evidence turn) are dropped for retrieval scoring."""
    path = path or os.path.join(DATA_DIR, "longmemeval", "longmemeval_s.json")
    fmt = "%Y/%m/%d (%a) %H:%M"
    hays, queries = [], []
    for item in json.load(open(path, encoding="utf-8"))[:limit]:
        h = Haystack(id=item["question_id"])
        gold = []
        for s, (sess, date) in enumerate(zip(item["haystack_sessions"], item["haystack_dates"])):
            t0 = _days(datetime.strptime(date, fmt))
            for i, turn in enumerate(sess):
                if turn.get("has_answer"):
                    gold.append(len(h.texts))
                h.texts.append(f"{turn['role']}: {turn['content']}")
                h.speakers.append(turn["role"])
                h.sessions.append(s)
                h.times.append(t0 + i / 1440.0)
        hays.append(h)
        if gold:
            queries.append(Query(h.id, item["question"], str(item["answer"]), gold,
                                 item["question_type"],
                                 _days(datetime.strptime(item["question_date"], fmt))))
    return hays, queries


LOADERS = {"locomo": load_locomo, "longmemeval": load_longmemeval}
