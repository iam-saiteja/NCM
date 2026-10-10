"""Who is this person? Pick the few memories that describe the speaker, not the topic.

A selection over the whole store: first-person statements of durable facts,
preferences, plans and habits score high; questions and small talk score low;
MMR keeps the picks from repeating each other. No model call.
"""

import re

import numpy as np

_FACT = re.compile(
    r"\b(?:i\s*(?:'m|’m)|i\s+am|i\s+have|i\s*(?:'ve|’ve)|i\s+own|i\s+work|i\s+live|"
    r"i\s+study|i\s+drive|i\s+teach|i\s+speak|i\s+was\s+born|i\s+grew\s+up|"
    r"my\s+(?:job|work|wife|husband|partner|son|daughter|kids?|children|dog|cat|pet|car|"
    r"house|home|family|mom|mother|dad|father|brother|sister|name|degree|major|hometown|birthday))\b",
    re.I)
_LEAD = re.compile(r"^\s*(?:\w+:\s*)?i\s+\w+", re.I)
_PREF = re.compile(
    r"\b(?:i\s+(?:really\s+)?(?:like|love|hate|prefer|enjoy|adore|dislike)\b|my\s+favou?rite\b|"
    r"i\s*(?:'d|’d|\s+would)\s+rather\b|i\s+can\s*(?:'t|’t|not)\s+stand\b|"
    r"i\s*(?:'m|’m|\s+am)\s+(?:really\s+)?into\b|not\s+a\s+fan)", re.I)
_PLAN = re.compile(
    r"\b(?:i\s*(?:'ll|’ll)|i\s+will|i\s*(?:'m|’m|\s+am)\s+(?:going\s+to|gonna|planning)|"
    r"i\s+plan\s+to|i\s+decided|i\s+want\s+to|i\s+hope\s+to)\b", re.I)
_HABIT = re.compile(
    r"\b(?:i\s+(?:always|never|usually|often|rarely)|i\s+tend\s+to|"
    r"every\s+(?:day|week|morning|evening|night|weekend))\b", re.I)


_SENT = re.compile(r"(?<=[.!?])\s+")
_ASK = re.compile(r"^\s*(?:what|where|when|why|who|whom|which|how|do|does|did|is|are|was|were|"
                  r"can|could|would|will|should|have|has|tell\s+me)", re.I)


def is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(_ASK.match(sentence))


def statements(text: str) -> list:
    """The non-question sentences of a turn."""
    return [s for s in _SENT.split(text.strip()) if s and not is_question(s)]


def _sentence_evidence(s: str) -> float:
    fact = 1.0 if _FACT.search(s) else 0.6 if _LEAD.search(s) else 0.0
    other = sum(bool(p.search(s)) for p in (_PREF, _PLAN, _HABIT))
    return min(1.0, 0.6 * fact + 0.3 * other + 0.1) if fact or other else 0.0


def evidence(text: str) -> float:
    """How much a turn says about its speaker, 0..1. Questions say nothing."""
    return max((_sentence_evidence(s) for s in statements(text)), default=0.0)


def persona(mem, speaker: str = "user", k: int = 6, diversity: float = 0.5) -> list:
    """Up to k statements that best describe `speaker`, most telling first."""
    rows = [r for r, sp in enumerate(mem.speakers) if sp == speaker]
    rel = np.array([evidence(mem.texts[r]) for r in rows], dtype=np.float32)
    keep = rel > 0
    rows, rel = [r for r, m in zip(rows, keep) if m], rel[keep]
    if not rows:
        return []
    # newer statements win ties: facts change, and the latest one is the true one
    rel = rel + 1e-3 * np.linspace(0, 1, len(rows), dtype=np.float32)
    emb = mem.embeddings[rows]
    sim = emb @ emb.T
    chosen = [int(np.argmax(rel))]
    penalty = sim[chosen[0]].copy()
    while len(chosen) < min(k, len(rows)):
        s = (1 - diversity) * rel - diversity * penalty
        s[chosen] = -np.inf
        chosen.append(int(np.argmax(s)))
        penalty = np.maximum(penalty, sim[chosen[-1]])
    return [mem.texts[rows[i]] for i in chosen]
