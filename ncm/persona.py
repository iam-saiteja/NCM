"""NCM - Persona payload: the few memories that describe the person, not the topic.

Retrieval answers "what is relevant to this query". A caller that wants a model to
*behave like* a particular person needs a second and different thing: a small set
of memories that describe who that person is, independent of what was just asked.
That is a selection problem over the whole store, not a ranking problem against a
query, and this module is the selection.

Nothing here calls a language model and nothing here calls the encoder. Evidence
comes from regular expressions over the stored text and diversity comes from the
semantic vectors the store already holds, so building a payload costs one matrix
multiply over the selected set and one regex pass per memory, with no inference.
That is deliberate: the cost argument for a memory layer collapses the moment the
memory layer needs its own model.

WHY THIS IS NOT A RETRIEVAL WEIGHT. The same evidence vector was measured as a
fifth channel inside the composite distance and it failed decisively. On 458
held-out validation cells, eleven arms in the weight grid, it lowered
persona-statement discrimination at every weight under two different query
operators. All sixteen paired differences cleared a Bonferroni threshold of
0.00625 in the wrong direction with every conversation-clustered bootstrap
interval below zero, and AUC fell monotonically as the state weight rose. Under
the mixed-query operator the best evidence arm scored 0.6889 AUC against 0.7183
for semantic distance alone (difference -0.0275, interval [-0.0412, -0.0134],
p 3.63e-06) and the worst scored 0.5000, which is chance (-0.2164, p 1.18e-119).
Under the speaker-only operator the best arm still lost, 0.7024 against 0.7248
(-0.0174, p 0.00118).

The reason is mechanical rather than statistical. The hard case in that task is
same-topic-wrong-person, "I own a Jeep" against "I drive a Honda", where the
discriminating information is the entity and already lives in the semantic
channel. A five-dimensional evidence type cannot separate two car brands. So the
evidence vector does not go in the distance. It goes here, where the question is
which turns to hand over rather than which turn is closest.

WHAT WAS MEASURED HERE. Coverage of a speaker's human-written Multi-Session Chat
persona statements by the k turns a selector picks, scored as the mean over
statements of the largest cosine to any selected turn, judged against ground
truth that no selector sees. The validation split selected the method and the
test split confirmed two directional hypotheses declared before it was read.
Test, 1002 cells over 501 conversations, against a diversity-only control holding
lambda and the redundancy penalty identical so the persona prior is the only
thing that differs:

  Diversity matters more than salience. Greedy MMR beat uniform random at every
  budget, while both salience baselines fell below random at k=10: longest 0.4664
  and nearest-centroid 0.4791 against random 0.4992. Salience-ranked selection
  picks near-duplicates that re-cover one statement and miss the rest. This
  replicated on both splits.

  The persona prior pays only at a real budget, and the crossover replicated.
  At k=10 the prior added 0.0210 coverage over the identical diversity control,
  0.5373 against 0.5163, interval [+0.0166, +0.0255], p 3.1e-19, and lifted
  hit-at-0.5 by 3.0 points, 0.5667 against 0.5364. At k=3 the same prior *cost*
  0.0290, p 1.17e-26. Both directions were declared in advance from the
  validation split, where they measured +0.0291 (p 7.57e-12) and -0.0160
  (p 3.82e-05), and both confirmed. At k=5 the prior was null on both splits
  (test -0.0043, p 0.199; valid +0.0033, p 0.491).

DEFAULT_PAYLOAD_SIZE is 10 because of that crossover and not by preference, and
MIN_PRIOR_BUDGET is 10 for the same reason: k=10 is the only budget at which the
prior was confirmed to help, so it is the only budget at which it is applied.
Below it, select_persona_exemplars falls back to diversity over semantic
salience, which is the control that won at k=3 and tied at k=5.

HONEST LIMITS. Coverage is scored by embedding cosine, which rewards paraphrase
and topic overlap and cannot verify that a turn entails a statement, so a high
number is necessary and not sufficient. The extractor is English, present tense
and first person, which is what a persona statement looks like in this corpus and
is not general. Negation is not modelled, so "I do not own a car" reads as
disclosure, correct as evidence type and wrong as polarity. And the ceiling is
still far off: a greedy oracle with access to the answers reaches 0.6668 at k=10
against 0.5373 here, so this selector closes 14.0 percent of the reachable gap
over the control, 18.2 percent on the validation split.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Evidence extraction
#
# Five dimensions, because ncm/persistence.py serialises auto_state_snapshot at a
# hard-coded width of five. A wider vector written into that field round-trips as
# silent corruption, so five is a constraint of the file format rather than a
# design preference, and the same width is reused here so an evidence vector can
# be stored in that field when a caller wants it persisted.
# ---------------------------------------------------------------------------

PERSONA_DIMS = ("disclosure", "preference", "commitment", "habit", "solicit")

# A first-person claim about a durable attribute. Restricted to frames that carry
# an attribute rather than an event, so "I own" is here and "I went" is not.
_DISCLOSURE = re.compile(
    r"\b(?:i\s*(?:'m|’m)|i\s+am|i\s+have|i\s*(?:'ve|’ve)|i\s+own|"
    r"i\s+work|i\s+live|i\s+study|i\s+drive|i\s+teach|i\s+speak|"
    r"my\s+(?:job|work|wife|husband|son|daughter|kid|kids|children|dog|cat|pet|"
    r"car|truck|house|home|family|mom|mother|dad|father|brother|sister|name|"
    r"degree|major|hometown|birthday))\b",
    re.I,
)
# A weaker fallback, because a persona statement need not use one of those frames.
# "I do not eat meat" is disclosure and matches none of them.
_FIRST_PERSON_LEAD = re.compile(r"^\s*i\s+\w+", re.I)

_PREFERENCE = re.compile(
    r"\b(?:i\s+(?:like|love|hate|prefer|enjoy|adore|dislike|fancy)\b|"
    r"i\s*(?:'d|’d|\s+would)\s+rather\b|my\s+favou?rite\b|"
    r"i\s+can\s*(?:'t|’t|not)\s+stand\b|not\s+(?:a\s+fan|really\s+into)\b|"
    r"i\s*(?:'m|’m|\s+am)\s+(?:really\s+)?into\b)",
    re.I,
)
_COMMITMENT = re.compile(
    r"\b(?:i\s*(?:'ll|’ll)|i\s+will|i\s*(?:'m|’m|\s+am)\s+(?:going\s+to|gonna)|"
    r"i\s+plan\s+to|i\s+decided|i\s+chose|i\s+want\s+to|i\s+need\s+to|"
    r"i\s+hope\s+to|i\s+intend\s+to)\b",
    re.I,
)
_HABIT = re.compile(
    r"\b(?:i\s+always|i\s+never|i\s+usually|i\s+often|i\s+rarely|i\s+tend\s+to|"
    r"every\s+(?:day|week|morning|evening|night|year|weekend)|"
    r"most\s+(?:days|nights|of\s+the\s+time)|all\s+the\s+time)\b",
    re.I,
)
# Interrogative opener, consulted only when no question mark is present, since a
# question mark is by far the more reliable signal.
_SOLICIT_LEAD = re.compile(
    r"^\s*(?:what|where|when|why|who|whom|which|how|do|does|did|are|is|was|were|"
    r"have|has|had|can|could|would|will|should|any|tell\s+me)\b",
    re.I,
)

_LEAD_WEIGHT = 0.6           # the fallback opener, graded because it is looser
_SOLICIT_LEAD_WEIGHT = 0.5   # an interrogative opener with no question mark
# Floor on every dimension so no turn yields the zero vector. A zero vector
# survives L2 normalization unchanged, which would place an evidence-free turn at
# a fixed distance from every reference by accident rather than by design. With
# the floor it points along the diagonal, between a matching turn and a
# contradicting one, which is where a turn that says nothing about the speaker
# belongs.
#
# Held as float32 rather than as a Python float on purpose. The vectors are
# float32, and float32(0.05) widens to 0.05000000074505806 when compared against
# the Python literal 0.05, so `value > 0.05` is True for a floor value and every
# dimension reads as fired. Comparing float32 to float32 is exact. Use _fired
# rather than an inline comparison so this cannot be got wrong twice.
_FLOOR = np.float32(0.05)


def _fired(value) -> bool:
    """Whether one dimension carries real evidence rather than the floor."""
    return bool(np.float32(value) > _FLOOR)


# The pure-disclosure reference direction. Relevance for exemplar selection is
# closeness to this axis, not closeness to the query, because the payload is
# supposed to describe the person no matter what was asked.
DISCLOSURE_AXIS = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

DEFAULT_PAYLOAD_SIZE = 10
DEFAULT_MMR_LAMBDA = 0.5
# Below this budget the persona prior is not applied. k=10 is the only budget at
# which it was confirmed to add coverage over the identical diversity control
# (test d +0.0210, p 3.1e-19), k=5 was null on both splits and k=3 measurably
# worse on both. Budgets between 6 and 9 were not measured, and the threshold
# sits at 10 rather than lower so that the untested range extrapolates toward the
# control rather than toward an unconfirmed mechanism. See the crossover in the
# module docstring.
MIN_PRIOR_BUDGET = 10


def persona_evidence_vector(text: str) -> np.ndarray:
    """The five-dimensional evidence vector for one turn.

    float32 of shape (5,), every component in [_FLOOR, 1]. Multi-hot on purpose:
    "I have two dogs, do you have any?" both tells and asks, and the honest
    reading of such a turn is a direction between the two poles rather than a
    choice between them.
    """
    s = text if isinstance(text, str) else ""
    v = np.full(5, _FLOOR, dtype=np.float32)
    if not s.strip():
        return v
    if _DISCLOSURE.search(s):
        v[0] = 1.0
    elif _FIRST_PERSON_LEAD.search(s):
        v[0] = _LEAD_WEIGHT
    if _PREFERENCE.search(s):
        v[1] = 1.0
    if _COMMITMENT.search(s):
        v[2] = 1.0
    if _HABIT.search(s):
        v[3] = 1.0
    if "?" in s:
        v[4] = 1.0
    elif _SOLICIT_LEAD.search(s):
        v[4] = _SOLICIT_LEAD_WEIGHT
    return v


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize, matching MemoryStore._normalize_state and retrieval._unit."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= 1e-8 else v.copy()


def _evidence_matrix(entries: Sequence) -> np.ndarray:
    """Evidence vectors for a list of entries, (n, 5) float32.

    Text is preferred over the stored snapshot, so a payload can be built from a
    store that was written in any way at all, including one loaded from an .ncm
    file that predates this module. The snapshot is the fallback for entries whose
    text was not retained, and for those the caller gets whatever was written
    there, which on a default store is the affect EMA and carries no persona
    evidence. That is a limitation of the stored data, not of this function.
    """
    rows = []
    for m in entries:
        text = getattr(m, "text", "")
        if isinstance(text, str) and text.strip():
            rows.append(persona_evidence_vector(text))
        else:
            snap = getattr(m, "auto_state_snapshot", None)
            if snap is None:
                rows.append(np.full(5, _FLOOR, dtype=np.float32))
            else:
                rows.append(np.asarray(snap, dtype=np.float32)[:5])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _disclosure_relevance(evidence: np.ndarray) -> np.ndarray:
    """Larger is more persona-bearing, on the same scale as a cosine.

    The distance is the one the state channel computes, so an exemplar chosen here
    and a memory ranked there agree about what "close to disclosure" means.
    """
    if evidence.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    m = evidence / np.maximum(np.linalg.norm(evidence, axis=1, keepdims=True), 1e-8)
    q = _unit(DISCLOSURE_AXIS)
    d = np.clip(np.linalg.norm(m - q[np.newaxis, :], axis=1) / np.sqrt(2.0), 0.0, 1.0)
    return (1.0 - d).astype(np.float32)


def _greedy_mmr(relevance, sem, k, lam, rng):
    """Pick k indices maximising lam*relevance - (1-lam)*max similarity to the
    chosen set. Relevance is larger-is-better and sem rows are unit vectors."""
    n = sem.shape[0]
    k = min(int(k), n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    sim = sem @ sem.T
    # The extractor is discrete, so relevance has dense ties. A tiny random key
    # stops insertion order from resolving them, which would otherwise make the
    # payload depend on the order memories happened to be added.
    rel = relevance.astype(np.float32) + rng.random(n).astype(np.float32) * 1e-6
    chosen = [int(np.argmax(rel))]
    penalty = sim[chosen[0]].copy()
    while len(chosen) < k:
        score = lam * rel - (1.0 - lam) * penalty
        score[chosen] = -np.inf
        pick = int(np.argmax(score))
        chosen.append(pick)
        penalty = np.maximum(penalty, sim[pick])
    return np.asarray(chosen, dtype=np.int64)


def select_persona_exemplars(
    store,
    k: int = DEFAULT_PAYLOAD_SIZE,
    tags: Optional[Sequence[str]] = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = 0,
) -> List:
    """The k memories that best describe whoever wrote them.

    Args:
        store: a MemoryStore. Only its cached semantic matrix and its entries are
            read; nothing is mutated and no memory is reinforced.
        k: budget. At k below MIN_PRIOR_BUDGET the persona prior is dropped and
            selection is diversity over semantic salience, because that is what
            measured better at small budgets.
        tags: keep only memories carrying at least one of these tags. This is how
            one speaker is separated from another in a shared store.
        mmr_lambda: relevance against diversity, 0 is pure diversity and 1 is pure
            relevance. The measured default is 0.5 and was not tuned.
        seed: fixes the tiebreak among equally-scored memories, so a payload is
            reproducible for a given store.

    Returns the selected entries in selection order, most persona-bearing first.
    """
    store._rebuild_cache()
    entries = [store._memories[mid] for mid in store._id_order]
    sem = store._sem_cache
    if tags:
        want = set(tags)
        keep = [i for i, m in enumerate(entries)
                if want.intersection(getattr(m, "tags", ()) or ())]
        if not keep:
            return []
        entries = [entries[i] for i in keep]
        sem = sem[np.asarray(keep, dtype=np.int64)]
    if not entries:
        return []

    sem = np.asarray(sem, dtype=np.float32)
    sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(seed)

    if int(k) < MIN_PRIOR_BUDGET:
        centroid = _unit(sem.mean(axis=0))
        relevance = sem @ centroid
    else:
        relevance = _disclosure_relevance(_evidence_matrix(entries))

    idx = _greedy_mmr(relevance, sem, k, float(mmr_lambda), rng)
    return [entries[int(i)] for i in idx]


@dataclass
class PersonaPayload:
    """Conditioning material for a caller's own model.

    The buckets are the evidence dimensions that fired, so they line up with what
    a caller actually wants to condition on: preferences are stance, commitments
    are decision precedents, and the selected set as a whole is style exemplars.
    A memory can appear in more than one bucket, because a turn that states a
    preference and a plan is evidence of both.
    """
    exemplars: List = field(default_factory=list)      # selection order
    facts: List = field(default_factory=list)          # disclosure
    preferences: List = field(default_factory=list)    # stance
    commitments: List = field(default_factory=list)    # decision precedents
    habits: List = field(default_factory=list)
    evidence: Optional[np.ndarray] = None              # (k, 5), for inspection

    def __len__(self) -> int:
        return len(self.exemplars)

    def to_prompt_block(self, header: str = "What I know about this person") -> str:
        """Render as plain text for the caller to place in its own prompt.

        Deliberately a string and not a completed answer. NCM is the memory layer:
        it supplies the conditioning and the caller's model does the generating,
        which is what keeps the hot path free of inference.

        Each memory is rendered exactly once, under the narrowest section it
        belongs to. The buckets overlap by design, and the disclosure bucket
        catches almost every first-person sentence, so rendering every membership
        repeated a third of the lines in testing: ten exemplars produced
        seventeen lines, with one sentence appearing under facts, preferences and
        habits. A repeated line tells the model nothing it has not already read
        and the caller pays tokens for it either way. The dataclass lists keep
        full multi-membership for programmatic use; only the rendered block
        deduplicates. Identical text from two different memories collapses too,
        for the same reason.
        """
        # Narrowest first, so a sentence that states a preference is filed as a
        # preference and only a sentence with nothing more specific to say falls
        # through to the general facts section.
        assignment_order = (
            ("Preferences and stance", self.preferences),
            ("Decisions and intentions", self.commitments),
            ("Habits and routines", self.habits),
            ("Stated facts", self.facts),
        )
        # Reading order, which puts the plain facts first because that is how a
        # description of a person usually opens.
        display_order = ("Stated facts", "Preferences and stance",
                         "Decisions and intentions", "Habits and routines")

        assigned = {title: [] for title, _ in assignment_order}
        claimed = set()
        for title, group in assignment_order:
            for m in group:
                text = (getattr(m, "text", "") or "").strip()
                if not text or text in claimed:
                    continue
                claimed.add(text)
                assigned[title].append(text)

        lines = [header + ":"]
        for title in display_order:
            if not assigned[title]:
                continue
            lines.append("")
            lines.append(title + ":")
            lines.extend("- " + t for t in assigned[title])

        rest = []
        for m in self.exemplars:
            text = (getattr(m, "text", "") or "").strip()
            if text and text not in claimed:
                claimed.add(text)
                rest.append(text)
        if rest:
            lines.append("")
            lines.append("Other things they said:")
            lines.extend("- " + t for t in rest)
        return "\n".join(lines)


def build_persona_payload(
    store,
    k: int = DEFAULT_PAYLOAD_SIZE,
    tags: Optional[Sequence[str]] = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    seed: int = 0,
) -> PersonaPayload:
    """Select k exemplars and sort them into the conditioning buckets.

    Costs one similarity matrix over the candidate set and one regex pass per
    memory. No model call, no encoder call, and the store is not modified.
    """
    chosen = select_persona_exemplars(
        store, k=k, tags=tags, mmr_lambda=mmr_lambda, seed=seed)
    if not chosen:
        return PersonaPayload(evidence=np.zeros((0, 5), dtype=np.float32))
    ev = _evidence_matrix(chosen)
    payload = PersonaPayload(exemplars=list(chosen), evidence=ev)
    buckets = (payload.facts, payload.preferences,
               payload.commitments, payload.habits)
    for row, mem in zip(ev, chosen):
        # A turn is filed under every dimension it actually asserts. Anything
        # above the floor is a real signal, and the solicit dimension is skipped
        # because asking a question is evidence about the other speaker rather
        # than about this one.
        for dim in range(4):
            if _fired(row[dim]):
                buckets[dim].append(mem)
    return payload
