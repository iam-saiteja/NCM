"""
EXP23: CADP Detection Precision and False Positive Characterisation
===================================================================

ALL DATA IN THIS EXPERIMENT IS HAND-AUTHORED SYNTHETIC TEXT written by the
experiment author. There is no external corpus, no human annotation study and
no inter-annotator agreement. Every count, rate, precision, recall and F1 value
produced here is computed over that hand-authored synthetic set and must be
cited as such in any downstream text.

Purpose
- EXP18 measures only the TRUE POSITIVE side of the Contradiction-Aware
  Distance Penalty (CADP): it shows corrected facts outrank contradicted ones.
  A detector that flagged every pair would score perfectly on EXP18.
- This experiment measures the FALSE POSITIVE side: how often
  MemoryStore.add() links an older memory as `contradicted_by` a newer one when
  the older memory is not in fact superseded.
- Negative pairs are grouped into families and scored per family, because a
  single pooled rate hides which family the detector fails on.
- A true-contradiction positive set is included so the report carries both a
  false positive rate and a true positive rate. A false positive rate alone is
  gameable by a detector that never fires.

Configuration note
- The shipped default for `contradiction_penalty` in MemoryProfile.custom is
  0.0 and `enable_contradiction_awareness` defaults to False, so CADP is
  opt-in. EXP18 ran at a penalty of 0.20, which is NOT the shipped default.
  This experiment records the exact value used in every condition.

Outputs
- experiments/results/exp23/exp23_cadp_false_positives.json
- experiments/results/exp23/exp23_cadp_false_positives.txt
- experiments/results/exp23/exp23_per_family_link_rate.png
- experiments/results/exp23/exp23_threshold_sweep.png
- experiments/results/exp23/exp23_false_positive_rank_shift.png
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import (  # noqa: E402
    MemoryEntry,
    MemoryProfile,
    MemoryStore,
    SentenceEncoder,
    retrieve_top_k_fast,
)

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Primary condition. These are the exact values EXP18 uses in
# experiments/python/exp18_contradiction_aware_retrieval.py::make_profile,
# so that this experiment characterises the same detector configuration whose
# true-positive behaviour EXP18 reports.
PRIMARY_SIM_THRESHOLD = 0.82
PRIMARY_REQUIRES_MARKER = True
PRIMARY_PENALTY = 0.20
PRIMARY_QUERY_GATE = 1.0
PRIMARY_WRITE_CONFLICT_TRACE = False

# Library defaults, read from ncm/memory.py and ncm/retrieval.py.
SHIPPED_PENALTY_DEFAULT = 0.0
SHIPPED_SIM_THRESHOLD_DEFAULT = 0.85
SHIPPED_REQUIRES_MARKER_DEFAULT = True
SHIPPED_ENABLE_DEFAULT = False

# Semantic similarity thresholds swept for the precision/recall curve.
SWEEP_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95, 0.99]

# The 7D state vector used for every write and every query, matching the
# state7() helper in EXP18 (5D auto-state padded to 7D with the constant 0.5).
FIXED_STATE_7D = np.full(7, 0.5, dtype=np.float32)

# ---------------------------------------------------------------------------
# HAND-AUTHORED SYNTHETIC DATASET
# ---------------------------------------------------------------------------
# Every pair below was written by hand by the experiment author. There is no
# external source and no second annotator. `label` is the author's ground truth
# for the question "is the older memory actually superseded by the newer one":
#   "link"      -> the older memory is genuinely superseded, a link is correct
#   "no_link"   -> the older memory is still valid, a link is a false positive
#   "ambiguous" -> the author judges the ground truth genuinely undecidable;
#                  these are scored separately and excluded from precision,
#                  recall and F1, because forcing an arbitrary label would make
#                  the precision number meaningless.
# `probe` is a hand-authored retrieval query aimed at the OLDER memory, used
# only for the retrieval-consequence measurement.

POSITIVE_PAIRS: list[dict[str, Any]] = [
    {
        "family": "true_positive_marked",
        "older": "I live in Berlin.",
        "newer": "Correction: I live in Munich.",
        "probe": "Where do I live?",
        "label": "link",
        "rationale": "Explicit correction marker and a single-valued slot. The older city is superseded.",
    },
    {
        "family": "true_positive_marked",
        "older": "My office is on the third floor.",
        "newer": "Correction: my office is on the fifth floor.",
        "probe": "Which floor is my office on?",
        "label": "link",
        "rationale": "Explicit correction of a single-valued slot.",
    },
    {
        "family": "true_positive_marked",
        "older": "My flight leaves at 9 am.",
        "newer": "Update: my flight leaves at 11 am.",
        "probe": "What time does my flight leave?",
        "label": "link",
        "rationale": "Marked update to a single-valued slot. The earlier departure time no longer holds.",
    },
    {
        "family": "true_positive_marked",
        "older": "My employee id is E-101.",
        "newer": "Correction: my employee id is E-222.",
        "probe": "What is my employee id?",
        "label": "link",
        "rationale": "Explicit correction of an identifier.",
    },
    {
        "family": "true_positive_marked",
        "older": "I drive a blue sedan.",
        "newer": "Actually I drive a grey sedan.",
        "probe": "What colour is my car?",
        "label": "link",
        "rationale": "The marker 'actually' here does correct the previous colour claim.",
    },
    {
        "family": "true_positive_marked",
        "older": "My hometown is Pune.",
        "newer": "Revised: my hometown is Mumbai.",
        "probe": "What is my hometown?",
        "label": "link",
        "rationale": "Explicit revision of a single-valued slot.",
    },
]

# "Unmarked" here means the sentence carries no explicit correction word such as
# "correction", "update", "revised" or "actually". A bare temporal adverb such
# as "now" is counted as unmarked, because a human reader does not treat "now"
# as a correction marker. The detector's marker list does contain "now", which
# is itself one of the findings of this experiment.
POSITIVE_PAIRS += [
    {
        "family": "true_positive_unmarked",
        "older": "My manager is Ana.",
        "newer": "My manager is Ravi now.",
        "probe": "Who is my manager?",
        "label": "link",
        "rationale": "Single-valued reporting relationship. The older manager is superseded.",
    },
    {
        "family": "true_positive_unmarked",
        "older": "My manager is Ana.",
        "newer": "My manager is Ravi.",
        "probe": "Who is my manager?",
        "label": "link",
        "rationale": "Same supersession with no temporal adverb at all.",
    },
    {
        "family": "true_positive_unmarked",
        "older": "My phone number is 555-0100.",
        "newer": "My phone number is 555-0199.",
        "probe": "What is my phone number?",
        "label": "link",
        "rationale": "Single-valued contact detail, silently replaced.",
    },
    {
        "family": "true_positive_unmarked",
        "older": "I live in Berlin.",
        "newer": "I live in Munich.",
        "probe": "Where do I live?",
        "label": "link",
        "rationale": "Single-valued residence, silently replaced.",
    },
    {
        "family": "true_positive_unmarked",
        "older": "My desk is in room 12.",
        "newer": "My desk is in room 30.",
        "probe": "Which room is my desk in?",
        "label": "link",
        "rationale": "Single-valued desk location, silently replaced.",
    },
    {
        "family": "true_positive_unmarked",
        "older": "My laptop password expires in June.",
        "newer": "My laptop password expires in August.",
        "probe": "When does my laptop password expire?",
        "label": "link",
        "rationale": "Single-valued expiry date, silently replaced.",
    },
]

# Ambiguous bucket. The author judges these genuinely undecidable: "now"
# signals a change of current activity, but the older statement was true when
# written and the underlying slot is not single-valued, so the older memory may
# still hold. These are reported separately and are excluded from precision,
# recall and F1.
AMBIGUOUS_PAIRS: list[dict[str, Any]] = [
    {
        "family": "incidental_temporal",
        "older": "I am reading a book about birds.",
        "newer": "I am now reading a book about volcanoes.",
        "probe": "What book am I reading?",
        "label": "ambiguous",
        "rationale": "A person can read two books at once, so the older may still hold.",
    },
    {
        "family": "incidental_temporal",
        "older": "I am working on the parser.",
        "newer": "I am now working on the optimizer.",
        "probe": "What am I working on?",
        "label": "ambiguous",
        "rationale": "Concurrent workstreams are normal, so supersession is not implied.",
    },
    {
        "family": "incidental_temporal",
        "older": "I take the bus to work.",
        "newer": "I am now cycling to work.",
        "probe": "How do I get to work?",
        "label": "ambiguous",
        "rationale": "Commute mode can vary by day, so the older may still hold.",
    },
    {
        "family": "incidental_temporal",
        "older": "I listen to jazz in the evening.",
        "newer": "I am now listening to classical music.",
        "probe": "What music do I listen to?",
        "label": "ambiguous",
        "rationale": "A momentary activity does not clearly cancel a stated habit.",
    },
]

NEGATIVE_PAIRS: list[dict[str, Any]] = [
    # Family 1: a correction marker is present but nothing is corrected.
    {
        "family": "marked_non_correction",
        "older": "The project started in March.",
        "newer": "Update: the project is still on schedule.",
        "probe": "When did the project start?",
        "label": "no_link",
        "rationale": "The marker introduces a status report, not a replacement of the start date.",
    },
    {
        "family": "marked_non_correction",
        "older": "I think the design is good.",
        "newer": "Actually, I agree with you.",
        "probe": "What do I think of the design?",
        "label": "no_link",
        "rationale": "'Actually' is used as a discourse particle expressing agreement, not correction.",
    },
    {
        "family": "marked_non_correction",
        "older": "My passport expires next year.",
        "newer": "I need to update my passport photo.",
        "probe": "When does my passport expire?",
        "label": "no_link",
        "rationale": "'update' is the main verb of a new task, not a correction of the expiry date.",
    },
    {
        "family": "marked_non_correction",
        "older": "We shipped the release on Tuesday.",
        "newer": "Update: no changes since yesterday.",
        "probe": "When did we ship the release?",
        "label": "no_link",
        "rationale": "The marker introduces an explicit statement that nothing changed.",
    },
    {
        "family": "marked_non_correction",
        "older": "I read the quarterly report.",
        "newer": "Actually, that was a useful report.",
        "probe": "Did I read the quarterly report?",
        "label": "no_link",
        "rationale": "An evaluative remark about the same object, not a replacement of it.",
    },
    {
        "family": "marked_non_correction",
        "older": "The server has 32 GB of memory.",
        "newer": "Instead of guessing, let us measure the server memory.",
        "probe": "How much memory does the server have?",
        "label": "no_link",
        "rationale": "'instead' governs a proposed action, not a revised memory figure.",
    },
    {
        "family": "marked_non_correction",
        "older": "My deadline is on Friday.",
        "newer": "Update: I have nothing new to report about the deadline.",
        "probe": "When is my deadline?",
        "label": "no_link",
        "rationale": "A marked non-update. The deadline is unchanged and explicitly so.",
    },
    # Family 2: the detector's marker test is a bare substring test, so words
    # that merely contain a marker as a substring trip the gate. These pairs
    # probe that. "know", "snow", "nowhere", "renowned" all contain "now".
    {
        "family": "marker_substring_artifact",
        "older": "My mentor is Sara.",
        "newer": "My mentor is renowned in the field.",
        "probe": "Who is my mentor?",
        "label": "no_link",
        "rationale": "A compliment about the same mentor. 'renowned' contains the substring 'now'.",
    },
    {
        "family": "marker_substring_artifact",
        "older": "The answer is unclear.",
        "newer": "I know the answer is 42.",
        "probe": "What is the answer?",
        "label": "no_link",
        "rationale": "Resolution of uncertainty. 'know' contains the substring 'now'.",
    },
    {
        "family": "marker_substring_artifact",
        "older": "The weather is warm.",
        "newer": "It snowed heavily last night.",
        "probe": "What is the weather like?",
        "label": "no_link",
        "rationale": "Two weather observations at different times. 'snowed' contains 'now'.",
    },
    {
        "family": "marker_substring_artifact",
        "older": "The parking lot is full.",
        "newer": "There is nowhere to park.",
        "probe": "Is there parking available?",
        "label": "no_link",
        "rationale": "A restatement, not a correction. 'nowhere' contains 'now'.",
    },
    {
        "family": "marker_substring_artifact",
        "older": "My knowledge of Rust is basic.",
        "newer": "My knowledge of Rust is still improving.",
        "probe": "How good is my Rust?",
        "label": "no_link",
        "rationale": "Compatible statements about the same skill. 'knowledge' contains 'now'.",
    },
    {
        "family": "marker_substring_artifact",
        "older": "The report is ready.",
        "newer": "It is snowing outside.",
        "probe": "Is the report ready?",
        "label": "no_link",
        "rationale": "Unrelated content that nonetheless contains the substring 'now'.",
    },
]

NEGATIVE_PAIRS += [
    # Family 3: paraphrases. The newer memory restates the same fact in
    # different words. Nothing is superseded, so a link is a false positive.
    {
        "family": "paraphrase_same_fact",
        "older": "My office is on the third floor.",
        "newer": "Actually my office is on floor three.",
        "probe": "Which floor is my office on?",
        "label": "no_link",
        "rationale": "Identical fact, different wording. No value changed.",
    },
    {
        "family": "paraphrase_same_fact",
        "older": "I was born in 1994.",
        "newer": "Update: my year of birth is 1994.",
        "probe": "When was I born?",
        "label": "no_link",
        "rationale": "Restatement of the same year under a marker.",
    },
    {
        "family": "paraphrase_same_fact",
        "older": "My flight leaves at 9 am.",
        "newer": "My flight departs at nine in the morning.",
        "probe": "What time does my flight leave?",
        "label": "no_link",
        "rationale": "Same departure time expressed differently.",
    },
    {
        "family": "paraphrase_same_fact",
        "older": "I work as a data engineer.",
        "newer": "My job is data engineering.",
        "probe": "What is my job?",
        "label": "no_link",
        "rationale": "Same occupation, rephrased.",
    },
    {
        "family": "paraphrase_same_fact",
        "older": "The meeting is at noon.",
        "newer": "The meeting is at 12 pm.",
        "probe": "When is the meeting?",
        "label": "no_link",
        "rationale": "Same time, two notations.",
    },
    # Family 4: topically related but jointly compatible. Both statements are
    # still true and the slot is not single-valued.
    {
        "family": "topically_related_compatible",
        "older": "I speak Tamil.",
        "newer": "I speak English.",
        "probe": "What languages do I speak?",
        "label": "no_link",
        "rationale": "Multilingualism. Both remain true.",
    },
    {
        "family": "topically_related_compatible",
        "older": "I own a bicycle.",
        "newer": "I own a car.",
        "probe": "What vehicles do I own?",
        "label": "no_link",
        "rationale": "Ownership is additive, not exclusive.",
    },
    {
        "family": "topically_related_compatible",
        "older": "One of my hobbies is chess.",
        "newer": "One of my hobbies is painting.",
        "probe": "What are my hobbies?",
        "label": "no_link",
        "rationale": "'One of' makes joint truth explicit. Same extracted subject, so this pair also probes the subject-aligned threshold bypass.",
    },
    {
        "family": "topically_related_compatible",
        "older": "I am allergic to peanuts.",
        "newer": "I am allergic to shellfish.",
        "probe": "What am I allergic to?",
        "label": "no_link",
        "rationale": "Allergies accumulate. Both remain true.",
    },
    {
        "family": "topically_related_compatible",
        "older": "I visited Japan in 2019.",
        "newer": "I visited Peru in 2022.",
        "probe": "Which countries have I visited?",
        "label": "no_link",
        "rationale": "Two distinct completed events.",
    },
    {
        "family": "topically_related_compatible",
        "older": "My cat is called Mochi.",
        "newer": "My dog is called Rex.",
        "probe": "What are my pets called?",
        "label": "no_link",
        "rationale": "Two different pets, high topical overlap.",
    },
]

NEGATIVE_PAIRS += [
    # Family 5: additive detail. The newer memory refines the older one rather
    # than replacing it, so the older memory is still entirely correct.
    {
        "family": "additive_detail",
        "older": "I work at a hospital.",
        "newer": "I work at a hospital in the cardiology ward.",
        "probe": "Where do I work?",
        "label": "no_link",
        "rationale": "The refinement entails the original statement.",
    },
    {
        "family": "additive_detail",
        "older": "My office is in Building B.",
        "newer": "My office is in Building B on the second floor.",
        "probe": "Where is my office?",
        "label": "no_link",
        "rationale": "Added detail, same building.",
    },
    {
        "family": "additive_detail",
        "older": "I studied physics.",
        "newer": "I studied physics at a university in Chennai.",
        "probe": "What did I study?",
        "label": "no_link",
        "rationale": "Added location, same subject of study.",
    },
    {
        "family": "additive_detail",
        "older": "My car is a hatchback.",
        "newer": "My car is a red hatchback.",
        "probe": "What kind of car do I have?",
        "label": "no_link",
        "rationale": "Added colour, same body type. Same extracted subject, so this pair also probes the subject-aligned threshold bypass.",
    },
    {
        "family": "additive_detail",
        "older": "I have a meeting on Monday.",
        "newer": "I have a meeting on Monday with the design team.",
        "probe": "When is my meeting?",
        "label": "no_link",
        "rationale": "Added attendee, same day.",
    },
    # Family 6: legitimate change over time with explicit time anchors. Both
    # statements remain true of their own time reference, so the older memory is
    # not wrong and should not be demoted.
    {
        "family": "legitimate_change_over_time",
        "older": "In 2019 I lived in Delhi.",
        "newer": "In 2023 I live in Berlin.",
        "probe": "Where did I live in 2019?",
        "label": "no_link",
        "rationale": "Each statement is anchored to its own year and both stay true.",
    },
    {
        "family": "legitimate_change_over_time",
        "older": "Last year my title was junior analyst.",
        "newer": "This year my title is senior analyst.",
        "probe": "What was my title last year?",
        "label": "no_link",
        "rationale": "A promotion. The historical title remains a correct record.",
    },
    {
        "family": "legitimate_change_over_time",
        "older": "In my first year I rented a studio flat.",
        "newer": "In my third year I rented a two bedroom flat.",
        "probe": "What did I rent in my first year?",
        "label": "no_link",
        "rationale": "Two separate tenancies, both true of their period.",
    },
    {
        "family": "legitimate_change_over_time",
        "older": "The 2021 budget was 40 thousand.",
        "newer": "The 2024 budget is 90 thousand.",
        "probe": "What was the 2021 budget?",
        "label": "no_link",
        "rationale": "Different fiscal years. Neither figure invalidates the other.",
    },
    # Family 7: same subject, opposite polarity, different time reference.
    # GROUND TRUTH DECISION AND REASONING: labelled no_link. "I used to X" is a
    # statement about the past that stays true permanently, and it already
    # entails that X does not hold now. The present-tense negation is therefore
    # consistent with it rather than a replacement of it. Demoting the older
    # memory would lose the person's history while adding no correctness.
    {
        "family": "polarity_time_shift",
        "older": "I used to smoke.",
        "newer": "I do not smoke.",
        "probe": "Did I ever smoke?",
        "label": "no_link",
        "rationale": "'Used to' already entails present non-smoking. Jointly consistent.",
    },
    {
        "family": "polarity_time_shift",
        "older": "I used to live in Delhi.",
        "newer": "I do not live in Delhi.",
        "probe": "Have I ever lived in Delhi?",
        "label": "no_link",
        "rationale": "Past residence and present non-residence are consistent.",
    },
    {
        "family": "polarity_time_shift",
        "older": "I was a vegetarian for ten years.",
        "newer": "I am not a vegetarian.",
        "probe": "Was I ever a vegetarian?",
        "label": "no_link",
        "rationale": "A completed ten year period remains a true record.",
    },
    {
        "family": "polarity_time_shift",
        "older": "I used to run marathons.",
        "newer": "I no longer run marathons.",
        "probe": "Did I ever run marathons?",
        "label": "no_link",
        "rationale": "The two sentences assert the same thing from opposite directions.",
    },
]

NEGATIVE_PAIRS += [
    # Family 8: unrelated statements with high lexical overlap. These probe
    # whether the semantic threshold alone separates topics, since a shared
    # ambiguous word inflates surface similarity.
    {
        "family": "lexical_overlap_unrelated",
        "older": "The bank by the river flooded.",
        "newer": "I opened an account at the bank.",
        "probe": "What happened at the river bank?",
        "label": "no_link",
        "rationale": "Two senses of 'bank'. Nothing is superseded.",
    },
    {
        "family": "lexical_overlap_unrelated",
        "older": "My python script is slow.",
        "newer": "I saw a python at the zoo.",
        "probe": "Why is my script slow?",
        "label": "no_link",
        "rationale": "Two senses of 'python'.",
    },
    {
        "family": "lexical_overlap_unrelated",
        "older": "My mouse stopped working.",
        "newer": "There is a mouse in the kitchen.",
        "probe": "What is wrong with my mouse?",
        "label": "no_link",
        "rationale": "Two senses of 'mouse'.",
    },
    {
        "family": "lexical_overlap_unrelated",
        "older": "I need to charge my phone.",
        "newer": "The charge on the invoice is wrong.",
        "probe": "Does my phone need charging?",
        "label": "no_link",
        "rationale": "Two senses of 'charge'.",
    },
    {
        "family": "lexical_overlap_unrelated",
        "older": "I booked a table for four.",
        "newer": "The table in the report has four rows.",
        "probe": "How many people is the restaurant booking for?",
        "label": "no_link",
        "rationale": "Two senses of 'table', shared numeral.",
    },
    # Family 9: plainly unrelated statements with low lexical overlap. These are
    # the easy negatives and act as a floor check on the detector.
    {
        "family": "unrelated_statements",
        "older": "I enjoy hiking on weekends.",
        "newer": "The printer on the second floor is jammed.",
        "probe": "What do I do on weekends?",
        "label": "no_link",
        "rationale": "No shared subject or topic.",
    },
    {
        "family": "unrelated_statements",
        "older": "My favourite cuisine is Korean food.",
        "newer": "The quarterly audit begins in November.",
        "probe": "What cuisine do I like?",
        "label": "no_link",
        "rationale": "No shared subject or topic.",
    },
    {
        "family": "unrelated_statements",
        "older": "I use Linux for development.",
        "newer": "My neighbour adopted two kittens.",
        "probe": "What operating system do I use?",
        "label": "no_link",
        "rationale": "No shared subject or topic.",
    },
    {
        "family": "unrelated_statements",
        "older": "My commute takes about 25 minutes.",
        "newer": "The library closes at eight on Sundays.",
        "probe": "How long is my commute?",
        "label": "no_link",
        "rationale": "No shared subject or topic.",
    },
    {
        "family": "unrelated_statements",
        "older": "I read science fiction novels.",
        "newer": "The boiler needs servicing before winter.",
        "probe": "What genre do I read?",
        "label": "no_link",
        "rationale": "No shared subject or topic.",
    },
]

ALL_PAIRS: list[dict[str, Any]] = POSITIVE_PAIRS + NEGATIVE_PAIRS + AMBIGUOUS_PAIRS

# Hand-authored multi-step chains. After all three are written, BOTH earlier
# memories are superseded and both should carry a contradicted_by link.
CHAINS: list[dict[str, Any]] = [
    {
        "chain_id": "chain_seat",
        "texts": [
            "My office seat is A-11.",
            "Update: my office seat is B-22.",
            "Correction: my office seat is C-33.",
        ],
    },
    {
        "chain_id": "chain_project_code",
        "texts": [
            "My project code is ALPHA.",
            "Update: my project code is BETA.",
            "Final correction: my project code is GAMMA.",
        ],
    },
    {
        "chain_id": "chain_city",
        "texts": [
            "I live in Berlin.",
            "Correction: I live in Munich.",
            "Correction: I live in Hamburg.",
        ],
    },
]

# Hand-authored distractor memories used only to give the retrieval-consequence
# measurement a realistic store to rank within.
DISTRACTORS: list[str] = [
    "I enjoy hiking on weekends.",
    "My favourite cuisine is Korean food.",
    "I usually work out in the evening.",
    "I am learning distributed systems.",
    "I read science fiction novels.",
    "I have a dentist appointment next Tuesday.",
    "I use Linux for development.",
    "I prefer quiet cafes for work.",
    "I like jazz and classical music.",
    "My commute takes about 25 minutes.",
    "The printer on the second floor is jammed.",
    "The quarterly audit begins in November.",
]


# ---------------------------------------------------------------------------
# ENCODING AND PROFILE HELPERS
# ---------------------------------------------------------------------------

class EncoderCache:
    """Encodes each distinct text once. Vectors are reused across conditions so
    that the only thing varying between conditions is the detector setting."""

    def __init__(self, encoder: SentenceEncoder):
        self.encoder = encoder
        self._sem: dict[str, np.ndarray] = {}
        self.emo = encoder.encode_emotional(FIXED_STATE_7D)
        self.state = encoder.encode_state(FIXED_STATE_7D)

    def semantic(self, text: str) -> np.ndarray:
        if text not in self._sem:
            self._sem[text] = self.encoder.encode(text)
        return self._sem[text]

    def entry(self, text: str, timestamp: int) -> MemoryEntry:
        return MemoryEntry(
            e_semantic=self.semantic(text).copy(),
            e_emotional=self.emo.copy(),
            s_snapshot=self.state.copy(),
            timestamp=timestamp,
            text=text,
        )


def detection_profile(threshold: float, requires_marker: bool) -> MemoryProfile:
    """Profile for the write-time detection measurement.

    `contradiction_penalty` is irrelevant here because no retrieval happens, but
    it is set to the primary value so the recorded profile is unambiguous.
    """
    profile = MemoryProfile(name=f"detect_t{threshold:.2f}_m{int(requires_marker)}")
    profile.set_custom("enable_contradiction_awareness", True)
    profile.set_custom("contradiction_similarity_threshold", threshold)
    profile.set_custom("contradiction_requires_marker", requires_marker)
    profile.set_custom("contradiction_penalty", PRIMARY_PENALTY)
    profile.set_custom("contradiction_query_gate", PRIMARY_QUERY_GATE)
    profile.set_custom("write_conflict_trace", PRIMARY_WRITE_CONFLICT_TRACE)
    return profile


def profile_settings(profile: MemoryProfile) -> dict[str, Any]:
    """Exact recorded settings for a condition, read back off the profile."""
    return {
        "name": profile.name,
        "custom": dict(profile.custom),
        "retrieval_weights": profile.retrieval_weights.to_dict(),
        "semantic_dim": profile.semantic_dim,
        "emotional_dim": profile.emotional_dim,
        "state_dim": profile.state_dim,
        "decay_rate": profile.decay_rate,
        "temperature": profile.temperature,
        "max_size": profile.max_size,
    }


def run_pair_detection(
    cache: EncoderCache,
    older: str,
    newer: str,
    threshold: float,
    requires_marker: bool,
    update_auto_state: bool = False,
) -> dict[str, Any]:
    """Write the two memories into a fresh two-item store and report whether
    add() linked the older one.

    update_auto_state defaults to False. MemoryStore._apply_contradiction_links
    reads only the two texts, the two semantic vectors and the two timestamps,
    so the auto-state has no effect on the linking decision. The equivalence is
    verified empirically by the auto_state_equivalence check rather than
    assumed.
    """
    store = MemoryStore(profile=detection_profile(threshold, requires_marker))
    m_old = cache.entry(older, timestamp=0)
    m_new = cache.entry(newer, timestamp=1)
    store.add(m_old, gate_check=False, update_auto_state=update_auto_state)
    store.step = 1
    store.add(m_new, gate_check=False, update_auto_state=update_auto_state)
    sim = float(np.dot(m_old.e_semantic, m_new.e_semantic))
    return {
        "linked": bool(m_old.contradicted_by == m_new.id),
        "older_contradicted_by_is_newer": bool(m_old.contradicted_by == m_new.id),
        "newer_was_flagged": bool(m_new.contradicted_by is not None),
        "cosine_similarity": round(sim, 6),
        "store_size_after": len(store.get_all_safe()),
    }


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def confusion_from_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confusion matrix for the LINKING decision.

    Positive class is "the older memory should be linked as superseded".
    Rows labelled ambiguous are excluded and counted separately.
    """
    tp = fp = fn = tn = 0
    ambiguous_total = 0
    ambiguous_linked = 0
    for r in rows:
        if r["label"] == "ambiguous":
            ambiguous_total += 1
            ambiguous_linked += 1 if r["linked"] else 0
            continue
        should = r["label"] == "link"
        did = bool(r["linked"])
        if should and did:
            tp += 1
        elif should and not did:
            fn += 1
        elif not should and did:
            fp += 1
        else:
            tn += 1

    scored = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    fp_rate = (fp / (fp + tn)) if (fp + tn) > 0 else None

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "scored_pairs": scored,
        "labelled_positive_pairs": tp + fn,
        "labelled_negative_pairs": fp + tn,
        "precision": None if precision is None else round(precision, 6),
        "precision_denominator": tp + fp,
        "recall": None if recall is None else round(recall, 6),
        "recall_denominator": tp + fn,
        "f1": None if f1 is None else round(f1, 6),
        "false_positive_rate": None if fp_rate is None else round(fp_rate, 6),
        "false_positive_rate_denominator": fp + tn,
        "ambiguous_pairs_excluded": ambiguous_total,
        "ambiguous_pairs_linked": ambiguous_linked,
    }


def per_family_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-family counts. Rates are always reported next to their raw counts."""
    families: dict[str, dict[str, Any]] = {}
    for r in rows:
        fam = families.setdefault(
            r["family"],
            {
                "family": r["family"],
                "ground_truth_label": r["label"],
                "n_pairs": 0,
                "n_linked": 0,
                "linked_pair_ids": [],
                "unlinked_pair_ids": [],
            },
        )
        if fam["ground_truth_label"] != r["label"]:
            raise ValueError(
                f"family {r['family']} mixes ground truth labels; families must be label-homogeneous"
            )
        fam["n_pairs"] += 1
        if r["linked"]:
            fam["n_linked"] += 1
            fam["linked_pair_ids"].append(r["pair_id"])
        else:
            fam["unlinked_pair_ids"].append(r["pair_id"])

    for fam in families.values():
        n = fam["n_pairs"]
        fam["link_rate"] = round(fam["n_linked"] / n, 6) if n else None
        fam["link_rate_as_fraction"] = f"{fam['n_linked']}/{n}"
        label = fam["ground_truth_label"]
        if label == "no_link":
            fam["errors_are"] = "false positives"
            fam["n_errors"] = fam["n_linked"]
        elif label == "link":
            fam["errors_are"] = "false negatives"
            fam["n_errors"] = n - fam["n_linked"]
        else:
            fam["errors_are"] = "not scored (ambiguous)"
            fam["n_errors"] = None
    return families


# ---------------------------------------------------------------------------
# MEASUREMENT 1: PRIMARY CONDITION
# ---------------------------------------------------------------------------

def measure_primary(cache: EncoderCache, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run every hand-authored pair through the primary detector configuration."""
    rows = []
    for idx, spec in enumerate(pairs):
        out = run_pair_detection(
            cache,
            spec["older"],
            spec["newer"],
            PRIMARY_SIM_THRESHOLD,
            PRIMARY_REQUIRES_MARKER,
        )
        rows.append({
            "pair_id": f"p{idx:03d}",
            "family": spec["family"],
            "older": spec["older"],
            "newer": spec["newer"],
            "probe": spec["probe"],
            "label": spec["label"],
            "rationale": spec["rationale"],
            "data_origin": "hand-authored synthetic",
            "linked": out["linked"],
            "cosine_similarity": out["cosine_similarity"],
            "decision_correct": (
                None if spec["label"] == "ambiguous"
                else bool(out["linked"] == (spec["label"] == "link"))
            ),
        })
    return rows


def measure_auto_state_equivalence(cache: EncoderCache, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify empirically that update_auto_state does not change the linking
    decision, which is the assumption that lets the sweep skip auto-state
    updates. Reported rather than assumed."""
    mismatches = []
    for idx, spec in enumerate(pairs):
        a = run_pair_detection(
            cache, spec["older"], spec["newer"],
            PRIMARY_SIM_THRESHOLD, PRIMARY_REQUIRES_MARKER,
            update_auto_state=False,
        )
        b = run_pair_detection(
            cache, spec["older"], spec["newer"],
            PRIMARY_SIM_THRESHOLD, PRIMARY_REQUIRES_MARKER,
            update_auto_state=True,
        )
        if a["linked"] != b["linked"]:
            mismatches.append(f"p{idx:03d}")
    return {
        "pairs_checked": len(pairs),
        "decisions_that_differ": len(mismatches),
        "mismatched_pair_ids": mismatches,
        "conclusion": (
            "auto-state does not affect the linking decision on this set"
            if not mismatches else
            "auto-state DOES affect the linking decision; the sweep assumption is invalid"
        ),
    }


def measure_chains(cache: EncoderCache) -> dict[str, Any]:
    """Write A, B, C in order into one store and check that linking reaches ALL
    superseded ancestors, not only the immediately preceding memory."""
    per_chain = []
    full_coverage = 0
    for spec in CHAINS:
        store = MemoryStore(
            profile=detection_profile(PRIMARY_SIM_THRESHOLD, PRIMARY_REQUIRES_MARKER)
        )
        entries = []
        for i, text in enumerate(spec["texts"]):
            m = cache.entry(text, timestamp=i)
            store.step = i
            store.add(m, gate_check=False, update_auto_state=False)
            entries.append(m)

        id_to_label = {m.id: chr(ord("A") + i) for i, m in enumerate(entries)}
        links = [
            (None if m.contradicted_by is None else id_to_label.get(m.contradicted_by, "unknown"))
            for m in entries
        ]
        a_flagged = entries[0].contradicted_by is not None
        b_flagged = entries[1].contradicted_by is not None
        covered = bool(a_flagged and b_flagged)
        full_coverage += 1 if covered else 0
        per_chain.append({
            "chain_id": spec["chain_id"],
            "texts": list(spec["texts"]),
            "data_origin": "hand-authored synthetic",
            "contradicted_by_label_per_memory": links,
            "oldest_ancestor_flagged": a_flagged,
            "middle_ancestor_flagged": b_flagged,
            "all_superseded_ancestors_flagged": covered,
        })
    return {
        "n_chains": len(CHAINS),
        "n_chains_with_all_ancestors_flagged": full_coverage,
        "coverage_as_fraction": f"{full_coverage}/{len(CHAINS)}",
        "per_chain": per_chain,
    }


# ---------------------------------------------------------------------------
# MEASUREMENT 2: THRESHOLD AND MARKER GATE SWEEP
# ---------------------------------------------------------------------------

def measure_sweep(cache: EncoderCache, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sweep contradiction_similarity_threshold crossed with
    contradiction_requires_marker, so the result characterises the detector
    rather than reporting one opaque score."""
    configs = []
    for requires_marker in (True, False):
        for threshold in SWEEP_THRESHOLDS:
            rows = []
            for idx, spec in enumerate(pairs):
                out = run_pair_detection(
                    cache, spec["older"], spec["newer"], threshold, requires_marker
                )
                rows.append({
                    "pair_id": f"p{idx:03d}",
                    "family": spec["family"],
                    "label": spec["label"],
                    "linked": out["linked"],
                })
            cm = confusion_from_decisions(rows)
            fams = per_family_breakdown(rows)
            configs.append({
                "contradiction_similarity_threshold": threshold,
                "contradiction_requires_marker": requires_marker,
                "confusion": cm,
                "false_positives_by_family": {
                    name: f["n_linked"]
                    for name, f in sorted(fams.items())
                    if f["ground_truth_label"] == "no_link" and f["n_linked"] > 0
                },
            })
    return {
        "thresholds_swept": list(SWEEP_THRESHOLDS),
        "marker_gate_values_swept": [True, False],
        "n_configs": len(configs),
        "note": (
            "All rates below are computed over the hand-authored synthetic pair "
            "set described in this file. Raw counts accompany every rate."
        ),
        "configs": configs,
    }


def measure_marker_gate(cache: EncoderCache) -> dict[str, Any]:
    """Map the marker gate empirically instead of trusting a hard-coded list.

    The prefixes below are artificial mechanism probes, not natural sentences.
    They hold the content change fixed and vary only the leading word, so a
    change in the linking decision is attributable to the marker test alone.
    """
    older = "My meeting room is R12."
    prefixes = [
        "", "correction: ", "update: ", "actually ", "instead ", "revised: ",
        "now ", "know ", "snow ", "nowhere ", "renowned ", "furthermore ",
        "moreover ", "additionally ",
    ]
    probes = []
    for prefix in prefixes:
        newer = f"{prefix}my meeting room is R30."
        with_gate = run_pair_detection(
            cache, older, newer, PRIMARY_SIM_THRESHOLD, True
        )
        without_gate = run_pair_detection(
            cache, older, newer, PRIMARY_SIM_THRESHOLD, False
        )
        probes.append({
            "leading_token": prefix.strip() or "(none)",
            "newer_text": newer,
            "linked_with_marker_gate_on": with_gate["linked"],
            "linked_with_marker_gate_off": without_gate["linked"],
            "cosine_similarity": with_gate["cosine_similarity"],
        })
    passing = [p["leading_token"] for p in probes if p["linked_with_marker_gate_on"]]
    return {
        "probe_kind": "artificial mechanism probe, hand-authored, not natural language",
        "fixed_older_text": older,
        "n_probes": len(probes),
        "tokens_that_satisfy_the_marker_gate": passing,
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# MEASUREMENT 3: RETRIEVAL CONSEQUENCE OF A FALSE POSITIVE
# ---------------------------------------------------------------------------

def _build_retrieval_store(cache: EncoderCache, older: str, newer: str) -> MemoryStore:
    """Distractors, then the pair, written the way EXP18 writes memories:
    update_auto_state=True and timestamp taken from an incrementing store.step.
    """
    store = MemoryStore(
        profile=detection_profile(PRIMARY_SIM_THRESHOLD, PRIMARY_REQUIRES_MARKER)
    )
    for text in DISTRACTORS + [older, newer]:
        m = cache.entry(text, timestamp=int(store.step))
        store.add(m, gate_check=False, update_auto_state=True)
        store.step += 1
    return store


def _ranked_texts(cache: EncoderCache, store: MemoryStore, query: str, k: int) -> list[tuple[str, float]]:
    """Retrieve with the shipped retrieval function, not a bespoke scorer."""
    rows = retrieve_top_k_fast(
        cache.semantic(query), cache.emo, store, cache.state,
        store.step, k=k, use_strength=False,
    )
    return [(m.text, float(dist)) for dist, _prob, m in rows]


def _rank_of(ranked: list[tuple[str, float]], needle: str) -> tuple[int, float]:
    for i, (text, dist) in enumerate(ranked, 1):
        if text == needle:
            return i, dist
    return -1, float("nan")


def measure_retrieval_consequence(
    cache: EncoderCache, false_positive_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """For each wrongly flagged pair, measure how far the flagged memory falls.

    The two measurements are taken on the SAME store, toggling only
    `contradiction_penalty` between 0.0 (the shipped default) and 0.20 (the
    value EXP18 used). Links, embeddings, timestamps and auto-state are
    therefore byte-identical across the two readings, so the entire rank change
    is attributable to the penalty and not to a difference in detection.
    """
    cases = []
    for row in false_positive_rows:
        store = _build_retrieval_store(cache, row["older"], row["newer"])
        flagged = [m for m in store.get_all_safe() if m.text == row["older"]]
        link_present = bool(flagged and flagged[0].contradicted_by is not None)

        # Pin the query state explicitly. retrieve_top_k_fast ignores its
        # s_current_normalized argument and reads store.auto_state instead
        # (ncm/retrieval.py:372), so the query state must be set on the store.
        store.auto_state.state = np.full(5, 0.5, dtype=np.float32)

        k = len(store.get_all_safe())
        for query_kind, query in (("hand_authored_probe", row["probe"]), ("exact_older_text", row["older"])):
            store.profile.set_custom("contradiction_penalty", 0.0)
            before = _ranked_texts(cache, store, query, k)
            store.profile.set_custom("contradiction_penalty", PRIMARY_PENALTY)
            after = _ranked_texts(cache, store, query, k)
            store.profile.set_custom("contradiction_penalty", PRIMARY_PENALTY)

            r_before, d_before = _rank_of(before, row["older"])
            r_after, d_after = _rank_of(after, row["older"])
            cases.append({
                "pair_id": row["pair_id"],
                "family": row["family"],
                "older_wrongly_flagged": row["older"],
                "newer": row["newer"],
                "query_kind": query_kind,
                "query": query,
                "link_present_in_store": link_present,
                "store_size": k,
                "rank_penalty_0.0": r_before,
                "rank_penalty_0.20": r_after,
                "rank_drop": (r_after - r_before) if (r_before > 0 and r_after > 0) else None,
                "distance_penalty_0.0": round(d_before, 6),
                "distance_penalty_0.20": round(d_after, 6),
                "left_top_1": bool(r_before == 1 and r_after != 1),
                "left_top_3": bool(r_before <= 3 and r_after > 3),
            })

    probe_cases = [c for c in cases if c["query_kind"] == "hand_authored_probe"]
    drops = [c["rank_drop"] for c in probe_cases if c["rank_drop"] is not None]
    return {
        "penalty_compared": {"baseline": 0.0, "cadp": PRIMARY_PENALTY},
        "shipped_default_penalty": SHIPPED_PENALTY_DEFAULT,
        "note": (
            "Measured on hand-authored synthetic pairs. Both readings come from "
            "the same store with only contradiction_penalty toggled."
        ),
        "n_false_positive_pairs_measured": len(false_positive_rows),
        "n_cases": len(cases),
        "probe_query_median_rank_drop": (
            float(np.median(drops)) if drops else None
        ),
        "probe_query_max_rank_drop": (max(drops) if drops else None),
        "probe_query_n_left_top_1": sum(1 for c in probe_cases if c["left_top_1"]),
        "probe_query_n_left_top_3": sum(1 for c in probe_cases if c["left_top_3"]),
        "probe_query_n_cases": len(probe_cases),
        "cases": cases,
    }


def measure_ignored_state_argument(cache: EncoderCache) -> dict[str, Any]:
    """Demonstrate that retrieve_top_k_fast ignores s_current_normalized.

    This is recorded as an observed behaviour of the shipped code, not as a
    claim about intent. It matters here because any experiment that believes it
    is controlling the query state through that argument is not controlling it.
    """
    store = _build_retrieval_store(
        cache, "My desk is in room 12.", "My desk is in room 30."
    )
    query = "Which room is my desk in?"
    k = len(store.get_all_safe())

    zeros = np.zeros(store.profile.state_dim, dtype=np.float32)
    ones = np.ones(store.profile.state_dim, dtype=np.float32)

    rows_zeros = retrieve_top_k_fast(
        cache.semantic(query), cache.emo, store, zeros, store.step, k=k, use_strength=False
    )
    rows_ones = retrieve_top_k_fast(
        cache.semantic(query), cache.emo, store, ones, store.step, k=k, use_strength=False
    )
    same_order = [m.id for _, _, m in rows_zeros] == [m.id for _, _, m in rows_ones]
    same_dists = bool(np.allclose(
        [d for d, _, _ in rows_zeros], [d for d, _, _ in rows_ones], atol=0.0, rtol=0.0
    ))

    # Now change the state the function actually reads.
    baseline = [d for d, _, _ in rows_zeros]
    store.auto_state.state = np.zeros(5, dtype=np.float32)
    rows_state_zero = retrieve_top_k_fast(
        cache.semantic(query), cache.emo, store, zeros, store.step, k=k, use_strength=False
    )
    store.auto_state.state = np.ones(5, dtype=np.float32)
    rows_state_one = retrieve_top_k_fast(
        cache.semantic(query), cache.emo, store, zeros, store.step, k=k, use_strength=False
    )
    store_state_changes_result = not bool(np.allclose(
        [d for d, _, _ in rows_state_zero], [d for d, _, _ in rows_state_one]
    ))

    return {
        "argument_under_test": "s_current_normalized",
        "read_site_in_source": "ncm/retrieval.py:372 reads store.auto_state.get_current_state()",
        "passing_zeros_vs_ones_gives_same_order": same_order,
        "passing_zeros_vs_ones_gives_identical_distances": same_dists,
        "assigning_store_auto_state_changes_distances": store_state_changes_result,
        "baseline_top_distance": round(float(baseline[0]), 6) if baseline else None,
        "conclusion": (
            "s_current_normalized is accepted and ignored; the query state must "
            "be set by assigning store.auto_state.state"
            if same_dists and store_state_changes_result else
            "behaviour did not match the expected pattern; inspect manually"
        ),
    }


# ---------------------------------------------------------------------------
# FIGURES. Every subtitle discloses that the data is hand-authored synthetic.
# ---------------------------------------------------------------------------

DISCLOSURE = "All data is hand-authored synthetic text written by the experiment author; raw counts shown"


def plot_per_family(families: dict[str, dict[str, Any]], out_path: str) -> str:
    ordered = sorted(
        families.values(),
        key=lambda f: (f["ground_truth_label"], -f["link_rate"], f["family"]),
    )
    names = [f["family"] for f in ordered]
    rates = [f["link_rate"] for f in ordered]
    colors = []
    for f in ordered:
        if f["ground_truth_label"] == "link":
            colors.append("#2e7d32" if f["link_rate"] >= 0.999 else "#a5d6a7")
        elif f["ground_truth_label"] == "no_link":
            colors.append("#c62828" if f["link_rate"] > 0 else "#90a4ae")
        else:
            colors.append("#f9a825")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ypos = np.arange(len(names))
    ax.barh(ypos, rates, color=colors, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{n}\n[{f['ground_truth_label']}]" for n, f in zip(names, ordered)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Fraction of pairs the detector linked as contradicted")
    for y, f in zip(ypos, ordered):
        ax.text(f["link_rate"] + 0.02, y, f["link_rate_as_fraction"], va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_title(
        "EXP23 CADP link rate per family\n"
        f"threshold={PRIMARY_SIM_THRESHOLD}, requires_marker={PRIMARY_REQUIRES_MARKER}. "
        "Green: correct links. Red: FALSE POSITIVES. Amber: ambiguous, unscored.\n"
        + DISCLOSURE,
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_threshold_sweep(sweep: dict[str, Any], out_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, marker_on in zip(axes, (True, False)):
        rows = [c for c in sweep["configs"] if c["contradiction_requires_marker"] is marker_on]
        rows.sort(key=lambda c: c["contradiction_similarity_threshold"])
        xs = [c["contradiction_similarity_threshold"] for c in rows]
        prec = [c["confusion"]["precision"] for c in rows]
        rec = [c["confusion"]["recall"] for c in rows]
        f1 = [c["confusion"]["f1"] for c in rows]
        fpr = [c["confusion"]["false_positive_rate"] for c in rows]
        ax.plot(xs, prec, "o-", label="precision", color="#1565c0")
        ax.plot(xs, rec, "s-", label="recall", color="#2e7d32")
        ax.plot(xs, f1, "^-", label="F1", color="#6a1b9a")
        ax.plot(xs, fpr, "x--", label="false positive rate", color="#c62828")
        ax.axvline(PRIMARY_SIM_THRESHOLD, color="#546e7a", ls=":", lw=1.5)
        n_pos = rows[0]["confusion"]["labelled_positive_pairs"] if rows else 0
        n_neg = rows[0]["confusion"]["labelled_negative_pairs"] if rows else 0
        ax.set_title(
            f"contradiction_requires_marker = {marker_on}\n"
            f"{n_pos} positive and {n_neg} negative hand-authored pairs",
            fontsize=9,
        )
        ax.set_xlabel("contradiction_similarity_threshold")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rate")
    fig.suptitle(
        "EXP23 CADP detector characterisation across the semantic threshold and the marker gate\n"
        "Dotted line marks the EXP18 threshold of 0.82. " + DISCLOSURE,
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_rank_shift(consequence: dict[str, Any], out_path: str) -> str:
    cases = [c for c in consequence["cases"] if c["query_kind"] == "hand_authored_probe"]
    fig, ax = plt.subplots(figsize=(11, 6.5))

    if not cases:
        ax.text(0.5, 0.5, "No false positives were produced,\nso there is no rank shift to plot.",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        for c in cases:
            before, after = c["rank_penalty_0.0"], c["rank_penalty_0.20"]
            ax.plot([0, 1], [before, after], "-", color="#90a4ae", lw=1.2, zorder=1)
            ax.scatter([0], [before], color="#1565c0", zorder=2)
            ax.scatter([1], [after], color="#c62828", zorder=2)
            ax.annotate(
                f"{c['pair_id']} {c['family']}",
                xy=(1.02, after), fontsize=7, va="center",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            "contradiction_penalty = 0.0\n(shipped default)",
            f"contradiction_penalty = {PRIMARY_PENALTY}\n(value used by EXP18)",
        ], fontsize=9)
        ax.set_xlim(-0.15, 1.6)
        ax.invert_yaxis()
        ax.set_ylabel("Rank of the wrongly flagged memory (1 = best)")
        ax.grid(True, alpha=0.3, axis="y")

    ax.set_title(
        "EXP23 retrieval consequence of a CADP false positive\n"
        f"{len(cases)} wrongly flagged pairs, same store, only the penalty toggled, "
        "hand-authored probe query\n" + DISCLOSURE,
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# TEXT REPORT
# ---------------------------------------------------------------------------

def write_text_report(results: dict[str, Any], out_path: str) -> None:
    cm = results["primary_condition"]["confusion"]
    L: list[str] = []
    A = L.append

    A("EXP23: CADP Detection Precision and False Positive Characterisation")
    A("=" * 78)
    A("")
    A("DATA DISCLOSURE")
    A("-" * 78)
    A("Every memory pair, chain and distractor used in this experiment is")
    A("HAND-AUTHORED SYNTHETIC TEXT written by the experiment author. There is no")
    A("external corpus, no independent annotator and no inter-annotator agreement")
    A("statistic. The ground-truth labels are one author's judgement. Every number")
    A("in this file is a property of that hand-authored set and of the detector, and")
    A("carries no claim about natural user data.")
    A("")
    A("CONFIGURATION")
    A("-" * 78)
    A(f"seed: {results['run_metadata']['seed']}")
    A(f"encoder_backend: {results['run_metadata']['encoder_backend']}")
    A(f"primary contradiction_similarity_threshold: {PRIMARY_SIM_THRESHOLD}")
    A(f"primary contradiction_requires_marker: {PRIMARY_REQUIRES_MARKER}")
    A(f"contradiction_penalty used for the retrieval measurement: {PRIMARY_PENALTY}")
    A(f"shipped default contradiction_penalty: {SHIPPED_PENALTY_DEFAULT}")
    A(f"shipped default enable_contradiction_awareness: {SHIPPED_ENABLE_DEFAULT}")
    A(f"shipped default contradiction_similarity_threshold: {SHIPPED_SIM_THRESHOLD_DEFAULT}")
    A("The primary threshold and marker settings are the values EXP18 uses, so this")
    A("experiment characterises the same configuration whose true-positive behaviour")
    A(f"EXP18 reports. A penalty of {PRIMARY_PENALTY} is NOT the shipped default.")
    A("")
    A("CONFUSION MATRIX FOR THE LINKING DECISION, PRIMARY CONDITION")
    A("-" * 78)
    A("Positive class: the older memory genuinely is superseded and should be linked.")
    A("Computed over the hand-authored synthetic pair set.")
    A(f"  true positives  : {cm['true_positives']}")
    A(f"  false positives : {cm['false_positives']}")
    A(f"  false negatives : {cm['false_negatives']}")
    A(f"  true negatives  : {cm['true_negatives']}")
    A(f"  scored pairs    : {cm['scored_pairs']} "
      f"({cm['labelled_positive_pairs']} positive, {cm['labelled_negative_pairs']} negative)")
    A(f"  ambiguous pairs excluded from all rates: {cm['ambiguous_pairs_excluded']} "
      f"(of which the detector linked {cm['ambiguous_pairs_linked']})")
    A("")

    def fmt(rate, num, den):
        if rate is None:
            return "undefined (zero denominator)"
        return f"{rate:.3f}  ({num}/{den})"

    A(f"  precision           : {fmt(cm['precision'], cm['true_positives'], cm['precision_denominator'])}")
    A(f"  recall              : {fmt(cm['recall'], cm['true_positives'], cm['recall_denominator'])}")
    A(f"  F1                  : {'undefined' if cm['f1'] is None else format(cm['f1'], '.3f')}")
    A(f"  false positive rate : {fmt(cm['false_positive_rate'], cm['false_positives'], cm['false_positive_rate_denominator'])}")
    A("")
    A("These denominators are small. A precision computed over a handful of items")
    A("must not be read as if it were computed over hundreds.")
    A("")
    A("PER-FAMILY BREAKDOWN, PRIMARY CONDITION")
    A("-" * 78)
    A("Families are hand-authored and label-homogeneous. 'linked' means add() set")
    A("contradicted_by on the older memory.")
    A("")
    A(f"  {'family':<32} {'truth':<10} {'linked':<9} {'errors':<8} error kind")
    fams = results["primary_condition"]["per_family"]
    for name in sorted(fams, key=lambda n: (fams[n]["ground_truth_label"], n)):
        f = fams[name]
        n_err = "n/a" if f["n_errors"] is None else str(f["n_errors"])
        A(f"  {name:<32} {f['ground_truth_label']:<10} "
          f"{f['link_rate_as_fraction']:<9} {n_err:<8} {f['errors_are']}")
    A("")
    failing = sorted(
        [f for f in fams.values() if f["ground_truth_label"] == "no_link" and f["n_linked"] > 0],
        key=lambda f: -f["n_linked"],
    )
    if failing:
        A("Negative families where the detector produced false positives:")
        for f in failing:
            A(f"  {f['family']}: {f['n_linked']} of {f['n_pairs']} pairs wrongly linked")
    else:
        A("No negative family produced a false positive in this condition.")
    A("")
    missed = sorted(
        [f for f in fams.values() if f["ground_truth_label"] == "link" and f["n_errors"]],
        key=lambda f: -f["n_errors"],
    )
    if missed:
        A("Positive families the detector missed:")
        for f in missed:
            A(f"  {f['family']}: {f['n_errors']} of {f['n_pairs']} genuine supersessions not linked")
    else:
        A("No genuine supersession was missed in this condition.")
    A("")
    A("MULTI-STEP CHAIN ANCESTOR COVERAGE")
    A("-" * 78)
    ch = results["chains"]
    A("Hand-authored three-step chains A then B then C. Correct behaviour links both")
    A("A and B once C is written.")
    A(f"  chains with all superseded ancestors flagged: {ch['coverage_as_fraction']}")
    for c in ch["per_chain"]:
        A(f"  {c['chain_id']}: contradicted_by per memory = {c['contradicted_by_label_per_memory']}, "
          f"all ancestors flagged = {c['all_superseded_ancestors_flagged']}")
    A("")
    A("AMBIGUOUS BUCKET")
    A("-" * 78)
    amb = results["ambiguous_bucket"]
    A(f"  pairs placed in the ambiguous bucket: {amb['n_pairs']}")
    A(f"  of those, the detector linked: {amb['n_linked']}")
    A("  Reason for the bucket: these pairs use 'now' non-correctively over a slot")
    A("  that is not single-valued, for example reading two books or varying a")
    A("  commute. The author judges supersession genuinely undecidable, so forcing a")
    A("  label would make the precision number meaningless. They are excluded from")
    A("  precision, recall, F1 and the false positive rate, and reported only here.")
    for p in amb["pairs"]:
        A(f"    [{'linked' if p['linked'] else 'not linked'}] {p['older']} -> {p['newer']}")
    A("")
    A("THRESHOLD AND MARKER GATE SWEEP")
    A("-" * 78)
    A("Rates over the hand-authored synthetic set. P = precision, R = recall,")
    A("FPR = false positive rate. TP/FP/FN/TN are raw counts.")
    A("")
    sw = results["threshold_sweep"]
    for marker_on in (True, False):
        A(f"  contradiction_requires_marker = {marker_on}")
        A(f"    {'thresh':<8} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4} "
          f"{'P':<8} {'R':<8} {'F1':<8} FPR")
        rows = [c for c in sw["configs"] if c["contradiction_requires_marker"] is marker_on]
        rows.sort(key=lambda c: c["contradiction_similarity_threshold"])
        for c in rows:
            k = c["confusion"]
            def n(v):
                return "n/a" if v is None else f"{v:.3f}"
            A(f"    {c['contradiction_similarity_threshold']:<8.2f} "
              f"{k['true_positives']:<4} {k['false_positives']:<4} "
              f"{k['false_negatives']:<4} {k['true_negatives']:<4} "
              f"{n(k['precision']):<8} {n(k['recall']):<8} {n(k['f1']):<8} {n(k['false_positive_rate'])}")
        A("")
    A("MARKER GATE MECHANISM PROBE")
    A("-" * 78)
    A("These probe strings are artificial and hand-authored, not natural language.")
    A("They hold the content change fixed and vary only the leading token, so any")
    A("change in the decision is attributable to the marker test alone.")
    mg = results["marker_gate_probe"]
    A(f"  fixed older text: {mg['fixed_older_text']}")
    A(f"  leading tokens that satisfied the marker gate: {mg['tokens_that_satisfy_the_marker_gate']}")
    A("")
    A("RETRIEVAL CONSEQUENCE OF A FALSE POSITIVE")
    A("-" * 78)
    rc = results["retrieval_consequence"]
    A("Measured on the same store with only contradiction_penalty toggled between")
    A(f"{rc['penalty_compared']['baseline']} and {rc['penalty_compared']['cadp']}, "
      f"so links, embeddings, timestamps and auto-state")
    A("are identical across the two readings and the whole rank change is")
    A("attributable to the penalty. Hand-authored synthetic pairs and hand-authored")
    A("distractor memories.")
    A(f"  wrongly flagged pairs measured: {rc['n_false_positive_pairs_measured']}")
    if rc["n_false_positive_pairs_measured"] == 0:
        A("  No false positives were produced in the primary condition, so there is no")
        A("  retrieval consequence to report for it.")
    else:
        A(f"  cases with a hand-authored probe query: {rc['probe_query_n_cases']}")
        A(f"  median rank drop: {rc['probe_query_median_rank_drop']}")
        A(f"  max rank drop: {rc['probe_query_max_rank_drop']}")
        A(f"  wrongly flagged memory pushed out of rank 1: "
          f"{rc['probe_query_n_left_top_1']}/{rc['probe_query_n_cases']}")
        A(f"  wrongly flagged memory pushed out of the top 3: "
          f"{rc['probe_query_n_left_top_3']}/{rc['probe_query_n_cases']}")
        A("")
        A("  Per case, hand-authored probe query:")
        for c in rc["cases"]:
            if c["query_kind"] != "hand_authored_probe":
                continue
            A(f"    {c['pair_id']} {c['family']}: rank {c['rank_penalty_0.0']} -> "
              f"{c['rank_penalty_0.20']} of {c['store_size']}, distance "
              f"{c['distance_penalty_0.0']:.4f} -> {c['distance_penalty_0.20']:.4f}")
            A(f"      wrongly flagged text: {c['older_wrongly_flagged']}")
            A(f"      query: {c['query']}")
    A("")
    A("OBSERVED IMPLEMENTATION BEHAVIOUR")
    A("-" * 78)
    for item in results["implementation_observations"]:
        A(f"  - {item}")
    A("")
    A("INTERNAL CONSISTENCY CHECKS")
    A("-" * 78)
    eq = results["auto_state_equivalence"]
    A(f"  auto-state equivalence: {eq['decisions_that_differ']} of {eq['pairs_checked']} "
      f"decisions differ when update_auto_state is toggled")
    A(f"    {eq['conclusion']}")
    ig = results["ignored_state_argument"]
    A(f"  retrieve_top_k_fast s_current_normalized ignored: "
      f"identical distances for zeros vs ones = {ig['passing_zeros_vs_ones_gives_identical_distances']}, "
      f"assigning store.auto_state.state changes distances = {ig['assigning_store_auto_state_changes_distances']}")
    A("")
    A("VERDICT")
    A("-" * 78)
    for line in results["verdict_lines"]:
        A(line)
    A("")
    A("Figures:")
    for name, rel in results["plots"].items():
        A(f"  {name}: {rel}")
    A("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------

def build_verdict(cm: dict[str, Any], fams: dict[str, dict[str, Any]], rc: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    fp = cm["false_positives"]
    fp_den = cm["false_positive_rate_denominator"]
    worst = sorted(
        [f for f in fams.values() if f["ground_truth_label"] == "no_link" and f["n_linked"] > 0],
        key=lambda f: -f["n_linked"],
    )
    if fp == 0:
        lines.append(
            f"On this hand-authored synthetic set the detector produced no false "
            f"positives ({fp}/{fp_den} negative pairs wrongly linked). This is a "
            f"small set and does not establish a low false positive rate on real data."
        )
    else:
        names = ", ".join(f"{f['family']} ({f['n_linked']}/{f['n_pairs']})" for f in worst)
        lines.append(
            f"On this hand-authored synthetic set the detector wrongly linked "
            f"{fp} of {fp_den} negative pairs. The failing families are: {names}."
        )
    if cm["false_negatives"]:
        lines.append(
            f"It also missed {cm['false_negatives']} of {cm['recall_denominator']} "
            f"genuine supersessions."
        )
    if rc["n_false_positive_pairs_measured"] and rc["probe_query_n_cases"]:
        lines.append(
            f"When a memory is wrongly flagged, the penalty of {PRIMARY_PENALTY} moved it "
            f"out of rank 1 in {rc['probe_query_n_left_top_1']} of "
            f"{rc['probe_query_n_cases']} cases and out of the top 3 in "
            f"{rc['probe_query_n_left_top_3']} of {rc['probe_query_n_cases']} cases, "
            f"with a median rank drop of {rc['probe_query_median_rank_drop']}."
        )
    lines.append(
        "Conservative reading: the false positive rate above is measured on a small "
        "hand-authored set chosen to span negative families, not sampled from any "
        "population, so it characterises the detector's failure modes rather than "
        "estimating a rate. Treat the identified failure modes as the finding and "
        "the rate itself as illustrative."
    )
    return lines


def run(sweep_enabled: bool = True, pair_limit: int | None = None) -> dict[str, Any]:
    encoder = SentenceEncoder(model_dir=os.path.join(ROOT_DIR, "models"))
    if encoder.backend != "sentence-transformers":
        raise RuntimeError(
            "Encoder fell back to its hash backend "
            f"(backend={encoder.backend!r}, error={encoder.backend_error!r}). "
            "The hash fallback preserves no semantic structure, so no number from "
            "this experiment would be meaningful. Aborting."
        )
    cache = EncoderCache(encoder)

    pairs = ALL_PAIRS if pair_limit is None else ALL_PAIRS[:pair_limit]

    primary_rows = measure_primary(cache, pairs)
    cm = confusion_from_decisions(primary_rows)
    fams = per_family_breakdown(primary_rows)

    fp_rows = [r for r in primary_rows if r["label"] == "no_link" and r["linked"]]
    consequence = measure_retrieval_consequence(cache, fp_rows)

    amb_rows = [r for r in primary_rows if r["label"] == "ambiguous"]
    ambiguous_bucket = {
        "n_pairs": len(amb_rows),
        "n_linked": sum(1 for r in amb_rows if r["linked"]),
        "why_ambiguous": (
            "These hand-authored pairs use 'now' non-correctively over a slot that "
            "is not single-valued. Supersession is genuinely undecidable, so they "
            "are excluded from precision, recall, F1 and the false positive rate."
        ),
        "pairs": amb_rows,
    }

    results: dict[str, Any] = {
        "experiment": "EXP23 CADP detection precision and false positive characterisation",
        "data_disclosure": (
            "ALL pairs, chains and distractor memories in this file are HAND-AUTHORED "
            "SYNTHETIC TEXT written by the experiment author. No external corpus, no "
            "independent annotator, no inter-annotator agreement. Every rate below is "
            "a property of this hand-authored set."
        ),
        "run_metadata": {
            "seed": SEED,
            "encoder_backend": encoder.backend,
            "encoder_backend_error": encoder.backend_error,
            "encoder_model_dir": os.path.join("models", "all-MiniLM-L6-v2"),
            "numpy_version": np.__version__,
            "matplotlib_backend": matplotlib.get_backend(),
            "sweep_enabled": sweep_enabled,
            "pair_limit": pair_limit,
            "n_pairs_run": len(pairs),
        },
        "shipped_defaults": {
            "enable_contradiction_awareness": SHIPPED_ENABLE_DEFAULT,
            "contradiction_penalty": SHIPPED_PENALTY_DEFAULT,
            "contradiction_similarity_threshold": SHIPPED_SIM_THRESHOLD_DEFAULT,
            "contradiction_requires_marker": SHIPPED_REQUIRES_MARKER_DEFAULT,
            "note": "CADP is opt-in. EXP18 used a penalty of 0.20, not the shipped default.",
        },
        "primary_condition": {
            "profile_settings": profile_settings(
                detection_profile(PRIMARY_SIM_THRESHOLD, PRIMARY_REQUIRES_MARKER)
            ),
            "contradiction_penalty_used_for_retrieval": PRIMARY_PENALTY,
            "confusion": cm,
            "per_family": fams,
            "labelled_dataset_with_decisions": primary_rows,
        },
        "ambiguous_bucket": ambiguous_bucket,
        "chains": measure_chains(cache),
        "auto_state_equivalence": measure_auto_state_equivalence(cache, pairs),
        "marker_gate_probe": measure_marker_gate(cache),
        "retrieval_consequence": consequence,
        "ignored_state_argument": measure_ignored_state_argument(cache),
    }

    if sweep_enabled:
        results["threshold_sweep"] = measure_sweep(cache, pairs)
    else:
        results["threshold_sweep"] = {
            "thresholds_swept": [],
            "marker_gate_values_swept": [],
            "n_configs": 0,
            "configs": [],
            "note": "sweep disabled for this run",
        }

    results["implementation_observations"] = [
        "MemoryStore._is_correction_pair tests correction markers with a bare "
        "substring test over the marker list, so any word containing a marker as a "
        "substring satisfies the gate. The marker 'now' is contained in 'know', "
        "'snow', 'nowhere', 'knowledge' and 'renowned'.",
        "When the subject extracted from both texts is equal, "
        "MemoryStore._is_correction_pair lowers the effective similarity threshold to "
        "min(threshold, 0.55), so raising contradiction_similarity_threshold above "
        "0.55 cannot tighten the decision for subject-aligned pairs.",
        "MemoryStore._extract_subject requires the copula ' is ' and returns None for "
        "sentences without it, so many natural statements have no extracted subject.",
        "retrieve_top_k_fast accepts s_current_normalized and never reads it; "
        "ncm/retrieval.py:372 uses store.auto_state.get_current_state() instead.",
        "Enabling a non-zero contradiction_penalty rescales all base distances by "
        "(1 - penalty) in addition to adding the penalty to flagged memories, so a "
        "flagged memory must beat an unflagged one by penalty/(1 - penalty) in raw "
        "distance to keep its position.",
    ]
    results["verdict_lines"] = build_verdict(cm, fams, consequence)

    plots = {
        "per_family_link_rate": plot_per_family(
            fams, os.path.join(RESULTS_DIR, f"{RESULT_BUCKET}_per_family_link_rate.png")
        ),
        "false_positive_rank_shift": plot_rank_shift(
            consequence, os.path.join(RESULTS_DIR, f"{RESULT_BUCKET}_false_positive_rank_shift.png")
        ),
    }
    if sweep_enabled:
        plots["threshold_sweep"] = plot_threshold_sweep(
            results["threshold_sweep"],
            os.path.join(RESULTS_DIR, f"{RESULT_BUCKET}_threshold_sweep.png"),
        )
    results["plots"] = {
        k: os.path.relpath(v, ROOT_DIR).replace("\\", "/") for k, v in plots.items()
    }

    out_json = os.path.join(RESULTS_DIR, f"{RESULT_BUCKET}_cadp_false_positives.json")
    out_txt = os.path.join(RESULTS_DIR, f"{RESULT_BUCKET}_cadp_false_positives.txt")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    write_text_report(results, out_txt)

    results["output_files"] = {
        "json": os.path.relpath(out_json, ROOT_DIR).replace("\\", "/"),
        "txt": os.path.relpath(out_txt, ROOT_DIR).replace("\\", "/"),
    }
    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="EXP23 CADP false positive characterisation")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: first 8 pairs, no threshold sweep")
    args = ap.parse_args()

    if args.smoke:
        res = run(sweep_enabled=False, pair_limit=8)
    else:
        res = run(sweep_enabled=True, pair_limit=None)

    cm = res["primary_condition"]["confusion"]
    print("EXP23 complete.")
    print(f"  encoder_backend: {res['run_metadata']['encoder_backend']}")
    print(f"  pairs run: {res['run_metadata']['n_pairs_run']}")
    print(f"  TP={cm['true_positives']} FP={cm['false_positives']} "
          f"FN={cm['false_negatives']} TN={cm['true_negatives']}")
    print(f"  precision={cm['precision']} recall={cm['recall']} f1={cm['f1']}")
    print(f"  false_positive_rate={cm['false_positive_rate']} "
          f"over {cm['false_positive_rate_denominator']} negative pairs")
    print(f"  ambiguous excluded: {cm['ambiguous_pairs_excluded']}")
    for line in res["verdict_lines"]:
        print(f"  verdict: {line}")
    print(f"  json: {res['output_files']['json']}")
    print(f"  txt:  {res['output_files']['txt']}")


if __name__ == "__main__":
    main()
