"""
EXP24: Paraphrase Robustness Diagnosis and Per-Channel Attribution
==================================================================

QUESTION
When a query is reworded instead of quoted verbatim, how much retrieval
quality does NCM lose, and WHICH channel of the composite distance is
responsible?

NCM ranks by d_raw = alpha*d_sem + beta*d_emo + gamma*d_state + delta*d_time
with shipped defaults alpha=0.4, beta=0.2, gamma=0.3, delta=0.1. This script
does not change those defaults. It decomposes the shipped ranking function.

DESIGN
Same-session episodic retrieval, the exp17 protocol. Relevance is the corpus
`session_id` field, which exists in the data and was not authored here. One
turn per eligible session is held out of the store and used as the query.
Two query conditions run against the IDENTICAL store, label and target, so
the only thing that varies is the surface form of the query:
  verbatim    the held-out turn's exact corpus text
  paraphrase  a HAND-AUTHORED rewording of that same turn (see DISCLOSURE)
Each condition is crossed with two retrieval systems:
  semantic_only  shipped retrieve_semantic_only, isolates the encoder
  ncm            shipped retrieve_top_k_fast, full composite distance
If both systems lose the same amount going verbatim -> paraphrase, the loss
is an encoder property. If ncm loses more, the composite adds fragility.

DISCLOSURE
The 48 paraphrases in PARAPHRASE_TABLE below are hand-authored by the
experimenter. Every number computed from the paraphrase arm is therefore a
number over hand-authored query text, and is labelled as such in the JSON
and the report. The relevance label stays corpus-derived on both arms.

NO ORACLE LEAK
No query-side input is derived from the relevance label. The query auto-state
is inferred from the query text alone by a fresh AutoStateTracker. There is no
oracle arm in this script.

API TRAP HANDLED
retrieve_top_k_fast accepts `s_current_normalized` and never reads it
(ncm/retrieval.py:346 signature, line 372 reads
store.auto_state.get_current_state() instead). The state channel is therefore
controlled by assigning store.auto_state.state directly, and probe_state_
controllability() proves both halves of that trap empirically before any
conclusion about the state channel is drawn.

Outputs
- experiments/results/exp24/exp24_paraphrase_diagnosis.json
- experiments/results/exp24/exp24_paraphrase_diagnosis.txt
- experiments/results/exp24/exp24_paraphrase_regression.png
- experiments/results/exp24/exp24_channel_variance.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ncm import AutoStateTracker, MemoryEntry, MemoryStore, SentenceEncoder
from ncm.retrieval import (
    EMO_NORM,
    STATE_NORM,
    retrieve_semantic_only,
    retrieve_top_k_fast,
)

RESULT_BUCKET = os.path.splitext(os.path.basename(__file__))[0].split("_")[0]
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results", RESULT_BUCKET)
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(ROOT_DIR, "experiments", "data", "real_world_corpus", "train.jsonl")

SEED = 20260820
MAX_CONVERSATIONS = 20
MIN_SESSIONS_PER_CONVERSATION = 2
MIN_SESSION_TURNS = 4
MIN_RELEVANT_PER_QUERY = 3
K_LIST = (5, 10)
# 200000, not 2000. At B=2000 the Monte Carlo standard error of a percentile
# bound on these discrete metrics is about 0.0036, larger than the distance
# from the exact tail mass to the 0.025 threshold, so the sign of a bound
# flipped with the seed. See BOOTSTRAP_NOTE.
BOOTSTRAP_RESAMPLES = 200000
BOOTSTRAP_BLOCK = 25000
BOOTSTRAP_CI = 95.0
BOOTSTRAP_NOTE = (
    "200000 resamples. A 400-seed sweep at 2000 resamples put the upper bound "
    "of the ncm p@5 paraphrase delta strictly below zero for 41.2 percent of "
    "seeds and exactly at zero for the rest, because these metrics are "
    "discrete and the bootstrap distribution of the mean has atoms. No "
    "conclusion here is gated on a strict inequality against such an atom; a "
    "bound within CI_ZERO_TOL of zero is reported as sitting on zero."
)
CI_ZERO_TOL = 1e-9
# Minimum detectable effect, two-sided normal approximation for a paired mean.
MDE_ALPHA = 0.05
MDE_POWER = 0.80
# Monte Carlo chance level, per metric, per query, using that query's own
# n_relevant and store_size.
RANDOM_BASELINE_PERMUTATIONS = 20000
WILCOXON_EXACT_MAX_N = 50
# Gate, not decoration. The run aborts above this.
DECOMPOSITION_TOL = 1e-5

CONDITIONS = ("verbatim", "paraphrase")
SYSTEMS = ("semantic_only", "ncm")
CHANNELS = ("sem", "emo", "state", "time")

# Phase-1 discovery record. The audit note quoted a "paraphrase regression" of
# 0.605 vs 0.876 on a category measure and 0.584 vs 0.767 on a state measure
# without naming a source. All four values were located. They are NOT a
# verbatim-vs-paraphrase contrast.
DISCOVERY_FINDINGS = {
    "status": "FOUND",
    "source_file": "experiments/results/exp6/exp6_current_memory_systems_vs_ncm.json",
    "producing_script": "experiments/python/exp6_current_memory_systems_vs_ncm.py",
    "fields": {
        "0.876": "standing[] entry system='semantic_emotional', field 'category_avg' = 0.8764",
        "0.767": "standing[] entry system='semantic_emotional', field 'state_avg' = 0.7672",
        "0.605": "standing[] entry system='ncm_cached_full', field 'category_avg' = 0.6050",
        "0.584": "standing[] entry system='ncm_cached_full', field 'state_avg' = 0.5835",
    },
    "also_in": (
        "experiments/EXPERIMENT_RESULTS.md lines 319-320 (Experiment 6 section). "
        "Line 319 carries the semantic_emotional pair and line 320 the "
        "ncm_cached_full pair. Lines 257-258 do NOT hold these values; that "
        "location is a speed claim about retrieve_semantic_only."
    ),
    "what_the_two_conditions_actually_are": (
        "Two retrieval SYSTEMS, not two query conditions. 'semantic_emotional' is "
        "the ncm.retrieval.retrieve_semantic_emotional baseline and "
        "'ncm_cached_full' is retrieve_top_k_fast with use_strength=True. Both "
        "run on the SAME 144 queries."
    ),
    "misattribution": (
        "exp6 has NO verbatim query arm. Every one of its 144 queries is a "
        "paraphrase: exp6_current_memory_systems_vs_ncm.py line 120 comments "
        "'Query prompts are category-level paraphrases, not exact memory "
        "strings', the JSON records dataset.query_type = 'category paraphrases "
        "(not exact memory strings)', and the txt header reads '144 paraphrase "
        "queries'. The audit note appears to have read that dataset descriptor "
        "as an experimental condition and reported a system-vs-system ablation "
        "as a paraphrase regression. No paraphrase regression was measured "
        "anywhere in exp6."
    ),
    "how_the_averages_are_formed": (
        "category_avg and state_avg are the unweighted mean of P@1, P@3, P@5 "
        "and P@10. Those four components are NOT in the standing[] record, "
        "which holds only system, category_avg, state_avg and latency_ms_avg. "
        "They are in the systems[] array of the same JSON file, in the entry "
        "with the matching 'system' field, under the dicts 'category_p_at_k' "
        "and 'state_p_at_k' keyed '1', '3', '5' and '10'. Verified: "
        "systems[] semantic_emotional category_p_at_k "
        "(0.9375+0.9167+0.8611+0.7903)/4 = 0.8764 and systems[] "
        "ncm_cached_full state_p_at_k (0.625+0.6111+0.5903+0.5076)/4 = 0.5835."
    ),
    "ruled_out": (
        "exp19, exp20 and exp21_multihop_reasoning are NOT the source. None of "
        "them computes a category_p_at_k or state_p_at_k metric, none writes a "
        "JSON result file, and none has a directory under experiments/results/. "
        "Attribution to exp6 is positive rather than by elimination: the exp6 "
        "JSON contains all four values as named fields and its per-k components "
        "reproduce them arithmetically."
    ),
    "separate_oracle_leak_found_in_exp6": (
        "NEEDS ACTION, reported for the orchestrator and not relied on here. In "
        "exp6 the query's state vector is STATE_ARCHETYPES[q['state_name']] "
        "(lines 265-266) while state_p_at_k's ground truth is defined as "
        "memories whose state tag equals that same q['state_name'] (line 273), "
        "and each memory's stored state is that archetype plus uniform noise of "
        "+/-0.05 (line 165). The query-side state input is therefore derived "
        "from the relevance label, which is an oracle leak by the definition "
        "used in this audit. Both the 0.7672 and the 0.5835 state_avg figures "
        "consume it, so neither is a deployable-condition number."
    ),
    "unrelated_number_checked": (
        "The exp4 latency triplet quoted elsewhere in the audit (49.346, 10.131, "
        "7.860 ms) is absent from "
        "experiments/results/run_all_experiments/exp4_speed_benchmarks.json. That "
        "file's 100000-memory row records semantic shipped median 95.9612 ms, "
        "semantic prebuilt median 2.6312 ms and manifold warm-cache median "
        "12.9269 ms. [NEEDS SOURCE: the 49.346/10.131/7.860 triplet]"
    ),
}

# HAND-AUTHORED PARAPHRASES. Written by the experimenter, not drawn from any
# corpus field. Keyed by (conversation id, session_id). `source_prefix` is the
# first 40 characters of the corpus turn the paraphrase was written against;
# load time asserts it still matches so a corpus change cannot silently
# repoint a paraphrase at a different turn. ASCII only.
PARAPHRASE_TABLE = [
    (0, 0, "What is your favorite meat to eat?",
     "Which kind of meat do you enjoy most?"),
    (0, 1, "Unfortunately I never had that in my l",
     "Sadly I have never tried that before, though I really want to. "
     "And you, is it something you enjoy?"),
    (0, 2, "I was told 4-6 weeks to recover, but th",
     "They said healing will take about a month and a half, which is fine, "
     "since I can sit still and do some wood carving. I will shape a few "
     "little statues."),
    (0, 3, "Basically my ankle is an smashed cookie",
     "My ankle is pretty much crushed at the moment, so yes, the situation "
     "is quite bad."),
    (1, 0, "That's cool my mom does the same thing",
     "Neat, my mother does that exact thing as well."),
    (1, 1, "How could you like the white walkers?! ",
     "Why would anyone be a fan of the white walkers? They exist purely to "
     "wipe out the rest of the cast and they are given no real development."),
    (1, 2, "That's too bad. Maybe she can help you ",
     "What a shame. Perhaps she could assist you in building a site about "
     "mermaid research, and I could add my technical skills to the effort."),
    (2, 0, "I did too. I do not get along with mine",
     "Same here. I get along badly with mine, they have no manners at all."),
    (2, 1, "I'll see if I can try one the next time",
     "Next time I go out I will attempt to sample one. Will you be at the "
     "studio working during this week?"),
    (2, 2, "Since I'm a bit shy, don't think I want",
     "Being fairly timid, a big corporation does not appeal to me. Perhaps "
     "I ought to remain at home and author a book instead. Your work must "
     "be thrilling."),
    (2, 3, "What is your conflict of interest?",
     "Which competing interest do you have?"),
    (5, 0, "Ok I see, that's your halloween costume",
     "Right, understood, so that is the outfit you wear for Halloween."),
    (5, 1, "When I say horror, I like mysteries wit",
     "By horror I mean mysteries carrying a shadowy, atmospheric tone. I "
     "enjoyed the Jonathan Creek novels, which also became a television "
     "programme. It followed a designer of magic illusions who typically "
     "ended up solving a killing. My favourite installments were the "
     "Halloween ones."),
    (5, 2, "Steven Kings is a great author too! I lo",
     "Stephen King is a wonderful writer as well. Halloween is fantastic "
     "and I would call it my favourite celebration too. Have you arranged "
     "anything for it this year?"),
    (5, 3, "I have been to a few, but I usually get",
     "I have attended a handful, though someone usually outbids me. Still, "
     "with a skilled auctioneer they are thrilling."),
    (7, 0, "Do you own your own company",
     "Is the business you run yours?"),
    (7, 1, "That sounds like a great gig. I am stud",
     "That seems like a wonderful job. My studies are in finance although "
     "writing appeals to me as well. If I cut back on going out I might get "
     "more done."),
    (7, 2, "Speaking of pizza, I'm craving some for",
     "On the subject of pizza, I really want some this evening. Which "
     "variety do you like best?"),
    (7, 3, "No!  I am dying to try it.  I have been",
     "Not at all, I am desperate to sample it. I have spent the whole day "
     "writing at my desk. Does it truly live up to its reputation?"),
    (8, 0, "Yes it does pay the bills",
     "Indeed, it covers my expenses."),
    (8, 1, "Ahh, I didn't realize I was chatting wi",
     "Wow, I had no clue I was talking to Steph Curry. Give me twenty "
     "attempts at a three pointer and I would miss them all, hitting the "
     "rim would be lucky."),
    (8, 2, "Oh do you know of a league that armatur",
     "Are you aware of any overseas competition open to amateur players? "
     "That is new to me."),
    (9, 0, "I agree. Have you seen goodfellas?",
     "I feel the same. Did you ever watch Goodfellas?"),
    (9, 1, "Nothing overseas, just a few family get",
     "No foreign trips, only some family holidays over the summer break. "
     "Have you picked any places to visit?"),
    (9, 2, "One is 21 and the other is 22. They are",
     "Their ages are twenty one and twenty two, separated by a single "
     "school year, and they are never apart. Have yours started looking at "
     "universities?"),
    (9, 3, "I wish I could say that I was but Im fa",
     "I would like to claim I am, but I am nowhere near organised. Which "
     "spots are you heading to on your summer trips? I want to visit "
     "somewhere nice."),
    (10, 0, "I'm a nurse who teaches nutrition class",
     "My work is nursing and I also run classes on nutrition."),
    (10, 1, "That's great to hear.  What type of mus",
     "Wonderful news. Which genre do you sing along with? Singing is "
     "something I enjoy too."),
    (10, 2, "Did she? I had no idea! Are you vegan a",
     "She did? That is news to me. Besides being vegetarian, are you also "
     "vegan?"),
    (11, 0, "Do you see a lot of animals around your",
     "Are there many creatures visiting your garden? Watching animals is "
     "something I really like."),
    (11, 1, "Around 11 am, I wanted to see some anim",
     "It was about eleven in the morning and I felt like watching some "
     "animals."),
    (11, 2, "He's friendly, I just leave him at home",
     "He is a gentle dog, but I keep him home when I head to the park at "
     "weekends since I would rather he not bother other people. He is a "
     "small dingo. Stop by sometime, I have been preparing the garden and "
     "it might inspire you to plant your own."),
    (11, 3, "Dalmations are cute. Do you like Siberi",
     "Dalmatians are adorable. Are Siberian Huskies a breed you enjoy?"),
    (12, 0, "I love doing anything outdoors. Especia",
     "Any activity in the open air appeals to me, summer most of all. What "
     "about you?"),
    (12, 1, "I also chose some place close to home. ",
     "I picked somewhere nearby as well. Do you follow the Michigan "
     "university basketball squad? They tend to be quite strong."),
    (12, 2, "Yeah there's a lot of great hiking spot",
     "There are plenty of good trails in this area and I aim to walk one "
     "weekly. Joining my college hiking group has been a fine way to meet "
     "people who share the interest. Do you camp at all?"),
    (13, 0, "Work is tiring. I would love to travel ",
     "My job wears me out. Journeying around the globe would suit me far "
     "better."),
    (13, 1, "I love to travel so I take a big vacati",
     "Travelling is a passion of mine, so once the Christmas busy period "
     "ends I take a long holiday."),
    (13, 2, "Yeah that's true! I can't believe how m",
     "That is quite right. The number of hours you put in astonishes me. "
     "Have you thought about finding different employment?"),
    (14, 0, "Oh nice! My mom went to beauty school t",
     "How lovely, my mother also attended cosmetology school."),
    (14, 1, "wow nice! I know from my husband that b",
     "That is impressive. Through my husband I have learned management "
     "carries a lot of pressure. His week runs past sixty hours."),
    (14, 2, "Jealous! It's been a long time. Is that",
     "I envy you. It has been ages. Does Seattle usually receive snow like "
     "that?"),
    (15, 0, "That's so cool. How do you like to spen",
     "How interesting. What do you enjoy doing with your spare hours?"),
    (15, 1, "Silence of the Lambs is great; I admit ",
     "I grant that Silence of the Lambs is excellent. Even so, the film I "
     "love most, and my top comedy, is Dr. Strangelove, which is wildly "
     "funny."),
    (15, 2, "I think it's on HBO or HBO Max so if yo",
     "I believe it streams on HBO or HBO Max, so it costs nothing if you "
     "subscribe. The story follows two ordinary British men trying to stay "
     "alive once a zombie outbreak begins."),
    (15, 3, "I see. Did you go snowboarding at all d",
     "Understood. Did you get out on a snowboard during the winter months?"),
    (16, 0, "Not like I want being a nurse on the mi",
     "It is not what I would prefer, working as a nurse at the military "
     "base."),
    (16, 1, "That does sound good!  I love bananas. ",
     "That really does sound appealing. Bananas are a favourite of mine. Do "
     "you ever make banana bread? It is one of the things I like best."),
]

PARAPHRASES = {
    (conv_id, session_id): {"source_prefix": prefix, "paraphrase": para}
    for conv_id, session_id, prefix, para in PARAPHRASE_TABLE
}


@dataclass
class Turn:
    text: str
    session_id: int


@dataclass
class Conversation:
    conv_id: int
    turns: list
    # session_id -> index into self.turns of the held-out query turn
    held_out: dict


def load_conversations(corpus_path, max_conversations):
    """
    Load conversations keeping every turn's session_id.

    The query turn for a session is the turn at index len(session)//2, chosen
    positionally so the query set is fully deterministic and does not depend
    on any RNG. That matters because the paraphrases were authored against
    these specific turns.
    """
    conversations = []
    integrity_failures = []

    try:
        handle = open(corpus_path, "r", encoding="utf-8")
    except FileNotFoundError:
        print("[exp24] ERROR: corpus not found at %s" % corpus_path)
        return [], integrity_failures

    with handle as f:
        for line in f:
            if len(conversations) >= max_conversations:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            sessions = data.get("sessions", [])
            if len(sessions) < MIN_SESSIONS_PER_CONVERSATION:
                continue

            conv_id = int(data.get("id", len(conversations)))
            turns = []
            held_out = {}

            for session_index, session in enumerate(sessions):
                session_id = int(session.get("session_id", session_index))
                dialogue = [
                    (t.get("text") or "").strip()
                    for t in session.get("dialogue", [])
                ]
                dialogue = [t for t in dialogue if t]
                if len(dialogue) < MIN_SESSION_TURNS:
                    for text in dialogue:
                        turns.append(Turn(text=text, session_id=session_id))
                    continue

                query_local_index = len(dialogue) // 2
                base = len(turns)
                for text in dialogue:
                    turns.append(Turn(text=text, session_id=session_id))

                key = (conv_id, session_id)
                if key not in PARAPHRASES:
                    continue
                source_text = dialogue[query_local_index]
                prefix = PARAPHRASES[key]["source_prefix"]
                if not source_text.startswith(prefix):
                    integrity_failures.append({
                        "key": "conv%d_session%d" % key,
                        "expected_prefix": prefix,
                        "found_prefix": source_text[:len(prefix)],
                    })
                    continue
                held_out[session_id] = base + query_local_index

            if turns and held_out:
                conversations.append(Conversation(
                    conv_id=conv_id, turns=turns, held_out=held_out,
                ))

    return conversations, integrity_failures


def precision_at_k(labels, k):
    top = labels[:k]
    if not top:
        return 0.0
    return float(sum(top)) / float(len(top))


def recall_at_k(labels, k, n_relevant):
    if n_relevant <= 0:
        return 0.0
    return float(sum(labels[:k])) / float(n_relevant)


def reciprocal_rank(labels):
    for rank, hit in enumerate(labels, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, k, n_relevant):
    gains = [1.0 if hit else 0.0 for hit in labels[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def decompose_channels(store, q_sem, q_emo, q_state_raw, current_step):
    """
    Recompute the four channels of the shipped composite distance over ALL
    candidates, exactly as ncm.retrieval.vectorized_manifold_distance does.

    Returns a dict with the raw per-candidate channel arrays, the weighted
    arrays, and the reconstructed total. The caller compares the reconstructed
    total against the shipped function's own output; if they agree, the
    decomposition IS the shipped ranking function and any statement about
    channel contribution is a statement about the real system.

    q_state_raw is the 5-dim query auto-state BEFORE L2 normalization.
    retrieve_top_k_fast normalizes it internally, so this repeats that step.
    """
    store._rebuild_cache()
    weights = store.profile.retrieval_weights
    alpha, beta, gamma, delta = weights.as_tuple()
    decay_rate = store.profile.decay_rate

    norm = float(np.linalg.norm(q_state_raw))
    if norm > 1e-8:
        s_current = (q_state_raw / norm).astype(np.float32)
    else:
        s_current = np.asarray(q_state_raw, dtype=np.float32)

    d_sem = np.clip(1.0 - (store._sem_cache @ q_sem), 0.0, 1.0)
    d_emo = np.clip(
        np.linalg.norm(store._emo_cache - q_emo[np.newaxis, :], axis=1) / EMO_NORM,
        0.0, 1.0,
    )
    d_state = np.clip(
        np.linalg.norm(
            store._auto_state_cache - s_current[np.newaxis, :], axis=1
        ) / STATE_NORM,
        0.0, 1.0,
    )
    delta_t = np.maximum(0, current_step - store._ts_cache).astype(np.float32)
    d_time = np.clip(1.0 - np.exp(-decay_rate * delta_t), 0.0, 1.0)

    raw = {"sem": d_sem, "emo": d_emo, "state": d_state, "time": d_time}
    weighted = {
        "sem": alpha * d_sem,
        "emo": beta * d_emo,
        "state": gamma * d_state,
        "time": delta * d_time,
    }
    total = sum(weighted.values())
    return {
        "raw": raw,
        "weighted": weighted,
        "total": np.asarray(total, dtype=np.float32),
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "s_current_normalized": s_current,
    }


def channel_stats(decomp):
    """
    Per-channel dispersion across candidates for ONE query.

    Ranking depends only on how much a channel VARIES across candidates. A
    channel with zero variance shifts every candidate's distance by the same
    amount and cannot reorder anything, whatever its weight. std and range of
    the WEIGHTED channel are therefore the quantities that matter, not the mean.
    """
    out = {}
    stds = {c: float(np.std(decomp["weighted"][c])) for c in CHANNELS}
    std_sum = sum(stds.values())
    total = decomp["total"]
    for c in CHANNELS:
        w = decomp["weighted"][c]
        r = decomp["raw"][c]
        if float(np.std(w)) < 1e-12 or float(np.std(total)) < 1e-12:
            spearman = 0.0
        else:
            spearman = float(stats.spearmanr(w, total).statistic)
        out[c] = {
            "raw_mean": float(np.mean(r)),
            "weighted_mean": float(np.mean(w)),
            "weighted_std": stds[c],
            "weighted_range": float(np.max(w) - np.min(w)),
            "std_share": (stds[c] / std_sum) if std_sum > 1e-12 else 0.0,
            "spearman_with_total": spearman,
        }
    return out


def build_store(conv, encoder):
    """
    One store per conversation holding every turn except the held-out queries.

    Mirrors exp17: add() with update_auto_state=True writes each memory's
    auto_state_snapshot from the store auto-state AFTER ingesting that
    memory's text, and _rebuild_cache reads those snapshots.

    Returns the list of PRE-update states as well. e_emotional is built from the
    pre-update state on the line below, while MemoryStore.add overwrites
    auto_state_snapshot with the post-update state, so the emotional projection
    and the state snapshot of one memory sit exactly one turn apart. That offset
    is measured, not assumed; see measure_emo_state_offset.
    """
    store = MemoryStore()
    session_of = {}
    pre_states = []
    held_positions = set(conv.held_out.values())

    for position, turn in enumerate(conv.turns):
        if position in held_positions:
            continue
        state_before = store.auto_state.get_current_state()
        pre_states.append(np.asarray(state_before, dtype=np.float32).copy())
        mem = MemoryEntry(
            e_semantic=encoder.encode(turn.text),
            e_emotional=encoder.encode_emotional(state_before),
            s_snapshot=encoder.encode_state(state_before),
            timestamp=int(store.step),
            text=turn.text,
        )
        stored = store.add(mem, update_auto_state=True)
        session_of[stored.id] = turn.session_id
        store.step += 1

    # Warm the vectorized cache before any timing. add() marks it dirty and
    # retrieve_top_k_fast would otherwise charge the first query of every
    # conversation for the whole rebuild.
    store._rebuild_cache()
    return store, session_of, pre_states


def measure_emo_state_offset(pre_states, post_snapshots):
    """
    C5: e_emotional and the state snapshot of the same memory are built from
    DIFFERENT state vectors, one turn apart, not from the same vector.

    Reports how often they coincide, the mean L2 gap between them, and the max
    L2 between pre_states[i] and post_snapshots[i-1], which is the shift-by-one
    identity that names the offset exactly.
    """
    out = {"n_memories_compared": 0}
    pairs = []
    shifted = []
    n_identical = 0
    for pre, post in zip(pre_states, post_snapshots):
        pre = np.asarray(pre, dtype=np.float64)
        post = np.asarray(post, dtype=np.float64)[:pre.shape[0]]
        d = float(np.linalg.norm(pre - post))
        pairs.append(d)
        if d == 0.0:
            n_identical += 1
    for i in range(1, min(len(pre_states), len(post_snapshots))):
        pre = np.asarray(pre_states[i], dtype=np.float64)
        prev = np.asarray(post_snapshots[i - 1], dtype=np.float64)[:pre.shape[0]]
        shifted.append(float(np.linalg.norm(pre - prev)))
    if pairs:
        out = {
            "n_memories_compared": len(pairs),
            "n_identical": int(n_identical),
            "mean_l2_same_index": float(np.mean(pairs)),
            "max_l2_same_index": float(np.max(pairs)),
            "max_l2_pre_i_vs_post_i_minus_1": float(np.max(shifted)) if shifted else 0.0,
            "note": (
                "pre_states[i] is the state passed to encode_emotional for "
                "memory i; post_snapshots[i] is memory i's auto_state_snapshot, "
                "written by MemoryStore.add after ingesting memory i's text. A "
                "zero in the shift-by-one field confirms the vectors are the "
                "same sequence offset by exactly one turn."
            ),
        }
    return out


def evaluate_queries(conversations, encoder, verbose=False):
    """
    For every held-out turn, run both query conditions against the identical
    store, label and target. Returns one record per query per condition.
    """
    records = []
    decomposition_errors = []
    strength_non_unit = 0
    store_sizes = []
    relevant_fractions = []
    all_snapshots = []
    snapshot_scope = []
    emo_offsets = []

    for conv in conversations:
        store, session_of, pre_states = build_store(conv, encoder)
        if len(store) == 0:
            continue
        store_sizes.append(len(store))

        if not np.allclose(store._str_cache, 1.0):
            strength_non_unit += 1

        # S6: every EVALUATED store contributes its stored auto-states, so the
        # stored-state dispersion figure is not scoped to one conversation.
        snaps = [
            m.auto_state_snapshot for m in store.get_all_safe()
            if m.auto_state_snapshot is not None
        ]
        if snaps:
            all_snapshots.append(np.stack(snaps))
            snapshot_scope.append({"conv_id": conv.conv_id, "n_snapshots": len(snaps)})
            emo_offsets.append(measure_emo_state_offset(pre_states, snaps))

        saved_state = store.auto_state.get_current_state()
        saved_turn = store.auto_state.turn

        for session_id, position in sorted(conv.held_out.items()):
            source_text = conv.turns[position].text
            paraphrase_text = PARAPHRASES[(conv.conv_id, session_id)]["paraphrase"]
            n_relevant = sum(
                1 for mid, sid in session_of.items() if sid == session_id
            )
            if n_relevant < MIN_RELEVANT_PER_QUERY:
                continue
            relevant_fractions.append(n_relevant / float(len(store)))

            for condition, query_text in (
                ("verbatim", source_text),
                ("paraphrase", paraphrase_text),
            ):
                rec = run_one_query(
                    store, session_of, encoder, conv.conv_id, session_id,
                    condition, query_text, n_relevant,
                )
                records.append(rec)
                decomposition_errors.append(rec.pop("_decomp_error"))

            # Restore the tracker so the next query starts from the same place.
            store.auto_state.state = saved_state.copy()
            store.auto_state.turn = saved_turn

        if verbose:
            print("[exp24] conv %d: %d memories, %d queries"
                  % (conv.conv_id, len(store), len(conv.held_out)))

    return {
        "records": records,
        "decomposition_max_abs_error": (
            float(np.max(decomposition_errors)) if decomposition_errors else 0.0
        ),
        "conversations_with_non_unit_strength": strength_non_unit,
        "store_sizes": store_sizes,
        "relevant_fractions": relevant_fractions,
        "store_snapshots": all_snapshots,
        "snapshot_scope": snapshot_scope,
        "emo_state_offset": _merge_emo_offsets(emo_offsets),
    }


def _merge_emo_offsets(per_conv):
    """Pool the per-conversation e_emotional / snapshot offset measurements."""
    kept = [o for o in per_conv if o.get("n_memories_compared")]
    if not kept:
        return {"n_memories_compared": 0}
    n = sum(o["n_memories_compared"] for o in kept)
    weighted = sum(o["mean_l2_same_index"] * o["n_memories_compared"] for o in kept)
    return {
        "n_memories_compared": int(n),
        "n_conversations": len(kept),
        "n_identical": int(sum(o["n_identical"] for o in kept)),
        "mean_l2_same_index": float(weighted / float(n)),
        "max_l2_same_index": float(max(o["max_l2_same_index"] for o in kept)),
        "max_l2_pre_i_vs_post_i_minus_1": float(
            max(o["max_l2_pre_i_vs_post_i_minus_1"] for o in kept)
        ),
        "note": kept[0]["note"],
    }


def run_one_query(store, session_of, encoder, conv_id, session_id,
                  condition, query_text, n_relevant):
    """
    Run one query text through both systems and decompose the composite.

    The query auto-state is inferred from the query text ALONE by a fresh
    AutoStateTracker. Nothing about the target session reaches the query side,
    so there is no oracle leak.

    Because retrieve_top_k_fast ignores its s_current_normalized argument and
    reads store.auto_state.get_current_state() instead, the inferred state is
    installed by assigning store.auto_state.state directly. The argument is
    still passed for signature compatibility but carries no effect; see
    probe_state_controllability for the empirical proof of both halves.
    """
    q_sem = encoder.encode(query_text)

    probe = AutoStateTracker()
    inferred_state = probe.update(query_text).astype(np.float32)
    q_emo = encoder.encode_emotional(inferred_state)
    q_state_encoded = encoder.encode_state(inferred_state)

    # --- semantic_only ------------------------------------------------
    t0 = perf_counter()
    hits = retrieve_semantic_only(q_sem, store, k=max(K_LIST))
    sem_latency = (perf_counter() - t0) * 1000.0
    sem_labels = [
        session_of.get(h[-1].id, -1) == session_id for h in hits
    ]

    # --- ncm ----------------------------------------------------------
    store.auto_state.state = inferred_state.copy()
    t0 = perf_counter()
    hits = retrieve_top_k_fast(
        q_sem, q_emo, store, q_state_encoded, int(store.step), k=max(K_LIST),
    )
    ncm_latency = (perf_counter() - t0) * 1000.0
    ncm_labels = [
        session_of.get(h[-1].id, -1) == session_id for h in hits
    ]
    shipped_topk_distances = np.array([h[0] for h in hits], dtype=np.float32)

    # --- channel decomposition of the same ranking decision -----------
    decomp = decompose_channels(
        store, q_sem, q_emo, inferred_state, int(store.step)
    )
    order = np.argsort(decomp["total"])[:max(K_LIST)]
    reconstructed = decomp["total"][order]
    decomp_error = float(np.max(np.abs(reconstructed - shipped_topk_distances)))

    rec = {
        "conv_id": conv_id,
        "session_id": session_id,
        "condition": condition,
        "query_chars": len(query_text),
        "n_relevant": n_relevant,
        "store_size": len(store),
        "inferred_state": inferred_state.tolist(),
        "metrics": {},
        "channels": channel_stats(decomp),
        "emo_state_raw_pearson": _safe_pearson(
            decomp["raw"]["emo"], decomp["raw"]["state"]
        ),
        "_decomp_error": decomp_error,
    }

    for system, labels, latency in (
        ("semantic_only", sem_labels, sem_latency),
        ("ncm", ncm_labels, ncm_latency),
    ):
        m = {}
        for k in K_LIST:
            m["p@%d" % k] = precision_at_k(labels, k)
            m["r@%d" % k] = recall_at_k(labels, k, n_relevant)
        m["ndcg@10"] = ndcg_at_k(labels, 10, n_relevant)
        m["mrr"] = reciprocal_rank(labels)
        m["latency_ms"] = latency
        rec["metrics"][system] = m

    return rec


def _safe_pearson(a, b):
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(stats.pearsonr(a, b).statistic)


def probe_state_controllability(store, encoder, query_text):
    """
    Prove BOTH halves of the retrieve_top_k_fast API trap before drawing any
    conclusion about the state channel.

    PROBE A  vary the s_current_normalized ARGUMENT, hold store.auto_state
             fixed. If the returned ranking is identical, the argument is
             ignored, which is the documented trap.
    PROBE B  vary store.auto_state.state to two extreme opposite states.
             If the ranking or the d_state spread changes, the channel is
             live and controllable, so a later finding of near-zero state
             variance under INFERRED states is a real property of the
             inference, not a failure to drive the input.

    Without probe B a null result on the state channel would be
    indistinguishable from never having changed the input at all.
    """
    q_sem = encoder.encode(query_text)
    probe = AutoStateTracker()
    inferred = probe.update(query_text).astype(np.float32)
    q_emo = encoder.encode_emotional(inferred)

    low = np.full(5, 0.02, dtype=np.float32)
    high = np.array([0.98, 0.02, 0.98, 0.02, 0.98], dtype=np.float32)

    def ids_for(auto_state, arg_state):
        store.auto_state.state = np.asarray(auto_state, dtype=np.float32).copy()
        hits = retrieve_top_k_fast(
            q_sem, q_emo, store, arg_state, int(store.step), k=max(K_LIST),
        )
        return [h[-1].id for h in hits], np.array([h[0] for h in hits])

    saved = store.auto_state.get_current_state()

    # PROBE A: argument varies, auto_state pinned.
    ids_arg_low, dist_arg_low = ids_for(inferred, encoder.encode_state(low))
    ids_arg_high, dist_arg_high = ids_for(inferred, encoder.encode_state(high))
    argument_is_ignored = (
        ids_arg_low == ids_arg_high
        and bool(np.allclose(dist_arg_low, dist_arg_high))
    )

    # PROBE B: auto_state varies.
    ids_state_low, dist_state_low = ids_for(low, encoder.encode_state(inferred))
    ids_state_high, dist_state_high = ids_for(high, encoder.encode_state(inferred))
    state_changes_ranking = ids_state_low != ids_state_high
    state_changes_distances = not bool(
        np.allclose(dist_state_low, dist_state_high)
    )

    # d_state spread that each installed state actually produces.
    spreads = {}
    for name, st in (("inferred", inferred), ("low", low), ("high", high)):
        d = decompose_channels(
            store, q_sem, q_emo, st, int(store.step)
        )
        spreads[name] = {
            "d_state_mean": float(np.mean(d["raw"]["state"])),
            "d_state_std": float(np.std(d["raw"]["state"])),
            "d_state_range": float(
                np.max(d["raw"]["state"]) - np.min(d["raw"]["state"])
            ),
        }

    store.auto_state.state = saved.copy()

    return {
        "probe_a_argument_is_ignored": argument_is_ignored,
        "probe_a_note": (
            "True confirms retrieve_top_k_fast ignores s_current_normalized; "
            "the argument cannot be used to control the state channel."
        ),
        "probe_b_state_assignment_changes_ranking": bool(state_changes_ranking),
        "probe_b_state_assignment_changes_distances": bool(state_changes_distances),
        "probe_b_note": (
            "True confirms store.auto_state.state DOES drive the channel, so a "
            "null state result under inferred states is a property of the "
            "inference and not an uncontrolled input."
        ),
        "d_state_by_installed_state": spreads,
    }


def measure_state_dispersion(records, store_snapshots, snapshot_scope=None):
    """
    Compare how far apart INFERRED QUERY states are from each other against how
    far apart the STORED memory auto-states are.

    Three scopes are kept separate, because pooling them mixes two different
    sources of variation:
      across_targets_verbatim   one state per target, verbatim arm only
      across_targets_paraphrase one state per target, paraphrase arm only
      pooled_both_conditions    every record, so across-target AND
                                within-target verbatim-vs-paraphrase variation
    The within-target verbatim-vs-paraphrase shift is reported on its own below.

    AutoStateTracker.update is a single EMA step from the neutral 0.5 vector,
    state = (1-alpha)*state + alpha*signal, with alpha in
    [0.15, 0.15, 0.15, 0.20, 0.25]. One utterance can move a dimension by at
    most alpha*|signal - 0.5| <= 0.125. Pairwise L2 between query states
    therefore has a hard analytic ceiling, which this reports alongside the
    measured values.
    """
    q_states = np.array([r["inferred_state"] for r in records], dtype=np.float32)
    out = {"n_query_states": int(q_states.shape[0])}

    def _pairwise(mat):
        if mat.shape[0] < 2:
            return None
        diffs = []
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[0]):
                diffs.append(float(np.linalg.norm(mat[i] - mat[j])))
        return {
            "min": float(np.min(diffs)),
            "mean": float(np.mean(diffs)),
            "max": float(np.max(diffs)),
            "n_pairs": len(diffs),
            "n_states": int(mat.shape[0]),
        }

    for condition in CONDITIONS:
        sub = np.array(
            [r["inferred_state"] for r in records if r["condition"] == condition],
            dtype=np.float32,
        )
        res = _pairwise(sub)
        if res:
            out["query_state_pairwise_l2_%s_only" % condition] = res

    if q_states.shape[0] >= 2:
        pooled = _pairwise(q_states)
        pooled["scope"] = (
            "POOLED over both query conditions, so it mixes across-target "
            "variation with the within-target verbatim-vs-paraphrase shift that "
            "verbatim_vs_paraphrase_state_l2 reports separately"
        )
        out["query_state_pairwise_l2"] = pooled
        out["query_state_per_dim_std"] = [
            float(v) for v in np.std(q_states, axis=0)
        ]
        out["query_state_max_abs_deviation_from_neutral"] = float(
            np.max(np.abs(q_states - 0.5))
        )

    # Verbatim vs paraphrase state shift for the SAME target.
    by_key = {}
    for r in records:
        by_key.setdefault((r["conv_id"], r["session_id"]), {})[r["condition"]] = r
    shifts = [
        float(np.linalg.norm(
            np.array(v["verbatim"]["inferred_state"], dtype=np.float32)
            - np.array(v["paraphrase"]["inferred_state"], dtype=np.float32)
        ))
        for v in by_key.values()
        if "verbatim" in v and "paraphrase" in v
    ]
    if shifts:
        out["verbatim_vs_paraphrase_state_l2"] = {
            "min": float(np.min(shifts)),
            "mean": float(np.mean(shifts)),
            "max": float(np.max(shifts)),
            "n": len(shifts),
        }

    if store_snapshots:
        allsnap = np.concatenate(store_snapshots, axis=0)
        out["stored_memory_state_per_dim_std"] = [
            float(v) for v in np.std(allsnap, axis=0)
        ]
        out["stored_memory_state_max_abs_deviation_from_neutral"] = float(
            np.max(np.abs(allsnap - 0.5))
        )
        out["n_stored_snapshots"] = int(allsnap.shape[0])
        out["n_stores_contributing_snapshots"] = len(store_snapshots)
        out["stored_snapshot_scope"] = (
            snapshot_scope if snapshot_scope is not None else
            "[NEEDS SOURCE: per-store snapshot counts were not passed in]"
        )

    out["analytic_single_step_per_dim_ceiling"] = [0.075, 0.075, 0.075, 0.10, 0.125]
    out["analytic_note"] = (
        "alpha*|signal-0.5| with signal bounded in [0,1] and alpha = "
        "[0.15,0.15,0.15,0.20,0.25]; the largest move one utterance can make "
        "in each dimension starting from the neutral 0.5 state."
    )
    return out


def check_state_padding(encoder):
    """
    encode_state and encode_emotional pad a 5-dim auto-state up to
    encoder.state_dim (default 7). Record whether the trailing dimensions are
    structurally zero rather than asserting it from the source.
    """
    probe = np.array([0.9, 0.1, 0.8, 0.2, 0.7], dtype=np.float32)
    enc = encoder.encode_state(probe)
    w_emo = encoder.w_emo
    emo = encoder.encode_emotional(probe)
    return {
        "encoder_state_dim": int(encoder.state_dim),
        "auto_state_dims_emitted": 5,
        "encode_state_output_len": int(enc.shape[0]),
        "w_emo_shape": [int(v) for v in w_emo.shape],
        "w_emo_rank_loss_note": (
            "W_emo has shape %s, so it maps the %d informative state dimensions "
            "into %d and cannot be inverted; encode_emotional then L2-normalizes "
            "the result, which discards its magnitude. Both are reasons the "
            "d_emo / d_state correlation is below 1."
            % (tuple(int(v) for v in w_emo.shape), 5, int(w_emo.shape[0]))
        ),
        "encode_emotional_output_norm": float(np.linalg.norm(emo)),
        "trailing_dims_are_zero": bool(
            enc.shape[0] > 5 and np.allclose(enc[5:], 0.0)
        ),
        "trailing_dim_values": [float(v) for v in enc[5:]],
        "note": (
            "AutoStateTracker emits 5 dimensions while SentenceEncoder.state_dim "
            "defaults to 7, so encode_state and encode_emotional zero-pad and "
            "dimensions 6 and 7 carry no information. Note that the composite "
            "state channel compares auto_state_snapshot directly at 5 dims, so "
            "the padding affects the emotional projection, not d_state."
        ),
    }


def paired_bootstrap(deltas, rng, n_resamples=BOOTSTRAP_RESAMPLES, ci=BOOTSTRAP_CI):
    """
    Percentile bootstrap CI for the mean of a PAIRED difference vector.

    Pairing is by target: the same store, the same label and the same target
    session under two query surface forms. Resampling is over targets.

    CAVEAT, do not read this interval as a generalization interval. The targets
    are NOT independent. Several targets come from the same conversation and
    share a store, an encoder and a topic, so the observations are clustered.
    This target-level interval treats them as independent and is therefore too
    narrow. cluster_bootstrap below resamples whole conversations and is the
    interval to quote; icc_and_effective_n reports the measured intraclass
    correlation and the effective sample size for the same vector.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.shape[0]
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    means = _blocked_resample_means(deltas, n, n_resamples, rng)
    lo = (100.0 - ci) / 2.0
    ci_low = float(np.percentile(means, lo))
    ci_high = float(np.percentile(means, 100.0 - lo))
    return {
        "n": int(n),
        "mean": float(np.mean(deltas)),
        "sd": float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_resamples": int(n_resamples),
        "resample_unit": "target",
        "bound_low_on_zero": bool(abs(ci_low) <= CI_ZERO_TOL),
        "bound_high_on_zero": bool(abs(ci_high) <= CI_ZERO_TOL),
        "n_nonzero_deltas": int(np.count_nonzero(deltas)),
        "mde": minimum_detectable_effect(deltas),
        "mde_definition": (
            "two-sided normal approximation, alpha=%.2f power=%.2f: "
            "(z_{1-alpha/2}+z_{power})*sd/sqrt(n). The smallest true paired "
            "mean difference this design would detect %.0f percent of the time."
            % (MDE_ALPHA, MDE_POWER, 100.0 * MDE_POWER)
        ),
    }


def _blocked_resample_means(values, n, n_resamples, rng, weights=None):
    """
    Bootstrap means in fixed-size blocks so a 200000-resample run never
    allocates a 200000 x n index matrix. Block size is a constant, so the
    stream of draws and therefore the result is deterministic for a given seed.
    """
    out = np.empty(n_resamples, dtype=np.float64)
    filled = 0
    while filled < n_resamples:
        take = int(min(BOOTSTRAP_BLOCK, n_resamples - filled))
        idx = rng.integers(0, n, size=(take, n))
        if weights is None:
            out[filled:filled + take] = values[idx].mean(axis=1)
        else:
            num = values[idx].sum(axis=1)
            den = weights[idx].sum(axis=1)
            out[filled:filled + take] = num / np.maximum(den, 1e-12)
        filled += take
    return out


def minimum_detectable_effect(deltas, alpha=MDE_ALPHA, power=MDE_POWER):
    """Smallest paired mean difference detectable at the given alpha and power."""
    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.shape[0]
    if n < 2:
        return None
    sd = float(np.std(deltas, ddof=1))
    if sd <= 0.0:
        return 0.0
    z_a = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_b = float(stats.norm.ppf(power))
    return float((z_a + z_b) * sd / np.sqrt(n))


def cluster_bootstrap(deltas, cluster_ids, rng,
                      n_resamples=BOOTSTRAP_RESAMPLES, ci=BOOTSTRAP_CI):
    """
    Percentile bootstrap CI that resamples whole CONVERSATIONS with
    replacement, so the interval respects the clustering of targets inside a
    conversation. The point estimate is the unweighted mean over targets, the
    same estimand as paired_bootstrap; only the resampling unit differs.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    if deltas.shape[0] == 0:
        return {"n": 0, "n_clusters": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    uniq = np.unique(ids)
    sums = np.array([deltas[ids == u].sum() for u in uniq], dtype=np.float64)
    counts = np.array([float((ids == u).sum()) for u in uniq], dtype=np.float64)
    means = _blocked_resample_means(sums, uniq.shape[0], n_resamples, rng,
                                   weights=counts)
    lo = (100.0 - ci) / 2.0
    ci_low = float(np.percentile(means, lo))
    ci_high = float(np.percentile(means, 100.0 - lo))
    return {
        "n": int(deltas.shape[0]),
        "n_clusters": int(uniq.shape[0]),
        "cluster_sizes": [int(c) for c in counts],
        "mean": float(np.mean(deltas)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_resamples": int(n_resamples),
        "resample_unit": "conversation",
        "bound_low_on_zero": bool(abs(ci_low) <= CI_ZERO_TOL),
        "bound_high_on_zero": bool(abs(ci_high) <= CI_ZERO_TOL),
    }


def icc_and_effective_n(values, cluster_ids):
    """
    One-way random-effects ICC(1) on a value vector whose observations are
    grouped by conversation, plus the design effect and effective sample size.

    ICC(1) = (MSB - MSW) / (MSB + (m0 - 1) * MSW) with m0 the mean cluster
    size. design_effect = 1 + (m0 - 1) * ICC. effective_n = n / design_effect.
    A positive ICC means targets inside a conversation resemble each other, so
    the target-level interval understates the uncertainty.
    """
    vals = np.asarray(values, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    n = vals.shape[0]
    uniq = np.unique(ids)
    k = uniq.shape[0]
    if n < 2 or k < 2 or k == n:
        return None
    grand = float(np.mean(vals))
    ssb = 0.0
    ssw = 0.0
    for u in uniq:
        g = vals[ids == u]
        ssb += g.shape[0] * (float(np.mean(g)) - grand) ** 2
        ssw += float(np.sum((g - float(np.mean(g))) ** 2))
    msb = ssb / float(k - 1)
    msw = ssw / float(n - k) if n > k else 0.0
    m0 = float(n) / float(k)
    denom = msb + (m0 - 1.0) * msw
    if abs(denom) < 1e-15:
        return None
    icc = float((msb - msw) / denom)
    design_effect = 1.0 + (m0 - 1.0) * icc
    return {
        "icc1": icc,
        "mean_cluster_size": m0,
        "n_clusters": int(k),
        "design_effect": float(design_effect),
        "effective_n": float(n / design_effect) if design_effect > 0 else float(n),
    }


def wilcoxon_or_none(a, b):
    """
    Wilcoxon signed-rank on paired samples; None when all differences tie.

    method="exact" is passed explicitly. On scipy 1.17 method="auto" falls back
    to the normal approximation as soon as zeros or ties are present, which is
    the case for every discrete metric here, and the approximation returns
    p-values up to 1.5x smaller than the exact test. The exact test is
    tractable only for small n, so for n above WILCOXON_EXACT_MAX_N the
    approximation is used and recorded as such in the "method" field.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] < 2 or np.allclose(a, b):
        return None
    n_nonzero = int(np.count_nonzero(b - a))
    method = "exact" if n_nonzero <= WILCOXON_EXACT_MAX_N else "approx"
    try:
        if method == "exact":
            res = stats.wilcoxon(a, b, method="exact")
        else:
            res = stats.wilcoxon(a, b, method="approx")
    except ValueError:
        return None
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "method": method,
        "n_nonzero_differences": n_nonzero,
        "method_note": (
            "scipy.stats.wilcoxon with method set explicitly; 'auto' would "
            "have used the normal approximation whenever zeros or ties occur"
        ),
    }


def _chance_level_for_query(n_relevant, store_size, n_perm, rng):
    """
    Monte Carlo chance level for ONE query, using that query's own n_relevant
    and store_size. n_perm uniformly random rankings of the store are drawn and
    scored with the SAME metric functions the experiment uses, so the baseline
    is comparable term by term.
    """
    m = int(min(max(K_LIST), store_size))
    order = np.argsort(rng.random((n_perm, store_size)), axis=1)[:, :m]
    labels = order < int(n_relevant)
    out = {}
    for k in K_LIST:
        kk = min(k, m)
        out["p@%d" % k] = float(labels[:, :kk].sum(axis=1).mean() / float(k))
    gains = labels[:, :min(10, m)].astype(np.float64)
    discount = 1.0 / np.log2(np.arange(gains.shape[1]) + 2.0)
    dcg = gains @ discount
    ideal = int(min(10, n_relevant))
    idcg = float(np.sum(1.0 / np.log2(np.arange(ideal) + 2.0))) if ideal > 0 else 0.0
    out["ndcg@10"] = float(np.mean(dcg / idcg)) if idcg > 0 else 0.0
    ranks = np.where(labels.any(axis=1), labels.argmax(axis=1) + 1, 0)
    rr = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    out["mrr"] = float(np.mean(rr))
    return out


def per_metric_random_baseline(records, rng, n_perm=RANDOM_BASELINE_PERMUTATIONS):
    """
    Chance level per metric, averaged over the evaluated targets.

    A single n_relevant/store_size ratio is the chance level for precision only.
    MRR and NDCG have different chance levels because they reward the position
    of the FIRST relevant hit, which a uniformly random ranking finds early far
    more often than the precision ratio suggests. Drawing one horizontal line
    for all four metrics understates the MRR baseline by a large factor.
    """
    seen = {}
    per_target = []
    cache = {}
    for r in records:
        key = (r["conv_id"], r["session_id"])
        if key in seen:
            continue
        seen[key] = True
        shape = (int(r["n_relevant"]), int(r["store_size"]))
        if shape not in cache:
            cache[shape] = _chance_level_for_query(shape[0], shape[1], n_perm, rng)
        per_target.append(cache[shape])
    if not per_target:
        return {}
    out = {
        name: float(np.mean([p[name] for p in per_target]))
        for name in ("p@5", "p@10", "ndcg@10", "mrr")
    }
    out["n_permutations_per_query"] = int(n_perm)
    out["n_targets"] = len(per_target)
    out["n_distinct_shapes"] = len(cache)
    out["definition"] = (
        "Monte Carlo over uniformly random rankings of each target's own store, "
        "scored with the same precision_at_k, ndcg_at_k and reciprocal_rank "
        "functions used for the real runs, then averaged over targets."
    )
    return out


BRITISH_TO_AMERICAN = (
    ("favourite", "favorite"), ("colour", "color"), ("flavour", "flavor"),
    ("honour", "honor"), ("humour", "humor"), ("labour", "labor"),
    ("neighbour", "neighbor"), ("behaviour", "behavior"), ("rumour", "rumor"),
    ("programme", "program"), ("organised", "organized"),
    ("organisation", "organization"), ("realised", "realized"),
    ("recognised", "recognized"), ("apologise", "apologize"),
    ("specialised", "specialized"), ("analyse", "analyze"),
    ("practise", "practice"), ("travelled", "traveled"),
    ("travelling", "traveling"), ("cancelled", "canceled"),
    ("theatre", "theater"), ("centre", "center"), ("metre", "meter"),
    ("litre", "liter"), ("defence", "defense"), ("licence", "license"),
    ("grey", "gray"), ("cheque", "check"), ("aeroplane", "airplane"),
    ("whilst", "while"), ("amongst", "among"), ("learnt", "learned"),
    ("spelt", "spelled"), ("mum", "mom"), ("maths", "math"),
    ("aluminium", "aluminum"), ("jewellery", "jewelry"),
    ("storey", "story"), ("kerb", "curb"), ("tyre", "tire"),
)


def _word_set(text):
    """Case-folded alphabetic tokens. Whole-word matching only, so that a
    substring such as 'mum' inside 'minimum' cannot register as a variant."""
    out = set()
    cur = []
    for ch in text.lower():
        if ch.isalpha():
            cur.append(ch)
        elif cur:
            out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return out


def audit_orthography_confound(conversations):
    """
    Second manipulated variable check.

    Rewording is the intended manipulation. If a hand-authored paraphrase also
    swaps American spelling for British spelling, then two variables moved at
    once and the paraphrase arm is not a pure rewording contrast. This counts
    the paraphrases that contain a British-spelling word absent from their
    source turn, and records whether the source used the American form.
    The stimuli are NOT rewritten here; changing them would break comparability
    with the previously shipped run. The count is disclosed instead.
    """
    hits = []
    n_checked = 0
    for conv in conversations:
        for session_id, position in sorted(conv.held_out.items()):
            key = (conv.conv_id, session_id)
            if key not in PARAPHRASES:
                continue
            n_checked += 1
            src = _word_set(conv.turns[position].text)
            par = _word_set(PARAPHRASES[key]["paraphrase"])
            found = []
            for brit, amer in BRITISH_TO_AMERICAN:
                if brit in par and brit not in src:
                    found.append({
                        "variant": brit,
                        "american_form": amer,
                        "source_uses_american_form": bool(amer in src),
                    })
            if found:
                hits.append({
                    "conv_id": conv.conv_id,
                    "session_id": session_id,
                    "variants": found,
                })
    return {
        "n_paraphrases_checked": n_checked,
        "n_paraphrases_with_spelling_variant": len(hits),
        "detail": hits,
        "wordlist_size": len(BRITISH_TO_AMERICAN),
        "method": (
            "fixed hand-built list of %d British/American spelling pairs, "
            "whole-word case-folded match; a paraphrase counts if it uses the "
            "British form and its source turn does not"
            % len(BRITISH_TO_AMERICAN)
        ),
        "limitation": (
            "the wordlist is hand-built and not exhaustive, so this count is a "
            "lower bound on orthographic divergence"
        ),
    }


def aggregate(records, rng, n_resamples=BOOTSTRAP_RESAMPLES):
    """Per-arm means, paired verbatim -> paraphrase deltas, and channel means."""
    by_key = {}
    for r in records:
        by_key.setdefault((r["conv_id"], r["session_id"]), {})[r["condition"]] = r
    paired_keys = sorted(
        k for k, v in by_key.items() if "verbatim" in v and "paraphrase" in v
    )

    metric_names = (
        ["p@%d" % k for k in K_LIST]
        + ["r@%d" % k for k in K_LIST]
        + ["ndcg@10", "mrr", "latency_ms"]
    )

    arms = {}
    for condition in CONDITIONS:
        for system in SYSTEMS:
            vals = {m: [] for m in metric_names}
            for k in paired_keys:
                m = by_key[k][condition]["metrics"][system]
                for name in metric_names:
                    vals[name].append(m[name])
            arm = {}
            for name in metric_names:
                arr = np.asarray(vals[name], dtype=np.float64)
                arm[name] = float(np.mean(arr))
                arm[name + "_sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            arm["n_queries"] = len(paired_keys)
            arms["%s__%s" % (condition, system)] = arm

    # Conversation id per paired target, the clustering unit for B4.
    cluster_ids = np.array([k[0] for k in paired_keys])

    regression = {}
    for system in SYSTEMS:
        for name in ("p@5", "p@10", "ndcg@10", "mrr"):
            v = [by_key[k]["verbatim"]["metrics"][system][name] for k in paired_keys]
            p = [by_key[k]["paraphrase"]["metrics"][system][name] for k in paired_keys]
            deltas = np.asarray(p, dtype=np.float64) - np.asarray(v, dtype=np.float64)
            entry = paired_bootstrap(deltas, rng, n_resamples=n_resamples)
            entry["verbatim_mean"] = float(np.mean(v))
            entry["paraphrase_mean"] = float(np.mean(p))
            entry["wilcoxon"] = wilcoxon_or_none(v, p)
            entry["cluster_bootstrap"] = cluster_bootstrap(
                deltas, cluster_ids, rng, n_resamples=n_resamples
            )
            entry["icc"] = icc_and_effective_n(deltas, cluster_ids)
            entry["wilcoxon_clustering_caveat"] = (
                "the signed-rank test assumes independent pairs; these targets "
                "are clustered by conversation, so the p-value is anticonservative"
            )
            regression["%s__%s" % (system, name)] = entry

    # Does NCM lose MORE than the encoder alone? Difference of differences.
    dod = {}
    for name in ("p@5", "ndcg@10"):
        d_ncm = np.array(
            [by_key[k]["paraphrase"]["metrics"]["ncm"][name]
             - by_key[k]["verbatim"]["metrics"]["ncm"][name] for k in paired_keys]
        )
        d_sem = np.array(
            [by_key[k]["paraphrase"]["metrics"]["semantic_only"][name]
             - by_key[k]["verbatim"]["metrics"]["semantic_only"][name]
             for k in paired_keys]
        )
        entry = paired_bootstrap(d_ncm - d_sem, rng, n_resamples=n_resamples)
        entry["cluster_bootstrap"] = cluster_bootstrap(
            d_ncm - d_sem, cluster_ids, rng, n_resamples=n_resamples
        )
        entry["icc"] = icc_and_effective_n(d_ncm - d_sem, cluster_ids)
        entry["interpretation"] = (
            "paraphrase drop for ncm minus paraphrase drop for semantic_only. A "
            "CI containing zero means no difference was DETECTED, which is not "
            "the same as none existing; read it against the mde field, and note "
            "that a bound sitting exactly on zero carries no effect-size "
            "information at all because it is forced by the number of nonzero "
            "paired differences"
        )
        dod["dod__%s" % name] = entry

    channels = {}
    for condition in CONDITIONS:
        per = {}
        for c in CHANNELS:
            per[c] = {
                field: float(np.mean([
                    by_key[k][condition]["channels"][c][field] for k in paired_keys
                ]))
                for field in (
                    "raw_mean", "weighted_mean", "weighted_std",
                    "weighted_range", "std_share", "spearman_with_total",
                )
            }
        per["emo_state_raw_pearson_mean"] = float(np.mean(
            [by_key[k][condition]["emo_state_raw_pearson"] for k in paired_keys]
        ))
        channels[condition] = per

    # S5: the largest movement in the table is MRR, and it RISES under
    # paraphrase. Whether the arms are length-matched is MEASURED here, not
    # assumed; at the shipped scale they are not. The paired per-target length
    # change is a separate question from the arm-level difference, so both are
    # reported.
    char_delta = np.array(
        [float(by_key[k]["paraphrase"]["query_chars"]
               - by_key[k]["verbatim"]["query_chars"]) for k in paired_keys]
    )
    length_correlation = {}
    for system in SYSTEMS:
        for name in ("p@5", "ndcg@10", "mrr"):
            d = np.array(
                [by_key[k]["paraphrase"]["metrics"][system][name]
                 - by_key[k]["verbatim"]["metrics"][system][name]
                 for k in paired_keys]
            )
            if float(np.std(char_delta)) < 1e-12 or float(np.std(d)) < 1e-12:
                continue
            res = stats.spearmanr(char_delta, d)
            length_correlation["%s__%s" % (system, name)] = {
                "spearman_rho": float(res.statistic),
                "p_value": float(res.pvalue),
                "n": int(char_delta.shape[0]),
            }
    length_match = {
        "n": int(char_delta.shape[0]),
        "verbatim_mean_chars": float(np.mean(
            [by_key[k]["verbatim"]["query_chars"] for k in paired_keys])),
        "paraphrase_mean_chars": float(np.mean(
            [by_key[k]["paraphrase"]["query_chars"] for k in paired_keys])),
        "mean_char_delta": float(np.mean(char_delta)),
        "wilcoxon_on_char_delta": wilcoxon_or_none(
            [by_key[k]["verbatim"]["query_chars"] for k in paired_keys],
            [by_key[k]["paraphrase"]["query_chars"] for k in paired_keys],
        ),
        "paired_length_vs_metric_spearman": length_correlation,
        "note": (
            "Two separate questions. wilcoxon_on_char_delta asks whether the arms "
            "differ in length at all. paired_length_vs_metric_spearman asks "
            "whether a metric's per-target movement tracks that target's own "
            "length change: x is the paired per-target character-count change "
            "from verbatim to hand-authored paraphrase, y is the paired "
            "per-target change in the named metric. A null rho does not clear an "
            "arm-level length difference; the two are different quantities."
        ),
    }

    return {
        "paired_keys": ["conv%d_session%d" % k for k in paired_keys],
        "n_paired_targets": len(paired_keys),
        "n_conversations_in_paired_set": int(np.unique(cluster_ids).shape[0]),
        "targets_per_conversation": [
            int((cluster_ids == u).sum()) for u in np.unique(cluster_ids)
        ],
        "clustering_note": (
            "the %d paired targets come from %d conversations, so the "
            "observations are clustered; every regression entry carries both a "
            "target-level and a conversation-level bootstrap CI plus a measured "
            "ICC(1) and effective n"
            % (len(paired_keys), int(np.unique(cluster_ids).shape[0]))
        ),
        "query_length_match": length_match,
        "arms": arms,
        "paraphrase_regression": regression,
        "difference_of_differences": dod,
        "channels": channels,
        "bootstrap_note": BOOTSTRAP_NOTE,
    }


ARM_COLORS = {
    "verbatim__semantic_only": "#c0392b",
    "verbatim__ncm": "#27ae60",
    "paraphrase__semantic_only": "#e8897f",
    "paraphrase__ncm": "#82d3a4",
}
CHANNEL_COLORS = {
    "sem": "#2980b9",
    "emo": "#8e44ad",
    "state": "#d35400",
    "time": "#7f8c8d",
}


def plot_regression(agg, random_baseline, path):
    """
    random_baseline is a PER-METRIC dict of Monte Carlo chance levels. A single
    horizontal line across all four metric groups was wrong: the chance level
    for MRR is far above the n_relevant/store_size ratio, because a uniformly
    random ranking still places some relevant item early most of the time.
    """
    arms = agg["arms"]
    metrics = ["p@5", "p@10", "ndcg@10", "mrr"]
    order = [
        "verbatim__semantic_only", "paraphrase__semantic_only",
        "verbatim__ncm", "paraphrase__ncm",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(metrics))
    width = 0.2
    for i, arm in enumerate(order):
        offset = (i - (len(order) - 1) / 2.0) * width
        vals = [arms[arm][m] for m in metrics]
        bars = ax1.bar(x + offset, vals, width, label=arm.replace("__", " / "),
                       color=ARM_COLORS[arm], edgecolor="black", linewidth=1.0)
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                     "%.3f" % b.get_height(), ha="center", va="bottom", fontsize=7.5)

    # One chance-level segment per metric group, not one line across all four.
    half = (len(order) / 2.0) * width
    for i, m in enumerate(metrics):
        base = random_baseline.get(m)
        if base is None:
            continue
        ax1.plot([x[i] - half, x[i] + half], [base, base], color="black",
                 linestyle=":", linewidth=1.8,
                 label="Monte Carlo chance level" if i == 0 else None)
        ax1.text(x[i] + half, base, " %.3f" % base, fontsize=7,
                 va="center", ha="left")
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["P@5", "P@10", "NDCG@10", "MRR"])
    ax1.set_title("Verbatim vs hand-authored paraphrase queries",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Paired deltas with BOTH bootstrap CIs: target-level, which assumes the
    # targets are independent, and conversation-level, which does not.
    reg = agg["paraphrase_regression"]
    keys = ["semantic_only__p@5", "ncm__p@5",
            "semantic_only__ndcg@10", "ncm__ndcg@10"]
    labels = ["sem_only\nP@5", "ncm\nP@5", "sem_only\nNDCG@10", "ncm\nNDCG@10"]
    means = [reg[k]["mean"] for k in keys]
    los = [reg[k]["mean"] - reg[k]["ci_low"] for k in keys]
    his = [reg[k]["ci_high"] - reg[k]["mean"] for k in keys]
    clo = [reg[k]["mean"] - reg[k]["cluster_bootstrap"]["ci_low"] for k in keys]
    chi = [reg[k]["cluster_bootstrap"]["ci_high"] - reg[k]["mean"] for k in keys]
    colors = ["#c0392b", "#27ae60", "#c0392b", "#27ae60"]

    xb = np.arange(len(keys))
    ax2.bar(xb, means, 0.55, color=colors, edgecolor="black", linewidth=1.0)
    ax2.errorbar(xb - 0.09, means, yerr=[los, his], fmt="none", ecolor="black",
                 elinewidth=1.6, capsize=5,
                 label="target-level bootstrap (assumes independence)")
    ax2.errorbar(xb + 0.09, means, yerr=[clo, chi], fmt="none", ecolor="#34495e",
                 elinewidth=1.6, capsize=5, linestyle=":",
                 label="conversation-level cluster bootstrap")
    ax2.axhline(0.0, color="black", linewidth=1.2)
    for i, m in enumerate(means):
        va = "top" if m < 0 else "bottom"
        ax2.text(i, m, "%+.3f" % m, ha="center", va=va, fontsize=9,
                 fontweight="bold")
    ax2.set_ylabel("Paraphrase minus verbatim")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_title("Paired paraphrase delta, 95 percent bootstrap CI",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "EXP24 paraphrase robustness, %d paired targets from %d conversations, "
        "relevance from corpus session_id, paraphrases hand-authored, "
        "%d bootstrap resamples"
        % (agg["n_paired_targets"], agg["n_conversations_in_paired_set"],
           reg["ncm__p@5"]["n_resamples"]),
        fontsize=10.5, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_channel_variance(agg, path):
    """
    Left: mean weighted channel value, which says how much each channel adds
    to the distance. Right: std of the weighted channel ACROSS candidates,
    which is the only thing that can reorder a ranking. A tall left bar with a
    flat right bar is an inert channel: it shifts every candidate equally.
    """
    ch = agg["channels"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(CHANNELS))
    width = 0.36

    for i, condition in enumerate(CONDITIONS):
        offset = (i - 0.5) * width
        vals = [ch[condition][c]["weighted_mean"] for c in CHANNELS]
        hatch = "" if condition == "verbatim" else "//"
        bars = ax1.bar(x + offset, vals, width, label=condition, hatch=hatch,
                       color=[CHANNEL_COLORS[c] for c in CHANNELS],
                       edgecolor="black", linewidth=1.0, alpha=0.9)
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylabel("Mean weighted channel value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["alpha*d_sem", "beta*d_emo", "gamma*d_state", "delta*d_time"],
                        fontsize=9)
    ax1.set_title("Contribution to the distance VALUE\n(does not imply ranking influence)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, condition in enumerate(CONDITIONS):
        offset = (i - 0.5) * width
        vals = [ch[condition][c]["weighted_std"] for c in CHANNELS]
        hatch = "" if condition == "verbatim" else "//"
        bars = ax2.bar(x + offset, vals, width, label=condition, hatch=hatch,
                       color=[CHANNEL_COLORS[c] for c in CHANNELS],
                       edgecolor="black", linewidth=1.0, alpha=0.9)
        for b, c in zip(bars, CHANNELS):
            share = ch[condition][c]["std_share"]
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     "%.4f\n(%.1f%%)" % (b.get_height(), 100.0 * share),
                     ha="center", va="bottom", fontsize=7.5)

    ax2.set_ylabel("Std of weighted channel across candidates")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["alpha*d_sem", "beta*d_emo", "gamma*d_state", "delta*d_time"],
                        fontsize=9)
    ax2.set_title("Dispersion across candidates\n(this is what reorders a ranking)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "EXP24 per-channel decomposition of the shipped composite distance, "
        "weights alpha=0.4 beta=0.2 gamma=0.3 delta=0.1",
        fontsize=10.5, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def write_report(results, path):
    agg = results["aggregate"]
    ds = results["dataset"]
    ch = agg["channels"]
    reg = agg["paraphrase_regression"]

    L = []
    L.append("EXP24: Paraphrase Robustness Diagnosis and Per-Channel Attribution")
    L.append("=" * 66)
    L.append("")
    L.append("Seed: %d    Encoder backend: %s    Encoder state_dim: %d"
             % (results["config"]["seed"], results["config"]["encoder_backend"],
                results["state_padding"]["encoder_state_dim"]))
    L.append("Retrieval weights: alpha=%.2f beta=%.2f gamma=%.2f delta=%.2f (shipped defaults, unchanged)"
             % tuple(results["config"]["weights"][k]
                     for k in ("alpha", "beta", "gamma", "delta")))
    L.append("")
    L.append("RUN SCALE")
    L.append("- smoke mode: %s" % results["config"]["smoke"])
    L.append("- max_conversations: %d (the committed default is %d)"
             % (results["config"]["max_conversations"], MAX_CONVERSATIONS))
    L.append("- conversations loaded: %d" % ds["conversations_loaded"])
    L.append("- held-out targets: %d" % ds["held_out_targets"])
    L.append("- paired targets evaluated: %d from %d conversations, %s per conversation"
             % (agg["n_paired_targets"], agg["n_conversations_in_paired_set"],
                agg["targets_per_conversation"]))
    L.append("- bootstrap resamples: %d" % results["config"]["bootstrap_resamples"])
    L.append("- %s" % results["config"]["scale_note"])
    L.append("")
    L.append("DISCLOSURE")
    L.append("The paraphrase arm uses %d HAND-AUTHORED paraphrases written by the"
             % ds["n_paraphrases_authored"])
    L.append("experimenter. Every paraphrase-arm number below is a number over")
    L.append("hand-authored query text. The relevance label is the corpus")
    L.append("session_id field on both arms and was not authored here.")
    L.append("")
    L.append("Dataset")
    L.append("- Corpus: %s" % results["config"]["corpus"])
    L.append("- Conversations loaded: %d" % ds["conversations_loaded"])
    L.append("- Paired targets evaluated: %d" % agg["n_paired_targets"])
    L.append("- Mean store size: %.1f memories" % ds["mean_store_size"])
    L.append("- Mean relevant per query: %.2f" % ds["mean_relevant_per_query"])
    L.append("- Random-guess precision: %.4f (mean over queries of n_relevant/store_size)"
             % ds["random_guess_precision"])
    rb = results["random_baseline_per_metric"]
    if rb:
        L.append("- Monte Carlo chance level per metric, %d random rankings per"
                 % rb["n_permutations_per_query"])
        L.append("  query using that query's own n_relevant and store_size:")
        L.append("  P@5 %.4f  P@10 %.4f  NDCG@10 %.4f  MRR %.4f"
                 % (rb["p@5"], rb["p@10"], rb["ndcg@10"], rb["mrr"]))
        L.append("  The MRR chance level is far above the precision ratio, because")
        L.append("  a uniformly random ranking still places some relevant item")
        L.append("  early most of the time. Comparing MRR against the precision")
        L.append("  ratio overstates the margin over chance.")
    L.append("- Paraphrase source-prefix integrity failures: %d"
             % ds["integrity_failures"])
    L.append("")
    orth = results["orthography_confound"]
    L.append("Second manipulated variable: orthography")
    L.append("- Hand-authored paraphrases checked: %d"
             % orth["n_paraphrases_checked"])
    L.append("- Paraphrases using a British spelling absent from their source")
    L.append("  turn: %d of %d"
             % (orth["n_paraphrases_with_spelling_variant"],
                orth["n_paraphrases_checked"]))
    L.append("- Method: %s" % orth["method"])
    if orth["n_paraphrases_with_spelling_variant"] > 0:
        for h in orth["detail"]:
            L.append("  conv %d session %d: %s"
                     % (h["conv_id"], h["session_id"],
                        ", ".join("%s (American form %s, source uses it: %s)"
                                  % (v["variant"], v["american_form"],
                                     v["source_uses_american_form"])
                                  for v in h["variants"])))
        L.append("  Rewording and orthography are therefore CONFOUNDED in this")
        L.append("  stimulus set. The paraphrase arm varies two things at once,")
        L.append("  wording and spelling convention, and this design cannot")
        L.append("  separate their contributions. The paraphrases were not")
        L.append("  rewritten, because changing the stimuli would break")
        L.append("  comparability with the previously shipped run.")
    else:
        L.append("  No British-spelling variant from the wordlist was found, so")
        L.append("  no orthographic confound is detected by this check. The")
        L.append("  wordlist is hand-built and not exhaustive, so this is a")
        L.append("  lower bound.")
    L.append("")
    L.append("Decomposition fidelity")
    L.append("- Max abs error between the reconstructed composite and the shipped")
    L.append("  retrieve_top_k_fast distances over the returned top-k: %.3e"
             % results["decomposition_max_abs_error"])
    L.append("  Below the %.1e gate, so the decomposition reproduces the shipped"
             % DECOMPOSITION_TOL)
    L.append("  ranking function and the channel figures below describe the real")
    L.append("  system. The run aborts if this gate is not met.")
    L.append("")

    L.append("Arm results (mean over %d paired targets)" % agg["n_paired_targets"])
    hdr = ("%-30s%9s%9s%11s%9s%10s" %
           ("arm", "P@5", "P@10", "NDCG@10", "MRR", "ms"))
    L.append(hdr)
    L.append("-" * len(hdr))
    for condition in CONDITIONS:
        for system in SYSTEMS:
            key = "%s__%s" % (condition, system)
            a = agg["arms"][key]
            L.append("%-30s%9.4f%9.4f%11.4f%9.4f%10.3f"
                     % (key, a["p@5"], a["p@10"], a["ndcg@10"], a["mrr"],
                        a["latency_ms"]))
    L.append("")
    L.append("Latency caveat: not a latency benchmark. retrieve_semantic_only")
    L.append("rebuilds its own (n,128) matrix per call while retrieve_top_k_fast")
    L.append("reads a cache warmed once per conversation. exp4 is the latency")
    L.append("measurement. These columns are recorded only for completeness.")
    L.append("")

    L.append("Paraphrase regression (paraphrase minus verbatim, paired)")
    L.append("Two CIs per row. 'target CI' resamples targets and assumes they are")
    L.append("independent, which they are not. 'cluster CI' resamples whole")
    L.append("conversations and is the interval to quote. Wilcoxon p is the exact")
    L.append("signed-rank test and also assumes independent pairs.")
    hdr2 = "%-24s%10s%10s%9s%19s%19s%10s" % (
        "system / metric", "verbatim", "paraphr.", "delta",
        "target CI", "cluster CI", "wilcox p")
    L.append(hdr2)
    L.append("-" * len(hdr2))
    for system in SYSTEMS:
        for name in ("p@5", "p@10", "ndcg@10", "mrr"):
            e = reg["%s__%s" % (system, name)]
            w = e["wilcoxon"]
            wp = "%.4f" % w["p_value"] if w else "n/a"
            cb = e["cluster_bootstrap"]
            L.append("%-24s%10.4f%10.4f%+9.4f%19s%19s%10s"
                     % ("%s %s" % (system, name), e["verbatim_mean"],
                        e["paraphrase_mean"], e["mean"],
                        "[%+.4f,%+.4f]" % (e["ci_low"], e["ci_high"]),
                        "[%+.4f,%+.4f]" % (cb["ci_low"], cb["ci_high"]), wp))
    L.append("")
    L.append("Sensitivity of each row (B3, B4, S1)")
    hdr3 = "%-24s%10s%12s%9s%12s%11s" % (
        "system / metric", "nonzero", "MDE(80pct)", "ICC(1)", "effective n",
        "wilcox")
    L.append(hdr3)
    L.append("-" * len(hdr3))
    for system in SYSTEMS:
        for name in ("p@5", "p@10", "ndcg@10", "mrr"):
            e = reg["%s__%s" % (system, name)]
            icc = e["icc"] or {}
            w = e["wilcoxon"] or {}
            L.append("%-24s%7d/%-2d%12s%9s%12s%11s"
                     % ("%s %s" % (system, name), e["n_nonzero_deltas"], e["n"],
                        ("%.4f" % e["mde"]) if e["mde"] is not None else "n/a",
                        ("%+.4f" % icc["icc1"]) if icc else "n/a",
                        ("%.2f" % icc["effective_n"]) if icc else "n/a",
                        w.get("method", "n/a")))
    L.append("nonzero is the count of targets whose metric actually changed,")
    L.append("out of the paired total. MDE is the smallest paired mean difference")
    L.append("this design would detect 80 percent of the time at two-sided 0.05.")
    L.append("A row whose nonzero count is small cannot produce an interval that")
    L.append("clears zero no matter what the effect is, so its CI carries no")
    L.append("effect-size information. ICC(1) above zero means targets inside a")
    L.append("conversation resemble each other and the target CI is too narrow. A")
    L.append("NEGATIVE ICC(1) means targets inside a conversation differ more than")
    L.append("targets drawn at random, which makes the design effect less than 1")
    L.append("and pushes the effective n above the nominal n. That is a real")
    L.append("property of these 14 conversations, not a gain in information, and")
    L.append("the cluster CI is then narrower than the target CI. Read the cluster")
    L.append("CI in both directions.")
    L.append("")

    L.append("Does the composite add fragility beyond the encoder?")
    for key, e in agg["difference_of_differences"].items():
        cb = e["cluster_bootstrap"]
        icc = e["icc"] or {}
        L.append("- %s: %+.4f, n=%d, %d of %d targets changed"
                 % (key, e["mean"], e["n"], e["n_nonzero_deltas"], e["n"]))
        L.append("  target CI [%+.4f, %+.4f], cluster CI [%+.4f, %+.4f]"
                 % (e["ci_low"], e["ci_high"], cb["ci_low"], cb["ci_high"]))
        L.append("  MDE at 80pct power, two-sided 0.05: %s%s"
                 % (("%.4f" % e["mde"]) if e["mde"] is not None else "n/a",
                    ("  ICC(1) %+.4f, effective n %.2f"
                     % (icc["icc1"], icc["effective_n"])) if icc else ""))
        if e["bound_low_on_zero"] or e["bound_high_on_zero"]:
            L.append("  A bound sits exactly on zero. With this many nonzero paired")
            L.append("  differences that bound is arithmetically forced and states")
            L.append("  only how many targets changed, not how large any effect is.")
    L.append("A CI containing zero means no difference between the composite and")
    L.append("the bare encoder was DETECTED. It does not establish that none")
    L.append("exists; any true difference smaller than the MDE above would be")
    L.append("missed by this design.")
    L.append("")
    lm = agg["query_length_match"]
    L.append("Query length across the two arms")
    L.append("- Mean characters: verbatim %.1f, hand-authored paraphrase %.1f"
             % (lm["verbatim_mean_chars"], lm["paraphrase_mean_chars"]))
    L.append("  paired mean change %+.1f characters (%+.1f percent)"
             % (lm["mean_char_delta"],
                100.0 * lm["mean_char_delta"] / max(lm["verbatim_mean_chars"], 1e-9)))
    if lm["wilcoxon_on_char_delta"]:
        lw = lm["wilcoxon_on_char_delta"]
        L.append("- Exact Wilcoxon on the paired character counts: p = %.4f"
                 % lw["p_value"])
        if lw["p_value"] < 0.05:
            L.append("  The arms are NOT length-matched. Query length is a second")
            L.append("  manipulated variable alongside rewording in this stimulus")
            L.append("  set, and this design cannot separate the two.")
        else:
            L.append("  No systematic length difference between the arms was")
            L.append("  detected at this sample size.")
    L.append("- Paired length change vs paired metric change, Spearman rho:")
    for key, c in sorted(lm["paired_length_vs_metric_spearman"].items()):
        flag = "  <- p < 0.05" if c["p_value"] < 0.05 else ""
        L.append("    %-24s rho %+.4f  p %.4f  n %d%s"
                 % (key, c["spearman_rho"], c["p_value"], c["n"], flag))
    L.append("  A significant rho means that metric's movement tracks the")
    L.append("  per-target query length change, which is a stronger statement")
    L.append("  than the arm-level length difference above: it would locate the")
    L.append("  length effect target by target and not only in the arm means.")
    L.append("  A null rho does not clear the arm-level difference, because a")
    L.append("  per-target rank correlation and a difference in arm means are")
    L.append("  different quantities.")
    L.append("")
    return L


def write_report_part2(results, L):
    agg = results["aggregate"]
    ch = agg["channels"]
    disp = results["state_dispersion"]
    ctrl = results["state_controllability"]

    L.append("Per-channel decomposition across candidates")
    L.append("Ranking depends only on how much a channel VARIES across")
    L.append("candidates. A channel with near-zero std shifts every candidate by")
    L.append("nearly the same amount and rarely reorders them, whatever its weight.")
    L.append("'std share' is each channel's share of the SUM OF THE FOUR")
    L.append("PER-CHANNEL STANDARD DEVIATIONS. It is not a variance")
    L.append("decomposition: std(sum) does not equal sum(std) unless the channels")
    L.append("are perfectly correlated, and d_emo and d_state are correlated here.")
    L.append("A variance-based share would put d_sem higher still, so this share")
    L.append("understates d_sem's dominance rather than overstating it.")
    L.append("")
    for condition in CONDITIONS:
        L.append("  condition = %s" % condition)
        hdr = "  %-14s%12s%12s%12s%12s%14s" % (
            "channel", "raw mean", "wtd mean", "wtd std", "std share", "spearman")
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for c in CHANNELS:
            e = ch[condition][c]
            L.append("  %-14s%12.4f%12.4f%12.6f%11.1f%%%14.4f"
                     % (c, e["raw_mean"], e["weighted_mean"], e["weighted_std"],
                        100.0 * e["std_share"], e["spearman_with_total"]))
        L.append("  d_emo vs d_state Pearson r across candidates: %.4f"
                 % ch[condition]["emo_state_raw_pearson_mean"])
        L.append("")

    off = results["emo_state_offset"]
    if off.get("n_memories_compared"):
        L.append("  e_emotional and the state snapshot are ONE TURN APART")
        L.append("  build_store passes the PRE-update auto-state to")
        L.append("  encode_emotional, while MemoryStore.add overwrites")
        L.append("  auto_state_snapshot with the POST-update state, so a memory's")
        L.append("  emotional projection and its state snapshot come from")
        L.append("  different vectors one turn apart. Measured over %d memories in"
                 % off["n_memories_compared"])
        L.append("  %d evaluated stores: %d pairs identical, mean L2 gap %.6f, max"
                 % (off["n_conversations"], off["n_identical"],
                    off["mean_l2_same_index"]))
        L.append("  %.6f. Shifting by one turn closes the gap exactly: max L2"
                 % off["max_l2_same_index"])
        L.append("  between pre-update state i and snapshot i-1 is %.3e."
                 % off["max_l2_pre_i_vs_post_i_minus_1"])
        L.append("  This offset is recorded, not changed. ncm/memory.py is not")
        L.append("  modified by this experiment.")
        L.append("")
    L.append("State channel controllability (the retrieve_top_k_fast API trap)")
    L.append("- Probe A, s_current_normalized argument varied with")
    L.append("  store.auto_state pinned: ranking identical = %s"
             % ctrl["probe_a_argument_is_ignored"])
    if ctrl["probe_a_argument_is_ignored"]:
        L.append("  The argument IS ignored, as documented at ncm/retrieval.py:346")
        L.append("  against line 372, which is why this script installs the state")
        L.append("  by assigning store.auto_state.state instead.")
    else:
        L.append("  The argument was NOT ignored on this run. That contradicts the")
        L.append("  premise for installing the state by assignment, so the state")
        L.append("  channel results below are not interpretable as written.")
    L.append("- Probe B, store.auto_state.state assigned two opposite extremes:")
    L.append("  ranking changed = %s, distances changed = %s"
             % (ctrl["probe_b_state_assignment_changes_ranking"],
                ctrl["probe_b_state_assignment_changes_distances"]))
    if ctrl["probe_b_state_assignment_changes_distances"]:
        L.append("  The distances DID move, so the state channel is live and was")
        L.append("  genuinely driven. A small measured state effect below is")
        L.append("  therefore not a false negative from a dead input.")
    else:
        L.append("  The distances did NOT move. The state input is dead on this")
        L.append("  run, so every state-channel null below is uninterpretable.")
    if not ctrl["probe_b_state_assignment_changes_ranking"]:
        L.append("  The RANKING did not change even at the two opposite extremes,")
        L.append("  which bounds how much the state channel can reorder this store")
        L.append("  at these weights. This is a statement about this store and")
        L.append("  these two extreme states, not a general claim.")
    for name, s in ctrl["d_state_by_installed_state"].items():
        L.append("  installed=%-9s d_state mean=%.4f std=%.6f range=%.6f"
                 % (name, s["d_state_mean"], s["d_state_std"], s["d_state_range"]))
    L.append("")

    L.append("Auto-state dispersion")
    L.append("Three scopes, kept apart. Across-target dispersion within one arm")
    L.append("and the within-target shift between arms are different quantities.")
    for condition in CONDITIONS:
        c = disp.get("query_state_pairwise_l2_%s_only" % condition)
        if c:
            L.append("- ACROSS TARGETS, %s arm only: pairwise L2 between %d query"
                     % (condition, c["n_states"]))
            L.append("  states over %d pairs: min %.4f, mean %.4f, max %.4f"
                     % (c["n_pairs"], c["min"], c["mean"], c["max"]))
    v = disp.get("verbatim_vs_paraphrase_state_l2")
    if v:
        L.append("- WITHIN TARGET, verbatim vs hand-authored paraphrase (n=%d):"
                 % v["n"])
        L.append("  min %.4f, mean %.4f, max %.4f" % (v["min"], v["mean"], v["max"]))
    q = disp.get("query_state_pairwise_l2")
    if q:
        L.append("- POOLED over both arms, %d query states over %d pairs:"
                 % (disp["n_query_states"], q["n_pairs"]))
        L.append("  min %.4f, mean %.4f, max %.4f" % (q["min"], q["mean"], q["max"]))
        L.append("  This pooled figure mixes the two scopes above and is reported")
        L.append("  only for continuity with the earlier run. Read the two")
        L.append("  separated scopes instead.")
    L.append("- Max abs deviation of any query state dimension from neutral 0.5: %.4f"
             % disp.get("query_state_max_abs_deviation_from_neutral", 0.0))
    if "stored_memory_state_max_abs_deviation_from_neutral" in disp:
        L.append("- Max abs deviation of any STORED memory state from 0.5: %.4f"
                 % disp["stored_memory_state_max_abs_deviation_from_neutral"])
        L.append("  Scope: all %d stored turns across all %d evaluated stores."
                 % (disp["n_stored_snapshots"],
                    disp["n_stores_contributing_snapshots"]))
    L.append("- Analytic ceiling on one EMA step per dimension: %s"
             % disp["analytic_note"])
    L.append("")

    sp = results["state_padding"]
    L.append("Structural zero padding")
    L.append("- AutoStateTracker emits %d dims, encoder.state_dim is %d, "
             "encode_state returns %d dims"
             % (sp["auto_state_dims_emitted"], sp["encoder_state_dim"],
                sp["encode_state_output_len"]))
    L.append("- Trailing dims all zero: %s, values %s"
             % (sp["trailing_dims_are_zero"], sp["trailing_dim_values"]))
    if not sp["trailing_dims_are_zero"]:
        L.append("- The trailing dimensions are NOT zero, so they are not")
        L.append("  structural padding and the note below does not apply. The run")
        L.append("  aborts on this condition.")
    L.append("- %s" % sp["note"])
    L.append("")

    L.append("Figures")
    L.append("- experiments/results/exp24/exp24_paraphrase_regression.png")
    L.append("  Left panel: the four arm means per metric, with a Monte Carlo")
    L.append("  chance level drawn separately for each metric. Right panel: the")
    L.append("  paired paraphrase delta with the target-level and the")
    L.append("  conversation-level bootstrap CI side by side.")
    L.append("- experiments/results/exp24/exp24_channel_variance.png")
    L.append("  Per-channel weighted mean and per-channel across-candidate")
    L.append("  standard deviation, for both query conditions.")
    L.append("")

    L.append("Reading of the result")
    for line in results["conclusion"]:
        L.append("- %s" % line)
    L.append("")
    return L


def _excludes_zero(entry):
    """
    True only when the interval clears zero by more than CI_ZERO_TOL.

    A bound landing exactly on zero is NOT treated as excluding it. These
    metrics are discrete, so the bootstrap distribution of the mean has atoms
    and a bound can sit precisely at zero; gating a verdict on a strict
    inequality against such an atom made the verdict a function of the seed.
    """
    return entry["ci_low"] > CI_ZERO_TOL or entry["ci_high"] < -CI_ZERO_TOL


def _zero_strictly_interior(entry):
    """True when zero is strictly inside the interval, not sitting on a bound."""
    return entry["ci_low"] < -CI_ZERO_TOL and entry["ci_high"] > CI_ZERO_TOL


def _bound_on_zero_phrase(entry):
    if entry.get("bound_low_on_zero") and entry.get("bound_high_on_zero"):
        return " Both bounds sit exactly on zero."
    if entry.get("bound_high_on_zero"):
        return " The upper bound sits exactly on zero, not below it."
    if entry.get("bound_low_on_zero"):
        return " The lower bound sits exactly on zero, not above it."
    return ""


def _mde_text(entry):
    if entry.get("mde") is None:
        return "[NEEDS SOURCE: mde could not be computed at this n]"
    return "%.4f" % entry["mde"]


def build_conclusion(agg, ctrl, disp, offset, padding):
    """
    Data-driven conclusion lines. Each statement is gated on the measurement
    that supports it, so the text cannot claim more than was observed.

    Two rules apply throughout. A directional claim requires an interval that
    clears zero by more than CI_ZERO_TOL. A null claim requires an interval with
    zero strictly interior, and is stated together with the minimum detectable
    effect, because an interval whose bound rests on zero only reports how many
    targets changed.

    Scope of every line below: one encoder (all-MiniLM-L6-v2), one corpus
    (experiments/data/real_world_corpus/train.jsonl), one machine, one protocol,
    and hand-authored paraphrases.
    """
    out = []
    ch = agg["channels"]
    reg = agg["paraphrase_regression"]

    ncm_p5 = reg["ncm__p@5"]
    sem_p5 = reg["semantic_only__p@5"]

    out.append(
        "Scale: %d paired targets from %d conversations, %d bootstrap resamples, "
        "hand-authored paraphrases."
        % (agg["n_paired_targets"], agg["n_conversations_in_paired_set"],
           ncm_p5["n_resamples"])
    )

    def _describe(entry, label):
        cb = entry["cluster_bootstrap"]
        base = (
            "%s changes by %+.4f from verbatim to hand-authored paraphrase "
            "queries. Target-level 95pct CI [%+.4f, %+.4f]; conversation-level "
            "cluster CI [%+.4f, %+.4f], which is the one to read because the "
            "targets are clustered."
            % (label, entry["mean"], entry["ci_low"], entry["ci_high"],
               cb["ci_low"], cb["ci_high"])
        )
        if _excludes_zero(cb):
            base += " The cluster CI excludes zero."
        else:
            base += (
                " The cluster CI includes zero, so no paraphrase regression is "
                "established at this sample size."
                + _bound_on_zero_phrase(cb)
            )
        base += (
            " %d of %d targets changed on this metric; the smallest difference "
            "this design would detect at 80 percent power is %s."
            % (entry["n_nonzero_deltas"], entry["n"], _mde_text(entry))
        )
        return base

    out.append(_describe(ncm_p5, "NCM P@5"))
    out.append(_describe(sem_p5, "The semantic-only baseline P@5"))

    icc = ncm_p5.get("icc")
    if icc:
        out.append(
            "The targets are clustered by conversation, not independent. "
            "Measured ICC(1) for the NCM P@5 delta is %+.4f over %d "
            "conversations of mean size %.2f, a design effect of %.3f and an "
            "effective sample size of %.2f against %d nominal observations."
            % (icc["icc1"], icc["n_clusters"], icc["mean_cluster_size"],
               icc["design_effect"], icc["effective_n"], ncm_p5["n"])
        )

    dod = agg["difference_of_differences"]["dod__p@5"]
    dod_cb = dod["cluster_bootstrap"]
    if _excludes_zero(dod_cb):
        out.append(
            "The composite loses %+.4f more than the bare encoder on P@5, "
            "conversation-level cluster CI [%+.4f, %+.4f], which excludes zero, "
            "so NCM's extra channels do add paraphrase fragility on this corpus "
            "with this encoder."
            % (dod["mean"], dod_cb["ci_low"], dod_cb["ci_high"])
        )
    elif _zero_strictly_interior(dod_cb):
        out.append(
            "No difference in paraphrase P@5 loss between the composite and the "
            "bare encoder was detected: %+.4f, conversation-level cluster CI "
            "[%+.4f, %+.4f], zero strictly interior. At n=%d the smallest "
            "difference this design would detect at 80 percent power is %s P@5, "
            "so a composite-added fragility below that remains possible. %d of "
            "%d targets changed on this measure."
            % (dod["mean"], dod_cb["ci_low"], dod_cb["ci_high"], dod["n"],
               _mde_text(dod), dod["n_nonzero_deltas"], dod["n"])
        )
    else:
        out.append(
            "No difference in paraphrase P@5 loss between the composite and the "
            "bare encoder was detected: %+.4f, conversation-level cluster CI "
            "[%+.4f, %+.4f].%s Only %d of %d targets changed on this measure, "
            "which is what forces that bound, so the interval reports the number "
            "of targets that moved and carries no effect-size information. At "
            "n=%d the smallest difference this design would detect at 80 percent "
            "power is %s P@5, so a composite-added fragility below that remains "
            "possible."
            % (dod["mean"], dod_cb["ci_low"], dod_cb["ci_high"],
               _bound_on_zero_phrase(dod_cb), dod["n_nonzero_deltas"], dod["n"],
               dod["n"], _mde_text(dod))
        )

    # S5: MRR is the largest movement in the table and it moves the OTHER way.
    ncm_mrr = reg["ncm__mrr"]
    sem_mrr = reg["semantic_only__mrr"]
    p5_mag = max(abs(ncm_p5["mean"]), abs(sem_p5["mean"]))
    mrr_mag = max(abs(ncm_mrr["mean"]), abs(sem_mrr["mean"]))
    direction = "RISES" if ncm_mrr["mean"] > 0 else "FALLS"
    line = (
        "MRR %s under hand-authored paraphrase, by %+.4f for NCM and %+.4f for "
        "the semantic-only baseline. That is %s in magnitude than the largest "
        "P@5 movement (%.4f) and %s in sign to it, so the word regression does "
        "not describe every metric in this table."
        % (direction, ncm_mrr["mean"], sem_mrr["mean"],
           "larger" if mrr_mag > p5_mag else "smaller", p5_mag,
           "opposite" if ncm_mrr["mean"] * ncm_p5["mean"] < 0 else "equal")
    )
    ncm_mrr_cb = ncm_mrr["cluster_bootstrap"]
    if _excludes_zero(ncm_mrr_cb):
        line += (
            " The NCM MRR conversation-level cluster CI is [%+.4f, %+.4f], which "
            "excludes zero." % (ncm_mrr_cb["ci_low"], ncm_mrr_cb["ci_high"])
        )
    else:
        line += (
            " The NCM MRR conversation-level cluster CI is [%+.4f, %+.4f], which "
            "includes zero, so the direction is not established."
            % (ncm_mrr_cb["ci_low"], ncm_mrr_cb["ci_high"])
        )
    out.append(line)

    lm = agg["query_length_match"]
    len_w = lm["wilcoxon_on_char_delta"]
    len_pct = 100.0 * lm["mean_char_delta"] / max(lm["verbatim_mean_chars"], 1e-9)
    if len_w is not None and len_w["p_value"] < 0.05:
        out.append(
            "The two arms are NOT length-matched at this scale. The hand-authored "
            "paraphrases are longer by %+.1f characters on average (%+.1f "
            "percent), exact Wilcoxon on the paired character counts p = %.4f. "
            "Query length is therefore a second manipulated variable alongside "
            "rewording, and this design cannot separate the two."
            % (lm["mean_char_delta"], len_pct, len_w["p_value"])
        )
    else:
        out.append(
            "The two arms are length-matched: mean change %+.1f characters (%+.1f "
            "percent), exact Wilcoxon on the paired character counts p = %s."
            % (lm["mean_char_delta"], len_pct,
               ("%.4f" % len_w["p_value"]) if len_w else "n/a")
        )

    sig_len = {k: c for k, c in lm["paired_length_vs_metric_spearman"].items()
               if c["p_value"] < 0.05}
    if sig_len:
        out.append(
            "The per-target length change also tracks the per-target metric "
            "change at p < 0.05 for %s, so for those metrics the movement cannot "
            "be attributed to rewording alone."
            % "; ".join("%s rho %+.4f p %.4f" % (k, c["spearman_rho"], c["p_value"])
                        for k, c in sorted(sig_len.items()))
        )
    else:
        arms_differ = len_w is not None and len_w["p_value"] < 0.05
        out.append(
            "No paired length-versus-metric Spearman correlation reaches p < 0.05 "
            "over the %d metrics tested, the largest being %s. %s"
            % (len(lm["paired_length_vs_metric_spearman"]),
               "; ".join(
                   "%s rho %+.4f p %.4f" % (k, c["spearman_rho"], c["p_value"])
                   for k, c in sorted(
                       lm["paired_length_vs_metric_spearman"].items(),
                       key=lambda kv: -abs(kv[1]["spearman_rho"]))[:1]),
               ("No metric's per-target movement was found to track per-target "
                "length change. This does NOT clear the arm-level length "
                "difference reported above: a per-target rank correlation and a "
                "difference in arm means are different quantities, and with %d "
                "targets this test is underpowered against a small one."
                % lm["n"]) if arms_differ else
               ("No metric's per-target movement was found to track per-target "
                "length change."))
        )

    for condition in CONDITIONS:
        shares = {c: ch[condition][c]["std_share"] for c in CHANNELS}
        top = max(shares, key=shares.get)
        out.append(
            "Under %s queries, d_%s supplies %.1f percent of the SUMMED "
            "per-channel across-candidate standard deviation; the shares are sem "
            "%.1f, emo %.1f, state %.1f, time %.1f percent. This is a share of a "
            "sum of standard deviations, not a variance decomposition, because "
            "std(sum) does not equal sum(std) unless the channels are perfectly "
            "correlated. A variance-based share would put d_sem higher, so this "
            "figure understates its dominance."
            % (condition, top, 100.0 * shares[top],
               100.0 * shares["sem"], 100.0 * shares["emo"],
               100.0 * shares["state"], 100.0 * shares["time"])
        )

    state_std = ch["verbatim"]["state"]["weighted_std"]
    sem_std = ch["verbatim"]["sem"]["weighted_std"]
    if sem_std > 0 and state_std / sem_std < 0.1:
        out.append(
            "gamma*d_state has %.1f times less across-candidate dispersion than "
            "alpha*d_sem (%.6f against %.6f), so despite carrying weight "
            "gamma=0.3 it rarely changes the order. It is not incapable of "
            "reordering: no single-channel ablation was run here, so the extra "
            "channels' contribution to the ranking is not attributed."
            % (sem_std / max(state_std, 1e-12), state_std, sem_std)
        )

    if ctrl["probe_a_argument_is_ignored"]:
        out.append(
            "retrieve_top_k_fast ignored its s_current_normalized argument: two "
            "opposite argument values produced an identical ranking."
        )
    else:
        out.append(
            "Probe A did NOT reproduce the ignored-argument behaviour on this "
            "run, so the premise for installing the query state by assigning "
            "store.auto_state.state does not hold and the state-channel numbers "
            "above are not interpretable as written."
        )
    if ctrl["probe_b_state_assignment_changes_distances"]:
        out.append(
            "Assigning store.auto_state.state DID change the distances, so the "
            "state channel was genuinely driven and the low dispersion above is "
            "not a false negative from an uncontrolled input."
        )
    else:
        out.append(
            "Assigning store.auto_state.state did NOT change the distances, so "
            "the state input was dead on this run and every state-channel null "
            "above is uninterpretable."
        )
    if not ctrl["probe_b_state_assignment_changes_ranking"]:
        out.append(
            "Probe B moved the distances but not the returned ranking, which "
            "bounds how far the state channel can reorder this one store at the "
            "shipped weights. That is a statement about this store and these two "
            "extreme installed states, not a general property."
        )

    for condition in CONDITIONS:
        c = disp.get("query_state_pairwise_l2_%s_only" % condition)
        if c:
            out.append(
                "Across targets within the %s arm alone, pairwise L2 between the "
                "%d inferred query states spans %.4f to %.4f over %d pairs, "
                "against a per-dimension single-step ceiling of at most 0.125, "
                "because AutoStateTracker.update applies one EMA step from the "
                "neutral 0.5 vector."
                % (condition, c["n_states"], c["min"], c["max"], c["n_pairs"])
            )
    v = disp.get("verbatim_vs_paraphrase_state_l2")
    if v:
        out.append(
            "Within a target, the verbatim-to-paraphrase state shift is a "
            "separate and smaller quantity: mean L2 %.4f over %d targets. The "
            "earlier pooled figure mixed these two scopes."
            % (v["mean"], v["n"])
        )

    emo_r = ch["verbatim"]["emo_state_raw_pearson_mean"]
    line = (
        "d_emo and d_state correlate at Pearson r = %.4f across candidates under "
        "verbatim queries. They are built from RELATED BUT DIFFERENT vectors, "
        "not the same one: e_emotional is a fixed linear projection W_emo of the "
        "PRE-update auto-state, while d_state compares the POST-update "
        "auto_state_snapshot of the same memory."
        % emo_r
    )
    if offset.get("n_memories_compared"):
        line += (
            " Measured over %d stored memories in %d stores, %d pairs are "
            "identical and the mean L2 gap is %.6f, while the max L2 between the "
            "pre-update state at index i and the snapshot at index i-1 is %.3e, "
            "which places them exactly one turn apart."
            % (offset["n_memories_compared"], offset["n_conversations"],
               offset["n_identical"], offset["mean_l2_same_index"],
               offset["max_l2_pre_i_vs_post_i_minus_1"])
        )
    line += (
        " r stays below 1 for two further reasons: W_emo has shape %s, so it maps "
        "the %d informative state dimensions into %d and loses rank, and "
        "encode_emotional L2-normalizes the projection, which discards its "
        "magnitude (measured output norm %.4f)."
        % (tuple(padding["w_emo_shape"]), padding["auto_state_dims_emitted"],
           padding["w_emo_shape"][0], padding["encode_emotional_output_norm"])
    )
    out.append(line)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="EXP24 paraphrase robustness diagnosis")
    p.add_argument("--max-conversations", type=int, default=MAX_CONVERSATIONS,
                   help="cap on conversations read from the corpus")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES,
                   help="bootstrap resamples for the paired CIs; this value is "
                        "threaded into aggregate() and is the number actually used")
    p.add_argument("--smoke", action="store_true",
                   help="reduced-scale validation run: 4 conversations, 2000 "
                        "resamples. NOT the shipped configuration; the shipped "
                        "numbers come from the defaults with no flags")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    max_conv = 4 if args.smoke else args.max_conversations
    n_boot = 2000 if args.smoke else args.bootstrap

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("[exp24] seed=%d max_conversations=%d bootstrap=%d%s"
          % (SEED, max_conv, n_boot, " (SMOKE, NOT the shipped configuration)"
             if args.smoke else ""))

    conversations, integrity_failures = load_conversations(CORPUS_PATH, max_conv)
    if not conversations:
        print("[exp24] ERROR: no usable conversations loaded. Aborting.")
        return 1
    if integrity_failures:
        print("[exp24] ERROR: %d paraphrase source-prefix mismatches. The corpus "
              "no longer matches the authored paraphrases. Aborting."
              % len(integrity_failures))
        for f in integrity_failures:
            print("[exp24]   %s expected %r found %r"
                  % (f["key"], f["expected_prefix"], f["found_prefix"]))
        return 1

    n_targets = sum(len(c.held_out) for c in conversations)
    print("[exp24] %d conversations, %d held-out targets"
          % (len(conversations), n_targets))

    encoder = SentenceEncoder(
        model_name="all-MiniLM-L6-v2", model_dir=os.path.join(ROOT_DIR, "models")
    )
    backend = encoder.backend
    print("[exp24] encoder backend: %s" % backend)
    if backend != "sentence-transformers":
        print("[exp24] ABORT: the hash fallback carries no semantic structure, so "
              "no retrieval number here would be meaningful.")
        print("[exp24]        reason: %s" % encoder.backend_error)
        return 1

    padding = check_state_padding(encoder)
    if not padding["trailing_dims_are_zero"]:
        print("[exp24] ABORT: encode_state trailing dims are not zero (%s), so "
              "they are not structural padding and the state-channel reading "
              "below would be wrong." % padding["trailing_dim_values"])
        return 1

    print("[exp24] evaluating %d targets x 2 conditions x 2 systems" % n_targets)
    ev = evaluate_queries(conversations, encoder, verbose=args.verbose)
    records = ev["records"]
    if not records:
        print("[exp24] ERROR: no queries scored. Aborting.")
        return 1

    if ev["decomposition_max_abs_error"] > DECOMPOSITION_TOL:
        print("[exp24] ABORT: decomposition_max_abs_error = %.3e exceeds the "
              "%.1e gate, so the four-channel decomposition is NOT the shipped "
              "ranking function and no channel attribution below would describe "
              "the real system."
              % (ev["decomposition_max_abs_error"], DECOMPOSITION_TOL))
        return 1

    if ev["conversations_with_non_unit_strength"] > 0:
        print("[exp24] ABORT: %d conversations have a non-unit strength cache. "
              "The strength modulator is then not exactly 1.0 and the "
              "four-channel decomposition is incomplete."
              % ev["conversations_with_non_unit_strength"])
        return 1

    agg = aggregate(records, rng, n_resamples=n_boot)
    if agg["n_paired_targets"] == 0:
        print("[exp24] ERROR: no paired targets. Aborting.")
        return 1

    random_baseline = per_metric_random_baseline(records, rng)
    orth = audit_orthography_confound(conversations)
    print("[exp24] orthography confound: %d of %d hand-authored paraphrases use "
          "a British spelling absent from their source turn"
          % (orth["n_paraphrases_with_spelling_variant"],
             orth["n_paraphrases_checked"]))

    # Controllability probe on the first conversation's store.
    probe_conv = conversations[0]
    probe_store, _, _ = build_store(probe_conv, encoder)
    probe_text = probe_conv.turns[sorted(probe_conv.held_out.values())[0]].text
    ctrl = probe_state_controllability(probe_store, encoder, probe_text)
    if not ctrl["probe_b_state_assignment_changes_distances"]:
        print("[exp24] ABORT: probe B did not move the distances, so the state "
              "input is dead and every state-channel null would be a false "
              "negative from an uncontrolled input.")
        return 1

    # S6: dispersion over the stored states of EVERY evaluated store, not just
    # the probe conversation.
    disp = measure_state_dispersion(
        records, ev["store_snapshots"], ev["snapshot_scope"]
    )

    weights = probe_store.profile.retrieval_weights
    results = {
        "experiment": "exp24_paraphrase_robustness_diagnosis",
        "config": {
            "seed": SEED,
            "corpus": "experiments/data/real_world_corpus/train.jsonl",
            "encoder_backend": backend,
            "encoder_model": "all-MiniLM-L6-v2",
            "max_conversations": max_conv,
            "bootstrap_resamples": n_boot,
            "bootstrap_resamples_note": (
                "this value is threaded into aggregate() and appears as "
                "n_resamples in every bootstrap entry; the two cannot disagree"
            ),
            "bootstrap_note": BOOTSTRAP_NOTE,
            "scale_note": (
                "the shipped numbers are the committed defaults with no flags: "
                "max_conversations=%d and bootstrap=%d. A --smoke run reads 4 "
                "conversations and is a validation run only, never the shipped "
                "result." % (MAX_CONVERSATIONS, BOOTSTRAP_RESAMPLES)
            ),
            "k_values": list(K_LIST),
            "smoke": bool(args.smoke),
            "weights": {
                "alpha": weights.alpha, "beta": weights.beta,
                "gamma": weights.gamma, "delta": weights.delta,
            },
            "weights_note": "shipped defaults, unmodified by this script",
            "decay_rate": probe_store.profile.decay_rate,
            "relevance_definition": (
                "a stored turn is relevant iff it shares the held-out query "
                "turn's corpus session_id"
            ),
            "oracle_leak": (
                "none; the query auto-state is inferred from the query text "
                "alone and no query-side input derives from the label"
            ),
            "state_channel_control": (
                "store.auto_state.state assigned directly, because "
                "retrieve_top_k_fast ignores its s_current_normalized argument"
            ),
            "timer": "time.perf_counter",
        },
        "discovery_phase": DISCOVERY_FINDINGS,
        "dataset": {
            "conversations_loaded": len(conversations),
            "held_out_targets": n_targets,
            "n_paraphrases_authored": len(PARAPHRASE_TABLE),
            "paraphrase_provenance": (
                "HAND-AUTHORED by the experimenter; not a corpus field. The "
                "relevance label is corpus-derived on both arms."
            ),
            "integrity_failures": len(integrity_failures),
            "mean_store_size": float(np.mean(ev["store_sizes"])) if ev["store_sizes"] else 0.0,
            "mean_relevant_per_query": float(np.mean(
                [r["n_relevant"] for r in records]
            )),
            "random_guess_precision": float(np.mean(ev["relevant_fractions"]))
            if ev["relevant_fractions"] else 0.0,
            "random_guess_definition": (
                "mean over queries of n_relevant / store_size; independent of k"
            ),
            "conversations_with_non_unit_strength": ev["conversations_with_non_unit_strength"],
            "strength_note": (
                "measured: %d of %d evaluated conversations have a non-unit "
                "strength cache. no reinforce() is called, so every strength is "
                "1.0 and the strength modulator is exactly 1.0, which makes the "
                "four-channel decomposition exact. The run aborts if this count "
                "is not zero."
                % (ev["conversations_with_non_unit_strength"], len(conversations))
            ),
        },
        "decomposition_max_abs_error": ev["decomposition_max_abs_error"],
        "decomposition_gate": {
            "tolerance": DECOMPOSITION_TOL,
            "passed": bool(ev["decomposition_max_abs_error"] <= DECOMPOSITION_TOL),
            "note": "the run returns 1 and writes nothing when this fails",
        },
        "random_baseline_per_metric": random_baseline,
        "orthography_confound": orth,
        "emo_state_offset": ev["emo_state_offset"],
        "state_padding": padding,
        "state_controllability": ctrl,
        "state_dispersion": disp,
        "aggregate": agg,
        "per_query_records": records,
    }
    results["conclusion"] = build_conclusion(
        agg, ctrl, disp, ev["emo_state_offset"], padding
    )

    json_path = os.path.join(RESULTS_DIR, "exp24_paraphrase_diagnosis.json")
    txt_path = os.path.join(RESULTS_DIR, "exp24_paraphrase_diagnosis.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = write_report(results, txt_path)
    lines = write_report_part2(results, lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[exp24] saved: %s" % json_path)
    print("[exp24] saved: %s" % txt_path)

    for path in (
        plot_regression(
            agg, results["random_baseline_per_metric"],
            os.path.join(RESULTS_DIR, "exp24_paraphrase_regression.png"),
        ),
        plot_channel_variance(
            agg, os.path.join(RESULTS_DIR, "exp24_channel_variance.png"),
        ),
    ):
        print("[exp24] saved: %s" % path)

    print("")
    for line in results["conclusion"]:
        print("[exp24] %s" % line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
