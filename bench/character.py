"""Does the AI character stay consistent with itself? Its own facts, asked in character.

Multi-Session Chat, test split. Speaker 2 plays the AI character, Speaker 1 the user.
The character's card says nothing but its name, so everything it knows about itself
must come from memory of the five sessions. For each character we turn its own persona
lines into in-character questions ("I own a Jeep" -> "What car do you drive?", gold
"Jeep"; drafted by the local LLM, kept only if the answer is copied from the fact and is
not in the question), then the character answers in its own words under each memory
setup. Scored by whether the reply contains the fact; replies that state no fact, or
admit not remembering, are counted separately.

  python -m bench.character --convs 80      # writes bench/results/character-msc-<model>.json
"""

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.data import DATA_DIR                          # noqa: E402
from bench.embed import embed                            # noqa: E402
from bench.metrics import paired_bootstrap               # noqa: E402
from bench.persona import Cached, persona_lines          # noqa: E402
from bench.qa import ask_raw, normalize                  # noqa: E402
from ncm import Memory                                   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHAR, USER = "Speaker 2", "Speaker 1"
PROBE_SYSTEM = ("You write quiz questions. Reply with JSON only: "
                '{"question": "...", "answer": "..."}')
CHARACTER_SYSTEM = (f"You are {CHAR}, chatting with {USER}, a friend you have talked with many times. "
                    "Stay in character and answer in one short sentence, in the first person. Use your "
                    "memories of your past conversations; if you do not remember, say so honestly.")
HEDGE = re.compile(r"\b(not sure|don'?t (?:remember|recall|know)|can'?t (?:remember|recall)|no idea|"
                   r"not certain|i forget|haven'?t mentioned|didn'?t mention)\b", re.I)


WH = re.compile(r"^(what|where|which|who|how|when)\b", re.I)
FEW_SHOT = """Statement: I own a Jeep and drive it to work.
{"question": "What car do you drive?", "answer": "Jeep"}

Statement: My favorite band is Coldplay.
{"question": "Which band is your favorite?", "answer": "Coldplay"}

Statement: I work as a nurse at the city hospital.
{"question": "What do you do for a living?", "answer": "nurse"}

Statement: I have two cats named Milo and Luna.
{"question": "What are your cats called?", "answer": "Milo and Luna"}"""


def make_probe(fact, cache):
    """An in-character question whose answer is a short span of the fact, or None."""
    if fact in cache:
        return cache[fact]
    prompt = (f"{FEW_SHOT}\n\nStatement: {fact}\n"
              "Write one question a friend could ask this person to check ONE specific detail of the "
              "statement. Start the question with What, Which, Where, Who, How or When. The answer is "
              "1 to 4 words copied exactly from the statement and must not appear in the question.")
    out = None
    try:
        d = json.loads(ask_raw(PROBE_SYSTEM, prompt, MODEL, num_predict=80, fmt="json"))
        q, a = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip()
        an, qn, fn = normalize(a), set(normalize(q)), " ".join(normalize(fact))
        if (q.endswith("?") and WH.match(q) and 0 < len(an) <= 4 and " ".join(an) in fn
                and not set(an) & qn and an[0] not in {"i", "my", "me", "yes", "no"}):
            out = {"question": q, "answer": a}
    except (json.JSONDecodeError, AttributeError):
        pass
    cache[fact] = out
    return out


def contains(reply, gold):
    r, g = " ".join(normalize(reply)), " ".join(normalize(gold))
    return bool(g) and g in r


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=80)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--arms", default="none,recent,vector,vector-k18,ncm-recall,ncm-context")
    a = ap.parse_args()
    MODEL = a.model
    convs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "msc", "test.jsonl"), encoding="utf-8")]
    rng = random.Random(1)
    picked = rng.sample(range(len(convs)), a.convs)

    probe_path = os.path.join(RES, "character-probes.json")
    pcache = json.load(open(probe_path, encoding="utf-8")) if os.path.exists(probe_path) else {}
    items = []
    for ci in picked:
        facts = [f for f in persona_lines(convs[ci], CHAR) if len(f.split()) >= 4]
        for f in rng.sample(facts, min(4, len(facts))):
            p = make_probe(f, pcache)
            if p:
                items.append({"conv": ci, "fact": f, **p})
    json.dump(pcache, open(probe_path, "w", encoding="utf-8"), indent=1)
    print(f"{len(items)} probes from {a.convs} characters", flush=True)

    turns = [f"{d['speaker']}: {d['text']}" for ci in picked for s in convs[ci]["sessions"] for d in s["dialogue"]]
    emb = Cached(dict(zip([(False, t) for t in turns], embed(turns, "minilm", "doc", tag="msc"))))
    mems = {}
    for ci in picked:
        m = Memory(embedder=emb)
        for s, sess in enumerate(convs[ci]["sessions"]):
            m.new_session()
            m.add_many([d["text"] for d in sess["dialogue"]], [d["speaker"] for d in sess["dialogue"]],
                       [1.7e9 + s * 7 * 86400 + i * 60 for i in range(len(sess["dialogue"]))])
        mems[ci] = m

    def lines(m, rows):
        who = lambda sp: "you" if sp == CHAR else sp   # noqa: E731
        return "Your memories of past conversations:\n" + "\n".join(
            f"- {who(m.speakers[r])}: {m.texts[r]}" for r in sorted(rows))

    cache_path = os.path.join(RES, "character-cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[(r["arm"], r["model"], r["key"])] = r
    out = open(cache_path, "a", encoding="utf-8")
    arms = a.arms.split(",")
    res = {arm: [] for arm in arms}
    t0 = time.time()
    for n, it in enumerate(items):
        m, q = mems[it["conv"]], it["question"]
        key = f"{it['conv']}|{it['fact']}|{q}"
        for arm in arms:
            if (arm, a.model, key) not in cache:
                if arm == "none":
                    ctx = ""
                elif arm == "recent":
                    ctx = lines(m, range(max(0, len(m) - 10), len(m)))
                elif arm == "vector":            # plain vector search over the whole conversation
                    qv = m.embedder([q], query=True)[0]
                    ctx = lines(m, np.argsort(-(m.embeddings @ qv), kind="stable")[:6])
                elif arm == "vector-k18":        # control: vector search with context()'s prompt budget
                    qv = m.embedder([q], query=True)[0]
                    ctx = lines(m, np.argsort(-(m.embeddings @ qv), kind="stable")[:18])
                elif arm == "ncm-recall":
                    ctx = lines(m, [m._row(h.id) for h in m.recall(q, k=6)])
                elif arm == "ncm-context":       # persona block about itself + its own lines as "you"
                    ctx = m.context(q, k=4, persona_of=CHAR, persona_k=6, self_name=CHAR).replace(
                        f"What you know about {CHAR}", "What you know about yourself").replace(
                        "Their mood lately", "Your mood lately")
                prompt = (f"{ctx}\n\n" if ctx else "") + f"{USER}: {q}"
                reply = ask_raw(CHARACTER_SYSTEM, prompt, a.model, num_predict=60)
                cache[(arm, a.model, key)] = {"arm": arm, "model": a.model, "key": key, "reply": reply,
                                              "gold": it["answer"], "chars": len(prompt)}
                out.write(json.dumps(cache[(arm, a.model, key)]) + "\n")
                out.flush()
            r = cache[(arm, a.model, key)]
            ok = contains(r["reply"], it["answer"])
            res[arm].append({"correct": float(ok), "hedged": float(not ok and bool(HEDGE.search(r["reply"]))),
                             "chars": r["chars"]})
        if n % 40 == 0:
            print(f"{n}/{len(items)} {time.time() - t0:.0f}s " + " ".join(
                f"{arm}={np.mean([x['correct'] for x in res[arm]]):.3f}" for arm in arms), flush=True)

    summary = {"model": a.model, "characters": a.convs, "probes": len(items), "arms": {}}
    for arm in arms:
        rows = res[arm]
        c = [x["correct"] for x in rows]
        summary["arms"][arm] = {
            "recalls_own_fact": float(np.mean(c)),
            "says_it_does_not_remember": float(np.mean([x["hedged"] for x in rows])),
            "other_wrong_or_vague": float(1 - np.mean(c) - np.mean([x["hedged"] for x in rows])),
            "prompt_chars": float(np.mean([x["chars"] for x in rows]))}
        if arm != "vector" and "vector" in res:
            v = [x["correct"] for x in res["vector"]]
            bs = paired_bootstrap(v, c)
            wins, losses = sum(1 for x, y in zip(v, c) if y > x), sum(1 for x, y in zip(v, c) if x > y)
            summary["arms"][arm]["vs_vector"] = bs | {
                "only_this_right": wins, "only_vector_right": losses,
                "mcnemar_p": binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0}
    json.dump(summary, open(os.path.join(RES, f"character-msc-{a.model.replace(':', '_')}.json"), "w"), indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, indent=1))


MODEL = "qwen2.5:3b"

if __name__ == "__main__":
    main()
