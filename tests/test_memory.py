import os
import threading
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

from ncm import Config, Memory
from ncm.lexical import BM25, tokenize
from ncm.memory import score
from ncm.persona import evidence, persona

DIM = 256


def bow(texts):
    """Deterministic bag-of-words embedder: shares words -> similar vectors."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            out[i, zlib.crc32(w.encode()) % DIM] += 1
    return out


def mem(**cfg):
    return Memory(embedder=bow, config=Config(**cfg))


def test_add_and_recall_finds_the_fact():
    m = mem()
    m.add_many(["I adopted a beagle called Max", "The weather is lovely today",
                "I work as a nurse in Boston"], times=[1, 2, 3])
    top = m.recall("what is the name of my beagle", k=1)
    assert top[0].text == "I adopted a beagle called Max"


def test_exact_repeat_reinforces_instead_of_duplicating():
    m = mem()
    a = m.add("Thanks!", time=1)
    b = m.add("thanks", time=2)
    c, d = m.add_many(["ok", "OK."], times=[3, 4])
    assert a == b and c == d and len(m) == 2
    assert m._uses[m._row(a)] == 1 and m._uses[m._row(c)] == 1


def test_same_text_different_speaker_is_kept():
    m = mem()
    m.add("hello", speaker="user", time=1)
    m.add("hello", speaker="assistant", time=2)
    assert len(m) == 2


def test_sessions_split_on_gap_and_explicitly():
    m = mem(session_gap=60)
    m.add_many(["a one", "b two"], times=[0, 30])
    m.add("c three", time=1000)
    m.new_session()
    m.add("d four", time=1001)
    assert list(m._sessions) == [0, 0, 1, 2]


def test_lift_reaches_previous_turns_only_by_default_and_never_across_sessions():
    dense = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    sessions = np.array([0, 0, 0, 0, 1])
    off = Config(lexical_weight=0, lift_previous=0)
    plain = score(dense, None, sessions, off)
    lift = score(dense, None, sessions, Config(lexical_weight=0)) - plain
    assert lift[1] > lift[0] > 0 and lift[3] == 0 and lift[4] == 0
    both = score(dense, None, sessions, Config(lexical_weight=0, lift_next=0.75)) - plain
    assert both[3] > 0 and both[4] == 0              # next turn lifted, other session untouched
    assert np.argmax(plain + lift) == 2               # the hit itself stays first


def test_speaker_filter():
    m = mem()
    m.add("I love jazz", speaker="user", time=1)
    m.add("I love jazz too", speaker="assistant", time=2)
    assert {r.speaker for r in m.recall("jazz", k=5, speaker="assistant")} == {"assistant"}


def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "m.ncm")
    m = Memory(p, embedder=bow, config=Config(lexical_weight=0.7))
    m.add_many(["I live in Paris", "I drive a red Jeep"], speakers=["user", "user"],
               times=[datetime(2024, 1, 1, tzinfo=timezone.utc), 5e8], metas=[{"mood": 1}, None])
    m.save()
    m2 = Memory(p, embedder=bow)
    assert m2.texts == m.texts and m2.ids == m.ids and m2.metas == m.metas
    assert m2.config.lexical_weight == 0.7
    assert np.allclose(m2.embeddings, m.embeddings, atol=1e-2)
    assert m2.recall("Jeep", k=1)[0].text == "I drive a red Jeep"
    assert m2.add("I live in Paris", time=6e8) == m.ids[0]   # dedupe index survives
    assert len(m2.add_many(["new one", "new two"], times=[7e8, 7e8 + 1])) == 2 and len(m2) == 4
    assert not os.path.exists(p + ".tmp")


def test_load_rejects_wrong_embedder(tmp_path):
    p = str(tmp_path / "m.ncm")
    m = Memory(p, embedder=bow)
    m.add("x y", time=1)
    m.save()
    with pytest.raises(ValueError):
        Memory(p, embedder=lambda t: np.ones((len(t), 3)))


def test_forget_and_capacity():
    m = mem(max_items=3)
    ids = [m.add(f"fact number {w}", time=t) for t, w in enumerate(["one", "two", "three"])]
    m.recall("fact number one", k=1, reinforce=True)
    m.add("fact number four", time=10)
    assert len(m) == 3 and ids[0] in m.ids     # the reinforced memory survived
    m.forget(m.ids[0])
    assert len(m) == 2 and ids[0] not in m.ids
    assert m.recall("four", k=1)[0].text == "fact number four"


def test_context_block_is_chronological_and_has_persona():
    m = mem()
    m.add_many(["I live in Boston", "nice", "Actually I moved, I live in Seattle now"],
               times=[1e9, 1e9 + 10, 1e9 + 1e7])
    block = m.context("where do I live", k=2)
    assert block.index("Boston") < block.rindex("Seattle")
    assert "What you know about user" in block and "Relevant memories" in block


def test_add_messages_skips_system():
    m = mem()
    m.add_messages([{"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi there"},
                    {"role": "assistant", "content": "hello!"}])
    assert m.speakers == ["user", "assistant"]


def test_concurrent_adds_are_safe():
    m = mem()
    ts = [threading.Thread(target=lambda i=i: [m.add(f"thread {i} item {j}", time=j) for j in range(50)])
          for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(m) == 200 and len(m._bm25) == 200 and m.embeddings.shape[0] == 200


def test_empty_store_and_bad_input():
    m = mem()
    assert m.recall("anything") == [] and m.context("anything") == ""
    with pytest.raises(ValueError):
        m.add("   ")


def test_bm25_ranks_rare_term():
    b = BM25()
    for t in ["the cat sat", "dogs run fast", "a cat and a dog"]:
        b.add(t)
    s = b.scores("dog")
    assert s[1] > 0 and s[2] > 0 and s[0] == 0
    assert tokenize("Cities and dogs") == ["city", "dog"]


def test_persona_prefers_self_disclosure():
    assert evidence("I work as a nurse") > evidence("what do you do?") == 0
    assert evidence("do I have a dog?") == 0 and evidence("I have two dogs. Do you?") > 0
    assert evidence("lol yeah") == 0
    assert evidence("do I have a dog") == 0             # a question without "?" is still a question
    m = mem()
    m.add_many(["lol yeah", "I work as a nurse", "how about you?", "I love hiking on weekends"],
               times=[1, 2, 3, 4])
    p = persona(m, k=5)
    assert set(p) == {"I work as a nurse", "I love hiking on weekends"}


def test_capacity_eviction_is_batched_and_index_stays_consistent():
    m = mem(max_items=40)
    for i in range(100):
        m.add(f"note {i} about topic{i}", time=i * 86400.0, meta={"salience": 0.0})
    assert 38 <= len(m) <= 40
    assert len(m._bm25) == len(m) and m.embeddings.shape[0] == len(m)
    last = m.recall("topic99", k=1)[0]
    assert last.text == "note 99 about topic99"
    gone = [i for i in range(100) if f"note {i} about topic{i}" not in m.texts]
    assert gone and max(gone) < 60                 # the oldest unused ones faded


def test_forget_many_then_bm25_rebuild():
    m = mem()
    ids = m.add_many([f"word{i} filler" for i in range(100)], times=range(100))
    m.forget(ids[:90])
    assert len(m) == 10 and m._bm25.n_docs == 10   # tombstones were compacted away
    assert m.recall("word95", k=1)[0].text == "word95 filler"


def test_affect_mood_and_salience():
    from ncm.affect import Affect, label
    m = mem()
    m.add_many(["I'm so happy, this is wonderful news!", "What a great day, I love this."], times=[1, 2])
    md = m.mood()
    assert md["valence"] > 0 and md["label"].startswith("positive")
    assert (m._salience > 0).all()
    m.add("plain note", time=3, meta={"salience": 0.0})
    assert m._salience[-1] == 0.0                      # explicit salience wins
    assert m.metas[0] == {}                            # computed salience is not written into meta
    assert label(-1, 1) == "negative, energetic" and label(0, 0) == "neutral"
    a = Affect(type("E", (), {"name": "sentence-transformers/all-MiniLM-L6-v2"})())
    assert a.W.shape == (384, 2)                        # shipped probe loads


def test_mood_roundtrips(tmp_path):
    m = mem()
    m.add("I'm so happy, this is wonderful news!", time=1)
    m.save(str(tmp_path / "a.ncm"))
    assert Memory(str(tmp_path / "a.ncm"), embedder=bow).mood() == m.mood()


def test_scores_identical_after_forget_and_roundtrip(tmp_path):
    m = mem()
    ids = m.add_many([f"alpha{i % 7} beta{i % 3} gamma{i}" for i in range(60)], times=range(60))
    m.forget(ids[5:20])
    before = m.scores("alpha3 beta1 gamma42")
    m.save(str(tmp_path / "s.ncm"))
    m2 = Memory(str(tmp_path / "s.ncm"), embedder=bow)
    assert np.allclose(m2.scores("alpha3 beta1 gamma42"), before, atol=1e-2)
    assert m2._bm25.n_docs == len(m2) == 45
    m2.add("alpha3 brand new", time=100)
    assert m2.recall("brand new", k=1)[0].text == "alpha3 brand new"


def test_import_v1_file_if_present():
    p = os.path.join(os.path.dirname(__file__), "..", "..", "ncm-data", "v1files", "memory_store.ncm")
    if not os.path.exists(p):
        pytest.skip("no v1 sample file")
    from ncm.legacy import import_v1
    m = import_v1(p, embedder=bow)
    assert len(m) > 0 and set(m.speakers) <= {"user", "assistant"}
    assert all(not t.startswith("user: ") for t in m.texts)


def test_add_many_returns_none_for_memories_forgotten_at_once():
    m = mem(max_items=10)
    ids = m.add_many([f"old note {i}" for i in range(30)], times=range(30),
                     metas=[{"salience": 0.0}] * 30)
    assert len(ids) == 30 and len(m) == 10
    assert all(i is None or i in m.ids for i in ids) and sum(i is not None for i in ids) == 10


def test_punctuation_and_emoji_only_messages_stay_distinct():
    m = mem()
    a, b, c = m.add_many(["!!!", "???", "!!!"], times=[1, 2, 3])
    assert a != b and a == c and len(m) == 2


def test_opening_a_v1_file_points_to_import_v1(tmp_path):
    p = tmp_path / "old.ncm"
    p.write_bytes(b"NCM\x02" + bytes(40))
    with pytest.raises(ValueError, match="import_v1"):
        Memory(str(p), embedder=bow)


def test_mood_skips_questions_and_persona_scales():
    m = mem()
    m.add_many(["I'm so sad, this is terrible news.", "what makes me happy and joyful and great"],
               times=[1, 2])
    assert m.mood()["valence"] == m._valence[0]        # the question did not count
    m.add_many([f"I have hobby number {i}" for i in range(3000)], times=range(3, 3003))
    assert len(persona(m, k=5)) == 5                    # no N x N matrix needed


def test_reranker_reorders_the_first_stage_shortlist():
    calls = []

    def by_length(query, texts):          # a stand-in cross-encoder: prefers longer texts
        calls.append(len(texts))
        return np.array([len(t) for t in texts], dtype=np.float32)

    m = Memory(embedder=bow, reranker=by_length, config=Config(rerank_top=3))
    m.add_many(["cat", "cat sat", "cat sat on the mat today", "dog"], times=[1, 2, 3, 4])
    top = m.recall("cat", k=2)
    assert calls == [3] and len(top) == 2
    assert top[0].text == "cat sat on the mat today" and top[0].score > top[1].score
    assert Memory(embedder=bow).reranker is None   # off by default


def test_emotional_and_used_memories_outlive_neutral_ones_at_capacity():
    m = mem(max_items=20)
    day = 86400.0
    ids = m.add_many([f"neutral fact {i}" for i in range(20)], times=[i * day for i in range(20)],
                     metas=[{"salience": 1.5 if i < 3 else 0.0} for i in range(20)])
    m.recall("neutral fact 5", k=1, reinforce=True)
    m.recall("neutral fact 5", k=1, reinforce=True)
    m.add_many([f"new thing {i}" for i in range(5)], times=[(30 + i) * day for i in range(5)],
               metas=[{"salience": 0.0}] * 5)
    kept = set(m.ids)
    assert {ids[0], ids[1], ids[2]} <= kept      # oldest, but emotional
    assert ids[5] in kept                        # old, but recalled twice
    assert ids[3] not in kept                    # old, neutral, never used: faded
