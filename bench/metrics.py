"""Ranking metrics over full rankings (arrays of turn indices, best first)."""

import numpy as np


def score_query(ranking, gold, sessions=None, ks=(1, 5, 10)):
    gold = set(gold)
    pos = {int(d): r for r, d in enumerate(ranking[:max(100, max(ks))])}
    out = {}
    for k in ks:
        top = set(int(d) for d in ranking[:k])
        out[f"recall@{k}"] = len(top & gold) / len(gold)
        out[f"hit@{k}"] = float(bool(top & gold))
    ranks = sorted(pos[g] for g in gold if g in pos)
    out["mrr"] = 1.0 / (ranks[0] + 1) if ranks else 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r in ranks if r < 10)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg
    if sessions is not None:
        gs = {sessions[g] for g in gold}
        top_s = {sessions[int(d)] for d in ranking[:10]}
        out["session_recall@10"] = len(gs & top_s) / len(gs)
    return out


def aggregate(rows, by=None):
    """Mean of every metric, optionally split by a key (e.g. category)."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    res = {"all": {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n": len(rows)}}
    if by:
        for g in sorted({r[by] for r in rows}):
            sub = [r for r in rows if r[by] == g]
            res[g] = {k: float(np.mean([r[k] for r in sub])) for k in keys} | {"n": len(sub)}
    return res


def paired_bootstrap(a, b, n=10000, seed=0):
    """95% CI and two-sided p for mean(b - a) over paired per-query values."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))], "p": float(max(p, 1.0 / n))}
