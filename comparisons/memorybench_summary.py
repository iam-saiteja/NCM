"""Summarise NCM vs MemoryBench's RAG baseline on LoCoMo, LongMemEval and ConvoMem.

Reads the MemoryBench reports (data/runs/{ncm,rag}-{benchmark}/report.json in the MemoryBench
fork) and writes comparisons/results/memorybench-bedrock.json. Both systems answer with
gpt-oss-120b and are graded by gpt-oss-20b, on the same questions; the difference per
benchmark is tested with an exact McNemar test on the paired verdicts. Accuracy is over
every question in the run: MemoryBench leaves a question unscored when the answer comes
back empty (gpt-oss spent its whole token budget reasoning); here that counts as wrong.

  python comparisons/memorybench_summary.py --runs ../memorybench/data/runs
"""

import argparse
import json
import os

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS = ("locomo", "longmemeval", "convomem")
SYSTEMS = ("ncm", "rag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "..", "memorybench", "data", "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "memorybench-bedrock.json"))
    a = ap.parse_args()
    out = {"answering_model": "gpt-oss-120b (Amazon Bedrock)", "judge": "gpt-oss-20b (Amazon Bedrock)",
           "benchmarks": {}}
    for b in BENCHMARKS:
        reps, v = {}, {}
        for s in SYSTEMS:
            p = os.path.join(a.runs, f"{s}-{b}", "report.json")
            if os.path.exists(p):
                reps[s] = json.load(open(p, encoding="utf-8"))
                every = json.load(open(os.path.join(a.runs, f"{s}-{b}", "checkpoint.json"), encoding="utf-8"))["questions"]
                scored = {e["questionId"]: e["score"] for e in reps[s]["evaluations"]}
                v[s] = {q: scored.get(q, 0) for q in every}
        if not reps:
            continue
        res = {}
        for s, r in reps.items():
            res[s] = {"questions": len(v[s]), "correct": sum(v[s].values()),
                      "accuracy": sum(v[s].values()) / len(v[s]),
                      "unscored_counted_wrong": len(v[s]) - r["summary"]["totalQuestions"],
                      "search_ms_median": r["latency"]["search"]["median"],
                      "context_tokens_mean": r["tokens"]["avgContextTokens"],
                      "retrieval": {k: r["retrieval"][k] for k in ("hitAtK", "mrr", "ndcg", "k")} if r.get("retrieval") else None,
                      "by_type": {t: {"total": v["total"], "accuracy": v["accuracy"]} for t, v in r["byQuestionType"].items()}}
        if len(reps) == 2:
            both = sorted(set(v["ncm"]) & set(v["rag"]))
            ncm_only = sum(v["ncm"][q] > v["rag"][q] for q in both)
            rag_only = sum(v["rag"][q] > v["ncm"][q] for q in both)
            res["paired"] = {"questions": len(both),
                             "ncm_accuracy": sum(v["ncm"][q] for q in both) / len(both),
                             "rag_accuracy": sum(v["rag"][q] for q in both) / len(both),
                             "ncm_right_rag_wrong": ncm_only, "rag_right_ncm_wrong": rag_only,
                             "mcnemar_p": binomtest(ncm_only, ncm_only + rag_only).pvalue if ncm_only + rag_only else 1.0}
        out["benchmarks"][b] = res
    paired = [r["paired"] for r in out["benchmarks"].values() if "paired" in r]
    if paired:
        n = sum(p["questions"] for p in paired)
        w, l = sum(p["ncm_right_rag_wrong"] for p in paired), sum(p["rag_right_ncm_wrong"] for p in paired)
        out["all_benchmarks"] = {"questions": n,
                                 "ncm_accuracy": sum(p["ncm_accuracy"] * p["questions"] for p in paired) / n,
                                 "rag_accuracy": sum(p["rag_accuracy"] * p["questions"] for p in paired) / n,
                                 "ncm_right_rag_wrong": w, "rag_right_ncm_wrong": l,
                                 "mcnemar_p": binomtest(w, w + l).pvalue}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)

    print(f"{'benchmark':12} {'NCM':>8} {'RAG':>8} {'paired n':>9} {'p':>8}   search ms (NCM/RAG)   context tok (NCM/RAG)")
    for b, res in out["benchmarks"].items():
        n, r, pr = res.get("ncm"), res.get("rag"), res.get("paired")
        fmt = lambda x: f"{100 * x['accuracy']:.1f}%" if x else "-"
        print(f"{b:12} {fmt(n):>8} {fmt(r):>8} {pr['questions'] if pr else '-':>9} "
              f"{pr['mcnemar_p'] if pr else float('nan'):>8.3g}   "
              f"{n and n['search_ms_median']}/{r and r['search_ms_median']}   "
              f"{n and n['context_tokens_mean']}/{r and r['context_tokens_mean']}")
    if "all_benchmarks" in out:
        t = out["all_benchmarks"]
        print(f"{'all':12} {100 * t['ncm_accuracy']:7.1f}% {100 * t['rag_accuracy']:7.1f}% {t['questions']:>9} {t['mcnemar_p']:>8.3g}")


if __name__ == "__main__":
    main()
