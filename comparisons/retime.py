"""Re-measure the reader-time column of the end-to-end table.

The original run reached Ollama at "http://localhost:11434". On Windows "localhost" is
tried over IPv6 first and every request waited ~2.1 s before falling back to IPv4, so
the recorded reader times were inflated by that much for every system. Answers and
accuracy were unaffected. This rebuilds each system's real prompt for a fixed sample of
questions and times the reader over 127.0.0.1.

  python -m comparisons.retime          # updates answer_seconds_p50 in results/e2e-stats.json
"""

import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.qa import SYSTEM                                         # noqa: E402
from comparisons.common import RESULTS, chat, e2e_data             # noqa: E402
from comparisons.end_to_end import systems                         # noqa: E402

SAMPLE = 40


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    hays, queries = e2e_data()
    hmap = {h.id: h for h in hays}
    sample = sorted(random.Random(0).sample(queries, SAMPLE), key=lambda q: (q.haystack, q.question))
    path = os.path.join(RESULTS, "e2e-stats.json")
    stats = json.load(open(path, encoding="utf-8"))
    for s in systems():
        s.prepare()
        s.ingest(hays)
        times = []
        for q in sample:
            ctx = s.context(hmap[q.haystack], q)
            prompt = (f"{ctx}\n\n" if ctx else "") + f"Question: {q.question}\nShort answer:"
            t0 = time.perf_counter()
            chat([{"role": "system", "content": SYSTEM["locomo"]}, {"role": "user", "content": prompt}],
                 num_ctx=s.num_ctx)
            times.append(time.perf_counter() - t0)
        st = stats[s.name]
        st["answer_seconds_p50_original_localhost"] = st.get("answer_seconds_p50_original_localhost",
                                                              st["answer_seconds_p50"])
        st["answer_seconds_p50"] = float(np.median(times))
        st["answer_seconds_sample"] = SAMPLE
        print(f"{s.name:50s} reader p50 {st['answer_seconds_p50']:.2f} s "
              f"(was {st['answer_seconds_p50_original_localhost']:.2f} s via localhost)", flush=True)
        json.dump(stats, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
