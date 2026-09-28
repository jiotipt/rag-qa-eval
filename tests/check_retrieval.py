import collections
import os

import rag_under_test
from eval_config import load_dataset
from runner import retrieval_hit

CATEGORIES = ["正常", "模糊", "多跳", "对抗", "越界", "追问"]


def main():
    rows = load_dataset()
    rag_under_test.ensure_ready()

    hit = collections.Counter()
    total = collections.Counter()
    misses = []

    for case in rows:
        docs = rag_under_test.retrieve(case["question"])
        sources = [d.metadata.get("source", "") for d in docs]
        result = retrieval_hit(case, sources)
        if result is None:
            continue
        total[case["category"]] += 1
        if result == 1.0:
            hit[case["category"]] += 1
        else:
            misses.append(
                (case["id"], case["category"], [os.path.basename(str(s)) for s in sources])
            )

    print("=== 检索命中率（只测检索层，不调 LLM）===")
    print(f"{'类别':<6}{'命中/总':>10}{'命中率':>10}")
    overall_hit = 0
    overall_total = 0
    for cat in CATEGORIES:
        if total[cat]:
            overall_hit += hit[cat]
            overall_total += total[cat]
            print(f"{cat:<6}{f'{hit[cat]}/{total[cat]}':>10}{hit[cat] / total[cat]:>10.0%}")
    if overall_total:
        print(f"{'合计':<6}{f'{overall_hit}/{overall_total}':>10}{overall_hit / overall_total:>10.0%}")

    print("\n=== 未命中明细（期望文档 vs 实际召回）===")
    for cid, cat, srcs in misses:
        print(f"{cid} [{cat}] 实际召回: {srcs}")


if __name__ == "__main__":
    main()
