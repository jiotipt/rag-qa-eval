import json
import statistics

from eval_config import RESULTS_PATH

CATEGORIES = ["正常", "模糊", "多跳", "对抗", "越界", "追问"]
METRICS = ["answer_relevancy", "faithfulness", "contextual_precision", "behavior"]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    print(f"总用例: {total}  通过: {passed}  通过率: {passed / total:.1%}\n")

    print(f"{'类别':<6}{'总数':>6}{'通过':>6}{'通过率':>8}")
    for cat in CATEGORIES:
        group = [r for r in rows if r["category"] == cat]
        if group:
            ok = sum(1 for r in group if r["passed"])
            print(f"{cat:<6}{len(group):>6}{ok:>6}{ok / len(group):>8.0%}")

    print("\n指标均值（仅统计有分数的用例）:")
    for name in METRICS:
        values = [
            r["metrics"].get(name)
            for r in rows
            if isinstance(r["metrics"].get(name), (int, float))
        ]
        if values:
            print(f"  {name:<22} 均值={statistics.mean(values):.3f}  n={len(values)}")

    if any("passed_fair" in r for r in rows):
        fair = [r for r in rows if r.get("passed_fair") is not None]
        errs = sum(1 for r in rows if r.get("metric_error"))
        if fair:
            fair_pass = sum(1 for r in fair if r["passed_fair"])
            print(
                f"\n解耦视图: 有效样本 {len(fair)}/{total}  有效通过率 "
                f"{fair_pass}/{len(fair)} = {fair_pass / len(fair):.1%}  度量异常(已剔除) {errs}"
            )
        ans = [r for r in rows if r.get("answer_passed") is not None]
        if ans:
            a_ok = sum(1 for r in ans if r["answer_passed"])
            r_ok = sum(1 for r in ans if r["retrieval_passed"])
            print(
                f"  作答类拆解: 回答通过(AR∧Faith) {a_ok}/{len(ans)} = {a_ok / len(ans):.1%}"
                f" | 检索通过(CP) {r_ok}/{len(ans)} = {r_ok / len(ans):.1%}"
            )

    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    if latencies:
        print(
            f"\n延迟(生成段): P50={percentile(latencies, 50):.2f}s "
            f"P95={percentile(latencies, 95):.2f}s "
            f"min={min(latencies):.2f}s max={max(latencies):.2f}s"
        )

    hits = [r["retrieval_hit"] for r in rows if r.get("retrieval_hit") is not None]
    if hits:
        print(f"检索命中率: {sum(hits) / len(hits):.1%}  (n={len(hits)})")


if __name__ == "__main__":
    main()
