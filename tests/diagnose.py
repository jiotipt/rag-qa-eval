import collections
import json
import os

from eval_config import RESULTS_PATH

CATEGORIES = ["正常", "模糊", "多跳", "对抗", "越界", "追问"]
REFUSAL_MARK = "无法回答"


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"未找到 {RESULTS_PATH}，请先运行 test_rag_quality.py")
        return
    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    by_cat = collections.Counter()
    pass_cat = collections.Counter()
    for r in rows:
        by_cat[r["category"]] += 1
        if r["passed"]:
            pass_cat[r["category"]] += 1

    print("=== 各类别通过情况 ===")
    print(f"{'类别':<6}{'总数':>6}{'通过':>6}{'失败':>6}")
    for cat in CATEGORIES:
        t = by_cat.get(cat, 0)
        p = pass_cat.get(cat, 0)
        print(f"{cat:<6}{t:>6}{p:>6}{t - p:>6}")

    fails = [r for r in rows if not r["passed"]]
    print(f"\n=== 失败明细（共 {len(fails)} 条）===")
    for r in fails:
        out = r.get("actual_output", "")
        refused = REFUSAL_MARK in out
        errors = {k: v for k, v in r["metrics"].items() if k.endswith("_error")}
        print(
            f"{r['id']} [{r['category']}/{r['behavior']}] hit={r['retrieval_hit']} "
            f"拒答={'是' if refused else '否'} metrics={r['metrics']}"
        )
        print(f"     输出: {out[:80]}")
        if errors:
            print(f"     错误: {errors}")

    hit0 = [r for r in fails if r["retrieval_hit"] == 0]
    hit_miss = [r for r in fails if r["retrieval_hit"] == 1 and REFUSAL_MARK in r.get("actual_output", "")]
    behavior_fail = [r for r in fails if r["behavior"] in ("拒答", "澄清")]
    print("\n=== 失败归因 ===")
    print(f"失败总数              : {len(fails)}")
    print(f"  检索未命中(hit=0)   : {len(hit0)}  ← 检索层问题（调 top-k/切块）")
    print(f"  命中却仍拒答        : {len(hit_miss)}  ← 生成/prompt 层问题")
    print(f"  行为类判定失败      : {len(behavior_fail)}  ← judge/行为标准问题")


if __name__ == "__main__":
    main()
