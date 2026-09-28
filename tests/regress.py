"""D4 · 回归对比：同一轨「改前 vs 改后」两份 eval_results 的前后 delta。

证明「评测套件 = 回归资产」：改一个变量（prompt / top-k …）→ 重跑 → 看变化。

用法（在 `RAG_QA` 目录、已激活 venv 下）：
    python tests/regress.py eval_results_before.json eval_results.json [out.md]

约定：第一份是「改前（baseline）」，第二份是「改后」。输出通过率 / 指标均值 /
命中率 / 延迟的变化，以及逐条的 pass↔fail 翻转清单。
"""
import json
import os
import statistics
import sys

METRICS = ["answer_relevancy", "faithfulness", "contextual_precision", "behavior"]


def load(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"未找到 {path}")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def stats(rows: dict) -> dict:
    vals = list(rows.values())
    n = len(vals)
    passed = sum(1 for r in vals if r.get("passed"))
    means = {}
    for name in METRICS:
        xs = [r["metrics"].get(name) for r in vals if isinstance(r["metrics"].get(name), (int, float))]
        means[name] = statistics.mean(xs) if xs else None
    hits = [r["retrieval_hit"] for r in vals if r.get("retrieval_hit") is not None]
    lats = [r["latency_s"] for r in vals if r.get("latency_s") is not None]
    fair = [r for r in vals if r.get("passed_fair") is not None]
    ans = [r for r in vals if r.get("answer_passed") is not None]
    return {
        "n": n,
        "passed": passed,
        "pass_rate": passed / n if n else 0.0,
        "fair_pass_rate": (sum(1 for r in fair if r["passed_fair"]) / len(fair)) if fair else None,
        "answer_rate": (sum(1 for r in ans if r["answer_passed"]) / len(ans)) if ans else None,
        "retrieval_rate": (sum(1 for r in ans if r["retrieval_passed"]) / len(ans)) if ans else None,
        "metric_errors": sum(1 for r in vals if r.get("metric_error")),
        "means": means,
        "hit_rate": (sum(hits) / len(hits)) if hits else None,
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
    }


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def _delta(a, b, pct=False):
    if a is None or b is None:
        return "-"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}" if pct else f"{sign}{d:.3f}"


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法：python tests/regress.py <before.json> <after.json> [out.md]")
    before_path, after_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    before, after = load(before_path), load(after_path)
    sb, sa = stats(before), stats(after)

    new_pass = sorted([i for i in after if i in before and not before[i].get("passed") and after[i].get("passed")])
    new_fail = sorted([i for i in after if i in before and before[i].get("passed") and not after[i].get("passed")])
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    lines = [
        "# D4 · 回归对比（改前 vs 改后）",
        "",
        f"> before：`{os.path.basename(before_path)}` ｜ after：`{os.path.basename(after_path)}`",
        "",
        "## 总览",
        "",
        "| 指标 | 改前 | 改后 | Δ |",
        "|---|---|---|---|",
        f"| 样本数 | {sb['n']} | {sa['n']} | {sa['n']-sb['n']:+d} |",
        f"| 通过数 | {sb['passed']} | {sa['passed']} | {sa['passed']-sb['passed']:+d} |",
        f"| 通过率 | {_fmt(sb['pass_rate'], True)} | {_fmt(sa['pass_rate'], True)} | {_delta(sb['pass_rate'], sa['pass_rate'], True)} |",
        f"| 有效通过率(剔除度量异常) | {_fmt(sb['fair_pass_rate'], True)} | {_fmt(sa['fair_pass_rate'], True)} | {_delta(sb['fair_pass_rate'], sa['fair_pass_rate'], True)} |",
        f"| 作答·回答通过率(AR∧F) | {_fmt(sb['answer_rate'], True)} | {_fmt(sa['answer_rate'], True)} | {_delta(sb['answer_rate'], sa['answer_rate'], True)} |",
        f"| 作答·检索通过率(CP) | {_fmt(sb['retrieval_rate'], True)} | {_fmt(sa['retrieval_rate'], True)} | {_delta(sb['retrieval_rate'], sa['retrieval_rate'], True)} |",
        f"| 度量异常数 | {sb['metric_errors']} | {sa['metric_errors']} | {sa['metric_errors'] - sb['metric_errors']:+d} |",
        f"| 检索命中率 | {_fmt(sb['hit_rate'], True)} | {_fmt(sa['hit_rate'], True)} | {_delta(sb['hit_rate'], sa['hit_rate'], True)} |",
        f"| 延迟 P50 | {_fmt(sb['p50'])} | {_fmt(sa['p50'])} | {_delta(sb['p50'], sa['p50'])} |",
        f"| 延迟 P95 | {_fmt(sb['p95'])} | {_fmt(sa['p95'])} | {_delta(sb['p95'], sa['p95'])} |",
        "",
        "## 指标均值",
        "",
        "| 指标 | 改前 | 改后 | Δ |",
        "|---|---|---|---|",
    ]
    for name in METRICS:
        lines.append(
            f"| {name} | {_fmt(sb['means'][name])} | {_fmt(sa['means'][name])} | "
            f"{_delta(sb['means'][name], sa['means'][name])} |"
        )

    lines += [
        "",
        "## 逐条翻转",
        "",
        f"- **转好（fail→pass，{len(new_pass)}）**：{'、'.join(new_pass) or '无'}",
        f"- **转坏（pass→fail，{len(new_fail)}）**：{'、'.join(new_fail) or '无'}",
        f"- 仅存在于 before：{'、'.join(only_before) or '无'}",
        f"- 仅存在于 after：{'、'.join(only_after) or '无'}",
        "",
        "> 结论写法：先看通过率/命中率是否提升，再看有没有「转坏」的回归项；有转坏即为**回归缺陷**。",
    ]

    text = "\n".join(lines) + "\n"
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[regress] 已写入 {out_path}")


if __name__ == "__main__":
    main()
