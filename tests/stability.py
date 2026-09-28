"""v2 · 稳定性复核：把「同套用例重复 N 次」的结果按**多数投票**聚合成稳定通过率。

为什么需要它（面试话术）：
  LLM 即使 temperature=0 也不完全确定（批处理/MoE 路由/浮点顺序），judge 亦不确定。
  单跑 43/50=86% 只是「一次抽样」，实测同代码重跑会差 ±1~2 条。
  所以对每条用例跑 N 次，按「N 次中过半数通过」判为**稳定通过**，
  再用「稳定通过数 / 用例数」得到可对外讲的通过率，并列出 flaky（时好时坏）清单。

与 summarize.py 的区别：
  summarize.py 把 (用例 × 轮次) 的所有行**拉平**算通过率 = 单跑期望值；
  本脚本**按用例多数投票**，回答「剔除运气后，真实水平是多少」。

用法（在 `RAG_QA` 目录、已激活 venv 下）：
    python tests/stability.py eval_results_v2b_n3.json [out.md]

输入：带 `run_idx` 的多轮结果 JSON（由 `EVAL_RUNS=N` 跑批产生）。
输出：控制台报告 +（可选）markdown。
"""
import json
import os
import statistics
import sys
from collections import defaultdict

METRICS = ["answer_relevancy", "faithfulness", "contextual_precision", "behavior"]
CATEGORIES = ["正常", "模糊", "多跳", "对抗", "越界", "追问"]


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"未找到 {path}")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} 不是预期的非空列表")
    return rows


def _natural_key(cid: str):
    digits = "".join(ch for ch in cid if ch.isdigit())
    return (int(digits) if digits else 1 << 30, cid)


def aggregate(rows: list[dict]) -> dict:
    """按用例聚合。返回 {cases, n_runs, ...}。"""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_case[r["id"]].append(r)

    cases = []
    for cid in sorted(by_case, key=_natural_key):
        runs = sorted(by_case[cid], key=lambda r: r.get("run_idx", 0))
        n_runs = len(runs)
        pass_count = sum(1 for r in runs if r.get("passed"))
        # 多数口径：严格过半（3 次 → ≥2）
        majority = pass_count * 2 > n_runs
        meta = runs[0]
        cases.append(
            {
                "id": cid,
                "category": meta.get("category", "?"),
                "behavior": meta.get("behavior", "?"),
                "n_runs": n_runs,
                "pass_count": pass_count,
                "majority": majority,
                "all_pass": pass_count == n_runs,
                "any_pass": pass_count > 0,
                "pass_flags": [bool(r.get("passed")) for r in runs],
                "metric_errors": sum(1 for r in runs if r.get("metric_error")),
                "hit_flags": [
                    r.get("retrieval_hit")
                    for r in runs
                    if r.get("retrieval_hit") is not None
                ],
            }
        )

    n = len(cases)
    n_runs = cases[0]["n_runs"] if cases else 0
    total_rows = len(rows)
    total_pass = sum(1 for r in rows if r.get("passed"))

    stable_pass = sum(1 for c in cases if c["majority"])
    all_pass = sum(1 for c in cases if c["all_pass"])
    any_pass = sum(1 for c in cases if c["any_pass"])

    means = {}
    for name in METRICS:
        xs = [
            r["metrics"].get(name)
            for r in rows
            if isinstance(r["metrics"].get(name), (int, float))
        ]
        means[name] = statistics.mean(xs) if xs else None

    return {
        "cases": cases,
        "n": n,
        "n_runs": n_runs,
        "total_rows": total_rows,
        "total_pass": total_pass,
        "naive_rate": (total_pass / total_rows) if total_rows else 0.0,
        "stable_pass": stable_pass,
        "stable_rate": (stable_pass / n) if n else 0.0,
        "all_pass": all_pass,
        "all_rate": (all_pass / n) if n else 0.0,
        "any_pass": any_pass,
        "any_rate": (any_pass / n) if n else 0.0,
        "means": means,
    }


def _pct(v):
    return f"{v:.1%}"


def _flags(flags: list[bool]) -> str:
    return "".join("✓" if f else "✗" for f in flags)


def build_report(path: str, agg: dict) -> str:
    cases = agg["cases"]
    n_runs = agg["n_runs"]

    flaky = [c for c in cases if 0 < c["pass_count"] < c["n_runs"]]
    stable_fail = [c for c in cases if c["pass_count"] == 0]

    lines = [
        f"# v2 · 稳定性复核（N={n_runs}，多数投票）",
        "",
        f"> 结果文件：`{os.path.basename(path)}` ｜ 用例数：{agg['n']} ｜ 总轮次：{agg['total_rows']}",
        f"> 判稳定口径：**过半数**通过（{n_runs} 次中 ≥{n_runs // 2 + 1} 次即算稳定通过）",
        "",
        "## 总览（多口径对照）",
        "",
        "| 口径 | 数值 | 含义 |",
        "|---|---|---|",
        f"| 单跑平均通过率 | {_pct(agg['naive_rate'])} ({agg['total_pass']}/{agg['total_rows']}) | 所有轮次拉平＝单跑期望值（≈summarize.py） |",
        f"| **稳定通过率（多数）** | **{_pct(agg['stable_rate'])} ({agg['stable_pass']}/{agg['n']})** | 剔除运气后的可对外值 |",
        f"| 全通过率（{n_runs}/{n_runs}） | {_pct(agg['all_rate'])} ({agg['all_pass']}/{agg['n']}) | 最严口径 |",
        f"| 至少一次通过 | {_pct(agg['any_rate'])} ({agg['any_pass']}/{agg['n']}) | 最松口径 |",
        "",
        "## 用例分类",
        "",
        "| 分类 | 条数 | 用例 |",
        "|---|---|---|",
        f"| 稳定通过（{n_runs}/{n_runs}） | {agg['all_pass']} | - |",
        f"| 多数通过·有波动（≥2/{n_runs} 但非全通过） | {len([c for c in flaky if c['majority']])} | "
        f"{'、'.join(c['id'] for c in flaky if c['majority']) or '无'} |",
        f"| 多数未过·有波动（<{n_runs // 2 + 1} 但至少过 1 次） | {len([c for c in flaky if not c['majority']])} | "
        f"{'、'.join(c['id'] for c in flaky if not c['majority']) or '无'} |",
        f"| 稳定失败（0/{n_runs}） | {len(stable_fail)} | "
        f"{'、'.join(c['id'] for c in stable_fail) or '无'} |",
        "",
        "## Flaky 清单（时好时坏 · 最值得面试讲）",
        "",
    ]
    if flaky:
        lines += [
            "| 用例 | 类别 | 通过次数 | 各轮 | 说明 |",
            "|---|---|---|---|---|",
        ]
        for c in flaky:
            note = "多数通过但有波动" if c["majority"] else "多数未过，偶发通过"
            lines.append(
                f"| {c['id']} | {c['category']} | {c['pass_count']}/{c['n_runs']} | "
                f"{_flags(c['pass_flags'])} | {note} |"
            )
    else:
        lines.append("无（所有用例要么全过、要么全败，非常稳定）。")

    lines += [
        "",
        f"## 逐条明细（共 {agg['n']} 条）",
        "",
        "| 用例 | 类别 | 行为 | 通过 | 判定 | 检索命中 |",
        "|---|---|---|---|---|---|",
    ]
    for c in cases:
        judge = "稳定通过" if c["all_pass"] else ("多数通过" if c["majority"] else ("多数未过" if c["any_pass"] else "稳定失败"))
        if c["hit_flags"]:
            hit = f"{sum(1 for h in c['hit_flags'] if h)}/{len(c['hit_flags'])}"
        else:
            hit = "-"
        lines.append(
            f"| {c['id']} | {c['category']} | {c['behavior']} | "
            f"{c['pass_count']}/{c['n_runs']} | {judge} | {hit} |"
        )

    lines += [
        "",
        "## 分类别（多数口径）",
        "",
        "| 类别 | 总数 | 稳定通过 | 稳定通过率 |",
        "|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        group = [c for c in cases if c["category"] == cat]
        if group:
            ok = sum(1 for c in group if c["majority"])
            lines.append(f"| {cat} | {len(group)} | {ok} | {_pct(ok / len(group))} |")

    lines += [
        "",
        "## 指标均值（全部轮次）",
        "",
        "| 指标 | 均值 |",
        "|---|---|",
    ]
    for name in METRICS:
        v = agg["means"][name]
        lines.append(f"| {name} | {v:.3f} |" if v is not None else f"| {name} | - |")

    lines += [
        "",
        "> 结论写法：稳定通过率若 ≈ 单跑通过率 ⇒ 单跑结果可信、可对外讲；"
        "若明显更低 ⇒ 存在「靠运气通过」的用例，需以稳定值报数，并复盘 flaky 清单。",
    ]
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法：python tests/stability.py <results.json> [out.md]")
    path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    rows = load(path)
    agg = aggregate(rows)
    if agg["n_runs"] < 2:
        print(f"[警告] 该文件每用例仅 {agg['n_runs']} 轮；稳定性复核需要 EVAL_RUNS>=2。仍按单轮输出。\n")
    report = build_report(path, agg)
    print(report)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[stability] 已写入 {out_path}")


if __name__ == "__main__":
    main()
