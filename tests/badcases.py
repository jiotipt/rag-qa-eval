"""D4 · badcase 清单生成（检索 / 生成 / 行为 三层归因），输出 Markdown。

先分层、再对症——这是 D4 与面试的核心能力。

用法（在 `RAG_QA` 目录、已激活 venv 下）：
    python tests/badcases.py                 # 读当前 EVAL_TRACK 的结果，写 <ROOT>/badcases.md
    python tests/badcases.py out_badcases.md # 指定输出文件
    $env:EVAL_TRACK="deepseek"; python tests/badcases.py   # 对对照轨生成

结果文件由 eval_config.RESULTS_PATH 决定（随 EVAL_TRACK 切换）。
"""
import json
import os
import sys

from eval_config import EVAL_TRACK, RESULTS_PATH, ROOT, THRESHOLDS

REFUSAL_MARKS = ("无法回答",)

# 归因层 → 建议动作（先分层，再对症）
LAYER_ACTION = {
    "检索层·未命中": "top-k 未含真值片段 → 调 top-k / 切块大小 / overlap，或换 embedding",
    "检索层·精度": "相关片段排名靠后或被近重复块干扰 → 语料去重 / 加 reranker 重排",
    "生成层·过度拒答": "命中真值却仍拒答 → 本地模型 grounding 不足；强化 prompt（有依据就答）",
    "生成层·质量": "答案未达指标阈值 → 改 prompt / 降 temperature / 加引用约束",
    "行为层": "该答未答 / 该拒未拒 / 该澄清未澄清 → 行为约束（prompt），或复核 judge 口径",
    "其他": "人工复核",
}


def _num(v):
    return v if isinstance(v, (int, float)) else None


def classify(row: dict) -> str:
    """把失败用例归到「检索层 / 生成层 / 行为层」之一（顺序即优先级）。"""
    behavior = row.get("behavior")
    out = row.get("actual_output", "") or ""
    refused = any(m in out for m in REFUSAL_MARKS)
    m = row.get("metrics", {}) or {}
    hit = row.get("retrieval_hit")
    cp = _num(m.get("contextual_precision"))
    ar = _num(m.get("answer_relevancy"))
    faith = _num(m.get("faithfulness"))

    if behavior in ("拒答", "澄清"):
        return "行为层"
    if hit == 0:
        return "检索层·未命中"
    if refused:
        return "生成层·过度拒答"
    if cp is not None and cp < THRESHOLDS["contextual_precision"]:
        return "检索层·精度"
    if (ar is not None and ar < THRESHOLDS["answer_relevancy"]) or (
        faith is not None and faith < THRESHOLDS["faithfulness"]
    ):
        return "生成层·质量"
    return "其他"


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(f"未找到 {RESULTS_PATH}，请先运行 test_rag_quality.py")

    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "badcases.md")
    fails = [r for r in rows if not r.get("passed")]

    by_layer: dict[str, list[str]] = {}
    lines = [
        "# D4 · badcase 清单与三层归因",
        "",
        f"> 轨：`{EVAL_TRACK}` ｜ 结果文件：`{os.path.basename(RESULTS_PATH)}` ｜ "
        f"失败 {len(fails)} / 总 {len(rows)}",
        "",
        "| id | 类别 | 行为 | 检索命中 | AR | Faith | CP | 归因层 | 现象 | 建议动作 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fails:
        layer = classify(r)
        by_layer.setdefault(layer, []).append(r["id"])
        out = (r.get("actual_output", "") or "").replace("\n", " ")
        phenomenon = "拒答" if any(m in out for m in REFUSAL_MARKS) else out[:50]
        m = r.get("metrics", {}) or {}
        lines.append(
            f"| {r['id']} | {r.get('category','')} | {r.get('behavior','')} | "
            f"{r.get('retrieval_hit','-')} | {_fmt(_num(m.get('answer_relevancy')))} | "
            f"{_fmt(_num(m.get('faithfulness')))} | {_fmt(_num(m.get('contextual_precision')))} | "
            f"{layer} | {phenomenon} | {LAYER_ACTION[layer]} |"
        )

    lines += ["", "## 归因汇总", ""]
    for layer in LAYER_ACTION:
        ids = by_layer.get(layer)
        if ids:
            lines.append(f"- **{layer}（{len(ids)}）**：{'、'.join(ids)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[badcase] 失败 {len(fails)}/{len(rows)}（轨={EVAL_TRACK}）")
    for layer in LAYER_ACTION:
        ids = by_layer.get(layer)
        if ids:
            print(f"  {layer}: {len(ids)} 条 → {', '.join(ids)}")
    print(f"[badcase] 已写入 {out_path}")


if __name__ == "__main__":
    main()
