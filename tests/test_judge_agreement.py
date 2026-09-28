import csv
import json
import os

import pytest

from eval_config import HUMAN_LABELS_PATH, RESULTS_PATH


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in json.load(f)}


def cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    p_obs = sum(1 for a, b in pairs if a == b) / n
    a1 = sum(1 for a, _ in pairs if a == 1) / n
    b1 = sum(1 for _, b in pairs if b == 1) / n
    p_exp = a1 * b1 + (1 - a1) * (1 - b1)
    if p_exp == 1:
        return None
    return (p_obs - p_exp) / (1 - p_exp)


def test_judge_agreement():
    if not os.path.exists(HUMAN_LABELS_PATH):
        pytest.skip(f"缺少人工标注文件：{HUMAN_LABELS_PATH}")
    labeled = [
        row
        for row in csv.DictReader(open(HUMAN_LABELS_PATH, encoding="utf-8-sig"))
        if row.get("human_pass", "").strip() in {"0", "1"}
    ]
    if not labeled:
        pytest.skip("human_labels.csv 尚未填写 human_pass")

    results = _load_json(RESULTS_PATH)
    if not results:
        pytest.skip(f"尚未生成 {RESULTS_PATH}，请先运行 test_rag_quality.py")

    pairs = []
    disagreements = []
    for row in labeled:
        case_id = row["id"]
        if case_id not in results:
            continue
        human = int(row["human_pass"])
        judge_pass = 1 if results[case_id]["passed"] else 0
        pairs.append((human, judge_pass))
        if human != judge_pass:
            disagreements.append(case_id)

    if not pairs:
        pytest.skip("人工标注的 id 与 eval_results.json 无交集")

    agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
    kappa = cohen_kappa(pairs)
    print(f"\n[judge 校验] 样本={len(pairs)} 一致率={agreement:.1%} Cohen's κ={kappa}")
    print(f"[judge 校验] 不一致样例：{disagreements or '无'}")

    assert agreement >= 0.8, f"judge 与人工一致率过低：{agreement:.1%}"
