import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from eval_config import THRESHOLDS

BEHAVIOR_CRITERIA = {
    "拒答": (
        "回答应明确表示无法根据给定知识库作答、或说明该问题超出范围；"
        "不得编造知识库中没有的具体事实。若回答编造了具体事实，判不通过。"
    ),
    "澄清": (
        "面对信息不足或指代不明的问题，回答应先提出澄清问题、或先说明所做假设再作答；"
        "不得在缺少必要信息时直接给出一个可能错误的确定结论。"
    ),
}


def _expected_files(case: dict) -> list[str]:
    ctx = case.get("ground_truth_context", "")
    return [
        part.split("#")[0].strip().replace("\\", "/")
        for part in ctx.split("、")
        if part.strip()
    ]


def retrieval_hit(case: dict, sources: list[str]) -> float | None:
    expected = _expected_files(case)
    if not expected:
        return None
    for src in sources:
        normalized = str(src).replace("\\", "/")
        if any(exp in normalized for exp in expected):
            return 1.0
    return 0.0


def _safe_measure(metric, test_case, retries: int = 1) -> tuple[float | None, bool, str | None]:
    last_error = None
    for _ in range(retries + 1):
        try:
            metric.measure(test_case)
            return metric.score, bool(metric.success), None
        except Exception as exc:  # noqa: BLE001 - 评测失败要记录而非中断整轮
            last_error = str(exc)[:200]
    return None, False, last_error


def run_case(rag, judge, case: dict) -> dict:
    out = rag.answer(case["question"])
    behavior = case["expected_behavior"]
    scores: dict = {}
    passed_parts: list[bool] = []

    if behavior == "作答":
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=out["output"],
            expected_output=case.get("ground_truth_answer") or None,
            retrieval_context=out["contexts"],
        )
        metric_map = {
            "answer_relevancy": AnswerRelevancyMetric(
                threshold=THRESHOLDS["answer_relevancy"], model=judge
            ),
            "faithfulness": FaithfulnessMetric(
                threshold=THRESHOLDS["faithfulness"], model=judge
            ),
            "contextual_precision": ContextualPrecisionMetric(
                threshold=THRESHOLDS["contextual_precision"], model=judge
            ),
        }
    else:
        test_case = LLMTestCase(
            input=case["question"], actual_output=out["output"]
        )
        metric_map = {
            "behavior": GEval(
                name="Behavior",
                criteria=BEHAVIOR_CRITERIA.get(behavior, ""),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
                model=judge,
                threshold=THRESHOLDS["behavior"],
            )
        }

    flags: dict[str, bool] = {}
    for name, metric in metric_map.items():
        score, ok, err = _safe_measure(metric, test_case)
        scores[name] = score
        flags[name] = ok
        if err:
            scores[f"{name}_error"] = err
        passed_parts.append(ok)

    frozen_passed = bool(passed_parts) and all(passed_parts)
    metric_error = any(f"{name}_error" in scores for name in metric_map)
    if behavior == "作答":
        answer_passed = flags.get("answer_relevancy", False) and flags.get("faithfulness", False)
        retrieval_passed = flags.get("contextual_precision", False)
    else:
        answer_passed = None
        retrieval_passed = None

    return {
        "id": case["id"],
        "category": case["category"],
        "behavior": behavior,
        "latency_s": round(out["latency_s"], 3),
        "retrieval_hit": retrieval_hit(case, out["sources"]),
        "sources": [os.path.basename(str(s)) for s in out["sources"]],
        "metrics": scores,
        "passed": frozen_passed,
        "answer_passed": answer_passed,
        "retrieval_passed": retrieval_passed,
        "metric_error": metric_error,
        "passed_fair": None if metric_error else frozen_passed,
        "actual_output": out["output"][:500],
    }
