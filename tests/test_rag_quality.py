import pytest

from runner import run_case


def test_rag_case(rag, judge, case, run_idx, record_result):
    result = run_case(rag, judge, case)
    result["run_idx"] = run_idx
    record_result(result)
    assert result["passed"], (
        f"{result['id']}({result['category']}) 未通过 | "
        f"metrics={result['metrics']} | retrieval_hit={result['retrieval_hit']}"
    )
