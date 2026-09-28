import json
import os

import pytest
from dotenv import load_dotenv

from eval_config import DATASET_PATH, JUDGE_MODEL, RESULTS_PATH, RUNS_PER_CASE, load_dataset

load_dotenv()

_RESULTS: list[dict] = []


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_dataset(DATASET_PATH)
        metafunc.parametrize("case", cases, ids=[c["id"] for c in cases])
    if "run_idx" in metafunc.fixturenames:
        metafunc.parametrize("run_idx", range(RUNS_PER_CASE))


@pytest.fixture(scope="session")
def judge():
    from deepeval.models import DeepSeekModel

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.exit("未找到 DEEPSEEK_API_KEY，无法运行 LLM-as-judge。", returncode=2)
    return DeepSeekModel(model=JUDGE_MODEL, api_key=api_key, temperature=0)


@pytest.fixture(scope="session")
def rag():
    import rag_under_test

    rag_under_test.ensure_ready()
    return rag_under_test


@pytest.fixture(scope="session")
def record_result():
    return _RESULTS.append


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_RESULTS, f, ensure_ascii=False, indent=2)
    total = len(_RESULTS)
    passed = sum(1 for r in _RESULTS if r["passed"])
    print(f"\n[汇总] 通过率 {passed}/{total} = {passed / total:.1%}")
    print(f"[汇总] 明细已写入 {RESULTS_PATH}")
