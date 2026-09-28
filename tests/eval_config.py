import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 默认数据集：dataset_sample.csv（12 条 / 六类齐全，配套随仓库发布的 kb_docs 示例语料，开箱可跑）。
# 另一个 dataset_v1.csv（50 条）是历史冻结评测集，配套原始内部语料（版权原因未随仓库发布）；
# 如需在自有语料上复现完整实验，用环境变量 EVAL_DATASET=dataset_v1.csv 指定。
DATASET_PATH = os.getenv("EVAL_DATASET", os.path.join(ROOT, "dataset_sample.csv"))
HUMAN_LABELS_PATH = os.getenv("EVAL_HUMAN_LABELS", os.path.join(HERE, "human_labels.csv"))

DB_DIR = os.getenv("EVAL_DB_DIR", os.path.join(ROOT, "chroma_db"))
EMB_MODEL = os.getenv("EVAL_EMB_MODEL", os.path.join(ROOT, "models", "bge-m3"))

# 检索 top-k：实测 k 过大（8）会引入大量跨技能干扰块、拉低 Contextual Precision；k=3 时正常类文件命中率已 95%，为当前最优
TOP_K = int(os.getenv("EVAL_TOP_K", "3"))

# 两阶段检索（v2）：向量召回 RECALL_K 条 → CrossEncoder 重排取 TOP_K。默认 off，保证与 v1 结果可比。
RECALL_K = int(os.getenv("EVAL_RECALL_K", "10"))
RERANK_ENABLED = os.getenv("EVAL_RERANK", "off").strip().lower() in ("1", "on", "true", "yes")
RERANK_MODEL = os.getenv("EVAL_RERANK_MODEL", os.path.join(ROOT, "models", "bge-reranker-v2-m3"))

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# 双轨（D4）：被测对象可切换 —— local（LM Studio 本地轨） / deepseek（API 对照轨）
#   用法：$env:EVAL_TRACK="deepseek"; python -m pytest tests/test_rag_quality.py -v
#   两轨共用同一检索层（同 embedding / 同库 / 同 top-k），只切换「被测生成模型」，
#   以保证双轨对比是控制变量（见 report.md §8）。
#   结果文件按轨分开：local→eval_results.json；deepseek→eval_results_deepseek.json，
#   这样 summarize.py / diagnose.py / badcases.py 会随 EVAL_TRACK 自动读对应结果。
# ---------------------------------------------------------------------------
EVAL_TRACK = os.getenv("EVAL_TRACK", "local").strip().lower()

_TRACKS = {
    "local": {
        "name": "本地轨(LM Studio)",
        "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        "model": os.getenv("EVAL_LLM_MODEL", "qwen3.6-35b-a3b-ud"),
        "reasoning_effort": "none",  # 本地思考模型必须显式关闭，否则返回空回答
        "results": os.path.join(ROOT, "eval_results.json"),
    },
    "deepseek": {
        "name": "对照轨(DeepSeek API)",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": os.getenv("EVAL_DS_MODEL", "deepseek-chat"),
        "reasoning_effort": None,  # DeepSeek 无需关闭思考
        "results": os.path.join(ROOT, "eval_results_deepseek.json"),
    },
}

if EVAL_TRACK not in _TRACKS:
    raise ValueError(f"未知 EVAL_TRACK={EVAL_TRACK!r}，可选：{list(_TRACKS)}")

TRACK = _TRACKS[EVAL_TRACK]
LLM_MODEL = TRACK["model"]  # 兼容旧引用

# 结果文件：显式 EVAL_RESULTS 优先；dataset_v1 按轨命名（历史证据 eval_results*.json）；
# 其他数据集（默认 dataset_sample）派生独立文件名，避免跑样例时覆盖历史证据
_dataset_stem = os.path.splitext(os.path.basename(DATASET_PATH))[0]
_default_results = (
    TRACK["results"]
    if _dataset_stem == "dataset_v1"
    else os.path.join(ROOT, f"{_dataset_stem}_results.json")
)
RESULTS_PATH = os.getenv("EVAL_RESULTS", _default_results)

# 准出标准：D3 跑批前冻结（冻结于 2026-09-25），冻结后不得修改
THRESHOLDS = {
    "answer_relevancy": float(os.getenv("TH_ANSWER_RELEVANCY", "0.5")),
    "faithfulness": float(os.getenv("TH_FAITHFULNESS", "0.5")),
    "contextual_precision": float(os.getenv("TH_CONTEXTUAL_PRECISION", "0.5")),
    "behavior": float(os.getenv("TH_BEHAVIOR", "0.5")),
}

# 非确定性处理规范：常规 N=1；稳定性专项设 EVAL_RUNS=3~5
RUNS_PER_CASE = int(os.getenv("EVAL_RUNS", "1"))


def load_dataset(path: str | None = None) -> list[dict]:
    with open(path or DATASET_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
