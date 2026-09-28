"""检索诊断：区分「召回不足」与「排序不佳」。

用法（RAG_QA 目录、venv 下）：
    $env:EVAL_RERANK="on"; $env:EVAL_RECALL_K="30"; python tests/inspect_rank.py

对每条有真值文档的用例，输出：
    召回@K 是否含金标 → 金标名次 → 最终 top-k 是否命中
用于判断该调「召回 k / embedding」还是「reranker」。
"""
import rag_under_test as rag
from eval_config import RECALL_K, TOP_K, load_dataset
from runner import _expected_files


def _gold_rank(case, docs):
    expected = _expected_files(case)
    for i, d in enumerate(docs):
        src = str(d.metadata.get("source", "")).replace("\\", "/")
        if any(e in src for e in expected):
            return i
    return None


def main():
    rows = load_dataset()
    rag.ensure_ready()
    vectorstore = rag._retriever.vectorstore

    recall_ok = 0
    final_ok = 0
    considered = 0
    dropped = []
    rerank_state = "on" if rag._reranker is not None else "off"
    print(f"召回 K={RECALL_K} ｜ 最终 top_k={TOP_K} ｜ 重排={rerank_state}")
    for case in rows:
        if not _expected_files(case):
            continue
        considered += 1
        recalled = vectorstore.similarity_search(case["question"], k=RECALL_K)
        rank = _gold_rank(case, recalled)
        final_hit = _gold_rank(case, rag.retrieve(case["question"])) is not None
        recall_ok += rank is not None
        final_ok += final_hit
        if rank is not None and not final_hit:
            dropped.append((case["id"], rank))
    print(f"召回覆盖(金标在 recall K 内): {recall_ok}/{considered}")
    print(f"最终 top{TOP_K} 命中: {final_ok}/{considered}")
    if dropped:
        print(f"被排序丢弃(召回有、top{TOP_K} 无): {dropped}")


if __name__ == "__main__":
    main()
