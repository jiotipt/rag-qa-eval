import sys

import rag_under_test
from eval_config import load_dataset


def _short_source(meta_value) -> str:
    src = str(meta_value or "").replace("\\", "/")
    if "/kb_docs/" in src:
        return src.split("/kb_docs/")[-1]
    return src


def main():
    ids = sys.argv[1:] or ["Q002"]
    cases = {c["id"]: c for c in load_dataset()}
    rag_under_test.ensure_ready()
    for cid in ids:
        case = cases.get(cid)
        if not case:
            print(f"[{cid}] 不存在")
            continue
        docs = rag_under_test.retrieve(case["question"])
        joined = "\n".join(d.page_content for d in docs)
        key = case["ground_truth_answer"][:12]
        print("=" * 70)
        print(f"[{cid}] 类别={case['category']} 期望行为={case['expected_behavior']}")
        print(f"问题: {case['question']}")
        print(f"期望答案: {case['ground_truth_answer']}")
        print(f"期望来源: {case['ground_truth_context']}")
        print(f"答案关键片段是否出现在召回文本中: {'✅ 是' if key in joined else '❌ 否'}  (key='{key}')")
        print(f"\n召回 {len(docs)} 段:")
        for i, d in enumerate(docs, 1):
            print(f"\n--- 片段{i} 来源: {_short_source(d.metadata.get('source'))}")
            print(d.page_content[:1000])
        print()


if __name__ == "__main__":
    main()
