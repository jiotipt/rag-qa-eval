# rag-qa-eval · RAG 问答质量评测套件

> 用 **pytest + DeepEval** 为一套 RAG 知识库问答搭建质量保障体系：分层测试集、指标断言、LLM-as-judge 及其人工校验、非确定性处理（N=3 稳定性）、双轨对比、badcase 三层归因、安全红队。

非确定性系统没有"唯一期望值"，传统 `assert` 失效。本项目把「质量」量化为**可复现的指标**，并给出**可判定的准出标准**。

---

## 这是什么

- **被测对象**：一个知识库问答 RAG（向量检索 + LLM 生成）。`kb_docs/` 随仓库提供一套**原创、虚构的「OfficeCraft 办公技能集」示例语料**（会议纪要 / 周报 / PPT 大纲，共 4 篇，MIT 协议），可直接建索引跑通全流程；替换为你自己的文档即可评测你的 RAG。
- **本项目不实现业务逻辑**，只提供**评测 / 准出 / 回归**能力：给定测试集，输出通过率、各项指标、延迟、检索命中率，并判定是否达标。

## 快速开始

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt

python download_model.py      # 下载 bge-m3 + bge-reranker-v2-m3 到 models/
python build_index.py         # 文档 → 切块(400/120) → 向量库 chroma_db/
                              # 示例语料期望输出：加载文档 4 篇 → 切分 chunk 11 个

# 起一个 OpenAI 兼容端点（示例：LM Studio 加载 qwen3.6-35b-a3b-ud @ http://127.0.0.1:1234/v1）
# 设置 judge 密钥（DeepSeek）
export DEEPSEEK_API_KEY=sk-xxxx        # Windows: $env:DEEPSEEK_API_KEY="sk-xxxx"

python -m pytest tests/test_rag_quality.py -v     # 全量跑批（默认 dataset_sample.csv，12 条）
python tests/summarize.py                          # 汇总
python tests/check_retrieval.py                   # 只验检索层（不调 LLM、不花 token）
```

> 完整步骤（建索引 / 冒烟 / 双轨 / 回归 / 红队 / 稳定性）见 [`RUNBOOK.md`](RUNBOOK.md)。

## 关键设计

| 维度 | 做法 |
|---|---|
| 测试集 | `dataset_sample.csv`：**12 条 / 6 类**（正常·模糊·多跳·对抗·越界·追问），默认数据集，配套公开示例语料开箱可跑；`dataset_v1.csv`：50 条历史冻结集（配套原始内部语料，未随仓库发布）。规范见 `dataset_v1_标注规范.md` |
| 指标 | Answer Relevancy / Faithfulness / Contextual Precision（工具指标）+ 拒答·澄清 GEval 行为判定 + 检索命中率 |
| 断言 | 用例级「全部指标 ≥ 阈值」= 通过（**不用字符串相等**） |
| 非确定性 | `temperature=0`；常规 N=1，稳定性 N=3（`EVAL_RUNS=3`），按**每用例多数投票**判「稳定通过」 |
| judge 校验 | 人工标注 → 与 judge 一致率 + Cohen's κ |
| 双轨对比 | `EVAL_TRACK=local\|deepseek`：同检索层，只换生成模型（控制变量） |
| 检索 | 两阶段 = 向量召回 `k=10` → `bge-reranker-v2-m3` 重排 top-3（`EVAL_RERANK=on`） |

## 结果（仓库自带证据）

> 以下结果基于作者的**原始内部语料 + 50 条冻结测试集**（`dataset_v1.csv`），该语料因版权原因未随仓库发布；`eval_results*.json` 与各报告 md 保留为**历史实验证据**。公开仓库自带的 12 条样例（`dataset_sample.csv` + OfficeCraft 语料）用于端到端复现评测流程，检索层实测 12/12 命中，质量跑批结果以你本地实测为准。

| 版本 | 通过率 | 说明 |
|---|---|---|
| v1 冻结基线 | 72%（36/50） | 向量 top-3 |
| v2b 两阶段检索 | 86%（43/50，单跑） | 召回 10 → 重排 top-3 |
| **N=3 稳定值** | **84%（42/50）** | 剔除运气（3 次中 ≥2 次通过） |

- 逐条对比：`regression_*.md`；badcase 三层归因：`badcases.md`；稳定性报告：`stability_v2b_n3.md`；测试报告：`report.md`。
- 原始明细：`eval_results*.json`。

## 目录结构

```
rag-qa-eval/
├── build_index.py            # 文档 → 切块 → 向量库
├── rag.py                    # 检索 + 生成（示例）
├── download_model.py         # 下载 embedding / reranker 模型
├── requirements.txt
├── pytest.ini
├── RUNBOOK.md                # 操作手册（按顺序执行）
├── LICENSE                   # MIT（示例语料同协议）
├── report.md                 # 测试报告（历史实验证据）
├── dataset_sample.csv        # 默认测试集（12 条，配套公开示例语料）
├── dataset_v1.csv            # 历史冻结测试集（50 条，配套未发布的内部语料）
├── dataset_v1_标注规范.md     # 标注规范
├── eval_results*.json        # 各版本原始结果（证据）
├── regression_*.md           # 前后对比
├── badcases*.md              # badcase 归因
├── stability_v2b_n3.md       # N=3 稳定性报告
├── kb_docs/                  # 原创虚构示例知识库（OfficeCraft，4 篇，MIT）
├── promptfoo/                # 安全红队配置与结果
└── tests/
    ├── eval_config.py        # 阈值 / 模型 / 路径配置
    ├── rag_under_test.py     # 被测 RAG 适配器（两阶段检索）
    ├── runner.py             # 单条评测逻辑（解耦字段 + 重试）
    ├── conftest.py           # judge / 参数化 / 结果汇总
    ├── test_rag_quality.py   # 主套件
    ├── test_judge_agreement.py
    ├── stability.py          # N=3 多数投票聚合
    ├── summarize.py / regress.py / badcases.py
    ├── check_retrieval.py / inspect_rank.py / diagnose.py / inspect_case.py
    └── human_labels.csv      # judge 校验的人工标注
```

## 技术栈

Python 3.12 · pytest · DeepEval · LangChain · ChromaDB · sentence-transformers（bge-m3 / bge-reranker-v2-m3）· LM Studio / DeepSeek（OpenAI 兼容）

## 说明

- **不包含模型权重**（4GB+）：用 `python download_model.py` 获取。
- **不包含向量库**：用 `python build_index.py` 重建。
- **不包含任何密钥 / 个人数据**。
- `kb_docs/` 为原创虚构示例知识库（MIT），可自由使用；自用时替换为你自己的文档并重建索引（chunk 参数可能需按语料调整）。
- 历史结果（72%/86%/84%）基于未发布的内部语料与 `dataset_v1.csv`；公开样例（`dataset_sample.csv`）用于复现流程与方法，不与历史分数直接比较。
