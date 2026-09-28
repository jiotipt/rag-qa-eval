# RUNBOOK · 测试执行步骤

> 被测 RAG 评测的完整执行手册。所有命令在 PowerShell、`RAG_QA` 目录、已激活 venv 下执行。

## 0. 前置检查

```powershell
cd "rag-qa-eval"
.\.venv\Scripts\Activate.ps1
python --version                 # 3.12
python -c "import deepeval, langchain_openai, chromadb; print('deps ok')"
```
- LM Studio 已启动且加载 `qwen3.6-35b-a3b-ud`，`http://127.0.0.1:1234/v1` 可访问。

## 1. 建索引（知识库有更新时才需要重跑）

```powershell
python build_index.py
# 公开示例语料期望：加载文档 4 篇 → 切分 chunk 11 个 → embedding 进度条 → 索引已写入 chroma_db
# （作者内部语料口径为 28 篇 / 703 chunk，语料未随仓库发布）
```

## 2. 配置密钥（judge 用）

```powershell
# 若已用 setx 永久设置过，重开终端即可；否则临时设置：
$env:DEEPSEEK_API_KEY = "sk-你的新key"
```

## 3. 冒烟测试（先跑 1 条，确认链路通）

```powershell
python -m pytest tests/test_rag_quality.py -v -k S01
# 期望：1 passed
```

## 4. 全量跑批（默认样例集 12 条）

```powershell
python -m pytest tests/test_rag_quality.py -v
# 默认数据集 dataset_sample.csv（12 条 / 六类），结束打印：[汇总] 通过率 x/12
# 明细写入：eval_results.json
```

> 跑历史冻结集 `dataset_v1.csv`（50 条，需自备与之配套的语料并重建索引）：
> `$env:EVAL_DATASET="dataset_v1.csv"; python -m pytest tests/test_rag_quality.py -v`

## 5. 稳定性专项（可选，N=3）

```powershell
$env:EVAL_RUNS = "3"
python -m pytest tests/test_rag_quality.py -v
# 或改用 pytest-repeat：Remove-Item Env:EVAL_RUNS; python -m pytest tests/test_rag_quality.py --count=3
```

## 6. Judge 校验（配套历史冻结集 v1）

> 仓库已附完成的人工标注 `tests/human_labels.csv`（18 条，Q 编号，对应 `dataset_v1.csv`）。
> 该步骤只在 v1 流程下生效；默认样例集（S 编号）与标注无交集时用例自动 skip，不会报错。

1. （复现历史实验时）切换数据集：`$env:EVAL_DATASET="dataset_v1.csv"`，先跑第 4 步生成结果。
2. 运行：

```powershell
python -m pytest tests/test_judge_agreement.py -v
# 历史结果：一致率 77.8%、Cohen's κ=0.507（低于 0.8 为预期失败，用以暴露 judge 偏差）
```

## 7. 汇总与报告

- 结果明细：`eval_results.json`
- 按 `report.md` 骨架填：通过率 / 各指标均值 / 延迟 P50·P95 / 检索命中率 / 准出结论
- 冻结准出标准见 `report.md` §3；judge 校验方法见 `tests/test_judge_agreement.py`，人工标注见 `tests/human_labels.csv`（18 条，一致率 77.8%、κ=0.507，结论见 `report.md` §10）。

## 8. 双轨对比（D4）

被测对象可切换：`EVAL_TRACK=local`（LM Studio 本地轨）/ `deepseek`（API 对照轨）。
两轨共用同一检索层（同 embedding / 同库 / 同 top-k），只换生成模型 → 控制变量。
结果文件按轨分开：本地→`eval_results.json`；对照→`eval_results_deepseek.json`。

```powershell
# 本地轨
$env:EVAL_TRACK="local"
python -m pytest tests/test_rag_quality.py -v      # → eval_results.json
python tests/summarize.py

# 对照轨（被测模型 = DeepSeek；judge 也走 DeepSeek，注意自评偏差）
$env:EVAL_TRACK="deepseek"
$env:DEEPSEEK_API_KEY = "sk-..."
python -m pytest tests/test_rag_quality.py -v      # → eval_results_deepseek.json
python tests/summarize.py

Remove-Item Env:EVAL_TRACK                          # 用完复位，避免影响后续
```

## 9. badcase 清单（D4）

```powershell
python tests/badcases.py                 # 生成 badcases.md（检索/生成/行为 三层归因）
# 对对照轨：$env:EVAL_TRACK="deepseek"; python tests/badcases.py badcases_deepseek.md
```

## 10. 回归演示（D4）

```powershell
Copy-Item eval_results.json eval_results_before.json   # 改前留档
# …只改 1 个变量（改 prompt 第 1 条 / 调 top-k）…
python -m pytest tests/test_rag_quality.py -v          # 重跑
python tests/regress.py eval_results_before.json eval_results.json regression.md
```

## 11. 安全红队（D4）

见 `promptfoo/README.md`（提示注入 / 越狱 / PII 泄露，接本地 OpenAI 兼容端点）。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `未找到 DEEPSEEK_API_KEY` | 终端未继承变量 | 重开终端，或临时 `$env:DEEPSEEK_API_KEY=...` |
| 回答为空 | 本地思考模型未关思考 | 确认 `rag_under_test.py` 里 `reasoning_effort="none"` |
| `ModuleNotFoundError` | venv 未激活/缺包 | `pip install -r requirements.txt` |
| 检索命中率全 0 | 索引与 embedding 不一致 | 重建索引（第 1 步）；确认 embedding 模型一致 |
| 指标报 API 错误 | judge key 或网络 | 检查 key、DeepSeek 余额/限流 |
| 单个 case 超时 | 本地模型慢 | 单独重跑该 id；必要时 `-k Qxxx` |
