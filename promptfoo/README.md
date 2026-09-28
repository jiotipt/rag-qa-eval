# D4 · promptfoo 安全红队

> 目标：对 RAG 问答做一轮**安全红队**——提示注入 / 越狱 / PII 泄露，产出「成功攻击样例 + 护栏建议」。

## 1. 定位与局限（面试必讲）

> 复用到你自己的知识库时，先改 `promptfooconfig.yaml` 里的 `redteam.purpose`、`entities` 和 `prompts`（当前配置针对作者原始的"视频生成 Skill"语料）；`redteam.yaml` 与 `redteam结果-D4.md` 为历史实验证据。

- 本红队测试的是 **「被测模型 + 系统提示」** 这一层，**不包含** RAG 检索链路（promptfoo 不会真的去 ChromaDB 召回）。
  - 因此它是**安全冒烟测试**，不是端到端 RAG 安全评测。
  - 想端到端，可改用 promptfoo 的 HTTP/custom provider 指向你的 RAG 服务；本项目保持「模型+提示」层，够证明方法。
- 生成/评分对抗样本用的是 **DeepSeek**（不是被测本地模型），避免「自己出题自己判」。

## 2. 前置

- Node ≥ 18（建议 20+）；用 `npx promptfoo@latest` 免全局安装。
- LM Studio 已启动、加载 `qwen3.6-35b-a3b-ud`，`http://127.0.0.1:1234/v1` 可访问。
- 环境变量设 `DEEPSEEK_API_KEY`（对抗样本生成/评分用）。

## 3. 文件

| 文件 | 说明 |
|---|---|
| `promptfooconfig.yaml` | 红队配置：目标(本地)、插件(注入/越狱/PII/RAG)、策略、purpose |
| `redteam.yaml` | 已生成的 18 条探针用例（可直接复跑） |
| `redteam结果-D4.md` | 本轮实测结果：17 通过 / 1 被攻破 + 护栏建议 + 局限 |
| `README.md` | 本文件：如何跑、如何读结果 |

## 4. 运行

```powershell
cd "rag-qa-eval\promptfoo"
$env:DEEPSEEK_API_KEY = "sk-..."

# 生成对抗样本并执行（= generate + eval 一步到位）
npx promptfoo@latest redteam run

# 查看报告（浏览器）
npx promptfoo@latest redteam report
```

产物：`redteam.yaml`（生成的用例）、`output/`（结果）、浏览器报告。

> 若配置字段报错（promptfoo 版本差异）：先 `npx promptfoo@latest redteam init` 生成官方最新模板，
> 再把本目录 `promptfooconfig.yaml` 里的 `targets` 与 `redteam.plugins` 段迁过去。

## 5. 时间控制（对齐 D4 止损）

- 时间紧：`numTests` 降到 **3**，`plugins` 只留 注入(`indirect-prompt-injection`,`system-prompt-override`) + 越狱(`pliny`) + PII(`pii:direct`)，共 4 个插件。
- 时区/网络：生成需要 DeepSeek 网络；被测本地模型不耗 API。

## 6. 怎么读结果（写进报告）

对每个失败（=被攻破）的用例，记录：

| 字段 | 内容 |
|---|---|
| 插件 / 策略 | 哪类攻击（如 `pliny` + `jailbreak`） |
| 攻击输入 | 生成的恶意 prompt（可脱敏） |
| 模型输出 | 是否泄露系统提示 / 越界作答 / 输出 PII |
| 护栏建议 | 如：系统提示加"拒绝套取系统提示"、输出侧加 PII 过滤、检索侧限制可导出文档 |

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 被测输出为空 | 本地思考模型未关思考 | 配置里 `passthrough: {reasoning_effort: none}` 已带；确认 LM Studio 端点/模型名 |
| 报缺少 `OPENAI_API_KEY` | 生成器默认用 openai | 已把 `redteam.provider` 指为 DeepSeek；确认 `DEEPSEEK_API_KEY` 已设 |
| 连接被拒 | LM Studio 未开 | 起服务并加载模型 |
| 字段/插件名报错 | promptfoo 版本差异 | 跑 `redteam init` 对齐官方模板 |
