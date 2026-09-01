# Text-to-SQL 大模型后训练实战——项目启动说明

> 项目定位：以公开、可执行的 Text-to-SQL 数据集为载体，完整练习开源大模型的环境搭建、Base 模型推理、LoRA/QLoRA SFT、GRPO/RLVR、统一评测和项目复盘。
>
> 项目性质：个人学习与相关实习求职准备项目。核心目标是掌握并能讲清 SFT 与 GRPO 后训练流程，形成有真实代码、日志、指标和失败分析的项目证据，而不是开发完整业务产品，也不是让 Codex 一次性生成整个项目。
>
> 建议周期：4～6 周。先完成最小闭环，再逐步扩大数据量和训练规模。
>
> 数据默认方案：优先直接使用 CSpider；若其下载或授权条件不适合，则使用 Spider 1.0。Olist 等领域数据只作为后续可选扩展，不是项目启动前置条件。

---

## 1. 项目核心目的

本项目不以开发完整产品、垂直业务系统或 Agent 系统为目标，而以掌握大模型后训练工程流程为目标。Text-to-SQL 的价值在于：输出可以通过 SQLite 真实执行，从而为 SFT 评测和 GRPO/RLVR 提供可验证信号。

最终需要真正掌握并能在面试中解释：

1. 如何检查 CUDA、PyTorch、Transformers 和 GPU 环境；
2. 如何加载、量化和部署一个 Qwen 系列开源模型；
3. 模型输入如何经过 Tokenizer、Embedding、Transformer Blocks 和 LM Head；
4. Base 模型在未训练前能达到什么水平；
5. 如何把公开 Text-to-SQL 数据转换、清洗和验证为 SFT/GRPO 数据；
6. LoRA/QLoRA 如何减少训练参数量和显存占用；
7. SFT Loss 如何计算，哪些 Token 参与 Loss；
8. 如何比较 Base、SFT 和 GRPO 模型的真实能力；
9. GRPO 如何生成多个 Completion、计算 Reward 和相对优势；
10. 如何设计基于 SQL 执行结果的可验证 Reward；
11. 如何记录实验、定位错误并形成可复现结论；
12. 如何根据真实代码、日志和实验结果回答项目面试问题，并说明工程取舍和实验局限。

面向实习求职，项目最终需要形成以下可展示证据：

- 一套可复现的 Base → SFT → GRPO 实验链路；
- SFT 数据格式、Label Mask、LoRA/QLoRA 配置和训练曲线；
- 可单元测试的 SQL 执行器、结果比较器和 GRPO Reward；
- Base、SFT、GRPO 使用同一测试集的对比结果；
- 显存、耗时、Checkpoint、错误样本和失败实验记录；
- 能由本人逐段解释的核心训练代码与项目复盘。

项目的主线是：

```text
环境与算力检查
→ 公开数据集下载、转换与验证
→ 模型架构解析
→ Base 模型部署与推理
→ Baseline 评测
→ LoRA/QLoRA SFT
→ SFT 推理与评测
→ GRPO/RLVR
→ GRPO 推理与评测
→ 最终部署、总结与模拟面试
```

---

## 2. 项目范围

### 2.1 要做的内容

- 输入数据库 Schema 和自然语言问题，生成可执行的 SQLite SQL；
- 直接接入带问题、Gold SQL 和 SQLite 数据库的公开数据集；
- 使用统一测试集对 Base、SFT、GRPO 三个阶段进行评测；
- 使用 LoRA 或 QLoRA 完成监督微调；
- 使用 SQL 执行结果构造 GRPO/RLVR Reward；
- 保存每个阶段的配置、Checkpoint、预测结果、指标和错误样本；
- 最后提供 CLI 或轻量 API 推理入口；
- 根据项目产物生成模拟面试问题。

### 2.2 暂时不做的内容

- 不实现多步自主 Agent Loop；
- 不加入复杂 Tool Calling、规划、记忆和停止策略；
- 不开发完整 Web 前端；
- 不接入真实企业数据库；
- 不把自建电商数据库和大规模人工造数作为前置任务；
- 不追求覆盖某个垂直行业的全部业务口径；
- 不进行大规模分布式训练；
- 不为了丰富技术栈加入 RAG、LangChain、CrewAI 等非必要模块；
- 不要求 GRPO 必须优于 SFT，允许得到可信的负面结果。

### 2.3 系统边界

模型负责：

```text
Natural Language Question + Database Schema → SQL
```

模型外部程序负责：

```text
SQL 提取 → 安全检查 → SQLite 执行 → 结果比较 → 指标/Reward
```

数据库执行器只是训练与评测环境，不是 Agent 工具。

---

## 3. Codex 协作原则

Codex 用于降低重复工程成本，但不能代替本人理解项目。

### 3.1 Codex可以做什么

- 按当前阶段创建目录、配置、脚本和单元测试；
- 解释生成代码的输入、输出和调用关系；
- 逐段讲解训练循环、Loss、LoRA 和 GRPO 代码；
- 帮助生成数据处理脚本和自然语言改写；
- 检查数据泄漏、格式错误和 SQL 执行失败；
- 分析日志、报错、显存和实验结果；
- 帮助撰写阶段报告、README 和模拟面试问题。

### 3.2 Codex不能直接替代什么

- 不一次性完成所有阶段；
- 不跳过 Base 模型实验直接开始微调；
- 不虚构训练指标、显存、耗时或提升幅度；
- 不把未经执行验证的 SQL 当作 Gold SQL；
- 不在本人未读核心代码前自动进入下一阶段；
- 不替本人决定实验结论；
- 不自动运行长时间训练，除非当前阶段明确授权。

### 3.3 每阶段推荐的协作提示词

```text
请阅读 PROJECT_KICKOFF.md，只完成阶段 X，不实现后续阶段。
先检查上一阶段的退出条件，再列出本阶段将新增或修改的文件。
实现后运行 smoke test，并解释：
1. 数据如何流动；
2. 核心代码如何计算；
3. 我需要亲自执行哪些命令；
4. 应观察哪些指标；
5. 哪些结果必须写入项目日志。
不要代替我运行正式长训练，也不要虚构结果。
```

---

## 4. 技术路线与默认选择

| 模块 | 默认选择 | 说明 |
|---|---|---|
| Base Model | Qwen3-4B-Instruct-2507 | RTX 3090 24GB 服务器候选；服务器 smoke test 后正式锁定 |
| 推理 | Transformers | 首先理解标准加载与生成流程 |
| 量化 | bitsandbytes 4-bit | 适配 11GB 显存 |
| SFT | Transformers + PEFT + TRL | QLoRA 优先 |
| GRPO | TRL GRPOTrainer | 先用小数据和小模型 smoke test |
| 数据库 | SQLite | 方便隔离、复制和执行验证 |
| 配置 | YAML | 禁止训练参数散落硬编码 |
| 日志 | JSONL/CSV + Markdown | 可选接入 W&B，但本地日志必须完整 |
| 测试 | pytest | 覆盖数据、SQL执行、Reward和配置 |
| 最终服务 | CLI，FastAPI 可选 | vLLM 仅在硬件和模型兼容时加入 |

模型一旦在阶段 0 锁定，Base、SFT、GRPO 必须使用同一个 Base Model 和同一测试集，否则结果不可直接比较。

---

## 5. 推荐项目结构

```text
llm-text2sql-posttraining/
├── README.md
├── PROJECT_KICKOFF.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── model.yaml
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── serving.yaml
│
├── data/
│   ├── raw/
│   │   ├── cspider/
│   │   └── spider/
│   ├── interim/
│   ├── processed/
│   │   ├── sft_train.jsonl
│   │   ├── sft_validation.jsonl
│   │   ├── grpo_train.jsonl
│   │   └── test.jsonl
│   └── databases/
│       └── <db_id>/<db_id>.sqlite
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── convert.py
│   │   ├── compute_gold_results.py
│   │   ├── clean.py
│   │   ├── split.py
│   │   └── validate.py
│   ├── model/
│   │   ├── load.py
│   │   └── inspect_architecture.py
│   ├── inference/
│   │   ├── generate.py
│   │   └── sql_parser.py
│   ├── execution/
│   │   ├── sqlite_executor.py
│   │   └── result_normalizer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   ├── sft/
│   │   ├── dataset.py
│   │   └── train.py
│   ├── grpo/
│   │   ├── rewards.py
│   │   └── train.py
│   ├── serving/
│   │   ├── cli.py
│   │   └── api.py
│   └── utils/
│       ├── check_env.py
│       ├── logging.py
│       ├── seed.py
│       └── memory_estimator.py
│
├── scripts/
│   ├── bootstrap.sh
│   ├── check_env.sh
│   ├── prepare_data.sh
│   ├── inspect_model.sh
│   ├── run_base_inference.sh
│   ├── evaluate_base.sh
│   ├── train_sft.sh
│   ├── evaluate_sft.sh
│   ├── train_grpo.sh
│   ├── evaluate_grpo.sh
│   ├── serve.sh
│   └── generate_interview.sh
│
├── tests/
│   ├── test_data.py
│   ├── test_sql_executor.py
│   ├── test_metrics.py
│   ├── test_sft_format.py
│   └── test_rewards.py
│
├── outputs/
│   ├── baseline/
│   ├── sft/
│   ├── grpo/
│   ├── checkpoints/
│   └── final_comparison/
│
└── docs/
    ├── architecture_notes.md
    ├── data_report.md
    ├── experiment_log.md
    ├── issue_log.md
    ├── compute_report.md
    ├── final_report.md
    └── interview_questions.md
```

所有模块通过配置和稳定的数据结构连接，后续阶段不得直接读取上一阶段的内部临时变量。

---

## 6. 核心接口约定

接口应在早期确定，后续只扩展、不随意修改。

### 6.1 标准任务样本

以下数值只用于展示字段结构，实际 `gold_result` 必须由对应数据库执行 `gold_sql` 后写入：

```json
{
  "id": "cspider_dev_0001",
  "question": "年龄超过56岁的负责人有多少位？",
  "db_id": "department_management",
  "db_path": "data/databases/department_management/department_management.sqlite",
  "schema": "CREATE TABLE ...",
  "gold_sql": "SELECT count(*) FROM head WHERE age > 56;",
  "gold_result": [[1]],
  "task_type": "aggregation",
  "difficulty": "easy",
  "source": "cspider",
  "split": "test"
}
```

### 6.2 标准模型输出

模型训练目标只输出 SQL：

```text
SELECT ...;
```

解析器统一转换为：

```json
{
  "raw_output": "SELECT ...;",
  "parsed_sql": "SELECT ...;",
  "format_valid": true
}
```

### 6.3 标准预测记录

```json
{
  "id": "cspider_dev_0001",
  "stage": "base",
  "model": "locked-model-name",
  "prompt": "...",
  "raw_output": "...",
  "predicted_sql": "...",
  "execution_success": true,
  "predicted_result": [[1]],
  "gold_result": [[1]],
  "execution_correct": true,
  "latency_seconds": 1.23,
  "input_tokens": 420,
  "output_tokens": 55
}
```

### 6.4 标准训练产物

每个训练阶段必须输出：

```text
checkpoint/
adapter/
trainer_state.json
training_metrics.json
resolved_config.yaml
environment.json
predictions.jsonl
evaluation_metrics.json
error_cases.jsonl
```

---

## 7. 数据策略

### 7.1 原则

训练数据优先来自已经公开、包含自然语言问题、Gold SQL、Schema 和 SQLite 数据库的数据集。项目不把“寻找原始业务数据并从零造问题”作为主要工作，人工时间集中投入到：

- 数据格式转换；
- 数据清洗；
- SQL执行验证；
- 数据库级划分和泄漏检查；
- 错误样本抽查；
- 高质量测试集审核。

人工不应把大量时间花在逐条编写普通训练样本上。第一版正式实验不依赖 GPT 批量生成或翻译数据。

### 7.2 推荐来源

按以下优先级选择，第一版只锁定一个主数据源：

1. **CSpider（默认）**：中文问题、Gold SQL 和 Spider SQLite 数据库，适合中文 Text-to-SQL 后训练练习；只使用其数据，不复用陈旧的训练代码。使用前记录来源与可用条款，不在未确认许可时重新分发原始数据。
2. **Spider 1.0（后备）**：英文跨领域 Text-to-SQL，数据与 SQLite 数据库完整，公开许可清晰，适合完整的执行评测与 GRPO Reward。
3. **Verified Text-to-SQL 1.5K（可选 SFT 热身）**：体积小、格式直接，但公开训练行没有稳定对应的数据库快照时，不作为 GRPO 或最终执行评测数据。
4. **BIRD（扩展实验）**：更真实也更大，下载和处理成本明显更高，不作为第一版前置条件。
5. **Olist 或其他领域数据（可选）**：只有在主流程完成后，才用于少量领域适配实验；不影响本项目完成判定。

无论来源如何，所有 Gold SQL 都要在对应 SQLite 数据库上重新执行，Gold Result 只能由程序生成，禁止由 GPT 编造。

### 7.3 数据使用与生成边界

第一版可以由 Codex 自动完成：

- 下载/导入脚本；
- 原始格式到 canonical、SFT、GRPO 格式的转换；
- Schema 序列化；
- Gold SQL 执行和 Gold Result 生成；
- 难度及 SQL 特征统计；
- 数据清洗脚本和验证脚本。

必须由程序或人工验证：

- 表之间的主外键关系；
- Gold SQL 是否可执行；
- Gold Result 是否正确；
- 官方数据是否存在已知错误或方言兼容问题；
- 训练数据库与测试数据库是否泄漏；
- 最终测试集质量。

问题改写、翻译、模板扩增和领域数据生成全部属于后续可选实验，不进入第一版关键路径。

### 7.4 数据规模采用逐级扩大

| 用途 | Smoke Test | 第一版正式实验 | 扩展实验 |
|---|---:|---:|---:|
| SFT Train | 50～100 | 800～2,000 | 3,000+ |
| Validation | 20～30 | 100～200 | 300+ |
| GRPO Train | 20～50 | 200～500 | 1,000+ |
| Test | 30～50 | 200～300 | 500+ |

禁止在 50 条数据都不能正常训练和评测时直接导入几千条数据。正式规模优先从主数据集确定性抽样，而不是重新生成。

### 7.5 防止数据泄漏

- 优先沿用数据集官方数据库级 train/dev 划分；
- 官方 dev 作为锁定测试集，从 train 中按 `db_id` 划出 validation；
- 同一数据库不得同时出现在 train、validation 和 test；
- 如果以后生成同义改写，同一 Gold SQL 的改写不得跨 split；
- 测试集一旦锁定，后续不得根据测试结果修改训练数据；
- 数据处理后生成哈希和统计报告。

---

## 8. 分阶段实施计划

### 阶段 0：环境、模型与算力可行性检查

### 目标

在写训练代码前确认模型、精度、量化方式和 GPU 是否可行，避免完成代码后才发现显存不足。

### Codex任务

- 创建最小项目骨架；
- 创建环境检查和显存估算脚本；
- 输出 Python、PyTorch、CUDA、GPU、驱动和磁盘信息；
- 创建模型加载 smoke test；
- 不进入正式推理或训练。

### 本人需要理解

- GPU显存不等于系统内存；
- 多张11GB显卡不会自动变成一张44GB显卡；
- FP32、FP16和4-bit参数显存估算；
- 推理显存还包括 KV Cache 和临时激活；
- 训练显存还包括梯度、优化器状态和激活。

### 启动命令

```bash
bash scripts/bootstrap.sh
bash scripts/check_env.sh
```

对应 Python 命令：

```bash
python -m src.utils.check_env
python -m src.utils.memory_estimator --config configs/model.yaml
```

### 退出条件

- 环境检查通过；
- 模型成功完成一次最小加载和生成；
- `docs/compute_report.md`记录模型、精度、显存峰值和可行配置；
- 锁定后续 Base/SFT/GRPO 使用的模型版本；
- 如果 4B GRPO 不可行，同时锁定一个小模型用于全流程练习。

---

### 阶段 1：公开数据集获取、转换与验证

### 目标

以 CSpider 为默认主数据源，建立“下载/导入 → 格式转换 → SQL 执行 → 划分 → 验证”的可复现流水线，不从零手工编写训练集。

### Codex任务

- 编写 CSpider/Spider 下载或本地导入、来源记录和文件校验脚本；
- 将原始 JSON、Gold SQL、Schema 和数据库路径转换为统一 canonical task；
- 编写 Gold SQL 执行和 Gold Result 缓存脚本；
- 编写按 `db_id` 划分、去重、错误过滤和泄漏检查脚本；
- 从公开数据中确定性抽取 smoke 子集，不生成正式规模的新问题。

### 本人需要理解

- 原始数据、canonical task、SFT、GRPO 和测试样本之间的区别；
- Schema如何放入Prompt；
- Gold SQL与Gold Result分别有什么作用；
- 为什么Gold Result不能由GPT直接生成；
- 为什么跨数据库 Text-to-SQL 必须按 `db_id` 控制划分；
- 为什么公开 benchmark 仍需重新执行和检查 Gold SQL。

### 启动命令

```bash
bash scripts/prepare_data.sh --mode smoke
bash scripts/prepare_data.sh --mode full
```

对应 Python 命令：

```bash
python -m src.data.download --config configs/data.yaml
python -m src.data.convert --config configs/data.yaml --mode smoke
python -m src.data.compute_gold_results --config configs/data.yaml
python -m src.data.clean --config configs/data.yaml
python -m src.data.split --config configs/data.yaml
python -m src.data.validate --config configs/data.yaml
```

### 退出条件

- 数据来源、版本、许可说明和文件哈希已记录；
- 保留样本的 Gold SQL 均可执行，不兼容样本有明确拒绝原因；
- 数据文件满足标准接口；
- 重复、空结果、错误 SQL 和 `db_id` 划分泄漏均有统计；
- 人工抽查测试集；
- 生成`docs/data_report.md`。

---

### 阶段 2：模型架构解析

### 目标

在训练前看懂模型由什么组成，以及输入输出张量如何变化。

### 必须学习和记录

- Tokenizer与Chat Template；
- Token ID和Attention Mask；
- Embedding层；
- Transformer Block数量；
- RMSNorm；
- Q/K/V投影、Multi-Head Attention或GQA；
- RoPE；
- MLP/Gated MLP；
- Residual Connection；
- LM Head；
- Logits形状和Next-token Prediction；
- 总参数量与可训练参数量。

### 启动命令

```bash
bash scripts/inspect_model.sh
```

对应 Python 命令：

```bash
python -m src.model.inspect_architecture --config configs/model.yaml
```

### 退出条件

- `docs/architecture_notes.md`包含架构图或模块树；
- 能解释一次Prompt如何变成下一个Token；
- 能说明模型参数、激活和KV Cache分别占用什么资源；
- 能解释为什么训练时需要Shift Labels。

---

### 阶段 3：Base模型部署、推理与Baseline记录

### 目标

使用未微调模型完成真实推理，保存后续比较所需的Baseline。

### 实验设置

- 固定模型版本；
- 固定Prompt模板；
- 固定测试集；
- 固定解码参数；
- 默认使用确定性解码进行正式评测；
- 先单样本推理，再批量评测。

### 启动命令

```bash
bash scripts/run_base_inference.sh --mode single
bash scripts/evaluate_base.sh
```

对应 Python 命令：

```bash
python -m src.inference.generate \
  --config configs/baseline.yaml \
  --stage base \
  --question "过去30天销售额最高的5个商品是什么？"

python -m src.evaluation.evaluate \
  --config configs/baseline.yaml \
  --stage base \
  --output-dir outputs/baseline
```

### 必须记录

- SQL格式正确率；
- SQL可执行率；
- Execution Accuracy；
- 简单/中等/困难任务准确率；
- 平均输入输出Token；
- 平均延迟；
- GPU峰值显存；
- 典型成功与失败样本。

### 退出条件

- `outputs/baseline/`产物完整；
- `docs/experiment_log.md`已有Baseline条目；
- 能解释Base模型主要失败类型；
- 不修改测试集来迎合模型结果。

---

### 阶段 4：LoRA/QLoRA SFT实现与Smoke Test

### 目标

先验证数据、Mask、Loss和Checkpoint链路，再运行正式微调。

### 必须理解的计算原理

#### SFT Loss

模型对每个位置预测下一个Token：

```text
input_ids[:, :-1] → labels[:, 1:]
```

使用Token级交叉熵：

```text
L = - Σ log p(y_t | y_<t, x)
```

如果采用assistant-only loss，System和User部分的Label设为`-100`，不参与Loss。

#### LoRA

冻结原权重`W`，只训练低秩增量：

```text
W' = W + (α/r)BA
```

必须记录LoRA注入层、rank、alpha、dropout和可训练参数比例。

#### QLoRA

Base权重量化为4-bit保存，LoRA参数仍以较高精度训练，以降低显存占用。

### 启动命令

```bash
bash scripts/train_sft.sh --mode smoke
```

对应 Python 命令：

```bash
python -m src.sft.train \
  --config configs/sft.yaml \
  --max-steps 10 \
  --output-dir outputs/checkpoints/sft_smoke
```

### Smoke退出条件

- 能完成前向、反向和参数更新；
- Loss为有限值并能正常记录；
- 只有目标LoRA参数可训练；
- Checkpoint可以保存并重新加载；
- 随机抽查确认只对Assistant SQL计算Loss；
- 记录实际显存峰值。

---

### 阶段 5：正式SFT、推理与评测

### 目标

完成第一版正式微调，并与Base模型进行同条件对比。

### 启动命令

```bash
bash scripts/train_sft.sh --mode full
bash scripts/evaluate_sft.sh
```

对应 Python 命令：

```bash
python -m src.sft.train \
  --config configs/sft.yaml \
  --output-dir outputs/checkpoints/sft

python -m src.evaluation.evaluate \
  --config configs/baseline.yaml \
  --stage sft \
  --adapter-path outputs/checkpoints/sft/best_adapter \
  --output-dir outputs/sft
```

### 必须记录和分析

- Train Loss和Validation Loss；
- 学习率曲线；
- 是否过拟合；
- Base与SFT的Execution Accuracy差异；
- 不同SQL难度的变化；
- 格式正确率与可执行率变化；
- SFT新引入的失败类型；
- 最佳Checkpoint选择依据。

### 退出条件

- SFT Adapter可独立加载；
- 使用与Baseline完全相同的测试集和解码配置；
- 形成Base vs SFT对比表；
- 可以逐段解释`src/sft/train.py`；
- 在日志中写明有效结论和不确定结论。

---

### 阶段 6：GRPO/RLVR环境与Reward验证

### 目标

先证明Reward正确，再消耗GPU训练。

### GRPO训练过程

```text
同一个Prompt
→ 当前策略生成G个SQL
→ SQL格式检查和数据库执行
→ 每个SQL获得Reward
→ 组内Reward标准化得到相对优势
→ 更新策略模型
```

简化的组内优势：

```text
A_i = (r_i - mean(r_1...r_G)) / (std(r_1...r_G) + ε)
```

如果同一组所有输出Reward相同，优势接近0，该Prompt几乎不产生学习信号。

### Reward优先级

核心Reward：

```text
生成SQL的执行结果是否与Gold Result一致
```

辅助Reward可以包括：

- 输出格式正确；
- SQL能够执行；
- 禁止写操作；
- 查询结果部分匹配；
- 轻微长度惩罚。

禁止把SQL字符串相似度作为主要正确性Reward，因为不同SQL可能得到相同正确结果。

### Reward测试命令

```bash
pytest -v tests/test_rewards.py
```

对应直接检查：

```bash
python -m src.grpo.rewards --config configs/grpo.yaml --self-test
```

### 退出条件

- Gold SQL得到最高正确性Reward；
- 等价但不同写法的SQL也能获得正确Reward；
- 错误SQL、危险SQL、空输出有明确结果；
- Reward测试全部通过；
- Reward各分量分开记录，不能只记录总分。

---

### 阶段 7：GRPO Smoke Test、正式训练与评测

### 目标

先完成小模型/小数据GRPO闭环，再判断是否扩大训练。

### 启动命令

```bash
bash scripts/train_grpo.sh --mode smoke
bash scripts/train_grpo.sh --mode full
bash scripts/evaluate_grpo.sh
```

对应 Python 命令：

```bash
python -m src.grpo.train \
  --config configs/grpo.yaml \
  --max-steps 5 \
  --output-dir outputs/checkpoints/grpo_smoke

python -m src.grpo.train \
  --config configs/grpo.yaml \
  --output-dir outputs/checkpoints/grpo

python -m src.evaluation.evaluate \
  --config configs/baseline.yaml \
  --stage grpo \
  --adapter-path outputs/checkpoints/grpo/best_adapter \
  --output-dir outputs/grpo
```

### 必须记录

- 每个Reward分量；
- 组内Reward标准差；
- 零方差Prompt比例；
- Completion平均长度；
- KL或与参考策略的偏移；
- SQL执行率与Execution Accuracy；
- Base、SFT、GRPO三阶段统一指标；
- 每个Checkpoint的验证结果；
- GPU峰值显存、耗时和失败次数。

### 停止条件

出现以下情况应停止扩大训练并先分析：

- Reward上升但验证集Execution Accuracy不升；
- 组内Reward标准差持续接近0；
- 输出明显变长或频繁截断；
- SQL格式提高但真实执行正确率下降；
- 显存接近上限且频繁OOM；
- 连续多个Checkpoint无提升。

### 退出条件

- 完成SFT vs GRPO的同条件评测；
- 能解释GRPO是否有效以及原因；
- 即使GRPO没有提升，也有完整的失败分析；
- 不以训练Reward代替最终任务指标。

---

### 阶段 8：最终部署与端到端验证

### 目标

将最佳模型接入统一推理入口，证明模型可以从问题生成并执行SQL。

### 启动命令

```bash
bash scripts/serve.sh --backend transformers
```

对应 Python 命令：

```bash
python -m src.serving.cli \
  --config configs/serving.yaml \
  --adapter-path outputs/checkpoints/best_adapter
```

可选API：

```bash
python -m src.serving.api --config configs/serving.yaml
```

如果硬件和模型版本确认兼容，再增加vLLM部署；它不是项目完成的必要条件。

### 退出条件

- 输入问题可以返回SQL、执行状态和结果；
- 只允许只读查询；
- 推理配置与评测配置可追踪；
- README中包含可复制的启动示例。

---

### 阶段 9：最终总结与AI模拟面试

### 目标

将真实项目产物转化为可讲述、可追问、可验证的面试材料。

### 前置材料

AI生成面试问题前必须读取：

- README；
- 本启动说明；
- 模型架构笔记；
- 数据报告；
- 实验日志；
- 问题日志；
- Base/SFT/GRPO指标；
- 核心训练与Reward代码。

### 启动命令

```bash
bash scripts/generate_interview.sh
```

对应 Python 命令：

```bash
python -m src.utils.generate_interview \
  --project-root . \
  --output docs/interview_questions.md
```

### 面试题必须覆盖

- 项目为什么选择Text-to-SQL；
- 数据来源、生成、清洗和泄漏控制；
- Qwen模型架构；
- Base模型如何部署；
- Prompt和Chat Template；
- SFT Loss和Label Mask；
- LoRA/QLoRA原理及参数选择；
- Train/Validation/Test区别；
- GRPO相对优势与Reward设计；
- 为什么使用执行结果而非SQL字符串相似度；
- Base、SFT、GRPO实验对比；
- OOM、环境冲突、Loss异常和Reward失配；
- 项目局限与下一步改进。

AI只能根据真实代码和日志出题，不能把未完成模块写成已完成成果。

### 退出条件

- 生成基础题、原理题、代码题、实验题和压力追问题；
- 每个问题标注对应项目证据位置；
- 本人能脱离文档回答核心问题；
- 不确定或失败实验能够如实解释。

---

## 9. 算力、内存与磁盘预估

以下为4B级模型在单张 RTX 3090 24GB 服务器上的保守起始估计，实际值取决于模型结构、序列长度、Batch Size、量化和软件版本。阶段0必须用服务器真实 Smoke Test 覆盖估算。

| 阶段 | CPU内存 | 单卡显存估计 | 磁盘 | RTX 3090 24GB适配判断 |
|---|---:|---:|---:|---|
| 数据处理/SQLite | 8～16GB | 不需要 | 5～20GB | 可行 |
| 4B BF16推理 | 16GB+ | 约10～14GB | 10～20GB | 可行，但正式对比优先保持统一量化设置 |
| 4B 4-bit推理 | 16GB+ | 约4～7GB | 5～15GB | 可行，阶段0默认配置 |
| 4B QLoRA SFT，短序列 | 24GB+ | 约10～18GB | 15～40GB | 通常可行；从batch=1、seq=512开始 |
| 4B普通LoRA BF16 | 32GB+ | 约18～24GB或更高 | 20～50GB | 较紧，不作为默认路线 |
| 4B GRPO，多Rollout | 32GB+ | 约18～24GB或更高 | 20～60GB | 高风险，必须先小group、短输出 smoke test |
| 0.6B～1.7B GRPO | 16～32GB | 约6～14GB | 10～30GB | 4B失败时的独立练习后备链 |

### 算力决策规则

1. Base、SFT、GRPO对比必须使用同一模型；
2. 如果4B只能完成Base和SFT，GRPO使用小模型时必须作为独立实验链，不能直接与4B SFT宣称提升；
3. 最稳妥方案是先用小模型跑通全流程，再决定是否用4B重复正式实验；
4. 每次正式训练前先运行10步以内的Smoke Test；
5. 记录`nvidia-smi`、PyTorch显存峰值、序列长度、Batch Size和量化设置；
6. 磁盘剩余空间不足时禁止启动训练；
7. 多卡训练只有在明确配置Accelerate/DeepSpeed/FSDP后才会拆分负载。

---

## 10. 阶段式消融与统一评测

分阶段开发本身形成最重要的消融，不需要为基础对比重复训练：

| 实验 | 模型 | 训练 | 目的 |
|---|---|---|---|
| E0 | Base | 无 | 未训练基线 |
| E1 | Base + SFT Adapter | QLoRA SFT | 测量监督微调贡献 |
| E2 | SFT + GRPO Adapter | SFT后继续GRPO | 测量执行奖励贡献 |

三组实验必须共享：

- 测试集；
- Prompt模板；
- SQL解析器；
- 执行器；
- 结果标准化器；
- 解码策略；
- 指标实现。

第一版不做大量超参数搜索。只有在发现明确问题时，才增加单变量实验，例如：

- LoRA rank 8 vs 16；
- 有无assistant-only loss；
- 纯执行Reward vs 执行Reward+格式Reward；
- Group Size 4 vs 8。

每次只改变一个核心因素，并复用已有Checkpoint。

---

## 11. 项目日志规范

### 11.1 实验日志：`docs/experiment_log.md`

每次实验追加：

```text
日期与实验ID：
目标/假设：
Git Commit：
模型与Checkpoint：
数据版本与哈希：
配置文件：
启动命令：
GPU与显存：
运行时间：
核心指标：
观察结果：
结论：
下一步：
```

### 11.2 问题日志：`docs/issue_log.md`

每次问题追加：

```text
问题ID和日期：
所处阶段：
错误信息：
复现命令：
环境信息：
初始假设：
定位过程：
根本原因：
解决方案：
验证方式：
是否可能再次发生：
```

只记录最终解决命令是不够的，必须记录为什么会发生以及如何验证修复。

### 11.3 配置与结果追踪

- 每次运行保存解析后的完整配置；
- 记录随机种子；
- 记录代码Commit；
- 日志和Checkpoint使用唯一实验ID；
- 不覆盖已有结果；
- 失败实验同样保留最小日志；
- 禁止手工修改最终指标文件。

---

## 12. README必须维护的启动命令

README应随着阶段完成逐步更新，而不是最后补写。最终至少包含以下两套等价入口。

### Shell入口

```bash
bash scripts/bootstrap.sh
bash scripts/check_env.sh
bash scripts/prepare_data.sh --mode smoke
bash scripts/inspect_model.sh
bash scripts/run_base_inference.sh --mode single
bash scripts/evaluate_base.sh
bash scripts/train_sft.sh --mode smoke
bash scripts/train_sft.sh --mode full
bash scripts/evaluate_sft.sh
bash scripts/train_grpo.sh --mode smoke
bash scripts/train_grpo.sh --mode full
bash scripts/evaluate_grpo.sh
bash scripts/serve.sh --backend transformers
bash scripts/generate_interview.sh
```

### Python入口

```bash
python -m src.utils.check_env
python -m src.data.validate --config configs/data.yaml
python -m src.model.inspect_architecture --config configs/model.yaml
python -m src.inference.generate --config configs/baseline.yaml --stage base
python -m src.evaluation.evaluate --config configs/baseline.yaml --stage base
python -m src.sft.train --config configs/sft.yaml
python -m src.evaluation.evaluate --config configs/baseline.yaml --stage sft
python -m src.grpo.train --config configs/grpo.yaml
python -m src.evaluation.evaluate --config configs/baseline.yaml --stage grpo
python -m src.serving.cli --config configs/serving.yaml
python -m src.utils.generate_interview --project-root .
```

Shell脚本只能负责环境变量、默认参数和命令封装，核心逻辑必须在Python模块中，防止两套入口行为不一致。

---

## 13. 项目完成标准

项目只有同时满足以下条件才算完成：

- 从空环境可以根据README完成安装；
- CSpider 或 Spider 数据可自动下载/导入、转换、执行验证、划分和复现；
- Base模型能够加载和推理；
- 有固定Baseline结果；
- 完成至少一次可复现QLoRA SFT；
- 完成SFT后的统一评测；
- GRPO至少完成Reward测试和Smoke Test；
- 如果算力允许，完成正式GRPO及评测；
- Base/SFT/GRPO结果可在同一张表比较；
- 训练代码和计算原理均有本人总结；
- 所有正式实验有配置、日志和问题记录；
- README同时提供`.sh`和`python -m`启动命令；
- 记录实际CPU内存、GPU显存、磁盘和训练耗时；
- 最终报告不夸大GRPO或SFT效果；
- AI基于真实项目生成模拟面试问题；
- 本人能够解释项目的核心代码、指标、失败和取舍；
- README 和最终报告能够清楚说明本人在数据、SFT、Reward、GRPO 和评测链路中的具体工作，形成可用于实习简历和面试的项目材料。

---

## 14. 推荐执行顺序

严格按以下顺序推进：

```text
阶段0：确认环境和算力
↓
阶段1：完成小规模可执行数据
↓
阶段2：解析模型架构
↓
阶段3：固定Base Baseline
↓
阶段4：完成SFT Smoke Test
↓
阶段5：完成正式SFT及对比
↓
阶段6：单独验证Reward
↓
阶段7：完成GRPO Smoke Test，再决定是否扩大
↓
阶段8：部署最佳模型
↓
阶段9：总结并生成模拟面试
```

任何阶段没有满足退出条件，都不进入下一阶段。

项目首先追求：

```text
跑通并理解 > 扩大规模
可验证 > 指标好看
可复现 > 模块数量
真实结论 > 预设GRPO有效
```
