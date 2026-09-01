# Text-to-SQL 后训练数据准备计划

> 本计划服务于 SFT 与 GRPO 后训练闭环。数据工作的目标是“快速得到可执行、可复现、可评测的数据”，不是建设垂直领域数据集。

阶段 1 的训练阶段数据矩阵、字段暴露边界、防泄漏规则和当前执行顺序已经冻结在 [`docs/stage1_data_plan.md`](docs/stage1_data_plan.md)。本文保留总体实施说明；发生冲突时，以阶段 1 文档为准。

执行状态：`cspider-smoke-v0` 已通过 Gold SQL 执行、token 长度、`db_id` 泄漏和文件哈希验证，实测结果见 [`docs/data_report.md`](docs/data_report.md)。

## 1. 默认方案

第一版直接使用 CSpider：

- 自然语言问题：中文；
- SQL 方言：SQLite；
- 数据内容：问题、Gold SQL、数据库 Schema、SQLite 数据库；
- 用途：Base 评测、SFT、GRPO 执行奖励和统一测试；
- 代码策略：只使用数据，不运行 CSpider 仓库中陈旧的训练代码；
- 许可策略：记录原始来源和说明，个人学习研究使用；未确认许可前不重新分发原始数据。

如果 CSpider 无法稳定下载或使用，则切换到 Spider 1.0。第一版不混合多个主数据源。

Olist、BIRD、自动翻译、问题改写和模板生成均为后续可选实验，不进入项目关键路径。

## 2. 数据产物

```text
data/
├── raw/
│   ├── cspider/                 # 原始文件，只读保留
│   └── manifests/               # 来源、版本、下载时间、哈希和许可说明
├── interim/
│   ├── canonical_all.jsonl
│   ├── execution_failures.jsonl
│   └── split_manifest.json
├── databases/
│   └── <db_id>/<db_id>.sqlite
└── processed/
    ├── canonical_train.jsonl
    ├── canonical_validation.jsonl
    ├── canonical_test.jsonl
    ├── sft_train.jsonl
    ├── sft_validation.jsonl
    ├── grpo_train.jsonl
    ├── test.jsonl
    └── dataset_manifest.json
```

`processed/` 文件必须由脚本生成，不允许手工修改。

## 3. Canonical task

以下数值只用于展示字段结构，实际 `gold_result` 必须由对应数据库执行 `gold_sql` 后写入：

```json
{
  "id": "cspider_train_000001",
  "source": "cspider",
  "question": "年龄超过56岁的负责人有多少位？",
  "db_id": "department_management",
  "db_path": "data/databases/department_management/department_management.sqlite",
  "schema": "CREATE TABLE ...",
  "schema_hash": "sha256:...",
  "gold_sql": "SELECT count(*) FROM head WHERE age > 56;",
  "gold_result": [[1]],
  "result_compare_mode": "unordered",
  "difficulty": "easy",
  "sql_features": ["filter", "aggregate"],
  "split": "train",
  "data_version": "cspider-smoke-v0"
}
```

关键约束：

- `db_path` 使用仓库相对路径；
- `gold_result` 只能由程序执行 Gold SQL 得到；
- Schema 序列化顺序固定；
- 同一 `db_id` 只能属于一个 split；
- 原始字段始终可追溯，不在转换时静默修正。

## 4. 数据划分

优先保留 CSpider/Spider 官方的跨数据库划分：

1. 官方 train 作为训练候选池；
2. 从 train 按 `db_id` 划出 validation；
3. 官方 dev 锁定为 test；
4. 不使用官方 test 的隐藏标签；
5. 同一数据库不得跨 train、validation、test；
6. 测试集锁定后，不根据模型结果改题或移动样本。

Smoke 版建议：

| 用途 | 数量 |
|---|---:|
| SFT Train | 80 |
| Validation | 24 |
| GRPO Train | 32 |
| Test | 40 |

正式 v1 建议：

| 用途 | 数量 |
|---|---:|
| SFT Train | 800～2,000 |
| Validation | 100～200 |
| GRPO Train | 200～500 |
| Test | 200～300 |

抽样使用固定随机种子，并按数据库与难度分层，避免少数数据库占据大部分样本。

## 5. 分步骤实施

### D0：锁定来源

1. 尝试获取 CSpider 完整 train、dev、Gold SQL、tables/schema 和 database 目录。
2. 记录来源链接、获取日期、压缩包和解压文件 SHA-256。
3. 确认原始文件可以相互关联。
4. 如果下载或条款存在阻碍，立即切换 Spider 1.0，不在找数据上持续耗时。

退出条件：选定唯一主数据源，原始文件和来源记录完整。

### D1：统一格式

1. 解析问题、Gold SQL、`db_id` 和 Schema。
2. 将数据库复制或映射到统一相对路径。
3. 生成稳定的 Schema 文本。
4. 转换成 canonical JSONL。
5. 保存无法关联数据库或 SQL 的拒绝记录。

退出条件：所有保留样本满足 canonical schema，ID 唯一且来源可追溯。

### D2：执行 Gold SQL

1. 只读打开对应 SQLite 数据库。
2. 只允许单条 `SELECT` 或只读 `WITH ... SELECT`。
3. 设置执行超时和最大返回行数。
4. 生成并缓存 Gold Result。
5. 分类记录语法错误、方言不兼容、缺表、缺列、超时和其他异常。

退出条件：进入 processed 的样本 Gold SQL 执行成功率为 100%；失败样本有 reason code。

### D3：划分和抽样

1. 沿用官方 train/dev 边界。
2. 从 train 按 `db_id` 确定 validation 数据库。
3. 在各 split 内按难度分层抽取 smoke 数量。
4. 将 split 和 seed 写入 manifest。
5. 检查 `db_id` 交集为空。

退出条件：固定 seed 可重现相同 ID 列表，数据库级泄漏为零。

### D4：派生 SFT 数据

SFT 使用 chat messages：

```json
{
  "id": "cspider_train_000001",
  "messages": [
    {"role": "system", "content": "你是 SQLite 查询生成器，只输出一条只读 SQL。"},
    {"role": "user", "content": "数据库结构：\n...\n\n问题：\n..."},
    {"role": "assistant", "content": "SELECT ...;"}
  ]
}
```

退出条件：最终 tokenizer 能处理全部样本；Assistant-only Label Mask 单元测试通过；没有意外截断 SQL。

### D5：派生 GRPO 数据

GRPO 行只向模型暴露 Schema 和问题，同时保存 Reward 所需的：

- `db_path`；
- `gold_result`；
- 结果比较模式；
- 浮点容差；
- 最大结果行数。

`gold_sql` 可留在审计数据中，但不得拼入模型输入。GRPO 子集只从 train 选择，不能根据 test 表现筛选。

退出条件：Reward 能区分正确、等价、错误、不可执行、空输出和危险 SQL。

### D6：验证并冻结版本

生成以下统计：

- 各 split 样本数和数据库数；
- SQL 执行成功率和失败类型；
- 空结果比例；
- 难度和 SQL 特征分布；
- Prompt/SQL token 长度分位数；
- 精确重复和数据库级泄漏；
- 文件、Schema、数据库和配置哈希。

冻结为 `cspider-smoke-v0`。任何修改产生新版本，不覆盖旧文件。

## 6. 进入训练前的硬门槛

- 数据来源和版本已记录；
- 保留样本 Gold SQL 执行成功率 100%；
- Gold Result 全部由程序生成；
- train/validation/test 的 `db_id` 交集为空；
- 测试集 ID 和文件哈希已经锁定；
- 危险 SQL 为 0；
- SFT Label Mask 测试通过；
- 所有样本能在最大序列长度内完整编码；
- Reward 单元测试覆盖正确、等价、错误和异常 SQL；
- `dataset_manifest.json` 和 `docs/data_report.md` 已生成。

## 7. 实际开工顺序

1. 先完成阶段 0，锁定模型和 tokenizer。
2. 尝试下载 CSpider；失败即切换 Spider 1.0。
3. 只做 canonical 转换和 SQLite 执行，不生成新问题。
4. 冻结 80/24/32/40 的 smoke 数据。
5. 完成 Base baseline。
6. 完成 SFT 10-step smoke test。
7. 单独验证 Reward 后完成 GRPO 5-step smoke test。
8. 全链路通过后再扩大到正式 v1。

项目先证明后训练链路正确，再讨论更多数据、领域适配或指标提升。
