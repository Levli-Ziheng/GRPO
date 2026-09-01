# 阶段 1：Text-to-SQL 数据资产规划

> 状态：已执行并验收。当前冻结版本为 `cspider-smoke-v0`；实测统计与哈希见 `docs/data_report.md`。

## 1. 数据源决策

### 主方案：CSpider

- 使用中文问题、Spider SQLite 数据库、Schema 和 Gold SQL；
- 只使用公开数据，不运行原仓库的旧训练代码；
- CSpider 官方入口：<https://github.com/taolusi/chisp>；
- 数据下载依赖 Google Drive 或百度网盘，开始 D0 后只给一次限时尝试。

### 快速回退：Spider 1.0

- 如果 CSpider 在 30 分钟内无法完整取得 `train/dev + tables.json + database/`，立即切换 Spider 1.0；
- Hugging Face 入口：<https://huggingface.co/datasets/xlangai/spider>；
- 官方仓库：<https://github.com/taoyds/spider>；
- v0 只能选择一个主数据源，不混合中文和英文样本。

选择原则是“文件完整、SQLite 可执行、来源可追溯”，不是为了中文而长期处理网盘问题。

## 2. 总体数据流

```text
公开原始数据
  └─ raw：问题、Gold SQL、tables/schema、SQLite 数据库
      └─ canonical：统一字段、执行 Gold SQL、缓存 Gold Result
          ├─ Base/Eval view：prompt + 隐藏评测答案
          ├─ SFT view：prompt + Assistant Gold SQL
          ├─ GRPO view：prompt + 仅供 Reward 使用的执行元数据
          └─ Reward test：人工构造的小型数据库与边界案例
```

`canonical` 是唯一事实来源。SFT、GRPO 和 Eval 文件只能由脚本派生，不允许手改。

## 3. 各阶段需要的数据

| 阶段 | 数据文件 | 模型能看到 | 程序保留但不放入 Prompt | 用途 |
|---|---|---|---|---|
| Base baseline | `eval_test.jsonl` | System、Schema、Question | Gold SQL、Gold Result、DB 路径、比较规则 | 固定 Base 起点 |
| SFT smoke/full | `sft_train.jsonl` | System、Schema、Question、目标 SQL | 来源、DB、token 统计 | Assistant-only CE Loss |
| SFT checkpoint | `sft_validation.jsonl` | 与训练相同 | Gold Result、执行元数据 | Validation Loss 与执行准确率 |
| Reward 自测 | `reward_cases.jsonl` + toy DB | 候选 SQL | 预期 Reward 分量 | 证明 Reward 正确、安全 |
| GRPO smoke/full | `grpo_train.jsonl` | System、Schema、Question | DB 路径、Gold Result、比较规则；Gold SQL 仅审计保存 | 生成多条 SQL 并执行打分 |
| GRPO checkpoint | `eval_validation.jsonl` | System、Schema、Question | Gold SQL、Gold Result | 选 checkpoint、早停 |
| 最终统一评测 | `eval_test.jsonl` | System、Schema、Question | 全部 Gold 信息 | Base/SFT/GRPO 同条件对比 |
| 错误分析 | `audit/*.jsonl` | 不用于训练 | 失败原因、预测、执行异常 | 项目复盘与面试材料 |

### 3.1 Base/Eval 数据

模型输入只有统一 Prompt，不出现答案。评测程序使用：

- `db_path`；
- `gold_sql`；
- `gold_result` 或其稳定哈希；
- `order_sensitive`；
- `float_tolerance`；
- SQL 执行超时和最大返回行数。

测试集一旦冻结，不根据 Base/SFT/GRPO 结果移动、删改或重新抽样。

### 3.2 SFT 数据

每行使用 chat messages：

```json
{
  "id": "source_train_000001",
  "messages": [
    {"role": "system", "content": "你是 SQLite 查询生成器，只输出一条只读 SQL。"},
    {"role": "user", "content": "数据库结构：\n...\n\n问题：\n..."},
    {"role": "assistant", "content": "SELECT ...;"}
  ],
  "source_id": "source_train_000001"
}
```

硬约束：

- 只对 Assistant SQL token 计算 Loss；
- `prompt + SQL + EOS` 必须完整落在训练长度内；
- 不训练解释、思维链或 Markdown 代码块；
- Gold SQL 必须在对应数据库执行成功；
- SFT validation 不参加梯度更新。

### 3.3 GRPO 数据

GRPO 行只向策略模型暴露 Prompt：

```json
{
  "id": "source_train_000001",
  "prompt": [
    {"role": "system", "content": "你是 SQLite 查询生成器，只输出一条只读 SQL。"},
    {"role": "user", "content": "数据库结构：\n...\n\n问题：\n..."}
  ],
  "reward_context": {
    "db_path": "data/databases/...sqlite",
    "gold_result_hash": "sha256:...",
    "order_sensitive": false,
    "float_tolerance": 1e-6,
    "timeout_ms": 3000,
    "max_result_rows": 1000
  }
}
```

硬约束：

- `gold_sql`、`gold_result` 不得拼入 Prompt；
- 数据库必须只读打开，只接受单条 `SELECT` 或只读 `WITH ... SELECT`；
- smoke v0 排除 Gold Result 为空的样本，降低“随便查空结果也得分”的 Reward 漏洞；
- 正式版可以纳入空结果，但必须单独统计并增加防投机规则；
- GRPO 只从 train pool 取样，不看 validation/test 表现选题；
- SFT 与 GRPO 训练问题允许重叠，默认记录重叠数。若 SFT 后某些训练 Prompt 的 rollout 全对或全错、组内 Reward 长期零方差，可在 train pool 内重建 curriculum 版本。

### 3.4 Reward 测试数据

Reward 测试数据不是公开 benchmark，也不参与训练。建立一个极小 toy SQLite 数据库并覆盖：

1. Gold SQL；
2. SQL 字符串不同但执行结果等价；
3. 错列、错表和语法错误；
4. `INSERT/UPDATE/DELETE/DROP` 等危险 SQL；
5. 空输出、解释文字、Markdown 包裹、多条 SQL；
6. 有序结果与无序结果；
7. 浮点误差、NULL、重复行；
8. 超时或超大结果集。

这些案例用于 `tests/test_rewards.py`，保证执行正确性 Reward 高于格式 Reward，危险 SQL 始终得到拒绝分。

## 4. Canonical 数据标准

每个公开样本首先转换成统一记录：

```json
{
  "id": "source_train_000001",
  "source": "cspider",
  "source_split": "train",
  "split": "train",
  "question": "...",
  "db_id": "...",
  "db_path": "data/databases/...sqlite",
  "db_sha256": "sha256:...",
  "schema": "CREATE TABLE ...",
  "schema_hash": "sha256:...",
  "gold_sql": "SELECT ...;",
  "gold_result": [["..."]],
  "gold_result_hash": "sha256:...",
  "order_sensitive": false,
  "float_tolerance": 1e-6,
  "difficulty": "medium",
  "sql_features": ["join", "aggregate"],
  "prompt_tokens": 300,
  "sql_tokens": 25,
  "data_version": "source-smoke-v0"
}
```

Gold Result 必须由 SQLite 执行产生，不能由模型生成。原始字段异常不得静默修正，统一写入 `audit/rejections.jsonl`。

## 5. 划分与防泄漏

1. 官方 train 是唯一训练候选池；官方 dev 锁定为最终 test；
2. validation 从官方 train 中按 `db_id` 整库划出；
3. `train_db_ids ∩ validation_db_ids ∩ test_db_ids` 必须两两为空；
4. SFT 与 GRPO 可共享 train DB 和问题，这是顺序后训练，不属于测试泄漏；
5. validation 用于 checkpoint 与超参数选择，test 只做 Base/SFT/GRPO 里程碑对比；
6. 记录问题精确重复、规范化 SQL 重复、Schema 重复和跨 split 数据库交集；
7. 所有抽样使用固定 seed，并保存 ID 清单与文件 SHA-256。

## 6. Smoke 与正式规模

### Smoke v0

| 资产 | 数量 | 说明 |
|---|---:|---|
| SFT train | 80 | 覆盖多数据库与难度 |
| GRPO train | 32 | 默认从 train pool/SFT 样本中确定性抽取 |
| Validation | 24 | SFT/GRPO 共用，不训练 |
| Test | 40 | Base/SFT/GRPO 共用并锁定 |
| Reward cases | 12～20 | 人工边界案例，不属于 benchmark |

### Full v1

| 资产 | 数量 | 说明 |
|---|---:|---|
| SFT train | 800～2,000 | 先按 token 长度、数据库、难度分层 |
| GRPO candidate | 200～500 | 受 3090 rollout 成本限制 |
| Validation | 100～200 | checkpoint 选择 |
| Test | 200～300 | 统一最终比较 |

先完成 smoke 闭环再决定正式规模；不为了凑数量生成合成问题。

## 7. 目录与冻结产物

```text
data/
├── raw/
│   ├── <source>/
│   └── manifests/source_manifest.json
├── databases/<db_id>/<db_id>.sqlite
├── interim/
│   ├── canonical_all.jsonl
│   ├── gold_execution_cache.jsonl
│   └── audit/
│       ├── rejections.jsonl
│       └── execution_failures.jsonl
├── processed/<data_version>/
│   ├── canonical_train.jsonl
│   ├── canonical_validation.jsonl
│   ├── canonical_test.jsonl
│   ├── sft_train.jsonl
│   ├── sft_validation.jsonl
│   ├── grpo_train.jsonl
│   ├── eval_validation.jsonl
│   ├── eval_test.jsonl
│   └── dataset_manifest.json
└── fixtures/
    ├── reward_cases.jsonl
    └── reward_test.sqlite
```

`raw/` 只读保存，`processed/<data_version>/` 冻结后不覆盖。任何过滤、Prompt 或比较规则变化都产生新版本。

## 8. 阶段 1 执行顺序

1. D0：限时获取 CSpider；失败立即切换 Spider 1.0，记录来源、许可说明和哈希；
2. D1：转换 canonical，统一 Schema 序列化与相对数据库路径；
3. D2：只读执行全部 Gold SQL，缓存 Gold Result，失败样本进入审计文件；
4. D3：按 `db_id` 划分，生成 80/24/32/40 smoke ID 清单；
5. D4：派生 SFT、GRPO、Eval 三类 view；
6. D5：创建 Reward toy DB 和边界案例；
7. D6：用 Qwen3-4B tokenizer 做长度检查，验证泄漏、重复和文件哈希；
8. D7：生成 `docs/data_report.md`，冻结 `source-smoke-v0`。

## 9. 阶段 1 退出条件

- [ ] 唯一主数据源已锁定，来源、版本、许可说明和 SHA-256 已记录；
- [ ] 保留样本 Gold SQL 执行成功率 100%；
- [ ] train/validation/test 的 `db_id` 泄漏为 0；
- [ ] SFT、GRPO、Eval view 均能由 canonical 确定性重建；
- [ ] GRPO Prompt 中不存在 Gold SQL/Gold Result；
- [ ] 所有 smoke 样本在配置的 token 长度内完整编码；
- [ ] Reward 边界案例齐全；
- [ ] `dataset_manifest.json` 与 `docs/data_report.md` 已生成；
- [ ] 测试集 ID 和文件哈希已冻结。

以上条件全部满足后，才能进入 Base baseline 和 SFT smoke。
