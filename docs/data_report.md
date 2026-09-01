# 阶段 1 数据报告

> 状态：已验收；数据版本：`cspider-smoke-v0`。

## 1. 来源与原始资产

- 来源：cspider / official-full；
- Train：8659 条，Dev：1034 条；
- Schema：166 个，SQLite：166 个；
- 官方 train/dev 数据库交集：0；
- 许可记录：CSpider is derived from Spider. The CSpider repository does not provide a clear standalone dataset license; use is limited to this learning/research project and raw data must not be redistributed until terms are confirmed.

原始文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `data/train.json` | `e9bac272df6f3057207bfac0ab53d536ddeecdeed4d45456bfba115336c01c7b` |
| `data/dev.json` | `a50f2db0cebf26688daf0fdbabd745046f164a622072c49a0c5c9b088131cf7c` |
| `data/tables.json` | `b68de9166952871d64554486fca4b25ff88509e983857a6af428250ffd58b67f` |

## 2. Gold SQL 执行与清洗

- 输入：9693；执行成功：9658；失败：35；
- 执行成功率：99.64%；成功记录中的空结果：3176；
- Token/去重后保留：6004；拒绝：3654；
- 重复问题组：4；重复 SQL 组：2355。

执行失败类型：`{'missing_table': 1, 'result_too_large': 32, 'sqlite_error': 2}`

清洗拒绝类型：`{'exact_duplicate': 65, 'grpo_sequence_too_long': 318, 'sft_sequence_too_long': 3271}`

所有进入 processed 的 Gold SQL 都已在对应 SQLite 数据库只读执行成功；失败记录保存在 `data/interim/audit/`。

## 3. Smoke 划分与派生视图

| 资产 | 数量 |
|---|---:|
| sft_train | 80 |
| sft_validation | 24 |
| grpo_train | 32 |
| eval_validation | 24 |
| eval_test | 40 |

数据库数量：train=60，validation=10，test=17。

数据库泄漏：train/validation=0，train/test=0，validation/test=0。

GRPO 样本：32；与 SFT train 重叠：32；空 Gold Result：0。

## 4. 长度与分布

- Prompt tokens：P50=240.0，P95=419.7，Max=448；
- SFT sequence tokens：P50=281.5，P95=455.4，Max=486；
- 难度分布（启发式 v1）：`{'easy': 39, 'extra': 19, 'hard': 37, 'medium': 49}`；
- 最大序列长度：512；GRPO 预留 completion：64 tokens。

## 5. Reward 测试资产

已生成 13 个案例，覆盖：`['correct', 'correct_but_empty', 'empty_output', 'execution_error', 'format_error', 'unsafe_sql', 'wrong_result']`。这些 fixture 不参与训练。

## 6. 冻结哈希

| 文件 | SHA-256 |
|---|---|
| `canonical_test.jsonl` | `1c9ae1f2dc496bc206f30af99f7765b0f3ab4cc20ccad05fb300f67ce5e344db` |
| `canonical_train.jsonl` | `977fdcb464859278d1b05dbc3774353a060b7d33d3d39a533c6bdb5440509073` |
| `canonical_validation.jsonl` | `55d5ceccbf3dc06e10eb2badb1e60aee3c6e9032ec21f8b04e1d91a8eef8cef8` |
| `eval_test.jsonl` | `ccf8acf06ea3e1f6778d95712358afdb725dfbed2769c26ab3652b1d1f930d7c` |
| `eval_validation.jsonl` | `34fdc55ce8797342220b4160b2c7a20e0cd749c3709ce41df767c8bf725d9a4b` |
| `grpo_train.jsonl` | `d417ca101c7dd4d2b17d8b6e4ffcdc68da6fd5c9fb06e59ddc9fddb6bcdecc7d` |
| `sft_train.jsonl` | `2a19910bfbebdb4027ad9b5c9f620d6867bcbc573b9903d38d359f6841a15b00` |
| `sft_validation.jsonl` | `45c1a8dcc77440ee74aecfb86d95f67f82dd55529c5c85f6f010653369133ffb` |

## 7. 验收

- [x] 数据来源、许可说明和哈希已记录；
- [x] 所有原始 Gold SQL 均有执行或拒绝记录；
- [x] `db_id` 跨 split 泄漏为 0；
- [x] GRPO smoke 不含空 Gold Result；
- [x] SFT/GRPO/Eval 视图、长度和哈希验证通过；

阶段 1 只冻结 smoke 数据。正式 v1 必须由相同配置与脚本重新派生，不得手工扩充 processed 文件。
