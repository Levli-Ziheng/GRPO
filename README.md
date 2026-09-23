# Text-to-SQL Post-training

一个面向中文 Text-to-SQL 的可复现后训练项目，覆盖数据预处理、Base 推理、QLoRA SFT、GRPO/RLVR、RL-ZVP（零方差优势塑形）、SQLite 执行评测和执行结果自一致性推理。

仓库只包含代码、配置、脚本和测试。数据集、模型权重、Adapter、训练输出、实验报告与个人笔记均不纳入版本控制。

## 功能

- 将 CSpider 原始数据转换为统一的 SFT、GRPO 和评测视图
- 使用 Qwen3-4B-Instruct 与 4-bit NF4 QLoRA 进行 SFT
- 使用执行正确性、部分结果相似度和 SQL 安全性奖励进行 GRPO
- 支持 RL-ZVP 熵引导优势塑形，利用组内零方差 prompt
- 提供 SQLite 只读执行器、结果归一化和严格 Execution Accuracy
- 支持基于执行结果投票的 self-consistency 推理
- 保存配置、哈希、资源占用和可独立重载的 LoRA Adapter

## 项目结构

```text
configs/   数据、模型、SFT、GRPO 和 RL-ZVP 配置
scripts/   环境检查、数据准备、训练与评测入口
src/       核心 Python 实现
tests/     单元测试和回归测试
```

运行时会使用但不会提交以下目录：

```text
data/      CSpider 原始数据及处理结果
models/    本地模型权重
outputs/   Checkpoint、预测和运行指标
docs/      本地实验报告和笔记
```

## 环境要求

- Linux
- Python 3.10+
- NVIDIA CUDA GPU；4B 模型的 4-bit 训练建议至少 24 GiB 显存
- 推荐使用独立虚拟环境或 Conda 环境

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需由脚本安装指定 CUDA 版本的 PyTorch：

```bash
TORCH_VERSION=2.8.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash scripts/bootstrap.sh
```

## 模型准备

默认模型为 `Qwen/Qwen3-4B-Instruct-2507`，固定 revision 记录在 `configs/model.yaml`。配置默认从以下本地目录离线加载：

```text
models/Qwen3-4B-Instruct-2507/
```

模型权重不属于本仓库。请遵守模型许可证，自行下载到该目录；如需直接从 Hugging Face 加载，可修改 `configs/model.yaml` 中的 `local_files_only` 和 `local_path`。

## 数据准备

本仓库不分发 CSpider。请从 [CSpider 项目](https://github.com/taolusi/chisp) 获取数据并确认其许可条款，然后放置为：

```text
data/train.json
data/dev.json
data/tables.json
data/database/<db_id>/<db_id>.sqlite
```

生成小规模或完整数据视图：

```bash
bash scripts/prepare_data.sh --mode smoke
bash scripts/prepare_data.sh --mode full
```

处理流程会校验数据库、执行 Gold SQL、清理异常样本并生成确定性数据划分。

## 快速检查

```bash
python -m pytest
bash scripts/check_env.sh
python -m src.model.load \
  --config configs/model.yaml \
  --smoke-test \
  --json-out outputs/stage0/model_smoke.json
```

## Base 推理与评测

```bash
bash scripts/run_base_inference.sh --mode single
bash scripts/evaluate_base.sh
```

## QLoRA SFT

Smoke test：

```bash
bash scripts/train_sft.sh \
  --mode smoke \
  --max-steps 10 \
  --output-dir outputs/sft/smoke-run
```

完整训练：

```bash
bash scripts/train_sft.sh \
  --mode full \
  --config configs/sft_full.yaml \
  --output-dir outputs/sft/full-run
```

输出目录必须是尚不存在的新目录，避免覆盖已有实验。

## GRPO 与 RL-ZVP

`configs/grpo_stage7.yaml` 启用 RL-ZVP。两个 `grpo_stage7_zvp_balanced_*.yaml` 配置提供难度均衡采样和更保守的学习率示例。

```bash
bash scripts/train_grpo.sh \
  --mode full \
  --config configs/grpo_stage7.yaml \
  --max-steps 50 \
  --output-dir outputs/grpo/run

bash scripts/train_grpo.sh --verify --output-dir outputs/grpo/run
bash scripts/evaluate_grpo.sh --action select --run-dir outputs/grpo/run
bash scripts/evaluate_grpo.sh --action evaluate --run-dir outputs/grpo/run
```

模型选择只使用验证集；测试集仅用于对选定候选做最终评测。

## 执行结果自一致性

该入口生成多个 SQL 候选，逐一执行后按预测结果投票；平票时优先 greedy 候选。它不会读取 Gold 正确性来选择答案。

```bash
python -m src.inference.self_consistency \
  --source-run outputs/grpo/run \
  --split validation \
  --num-candidates 4 \
  --output-dir outputs/inference/self-consistency-validation
```

## 配置说明

- `configs/model.yaml`：模型、量化和运行设备
- `configs/data.yaml`：原始数据路径、转换与划分
- `configs/sft*.yaml`：QLoRA 与训练预算
- `configs/grpo*.yaml`：奖励、采样、GRPO 与 RL-ZVP

Linux 脚本默认使用 `python3`。可通过 `PYTHON_BIN` 指定解释器，通过 `CUDA_VISIBLE_DEVICES` 指定 GPU：

```bash
PYTHON_BIN="$(command -v python3)" CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh --help
```

## 测试

```bash
python -m pytest -q
```

测试覆盖数据转换、SQL 解析与执行、SFT 标签 Mask、GRPO Reward/Advantage、RL-ZVP 和 self-consistency 选择逻辑。
