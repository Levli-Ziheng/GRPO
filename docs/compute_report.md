# 阶段 0 算力与模型可行性报告

> 状态：已验收。实测日期：2026-08-31；服务器目录：`/home/rslab/lzh_Levi/Text-to-SQL`。

## 1. 候选锁定配置

| 项目 | 配置 |
|---|---|
| 服务器 GPU | 2 × NVIDIA GeForce RTX 3090 24GB；阶段 0 使用 `cuda:0` 单卡 |
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` |
| 模型许可 | Apache-2.0 |
| 参数量 | 4.0B |
| 推理/训练权重 | bitsandbytes 4-bit NF4 |
| 计算精度 | BF16 |
| 初始最大序列长度 | 512 |
| 初始 batch size | 1 |
| 后续训练 | QLoRA SFT → GRPO/RLVR |

选择理由：4B 模型能提供比极小模型更有意义的 Text-to-SQL 基线；RTX 3090 支持 BF16，24GB 显存适合先用 4-bit、短序列和小 batch 跑通后训练闭环。Base、SFT、GRPO 主实验统一锁定该模型与 commit；正式训练显存仍在对应阶段用 smoke test 验证。

## 2. 服务器环境实测

运行：

```bash
bash scripts/stage0.sh
```

| 项目 | 实测值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS；Linux 6.8.0-136-generic x86_64 |
| Python | 3.12.13；`/home/rslab/anaconda3/envs/lzh_llm/bin/python` |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8；CUDA available=True；BF16=True |
| NVIDIA Driver | 580.173.02 |
| GPU | 2 × NVIDIA GeForce RTX 3090，Compute Capability 8.6 |
| 总显存 | 每卡 24,576 MiB（PyTorch 显示约 23.56 GiB） |
| 空闲显存 | 正式 warm smoke 前 GPU0 24,119 MiB、GPU1 24,098 MiB |
| CPU | 16 核 / 32 逻辑线程 |
| 系统内存 | 125.70 GiB 总计；检查时 98.03 GiB 可用 |
| 项目磁盘剩余空间 | 278G（`/`） |
| transformers | 5.16.1 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.2 |
| peft | 0.20.0 |
| trl | 1.12.0 |

环境使用用户已有 Conda 环境 `lzh_llm`，未创建 `.env` 或 `.venv`。官方 Hugging Face 端点在服务器上连接超时，本次下载通过进程级 `HF_ENDPOINT=https://hf-mirror.com` 完成，未修改全局配置。

## 3. 模型 Smoke Test

| 项目 | 实测值 |
|---|---|
| 模型是否成功加载 | 是，退出码 0 |
| 模型 revision/commit | `main` / `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 量化方式 | bitsandbytes 4-bit NF4 + double quant；BF16 compute |
| 输入 token | 62 |
| 输出 token | 18 |
| 生成结果 | `SELECT COUNT(*) FROM department WHERE budget > 1000000;` |
| 加载耗时 | 10.095 秒（缓存命中后的 warm load） |
| 生成耗时 | 0.959 秒 |
| PyTorch 峰值 allocated | 2.564 GiB |
| PyTorch 峰值 reserved | 2.594 GiB |
| `nvidia-smi` 峰值显存 | 2,985 MiB（GPU0，0.2 秒间隔采样） |

首次下载加加载耗时为 646.913 秒，其中主要是约 8.04GB 权重下载；该数字不用于估算日常启动耗时。冒烟 SQL 与问题语义一致。

## 4. 可行性结论

- Base 4-bit 推理：已验证可行，实测峰值约 3GB。
- QLoRA SFT（batch=1、seq=512、gradient checkpointing）：规划估算 9.11–17.11 GiB，单张 3090 有可行空间，待阶段 4 训练 smoke test 验证。
- GRPO（小 group、短 completion）：规划估算 16.21–24.21 GiB，单卡高风险；服务器可见两张 3090，但多卡方案必须在阶段 7 单独验证，不能把显存简单相加。
- 主实验链锁定 `Qwen/Qwen3-4B-Instruct-2507`，Base/SFT/GRPO 必须使用相同 commit 与测试集。
- 后备练习链锁定 `Qwen/Qwen3-1.7B`：仅当 4B GRPO smoke test OOM 时启用，并从 Base、SFT 到 GRPO 全部独立运行；不得把 1.7B GRPO 与 4B SFT 串成同一实验结论。

## 5. 阶段 0 退出条件

- [x] 环境检查通过；
- [x] 候选模型完成一次最小加载和生成；
- [x] 记录实际峰值显存和耗时；
- [x] 确认 Base/SFT/GRPO 使用的同一模型；
- [x] 若 4B GRPO 风险过高，明确记录独立的小模型练习链。

阶段 0 已验收。服务器产物位于 `outputs/stage0/`，单元测试结果为 `2 passed`。
