# Text-to-SQL Post-training

用于练习 Base 推理、QLoRA SFT、GRPO/RLVR 和统一执行评测的个人项目。

当前进度：阶段 0、阶段 1 均已验收。CSpider 已转换并冻结为 `cspider-smoke-v0`，包含 Base/Eval、SFT 和 GRPO 数据视图。算力实测见 `docs/compute_report.md`，数据实测见 `docs/data_report.md`。

阶段 1 在已有 Conda 环境中执行：

```bash
conda activate lzh_llm
cd /home/rslab/lzh_Levi/Text-to-SQL
bash scripts/prepare_data.sh --mode smoke
```

脚本只读使用 `data/train.json`、`data/dev.json`、`data/tables.json` 和 `data/database/`，生成物写入 `data/interim/`、`data/processed/` 和 `docs/data_report.md`。

## 阶段 0：Linux 服务器

推荐 Python 3.11。请先激活你自己的 Conda/系统环境，再进入项目根目录执行；脚本不会创建虚拟环境：

```bash
bash scripts/bootstrap.sh
bash scripts/check_env.sh
bash scripts/smoke_model.sh
```

直接运行 Python 模块：

```bash
python -m src.utils.check_env --strict --json-out outputs/stage0/environment.json
python -m src.utils.memory_estimator --config configs/model.yaml --json-out outputs/stage0/memory_estimate.json
python -m src.model.load --config configs/model.yaml --smoke-test --json-out outputs/stage0/model_smoke.json
```

如果当前环境的解释器不是 `python3`，可以显式指定：

```bash
export PYTHON_BIN=/path/to/your/python
bash scripts/stage0.sh
```

默认安装 PyTorch 2.8.0 + CUDA 12.8 wheel。若服务器驱动不支持，可以在安装前覆盖版本或 CUDA wheel：

```bash
export TORCH_VERSION=2.8.0
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
bash scripts/bootstrap.sh
```

不要依据预估值宣称阶段 0 完成。只有服务器上的 `environment.json` 和 `model_smoke.json` 成功生成后，才能回填 `docs/compute_report.md`。

Windows PowerShell 脚本仅用于本地代码检查，不作为训练环境依据。
