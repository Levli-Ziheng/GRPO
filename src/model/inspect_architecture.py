from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.model.load import resolve_model_source
from src.utils.config import load_yaml, write_json


def _get(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def inspect_architecture(project_config: dict[str, Any]) -> dict[str, Any]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_config = project_config["model"]
    source, source_is_local = resolve_model_source(model_config)
    kwargs = {
        "trust_remote_code": bool(model_config["trust_remote_code"]),
        "local_files_only": bool(model_config["local_files_only"]),
    }
    if not source_is_local:
        kwargs["revision"] = model_config["revision"]
    config = AutoConfig.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    raw_config = {}
    weight_index = {}
    if source_is_local:
        raw_config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
        index_path = Path(source) / "model.safetensors.index.json"
        if index_path.is_file():
            weight_index = json.loads(index_path.read_text(encoding="utf-8"))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    hidden = int(_get(config, "hidden_size"))
    layers = int(_get(config, "num_hidden_layers"))
    heads = int(_get(config, "num_attention_heads"))
    kv_heads = int(_get(config, "num_key_value_heads", heads))
    head_dim = int(_get(config, "head_dim", hidden // heads))
    intermediate = int(_get(config, "intermediate_size"))
    vocab = int(_get(config, "vocab_size"))
    sample_messages = [
        {"role": "system", "content": "你是 SQLite 查询生成器，只输出 SQL。"},
        {"role": "user", "content": "CREATE TABLE t(id INTEGER); 查询行数。"},
    ]
    encoded = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]

    structural_parameters = sum(parameter.numel() for parameter in model.parameters())
    total_weight_bytes = int(weight_index.get("metadata", {}).get("total_size", 0))
    checkpoint_dtype = str(raw_config.get("torch_dtype", "bfloat16"))
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}.get(checkpoint_dtype)
    checkpoint_parameters = total_weight_bytes // dtype_bytes if dtype_bytes else 0
    total_parameters = checkpoint_parameters or structural_parameters
    kv_bytes_per_token_bf16 = 2 * layers * kv_heads * head_dim * 2
    weight_files = sorted(Path(source).glob("*.safetensors")) if source_is_local else []
    return {
        "model_id": model_config["id"],
        "model_source": source,
        "revision": model_config["revision"],
        "architecture_class": (_get(config, "architectures", []) or [type(model).__name__])[0],
        "model_type": _get(config, "model_type"),
        "dimensions": {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "query_projection_size": heads * head_dim,
            "key_value_projection_size": kv_heads * head_dim,
            "gqa_query_heads_per_kv_head": heads // kv_heads,
            "max_position_embeddings": int(_get(config, "max_position_embeddings")),
        },
        "attention": {
            "rope_theta": raw_config.get("rope_theta", _get(config, "rope_theta")),
            "rope_scaling": raw_config.get("rope_scaling", _get(config, "rope_scaling")),
            "attention_bias": _get(config, "attention_bias"),
            "attention_dropout": _get(config, "attention_dropout"),
        },
        "mlp": {"activation": _get(config, "hidden_act"), "gated": True},
        "normalization": {"type": "RMSNorm", "epsilon": _get(config, "rms_norm_eps")},
        "embeddings": {
            "tie_word_embeddings": bool(_get(config, "tie_word_embeddings")),
            "embedding_shape": [vocab, hidden],
            "lm_head_shape": [vocab, hidden],
        },
        "parameters": {
            "total": total_parameters,
            "trainable_before_adapter": total_parameters,
            "meta_structural_count_before_tying": structural_parameters,
            "checkpoint_dtype": checkpoint_dtype,
            "weight_file_count": len(weight_files),
            "weight_bytes_on_disk": total_weight_bytes or sum(path.stat().st_size for path in weight_files),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_present": bool(tokenizer.chat_template),
            "sample_input_tokens": len(input_ids),
            "sample_input_ids_head": input_ids[:16],
            "sample_attention_mask_shape": [1, len(attention_mask)],
        },
        "tensor_flow": {
            "input_ids": [1, "sequence_length"],
            "embedding_output": [1, "sequence_length", hidden],
            "q_projection": [1, "sequence_length", heads, head_dim],
            "k_v_projection": [1, "sequence_length", kv_heads, head_dim],
            "logits": [1, "sequence_length", vocab],
        },
        "kv_cache_bf16": {
            "bytes_per_token_batch_1": kv_bytes_per_token_bf16,
            "gib_at_512_tokens_batch_1": round(kv_bytes_per_token_bf16 * 512 / 2**30, 4),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    d = report["dimensions"]
    p = report["parameters"]
    t = report["tokenizer"]
    a = report["attention"]
    return f"""# Qwen3-4B-Instruct-2507 模型架构解析

> 状态：已由服务器本地模型配置和 tokenizer 生成；模型 commit：`{report['revision']}`。

## 1. 选择结论

本项目锁定 `{report['model_id']}` 作为 Base、SFT 和 GRPO 主实验的共同起点。服务器有两张 RTX 3090 24GB；阶段 0 已实测 4-bit 单样本推理峰值约 3GB，因此 4B 模型适合 Base 推理和 QLoRA SFT。GRPO 仍需小 group、短 completion 的独立 smoke test。这里的“Base”表示后训练前的起点 checkpoint；该 checkpoint 本身经过指令微调。

## 2. 实测配置

| 项目 | 值 |
|---|---:|
| 架构类 | `{report['architecture_class']}` |
| Checkpoint 参数量 | {p['total']:,} |
| 词表 | {d['vocab_size']:,} |
| 隐藏维度 | {d['hidden_size']:,} |
| Decoder 层数 | {d['num_hidden_layers']} |
| Attention heads / KV heads | {d['num_attention_heads']} / {d['num_key_value_heads']} |
| Head dim | {d['head_dim']} |
| MLP 中间维度 | {d['intermediate_size']:,} |
| 最大位置长度 | {d['max_position_embeddings']:,} |
| 本地权重分片 / 大小 | {p['weight_file_count']} / {p['weight_bytes_on_disk'] / 2**30:.2f} GiB |

## 3. 模块树

```text
Qwen3ForCausalLM
├─ model.embed_tokens: token id → hidden vector
├─ model.layers × {d['num_hidden_layers']}
│  ├─ input_layernorm: RMSNorm
│  ├─ self_attn
│  │  ├─ q_proj + q_norm: {d['num_attention_heads']} × {d['head_dim']}
│  │  ├─ k_proj + k_norm: {d['num_key_value_heads']} × {d['head_dim']}
│  │  ├─ v_proj: {d['num_key_value_heads']} × {d['head_dim']}
│  │  ├─ RoPE
│  │  └─ o_proj
│  ├─ residual connection
│  ├─ post_attention_layernorm: RMSNorm
│  ├─ mlp: gate_proj + up_proj → SiLU gate → down_proj
│  └─ residual connection
├─ model.norm: RMSNorm
└─ lm_head: hidden vector → vocabulary logits
```

## 4. Attention、GQA 与 RoPE

Query 有 {d['num_attention_heads']} 个头，Key/Value 各有 {d['num_key_value_heads']} 个头，每 {d['gqa_query_heads_per_kv_head']} 个 Query 头共享一组 Key/Value，这就是 Grouped Query Attention。它保留较多 Query 表达能力，同时降低 KV Cache。Q 投影宽度是 {d['query_projection_size']}，K/V 投影宽度各是 {d['key_value_projection_size']}。

RoPE 的 `theta` 为 `{a['rope_theta']}`，缩放配置为 `{a['rope_scaling']}`。它把位置信息编码进 Q/K 的旋转关系，而不是额外加一个位置向量。

BF16、batch=1、512 tokens 时，按 K 和 V 两份缓存估算 KV Cache 约 {report['kv_cache_bf16']['gib_at_512_tokens_batch_1']} GiB；实际显存还包括量化权重、激活、临时张量和框架开销。

## 5. 输入如何变成下一个 Token

1. tokenizer 按 chat template 把 System/User 文本编码为 token ids；本次样例得到 {t['sample_input_tokens']} 个 token，Attention Mask 形状为 `{t['sample_attention_mask_shape']}`。
2. Embedding 把 `[batch, seq]` 变成 `[batch, seq, {d['hidden_size']}]`。
3. 每个 Decoder 层依次做因果自注意力、残差、RMSNorm、门控 MLP 和第二次残差。
4. 最终 RMSNorm 与 LM Head 产生 `[batch, seq, {d['vocab_size']}]` logits。
5. 最后一个位置的 logits 经解码策略选择下一个 token，再把它追加到序列并重复前向过程。

## 6. SFT 为什么 Shift Labels

因果语言模型在位置 `t` 的输出预测位置 `t+1`，所以训练时比较 `logits[:, :-1]` 与 `labels[:, 1:]`。本项目还需要把 System 和 User 区域的 label 设为 `-100`，只让 Assistant SQL token 参与交叉熵。这样优化目标是“根据问题和 Schema 生成 SQL”，而不是复述输入提示。

## 7. Tokenizer 检查

- tokenizer 类：`{t['class']}`；有效词表大小：{t['vocab_size']:,}；
- EOS / PAD：`{t['eos_token_id']}` / `{t['pad_token_id']}`；
- chat template：{'存在' if t['chat_template_present'] else '缺失'}；
- 样例 token ids 前 16 个：`{t['sample_input_ids_head']}`。

机器可读报告位于 `outputs/stage2/architecture.json`。参数量由 safetensors 索引中的 BF16 总字节数换算，避免 meta-device 构造尚未绑定共享 Embedding/LM Head 时重复计数；权重大小也取自该索引。未在本阶段运行训练。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the locked model architecture.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_architecture(load_yaml(args.config))
    write_json(args.json_out, report)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(f"Model: {report['model_id']} @ {report['revision']}")
    print(f"Parameters: {report['parameters']['total']:,}")
    print(f"Wrote: {args.json_out}")
    print(f"Wrote: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
