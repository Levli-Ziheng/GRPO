"""Shared QLoRA preparation for training and a fresh-process reload."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from src.model.load import _torch_dtype, resolve_model_source

def load_base(config, gradient_checkpointing=True):
    source, local=resolve_model_source(config["model"])
    if not local or not config["model"]["local_files_only"]:
        raise ValueError("Only the verified offline model is allowed")
    q=config["quantization"]
    model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
        dtype=_torch_dtype(torch,config["runtime"]["dtype"]),device_map={"":0},attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=q["quant_type"],
            bnb_4bit_compute_dtype=_torch_dtype(torch,q["compute_dtype"]),bnb_4bit_use_double_quant=q["use_double_quant"]))
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    model.config.use_cache=False
    return model

def load_tokenizer(config):
    source,_=resolve_model_source(config["model"])
    return AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
