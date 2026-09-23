"""Fresh-process checkpoint/adapter reload verification, no optimizer update."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import GenerationConfig
from src.sft.model import load_base,load_tokenizer
from src.sft.loss import assistant_loss
from src.utils.config import load_yaml,write_json
from src.inference.generate import Generator
from src.data.common import read_jsonl,write_jsonl
from src.evaluation.evaluate import evaluate_record,write_evaluation

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    args=parser.parse_args()
    out=Path(args.run_dir)
    resolved=load_yaml(out/"resolved_config.yaml")
    config,model_config=resolved["sft"],resolved["model"]
    status=json.loads((out/"status.json").read_text())
    if status["status"]!="trained_pending_reload":
        raise ValueError("Run must have completed training and not yet been verified")
    ckpt=out/f"checkpoint-{config['training']['max_steps']}"
    for name in ("optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json","adapter_model.safetensors","adapter_config.json"):
        if not (ckpt/name).is_file():
            raise FileNotFoundError(ckpt/name)
    state=json.loads((ckpt/"trainer_state.json").read_text())
    assert state["global_step"]==config["training"]["max_steps"]
    optimizer=torch.load(ckpt/"optimizer.pt",map_location="cpu",weights_only=True)
    optimizer_steps=sorted({int(v["step"]) for v in optimizer["state"].values()})
    assert optimizer_steps==[config["training"]["max_steps"]]
    del optimizer
    checkpoint_tensors=load_file(ckpt/"adapter_model.safetensors")
    exported_tensors=load_file(out/"adapter"/"adapter_model.safetensors")
    assert checkpoint_tensors.keys()==exported_tensors.keys()
    assert all(torch.equal(v,exported_tensors[k]) for k,v in checkpoint_tensors.items())
    del checkpoint_tensors,exported_tensors
    started=time.perf_counter()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base=load_base(model_config,gradient_checkpointing=False)
    model=PeftModel.from_pretrained(base,ckpt,is_trainable=False).eval()
    tokenizer=load_tokenizer(model_config)
    reference=torch.load(out/"reload_reference.pt",map_location="cpu",weights_only=True)
    batch={k:reference[k].to("cuda:0") for k in ("input_ids","attention_mask","labels")}
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        result=model(**batch)
        loss=assistant_loss(result.logits,batch["labels"])
    logits=result.logits[:,-1,:].float().cpu()
    difference=(logits-reference["last_logits"]).abs().max().item()
    tolerance=config["verification"]["reload_atol"]
    assert torch.allclose(logits,reference["last_logits"],atol=tolerance,rtol=0),difference
    assert abs(loss.item()-reference["loss"])<=tolerance
    del result,logits,batch
    # A single validation question tests generation after reload; never a test-set score.
    baseline=load_yaml("configs/baseline.yaml")
    task=next(read_jsonl(Path(config["dataset_root"])/"canonical_validation.jsonl"))
    from src.data.common import build_messages
    generator=Generator.__new__(Generator)
    generator.torch=torch; generator.model=model; generator.tokenizer=tokenizer
    generator.device=torch.device("cuda:0"); generator.config=baseline["generation"]
    generator.decoding=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,temperature=None,top_p=None,top_k=None,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    model.config.use_cache=True
    prediction=generator.generate(build_messages(load_yaml(config["data_config"]),task["schema"],task["question"]))
    prediction.update({"id":task["id"],"stage":"sft_smoke","model":model_config["model"]["id"],
                       "model_revision":model_config["model"]["revision"]})
    prediction=evaluate_record(prediction,task,baseline["evaluation"])
    if prediction["truncated"] or not prediction["raw_output"]:
        raise RuntimeError("Reloaded generation empty or truncated")
    write_jsonl(out/"predictions.jsonl",[prediction])
    write_evaluation(out,[prediction])
    report={"evidence_state":"executed","fresh_process":True,"checkpoint_step":state["global_step"],
        "optimizer_state_steps":optimizer_steps,"checkpoint_and_exported_adapter_tensors_equal":True,
        "max_abs_logit_difference":difference,"loss_before_save":reference["loss"],"loss_after_reload":loss.item(),
        "reload_atol":tolerance,"all_parameters_frozen_for_verification":not any(p.requires_grad for p in model.parameters()),
        "generation_sample_id":task["id"],"generation_split":"validation","generation_execution_correct":prediction["execution_correct"],
        "generation_nonempty":True,"generation_truncated":False,
        "elapsed_seconds":time.perf_counter()-started,
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
    write_json(out/"reload_verification.json",report)
    status.update({"status":"completed","reload_verified":True})
    write_json(out/"status.json",status)
    print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
