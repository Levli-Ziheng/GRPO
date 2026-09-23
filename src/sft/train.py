"""Stage 4 only: bounded QLoRA SFT, explicit masks and checkpoint evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed, GenerationConfig
from src.sft.dataset import SQLDataset, SQLCollator, load_frozen_rows, mask_audit
from src.sft.loss import assistant_loss
from src.sft.model import load_base, load_tokenizer
from src.inference.generate import verify_model
from src.utils.config import load_yaml, write_json
from src.data.common import sha256_file

def parameter_hashes(model, frozen_only=True):
    result={}
    for name,p in model.named_parameters():
        if frozen_only and p.requires_grad:
            continue
        raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        result[name]=hashlib.sha256(raw).hexdigest()
    return result

class AuditCallback(TrainerCallback):
    def __init__(self,path,start,config):
        self.path,self.start,self.config=path,start,config
        self.gradient_checks=[]
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        names=[]
        for name,p in model.named_parameters():
            if p.grad is not None:
                if not p.requires_grad or ".lora_" not in name:
                    raise RuntimeError("Unexpected gradient: "+name)
                if not torch.isfinite(p.grad).all():
                    raise RuntimeError("Non-finite gradient: "+name)
                names.append(name)
        if not names:
            raise RuntimeError("No LoRA gradients")
        self.gradient_checks.append({"step":state.global_step+1,"finite_lora_gradients":len(names)})
    def on_step_end(self,args,state,control,**kwargs):
        if time.perf_counter()-self.start>self.config["budget"]["max_seconds"]:
            raise TimeoutError("SFT budget exceeded")
        if torch.cuda.mem_get_info()[0]<self.config["budget"]["gpu_safety_margin_gib"]*2**30:
            raise RuntimeError("GPU safety margin exhausted")
    def on_log(self,args,state,control,logs=None,**kwargs):
        for key,value in (logs or {}).items():
            if isinstance(value,(int,float)) and not math.isfinite(value):
                raise RuntimeError("Non-finite metric: "+key)
        with self.path.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps({"step":state.global_step,**(logs or {})})+"\n")

class SQLTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=False
        self.observed_train_sequences=[]
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        labels=inputs["labels"]
        outputs=model(**{k:v for k,v in inputs.items() if k!="labels"})
        loss=assistant_loss(outputs.logits,labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite assistant loss")
        if model.training:
            self.observed_train_sequences.extend(inputs["input_ids"].detach().cpu().tolist())
        return (loss,outputs) if return_outputs else loss

def probe(model,batch):
    model.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        outputs=model(**batch)
        manual=assistant_loss(outputs.logits,batch["labels"])
        if not torch.allclose(outputs.loss,manual,atol=1e-5,rtol=1e-5):
            raise RuntimeError("Manual shifted loss differs from model loss")
    return manual.item(),outputs.logits[:,-1,:].detach().float().cpu()

def run(args):
    config=load_yaml(args.config)
    steps=args.max_steps or config["training"]["max_steps"]
    if not 1<=steps<=config["budget"]["max_steps"] or args.mode!="smoke":
        raise ValueError("Stage 4 is limited to 1–10 smoke steps")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Select one CUDA_VISIBLE_DEVICES device")
    if config["training"]["gradient_accumulation_steps"]!=1:
        raise ValueError("This smoke loss audit currently requires gradient accumulation 1")
    model_config=load_yaml(config["model_config"])
    out=Path(args.output_dir) if args.output_dir else Path("outputs/checkpoints")/(
        "sft-smoke-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    command="CUDA_VISIBLE_DEVICES="+os.environ["CUDA_VISIBLE_DEVICES"]+" "+shlex.join([sys.executable,"-m","src.sft.train",*sys.argv[1:]])
    write_json(out/"status.json",{"status":"running","command":command})
    try:
        if shutil.disk_usage(".").free<config["budget"]["min_free_disk_gib"]*2**30:
            raise RuntimeError("Insufficient disk")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.cuda.set_device(0)
        if torch.cuda.mem_get_info()[0]<config["budget"]["min_free_gpu_gib"]*2**30:
            raise RuntimeError("Insufficient GPU headroom")
        torch.cuda.reset_peak_memory_stats()
        (out/"gpu_before.csv").write_text(subprocess.check_output(
            ["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True))
        source=verify_model(model_config)
        train_rows,val_rows=load_frozen_rows(config)
        config["training"]["max_steps"]=steps
        resolved={"sft":config,"model":model_config,"command":command,
                  "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
                  "loss":"mean shifted cross-entropy over assistant SQL and EOS; exclude prompt/padding/trailing newline",
                  "trainer":"Transformers Trainer + PEFT QLoRA; explicit tokenized labels"}
        (out/"resolved_config.yaml").write_text(yaml.safe_dump(resolved,allow_unicode=True,sort_keys=False))
        with (out/"environment.log").open("w") as log:
            subprocess.run([sys.executable,"-m","src.utils.check_env","--json-out",str(out/"environment.json")],stdout=log,check=True)
        cmd=[sys.executable,".agents/skills/ai-experiment/scripts/capture_context.py","--repo-root",".",
             "--experiment-id",out.name,"--output",str(out/"run_context.json"),"--command",command]
        for path in [args.config,config["model_config"],config["data_config"],
                     str(Path(config["dataset_root"])/config["manifest_file"]),
                     str(Path(source)/"download_manifest.json"),str(out/"environment.json")]:
            cmd+=["--artifact",path]
        subprocess.run(cmd,check=True)
        set_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        tokenizer=load_tokenizer(model_config)
        train_data=SQLDataset(train_rows,tokenizer,config["max_sequence_length"])
        val_data=SQLDataset(val_rows,tokenizer,config["max_sequence_length"])
        collator=SQLCollator(tokenizer.pad_token_id)
        write_json(out/"mask_audit.json",{"all_train_checked":len(train_data),"all_validation_checked":len(val_data),
            "train_examples":mask_audit(train_data,tokenizer,config["seed"],config["verification"]["mask_sample_count"]),
            "validation_examples":mask_audit(val_data,tokenizer,config["seed"],config["verification"]["mask_sample_count"])})
        load_start=time.perf_counter()
        base=load_base(model_config,gradient_checkpointing=True)
        model=get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",**config["lora"]))
        load_seconds=time.perf_counter()-load_start
        trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        if not trainable or any(".lora_" not in n for n in trainable):
            raise RuntimeError("Non-LoRA trainable parameters")
        targets=config["lora"]["target_modules"]
        matched=[n for n,module in model.named_modules() if hasattr(module,"lora_A")]
        expected_count=model.config.num_hidden_layers*len(targets)
        if len(matched)!=expected_count or any(n.rsplit(".",1)[-1] not in targets for n in matched):
            raise RuntimeError("LoRA target coverage mismatch")
        num_trainable,num_total=model.get_nb_trainable_parameters()
        write_json(out/"parameter_audit.json",{"trainable":num_trainable,"total_logical_parameters":num_total,
            "trainable_fraction":num_trainable/num_total,"trainable_names":list(trainable),
            "trainable_dtypes":sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
            "target_module_count":len(matched),"target_modules":matched,"lora":config["lora"],
            "base_is_4bit":bool(model.is_loaded_in_4bit)})
        frozen_before=parameter_hashes(model)
        write_json(out/"frozen_parameter_hashes_before.json",frozen_before)
        batch={k:v.to("cuda:0") for k,v in collator([train_data[0]]).items()}
        initial_loss,_=probe(model,batch)
        callback=AuditCallback(out/"training_log.jsonl",started,config)
        training=TrainingArguments(output_dir=str(out),**config["training"],seed=config["seed"],data_seed=config["seed"],
            logging_steps=1,logging_strategy="steps",logging_nan_inf_filter=False,
            save_strategy="steps",save_steps=steps,eval_strategy="no",report_to="none",
            remove_unused_columns=False,label_names=["labels"],dataloader_num_workers=0,
            prediction_loss_only=True)
        (out/"training_arguments.json").write_text(training.to_json_string())
        trainer=SQLTrainer(model=model,args=training,train_dataset=train_data,eval_dataset=val_data,
            data_collator=collator,processing_class=tokenizer,callbacks=[callback])
        before=trainer.evaluate(metric_key_prefix="before")
        result=trainer.train()
        after=trainer.evaluate(metric_key_prefix="after")
        trainer.save_model(str(out/"adapter"))
        trainer.save_state()
        tokenizer.save_pretrained(out/"tokenizer")
        model.gradient_checkpointing_disable()
        final_loss,last_logits=probe(model,batch)
        torch.save({"last_logits":last_logits,"loss":final_loss,"input_ids":batch["input_ids"].cpu(),
                    "attention_mask":batch["attention_mask"].cpu(),"labels":batch["labels"].cpu()},out/"reload_reference.pt")
        del last_logits
        frozen_after=parameter_hashes(model)
        if frozen_before!=frozen_after:
            raise RuntimeError("Frozen base parameters changed")
        updates={}
        for name,p in model.named_parameters():
            if name in trainable:
                delta=p.detach().cpu()-trainable[name]
                updates[name]={"max_abs_delta":delta.abs().max().item(),"changed":bool(torch.count_nonzero(delta))}
        if not any(x["changed"] for x in updates.values()):
            raise RuntimeError("LoRA parameters did not update")
        write_json(out/"parameter_updates.json",{"frozen_base_unchanged":True,"updates":updates,
            "gradient_checks":callback.gradient_checks})
        seq_to_id={tuple(f["input_ids"]):r["id"] for f,r in zip(train_data.features,train_rows)}
        seen=[seq_to_id[tuple(s)] for s in trainer.observed_train_sequences]
        write_json(out/"sample_trace.json",{"seen_train_ids":seen,"optimizer_steps":trainer.state.global_step,
                    "validation_ids":[r["id"] for r in val_rows],"test_used_for_training":False})
        if trainer.state.global_step!=steps or len(callback.gradient_checks)!=steps:
            raise RuntimeError("Optimizer step/gradient audit incomplete")
        metrics={"evidence_state":"executed","train":result.metrics,"validation_before":before,"validation_after":after,
            "probe_initial_loss":initial_loss,"probe_final_loss":final_loss,"optimizer_steps":trainer.state.global_step,
            "load_seconds":load_seconds,"elapsed_seconds":time.perf_counter()-started,
            "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
            "peak_process_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            "trainable_parameters":num_trainable,"trainable_fraction":num_trainable/num_total}
        write_json(out/"training_metrics.json",metrics)
        load_frozen_rows(config)
        write_json(out/"status.json",{"status":"trained_pending_reload","command":command})
        print(json.dumps({"output_dir":str(out),**metrics},ensure_ascii=False),flush=True)
        return 0
    except BaseException as exc:
        write_json(out/"status.json",{"status":"failed","error":repr(exc),"command":command,
                                     "elapsed_seconds":time.perf_counter()-started})
        raise

def main():
    def terminate(signum,frame):
        raise TimeoutError("SFT wall-clock budget exhausted")
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--mode",choices=["smoke", "full"],default="smoke")
    parser.add_argument("--max-steps",type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.mode == "full":
        from src.sft.full import train
        return train(args.config or "configs/sft_full.yaml", args.output_dir, max_steps=args.max_steps)
    args.config = args.config or "configs/sft.yaml"
    return run(args)

if __name__=="__main__":
    raise SystemExit(main())
