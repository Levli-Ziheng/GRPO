"""Stage 5: bounded full-data QLoRA and shared locked-test evaluation.

Artifacts: metrics.json (configuration/provenance/audits), training_curve.csv,
best_adapter/ and test_predictions.jsonl. Only validation loss selects weights.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from src.sft.dataset import SQLDataset, SQLCollator, load_frozen_rows
from src.sft.loss import assistant_loss
from src.sft.model import load_base, load_tokenizer
from src.sft.train import parameter_hashes, probe
from src.data.common import read_jsonl, sha256_file, build_messages
from src.data.validate import validate_dataset
from src.inference.generate import Generator, load_tasks, verify_model
from src.evaluation.evaluate import evaluate_record
from src.evaluation.metrics import summarize
from src.utils.config import load_yaml
from src.utils.check_env import collect_environment


def save_metrics(out, metrics):
    target=out/'metrics.json'
    temp=out/'metrics.json.tmp'
    temp.write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(target)


def context(config_path, out):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'context.json'
        subprocess.run([sys.executable,'.agents/skills/ai-experiment/scripts/capture_context.py',
            '--output',str(path),'--experiment-id',out.name,'--artifact',config_path,
            '--artifact','data/processed/cspider-v1/dataset_manifest.json'],check=True,capture_output=True)
        return json.loads(path.read_text())


def guard(config, started):
    if time.perf_counter()-started > config['budget']['max_seconds']:
        raise TimeoutError('Wall-clock budget exceeded')
    if torch.cuda.mem_get_info()[0] < config['budget']['gpu_safety_margin_gib']*2**30:
        raise RuntimeError('GPU safety margin exhausted')
    if psutil.virtual_memory().available < 2*2**30:
        raise MemoryError('System RAM below 2 GiB; stop before considering streaming fallback')


def preflight(config):
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Select exactly one CUDA_VISIBLE_DEVICES device')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < config['budget']['min_free_gpu_gib']*2**30:
        raise RuntimeError('Insufficient free GPU memory')
    if psutil.disk_usage('.').free < config['budget']['min_free_disk_gib']*2**30:
        raise RuntimeError('Insufficient disk space')
    torch.cuda.reset_peak_memory_stats()
    return gpu


def protected_hashes():
    return {str(p):sha256_file(p) for root in ('data/processed/cspider-smoke-v0',
        'outputs/baseline/base-cspider-smoke-v0-20260908') for p in Path(root).rglob('*') if p.is_file()}


class FullSQLTrainer(Trainer):
    """Mean SQL+EOS token loss per microbatch; Trainer averages microbatches.

    Batch size is fixed at one, so gradient accumulation minimizes the mean of
    per-example SQL-token means. Do not pass num_items_in_batch token counts.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=False
        self.sample_counts=Counter()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels=inputs['labels']
        if labels.shape[0]!=1: raise ValueError('Sample-mean loss requires microbatch size 1')
        outputs=model(**{k:v for k,v in inputs.items() if k!='labels'})
        loss=assistant_loss(outputs.logits,labels)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        if model.training:
            fingerprint=hashlib.sha256(inputs['input_ids'].detach().cpu().numpy().tobytes()).hexdigest()
            self.sample_counts[fingerprint]+=1
        return (loss,outputs) if return_outputs else loss


class FullAudit(TrainerCallback):
    def __init__(self,out,config,started,metrics):
        self.out,self.config,self.started,self.metrics=out,config,started,metrics
        self.best=math.inf
        self.best_step=None
        self.gradient_checks=0
        self.curve=out/'training_curve.csv'
        with self.curve.open('x',newline='') as h:
            csv.writer(h).writerow(['step','epoch','train_loss','validation_loss','learning_rate','grad_norm','elapsed_seconds'])
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        seen=0
        for name,p in model.named_parameters():
            if p.grad is not None:
                if not p.requires_grad or '.lora_' not in name or not torch.isfinite(p.grad).all():
                    raise RuntimeError('Unexpected or non-finite gradient: '+name)
                seen+=1
        if not seen: raise RuntimeError('No LoRA gradients')
        self.gradient_checks+=1
    def on_step_end(self,args,state,control,**kwargs):
        guard(self.config,self.started)
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        if any(not math.isfinite(v) for v in logs.values() if isinstance(v,(int,float))):
            raise RuntimeError('Non-finite metric')
        with self.curve.open('a',newline='') as h:
            csv.writer(h).writerow([state.global_step,state.epoch,logs.get('loss',''),logs.get('eval_loss',''),
                logs.get('learning_rate',''),logs.get('grad_norm',''),round(time.perf_counter()-self.started,3)])
    def on_evaluate(self,args,state,control,metrics=None,model=None,**kwargs):
        value=metrics.get('eval_loss')
        if value is None: return
        guard(self.config,self.started)
        self.metrics.setdefault('validation_curve',[]).append({'step':state.global_step,'epoch':state.epoch,'loss':value})
        # Step zero is a diagnostic baseline, never a candidate trained adapter.
        if state.global_step>0 and value<self.best:
            self.best,self.best_step=value,state.global_step
            model.save_pretrained(self.out/'best_adapter',safe_serialization=True)
            self.metrics['best_checkpoint']={'step':state.global_step,'validation_loss':value,
                'selection':'minimum validation sample-mean SQL+EOS cross-entropy; no test access'}
        save_metrics(self.out,self.metrics)


def train(config_path, output_dir=None, max_steps=None, smoke=False):
    config=load_yaml(config_path)
    if max_steps is not None: config['training']['max_steps']=max_steps
    if smoke:
        config['training'].update(max_steps=2,warmup_steps=0)
        config['evaluation_steps']=1
        config['logging_steps']=1
        config['budget']['max_seconds']=600
    steps=config['training']['max_steps']
    if not 1<=steps<=config['budget']['max_steps']: raise ValueError('Step budget exceeded')
    if config['training']['per_device_train_batch_size']!=1 or config['training']['per_device_eval_batch_size']!=1:
        raise ValueError('Use batch size 1')
    out=Path(output_dir or config['output_dir'])
    if out.exists():
        metrics=json.loads((out/'metrics.json').read_text())
        if metrics.get('status')!='prepared' or sorted(p.name for p in out.iterdir())!=['metrics.json']:
            raise FileExistsError('Refusing to overwrite an existing experiment')
    else:
        out.mkdir(parents=True,exist_ok=False)
        metrics={}
    started=time.perf_counter()
    def timeout(signum,frame): raise TimeoutError('SFT wall-clock budget exhausted')
    old_handler=signal.signal(signal.SIGALRM,timeout)
    signal.alarm(config['budget']['max_seconds'])
    try:
        metrics.update(status='preflight',configuration=config,command=' '.join(sys.argv),smoke=smoke,
            source=context(config_path,out),environment=collect_environment(),streaming_used=False)
        save_metrics(out,metrics)
        metrics['gpu_before']=preflight(config)
        model_cfg=load_yaml(config['model_config'])
        metrics['model_config']=model_cfg
        source=verify_model(model_cfg)
        hashes=protected_hashes()
        data_cfg=load_yaml(config['data_config'])
        validation,errors=validate_dataset(data_cfg,'full')
        if errors: raise ValueError(errors)
        metrics['data_validation']=validation
        train_rows,val_rows=load_frozen_rows(config)
        tokenizer=load_tokenizer(model_cfg)
        all_train=SQLDataset(train_rows,tokenizer,config['max_sequence_length'])
        all_val=SQLDataset(val_rows,tokenizer,config['max_sequence_length'])
        metrics['mask_checked']={'train':len(all_train),'validation':len(all_val),'max_tokens':max(len(f['input_ids']) for d in (all_train,all_val) for f in d.features)}
        if smoke:
            # Exercise the longest full-data sequence and the actual accumulation path.
            indices=sorted(range(len(all_train)),key=lambda i:len(all_train[i]['input_ids']),reverse=True)[:16]
            train_data=[all_train[i] for i in indices]
            val_data=sorted(all_val.features,key=lambda f:len(f['input_ids']),reverse=True)[:16]
        else: train_data,val_data=all_train,all_val
        set_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=get_peft_model(load_base(model_cfg),LoraConfig(task_type='CAUSAL_LM',**config['lora']))
        trainable,total=model.get_nb_trainable_parameters()
        if any('.lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
            raise RuntimeError('Unexpected trainable base parameters')
        frozen=parameter_hashes(model)
        adapter_before=parameter_hashes(model,False)
        adapter_before={n:h for n,h in adapter_before.items() if '.lora_' in n}
        audit=FullAudit(out,config,started,metrics)
        arguments=TrainingArguments(output_dir=str(out),**config['training'],seed=config['seed'],data_seed=config['seed'],
            logging_steps=config['logging_steps'],logging_strategy='steps',logging_nan_inf_filter=False,
            eval_strategy='steps',eval_steps=config['evaluation_steps'],save_strategy='no',report_to='none',
            remove_unused_columns=False,label_names=['labels'],dataloader_num_workers=0,dataloader_pin_memory=False,
            prediction_loss_only=True,disable_tqdm=True)
        metrics['training_arguments']=arguments.to_dict()
        trainer=FullSQLTrainer(model=model,args=arguments,train_dataset=train_data,eval_dataset=val_data,
            data_collator=SQLCollator(tokenizer.pad_token_id),processing_class=tokenizer,callbacks=[audit])
        metrics['validation_before']=trainer.evaluate()
        metrics['status']='training'
        save_metrics(out,metrics)
        result=trainer.train()
        if not audit.best_step: raise RuntimeError('No trained checkpoint evaluated')
        if trainer.state.global_step!=steps or audit.gradient_checks!=steps: raise RuntimeError('Incomplete training steps')
        if parameter_hashes(model)!=frozen: raise RuntimeError('Frozen base changed')
        state=load_file(str(out/'best_adapter'/'adapter_model.safetensors'))
        set_peft_model_state_dict(model,state)
        del state
        model.gradient_checkpointing_disable()
        batch={k:v.to('cuda:0') for k,v in SQLCollator(tokenizer.pad_token_id)([all_val[0]]).items()}
        reference_loss,reference_logits=probe(model,batch)
        adapter_after=parameter_hashes(model,False)
        changes=sum(adapter_after[n]!=h for n,h in adapter_before.items())
        if changes==0: raise RuntimeError('Adapter did not change')
        if any(sha256_file(p)!=h for p,h in hashes.items()): raise RuntimeError('Frozen smoke/baseline changed')
        load_frozen_rows(config)
        metrics.update(status='trained_pending_reload',train=result.metrics,optimizer_steps=trainer.state.global_step,
            gradient_checks=audit.gradient_checks,trainable_parameters=trainable,trainable_fraction=trainable/total,
            frozen_base_unchanged=True,changed_adapter_tensors=changes,frozen_smoke_baseline_unchanged=True,
            sample_coverage={'unique_token_sequences':len(trainer.sample_counts),'microbatches':sum(trainer.sample_counts.values()),
                'min_visits':min(trainer.sample_counts.values()),'max_visits':max(trainer.sample_counts.values())},
            reload_reference={'validation_id':val_rows[0]['id'],'loss':reference_loss,'last_logits_first32':reference_logits[0,:32].tolist()},
            adapter_sha256=sha256_file(out/'best_adapter'/'adapter_model.safetensors'),
            elapsed_seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        save_metrics(out,metrics)
        print(json.dumps({k:metrics[k] for k in ('status','best_checkpoint','train','sample_coverage','peak_allocated_gib','elapsed_seconds')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        metrics.update(status='failed',error=repr(exc),elapsed_seconds=time.perf_counter()-started)
        save_metrics(out,metrics)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_handler)
    return 0


def verify(out):
    out=Path(out)
    m=json.loads((out/'metrics.json').read_text())
    if m['status']!='trained_pending_reload': raise ValueError('Expected trained_pending_reload')
    c=m['configuration']
    preflight(c)
    if sha256_file(out/'best_adapter'/'adapter_model.safetensors')!=m['adapter_sha256']: raise ValueError('Adapter drift')
    _,val=load_frozen_rows(c)
    tokenizer=load_tokenizer(m['model_config'])
    data=SQLDataset(val[:1],tokenizer,c['max_sequence_length'])
    model=PeftModel.from_pretrained(load_base(m['model_config'],False),out/'best_adapter',is_trainable=False)
    batch={k:v.to('cuda:0') for k,v in SQLCollator(tokenizer.pad_token_id)([data[0]]).items()}
    loss,logits=probe(model,batch)
    reference=m['reload_reference']
    delta=float((logits[0,:32]-torch.tensor(reference['last_logits_first32'])).abs().max())
    if abs(loss-reference['loss'])>1e-5 or delta>1e-5: raise RuntimeError('Fresh-process adapter reload mismatch')
    m.update(status='verified',reload_verification={'loss':loss,'loss_delta':abs(loss-reference['loss']),
        'selected_logits_max_abs_delta':delta,'validation_id':val[0]['id'],'fresh_process':True})
    save_metrics(out,m)
    print(json.dumps(m['reload_verification']),flush=True)
    return 0


def evaluate(out, stage, resume=False, allow_budget_truncation=False):
    out=Path(out)
    m=json.loads((out/'metrics.json').read_text())
    if m['status'] not in ('verified','evaluating','completed'): raise ValueError('Verify independent reload before evaluation')
    if stage in m.get('evaluation',{}): raise FileExistsError('Stage already evaluated')
    c=m['configuration']
    if m.get('smoke'): raise ValueError('No test evaluation for smoke selection')
    preflight(c)
    cfg=load_yaml(c['baseline_config'])
    data_cfg=load_yaml(c['data_config'])
    smoke_tasks=load_tasks(cfg,data_cfg)
    full_cfg=copy.deepcopy(cfg)
    root=Path(c['dataset_root'])
    full_cfg.update(dataset=str(root/'canonical_test.jsonl'),dataset_manifest=str(root/'dataset_manifest.json'),expected_samples=300)
    full_tasks=load_tasks(full_cfg,data_cfg)
    # Union generates overlapping test IDs once; both frozen splits are scored separately.
    tasks={r['id']:r for r in full_tasks}
    for row in smoke_tasks:
        if row['id'] in tasks:
            if any(tasks[row['id']][k]!=row[k] for k in ('question','schema','gold_sql','db_id','gold_result')):
                raise ValueError('Different tasks under same ID')
        tasks[row['id']]=row
    source=verify_model(m['model_config'])
    set_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    def timeout(signum,frame): raise TimeoutError('Evaluation wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(c['evaluation_budget_seconds'])
    existing=list(read_jsonl(out/'test_predictions.jsonl')) if (out/'test_predictions.jsonl').exists() else []
    retained=[p for p in existing if p['stage']==stage]
    if retained and not resume: raise FileExistsError('Partial predictions exist; explicit --resume required')
    if [p['id'] for p in retained]!=list(tasks)[:len(retained)]: raise ValueError('Partial prediction IDs are not the exact task prefix')
    for prediction in retained:
        task=tasks[prediction['id']]
        checked=evaluate_record(prediction,task,cfg['evaluation'])
        if any(checked[k]!=prediction[k] for k in ('execution_correct','execution_success','format_valid','predicted_result')):
            raise ValueError('Partial prediction CPU recheck failed')
    if stage in m.get('evaluation_attempts',{}):
        m.setdefault('evaluation_history',[]).append({'stage':stage,**m['evaluation_attempts'][stage]})
    m['status']='evaluating'
    m['evaluation_source']=context('configs/sft_full.yaml',out)
    m.setdefault('evaluation_attempts',{})[stage]={'status':'running','decoding':cfg['generation'],
        'unique_test_samples':len(tasks),'retained_prefix_count':len(retained),
        'command':' '.join(sys.argv),'length_budget_policy':'record budget hits and apply unchanged SQL scoring' if allow_budget_truncation else 'stop for diagnosis'}
    save_metrics(out,m)
    try:
        generator=Generator(m['model_config'],cfg['generation'],source)
        if stage=='sft':
            if sha256_file(out/'best_adapter'/'adapter_model.safetensors')!=m['adapter_sha256']: raise ValueError('Adapter drift')
            generator.model=PeftModel.from_pretrained(generator.model,out/'best_adapter',is_trainable=False).eval()
        predictions=list(retained)
        with (out/'test_predictions.jsonl').open('a',encoding='utf-8') as handle:
            for idx,row in enumerate(tasks.values(),1):
                if idx<=len(retained): continue
                guard(c,started)
                prediction=generator.generate(build_messages(data_cfg,row['schema'],row['question']))
                prediction.update(id=row['id'],stage=stage,model=m['model_config']['model']['id'],model_revision=m['model_config']['model']['revision'],
                    test_splits=[name for name,rows in (('formal300',full_tasks),('locked40',smoke_tasks)) if row['id'] in {r['id'] for r in rows}])
                prediction=evaluate_record(prediction,row,cfg['evaluation'])
                # Schema-bearing prompt can be reconstructed from immutable source/config.
                prediction.pop('prompt',None)
                handle.write(json.dumps(prediction,ensure_ascii=False)+'\n')
                handle.flush()
                predictions.append(prediction)
                if idx%25==0 or idx==len(tasks): print(f'{stage}: {idx}/{len(tasks)}',flush=True)
                if prediction['truncated'] and not allow_budget_truncation:
                    raise RuntimeError('Output truncated; diagnose before proceeding')
        by_id={p['id']:p for p in predictions}
        m.setdefault('evaluation',{})[stage]={name:summarize([by_id[r['id']] for r in rows]) for name,rows in (('formal300',full_tasks),('locked40',smoke_tasks))}
        m['evaluation'][stage]['runtime']={'seconds_this_process':time.perf_counter()-started,'retained_prefix_count':len(retained),
            'total_generation_seconds':sum(p['latency_seconds'] for p in predictions),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        m['evaluation_attempts'][stage]['status']='completed'
        load_tasks(cfg,data_cfg)
        load_tasks(full_cfg,data_cfg)
        m['status']='completed' if set(m['evaluation'])=={'base','sft'} else 'evaluating'
        save_metrics(out,m)
        print(json.dumps(m['evaluation'][stage],ensure_ascii=False),flush=True)
    except BaseException as exc:
        m['evaluation_attempts'][stage].update(status='failed',error=repr(exc))
        save_metrics(out,m)
        raise
    finally: signal.alarm(0)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['train','verify','evaluate'],required=True)
    parser.add_argument('--config',default='configs/sft_full.yaml')
    parser.add_argument('--run-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stage',choices=['base','sft'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-budget-truncation',action='store_true')
    args=parser.parse_args()
    if args.action=='train': return train(args.config,args.run_dir,smoke=args.smoke)
    if not args.run_dir: parser.error('--run-dir required')
    if args.action=='verify': return verify(args.run_dir)
    if not args.stage: parser.error('--stage required for evaluate')
    return evaluate(args.run_dir,args.stage,args.resume,args.allow_budget_truncation)

if __name__=='__main__': raise SystemExit(main())
