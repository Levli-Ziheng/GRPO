"""Stage 7: bounded five-step GRPO smoke and fresh-process adapter verification."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainerCallback, GenerationConfig, set_seed
from trl import GRPOConfig, GRPOTrainer
from src.data.common import read_jsonl, sha256_file, build_messages
from src.grpo.bridge import (load_rows, prioritize_rows, RewardAudit, arrow_rows,
                             entropy_guided_advantages, proportional_interleave_rows)
from src.sft.model import load_base, load_tokenizer
from src.sft.train import parameter_hashes
from src.inference.generate import verify_model, Generator
from src.evaluation.evaluate import evaluate_record
from src.utils.config import load_yaml, write_json


def gpu_preflight(c):
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Select exactly one CUDA_VISIBLE_DEVICES')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    if shutil.disk_usage('.').free < c['budget']['min_free_disk_gib']*2**30:
        raise RuntimeError('Insufficient disk space')
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < c['budget']['min_free_gpu_gib']*2**30:
        raise RuntimeError('Insufficient GPU memory')
    torch.cuda.reset_peak_memory_stats()
    return gpu


class AuditCallback(TrainerCallback):
    def __init__(self,out,c):
        self.out,self.c = out,c
        self.steps, self.nonzero_steps = 0,0
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        norm2 = 0.0
        for n,p in model.named_parameters():
            if p.grad is None: continue
            if not p.requires_grad or '.lora_' not in n or '.default.' not in n:
                raise RuntimeError('Unexpected gradient: '+n)
            if not torch.isfinite(p.grad).all(): raise RuntimeError('Non-finite gradient')
            norm2 += p.grad.float().square().sum().item()
        self.steps += 1
        self.nonzero_steps += int(norm2 > 0)
        with (self.out/'gradients.jsonl').open('a') as h:
            h.write(json.dumps({'step':self.steps,'gradient_norm':math.sqrt(norm2)})+'\n')
    def on_step_end(self,args,state,control,**kwargs):
        if torch.cuda.mem_get_info()[0] < self.c['budget']['gpu_safety_margin_gib']*2**30:
            raise RuntimeError('GPU safety margin exhausted')
        interval=int(self.c.get('checkpoint_interval',25))
        if interval <= 0:
            raise ValueError('checkpoint_interval must be positive')
        if self.c.get('mode')=='full' and state.global_step % interval == 0:
            kwargs['model'].save_pretrained(self.out/f'adapter-step-{state.global_step}',
                                            selected_adapters=['default'],safe_serialization=True)
    def on_log(self,args,state,control,logs=None,**kwargs):
        if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in (logs or {}).values()):
            raise RuntimeError('Non-finite training metric')
        with (self.out/'training_metrics.jsonl').open('a') as h:
            h.write(json.dumps({'step':state.global_step,**(logs or {})})+'\n')


class RLZVPGRPOTrainer(GRPOTrainer):
    """Use entropy-guided token advantages for zero-variance prompt groups."""

    def __init__(self, *args, reward_audit, rl_zvp_alpha, rl_zvp_enabled=True, **kwargs):
        self.reward_audit = reward_audit
        self.rl_zvp_alpha = float(rl_zvp_alpha)
        self.rl_zvp_enabled = bool(rl_zvp_enabled)
        self.rl_zvp_groups = 0
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        if len(inputs) != self.num_generations or len({row['id'] for row in inputs}) != 1:
            raise RuntimeError('RL-ZVP requires exactly one complete GRPO group per generation batch')
        prepared = super()._generate_and_score_completions(inputs)
        record = self.reward_audit.last_group_record
        if record is None or record['id'] != inputs[0]['id']:
            raise RuntimeError('Reward audit and generated group are out of sync')
        kind = 0
        if record['zero_variance']:
            if self.rl_zvp_enabled:
                kind = 1 if record['all_execution_correct'] else -1
                self.rl_zvp_groups += 1
            else:
                self.reward_audit.skipped_zero_groups += 1
        prepared['zvp_kind'] = torch.full_like(prepared['advantages'], kind, dtype=torch.int8)
        return prepared

    def _compute_loss(self, model, inputs):
        kinds = inputs.get('zvp_kind')
        if kinds is None or not torch.any(kinds != 0):
            return super()._compute_loss(model, inputs)
        prompt_ids, prompt_mask = inputs['prompt_ids'], inputs['prompt_mask']
        completion_ids, completion_mask = inputs['completion_ids'], inputs['completion_mask']
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        with torch.no_grad():
            _, entropies, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, completion_ids.size(1),
                batch_size=self.args.per_device_train_batch_size,
                compute_entropy=True, compute_aux_loss=False,
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                num_images=inputs.get('num_images'),
                pixel_attention_mask=inputs.get('pixel_attention_mask'),
                spatial_shapes=inputs.get('spatial_shapes'),
                num_tiles=inputs.get('num_tiles'), image_sizes=inputs.get('image_sizes'),
                token_type_ids=inputs.get('token_type_ids'),
                mm_token_type_ids=inputs.get('mm_token_type_ids'),
                image_position_ids=inputs.get('image_position_ids'))
        shaped = entropy_guided_advantages(entropies, completion_mask, kinds, self.rl_zvp_alpha)
        ordinary = inputs['advantages']
        if ordinary.ndim == 1:
            ordinary = ordinary.unsqueeze(1).expand_as(shaped)
        effective = torch.where(kinds.unsqueeze(1) == 0, ordinary, shaped)
        active = completion_mask.bool()
        with (Path(self.args.output_dir)/'rl_zvp_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({
                'kind': 'positive' if torch.all(kinds > 0) else 'negative',
                'alpha': self.rl_zvp_alpha,
                'mean_token_entropy': entropies[active].float().mean().item(),
                'mean_token_advantage': effective[active].float().mean().item(),
                'max_abs_token_advantage': effective[active].float().abs().max().item(),
            })+'\n')
        return super()._compute_loss(model, dict(inputs, advantages=effective))


def probe(model,tokenizer,prompt):
    model.eval()
    tokens = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=True,
                                          return_dict=True,return_tensors='pt').to('cuda:0')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        result = model(**tokens,use_cache=False).logits[0,-1,:].float().cpu()
    return result


def restore_sft_precision(model,original):
    # TRL casts trainable QLoRA to BF16. Restore FP32 AND original values after
    # construction, before optimizer creation, so initial policy equals SFT.
    if 'ref' not in model.peft_config:
        raise RuntimeError('Missing frozen SFT reference adapter')
    for name,p in model.named_parameters():
        if '.lora_' in name: p.data = p.data.float()
    set_peft_model_state_dict(model,original,adapter_name='default')
    set_peft_model_state_dict(model,original,adapter_name='ref')
    model.set_adapter('default')
    for name,p in model.named_parameters():
        expected = '.lora_' in name and '.default.' in name
        if p.requires_grad != expected:
            raise RuntimeError('Unexpected trainable state: '+name)
    for name,p in model.named_parameters():
        if '.lora_' in name and '.default.' in name:
            if not torch.equal(p,model.get_parameter(name.replace('.default.','.ref.'))):
                raise RuntimeError('SFT reference initialization mismatch')


def run(args):
    c=load_yaml(args.config)
    c['mode']=args.mode
    budget_key='formal' if args.mode=='full' else 'smoke'
    steps=args.max_steps if args.max_steps is not None else c['budget'][budget_key+'_max_steps']
    if not 1<=steps<=c['budget'][budget_key+'_max_steps']:
        raise ValueError('Step budget exceeded')
    c['training']['max_steps']=steps
    zvp=c.get('rl_zvp',{})
    if zvp.get('enabled',False) and float(zvp.get('alpha',0)) <= 0:
        raise ValueError('Enabled RL-ZVP requires a positive alpha')
    root=c['full_dataset_root'] if args.mode=='full' else c['smoke_dataset_root']
    c['dataset_root']=root
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    m={'status':'running','mode':args.mode,'config':c,'steps_requested':steps,
       'command':'CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','')+' '+shlex.join([sys.executable,'-m','src.grpo.train',*sys.argv[1:]])}
    callback=None
    def timeout(signum,frame): raise TimeoutError('Training wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(c['budget'][budget_key+'_max_seconds'])
    try:
        m['gpu_before']=gpu_preflight(c)
        mc=load_yaml(c['model_config'])
        m['model_config']=mc
        verify_model(mc)
        expected=json.loads((Path(c['adapter_path']).parent/'metrics.json').read_text())['adapter_sha256']
        if sha256_file(Path(c['adapter_path'])/'adapter_model.safetensors')!=expected:
            raise ValueError('Stage5 adapter drift')
        m['sft_adapter_sha256']=expected
        subprocess.run([sys.executable,'.agents/skills/ai-experiment/scripts/capture_context.py',
            '--output',str(out/'run_context.json'),'--experiment-id',out.name,'--artifact',args.config,
            '--artifact',root+'/dataset_manifest.json','--command',m['command']],check=True)
        subprocess.run([sys.executable,'-m','src.utils.check_env','--json-out',str(out/'environment.json')],
                       check=True,stdout=subprocess.DEVNULL)
        set_seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        tokenizer=load_tokenizer(mc)
        tokenizer.padding_side='left'
        rows=load_rows(root,load_yaml(c['data_config']),tokenizer,
                       c['max_sequence_length'],c['training']['max_completion_length'])
        strategy=c.get('sampling_strategy','priority')
        if strategy=='priority':
            rows=prioritize_rows(rows,tuple(c['difficulty_priority']))
        elif strategy=='proportional_interleave':
            rows=proportional_interleave_rows(rows,tuple(c['difficulty_priority']))
        else:
            raise ValueError('Unknown sampling_strategy: '+str(strategy))
        m['train_pool_size']=len(rows)
        m['difficulty_priority']=c['difficulty_priority']
        m['sampling_strategy']=strategy
        model=PeftModel.from_pretrained(load_base(mc),c['adapter_path'],is_trainable=True)
        original={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        reward=RewardAudit(load_yaml(c['reward_config']),out/'rollouts.jsonl',c['training']['num_generations'],
                           c['training']['max_completion_length'],tokenizer.eos_token_id)
        callback=AuditCallback(out,c)
        training=dict(c['training'],max_steps=steps,seed=c['seed'],data_seed=c['seed'],
                      logging_steps=1,disable_tqdm=True,dataloader_num_workers=0,
                      dataloader_pin_memory=False,disable_dropout=True)
        encoded_rows=arrow_rows(rows)
        trainer=RLZVPGRPOTrainer(model=model,args=GRPOConfig(output_dir=str(out),**training),
                            reward_funcs=reward,train_dataset=Dataset.from_list(encoded_rows),
                            processing_class=tokenizer,callbacks=[callback],reward_audit=reward,
                            rl_zvp_enabled=zvp.get('enabled',False),
                            rl_zvp_alpha=zvp.get('alpha',0.1))
        restore_sft_precision(model,original)
        del original
        frozen=parameter_hashes(model)
        before={n:h for n,h in parameter_hashes(model,False).items() if '.default.' in n}
        m.update(adapter_dtype='float32',reference='frozen stage5 SFT ref adapter',
                 reference_matches_initial_policy=True,advantage='sample std (ddof=1), epsilon=1e-4; TRL GRPO loss')
        m['advantage_method']='RL-ZVP entropy-guided token advantage' if zvp.get('enabled',False) else 'GRPO'
        write_json(out/'metrics.json',m)
        try:
            result=trainer.train()
            train_metrics=result.metrics
            m['stop_reason']=None
        except RuntimeError as exc:
            # These guards fire before the failing rollout gets a backward/update.
            # Preserve the last completed policy; never relax the guard to run longer.
            if args.mode!='full' or str(exc) not in (
                'Truncated training completion; diagnose before expansion',
                'Unsafe SQL generated'):
                raise
            m['stop_reason']=str(exc)
            train_metrics={'controlled_early_stop':True,'completed_steps':trainer.state.global_step}
        if (m['stop_reason'] is None and callback.steps != steps) or callback.nonzero_steps == 0:
            raise RuntimeError('No verified policy-learning steps')
        after=parameter_hashes(model,False)
        changed=sum(after[n]!=h for n,h in before.items())
        if changed==0: raise RuntimeError('No policy adapter updates')
        if parameter_hashes(model)!=frozen: raise RuntimeError('Frozen Base/reference changed')
        model.gradient_checkpointing_disable()
        reference=probe(model,tokenizer,rows[0]['prompt'])
        torch.save(reference,out/'reload_logits.pt')
        model.save_pretrained(out/'adapter',selected_adapters=['default'],safe_serialization=True)
        m.update(status='trained_pending_reload',train=train_metrics,optimizer_steps=trainer.state.global_step,
                 nonzero_gradient_steps=callback.nonzero_steps,changed_adapter_tensors=changed,
                 frozen_base_reference_unchanged=True,reward_groups=reward.groups,
                 zero_variance_groups=reward.zero_groups,
                 skipped_zero_variance_groups=reward.skipped_zero_groups,
                 rl_zvp_groups=trainer.rl_zvp_groups,
                 positive_zero_variance_groups=reward.positive_zero_groups,
                 negative_zero_variance_groups=reward.negative_zero_groups,
                 sampled_difficulties=reward.sampled_difficulties,
                 adapter_sha256=sha256_file(out/'adapter'/'adapter_model.safetensors'),
                 reload_prompt_id=rows[0]['id'])
        m['checkpoint_policy']='adapter-only snapshots at steps 25/50 and terminal policy; no optimizer resume'
        m['checkpoints']=[{'step':int(p.name.split('-')[-1]),'path':str(p),
            'sha256':sha256_file(p/'adapter_model.safetensors')} for p in sorted(out.glob('adapter-step-*'))]
        if not any(x['step']==trainer.state.global_step for x in m['checkpoints']):
            m['checkpoints'].append({'step':trainer.state.global_step,'path':str(out/'adapter'),
                                     'sha256':m['adapter_sha256']})
        load_rows(root,load_yaml(c['data_config']),tokenizer,
                  c['max_sequence_length'],c['training']['max_completion_length'])
        if sha256_file(Path(c['adapter_path'])/'adapter_model.safetensors')!=expected:
            raise RuntimeError('Stage5 adapter changed')
    except BaseException as exc:
        m.update(status='failed',error=repr(exc),optimizer_steps=callback.steps if callback else 0)
        raise
    finally:
        signal.alarm(0)
        m.update(elapsed_seconds=time.perf_counter()-started,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_initialized() else 0,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_initialized() else 0,
                 peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        write_json(out/'metrics.json',m)
        print(json.dumps(m,ensure_ascii=False),flush=True)


def verify(args):
    out=Path(args.output_dir)
    m=json.loads((out/'metrics.json').read_text())
    if m['status']!='trained_pending_reload': raise ValueError('Expected trained_pending_reload')
    c=m['config']
    gpu_preflight(c)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if sha256_file(out/'adapter'/'adapter_model.safetensors')!=m['adapter_sha256']:
        raise ValueError('Adapter hash mismatch')
    tokenizer=load_tokenizer(m['model_config'])
    root=c.get('dataset_root',c['smoke_dataset_root'])
    rows=load_rows(root,load_yaml(c['data_config']),tokenizer)
    model=PeftModel.from_pretrained(load_base(m['model_config'],False),out/'adapter',is_trainable=False).eval()
    reload_row=next((row for row in rows if row['id']==m['reload_prompt_id']),None)
    if reload_row is None: raise ValueError('Reload prompt ID missing from frozen data')
    logits=probe(model,tokenizer,reload_row['prompt'])
    expected=torch.load(out/'reload_logits.pt',map_location='cpu',weights_only=True)
    delta=(logits-expected).abs().max().item()
    if delta>1e-5: raise RuntimeError('Fresh reload logits mismatch: '+str(delta))
    baseline=load_yaml('configs/baseline.yaml')
    task=next(read_jsonl(Path(root)/'canonical_validation.jsonl'))
    generator=Generator.__new__(Generator)
    generator.torch=torch; generator.model=model; generator.tokenizer=tokenizer
    generator.device=torch.device('cuda:0'); generator.config=baseline['generation']
    generator.decoding=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    prediction=generator.generate(build_messages(load_yaml(c['data_config']),task['schema'],task['question']))
    prediction=evaluate_record(prediction,task,baseline['evaluation'])
    write_json(out/'reload_generation.json',prediction)
    if not prediction['raw_output'] or prediction['truncated']:
        raise RuntimeError('Reload generation empty/truncated')
    m.update(status='completed',reload={'fresh_process':True,'max_abs_logits_difference':delta,
        'validation_id':task['id'],'execution_correct':prediction['execution_correct']})
    write_json(out/'metrics.json',m)
    print(json.dumps(m['reload'],ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/grpo_stage7.yaml')
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--max-steps',type=int)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    return verify(args) if args.verify else run(args)


if __name__=='__main__': main()
