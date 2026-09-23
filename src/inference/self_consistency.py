"""Execution-result self-consistency for Text-to-SQL validation/test inference."""
from __future__ import annotations

import argparse
import copy
import json
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import GenerationConfig, set_seed

from src.data.common import build_messages
from src.evaluation.evaluate import evaluate_record
from src.evaluation.metrics import summarize
from src.grpo.full import generator_for, load_split
from src.utils.config import load_yaml, write_json


def result_key(candidate):
    if not candidate.get('execution_success'):
        return None
    return json.dumps(candidate.get('predicted_result'), ensure_ascii=False,
                      sort_keys=True, separators=(',', ':'))


def select_by_result_vote(candidates):
    """Choose the most common executable result; ties prefer greedy/index 0."""
    if not candidates:
        raise ValueError('At least one candidate is required')
    groups = defaultdict(list)
    for index, candidate in enumerate(candidates):
        key = result_key(candidate)
        if key is not None:
            groups[key].append(index)
    if not groups:
        return 0, 0
    indices = max(groups.values(), key=lambda group: (len(group), 0 in group, -min(group)))
    return (0 if 0 in indices else min(indices)), len(indices)


def generate_sampled(generator, messages, count, temperature, top_p, seed):
    tokenizer, model, device = generator.tokenizer, generator.model, generator.device
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors='pt').to(device)
    prompt_tokens = inputs['input_ids'].shape[-1]
    allowance = min(generator.config['max_new_tokens'],
                    generator.config['max_sequence_length'] - prompt_tokens)
    if allowance <= 0:
        raise ValueError('Prompt exceeds context budget')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    decoding = GenerationConfig(do_sample=True, num_beams=1, num_return_sequences=count,
        temperature=temperature, top_p=top_p, top_k=0, use_cache=True,
        max_new_tokens=allowance, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=decoding)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result=[]
    for output in outputs:
        tokens=output[prompt_tokens:].tolist()
        result.append({'raw_output':tokenizer.decode(tokens,skip_special_tokens=True).strip(),
            'input_tokens':prompt_tokens,'output_tokens':len(tokens),
            'max_new_tokens_effective':allowance,'latency_seconds':elapsed/count,
            'truncated':len(tokens)>=allowance and tokens[-1]!=tokenizer.eos_token_id})
    return result


def run(args):
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source_run)
    metrics=json.loads((source/'metrics.json').read_text())
    c=metrics['config']; data_config=load_yaml(c['data_config']); baseline=load_yaml('configs/baseline.yaml')
    tasks=load_split(c['full_dataset_root'],args.split,data_config)
    if args.max_tasks is not None:
        tasks=tasks[:args.max_tasks]
    if args.num_candidates < 2:
        raise ValueError('num_candidates must include greedy plus at least one sample')
    command=' '.join(args._argv)
    state={'status':'running','split':args.split,'task_count':len(tasks),
        'num_candidates':args.num_candidates,'temperature':args.temperature,'top_p':args.top_p,
        'seed':c['seed'],'source_run':str(source),'command':command}
    write_json(out/'metrics.json',state)
    context=[sys.executable,'.agents/skills/ai-experiment/scripts/capture_context.py',
        '--output',str(out/'run_context.json'),'--experiment-id',out.name,
        '--artifact','configs/baseline.yaml','--artifact',c['data_config'],
        '--artifact',str(Path(c['full_dataset_root'])/'dataset_manifest.json'),
        '--command',command]
    subprocess.run(context,check=True)
    def timeout(*_):
        raise TimeoutError('Self-consistency wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(args.max_seconds)
    started=time.perf_counter()
    try:
        set_seed(c['seed'])
        generator=generator_for(metrics,baseline)
        selected=[]; oracle_correct=0; executable_candidates=0
        with (out/'candidates.jsonl').open('x',encoding='utf-8') as all_handle, \
             (out/'predictions.jsonl').open('x',encoding='utf-8') as selected_handle:
            for task_index,task in enumerate(tasks):
                messages=build_messages(data_config,task['schema'],task['question'])
                greedy=generator.generate(messages)
                samples=generate_sampled(generator,messages,args.num_candidates-1,
                    args.temperature,args.top_p,c['seed']+task_index+1)
                candidates=[]
                for index,candidate in enumerate([greedy,*samples]):
                    candidate.update(id=task['id'],stage='sft_self_consistency_candidate',
                                     candidate_index=index)
                    candidate=evaluate_record(candidate,task,baseline['evaluation'])
                    candidates.append(candidate)
                    executable_candidates += int(candidate['execution_success'])
                chosen_index,votes=select_by_result_vote(candidates)
                choice=copy.deepcopy(candidates[chosen_index])
                choice.update(stage='sft_self_consistency',selected_candidate_index=chosen_index,
                              result_vote_count=votes,candidate_count=len(candidates))
                selected.append(choice)
                oracle_correct += int(any(x['execution_correct'] for x in candidates))
                all_handle.write(json.dumps({'id':task['id'],'candidates':candidates},ensure_ascii=False)+'\n')
                selected_handle.write(json.dumps(choice,ensure_ascii=False)+'\n')
                all_handle.flush(); selected_handle.flush()
                if (task_index+1)%10==0 or task_index+1==len(tasks):
                    print(f'{task_index+1}/{len(tasks)} selected_correct={sum(x["execution_correct"] for x in selected)} '
                          f'oracle={oracle_correct}',flush=True)
        state.update(status='completed',metrics=summarize(selected),oracle_pass_at_k=oracle_correct/len(tasks),
            oracle_correct=oracle_correct,executable_candidate_rate=executable_candidates/(len(tasks)*args.num_candidates),
            elapsed_seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        signal.alarm(0)
        write_json(out/'metrics.json',state)
    print(json.dumps(state,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--num-candidates',type=int,default=8)
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--top-p',type=float,default=0.95)
    p.add_argument('--max-tasks',type=int)
    p.add_argument('--max-seconds',type=int,default=3600)
    args=p.parse_args(); args._argv=['python','-m','src.inference.self_consistency',*__import__('sys').argv[1:]]
    run(args)


if __name__=='__main__': main()
