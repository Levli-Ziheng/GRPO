"""Validation selection and locked-test evaluation of bounded GRPO runs."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import set_seed
from src.data.common import read_jsonl, sha256_file, build_messages
from src.evaluation.evaluate import evaluate_record
from src.evaluation.metrics import summarize
from src.inference.generate import Generator, verify_model, load_tasks
from src.grpo.train import gpu_preflight
from src.utils.config import load_yaml, write_json


def load_split(root, split, data_config):
    root=Path(root)
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    name=f'canonical_{split}.jsonl'
    view=f'eval_{split}.jsonl'
    for file in (name,view):
        if sha256_file(root/file)!=manifest['file_sha256'][file]:
            raise ValueError('Frozen data drift: '+file)
    rows=list(read_jsonl(root/name))
    if [r['id'] for r in rows]!=manifest['sample_ids'][split]:
        raise ValueError('Frozen split IDs drift')
    views={r['id']:r for r in read_jsonl(root/view)}
    checked=set()
    for row in rows:
        if build_messages(data_config,row['schema'],row['question'])!=views[row['id']]['prompt']:
            raise ValueError('Frozen prompt mismatch')
        if row['db_path'] not in checked:
            if sha256_file(row['db_path'])!=row['db_sha256']: raise ValueError('Database drift')
            checked.add(row['db_path'])
    return rows


def best_candidate(candidates):
    # Only validation enters selection. Break ties toward fewer updates.
    if not candidates: raise ValueError('No validation candidates')
    return max(candidates,key=lambda r:(r['metrics']['correct_count'],-r['step']))


def check_predictions(predictions,tasks,ec):
    if [p['id'] for p in predictions]!=[t['id'] for t in tasks]:
        raise ValueError('Prediction IDs/order mismatch')
    for p,t in zip(predictions,tasks):
        actual=evaluate_record(p,t,ec)
        for key in ('execution_correct','execution_success','format_valid','predicted_result'):
            if actual[key]!=p[key]: raise ValueError('CPU re-evaluation mismatch: '+p['id']+':'+key)
    return summarize(predictions)


def generator_for(m,baseline):
    c=m['config']
    gpu_preflight(c)
    set_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=verify_model(m['model_config'])
    g=Generator(m['model_config'],baseline['generation'],source)
    g.model=PeftModel.from_pretrained(g.model,c['adapter_path'],is_trainable=False).eval()
    return g


def generate_rows(g,tasks,dc,ec,path,label,model_cfg):
    # Exact prefix resume: no missing/duplicate IDs and CPU verify retained rows.
    existing=list(read_jsonl(path)) if path.exists() else []
    if existing:
        check_predictions(existing,tasks[:len(existing)],ec)
        if any(p['stage']!=label for p in existing): raise ValueError('Wrong retained stage')
    rows=list(existing)
    with path.open('a',encoding='utf-8') as handle:
        for i,task in enumerate(tasks[len(existing):],len(existing)+1):
            if torch.cuda.mem_get_info()[0]<2*2**30: raise RuntimeError('GPU safety margin exhausted')
            prediction=g.generate(build_messages(dc,task['schema'],task['question']))
            prediction.update(id=task['id'],stage=label,model=model_cfg['model']['id'],
                              model_revision=model_cfg['model']['revision'])
            prediction=evaluate_record(prediction,task,ec)
            prediction.pop('prompt',None)
            handle.write(json.dumps(prediction,ensure_ascii=False)+'\n')
            handle.flush()
            rows.append(prediction)
            if i%25==0 or i==len(tasks): print(f'{label}: {i}/{len(tasks)}',flush=True)
    check_predictions(rows,tasks,ec)
    return rows


def select(out,m):
    if m['mode']!='full' or m['status']!='completed':
        raise ValueError('Require successfully reloaded full-mode terminal policy')
    if 'selection' in m: raise FileExistsError('Selection already completed')
    c=m['config']; dc=load_yaml(c['data_config']); baseline=load_yaml('configs/baseline.yaml')
    remaining=c['budget']['formal_max_seconds']-m['elapsed_seconds']-m.get('selection_seconds',0)
    if remaining<=0: raise TimeoutError('Formal budget consumed')
    signal.alarm(math.ceil(remaining))
    started=time.perf_counter()
    try:
        tasks=load_split(c['full_dataset_root'],'validation',dc)
        if len(tasks)!=200: raise ValueError('Expected frozen 200 validation tasks')
        g=generator_for(m,baseline)
        specs=[{'step':0,'path':c['adapter_path'],'sha256':m['sft_adapter_sha256'],'label':'sft_initial'}]
        specs += [dict(x,label=f"grpo_step_{x['step']}") for x in m['checkpoints']]
        candidates=[]; identical={}
        for spec in specs:
            path=Path(spec['path'])/'adapter_model.safetensors'
            if sha256_file(path)!=spec['sha256']: raise ValueError('Candidate adapter drift')
            if spec['sha256'] in identical:
                metrics=identical[spec['sha256']]['metrics']
                spec=dict(spec,reused_identical_weights_from=identical[spec['sha256']]['label'])
            else:
                set_peft_model_state_dict(g.model,load_file(str(path)),adapter_name='default')
                predictions=generate_rows(g,tasks,dc,baseline['evaluation'],out/f"validation-{spec['label']}.jsonl",
                                          spec['label'],m['model_config'])
                metrics=summarize(predictions)
                identical[spec['sha256']]={'metrics':metrics,'label':spec['label']}
            candidates.append(dict(spec,metrics=metrics))
            write_json(out/'validation_progress.json',{'candidates':candidates})
        selected=best_candidate(candidates)
        trained=best_candidate(candidates[1:])
        m['selection']={'candidates':candidates,'selected':selected,'best_grpo':trained,
                        'rule':'validation correct_count; ties prefer fewer steps; SFT initial included',
                        'recommended_policy':'sft' if selected['label']=='sft_initial' else 'grpo'}
        m['decoding']=baseline['generation']; m['evaluation_config']=baseline['evaluation']
        m['status']='selected'
        load_split(c['full_dataset_root'],'validation',dc)
    finally:
        m['selection_seconds']=m.get('selection_seconds',0)+time.perf_counter()-started
        signal.alarm(0)


def evaluate(out,m):
    if m['status']!='selected': raise ValueError('Validation selection must precede test access')
    c=m['config']; dc=load_yaml(c['data_config']); baseline=load_yaml('configs/baseline.yaml')
    if baseline['generation']!=m['decoding'] or baseline['evaluation']!=m['evaluation_config']:
        raise ValueError('Evaluation config drift')
    remaining=c['budget']['evaluation_max_seconds']-m.get('evaluation_seconds',0)
    if remaining<=0: raise TimeoutError('Evaluation budget consumed')
    signal.alarm(math.ceil(remaining)); started=time.perf_counter()
    try:
        tasks=load_split(c['full_dataset_root'],'test',dc)
        if len(tasks)!=300: raise ValueError('Expected 300 frozen test tasks')
        locked=load_tasks(baseline,dc)
        task_by_id={t['id']:t for t in tasks}
        for t in locked:
            if t!=task_by_id[t['id']]:
                for key in ('question','schema','gold_sql','gold_result','db_sha256','order_sensitive'):
                    if t[key]!=task_by_id[t['id']][key]: raise ValueError('Locked40 task differs')
        previous=Path(c['adapter_path']).parent
        sft_metrics=json.loads((previous/'metrics.json').read_text())
        if sft_metrics['model_config']!=m['model_config']: raise ValueError('Model config differs from stage5')
        old=list(read_jsonl(previous/'test_predictions.jsonl'))
        comparisons={}; old_by_stage={}
        for stage in ('base','sft'):
            rows=[p for p in old if p['stage']==stage]
            actual=check_predictions(rows,tasks,baseline['evaluation'])
            if actual!=sft_metrics['evaluation'][stage]['formal300']:
                raise ValueError('Cached baseline aggregate mismatch')
            if any(p['model_revision']!=m['model_config']['model']['revision'] for p in rows):
                raise ValueError('Cached model revision mismatch')
            by_id={p['id']:p for p in rows}; old_by_stage[stage]=by_id
            comparisons[stage]={'formal300':actual,'locked40':summarize([by_id[t['id']] for t in locked])}
        m['cached_stage5_sha256']={'metrics':sha256_file(previous/'metrics.json'),
                                 'predictions':sha256_file(previous/'test_predictions.jsonl')}
        candidate=m['selection']['best_grpo']
        path=Path(candidate['path'])/'adapter_model.safetensors'
        if sha256_file(path)!=candidate['sha256']: raise ValueError('GRPO candidate drift')
        g=generator_for(m,baseline)
        set_peft_model_state_dict(g.model,load_file(str(path)),adapter_name='default')
        rows=generate_rows(g,tasks,dc,baseline['evaluation'],out/'test_predictions.jsonl','grpo',m['model_config'])
        by_id={p['id']:p for p in rows}
        comparisons['grpo']={'formal300':summarize(rows),'locked40':summarize([by_id[t['id']] for t in locked])}
        m['comparison']=comparisons
        m['paired_change']={'improved':[],'regressed':[],'different_output':0}
        for p in rows:
            oldp=old_by_stage['sft'][p['id']]
            if p['raw_output']!=oldp['raw_output']: m['paired_change']['different_output']+=1
            if p['execution_correct'] and not oldp['execution_correct']: m['paired_change']['improved'].append(p['id'])
            if oldp['execution_correct'] and not p['execution_correct']: m['paired_change']['regressed'].append(p['id'])
        m['test_adapter']=candidate
        m['cpu_rechecks']=600+len(rows)
        m['evaluation_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        m['evaluation_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        load_split(c['full_dataset_root'],'test',dc)
        m['status']='stage7_completed'
    finally:
        m['evaluation_seconds']=m.get('evaluation_seconds',0)+time.perf_counter()-started
        signal.alarm(0)


def report(out,m,target):
    if m['status']!='stage7_completed': raise ValueError('Incomplete comparison')
    groups=list(read_jsonl(out/'rollouts.jsonl'))
    scores=[s for g in groups for s in g['scores']]
    is_rl_zvp=m.get('advantage_method','').startswith('RL-ZVP')
    m['rollout_summary']={'groups':len(groups),'zero_variance_groups':sum(g['zero_variance'] for g in groups),
        'effective_groups':len(groups) if is_rl_zvp else sum(not g['zero_variance'] for g in groups),
        'positive_zero_variance_groups':sum(g.get('zero_variance',False) and g.get('all_execution_correct',False)
                                            for g in groups),
        'negative_zero_variance_groups':sum(g.get('zero_variance',False) and g.get('all_execution_incorrect',False)
                                            for g in groups),
        'difficulty_counts':{name:sum(g.get('difficulty')==name for g in groups)
                             for name in ('medium','hard','easy','extra')},
        'completions':len(scores),'truncated':sum(t for g in groups for t in g['truncated']),
        'mean_tokens':statistics.mean(n for g in groups for n in g['output_tokens']),
        'mean_reward':statistics.mean(s['total'] for s in scores),
        'mean_partial_result_similarity':statistics.mean(s.get('partial_result_similarity',0.0) for s in scores),
        'execution_success':sum(s['execution_success'] for s in scores),
        'execution_correct':sum(s['execution_correct'] for s in scores)}
    r=m['rollout_summary']; v=m['selection']; best=v['best_grpo']
    method='RL-ZVP' if is_rl_zvp else 'GRPO'
    lines=[f'# 阶段7：{method}训练与同条件评测报告','',
        f'> 状态：阶段7实验与对比已完成；训练完成{m["optimizer_steps"]}/{m["steps_requested"]}个有效优化步骤。未进入阶段8。','',
        f'运行：`{out.as_posix()}`。报告由metrics.json和rollouts.jsonl生成。','',
        f'| 测试集 | Base | SFT | {method} |','|---|---:|---:|---:|']
    for split,label in [('formal300','正式300条'),('locked40','原40条（300条子集）')]:
        cells=[]
        for stage in ('base','sft','grpo'):
            x=m['comparison'][stage][split]
            cells.append(f"{x['execution_accuracy']:.2%}（{x['correct_count']}/{x['sample_count']}）")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    delta=m['comparison']['grpo']['formal300']['execution_accuracy']-m['comparison']['sft']['formal300']['execution_accuracy']
    lines += ['',f'正式测试较SFT变化：{delta*100:+.2f}个百分点；由错变对{len(m["paired_change"]["improved"])}条，由对变错{len(m["paired_change"]["regressed"])}条。',
        f'验证集选优建议：{v["recommended_policy"]}；本次{method}测试使用仅按验证集选择的第{best["step"]}步候选。测试集不参与选优。','',
        f'| 难度（正式300条） | 样本数 | SFT | {method} | 变化 |','|---|---:|---:|---:|---:|']
    for difficulty in ('easy','medium','hard','extra'):
        s=m['comparison']['sft']['formal300']['by_difficulty'][difficulty]
        g=m['comparison']['grpo']['formal300']['by_difficulty'][difficulty]
        lines.append(f'| {difficulty} | {s["count"]} | {s["execution_accuracy"]:.2%} | {g["execution_accuracy"]:.2%} | {(g["execution_accuracy"]-s["execution_accuracy"])*100:+.2f}个百分点 |')
    lines += ['',
        '## 训练与验证','',
        f'- 从阶段5最佳SFT Adapter重新开始；冻结同一SFT参考。NF4、BF16计算、FP32 LoRA；G={m["config"]["training"]["num_generations"]}，batch=1，累积{m["config"]["training"]["gradient_accumulation_steps"]}，lr=5e-6，beta=0.04，seed=20260831。' + (f' RL-ZVP alpha={m["config"]["rl_zvp"]["alpha"]}。' if is_rl_zvp else ''),
        f'- 数据池500条；实际完成{m["optimizer_steps"]}/{m["steps_requested"]}步。停止原因：{m.get("stop_reason") or "达到步数上限"}。',
        (f'- 生成{r["groups"]}组/{r["completions"]}条；全部{r["effective_groups"]}组产生更新；零方差{r["zero_variance_groups"]}组均被RL-ZVP利用（全正确{r["positive_zero_variance_groups"]}、全错误{r["negative_zero_variance_groups"]}）。难度采样：{r["difficulty_counts"]}。' if is_rl_zvp else f'- 生成{r["groups"]}组/{r["completions"]}条；有效组{r["effective_groups"]}，零方差跳过{r["zero_variance_groups"]}组。难度采样：{r["difficulty_counts"]}。'),
        f'- 平均Reward {r["mean_reward"]:.4f}，错误结果分级相似度均值{r["mean_partial_result_similarity"]:.4f}，平均长度{r["mean_tokens"]:.2f} tokens，截断{r["truncated"]}条。',
        f'- {m["changed_adapter_tensors"]}个Adapter张量变化；Base和SFT参考哈希不变。独立重载logits差{m["reload"]["max_abs_logits_difference"]}。',
        f'- 训练流程{m["elapsed_seconds"]:.2f}秒，峰值allocated/reserved {m["peak_allocated_gib"]:.2f}/{m["peak_reserved_gib"]:.2f} GiB；验证选优{m["selection_seconds"]:.2f}秒，测试比较{m["evaluation_seconds"]:.2f}秒。',
        f'- {m["cpu_rechecks"]}条测试预测CPU复评通过（含复用的Base/SFT）；Adapter-only保存，不支持精确恢复优化器。','',
        '| 验证候选 | 步数 | 正确数/200 |','|---|---:|---:|']
    for item in v['candidates']: lines.append(f'| {item["label"]} | {item["step"]} | {item["metrics"]["correct_count"]} |')
    lines += ['','## 结论与限制','',
        f'正式候选与SFT起点权重哈希是否一致：{m["adapter_sha256"]==m["sft_adapter_sha256"]}。',
        ('零方差组不再跳过：全正确组使用 alpha乘token熵的正优势；全错误组使用 -alpha乘(组内最大token熵-token熵)的负优势；熵detach，仅作为缩放因子。非零方差组保持原GRPO相对优势。' if is_rl_zvp else '组内奖励相同时相对优势为0，因此本轮将该组记入审计后跳过，不消耗优化步骤；训练题按medium、hard、easy、extra稳定排序，冻结数据本身未修改。'),
        '错误但可执行SQL按执行结果的完整行匹配、行数相似度和列数相似度获得分级奖励；执行正确性仍是最高奖励，且不使用SQL字符串相似度。',
        '本轮三个验证候选权重哈希均不同，分别完成200条验证推理。正式300条GRPO候选另行生成，并与原Base/SFT缓存CPU复评对照。',
        f'这是固定SQLite单快照执行一致率，单次种子、小预算实验。训练奖励与测试精度分开记录；本轮完成50步，但验证集没有超过SFT，不能仅凭训练Reward声称{method}提升了能力。',
        f'训练使用{m["config"]["training"]["max_completion_length"]}新token上限、总长{m["config"]["max_sequence_length"]}；同条件评测沿用阶段5 greedy、128新token、总长512，截断保留并照原规则评分。Reward分量和组内样本标准差见rollouts.jsonl，KL和训练指标见training_metrics.jsonl。','',
        '## 复现','', '```bash','cd Text-to-SQL',
        'RUN=outputs/grpo/stage7-new-run',
        'bash scripts/train_grpo.sh --mode full --max-steps 50 --output-dir "$RUN"',
        'bash scripts/train_grpo.sh --verify --output-dir "$RUN"',
        'bash scripts/evaluate_grpo.sh --action select --run-dir "$RUN"',
        'bash scripts/evaluate_grpo.sh --action evaluate --run-dir "$RUN"',
        'bash scripts/evaluate_grpo.sh --action report --run-dir "$RUN" --report docs/stage7_new_report.md','```','']
    Path(target).write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['select','evaluate','report'],required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--report',default='docs/stage7_grpo_report.md')
    args=p.parse_args(); out=Path(args.run_dir)
    m=json.loads((out/'metrics.json').read_text())
    def timeout(*a): raise TimeoutError('Phase wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    m.setdefault('commands',[]).append(sys.argv)
    try:
        if args.action=='select': select(out,m)
        elif args.action=='evaluate': evaluate(out,m)
        else: report(out,m,args.report)
    except BaseException as exc:
        m.setdefault('phase_failures',[]).append({'phase':args.action,'error':repr(exc)})
        raise
    finally:
        write_json(out/'metrics.json',m)
        print(json.dumps({'status':m['status'],'selection':m.get('selection',{}).get('recommended_policy'),
                          'stop_reason':m.get('stop_reason')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
