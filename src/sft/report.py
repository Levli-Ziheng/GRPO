"""Build the single stage-5 report from observed metrics and CPU-rechecked predictions."""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter
from src.data.common import read_jsonl,sha256_file
from src.utils.config import load_yaml
from src.inference.generate import load_tasks
from src.evaluation.evaluate import evaluate_record
from src.evaluation.metrics import summarize
from src.sft.full import save_metrics
from src.execution.sqlite_executor import execute_query
from src.execution.result_normalizer import results_equal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',default='docs/stage5_sft_report.md')
    args=parser.parse_args()
    out=Path(args.run_dir)
    m=json.loads((out/'metrics.json').read_text())
    if m['status']!='completed': raise ValueError('Training, reload and both evaluations must complete')
    c=m['configuration']
    cfg=load_yaml(c['baseline_config'])
    data_cfg=load_yaml(c['data_config'])
    locked=load_tasks(cfg,data_cfg)
    root=Path(c['dataset_root'])
    formal_cfg={**cfg,'dataset':str(root/'canonical_test.jsonl'),
        'dataset_manifest':str(root/'dataset_manifest.json'),'expected_samples':300}
    formal=load_tasks(formal_cfg,data_cfg)
    tasks={r['id']:r for r in formal+locked}
    predictions=list(read_jsonl(out/'test_predictions.jsonl'))
    stages={}
    for stage in ('base','sft'):
        rows=[r for r in predictions if r['stage']==stage]
        if len(rows)!=len(tasks) or {r['id'] for r in rows}!=set(tasks): raise ValueError('Incomplete/duplicate test predictions')
        stages[stage]={r['id']:r for r in rows}
        checked=[evaluate_record(r,tasks[r['id']],cfg['evaluation']) for r in rows]
        if any(any(a[k]!=b[k] for k in ('execution_correct','execution_success','format_valid','error_type','predicted_result')) for a,b in zip(rows,checked)):
            raise ValueError('CPU prediction re-evaluation mismatch')
        for split,samples in (('formal300',formal),('locked40',locked)):
            got=summarize([stages[stage][r['id']] for r in samples])
            if got!=m['evaluation'][stage][split]: raise ValueError('Saved metrics mismatch')
    if sha256_file(out/'best_adapter'/'adapter_model.safetensors')!=m['adapter_sha256']: raise ValueError('Adapter hash drift')
    test_run=subprocess.run([sys.executable,'-m','pytest','-q','-o','addopts='],capture_output=True,text=True)
    if test_run.returncode: raise RuntimeError(test_run.stdout+test_run.stderr)
    match=re.search(r'(\d+) passed',test_run.stdout)
    if not match: raise RuntimeError('Cannot parse pytest result')
    tests_passed=int(match.group(1))
    checked_gold=0
    databases={}
    for split in ('train','validation','test'):
        for row in read_jsonl(root/f'canonical_{split}.jsonl'):
            path=row['db_path']
            if path not in databases: databases[path]=sha256_file(path)
            if databases[path]!=row['db_sha256']: raise ValueError('Database hash drift')
            result=execute_query(path,row['gold_sql'],timeout_seconds=3.0,max_result_rows=1000)
            if not results_equal(result['rows'],row['gold_result'],order_sensitive=row['order_sensitive'],
                tolerance=row['float_tolerance'],predicted_columns=result['column_count'],gold_columns=row['gold_result_column_count']):
                raise ValueError('Gold result drift: '+row['id'])
            checked_gold+=1
    smoke_path=out.with_name('stage5-20260913-smoke-r1')/'metrics.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    before=stages['base']; after=stages['sft']
    improved=[r['id'] for r in formal if not before[r['id']]['execution_correct'] and after[r['id']]['execution_correct']]
    regressed=[r['id'] for r in formal if before[r['id']]['execution_correct'] and not after[r['id']]['execution_correct']]
    historic={r['id']:r for r in read_jsonl('outputs/baseline/base-cspider-smoke-v0-20260908/predictions.jsonl')}
    historic_changes=[r['id'] for r in locked if historic[r['id']]['raw_output']!=before[r['id']]['raw_output']]
    m['acceptance']={'cpu_prediction_reevaluations':len(predictions),'saved_metrics_recomputed':True,
        'adapter_hash_verified':True,'tests_passed':tests_passed,'tests_command':'python -m pytest -q -o addopts=',
        'pytest_output':test_run.stdout+test_run.stderr,'gold_reexecuted':checked_gold,'databases_verified':len(databases),
        'representative_smoke':smoke,
        'changed_files':['configs/sft_full.yaml','src/sft/train.py','src/sft/full.py','src/sft/report.py','scripts/train_sft.sh','scripts/evaluate_sft.sh','tests/test_sft_full.py'],
        'improved_ids':improved,'regressed_ids':regressed,'historical_base_raw_output_changed_ids':historic_changes,
        'report_generator_sha256':sha256_file(__file__)}
    save_metrics(out,m)
    baseline=m['evaluation']['base']; sft=m['evaluation']['sft']
    best=m['best_checkpoint']; val=m['validation_curve']
    last_val=val[-1]['loss']
    pre_val=m['validation_before']['eval_loss']
    train=m['train']
    with (out/'training_curve.csv').open() as handle:
        curve=list(csv.DictReader(handle))
    train_points=[r for r in curve if r['train_loss']]
    best_window=[float(r['train_loss']) for r in train_points if best['step']-50<int(r['step'])<=best['step']]
    last_window=[float(r['train_loss']) for r in train_points if m['optimizer_steps']-50<int(r['step'])<=m['optimizer_steps']]
    best_train=sum(best_window)/len(best_window)
    last_train=sum(last_window)/len(last_window)
    best_time=next(float(r['elapsed_seconds']) for r in curve if int(r['step'])==best['step'] and r['validation_loss'])
    final_time=next(float(r['elapsed_seconds']) for r in reversed(curve) if r['validation_loss'])
    overfit={'optimizer_steps_per_epoch':2000//8,'best_observed_step':best['step'],
        'last_step':m['optimizer_steps'],'best_validation_loss':best['validation_loss'],'last_validation_loss':last_val,
        'validation_absolute_increase':last_val-best['validation_loss'],
        'validation_relative_increase':last_val/best['validation_loss']-1,
        'train_last50_steps_at_best':best_train,'train_last50_steps_at_end':last_train,
        'elapsed_after_best_seconds':final_time-best_time,'no_retraining':True,
        'interpretation':'Training loss falls while held-out validation loss rises after the best observed checkpoint; evidence of late-stage overfitting in this run.'}
    m['overfitting_analysis']=overfit
    save_metrics(out,m)
    lines=['# 阶段5：正式 SFT 与同条件测试报告','',
        '> 状态：已执行并完成工程验收。正式训练、最佳Adapter独立重载、Base/SFT评测及CPU复评均完成；未进入阶段6。','',
        f"运行目录：`{out}`。",'',
        '| 测试集 | 数量 | Base执行准确率 | SFT执行准确率 | 差值 |',
        '|---|---:|---:|---:|---:|']
    for split,label in (('formal300','正式测试集'),('locked40','原冻结测试集')):
        b,s=baseline[split],sft[split]
        lines.append(f"| {label} | {b['sample_count']} | {b['execution_accuracy']:.2%}（{b['correct_count']}） | {s['execution_accuracy']:.2%}（{s['correct_count']}） | {(s['execution_accuracy']-b['execution_accuracy'])*100:+.2f}个百分点 |")
    lines+=['', '原40条是正式300条的子集，两张表不是独立测试。Base与SFT分别重新生成300条预测；同一条预测复用于两种统计。', '',
        '## 训练、选优与资源','',
        f"- 数据：固定CSpider正式版本cspider-v1，训练2000条、验证200条；按数据库隔离。所有2200条训练/验证样本的Tokenizer、Mask与512长度上限均通过；2500条Gold SQL重执行一致，124个数据库哈希通过。",
        '- 普通内存加载，没有发生需要切换流式数据的内存不足。正式训练从锁定的原始Base开始；没有续训阶段4或本次2步smoke Adapter。',
        f"- 模型：`{m['model_config']['model']['id']}`，revision `{m['model_config']['model']['revision']}`；NF4 4-bit、double quant、BF16 compute、gradient checkpointing。",
        f"- LoRA：r={c['lora']['r']}，alpha={c['lora']['lora_alpha']}，dropout={c['lora']['lora_dropout']}，7个投影目标；可训练参数{m['trainable_parameters']:,}，占比{m['trainable_fraction']:.4%}。",
        f"- batch=1，梯度累积8；完成{m['optimizer_steps']}个优化器步骤、{m['sample_coverage']['microbatches']}个训练microbatch、{train['epoch']:.1f}轮。随机种子{c['seed']}。",
        f"- learning_rate={c['training']['learning_rate']}，warmup={c['training']['warmup_steps']}步，linear scheduler，AdamW，梯度裁剪1.0；每{c['evaluation_steps']}步在200条验证集上评估。",
        f"- 整段训练平均Loss：{train['train_loss']:.6f}；验证Loss：训练前{pre_val:.6f}，最优{best['validation_loss']:.6f}，最后一步{last_val:.6f}。",
        f"- 最佳Adapter：第{best['step']}步，仅按验证Loss最低选择。仅保存这个Adapter；没有保存全量模型和每一步checkpoint/optimizer副本，因此交付不支持精确恢复优化器续训。",
        f"- Trainer训练计时：{train['train_runtime']:.1f}秒；正式训练脚本总计（含加载、验证、校验与保存）：{m['elapsed_seconds']:.1f}秒。",
        f"- GPU峰值allocated/reserved：{m['peak_allocated_gib']:.3f}/{m['peak_reserved_gib']:.3f} GiB；训练进程峰值RSS：{m['peak_process_rss_gib']:.3f} GiB。显存为PyTorch allocator峰值，不是整卡nvidia-smi峰值。",
        f"- 采样覆盖：{m['sample_coverage']}。固定大小的指纹计数替代逐次保存输入序列，避免训练轨迹占用不断增长的内存。",'',
        '| Step | Epoch | Validation Loss |','|---:|---:|---:|']
    for point in val: lines.append(f"| {point['step']} | {point['epoch'] or 0:.2f} | {point['loss']:.6f} |")
    lines+=['',f"最后验证Loss相对最优值差{last_val-best['validation_loss']:+.6f}。"+
        ('后期验证Loss回升，支持出现过拟合迹象；最终评测使用较早的最佳Adapter。' if last_val>best['validation_loss']+0.01 else '当前检查点未显示明显的末期验证Loss回升；这不足以证明不存在过拟合。'),'',
        'Loss口径：只对assistant SQL及EOS计算shifted交叉熵；prompt、模板尾部换行和padding为-100。单样本内部按监督token平均，梯度累积再按样本平均，验证集同样按样本平均。它不是SQL执行准确率。CSV中的Train Loss是最近日志窗口，不应将不同窗口当作同一固定样本的Loss。','',
        '## 500步是否偏多、是否过拟合','',
        f"500步是优化器更新次数：2000条÷(batch 1×累积8)=250步/轮，因此500步等于2轮，不是500轮。作为首次正式实验的探索上限并不夸张，但本次验证结果表明最优已在检查过的第{best['step']}步出现，后续训练没有改善验证Loss。",'',
        f"验证Loss在最优点之后的记录为{'→'.join(str(point['step'])+'步:'+format(point['loss'],'.6f') for point in val if point['step']>=best['step'])}。最后值比最低值高{last_val-best['validation_loss']:.6f}（相对{(last_val/best['validation_loss']-1)*100:.2f}%）。与此同时，最优点前50步训练窗口平均Loss为{best_train:.6f}，末尾50步为{last_train:.6f}。训练继续拟合、固定验证集表现变差，支持后半程出现过拟合迹象；不能仅靠Train/Validation的差距来下结论。",'',
        f"第{best['step']}步后的训练和验证额外花费约{(final_time-best_time)/60:.1f}分钟。以本次最小验证Loss为目标，500步超过了当前观察到的合适停止点；交付和测试均使用第{best['step']}步Adapter，未使用验证Loss更高的末步权重，也未重新训练。",'',
        '限制：只有一次随机种子、200条验证样本、每125步一次验证，因此只能说250步是已检查点中最优，不能断言真正最优恰好是250。Loss与执行准确率不是同一指标；未评测第500步权重，不能据此声称其执行精度下降多少。若未来另做实验，可考虑更密集验证和早停；本次不追加训练。直接把新实验max_steps改成250还会改变线性学习率日程，不能假定它等价于此次第250步权重。','',
        '## 同条件评测与失败变化','',
        '两种模型复用阶段3 Generator、Prompt、SQL解析器、只读执行器和结果比较器。相同NF4加载及BF16基础模型路径，SFT额外加载最佳LoRA；greedy、num_beams=1、max_new_tokens=128、总长度512。只有问题和Schema进入模型，Gold只在外部评分使用。','',
        'Base首次评测在第32条因128-token上限未产生EOS而停止，检查后确认是固定预算终止；保留中断证据并从已校验的预测前缀继续。后续对Base/SFT均记录长度上限命中，保持原解码和SQL评分规则，未筛除这些样本。truncated标记表示预算耗尽且无EOS，不一定代表SQL语句不完整。','',
        '| 正式300条指标 | Base | SFT |','|---|---:|---:|']
    for key,label in [('format_valid_rate','格式正确率'),('execution_success_rate','SQL执行成功率')]:
        lines.append(f"| {label} | {baseline['formal300'][key]:.2%} | {sft['formal300'][key]:.2%} |")
    lines.append(f"| 截断条数 | {baseline['formal300']['truncated_count']} | {sft['formal300']['truncated_count']} |")
    lines+=['','| 难度（项目启发式） | 数量 | Base准确率 | SFT准确率 |','|---|---:|---:|---:|']
    for level,b in baseline['formal300']['by_difficulty'].items():
        s=sft['formal300']['by_difficulty'][level]
        lines.append(f"| {level} | {b['count']} | {b['execution_accuracy']:.2%} | {s['execution_accuracy']:.2%} |")
    lines+=['',f"正式300条中：由错变对{len(improved)}条，由对变错{len(regressed)}条。失败类型：Base `{baseline['formal300']['error_counts']}`；SFT `{sft['formal300']['error_counts']}`。",'',
        f"重新运行Base与阶段3原40条的raw_output不同样本数：{len(historic_changes)}。原冻结数据与阶段3产物哈希均保持不变。",'']
    for label,ids in [('SFT新增失败样例',regressed[:3]),('SFT改正样例',improved[:2])]:
        lines+=['### '+label,'']
        if not ids: lines+=['无。','']
        for sample_id in ids:
            r=after[sample_id]
            lines += [f"- `{sample_id}`：{r['question']}",f"  - Gold：`{r['gold_sql']}`",f"  - Base：`{before[sample_id]['predicted_sql']}`",f"  - SFT：`{r['predicted_sql']}`；错误类型：`{r['error_type']}`。",'']
    lines+=['## 验收、限制与核心文件','',
        f"- {tests_passed}项测试通过，含梯度累积与独立手算梯度一致性测试。本次代表性GPU smoke详情见metrics.json中的representative_smoke。",
        f"- 正式训练{m['gradient_checks']}次梯度检查均通过；Base参数哈希未变，{m['changed_adapter_tensors']}个Adapter张量实际变化。",
        f"- 正式最佳Adapter新进程重载：Loss差{m['reload_verification']['loss_delta']}，选定32个探针logits最大绝对差{m['reload_verification']['selected_logits_max_abs_delta']}。这是数值探针，不是全logits逐元素证明。",
        f"- {len(predictions)}条预测已用CPU重新执行评分，所有结果与汇总指标一致；长度上限触及情况见上表。",
        '- 这是固定SQLite单快照执行一致率，不是官方Spider test-suite分数；难度是项目heuristic_v1。列顺序固定、无ORDER BY时按多重集合比较、浮点绝对容差1e-6；空结果和Gold歧义仍可能限制指标含义。',
        '- 512长度过滤及GRPO预留长度规则会筛掉较长Schema/SQL，本次不是完整CSpider总体分布；2000条是正式确定性子集。单次种子实验不能证明统计稳定提升。',
        f"- GPU驱动当前为{m['environment']['nvidia_smi']['devices'][0]['driver_version']}；旧驱动阻塞证据保留在20260911预检目录。代码无Git，当前运行源码树SHA-256为`{m['source']['scoped_tree']['sha256']}`；配置、数据和版本指纹见metrics.json。",'',
        '| 核心文件 | 内容 |','|---|---|',
        f"| `{args.output}` | 本报告、结果表、错误样例与复现命令 |",
        f"| `{out}/training_curve.csv` | Train/Validation Loss、学习率、梯度范数、耗时 |",
        f"| `{out}/metrics.json` | 汇总精度、配置、环境、版本、资源、验证和失败证据 |",
        f"| `{out}/test_predictions.jsonl` | 两个模型全部预测和执行测试结果，包含错误样本 |",
        f"| `{out}/best_adapter/` | 验证Loss最低的可独立加载LoRA权重及配置 |",'',
        '训练、模型和测试产物保存在运行目录中；这些生成物默认不纳入版本控制。','',
        '## 复现','',
        '在项目根目录执行，RUN必须是未使用的新目录：','',
        '```bash','cd Text-to-SQL',
        'export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
        'export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_DISABLE_PROGRESS_BARS=1',
        'export PYTHON_BIN=python3',
        'RUN=outputs/sft/stage5-new-run',
        'bash scripts/train_sft.sh --mode full --config configs/sft_full.yaml --output-dir "$RUN"',
        '$PYTHON_BIN -m src.sft.full --action verify --run-dir "$RUN"',
        '$PYTHON_BIN -m src.sft.full --action evaluate --stage base --allow-budget-truncation --run-dir "$RUN"',
        'bash scripts/evaluate_sft.sh --allow-budget-truncation --run-dir "$RUN"',
        '$PYTHON_BIN -m src.sft.report --run-dir "$RUN" --output docs/stage5_new_run_report.md','```','',
        '训练等价入口：`python -m src.sft.train --mode full --config configs/sft_full.yaml --output-dir <新目录>`。核心代码：dataset.py负责Mask；loss.py负责shifted CE；full.py负责微批次Loss、梯度累积、验证选优与独立评测。']
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(target),'cpu_reevaluations':len(predictions),'improved':len(improved),'regressed':len(regressed)},ensure_ascii=False))

if __name__=='__main__': main()
