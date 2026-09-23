"""CPU-only stage-7 readiness checks; does not instantiate a policy or trainer."""
import argparse
import importlib.metadata
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.common import build_messages, read_jsonl, sha256_file
from src.data.validate import validate_dataset
from src.grpo.rewards import score_candidate, run_self_test
from src.utils.config import load_yaml, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/grpo_stage7.yaml')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'preflight.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    c = load_yaml(args.config)
    dc = load_yaml(c['data_config'])
    rc = load_yaml(c['reward_config'])
    report = {'status': 'running', 'gpu_training_executed': False, 'command': sys.argv,
              'configuration': c, 'datasets': {}}
    try:
        from trl import GRPOConfig, GRPOTrainer
        report['trl_version'] = importlib.metadata.version('trl')
        report['trainer_source'] = inspect.getfile(GRPOTrainer)
        # CPU config validation only: override mixed precision for this constructor.
        values = dict(c['training'], bf16=False, use_cpu=True)
        gc = GRPOConfig(output_dir=str(out / 'unused-trainer-output'), **values)
        report['config_check'] = {'passed': True, 'generation_batch_size': gc.generation_batch_size,
                                  'num_generations': gc.num_generations, 'loss_type': gc.loss_type}
        for mode, key in [('smoke', 'smoke_dataset_root'), ('full', 'full_dataset_root')]:
            validation, errors = validate_dataset(dc, mode)
            if errors:
                raise ValueError(errors)
            root = Path(c[key])
            canonical = {r['id']: r for r in read_jsonl(root / 'canonical_train.jsonl')}
            rows = list(read_jsonl(root / 'grpo_train.jsonl'))
            checked = 0
            for row in rows:
                task = canonical[row['id']]
                assert row['prompt'] == build_messages(dc, task['schema'], task['question'])
                assert row['prompt_tokens'] + c['training']['max_completion_length'] <= c['max_sequence_length']
                context = dict(row['reward_context'], gold_result_column_count=task['gold_result_column_count'])
                assert context['gold_result'], 'GRPO pool must exclude empty gold results'
                scored = score_candidate(task['gold_sql'], context, rc)
                assert scored['execution_correct'], row['id']
                assert scored['total'] == 1.2, (row['id'], scored)
                checked += 1
            report['datasets'][mode] = {'validation_passed': True, 'gold_reward_checked': checked,
                'prompt_checked': checked, 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows),
                'database_overlap': validation['database_overlap'],
                'manifest_sha256': sha256_file(root / 'dataset_manifest.json')}
        sft = json.loads((Path(c['adapter_path']).parent / 'metrics.json').read_text())
        adapter_hash = sha256_file(Path(c['adapter_path']) / 'adapter_model.safetensors')
        assert sft['status'] == 'completed' and adapter_hash == sft['adapter_sha256']
        report['adapter_sha256'] = adapter_hash
        report['reward_self_test_cases'] = run_self_test(rc)['case_count']
        tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_rewards.py',
                                'tests/test_baseline.py'], capture_output=True, text=True)
        report['targeted_tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        tests.check_returncode()
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
