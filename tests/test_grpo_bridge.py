import json
import sqlite3
import statistics
import pytest
import torch
from src.grpo.bridge import (RewardAudit, completion_text, arrow_rows, prioritize_rows,
                             entropy_guided_advantages, proportional_interleave_rows)
from src.grpo.rewards import score_candidate, partial_result_similarity
from src.evaluation.evaluate import evaluate_record
from src.utils.config import load_yaml


@pytest.fixture
def context(tmp_path):
    path=tmp_path/'db.sqlite'
    with sqlite3.connect(path) as con:
        con.executescript('CREATE TABLE t(x); INSERT INTO t VALUES(1),(2);')
    return dict(db_path=str(path),gold_result=[[2]],gold_result_column_count=1,order_sensitive=False,
                float_tolerance=1e-6,timeout_seconds=0.1,max_result_rows=100)


def test_callback_and_group_math(tmp_path,context):
    audit=RewardAudit(load_yaml('configs/grpo.yaml'),tmp_path/'log.jsonl',2,64,99)
    result=audit(prompts=['q']*2,completions=[[{'role':'assistant','content':'SELECT count(*) FROM t'}],
        [{'role':'assistant','content':'SELECT 100'}]],completion_ids=[[3,99],[4,99]],reward_context=[context]*2,
        id=['x']*2,difficulty=['medium']*2)
    assert result==pytest.approx([1.2,0.44])
    log=json.loads((tmp_path/'log.jsonl').read_text())
    assert log['reward_std_sample']==pytest.approx(statistics.stdev(result))
    assert sum(log['advantages'])==pytest.approx(0)
    assert log['scores'][0]['components']['execution_correctness']==1


@pytest.mark.parametrize('sql',['SELECT count(*) FROM t','SELECT 100','SELECT missing FROM t',
    '```sql\nSELECT count(*) FROM t\n```','DELETE FROM t','SELECT x,x FROM t WHERE 0'])
def test_reward_matches_evaluator(context,sql):
    # Include the empty-results/column-count edge case.
    if 'WHERE 0' in sql: context=dict(context,gold_result=[])
    reward=score_candidate(sql,context,load_yaml('configs/grpo.yaml'))
    task=dict(context,db_id='test',difficulty='easy',question='q',gold_sql='SELECT count(*) FROM t')
    evaluation=evaluate_record({'raw_output':sql},task,context)
    assert reward['execution_correct']==evaluation['execution_correct']
    assert reward['execution_success']==evaluation['execution_success']


def test_stop_preserves_truncated_evidence(tmp_path,context):
    audit=RewardAudit(load_yaml('configs/grpo.yaml'),tmp_path/'log.jsonl',2,2,99)
    with pytest.raises(RuntimeError,match='Truncated'):
        audit(['q']*2,['SELECT 1','SELECT 2'],[[1,2],[1,99]],[context]*2,['x']*2,['medium']*2)
    assert json.loads((tmp_path/'log.jsonl').read_text())['truncated']==[True,False]


def test_zero_variance_is_recorded_without_early_stop(tmp_path,context):
    audit=RewardAudit(load_yaml('configs/grpo.yaml'),tmp_path/'log.jsonl',2,64,99)
    kwargs=dict(prompts=['q']*2,completions=['SELECT 1']*2,completion_ids=[[1,99]]*2,
                reward_context=[context]*2,id=['x']*2,difficulty=['medium']*2)
    audit(**kwargs)
    audit(**kwargs)
    assert audit.groups==2
    assert audit.negative_zero_groups==2
    assert audit.positive_zero_groups==0
    assert audit.last_group_record['all_execution_incorrect']
    with pytest.raises(ValueError,match='mixed'): audit(**dict(kwargs,id=['x','y']))


def test_entropy_guided_advantages_match_rl_zvp_equation():
    entropy=torch.tensor([[1.0,3.0,99.0],[2.0,5.0,0.0]])
    mask=torch.tensor([[1,1,0],[1,1,0]])
    kinds=torch.tensor([1,-1])
    actual=entropy_guided_advantages(entropy,mask,kinds,0.1)
    assert actual==pytest.approx(torch.tensor([[0.1,0.3,0.0],[-0.3,0.0,0.0]]))
    assert not actual.requires_grad


def test_entropy_guided_advantages_validate_inputs():
    with pytest.raises(ValueError,match='alpha'):
        entropy_guided_advantages(torch.ones(1,2),torch.ones(1,2),torch.ones(1),0)


def test_rejects_malformed_completion():
    with pytest.raises(ValueError): completion_text([{'role':'user','content':'SELECT 1'}])


def test_arrow_mixed_sql_cells_roundtrip(tmp_path,context):
    from datasets import Dataset
    mixed=dict(context,gold_result=[['Town',5706.0,None],['Village',1009.75,None]])
    rows=[dict(id='x',prompt=[{'role':'user','content':'q'}],reward_context=mixed)]
    encoded=Dataset.from_list(arrow_rows(rows))[0]
    assert encoded['prompt']==rows[0]['prompt']
    assert json.loads(encoded['reward_context'])==mixed
    audit=RewardAudit(load_yaml('configs/grpo.yaml'),tmp_path/'log.jsonl',2,64,99)
    encoded_context=json.dumps(context)
    assert audit(['q']*2,['SELECT count(*) FROM t']*2,[[99]]*2,[encoded_context]*2,
                 ['x']*2,['medium']*2)==[1.2,1.2]


def test_medium_rows_are_prioritized_stably():
    rows=[{'id':'e','difficulty':'easy'},{'id':'m1','difficulty':'medium'},
          {'id':'h','difficulty':'hard'},{'id':'m2','difficulty':'medium'}]
    assert [r['id'] for r in prioritize_rows(rows)]==['m1','m2','h','e']


def test_proportional_interleave_preserves_distribution_and_bucket_order():
    rows=([{'id':f'e{i}','difficulty':'easy'} for i in range(5)]
          +[{'id':f'm{i}','difficulty':'medium'} for i in range(3)]
          +[{'id':f'h{i}','difficulty':'hard'} for i in range(2)])
    actual=proportional_interleave_rows(rows)
    assert sorted(r['id'] for r in actual)==sorted(r['id'] for r in rows)
    for difficulty in ('easy','medium','hard'):
        assert [r['id'] for r in actual if r['difficulty']==difficulty] == [
            r['id'] for r in rows if r['difficulty']==difficulty]
    assert len({r['difficulty'] for r in actual[:5]})==3


def test_partial_result_similarity_is_graded_and_bounded():
    partial=partial_result_similarity([[1],[3]],[[1],[2]],order_sensitive=False,tolerance=1e-6,
                                      predicted_columns=1,gold_columns=1)
    unrelated=partial_result_similarity([[9]],[[1],[2]],order_sensitive=False,tolerance=1e-6,
                                        predicted_columns=1,gold_columns=1)
    assert 0 < unrelated < partial < 1
