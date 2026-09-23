"""TRL reward callback, frozen-data view, and group-level audit."""
import json
import statistics
from pathlib import Path
import torch
from src.data.common import read_jsonl, sha256_file, build_messages
from src.grpo.rewards import score_candidate


DIFFICULTY_PRIORITY = ('medium', 'hard', 'easy', 'extra')


def entropy_guided_advantages(entropies, completion_mask, kinds, alpha):
    """Equation (5) from RL-ZVP, with entropy detached from the graph.

    ``kinds`` is +1 for an all-correct group, -1 for an all-incorrect group,
    and 0 for an ordinary GRPO group. The caller retains ordinary GRPO
    advantages for kind 0; this helper only constructs the ZVP branch.
    """
    if entropies.ndim != 2 or completion_mask.shape != entropies.shape:
        raise ValueError('Entropy/mask shape mismatch')
    if kinds.ndim != 1 or kinds.shape[0] != entropies.shape[0]:
        raise ValueError('ZVP kind shape mismatch')
    if alpha <= 0:
        raise ValueError('RL-ZVP alpha must be positive')
    if not torch.all((kinds == -1) | (kinds == 0) | (kinds == 1)):
        raise ValueError('ZVP kinds must be -1, 0, or 1')
    mask = completion_mask.bool()
    if not torch.all(mask.any(dim=1)):
        raise ValueError('Every completion must contain at least one active token')
    detached = entropies.detach()
    max_entropy = detached.masked_fill(~mask, float('-inf')).max(dim=1, keepdim=True).values
    positive = alpha * detached
    negative = -alpha * (max_entropy - detached)
    shaped = torch.where(kinds.unsqueeze(1) > 0, positive, negative)
    return shaped.masked_fill(~mask, 0.0)


def prioritize_rows(rows, priority=DIFFICULTY_PRIORITY):
    """Stable runtime view that favors medium tasks without changing frozen data."""
    rank = {name: index for index, name in enumerate(priority)}
    return sorted(rows, key=lambda row: rank.get(row['difficulty'], len(rank)))


def proportional_interleave_rows(rows, priority=DIFFICULTY_PRIORITY):
    """Stable interleave whose every prefix tracks the pool distribution.

    This prevents a short bounded run from seeing only the first difficulty
    bucket while preserving the original order within each difficulty.
    """
    buckets = {name: [] for name in priority}
    extras = []
    for row in rows:
        name = row.get('difficulty')
        if name in buckets:
            buckets[name].append(row)
        else:
            extras.append(row)
    if extras:
        buckets['__other__'] = extras
    names = [name for name in (*priority, '__other__') if name in buckets and buckets[name]]
    totals = {name: len(buckets[name]) for name in names}
    emitted = {name: 0 for name in names}
    result = []
    while len(result) < len(rows):
        position = len(result) + 1
        available = [name for name in names if emitted[name] < totals[name]]
        name = max(available, key=lambda item: (totals[item] / len(rows) * position - emitted[item],
                                                -names.index(item)))
        result.append(buckets[name][emitted[name]])
        emitted[name] += 1
    return result


def load_rows(root, data_config, tokenizer, max_length=512, completion_length=64):
    root = Path(root)
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    for name in ('grpo_train.jsonl', 'canonical_train.jsonl'):
        if sha256_file(root/name) != manifest['file_sha256'][name]:
            raise ValueError('Frozen data hash mismatch: '+name)
    canonical = {r['id']: r for r in read_jsonl(root/'canonical_train.jsonl')}
    result, checked = [], set()
    for row in read_jsonl(root/'grpo_train.jsonl'):
        task = canonical[row['id']]
        context = dict(row['reward_context'], gold_result_column_count=task['gold_result_column_count'])
        if row['prompt'] != build_messages(data_config, task['schema'], task['question']):
            raise ValueError('Prompt drift')
        if context['db_path'] not in checked:
            if sha256_file(context['db_path']) != context['db_sha256']:
                raise ValueError('Database hash drift')
            checked.add(context['db_path'])
        encoded = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True,
                                                return_dict=True)
        if len(encoded['input_ids']) + completion_length > max_length:
            raise ValueError('Prompt exceeds total token budget: '+row['id'])
        if row['difficulty'] != task['difficulty']:
            raise ValueError('Difficulty drift: '+row['id'])
        result.append({'id':row['id'], 'difficulty':row['difficulty'],
                       'prompt':row['prompt'], 'reward_context':context})
    if [r['id'] for r in result] != manifest['sample_ids']['grpo']:
        raise ValueError('GRPO IDs/order drift')
    return result


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and len(completion) == 1
            and completion[0].get('role') == 'assistant'
            and isinstance(completion[0].get('content'), str)):
        return completion[0]['content']
    raise ValueError('Unexpected completion structure')


def arrow_rows(rows):
    # SQLite cells can mix strings/numbers/nulls; Arrow nested lists cannot.
    # Encode external scoring metadata only, preserving prompts unchanged.
    return [dict(r,reward_context=json.dumps(r['reward_context'],ensure_ascii=False,allow_nan=False)) for r in rows]


class RewardAudit:
    __name__ = 'execution_reward'

    def __init__(self, config, path, group_size, max_tokens, eos_id):
        self.config, self.path = config, Path(path)
        self.group_size, self.max_tokens, self.eos_id = group_size, max_tokens, eos_id
        self.groups, self.zero_groups = 0, 0
        self.skipped_zero_groups = 0
        self.positive_zero_groups = 0
        self.negative_zero_groups = 0
        self.last_group_record = None
        self.sampled_task_ids = set()
        self.sampled_difficulties = {}

    def __call__(self, prompts, completions, completion_ids, reward_context, id, difficulty, **kwargs):
        reward_context = [json.loads(c) if isinstance(c,str) else c for c in reward_context]
        n = len(completions)
        if not n or any(len(v) != n for v in (prompts, completion_ids, reward_context, id, difficulty)) or n % self.group_size:
            raise ValueError('Reward batch/group mismatch')
        rewards, stop = [], None
        for start in range(0,n,self.group_size):
            end = start+self.group_size
            if any(id[i] != id[start] or difficulty[i] != difficulty[start] or prompts[i] != prompts[start]
                   or reward_context[i] != reward_context[start]
                   for i in range(start,end)):
                raise ValueError('A GRPO group mixed different tasks')
            scores = [score_candidate(completion_text(completions[i]),reward_context[i],self.config)
                      for i in range(start,end)]
            totals = [s['total'] for s in scores]
            std = statistics.stdev(totals)  # TRL nanstd uses sample std, not population std.
            zero = std < 1e-8
            all_correct = all(s['execution_correct'] for s in scores)
            all_incorrect = not any(s['execution_correct'] for s in scores)
            if zero and not (all_correct or all_incorrect):
                raise RuntimeError('Zero-reward-variance group has mixed correctness')
            self.groups += 1
            self.zero_groups += int(zero)
            self.positive_zero_groups += int(zero and all_correct)
            self.negative_zero_groups += int(zero and all_incorrect)
            self.sampled_task_ids.add(id[start])
            self.sampled_difficulties[difficulty[start]] = self.sampled_difficulties.get(difficulty[start], 0)+1
            lengths = [len(completion_ids[i]) for i in range(start,end)]
            truncated = [len(completion_ids[i]) >= self.max_tokens and self.eos_id not in completion_ids[i]
                         for i in range(start,end)]
            record = {'group':self.groups, 'id':id[start], 'difficulty':difficulty[start],
                'scores':scores, 'reward_std_sample':std,
                'advantages':[(r-statistics.mean(totals))/(std+1e-4) for r in totals],
                'zero_variance':zero, 'all_execution_correct':all_correct,
                'all_execution_incorrect':all_incorrect,
                'output_tokens':lengths, 'truncated':truncated}
            self.last_group_record = record
            with self.path.open('a',encoding='utf-8') as h:
                h.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            rewards.extend(totals)
            if any(truncated): stop = 'Truncated training completion; diagnose before expansion'
            if any(s['class']=='unsafe_sql' for s in scores): stop = 'Unsafe SQL generated'
        if stop:
            raise RuntimeError(stop)
        return rewards
