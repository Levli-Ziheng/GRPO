import copy
from types import SimpleNamespace
import pytest
import torch
from transformers import TrainingArguments
from src.sft.full import FullSQLTrainer
from src.sft.dataset import SQLCollator
from src.sft.loss import assistant_loss

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(5,7))
    def forward(self,input_ids,attention_mask=None):
        return SimpleNamespace(logits=self.weight[input_ids])

def test_accumulation_equals_mean_per_example_gradient(tmp_path):
    torch.manual_seed(7)
    model=TinyModel()
    reference=copy.deepcopy(model)
    rows=[{'input_ids':[0,1,2], 'attention_mask':[1,1,1], 'labels':[-100,3,4]},
          {'input_ids':[0,1,2,3,4], 'attention_mask':[1]*5, 'labels':[-100,-100,2,1,3]}]
    collator=SQLCollator(0)
    loss=sum(assistant_loss(reference(**{k:v for k,v in collator([r]).items() if k!='labels'}).logits,
                            collator([r])['labels']) for r in rows)/2
    loss.backward()
    expected=reference.weight.detach()-0.1*reference.weight.grad
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,max_steps=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=2,learning_rate=0.1,optim='sgd',weight_decay=0,max_grad_norm=0,
        warmup_steps=0,lr_scheduler_type='constant',save_strategy='no',report_to='none',disable_tqdm=True,
        remove_unused_columns=False,dataloader_pin_memory=False)
    trainer=FullSQLTrainer(model=model,args=args,train_dataset=rows,data_collator=collator)
    trainer.train()
    assert torch.allclose(model.weight,expected,atol=1e-6)
    assert sum(trainer.sample_counts.values())==2

def test_rejects_multi_sample_microbatch(tmp_path):
    model=TinyModel()
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,report_to='none')
    trainer=FullSQLTrainer(model=model,args=args)
    with pytest.raises(ValueError,match='microbatch'):
        trainer.compute_loss(model,{'input_ids':torch.zeros((2,3),dtype=torch.long),
            'labels':torch.zeros((2,3),dtype=torch.long)})
