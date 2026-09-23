import pytest
import torch
from src.sft.dataset import encode_example, SQLCollator
from src.sft.loss import assistant_loss

class Tokenizer:
    eos_token_id=99
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict):
        assert return_dict is False
        return [1,2,3] if add_generation_prompt else [1,2,3,10,11,99,4]
    def decode(self,tokens,skip_special_tokens=False):
        return "SELECT 1" if tokens==[10,11] else "wrong"

def example():
    return {"id":"test","messages":[{"role":"system","content":"s"},
            {"role":"user","content":"u"},{"role":"assistant","content":"SELECT 1"}]}

def test_sql_and_eos_only():
    feature=encode_example(example(),Tokenizer(),8)
    assert feature["labels"]==[-100,-100,-100,10,11,99,-100]
    with pytest.raises(ValueError,match="truncate"):
        encode_example(example(),Tokenizer(),6)

def test_padding_mask():
    batch=SQLCollator(0)([{"input_ids":[1,2,3],"attention_mask":[1,1,1],"labels":[-100,2,3]},
                          {"input_ids":[1,2],"attention_mask":[1,1],"labels":[-100,2]}])
    assert batch["input_ids"].tolist()==[[1,2,3],[1,2,0]]
    assert batch["attention_mask"].tolist()==[[1,1,1],[1,1,0]]
    assert batch["labels"].tolist()==[[-100,2,3],[-100,2,-100]]

def test_shift_mask_and_gradients():
    logits=torch.randn(1,5,7,requires_grad=True)
    labels=torch.tensor([[-100,-100,3,4,-100]])
    loss=assistant_loss(logits,labels)
    expected=torch.nn.functional.cross_entropy(logits[:,1:3,:].reshape(-1,7),torch.tensor([3,4]))
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:,0,:])==0
    assert torch.count_nonzero(logits.grad[:,3:,:])==0
    assert torch.count_nonzero(logits.grad[:,1:3,:])>0
    changed=logits.detach().clone(); changed[:,0,:]=1000
    assert torch.allclose(assistant_loss(changed,labels),loss)
    with pytest.raises(ValueError,match="supervised"):
        assistant_loss(logits,torch.full_like(labels,-100))
