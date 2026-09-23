import torch.nn.functional as F

def assistant_loss(logits, labels):
    """Position t predicts label t+1. Mean CE over SQL + EOS only."""
    shifted_labels = labels[:, 1:]
    active = shifted_labels != -100
    if not active.any():
        raise ValueError("No supervised assistant tokens")
    return F.cross_entropy(logits[:, :-1, :][active].float(), shifted_labels[active])
