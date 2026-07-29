import torch
import torch.nn.functional as F
from src.model import FocalLoss

def test_focal_loss_pt_computation():
    inputs = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]]) # Shape: (2, 3)
    targets = torch.tensor([0, 1])
    
    # Unweighted
    loss_fn = FocalLoss(gamma=2.0, reduction='none')
    loss = loss_fn(inputs, targets)
    
    # Manually compute expected
    probs = F.softmax(inputs, dim=-1)
    pt = probs[torch.arange(2), targets]
    
    ce_loss = -torch.log(pt)
    expected_loss = ((1 - pt) ** 2.0) * ce_loss
    
    assert torch.allclose(loss, expected_loss), "Unweighted focal loss does not match expected."
    
def test_focal_loss_weighted():
    inputs = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]]) # Shape: (2, 3)
    targets = torch.tensor([0, 1])
    weights = torch.tensor([0.5, 2.0, 1.0])
    
    loss_fn = FocalLoss(weight=weights, gamma=2.0, reduction='none')
    loss = loss_fn(inputs, targets)
    
    # Manually compute expected
    probs = F.softmax(inputs, dim=-1)
    pt = probs[torch.arange(2), targets]
    
    ce_loss_unweighted = -torch.log(pt)
    ce_loss_weighted = ce_loss_unweighted * weights[targets]
    
    expected_loss = ((1 - pt) ** 2.0) * ce_loss_weighted
    
    assert torch.allclose(loss, expected_loss), "Weighted focal loss does not match mathematically corrected implementation."
