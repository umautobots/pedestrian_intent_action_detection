import torch
import torch.nn.functional as F
import pdb

def cross_entropy_loss(pred, target, reduction='mean'):
    '''
    pred: (batch, seg_len, num_class)
    target: (batch, seg_len)
    '''
    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1)
    return F.cross_entropy(pred, target, reduction=reduction)

def binary_cross_entropy_loss(pred, target, reduction='mean'):
    '''
    pred: logits, (batch, seg_len, 1)
    target: (batch, seg_len) or (batch, seg_len, 1)
    '''
    if pred.shape != target.shape:
        num_class =  pred.shape[-1]
        pred = pred.view(-1, num_class)
        target = target.view(-1, num_class).type(torch.float)
    return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)

def trn_loss(pred, target, reduction='mean'):
    '''
    pred: (batch, seg_len, pred_len, num_class)
    target: (batch, seg_len + pred_len, num_class)
    '''
    batch, seg_len, pred_len, num_class = pred.shape
    assert seg_len + pred_len == target.shape[1]

    # collect all targets
    flattened_targets = []
    for i in range(1, seg_len+1):
        flattened_targets.append(target[:, i:i+pred_len])
    
    flattened_targets = torch.cat(flattened_targets, dim=1)
    # compute loss
    return cross_entropy_loss(pred.view(batch, -1, num_class), flattened_targets, reduction=reduction)