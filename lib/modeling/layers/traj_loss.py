import torch
import torch.nn.functional as F
import pdb

def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def bom_traj_loss(pred, target):
    '''
    pred: (B, T, K, dim)
    target: (B, T, dim)
    '''
    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    return loss_traj
    
def fol_rmse(x_true, x_pred):
    '''
    Params:
        x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
        x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=-1))#
    L2_diff = torch.sum(L2_diff, dim=-1).mean()
    # sum of all batches
    # L2_mean_pred = torch.mean(L2_all_pred)

    return L2_diff

def masked_mse(y_true, y_pred):
    '''
    some keypoints invisible, thus only compute mse on visible keypoints
    y_true: (B, T, 50)
    y_pred: (B, T, 50)

    NOTE: March 21, new loss is the sum over prediction horizon instead of mean
    '''
    # pdb.set_trace()
    mask = y_true != 0.0
    diff = (y_pred - y_true) ** 2
    num_good_kpts = mask.sum(dim=-1, keepdims=True)
    a = torch.ones_like(num_good_kpts)
    num_good_kpts = torch.where(num_good_kpts > 0.0, num_good_kpts, a)
    mse_per_traj_per_frame = torch.sum((diff * mask) / num_good_kpts, dim=-1)
    
    return mse_per_traj_per_frame.sum(dim=-1).mean()#

def mse_loss(gt_frames, gen_frames):
    return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))

def bce_heatmap_loss(pred, target):
    '''
    sum over each image, then mean over batch
    '''
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    bce_loss = bce_loss.sum((1,2)).mean()
    return bce_loss

def l2_heatmap_loss(pred, target):
    '''
    sum over each image, then mean over batch
    '''
    bce_loss = ((pred - target)**2).sum((1,2)).mean()
    return bce_loss