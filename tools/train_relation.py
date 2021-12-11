
import os
import sys
sys.path.append('../intention2021ijcai')

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse
from configs import cfg

from datasets import make_dataloader
from lib.modeling import make_model
from lib.engine.trainer_relation import  do_train_iteration

import logging
from termcolor import colored 
from lib.utils.logger import Logger
import logging


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument(
    "--config_file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cfg.freeze()


if cfg.USE_WANDB:
    logger = Logger("relation_embedding",
                    cfg,
                    project = cfg.PROJECT,
                    viz_backend="wandb"
                    )
    run_id = logger.run_id
else:
    logger = logging.Logger("relation_embedding")
    run_id = 'no_wandb'

# make model
model = make_model(cfg).to(cfg.DEVICE)

num_params = 0
for name, param in model.named_parameters():
    _num = 1
    for a in param.shape:
        _num *= a
    num_params += _num
    print("{}:{}".format(name, param.shape))
print(colored("total number of parameters: {}".format(num_params), 'white', 'on_green'))

# make dataloader
train_dataloader = make_dataloader(cfg, split='train')
val_dataloader = make_dataloader(cfg, split='val')
test_dataloader = make_dataloader(cfg, split='test')

# optimizer
optimizer = optim.RMSprop(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.L2_WEIGHT, alpha=0.9, eps=1e-7)# the weight of L2 regularizer is 0.001
if cfg.SOLVER.SCHEDULER == 'exp':
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
elif cfg.SOLVER.SCHEDULER == 'plateau':
    # Same to original PIE implementation
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10,#0.2
                                                        min_lr=1e-07, verbose=1)
else:
    lr_scheduler = None
    
# checkpoints
save_checkpoint_dir = os.path.join(cfg.CKPT_DIR, run_id)
if not os.path.exists(save_checkpoint_dir):
    os.makedirs(save_checkpoint_dir)

do_train_iteration(cfg, model, optimizer, 
                    train_dataloader, val_dataloader, test_dataloader, 
                    cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler, save_checkpoint_dir=save_checkpoint_dir)
