
import os
import sys
sys.path.append('../intention2021icra')

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse
from configs import cfg

from datasets import make_dataloader
from lib.modeling import make_model
from lib.engine.trainer import do_train, do_val, do_train_iteration
from lib.engine.inference import inference
from lib.utils.meter import AverageValueMeter
from lib.utils.scheduler import ParamScheduler, sigmoid_anneal


import logging
from termcolor import colored 
from lib.utils.logger import Logger
import logging
from tqdm import tqdm

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

# num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
# args.distributed = num_gpus > 1

# if args.distributed:
#     torch.cuda.set_device(args.local_rank)
#     torch.distributed.init_process_group(
#         backend="nccl", init_method="env://"
#     )
#     synchronize()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cfg.freeze()


if cfg.USE_WANDB:
    logger = Logger("action_intent",
                    cfg,
                    project = cfg.PROJECT,
                    viz_backend="wandb"
                    )
    run_id = logger.run_id
else:
    logger = logging.Logger("action_intent")
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
    # NOTE: June 10, think about using Trajectron++ shceduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
elif cfg.SOLVER.SCHEDULER == 'plateau':
    # Same to original PIE implementation
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10,#0.2
                                                        min_lr=1e-07, verbose=1)
else:
    lr_scheduler = None #optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)
    
# checkpoints
if os.path.isfile(cfg.CKPT_DIR):
    model.load_state_dict(torch.load(cfg.CKPT_DIR))
    save_checkpoint_dir = os.path.join('/'.join(cfg.CKPT_DIR.split('/')[:-2]), run_id)
    print(colored("Train from checkpoint: {}".format(cfg.CKPT_DIR), 'white', 'on_green'))
else:
    save_checkpoint_dir = os.path.join(cfg.CKPT_DIR, run_id)
if not os.path.exists(save_checkpoint_dir):
    os.makedirs(save_checkpoint_dir)

# NOTE: Setup parameter scheduler
if cfg.SOLVER.INTENT_WEIGHT_MAX != -1:
    model.param_scheduler = ParamScheduler()
    model.param_scheduler.create_new_scheduler(
                                        name='intent_weight',
                                        annealer=sigmoid_anneal,
                                        annealer_kws={
                                            'device': cfg.DEVICE,
                                            'start': 0,
                                            'finish': cfg.SOLVER.INTENT_WEIGHT_MAX,# 20.0
                                            'center_step': cfg.SOLVER.CENTER_STEP,#800.0,
                                            'steps_lo_to_hi': cfg.SOLVER.STEPS_LO_TO_HI, #800.0 / 4.
                                        })
torch.autograd.set_detect_anomaly(True)
# NOTE: try different way to sample data for training.
if cfg.DATALOADER.ITERATION_BASED:
    do_train_iteration(cfg, model, optimizer, 
                       train_dataloader, val_dataloader, test_dataloader, 
                       cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler, save_checkpoint_dir=save_checkpoint_dir)
else:
    # trainning loss meters
    loss_act_det_meter = AverageValueMeter()
    loss_act_pred_meter = AverageValueMeter()
    loss_intent_meter = AverageValueMeter()

    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.DEVICE, loss_act_det_meter, loss_act_pred_meter, loss_intent_meter, logger=logger, lr_scheduler=lr_scheduler)
        loss_val = do_val(cfg, epoch, model, val_dataloader, cfg.DEVICE, logger=logger)

        if epoch % cfg.TEST.INTERVAL == 0:
            result_dict = inference(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger)
            torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))
        if cfg.SOLVER.SCHEDULER == 'plateau':
            lr_scheduler.step(loss_val)