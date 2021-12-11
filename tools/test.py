
import os
import sys
sys.path.append('../pedestrian_intent_action_detection')

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import argparse
from configs import cfg

from datasets import make_dataloader
from lib.modeling import make_model
from lib.engine.trainer import do_train, do_val
from lib.engine.inference import inference
import glob

import pickle as pkl
import logging
from termcolor import colored 
from lib.utils.logger import Logger
import logging
import pdb


parser = argparse.ArgumentParser(description="PyTorch intention detection testing")
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
    logger = Logger("FOL",
                    cfg,
                    project = cfg.PROJECT,
                    viz_backend="wandb"
                    )
    run_id = logger.run_id
else:
    logger = logging.Logger("FOL")
    run_id = 'no_wandb'

# make dataloader
test_dataloader = make_dataloader(cfg, split='test')
# make model
model = make_model(cfg).to(cfg.DEVICE)
if os.path.isfile(cfg.CKPT_DIR):
    checkpoints = [cfg.CKPT_DIR]
else:
    checkpoints = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, '*.pth')), key=os.path.getmtime)
if not checkpoints:
    print(colored("Checkpoint not loaded !!", 'white', 'on_red'))
    result_dict = inference(cfg, 0, model, test_dataloader, cfg.DEVICE, logger=logger)
else:
    for checkpoint in checkpoints:    
        model.load_state_dict(torch.load(checkpoint))
        print(colored("Checkpoint loaded: {}".format(checkpoint), 'white', 'on_green'))
        result_dict = inference(cfg, 0, model, test_dataloader, cfg.DEVICE, logger=logger)
