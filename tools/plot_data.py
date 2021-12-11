
import os
import sys
sys.path.append('../intention2021icra')

import argparse
from configs import cfg

from datasets import make_dataloader
from lib.utils.visualization import Visualizer, vis_results

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
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
cfg.freeze()


# make dataloader
train_dataloader = make_dataloader(cfg, split='train')
viz = Visualizer(mode='image')
for iters, batch in enumerate(tqdm(train_dataloader)):
    if iters % 5 != 0:
        continue
    bboxes = batch['obs_bboxes']
    img_paths = batch['image_files']
    target_intent = batch['obs_intent'].numpy()
    target_action = batch['obs_action'].numpy()
    target_crossing = batch['obs_crossing'].numpy()
    
    # visualize data    
    id_to_show = 0
    for t in range(bboxes.shape[1]):
        gt_behaviors = {
                        'action': int(target_action[id_to_show, t]),
                        'intent': int(target_intent[id_to_show, t]),
                        'crossing': int(target_crossing[id_to_show, t])
                        }
        viz_img = vis_results(viz, 
                                img_paths[t][id_to_show], 
                                bboxes[id_to_show][t], 
                                gt_behaviors=gt_behaviors,
                                pred_behaviors=None,
                                name='',
                                logger=None)
        path_list = img_paths[t][id_to_show].split('/')
        sid, vid, img_id = path_list[-3], path_list[-2], path_list[-1]
        save_path = os.path.join('viz_annos',sid, vid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Image.fromarray(viz_img).save(os.path.join(save_path, img_id))
    