'''
the trainer to pretrain the relation embedding network
'''
import torch
import os
import numpy as np
import torch
import torch.nn.functional as F
from lib.utils.visualization import Visualizer, vis_results, print_info
from lib.modeling.layers.cls_loss import binary_cross_entropy_loss, cross_entropy_loss, trn_loss
from lib.utils.meter import AverageValueMeter
from lib.engine.inference_relation import inference
from tqdm import tqdm
import time
import pdb
def do_val(cfg, epoch, model, dataloader, device, logger=None, iteration_based=False):
    model.eval()
    loss_intent_meter = AverageValueMeter()

    loss_act, loss_intent = [], []
    loss_func = {}
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss
    
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
    
            x_ped = batch['obs_bboxes'].to(device)
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO else None
            x_neighbor = batch['neighbor_bboxes']
            cls_neighbor = batch['neighbor_classes']
            x_light = batch['traffic_light']
            x_sign = batch['traffic_sign']
            x_crosswalk = batch['crosswalk']
            x_station = batch['station']

            img_path = batch['cur_image_file']
            target_intent = batch['obs_intent'].to(device)
            # target_action = batch['obs_action'].to(device)

            int_det_scores, relation_feature = model(x_ped, 
                                                     x_neighbor, 
                                                     cls_neighbor, 
                                                     x_ego=ego_motion, 
                                                     x_light=x_light, 
                                                     x_sign=x_sign, 
                                                     x_crosswalk=x_crosswalk, 
                                                     x_station=x_station)
                       
            if int_det_scores is not None:
                loss_intent_meter.add(loss_func['int_det'](int_det_scores, target_intent).item())
            
    loss_dict = {}
    
    if 'intent' in cfg.MODEL.TASK:
        loss_dict['loss_intent_val'] = loss_intent_meter.mean
    print_info(epoch, model, loss_dict, optimizer=None, logger=logger, iteration_based=iteration_based)
    
    return sum([v for v in loss_dict.values()])


def do_train_iteration(cfg, model, optimizer, 
                       train_dataloader, val_dataloader, test_dataloader, 
                       device, logger=None, lr_scheduler=None, save_checkpoint_dir=None):
    model.train()
    max_iters = len(train_dataloader)
    viz = Visualizer(cfg, mode='image')
    # trainning loss meters
    loss_intent_meter = AverageValueMeter()
    # loss functions
    loss_func = {}
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss
    with torch.set_grad_enabled(True):
        end = time.time()
        for iters, batch in enumerate(tqdm(train_dataloader), start=1):
            data_time = time.time() - end

            x_ped = batch['obs_bboxes'].to(device)
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO else None
            x_neighbor = batch['neighbor_bboxes']
            cls_neighbor = batch['neighbor_classes']
            x_light = batch['traffic_light']
            x_sign = batch['traffic_sign']
            x_crosswalk = batch['crosswalk']
            x_station = batch['station']

            img_path = batch['cur_image_file']
            target_intent = batch['obs_intent'].to(device)
            # target_action = batch['obs_action'].to(device)

            int_det_scores, relation_feature = model(x_ped, 
                                                     x_neighbor, 
                                                     cls_neighbor, 
                                                     x_ego=ego_motion, 
                                                     x_light=x_light, 
                                                     x_sign=x_sign, 
                                                     x_crosswalk=x_crosswalk, 
                                                     x_station=x_station)
            # get loss and update loss meters
            loss, loss_dict = 0.0, {}

            if int_det_scores is not None:
                loss_intent = loss_func['int_det'](int_det_scores, target_intent)
                if False: #act_det_scores is not None and hasattr(model, 'param_scheduler'):
                    loss += model.param_scheduler.intent_weight * loss_intent
                    loss_dict['intent_weight'] = model.param_scheduler.intent_weight.item()
                else:
                    loss += loss_intent
                loss_intent_meter.add(loss_intent.item())
                loss_dict['loss_int_det_train'] = loss_intent_meter.mean
            
            # weight
            if hasattr(model, 'param_scheduler'):
                model.param_scheduler.step()

            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # gradient clip
            loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            batch_time = time.time() - end
            loss_dict['batch_time'] = batch_time
            loss_dict['data_time'] = data_time

            # model.param_scheduler.step()
            if cfg.SOLVER.SCHEDULER == 'exp':
                lr_scheduler.step()
            # print log
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(iters, model, loss_dict, optimizer=optimizer, logger=logger, iteration_based=True)
            # visualize
            if cfg.VISUALIZE and iters % 50 == 0 and hasattr(logger, 'log_image'):
                bboxes = x_ped.detach().cpu().numpy()
                if cfg.DATASET.BBOX_NORMALIZE:
                    # NOTE: denormalize bboxes
                    _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
                    _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
                    bboxes = bboxes * (_max - _min) + _min
                
                id_to_show = np.random.randint(bboxes.shape[0])
                gt_behaviors, pred_behaviors = {}, {}

                if 'intent' in cfg.MODEL.TASK:
                    target_intent = target_intent.detach().cpu().numpy()
                    if int_det_scores.shape[-1] == 1:
                        int_det_scores = int_det_scores.sigmoid().detach().cpu().numpy()
                    else:
                        int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy() 
                    gt_behaviors['intent'] = target_intent[id_to_show, -1]
                    pred_behaviors['intent'] = int_det_scores[id_to_show, -1]

                # visualize result  
                vis_results(viz, 
                            img_path[id_to_show], 
                            bboxes[id_to_show][-1], 
                            gt_behaviors=gt_behaviors,
                            pred_behaviors=pred_behaviors,
                            name='intent_train',
                            logger=logger)
                
            end = time.time()
            # do validation
            if iters % 100 == 0:
                loss_val = do_val(cfg, iters, model, val_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if cfg.SOLVER.SCHEDULER == 'plateau':
                    lr_scheduler.step(loss_val)
            # do test
            if iters % 250 == 0:
                result_dict = inference(cfg, iters, model, test_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if 'intent' in cfg.MODEL.TASK:
                    save_file = os.path.join(save_checkpoint_dir, 
                                        'iters_{}_acc_{:.3}_f1_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                            result_dict['intent_accuracy'],
                                                                            result_dict['intent_f1']))
                else:
                    save_file = os.path.join(save_checkpoint_dir, 
                                    'iters_{}_mAP_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                        result_dict['mAP']))
                torch.save(model.state_dict(), save_file)