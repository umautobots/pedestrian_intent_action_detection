import numpy as np
from collections import defaultdict
import torch
from lib.utils.visualization import Visualizer, vis_results, print_info
from lib.utils.eval_utils import compute_acc_F1, compute_AP, compute_auc_ap
from tqdm import tqdm
import time

def inference(cfg, epoch, model, dataloader, device, logger=None, iteration_based=False):
    model.eval()
    max_iters = len(dataloader)
    
    viz = Visualizer(cfg, mode='image')
    
    # Collect outputs
    gt_actions, gt_intents = defaultdict(list), defaultdict(list)
    det_actions, pred_actions, det_intents, det_attentions = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    gt_bboxes, all_image_pathes =  defaultdict(list), defaultdict(list)
    # gt_traffics = defaultdict(list)
    dataloader.dataset.__getitem__(0)
    total_times = []
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            x = batch['img_patches'].to(device)
            bboxes = batch['obs_bboxes'].to(device)
            local_bboxes = batch['local_bboxes'].to(device) if batch['local_bboxes'] is not None else None
            masks = None #batch['masks'].to(device)
            img_path = batch['image_files']
            target_intent = batch['obs_intent'].numpy()
            target_action = batch['obs_action'].numpy()

            track_ids = batch['pids']
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO or cfg.MODEL.WITH_TRAFFIC else None
            x_traffic = None
            if cfg.MODEL.WITH_TRAFFIC:
                # gt_traffic = {}
                if cfg.MODEL.PRETRAINED:
                    x_traffic = batch['traffic_features'].to(device)
                    
                else:
                    x_traffic = {}
                    if 'x_neighbor' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_neighbor'] = batch['neighbor_bboxes']
                        x_traffic['cls_neighbor'] = batch['neighbor_classes']
                        # gt_traffic['neighbor'] = batch['neighbor_orig'] if 'neighbor_orig' in batch else None 
                    if 'x_light' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_light'] = batch['traffic_light']
                        x_traffic['cls_light'] = batch['traffic_light_classes']
                        # gt_traffic['traffic_light'] = batch['traffic_light_orig'] if 'traffic_light_orig' in batch else None 
                    if 'x_sign' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_sign'] = batch['traffic_sign']
                        x_traffic['cls_sign'] = batch['traffic_sign_classes']
                        # gt_traffic['traffic_sign'] = batch['traffic_sign_orig'] if 'traffic_sign_orig' in batch else None 
                    if 'x_crosswalk' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_crosswalk'] = batch['crosswalk']
                        x_traffic['cls_crosswalk'] = batch['crosswalk_classes']
                        # gt_traffic['crosswalk'] = batch['crosswalk_orig'] if 'crosswalk_orig' in batch else None 
                    if 'x_station' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_station'] = batch['station']
                        x_traffic['cls_station'] = batch['station_classes']
                        # gt_traffic['station'] = batch['station_orig'] if 'station_orig' in batch else None 
            
            # start = time.time()
            act_det_scores, act_pred_scores, int_det_scores, attentions = model(x, 
                                                                                        bboxes, 
                                                                                        x_ego=ego_motion,
                                                                                        x_traffic=x_traffic, 
                                                                                        local_bboxes=local_bboxes, 
                                                                                        masks=masks)
            # total_times.append((time.time() - start)/x.shape[1])
            # continue
            for i in range(len(attentions)):
                for k in attentions[i].keys():
                    attentions[i][k] = attentions[i][k].cpu().numpy()

            if act_det_scores is not None:
                if act_det_scores.shape[-1] == 1:
                    act_det_scores = act_det_scores.sigmoid().detach().cpu().numpy()
                else:
                    act_det_scores = act_det_scores.softmax(dim=-1).detach().cpu().numpy()
            if act_pred_scores is not None:
                if act_pred_scores.shape[-1] == 1:
                    act_pred_scores = act_pred_scores.sigmoid().detach().cpu().numpy()
                else:
                    act_pred_scores = act_pred_scores.softmax(dim=-1).detach().cpu().numpy()
            if int_det_scores is not None:
                if int_det_scores.shape[-1] == 1:
                    int_det_scores = int_det_scores.sigmoid().detach().cpu().numpy()
                else:
                    int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy()
            # NOTE: collect outputs
            bboxes = bboxes.detach().cpu().numpy()   
            for i, trk_id in enumerate(track_ids):
                gt_actions[trk_id].append(target_action[i])
                gt_intents[trk_id].append(target_intent[i])
                gt_bboxes[trk_id].append(bboxes[i])
                all_image_pathes[trk_id].append(img_path[i])
                
                det_actions[trk_id].append(act_det_scores[i])
                pred_actions[trk_id].append(act_pred_scores[i])
                det_intents[trk_id].append(int_det_scores[i])
                if len(track_ids) == 1:
                    det_attentions[trk_id] = attentions
                else:
                    det_attentions[trk_id].append(attentions[i])
                # gt_traffics[trk_id].append(gt_traffic)
            
            if cfg.VISUALIZE and iters % max(int(len(dataloader)/15), 1) == 0:
                if cfg.DATASET.BBOX_NORMALIZE:
                    # NOTE: denormalize bboxes
                    _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
                    _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
                    bboxes = bboxes * (_max - _min) + _min

                id_to_show = np.random.randint(bboxes.shape[0])
                gt_behaviors, pred_behaviors = {}, {}
                if 'action' in cfg.MODEL.TASK:
                    gt_behaviors['action'] = target_action[id_to_show, -1]
                    pred_behaviors['action'] = act_det_scores[id_to_show, -1]

                if 'intent' in cfg.MODEL.TASK:
                    gt_behaviors['intent'] = target_intent[id_to_show, -1]
                    pred_behaviors['intent'] = int_det_scores[id_to_show, -1]
                # visualize input
                input_images = []
                for i in range(4):
                    row = []
                    for j in range(4):
                        if i*4+j < x.shape[2]:
                            row.append(x[id_to_show, :, i*4+j,...].detach().cpu())
                        else:
                            row.append(torch.zeros_like(x[id_to_show, :, 0, ...]).cpu())
                    input_images.append(torch.cat(row, dim=2))
                input_images = torch.cat(input_images, dim=1).permute(1, 2, 0).numpy() 
                input_images = 255 * (input_images+1) / 2
                logger.log_image(input_images, label='input_test')

                vis_results(viz, 
                            img_path[id_to_show][-1], 
                            bboxes[id_to_show][-1], 
                            gt_behaviors=gt_behaviors,
                            pred_behaviors=pred_behaviors,
                            name='intent_test',
                            logger=logger)  

    predictions = {'gt_bboxes': gt_bboxes,
                    'gt_intents': gt_intents,
                    'det_intents': det_intents,
                    'gt_actions': gt_actions,
                    'det_actions': det_actions,
                    'pred_actions': pred_actions,
                    'frame_id': all_image_pathes,
                    'attentions': det_attentions,
                    # 'gt_traffics': gt_traffics,
                    }

    # compute accuracy and F1 scores
    # NOTE: PIE paper uses simple acc and f1 computation: score > 0.5 is positive, score < 0.5 is negative
    result_dict = {}
    if iteration_based:
        info = 'Iters: {}; \n'.format(epoch)
    else:
        info = 'Epoch: {}; \n'.format(epoch)
    if 'action' in cfg.MODEL.TASK:
        tmp_gt_actions, tmp_det_actions = [], []
        for k, v in gt_actions.items():
            tmp_gt_actions.extend(v)
            tmp_det_actions.extend(det_actions[k])

        if cfg.STYLE == 'PIE':
            gt_actions = np.concatenate(tmp_gt_actions, axis=0)
            det_actions = np.concatenate(tmp_det_actions, axis=0)
            gt_actions = gt_actions.reshape(-1)
            det_actions = det_actions.reshape(-1, det_actions.shape[-1])
        elif cfg.STYLE == 'SF-GRU':
            gt_actions = np.stack(tmp_gt_actions)
            det_actions = np.stack(tmp_det_actions)
            gt_actions = gt_actions[:, -1]# only last frame
            det_actions = det_actions[:, -1]# only last frame
            
        else:
            raise ValueError(cfg.STYLE)

        info += 'Action:\n'
        if cfg.DATASET.NUM_ACTION == 2:
            res, info = compute_acc_F1(det_actions, gt_actions, info, _type='action')
        else:
            res, info = compute_AP(det_actions, gt_actions, info, _type='action')
        result_dict.update(res)
        info += '\n'
    if 'intent' in cfg.MODEL.TASK:
        tmp_gt_intents, tmp_det_intents = [], []
        for k, v in gt_intents.items():
            tmp_gt_intents.extend(v)
            tmp_det_intents.extend(det_intents[k])
        
        if cfg.STYLE == 'PIE':
            gt_intents = np.concatenate(tmp_gt_intents, axis=0)
            det_intents = np.concatenate(tmp_det_intents, axis=0)
            gt_intents = gt_intents.reshape(-1)
            det_intents = det_intents.reshape(-1, det_intents.shape[-1])
        elif cfg.STYLE == 'SF-GRU':
            gt_intents = np.stack(tmp_gt_intents)
            det_intents = np.stack(tmp_det_intents)
            gt_intents = gt_intents[:, -1] # only last frame
            det_intents = det_intents[:, -1] # only last frame
        else:
            raise ValueError(cfg.STYLE)
        
        info += 'Intent:\n'
        if cfg.DATASET.NUM_INTENT == 2:
            res, info = compute_auc_ap(det_intents, gt_intents, info, _type='intent')
            res_acc_F1, info = compute_acc_F1(det_intents, gt_intents, info, _type='intent')
            res.update(res_acc_F1)
            res['score_difference'] = np.mean(det_intents[gt_intents==1]) - np.mean(det_intents[gt_intents==0])
            info += 'score_difference:{:3}; '.format(res['score_difference'])
        else:
            res, info = compute_AP(det_intents, det_intents, info, _type='intent')
        result_dict.update(res)
   
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(result_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)
    
    return result_dict