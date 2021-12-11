import numpy as np
import torch
from lib.utils.visualization import Visualizer, vis_results
from lib.utils.eval_utils import compute_acc_F1, compute_AP, compute_auc_ap
from tqdm import tqdm
import pickle as pkl

def inference(cfg, epoch, model, dataloader, device, logger=None, iteration_based=False):
    model.eval()
    max_iters = len(dataloader)
    
    viz = Visualizer(cfg, mode='image')
    # loss_act, loss_intent = 0, 0
    gt_bboxes, gt_intents, det_intents, all_image_pathes = [],[],[],[]
    dataloader.dataset.__getitem__(0)
    all_relation_features = {}
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

            cur_img_path = batch['cur_image_file']
            image_files = batch['image_files']
            pids = batch['pids']
            target_intent = batch['obs_intent']
            
            int_det_scores, relation_feature = model(x_ped, 
                                                     x_neighbor, 
                                                     cls_neighbor, 
                                                     x_ego=ego_motion, 
                                                     x_light=x_light, 
                                                     x_sign=x_sign, 
                                                     x_crosswalk=x_crosswalk, 
                                                     x_station=x_station)
            relation_feature = relation_feature.cpu().numpy()
            for i in range(len(image_files)):
                for t in range(len(image_files[i])):
                    img_id = image_files[i][t].split('/')[-1].split('.')[0]
                    key = pids[i] + '_' + img_id
                    if key not in all_relation_features:
                        all_relation_features[key] = relation_feature[i, t:t+1]
            bboxes = x_ped
            gt_intents.append(target_intent.view(-1).numpy())
            gt_bboxes.append(bboxes.detach().cpu().numpy())
          
            if int_det_scores is not None:
                if int_det_scores.shape[-1] == 1:
                    int_det_scores = int_det_scores.sigmoid().detach().cpu()
                else:
                    int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu()
                det_intents.append(int_det_scores.view(-1, int_det_scores.shape[-1]).numpy())
            if cfg.VISUALIZE and iters % max(int(len(dataloader)/15), 1) == 0:
                bboxes = bboxes.detach().cpu().numpy()
                if cfg.DATASET.BBOX_NORMALIZE:
                    # NOTE: denormalize bboxes
                    _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
                    _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
                    bboxes = bboxes * (_max - _min) + _min

                id_to_show = np.random.randint(bboxes.shape[0])
                gt_behaviors, pred_behaviors = {}, {}
                
                if 'intent' in cfg.MODEL.TASK:
                    target_intent = target_intent.detach().cpu().numpy()
                    int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy() 
                    gt_behaviors['intent'] = target_intent[id_to_show, -1]
                    pred_behaviors['intent'] = int_det_scores[id_to_show, -1]

                vis_results(viz, 
                            cur_img_path[id_to_show], 
                            bboxes[id_to_show][-1], 
                            gt_behaviors=gt_behaviors,
                            pred_behaviors=pred_behaviors,
                            name='intent_test',
                            logger=logger)  
    predictions = {'gt_bboxes': gt_bboxes,
                    'gt_intents': gt_intents,
                    'det_intents': det_intents,
                    'frame_id': all_image_pathes,
                    }
    pkl.dump(all_relation_features, open('relation_features_test.pkl', 'wb'))

    # compute accuracy and F1 scores
    # NOTE: PIE paper uses simple acc and f1 computation: score > 0.5 is positive, score < 0.5 is negative
    result_dict = {}
    if iteration_based:
        info = 'Iters: {}; \n'.format(epoch)
    else:
        info = 'Epoch: {}; \n'.format(epoch)

    if 'intent' in cfg.MODEL.TASK:
        gt_intents = np.concatenate(gt_intents, axis=0)
        det_intents = np.concatenate(det_intents, axis=0)
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
        logger.log_values(result_dict)
    else:
        print(info)
    
    
    return result_dict