import os
from PIL import Image
import numpy as np
import cv2
from .box_utils import cxcywh_to_x1y1x2y2

neighbor_class_to_name = {0:'pedestrian', 1:'car', 2:'truck', 3:'bus', 4:'train', 5:'bicycle', 6:'bike'}
traffic_light_state_to_name = {1:'red', 2:'yellow', 3:'green'}
traffic_light_class_to_name = {0:'regular', 1:'transit', 2:'pedestrian'}
traffic_sign_class_to_name = {0:'ped_blue', 1:'ped_yellow', 2:'ped_white', 3:'ped_text', 
                     4:'stop_sign', 5:'bus_stop', 6:'train_stop', 7:'construction', 8:'other'}

def print_info(epoch, model, loss_dict, optimizer=None, logger=None, iteration_based=False):
    # loss_dict['kld_weight'] = model.param_scheduler.kld_weight.item()
    # loss_dict['z_logit_clip'] = model.param_scheduler.z_logit_clip.item()
    if iteration_based:
        info = 'Iters:{},'.format(epoch)
    else:
        info = 'Epoch:{},'.format(epoch)
    if hasattr(optimizer, 'param_groups'):
        info += '\t lr:{:6},'.format(optimizer.param_groups[0]['lr'])
        loss_dict['lr'] = optimizer.param_groups[0]['lr']
    for key, v in loss_dict.items():
        info += '\t {}:{:.4f},'.format(key, v) 
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)
    else:
        print(info)

def vis_results(viz, img_path, bboxes, 
                gt_behaviors=None, pred_behaviors=None, 
                neighbor_bboxes=[], neighbor_classes=[],
                traffic_light_bboxes=[], traffic_light_classes=[], traffic_light_states=[],
                traffic_sign_bboxes=[], traffic_sign_classes=[],
                crosswalk_bboxes=[], station_bboxes=[],
                name='', logger=None):
    # 1. initialize visualizer
    viz.initialize(img_path=img_path)

    # 2. draw target pedestrian
    viz.draw_single_bbox(bboxes, gt_behaviors=gt_behaviors, pred_behaviors=pred_behaviors, color=(255., 0, 0))
    
    # 3. draw neighbor
    if len(neighbor_bboxes) > 0:
        for nei_bbox, cls in zip(neighbor_bboxes[:, t], neighbor_classes[:,t]):
            viz.draw_single_bbox(nei_bbox, 
                                color=(0, 255., 0), 
                                class_label=neighbor_class_to_name[int(cls)])

    # draw traffic light 
    if len(traffic_light_bboxes) > 0:
        for light_bbox, cls, state in zip(traffic_light_bboxes[:,t], traffic_light_classes[:,t], traffic_light_states[:,t]):
            viz.draw_single_bbox(light_bbox, color=(0, 125, 255.), 
                                class_label=traffic_light_class_to_name[int(cls)],
                                state_label=traffic_light_state_to_name[int(state)])
    # draw traffic sign
    if len(traffic_sign_bboxes) > 0:
        for sign_bbox, cls in zip(traffic_sign_bboxes[:,t], traffic_sign_classes[:,t]):
            viz.draw_single_bbox(sign_bbox, 
                                color=(125, 0, 125.), 
                                class_label=traffic_sign_class_to_name[int(cls)])

    # draw crosswalk and station
    if len(crosswalk_bboxes) > 0:
        for crosswalk_bbox in crosswalk_bboxes[:,t]:
            viz.draw_single_bbox(crosswalk_bbox, color=(255., 125., 0), 
                                class_label='crosswalk')
    if len(station_bboxes) > 0:
        for station_bbox in station_bboxes[:,t]:
            viz.draw_single_bbox(station_bbox, color=(255., 125., 0), 
                                class_label='transit station')
    viz_img = viz.img
    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)
    return viz_img

class Visualizer():
    def __init__(self, cfg, mode='image'):
        self.mode = mode
        self.cross_type = {0: 'not crossing', 1: 'crossing ego', -1: 'crossing others'}
        if cfg.DATASET.NUM_ACTION == 2:
            self.action_type = {0: 'standing', 1: 'walking'}
        elif cfg.DATASET.NUM_ACTION == 7:
            self.action_type = {0: 'standing', 1: 'waiting', 2: 'going towards', 
                                3: 'crossing', 4: 'crossed and standing', 5: 'crossed and walking', 6: 'other walking'}
        else:
            raise ValueError(cfg.DATASET.NUM_ACTION)
        self.intent_type = {0: 'will not cross', 1: "will cross"}
        if self.mode == 'image':
            self.img = None
        else:
            raise NameError(mode)
            
    def initialize(self, img=None, img_path=None):
        if self.mode == 'image':
            self.img = np.array(Image.open(img_path)) if img is None else img
            self.H, self.W, self.CH = self.img.shape
        # elif self.mode == 'plot':
        #     self.fig, self.ax = plt.subplots()
    
    def visualize(self, 
                  inputs, 
                  id_to_show=0,
                  normalized=False, 
                  bbox_type='x1y1x2y2',
                  color=(255,0,0), 
                  thickness=4, 
                  radius=5,
                  label=None,  
                  viz_type='point', 
                  viz_time_step=None):
        if viz_type == 'bbox':
            self.viz_bbox_trajectories(inputs, normalized=normalized, bbox_type=bbox_type, color=color, viz_time_step=viz_time_step)
        # elif viz_type == 'point':
        #     self.viz_point_trajectories(inputs, color=color, label=label, thickness=thickness, radius=radius)
        # elif viz_type == 'distribution':
        #     self.viz_distribution(inputs, id_to_show, thickness=thickness, radius=radius)

    def draw_single_bbox(self, bbox, class_label=None, state_label=None, gt_behaviors=None, pred_behaviors=None, color=None):
        '''
        img: a numpy array
        bbox: a list or 1d array or tensor with size 4, in x1y1x2y2 format
        behaviors: {'action':0/1, 
                    'crossing':0/1, 
                    'intent':0/1/-1}
        '''
        if color is None:
            color = np.random.rand(3) * 255
        
        cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), color, thickness=2)
        pos = [int(bbox[0]), int(bbox[1])-12]
        cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1]-60)), 
                        (int(bbox[0]+200), int(bbox[1])), color, thickness=-1)
        if class_label is not None:
            cv2.putText(self.img, class_label,  
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2)
            pos[1] -= 20
        if state_label is not None:
            cv2.putText(self.img, 'state: ' + state_label,  
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2)
            pos[1] -= 20

        if gt_behaviors is not None:
            
            if 'action' in gt_behaviors:
                cv2.putText(self.img, 'act: ' + self.action_type[gt_behaviors['action']],  
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
                pos[1] -= 20
            if 'crossing' in gt_behaviors:
                cv2.putText(self.img, 'cross: ' + self.cross_type[gt_behaviors['crossing']],  
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
                pos[1] -= 20
            if 'intent' in gt_behaviors:
                cv2.putText(self.img, 'int: ' + self.intent_type[gt_behaviors['intent']], 
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,255), thickness=2)
                pos[1] -= 20
        if pred_behaviors is not None:
            if 'action' in pred_behaviors:
                cv2.putText(self.img, 'act: ' + str(np.round(pred_behaviors['action'], decimals=2)),  
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,0), thickness=2)
                pos[1] -= 20
            if 'crossing' in pred_behaviors:
                cv2.putText(self.img, 'cross: ' + str(np.round(pred_behaviors['crossing'], decimals=2)),
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,0), thickness=2)
                pos[1] -= 20
            if 'intent' in pred_behaviors:
                cv2.putText(self.img, 'int: ' + str(np.round(pred_behaviors['intent'], decimals=2)), 
                            tuple(pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255,255,0), thickness=2)
    
    def viz_bbox_trajectories(self, bboxes, normalized=False, bbox_type='x1y1x2y2', color=None, thickness=4, radius=5, viz_time_step=None):
        '''
        bboxes: (T,4) or (T, K, 4)
        '''
        if len(bboxes.shape) == 2:
            bboxes = bboxes[:, None, :]

        if normalized:
            bboxes[:,[0, 2]] *= self.W
            bboxes[:,[1, 3]] *= self.H
        if bbox_type == 'cxcywh':
            bboxes = cxcywh_to_x1y1x2y2(bboxes)
        elif bbox_type == 'x1y1x2y2':
            pass
        else:
            raise ValueError(bbox_type)
        bboxes = bboxes.astype(np.int32)
        T, K, _ = bboxes.shape

        # also draw the center points
        center_points = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]])/2 # (T, K, 2)
        self.viz_point_trajectories(center_points, color=color, thickness=thickness, radius=radius)

        # draw way point every several frames, just to make it more visible
        if viz_time_step:
            bboxes = bboxes[viz_time_step, :]
            T = bboxes.shape[0]
        for t in range(T):
            for k in range(K):
                self.draw_single_bbox(bboxes[t, k, :], color=color)
        
    