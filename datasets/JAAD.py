import os
from collections import defaultdict
import json
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from datasets.JAAD_origin import JAAD

import copy
import glob
import time
import pdb

class JAADDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.root = cfg.DATASET.ROOT
        self.cfg = cfg
        self.dt = 1/self.cfg.DATASET.FPS
        # NOTE: add downsample function
        self.downsample_step = int(30/self.cfg.DATASET.FPS)
        
        if self.cfg.DATASET.NUM_ACTION == 2:
            self.action_type = {0: 'standing', 1: 'walking'}
        elif self.cfg.DATASET.NUM_ACTION == 7:
            self.action_type = {0: 'standing', 1: 'waiting', 2: 'going towards', 
                                3: 'crossing', 4: 'crossed and standing', 5: 'crossed and walking', 6: 'other walking'}
        else:
            raise ValueError(self.cfg.DATASET.NUM_ACTION)
        
        self.traffic_classes = {0:'pedestrian', 1:'car', 2:'truck', 3:'bus', 4:'train', 5:'bicycle', 6:'bike'}
        self.traffic_light_states = {1:'red', 2:'yellow', 3:'green'}
        self.traffic_light_type = {0:'regular', 1:'transit', 2:'pedestrian'}
        self.traffic_sign_type = {0:'ped_blue', 1:'ped_yellow', 2:'ped_white', 3:'ped_text', 
                                  4:'stop_sign', 5:'bus_stop', 6:'train_stop', 7:'construction', 8:'other'}
        
        if self.cfg.STYLE == 'PIE':
            intent_data_opts = {'fstride': 1,
                                'sample_type': 'all',#, 'beh', #
                                'height_rng': [0, float('inf')],
                                'squarify_ratio': 0,
                                'data_split_type': 'default',  #  kfold, random, default
                                'seq_type': 'intention', #  crossing , intention
                                'max_size_observe': self.cfg.MODEL.SEG_LEN,  # number of observation frames
                                'max_size_predict': self.cfg.MODEL.PRED_LEN,  # number of prediction frames
                                'seq_overlap_rate': self.cfg.DATASET.OVERLAP,  # how much consecutive sequences overlap
                                'balance': False,  # balance the training and testing samples
                                'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                                'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                                'encoder_input_type': [],
                                'decoder_input_type': ['bbox'],
                                'output_type': ['intention_binary'],
                                'min_track_size': 15, #  discard tracks that are shorter
                                }
        # ------------------------------------------------------------------------ 
        elif self.cfg.STYLE == 'SF-GRU':
            # NOTE: added to test in SF_GRU mode
            intent_data_opts ={'fstride': 1,
                'sample_type': 'all', #'beh',
                'subset': 'default',
                'data_split_type': 'default',  # kfold, random, default
                'seq_type': 'intention',
                'encoder_input_type': [],
                'decoder_input_type': ['bbox'],
                'output_type': ['intention_binary'],
                'balance': False,  # balance the training and testing samples
                'max_size_observe': 15, #self.cfg.MODEL.INPUT_LEN,  # number of observation frames
                'max_size_predict': 5,  # number of prediction frames
                'seq_overlap_rate': 0.5, #self.cfg.DATASET.OVERLAP,  # how much consecutive sequences overlap
                'min_track_size': 75} ## for obs length of 15 frames + 60 frames tte. This should be adjusted for different setup
        # ------------------------------------------------------------------------

        
        imdb = JAAD(data_path=self.root, style=self.cfg.STYLE)
        self.beh_seq = imdb.generate_data_trajectory_sequence(self.split, **intent_data_opts)
        
        if split in ['train', 'val'] and intent_data_opts['balance']:
            self.beh_seq = imdb.balance_samples_count(self.beh_seq, label_type='intention_binary')
        data_type = intent_data_opts['encoder_input_type'] + intent_data_opts['decoder_input_type'] + intent_data_opts['output_type']
        
        if self.cfg.DATASET.NUM_ACTION == 7:
            new_action_seq = self.generate_new_actions(self.beh_seq)
            self.beh_seq['action_seq'] = new_action_seq

        if self.cfg.MODEL.WITH_TRAFFIC:
            traffic_feature_path = os.path.join(self.root, 'relation_features', self.split+'.pkl')
            if self.cfg.MODEL.PRETRAINED and os.path.exists(traffic_feature_path):
                self.traffic_features = pkl.load(open(traffic_feature_path, 'rb'))
            else:
                self.traffic_features = None
        

        # ------------------------------------------------------------------------
        if self.cfg.STYLE == 'PIE':
            model_opts = {'time_to_event':None}
        elif self.cfg.STYLE == 'SF-GRU':
            # NOTE: added to test in SF_GRU mode
            model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
                'enlarge_ratio': 1.5,
                'pred_target_type': ['crossing'],
                'obs_length': 15,  # Determines min track size
                'time_to_event': [30, 60], # Determines min track size, use 60 if compare with SF-GRU
                'dataset': 'pie',
                'normalize_boxes': True}
        # ------------------------------------------------------------------------
        self.data = self.get_intent_data(self.beh_seq, data_type, 
                                            intent_data_opts['max_size_observe'], 
                                            intent_data_opts['max_size_predict'], 
                                            intent_data_opts['seq_overlap_rate'],
                                            model_opts['time_to_event'])
        for k, v in self.data.items():
            if len(v) != self.__len__():
                raise ValueError("{} length is wrong: {}".format(k, len(v)))

        # get sample weights based on intention type
        self.get_weights()
        
    def __getitem__(self, index):

        pids = self.data['obs_pid'][index]
        obs_bboxes = torch.FloatTensor(self.data['obs_bbox'][index])
        pred_bboxes = torch.FloatTensor(self.data['pred_bbox'][index])
        if self.split == 'test':
            cur_image_file = self.data['obs_image'][index]
        else:
            cur_image_file = self.data['obs_image'][index][-1]
        resolution = torch.FloatTensor(self.data['obs_resolution'][index])
        
        # normalize boxes
        if self.cfg.DATASET.BBOX_NORMALIZE:
            obs_bboxes = self.convert_normalize_bboxes(obs_bboxes, resolution, normalize='zero-one', bbox_type='x1y1x2y2')
            pred_bboxes = self.convert_normalize_bboxes(pred_bboxes, resolution, normalize='zero-one', bbox_type='x1y1x2y2')
        
        ret = {'obs_bboxes':obs_bboxes, 'pred_bboxes':pred_bboxes, 'cur_image_file':cur_image_file, 'resolution':resolution}
        
        # get target info
        ret['image_files'] = self.data['obs_image'][index]
        ret['pids'] = pids[0][0]
        # end = time.time()
        if self.cfg.MODEL.TYPE == 'conv3d':
            ret['img_patches'], ret['local_bboxes'] = self.load_image_patches(pids, self.data['obs_image'][index])
        elif 'rnn' in self.cfg.MODEL.TYPE:
            ret['local_bboxes'] = None
            ret['img_patches'] = self.load_extracted_features(pids, self.data['obs_image'][index]) #ret['local_bboxes'] 
        # else:
        #     raise NameError(self.cfg.MODEL.TYPE)
            
        ret['obs_intent'] = torch.tensor(self.data['obs_intent'][index]).squeeze()
        ret['obs_action'] = torch.tensor(self.data['obs_action'][index])
        ret['pred_action'] = torch.tensor(self.data['pred_action'][index])
        ret['obs_crossing'] = torch.tensor(self.data['obs_crossing'][index])

        
        # get neighbor info
        if self.cfg.MODEL.WITH_TRAFFIC:
            if self.traffic_features is None:
                ret.update(self.extract_traffic_features(index, obs_bboxes))
            else:
                ret['traffic_features'] = []
                for pid, img_file in zip(pids, self.data['obs_image'][index]):
                    key = pid[0] + '_' + img_file.split('/')[-1].split('.')[0] 
                    ret['traffic_features'].append(torch.FloatTensor(self.traffic_features[key]))
                ret['traffic_features'] = torch.cat(ret['traffic_features'], dim=0)
        # get ego info
        ret['obs_ego_motion'] = torch.FloatTensor(self.data['obs_ego_motion'][index]).unsqueeze(-1)
        ret['pred_ego_motion'] = torch.FloatTensor(self.data['pred_ego_motion'][index])
        
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def extract_traffic_features(self, index, ped_bboxes):
        keys = ['neighbor_bboxes', 
                'neighbor_classes', 
                'neighbor_ids', 
                'traffic_light',
                'traffic_light_classes', 
                'traffic_sign', 
                'traffic_sign_classes',
                'crosswalk', 
                'crosswalk_classes',
                ]
    
        ret = {k:[torch.FloatTensor([])] for k in keys}
        ped_x_c, ped_y_c, ped_y_b = (ped_bboxes[:, 0]+ped_bboxes[:,2])/2, (ped_bboxes[:,1]+ped_bboxes[:,3])/2, ped_bboxes[:, 3]
        
        # 1.get pedestrian feature
        if len(self.data['obs_traffic_bbox'][index]) > 0:
            # neighbor feature is the bbox difference [x1_diff, y1_diff, x2_diff, y2_diff]
            neighbor_bboxes = torch.FloatTensor(self.data['obs_traffic_bbox'][index])
            neighbor_classes = torch.FloatTensor(self.data['obs_traffic_class'][index])
            # NOTE: change the class to -1 if object does not exit on that frame.
            neighbor_classes[neighbor_bboxes.max(dim=-1)[0] == 0] = -1 
            
            ret['neighbor_bboxes'] = [neighbor_bboxes - ped_bboxes.unsqueeze(0)]
            ret['neighbor_classes'] = [neighbor_classes]
            ret['neighbor_ids'] = [self.data['obs_traffic_obj_id'][index]]
        # 2. get sign, light, crosswalk
        ret['traffic_sign'] = torch.FloatTensor(self.data['obs_traffic_sign_class'][index])
        ret['traffic_light'] = torch.FloatTensor(self.data['obs_traffic_light_class'][index]).unsqueeze(-1)
        ret['crosswalk'] = torch.FloatTensor(self.data['obs_crosswalk_class'][index]).unsqueeze(-1)
        ret['traffic_sign_classes'] = torch.ones((ret['traffic_sign'].shape[0]))
        ret['traffic_light_classes'] = torch.ones((ret['traffic_light'].shape[0]))
        ret['crosswalk_classes'] = torch.ones((ret['crosswalk'].shape[0]))

        return ret 

    def get_weights(self):
        # NOTE: add weights to data samples based intent, action or intent+action classes
        if self.split == 'train' and self.cfg.DATALOADER.WEIGHTED == 'intent':
            weights = [0 for i in range(self.cfg.DATASET.NUM_INTENT)]
            for obs_intent in self.data['obs_intent']:
                weights[obs_intent[-1][0]] += 1
            weights[0], weights[1] = weights[0]/weights[0], weights[0]/weights[1]

            self.weights = []
            for obs_intent in self.data['obs_intent']:
                self.weights.append(weights[obs_intent[-1][0]])
        elif self.split == 'train' and self.cfg.DATALOADER.WEIGHTED == 'action':
            weights = [0 for i in range(self.cfg.DATASET.NUM_ACTION)]
            for obs_action in self.data['obs_action']:
                weights[obs_action[-1]] += 1
            base = weights[-1]
            for i in range(self.cfg.DATASET.NUM_ACTION):
                weights[i] = base/(weights[i] + 1e-7)

            self.weights = []
            for obs_action in self.data['obs_action']:
                self.weights.append(weights[obs_action[-1]])
        elif self.split == 'train' and self.cfg.DATALOADER.WEIGHTED == 'action_intent':
            weights = [0 for i in range(self.cfg.DATASET.NUM_ACTION * self.cfg.DATASET.NUM_INTENT)]
            for obs_action, obs_intent in zip(self.data['obs_action'], self.data['obs_intent']):
                action = obs_action[-1]
                intent = obs_intent[-1][0]
                weights[self.cfg.DATASET.NUM_ACTION*intent + action] += 1
            base = weights[-4] # use intent to cross and crossing as a base class to compute weight
            for i in range(self.cfg.DATASET.NUM_ACTION * self.cfg.DATASET.NUM_INTENT):
                # NOTE: Nov 11th, ignore classes that have less than 15 samples, they can be false annotations
                weights[i] = base/(weights[i]) if weights[i] >15 else 0 
            self.weights = []
            for obs_action, obs_intent in zip(self.data['obs_action'], self.data['obs_intent']):
                action = obs_action[-1]
                intent = obs_intent[-1][0]
                self.weights.append(weights[self.cfg.DATASET.NUM_ACTION*intent + action])
        elif self.split == 'val':
            self.weights = [1 for i in range(len(self.data['obs_intent']))]
        elif self.split == 'test' or self.cfg.DATALOADER.WEIGHTED == 'none':
            self.weights = []
        else:
            raise ValueError("Unknown weight config: "+self.cfgDATALOADER.WEIGHTED)

    def load_extracted_features(self, pids, img_files):
        features = []
        # local_bboxes = []
        for t, img_path in enumerate(img_files):
            pid = pids[t][0]
            path_list = img_path.split('/')
            sid, vid, img_id = path_list[-3], path_list[-2], path_list[-1].strip('.png')
            
            if self.cfg.DATASET.NAME == 'JAAD':
               name = 'jaad'
            elif self.cfg.DATASET.NAME == 'PIE':
               name = 'pie' 
            data_path = os.path.join(self.cfg.DATASET.ROOT, 'saved_output/data/', name, self.split, 'features_context_pad_resize/vgg16_none',
                                     sid, vid, img_id + '_' + pid + '.pkl')
            feature = pkl.load(open(data_path, 'rb'))
            features.append(torch.tensor(feature).squeeze(0))
            
        return torch.stack(features, dim=0).permute(0,3,1,2)
        
    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize, bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))
            

        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_intent_data(self, data, data_type, observe_length, pred_length, overlap, time_to_event=None):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """
        # NOTE: add the downsample function in get_intent_tracks
        if self.cfg.STYLE == 'PIE':
            tracks = self.get_intent_tracks(data, data_type, observe_length, pred_length, overlap)
        elif self.cfg.STYLE == 'SF-GRU':
            tracks = self.get_intent_tracks_new(data, data_type, observe_length, pred_length, overlap, time_to_event, normalize=False)
        else:
            raise NameError(self.cfg.STYLE)
        
        if self.cfg.MODEL.TYPE == 'rnn' and self.split == 'test':
            pred_length = 0
        obs_slices = defaultdict(list)
        pred_slices = defaultdict(list)
        
        # Generate observation data input to encoder        
        for k, v in tracks.items():
            # obs_slices[k] = []
            # pred_slices[k] = []
            if k in ['traffic_bbox', 'traffic_class', 'traffic_obj_id']:
                obs_slices[k].extend([d[:, :d.shape[1]-pred_length] if len(d) >0 else [] for d in tracks[k]])
                pred_slices[k].extend([d[:, d.shape[1]-pred_length:] if len(d) >0 else [] for d in tracks[k]])
            else:
                obs_slices[k].extend([d[:len(d)-pred_length] for d in tracks[k]])
                pred_slices[k].extend([d[len(d)-pred_length:] for d in tracks[k]])
        ret =  {
                'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'obs_resolution': obs_slices['resolution'],
                'obs_bbox': obs_slices['bbox'], # enc_input
                'obs_crossing':obs_slices['crossing_seq'],
                'obs_action':obs_slices['action_seq'],
                'obs_intent':obs_slices['intention_binary'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'pred_resolution': pred_slices['resolution'],
                'pred_bbox': pred_slices['bbox'], #pred_target,
                'pred_crossing':pred_slices['crossing_seq'],
                'pred_action':pred_slices['action_seq'],
                'pred_intent':pred_slices['intention_binary'],
                'obs_traffic_bbox': obs_slices['traffic_bbox'], # for interaction learning purpose.
                'obs_traffic_class': obs_slices['traffic_class'],
                'obs_traffic_obj_id': obs_slices['traffic_obj_id'],
                'pred_traffic_bbox': pred_slices['traffic_bbox'],
                'pred_traffic_class': pred_slices['traffic_class'],
                'pred_traffic_obj_id': pred_slices['traffic_obj_id'],
                'obs_traffic_light_class':obs_slices['traffic_light_class'],
                'obs_traffic_sign_class':obs_slices['traffic_sign_class'],
                'obs_crosswalk_class':obs_slices['crosswalk_class'],
                'obs_ego_motion': obs_slices['ego_motion'],
                'pred_ego_motion': pred_slices['ego_motion'],
                }

        return ret
            
    def get_intent_tracks(self, dataset, data_type, observe_length, predict_length, overlap):
        down = self.downsample_step
        seq_length = observe_length * down + predict_length * down
        overlap_stride = observe_length * down if overlap == 0 else max(int((1 - overlap) * seq_length), 1)

        data_type = set(data_type)
        data_type.update(set(['image', 'pid', 'bbox', 'resolution', 
                              'intention_binary', 'crossing_seq','action_seq',
                              'traffic_bbox', 'traffic_class', 'traffic_obj_id',
                              'traffic_light_class', 'traffic_sign_class', 'crosswalk_class', 'ego_motion',
                             ]))

        d = {key: dataset[key].copy() for key in data_type}
        num_traffics = []
        all_track_lengths = [len(v) for v in d['pid']]
        for k in d.keys():
            tracks = []
            if k in ['traffic_bbox', 'traffic_class', 'traffic_obj_id']:
                for i, track in enumerate(d[k]):
                    num = len(track)
                    num_traffics.append(num)
                    trk_len = all_track_lengths[i]
                    if self.cfg.MODEL.TYPE == 'rnn' and self.split == 'test':
                        # NOTE: RNN models are tested online and the predict_length is removed
                        seq_length = trk_len
                        overlap_stride = trk_len
                    if num == 0:
                        for i in range(0, trk_len - seq_length + 1, overlap_stride):
                            tracks.append([])
                    else:
                        assert trk_len == track.shape[1]
                        for i in range(0, trk_len - seq_length + 1, overlap_stride):
                            tracks.append(track[:, i:i + seq_length:down])
                
            else: 
                for i, track in enumerate(d[k]):
                    trk_len = all_track_lengths[i]
                    if self.cfg.MODEL.TYPE == 'rnn' and self.split == 'test':
                        seq_length = trk_len 
                        overlap_stride = trk_len       
                    tracks.extend([track[i:i+seq_length:down] for i in\
                                range(0, trk_len - seq_length + 1, overlap_stride)])
            d[k] = tracks
        if 'traffic_bbox' in data_type:
            print("Max_num_traffics:{}, avg_num_traffics:{}".format(max(num_traffics), sum(num_traffics)/len(num_traffics)))
        return d

    def map_to_new_action(self, _int, _act, _cro, crossed, prev_act=None):
        if _int == 0 and _act == 0 and _cro == 1 and not crossed:
            new_act = 0 # standing - have not crossed and will not cross
        elif _int == 1 and _act == 0 and _cro == 1 and not crossed:
            new_act = 1 # waiting - have not crossed yet and will cross
        elif _int == 1 and _act == 1 and _cro == 1 and not crossed:
            new_act = 2 # going towards - have not crossed yet and will cross
        elif _cro in [0, 2]: #_act==1 and (_cro in [0, 2] or prev_act in [1, 3]):
            new_act = 3 # crossing - crossing right now or previous action is wating or crossing
        elif crossed and _cro == 1 and _act == 0:
            new_act = 4 # finished crossing and standing
        elif crossed and _cro == 1 and _act == 1:
            new_act = 5 # finished crossing and walking
        elif _act == 1:
            new_act = 6 # other walking
        else:
            pdb.set_trace()
        return new_act

    def generate_new_actions(self, beh_seq):
        # NOTE: Oct 8th, map the 2 actions to 7 actions
        new_action_seq = []
        for i, (bboxes, intent, action, cross) in enumerate(zip(beh_seq['bbox'],
                                             beh_seq['intention_binary'],
                                             beh_seq['action_seq'], 
                                             beh_seq['crossing_seq'])):
            intent = [tmp[0] for tmp in intent]
            crossed = False
            new_action_track = []
            
            for _int, _act, _cro, _box in zip(intent, action, cross, bboxes):
                _cro += 1
                if _cro in [0, 2]:
                    crossed = True
                # map to new action
                new_action = self.map_to_new_action(_int, _act, _cro, crossed)
                new_action_track.append(new_action)
            new_action_seq.append(new_action_track)
        return new_action_seq
        
    # Add downsample function
    def get_intent_tracks_new(self, data_raw, data_type, obs_length, pred_length, overlap, time_to_event, normalize=False):
        
        down = self.downsample_step
        seq_len = obs_length if self.split == 'test' else obs_length + pred_length 
        overlap_stride = obs_length * down if overlap == 0 else \
        max(int((1 - overlap) * obs_length), 1)

        data_type = set(data_type)
        data_type.update(set(['image', 'pid', 'bbox', 'resolution', 
                              'intention_binary', 'crossing_seq','action_seq',
                              'traffic_bbox', 'traffic_class', 'traffic_obj_id',
                              'traffic_light_class', 'traffic_sign_class', 'crosswalk_class', 'ego_motion',
                             ]))
        
        d = {key: data_raw[key].copy() for key in data_type}
        
        num_traffics = []
        all_track_lengths = [len(v) for v in d['pid']]

        for k in d.keys():
            tracks = []
            if k in ['traffic_bbox', 'traffic_class', 'traffic_obj_id']:
                # NOTE: number of traffic objects is different indifference sncenes so we 
                for i, track in enumerate(d[k]):
                    num = len(track)
                    num_traffics.append(num)
                    trk_len = all_track_lengths[i]
                    start_idx = trk_len - obs_length - time_to_event[1]
                    end_idx = trk_len - obs_length - time_to_event[0]
                    
                    if num == 0:
                        tracks.extend([[] for i in range(start_idx, end_idx + 1, overlap_stride)])
                    else:
                        assert trk_len == track.shape[1]
                        tracks.extend([track[:,i:i + seq_len] for i in range(start_idx, end_idx + 1, overlap_stride)])
            else: 
                for i, track in enumerate(d[k]):
                    trk_len = all_track_lengths[i]                    
                    start_idx = trk_len - obs_length - time_to_event[1]
                    end_idx = trk_len - obs_length - time_to_event[0]
                    tracks.extend([track[i:i + seq_len] for i in range(start_idx, end_idx + 1, overlap_stride)])
            d[k] = tracks
        if 'traffic_bbox' in data_type:
            print("Max_num_traffics:{}, avg_num_traffics:{}".format(max(num_traffics), sum(num_traffics)/len(num_traffics)))
        return d