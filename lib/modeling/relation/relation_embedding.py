'''
Nov 16th the relation embedding network.
The networks takes the target object and the traffic objects and 
'''
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modeling.poolers import Pooler
from lib.modeling.layers.attention import AdditiveAttention
import time
import pdb

class RelationEmbeddingNet(nn.Module):
    '''
    Embed the relation information for each time step.
    The model ignores temporal imformation to focus on relational information.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_box_embedding = nn.Sequential(nn.Linear(4, 32),
                                                  nn.ReLU())
        self.traffic_keys = self.cfg.MODEL.TRAFFIC_TYPES#['x_ego', 'x_neighbor', 'x_crosswalk', 'x_light', 'x_sign', 'x_station']
        if self.cfg.DATASET.NAME == 'PIE':
            self.traffic_embedding = nn.ModuleDict({
                                        'x_neighbor': nn.Sequential(nn.Linear(4, 32),
                                                                    nn.ReLU()),
                                        'x_light':nn.Sequential(nn.Linear(6, 32),
                                                                nn.ReLU()),
                                        'x_sign': nn.Sequential(nn.Linear(5, 32),
                                                                nn.ReLU()),
                                        'x_crosswalk': nn.Sequential(nn.Linear(7, 32),
                                                                    nn.ReLU()), 
                                        'x_station': nn.Sequential(nn.Linear(7, 32),
                                                                nn.ReLU()),
                                        'x_ego': nn.Sequential(nn.Linear(4, 32),
                                                            nn.ReLU())
                                                                        })
        elif cfg.DATASET.NAME == 'JAAD':
            self.traffic_embedding = nn.ModuleDict({
                                        'x_neighbor': nn.Sequential(nn.Linear(4, 32),
                                                                    nn.ReLU()),
                                        'x_light':nn.Sequential(nn.Linear(1, 32),
                                                                nn.ReLU()),
                                        'x_sign': nn.Sequential(nn.Linear(2, 32),
                                                                nn.ReLU()),
                                        'x_crosswalk': nn.Sequential(nn.Linear(1, 32),
                                                                    nn.ReLU()), 
                                        'x_ego': nn.Sequential(nn.Linear(1, 32),
                                                            nn.ReLU())
                                                                        })    
        if 'relation' in self.cfg.MODEL.TASK:
            self.classifier =  nn.Sequential(nn.Linear(32 * (len(self.traffic_keys)+1), 32),
                                            nn.Dropout(0.1),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.Dropout(0.1),
                                            nn.ReLU(),
                                            nn.Linear(32, 1),)
        if self.cfg.MODEL.TRAFFIC_ATTENTION != 'none':
            # NOTE: NOV 24 add attention to objects.
            self.attention = AdditiveAttention(32, 128)
    
    def embed_traffic_features(self, x_ped, x_traffics):
        '''
        run the fully connected embedding networks on all inputs 
        '''
        self.x_traffics = x_traffics
        self.x_ped = self.target_box_embedding(x_ped)

        
        # embed neighbor objects
        self.num_traffics = {}
        self.num_traffics = {k:[len(v) if isinstance(traffic, list) else 1 for v in traffic ] for k, traffic in self.x_traffics.items()}
        self.x_traffics['cls_ego'] = torch.ones(x_ped.shape[0], self.x_ped.shape[1])
        self.other_traffic = self.x_traffics

        # embed other traffics
        for k in self.traffic_keys:
            # traffic = self.other_traffic[k]
            traffic = self.x_traffics[k]
            if isinstance(traffic, list):
                traffic = torch.cat(traffic, dim=0).to(x_ped.device)
                if len(traffic) > 0:
                    self.x_traffics[k] = self.traffic_embedding[k](traffic)
                else:
                    self.x_traffics[k] = []
            elif isinstance(traffic, torch.Tensor):
                # ego motion is a tensor not a list.
                self.x_traffics[k] = self.traffic_embedding[k](traffic.to(x_ped.device))
            else:
                raise TypeError("traffic type unknown: "+type(traffic))
    
    def concat_traffic_features(self):
        # simply sum in each batch and concate different features 
        batch_size, T = self.x_ped.shape[0:2]
        all_traffic_features = []
        pdb.set_trace()
        for k in self.traffic_keys:
            traffic_cls = 'cls_'+k.split('_')[-1]
            if isinstance(self.other_traffic[traffic_cls], torch.Tensor):#k == 'x_ego':
                # NOTE: if traffic_cls is tensor format, it means the object has only 1 instance for all frames.
                # thus we don't need to mask or attend it
                all_traffic_features.append(self.x_traffics[k])
                continue
            
            num_objects = sum(self.num_traffics[k])
            if num_objects <= 0:
                # no such objects, skip
                all_traffic_features.append(torch.zeros(batch_size, self.x_ped.shape[-1]).to(self.x_ped.device))
                continue

            # 1. formulate the mapping matrix (B x num_objects matrix with 0 and 1) for in-batch sum 
            batch_traffic_id_map = torch.zeros(batch_size, num_objects).to(self.x_ped.device)
            indices = torch.repeat_interleave(torch.tensor(range(batch_size)), torch.tensor(self.num_traffics[k])).to(self.x_ped.device)
            batch_traffic_id_map[indices, range(num_objects)] = 1

            # 2. objects with class=-1 does not exist, so set feature to 0
            masks = (torch.cat(self.other_traffic[traffic_cls], dim=0)!=-1).to(self.x_ped.device)
            traffic_feature = self.x_traffics[k] * masks.unsqueeze(-1)

            # 3. do in-batch sum using matrix multiplication.
            traffic_feature = torch.matmul(batch_traffic_id_map, traffic_feature.view(num_objects, -1))
            traffic_feature = traffic_feature.view(batch_size, T, -1)
            all_traffic_features.append(traffic_feature)

        all_traffic_features = torch.cat([self.x_ped] + all_traffic_features, dim=-1)
        return all_traffic_features

    def attended_traffic_features(self, h_ped, t):
        all_traffic_features = []
        all_traffic_attentions = {}
        batch_size = h_ped.shape[0]
        
        #################### use separate attention for each object type #########################
        for k in self.traffic_keys:
            traffic_cls = 'cls_'+k.split('_')[-1]
            if isinstance(self.other_traffic[traffic_cls], torch.Tensor):#k == 'x_ego':
                # NOTE: if traffic_cls is tensor format, it means the object has only 1 instance for all frames.
                # thus we don't need to mask or attend it
                all_traffic_features.append(self.x_traffics[k][:, t])
                continue
            
            # 1. update the number of object for time t, based on the class label != -1
            self.num_traffics[k] = [len(torch.nonzero(v[:, t] != -1)) if len(v) > 0 else 0 for v in self.other_traffic[traffic_cls]]
            num_objects = sum(self.num_traffics[k])
            if num_objects <= 0:
                # no such objects, skip
                all_traffic_features.append(torch.zeros(batch_size, self.x_ped.shape[-1]).to(self.x_ped.device))
                continue
            masks = (torch.cat(self.other_traffic[traffic_cls], dim=0)!=-1).to(self.x_ped.device)
            masks = masks[:, t] if len(masks) > 0 else masks
            traffic_feature = self.x_traffics[k][masks][:, t]

            # 2. get attention score (logits) vector
            h_ped_tiled = torch.repeat_interleave(h_ped, torch.tensor(self.num_traffics[k]).to(h_ped.device), dim=0)
            if len(h_ped_tiled) > 0:
                # NOTE: if len(h_ped_tiled) == 0, there is no traffic in any batch.
                score_vec = self.attention.get_score_vec(self.x_traffics[k][masks][:, t:t+1], h_ped_tiled)
            
            # 3. create the attended batch_traffic_id_map
            batch_traffic_id_map = torch.zeros(batch_size, num_objects).to(self.x_ped.device)
            indices = torch.repeat_interleave(torch.tensor(range(batch_size)), torch.tensor(self.num_traffics[k])).to(self.x_ped.device)
            batch_traffic_id_map[indices, range(num_objects)] = 1
            if self.cfg.MODEL.TRAFFIC_ATTENTION == 'softmax':
                # NOTE: self-implemented softmax with selected slices along a dim
                attention_probs = torch.exp(score_vec) / torch.repeat_interleave(torch.matmul(batch_traffic_id_map, 
                                                                                              torch.exp(score_vec)), 
                                                                                 torch.tensor(self.num_traffics[k]).to(h_ped.device), dim=0)

            elif self.cfg.MODEL.TRAFFIC_ATTENTION == 'sigmoid':
                attention_probs = torch.sigmoid(score_vec)
            else:
                raise NameError(self.cfg.MODEL.TRAFFIC_ATTENTION)
            all_traffic_attentions[k] = attention_probs

            traffic_feature *= attention_probs
            traffic_feature = torch.matmul(batch_traffic_id_map, traffic_feature)
            all_traffic_features.append(traffic_feature)
            
        # We use defined order of concatenation.
        all_traffic_features = torch.cat([self.x_ped[:, t ]] + all_traffic_features, dim=-1)
        return all_traffic_features, all_traffic_attentions

    def forward(self, x_ped, x_traffics, h_ped=None, t=None):#x_neighbor, cls_neighbor, **other_traffic
        '''
        Run FC on each neighbor features, the sum and concatenate
        '''
        self.embed_traffic_features(x_ped, x_traffics)
        
        if self.cfg.MODEL.TRAFFIC_ATTENTION != 'none':
            all_traffic_features, all_traffic_attentions = self.attended_traffic_features(h_ped, t)
        else:
            all_traffic_features = self.concat_traffic_features(x_ped)
            all_traffic_attentions = {}
        
        
        if 'relation' in self.cfg.MODEL.TASK:
            int_det_score = self.classifier(all_traffic_features)
        else:
            int_det_score = None
        return int_det_score, all_traffic_features, all_traffic_attentions
        