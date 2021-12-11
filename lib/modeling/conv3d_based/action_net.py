'''
we need to make it generalize to any 3D Conv network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .action_detectors import make_model
from lib.modeling.poolers import Pooler

import pdb
class ActionNet(nn.Module):
    def __init__(self, cfg, base_model=None):
        '''
        base_model: the base model for action net, a new base model is created if based_model is None
        '''
        super().__init__()
        # if base_model is None:
        #     network_name = cfg.MODEL.ACTION_NET
        #     self.base_model = make_model(network_name, num_classes=cfg.DATASET.NUM_ACTION, pretrained=cfg.MODEL.PRETRAINED)
        # else:
        #     self.base_model = base_model
        self.cfg = cfg
        self.classifier = nn.Linear(1024, cfg.DATASET.NUM_INTENT)
        self.pooler = Pooler(output_size=(self.cfg.MODEL.ROI_SIZE, self.cfg.MODEL.ROI_SIZE),
                            scales=self.cfg.MODEL.POOLER_SCALES,
                            sampling_ratio=self.cfg.MODEL.POOLER_SAMPLING_RATIO,
                            canonical_level=1)
    def forward(self, x, bboxes, masks):
        '''
        take input image patches and classify to action
        Params:
            x: (Batch, channel, T, H, W)
        Return:
            action: action classifictaion logits, (Batch, num_actions)
        '''
        
        # 1. apply mask to the input to get pedestrian patch
        if self.cfg.MODEL.ACTION_NET_INPUT == 'masked':
            roi_features = x * masks.unsqueeze(1)
        elif self.cfg.MODEL.ACTION_NET_INPUT == 'pooled':
            B, C, T, W, H = x.shape
            seq_len = bboxes.shape[1]
            starts = torch.arange(0, seq_len+1, int(seq_len/T))[:-1]
            ends = torch.arange(0, seq_len+1, int(seq_len/T))[1:]
            merged_bboxes = []
            for s, e in zip(starts, ends):
                merged_bboxes.append((bboxes[:, s:e].type(torch.float)).mean(dim=1))
            merged_bboxes = torch.stack(merged_bboxes, dim=1)#.type(torch.long)
            
            x = x.permute(0,2,1,3,4).reshape(B*T, C, W, H) # BxCxTxWxH -> (B*T)xCxWxH 
            merged_bboxes = merged_bboxes.reshape(-1, 1, 4)
            roi_features = self.pooler(x, merged_bboxes)
            roi_features = roi_features.reshape(B, T, C, W, H).permute(0,2,1,3,4)
            
        else:
            raise NameError()
        
        # 2. run action classification
        roi_features = F.dropout(F.avg_pool3d(roi_features, kernel_size=(2,7,7), stride=(1,1,1)), p=0.5, training=self.training)
        roi_features = roi_features.squeeze(-1).squeeze(-1).squeeze(-1)
        action_logits = self.classifier(roi_features)

        return action_logits, roi_features
    
    # def apply_mask(self, x):
    #     '''
    #     create mask from box and apply to input x
    #     '''
    #     pdb.set_trace()
        


