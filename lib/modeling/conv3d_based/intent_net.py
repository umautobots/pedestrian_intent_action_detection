'''
we need to make it generalize to any 3D Conv network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .action_detectors import make_model
import pdb
class IntentNet(nn.Module):
    def __init__(self, cfg, base_model=None):
        super().__init__()
        # if base_model is None:
        #     network_name = cfg.MODEL.INTENT_NET
        #     self.base_model = make_model(network_name, num_classes=cfg.DATASET.NUM_INTENT, pretrained=cfg.MODEL.PRETRAINED)
        # else:
        #     self.base_model = base_model
        self.cfg = cfg
        self.classifier = nn.Linear(1024,  cfg.DATASET.NUM_INTENT)                         
        self.merge_classifier = nn.Linear(1024 + 1024, cfg.DATASET.NUM_INTENT)
        # self.merge_classifier = nn.Sequential(
        #                                     nn.Linear(cfg.DATASET.NUM_ACTION + cfg.DATASET.NUM_INTENT, 256),
        #                                     nn.Dropout(0.5),
        #                                     nn.ReLU(),
        #                                     nn.Linear(256, cfg.DATASET.NUM_INTENT)
        #                                     )
    def forward(self, x, action_logits=None, roi_features=None):
        '''
        take input image patches and classify to intention
        Params:
            x: (Batch, channel, T, H, W)
            action: (Batch, num_actions)
        Return:
            intent: intention classification logits (Batch, num_intents)
        '''
        # intent = self.base_model(x)
        # pdb.set_trace()
        x = F.dropout(F.avg_pool3d(x, kernel_size=(2,7,7), stride=(1,1,1)), p=0.5, training=self.training)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        # if action is not None:
            # intent = self.merge_classifier(torch.cat([intent_logits, action_logits], dim=-1))
        if roi_features is not None:
            intent = self.merge_classifier(torch.cat([x, roi_features], dim=-1))
        else:
            intent = self.classifier(x)
        return intent