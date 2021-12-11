'''
main function of our action-intention detection model
Action head
Intention head
'''
import torch
import torch.nn as nn
from .action_net import ActionNet
from .intent_net import IntentNet
from .action_detectors import make_model
# from .poolers import Pooler
import pdb

class ActionIntentionDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # if cfg.MODEL.TASK == 'intent_action':
        # we only use the top layers of the the base model 
        self.base_model = make_model(cfg.MODEL.INTENT_NET, num_classes=2, pretrained=cfg.MODEL.PRETRAINED)
        if 'action' in cfg.MODEL.TASK:
            self.action_model = ActionNet(cfg)
        if 'intent' in cfg.MODEL.TASK:
            self.intent_model = IntentNet(cfg)
        # else:
        #     raise NameError("Unknown model task", cfg.MODEL.TASK)

        # self.pooler = Pooler(output_size=(self.cfg.ROI_SIZE, self.cfg.ROI_SIZE),
        #                     scales=self.cfg.POOLER_SCALES,
        #                     sampling_ratio=self.cfg.POOLER_SAMPLING_RATIO,
        #                     canonical_level=1)

    def forward(self, x, bboxes, masks):
        '''
        x: input feature of the pedestrian
        bboxes: the local bbox of the pedestrian in the patch
        masks: the binary mask of the pedestrian box in the patch
        '''
        action_logits = None
        roi_features = None
        intent_logits = None
        x = self.base_model(x, extract_features=True)
        
        # if self.cfg.MODEL.TASK == 'action_intent':
        #     self.base_model(x)
        if 'action' in self.cfg.MODEL.TASK:
            # 1. get action detection
            action_logits, roi_features = self.action_model(x, bboxes, masks)
        if 'intent' in self.cfg.MODEL.TASK:
            # 2. get intent detection
            intent_logits = self.intent_model(x, action_logits, roi_features)

        return action_logits, intent_logits

