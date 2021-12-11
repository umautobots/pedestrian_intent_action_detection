'''
main function of our action-intention detection model
Action head
Intention head
'''
import torch
import torch.nn as nn
from .action_net import ActionNet
from .intent_net import IntentNet
from .action_intent_net import ActionIntentNet
from lib.modeling.layers.attention import AdditiveAttention2D
from lib.modeling.relation import RelationNet


class ActionIntentionDetection(nn.Module):
    def __init__(self, cfg, parameter_scheduler=None):
        super().__init__()
        self.cfg = cfg
        self.parameter_scheduler = parameter_scheduler
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        
        self.bbox_embedding = nn.Sequential(nn.Linear(4, 16),
                                            nn.ReLU())
        if self.cfg.MODEL.WITH_TRAFFIC:
            self.relation_model = RelationNet(cfg)

        self.x_visual_extractor = None
        if 'action' in self.cfg.MODEL.TASK and 'intent' in self.cfg.MODEL.TASK and 'single' in self.cfg.MODEL.TASK:
            if 'convlstm' not in self.cfg.MODEL.INTENT_NET:
                self._init_visual_extractor()
            self.action_intent_model = ActionIntentNet(cfg, x_visual_extractor=self.x_visual_extractor)
        else:
            if 'action' in self.cfg.MODEL.TASK:
                if 'convlstm' not in self.cfg.MODEL.ACTION_NET:
                    self._init_visual_extractor()
                self.action_model = ActionNet(cfg, x_visual_extractor=self.x_visual_extractor)
            if 'intent' in self.cfg.MODEL.TASK:
                if 'convlstm' not in self.cfg.MODEL.INTENT_NET and self.x_visual_extractor is None:
                    self._init_visual_extractor()
                self.intent_model = IntentNet(cfg, x_visual_extractor=self.x_visual_extractor)
        

    def _init_visual_extractor(self):
        if self.cfg.MODEL.INPUT_LAYER == 'avg_pool':
            self.x_visual_extractor = nn.Sequential(nn.Dropout2d(0.4),
                                                    nn.AvgPool2d(kernel_size=[7,7], stride=(1,1)),
                                                    nn.Flatten(start_dim=1, end_dim=-1),
                                                    nn.Linear(512, 128),
                                                    nn.ReLU())
        elif self.cfg.MODEL.INPUT_LAYER == 'conv2d':
            self.x_visual_extractor = nn.Sequential(nn.Dropout2d(0.4),
                                                    nn.Conv2d(in_channels=512, out_channels=64, kernel_size=[2,2]),
                                                    nn.Flatten(start_dim=1, end_dim=-1),
                                                    nn.ReLU())
        elif self.cfg.MODEL.INPUT_LAYER == 'attention':
            self.x_visual_extractor = AdditiveAttention2D(self.cfg)
        else:
            raise NameError(self.cfg.MODEL.INPUT_LAYER)

    def _init_hidden_states(self, x, net_type='gru', task_exists=True):
        batch_size = x.shape[0]
        if not task_exists:
            return None
        elif 'convlstm' in net_type:
            return [x.new_zeros(batch_size, self.cfg.MODEL.CONVLSTM_HIDDEN, 6, 6),
                      x.new_zeros(batch_size, self.cfg.MODEL.CONVLSTM_HIDDEN, 6, 6),
                      x.new_zeros(batch_size, self.hidden_size)]
        elif 'gru' in net_type:
            return x.new_zeros(batch_size, self.hidden_size)
        else:
            raise ValueError(net_type)

    def forward(self, 
                x_visual, 
                x_bbox=None, 
                x_ego=None, 
                x_traffic=None,
                dec_inputs=None, 
                local_bboxes=None, 
                masks=None):
        
        if 'action' in self.cfg.MODEL.TASK and 'intent' in self.cfg.MODEL.TASK and 'single' in self.cfg.MODEL.TASK:
            return self.forward_single_stream(x_visual, 
                                              x_bbox=x_bbox, 
                                              x_ego=x_ego, 
                                              x_traffic=x_traffic, 
                                              dec_inputs=dec_inputs)
        else:
            return self.forward_two_stream(x_visual, 
                                           x_bbox=x_bbox, 
                                           x_ego=x_ego, 
                                           dec_inputs=dec_inputs, 
                                           local_bboxes=local_bboxes, 
                                           masks=masks)

    def embed_traffic_features(self, x_bbox, x_ego, x_traffic):
        x_traffic['x_ego'] = x_ego
        if self.cfg.DATASET.NAME == 'PIE':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        elif self.cfg.DATASET.NAME == 'JAAD':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        else:
            raise NameError(self.cfg.DATASET.NAME)
    def concat_traffic_features(self):
        return self.relation_model.concat_traffic_features()
    
    def attended_traffic_features(self, h_ped, t):
        return self.relation_model.attended_traffic_features(h_ped, t)

    def forward_single_stream(self, x_visual, x_bbox=None, x_ego=None, x_traffic=None, dec_inputs=None):
        '''
        NOTE: Action and Intent net share the same encoder network, but different classifiers
        '''
        seg_len = x_visual.shape[1]
        # initialize inputs and hidden states for encoders
        future_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.INTENT_NET else None
        enc_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.ACTION_NET,  task_exists=True)
        enc_h_ego = x_visual.new_zeros(x_visual.shape[0], 32) if self.cfg.MODEL.WITH_EGO else None
        action_detection_scores, intent_detection_scores, action_prediction_scores = [], [], []
        all_attentions = []
        
        if self.cfg.MODEL.WITH_TRAFFIC and not self.cfg.MODEL.PRETRAINED:
            self.embed_traffic_features(x_bbox, x_ego, x_traffic)
            if self.cfg.MODEL.TRAFFIC_ATTENTION == 'none':
                # NOTE: get all traffic features in advance if do not use attention, otherwise get traffic feature iteratively.
                x_traffic = self.concat_traffic_features()
        for t in range(seg_len):
            # Run one step of action detector/predictor
            x_ego_input = x_ego[:, t] if x_ego is not None else None
            x_traffic_input, traffic_attentions = None, None
            if isinstance(x_traffic, torch.Tensor):
                x_traffic_input = x_traffic[:, t]
            elif isinstance(x_traffic, dict):
                # end = time.time()
                x_traffic_input, traffic_attentions = self.attended_traffic_features(enc_hx, t)

            ret = self.step_one_stream(x_visual[:, t], 
                                       enc_hx, 
                                       x_bbox=x_bbox[:, t],
                                       x_ego=x_ego_input,
                                       x_traffic=x_traffic_input,
                                       enc_h_ego=enc_h_ego,
                                       future_inputs=future_inputs, 
                                       dec_inputs=dec_inputs)
            enc_act_scores, enc_int_scores, enc_hx, dec_act_scores, future_inputs, enc_h_ego = ret
            action_detection_scores.append(enc_act_scores)
            intent_detection_scores.append(enc_int_scores)
            all_attentions.append(traffic_attentions)
            
            if dec_act_scores is not None:
                action_prediction_scores.append(dec_act_scores)
            
        action_detection_scores = torch.stack(action_detection_scores, dim=1) if action_detection_scores else None
        action_prediction_scores = torch.stack(action_prediction_scores, dim=1) if action_prediction_scores else None
        intent_detection_scores = torch.stack(intent_detection_scores, dim=1) if intent_detection_scores else None
        return action_detection_scores, action_prediction_scores, intent_detection_scores, all_attentions
        
    def step_one_stream(self, x_visual, enc_hx, x_bbox=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None, dec_inputs=None):
        if x_bbox is not None:
            x_bbox = self.bbox_embedding(x_bbox)
        ret = self.action_intent_model.step(x_visual, 
                                            enc_hx, 
                                            x_bbox=x_bbox, 
                                            x_ego=x_ego,
                                            x_traffic=x_traffic,
                                            enc_h_ego=enc_h_ego,
                                            future_inputs=future_inputs, 
                                            dec_inputs=dec_inputs)
        enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego = ret
        return enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego
    
    def forward_two_stream(self, x_visual, x_bbox=None, x_ego=None, dec_inputs=None, local_bboxes=None, masks=None):
        '''
        NOTE: Action and Intent net use separate encoder networks
        for training only !
        x_visual: extracted features, (batch_size, SEG_LEN, 512, 7, 7)
        x_bbox: bounding boxes(batch_size, SEG_LEN, 4)
        '''
        seg_len = x_visual.shape[1]
        act_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.ACTION_NET,  task_exists='action' in self.cfg.MODEL.TASK)
        int_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.INTENT_NET,  task_exists='intent' in self.cfg.MODEL.TASK)
        future_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.ACTION_NET else None
        
        action_detection_scores = []
        action_prediction_scores = []
        intent_detection_scores = []
        for t in range(seg_len):
            dec_input = dec_inputs[:, t] if dec_inputs else None
            ret = self.step_two_stream(x_visual[:, t], act_hx, int_hx, x_bbox[:, t], future_inputs, dec_input)
            enc_act_score, dec_act_score, enc_int_score, act_hx, int_hx, future_inputs = ret
            if enc_act_score is not None:
                action_detection_scores.append(enc_act_score)
            if dec_act_score is not None:
                action_prediction_scores.append(dec_act_score)
            if enc_int_score is not None:
                intent_detection_scores.append(enc_int_score)
        action_detection_scores = torch.stack(action_detection_scores, dim=1) if action_detection_scores else None
        action_prediction_scores = torch.stack(action_prediction_scores, dim=1) if action_prediction_scores else None
        intent_detection_scores = torch.stack(intent_detection_scores, dim=1) if intent_detection_scores else None
        return action_detection_scores, action_prediction_scores, intent_detection_scores, None
    
    def step_two_stream(self, x_visual, act_hx, int_hx, x_bbox=None, future_inputs=None, dec_inputs=None):
        '''
        Directly call step when run inferencing.
        Params:
            x_visual:
            act_hx:
            int_hx:
            x_bbox:
            future_inputs:
        Return
        '''
        if x_bbox is not None:
            x_bbox = self.bbox_embedding(x_bbox)
        enc_act_score, dec_act_score, enc_int_score  = None, None, None
        if 'action' in self.cfg.MODEL.TASK:
            ret = self.action_model.step(x_visual, act_hx, x_bbox=x_bbox, dec_inputs=dec_inputs, future_inputs=future_inputs)
            enc_act_score, act_hx, dec_act_score, future_inputs = ret
            
        if 'intent' in self.cfg.MODEL.TASK:
            if 'trn' in self.cfg.MODEL.ACTION_NET:
                # TRN model use future action feature as input to intent net
                int_hx, enc_int_score = self.intent_model.step(x_visual, int_hx, x_bbox=x_bbox, future_inputs=future_inputs)
            else:
                # otherwise use action encoder 
                int_hx, enc_int_score = self.intent_model.step(x_visual, int_hx, x_bbox=x_bbox, future_inputs=act_hx)

        return enc_act_score, dec_act_score, enc_int_score, act_hx, int_hx, future_inputs