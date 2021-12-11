'''
The action net take stack of observed image features
and detect the observed actions (and predict the futrue actions)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modeling.poolers import Pooler
from lib.modeling.layers.convlstm import ConvLSTMCell

import pdb

class ActionIntentNet(nn.Module):
    def __init__(self, cfg, x_visual_extractor=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = self.cfg.MODEL.PRED_LEN
        # The encoder RNN to encode observed image features
        # NOTE: there are two ways to encode the feature
        self.enc_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)
        self.recurrent_drop = nn.Dropout(self.cfg.MODEL.RECURRENT_DROPOUT)
        if 'convlstm' in self.cfg.MODEL.INTENT_NET:
            # a. use ConvLSTM then ,ax/avg pool or flatten the hidden feature.
            self.enc_cell = ConvLSTMCell((7, 7), 
                                         512, self.cfg.MODEL.CONVLSTM_HIDDEN, #self.hidden_size, 
                                         kernel_size=(2,2),
                                         input_dropout=0.4,
                                         recurrent_dropout=0.2,
                                         attended=self.cfg.MODEL.INPUT_LAYER=='attention')
            enc_input_size = 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN + self.hidden_size if 'trn' in self.cfg.MODEL.INTENT_NET else 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN
            self.enc_fused_cell = nn.GRUCell(enc_input_size, self.hidden_size)                             
        elif 'gru' in self.cfg.MODEL.INTENT_NET:
            if self.cfg.MODEL.INPUT_LAYER == 'conv2d':
                enc_input_size = 6*6*64 + 16 + self.hidden_size if 'trn' in self.cfg.MODEL.INTENT_NET else 6*6*64 + 16
            else:
                enc_input_size = 128 + 16 + self.hidden_size if 'trn' in self.cfg.MODEL.INTENT_NET else 128 + 16 
            # use max/avg pooling to get 1d vector then use regular GRU
            # NOTE: use max pooling on pre-extracted feature can be problematic since some features will be lost constantly.
            if x_visual_extractor is not None:
                # use an initialized feature extractor 
                self.x_visual_extractor = x_visual_extractor
            elif self.cfg.MODEL.INPUT_LAYER == 'avg_pool':
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
            else:
                raise NameError(self.cfg.MODEL.INPUT_LAYER)
            # NOTE: add ego motion encoder
            if self.cfg.MODEL.WITH_EGO:
                self.ego_enc_cell = nn.GRUCell(4, 32)
                enc_input_size += 32
            if self.cfg.MODEL.WITH_TRAFFIC:
                enc_input_size += 32 * (len(self.cfg.MODEL.TRAFFIC_TYPES)+1) # NOTE: Nov 18, traffic feature dim is 224 
            self.enc_cell = nn.GRUCell(enc_input_size, self.hidden_size)
        else:
            raise NameError(self.cfg.MODEL.INTENT_NET)

        # The decoder RNN to predict future actions
        self.dec_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)
        self.dec_input_linear = nn.Sequential(nn.Linear(self.cfg.DATASET.NUM_ACTION, self.hidden_size),
                                              nn.ReLU())
        self.future_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU())
        self.dec_cell = nn.GRUCell(self.hidden_size, self.hidden_size)

        # The classifier layer
        self.action_classifier = nn.Linear(self.hidden_size, self.cfg.DATASET.NUM_ACTION)
        self.intent_classifier = nn.Linear(self.hidden_size, 1) #self.cfg.DATASET.NUM_INTENT

    def enc_step(self, x_visual, enc_hx, x_bbox=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None):
        '''
        Run one step of the encoder
        x_visual: visual feature as the encoder inputs
        x_bbox: bounding boxes as the encoder inputs
        x_traffic: traffic features as the encoder inputs
        future_inputs: encoder inputs from the decoder end (TRN)
        '''
        batch_size = x_visual.shape[0]
        if 'convlstm' in self.cfg.MODEL.INTENT_NET:
            h_fused = enc_hx[2]
            # run ConvLSTM
            h, c = self.enc_cell(x_visual, enc_hx[:2], future_inputs)
            # get input for GRU
            fusion_input = h.view(batch_size, -1)
            if future_inputs is not None:
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            fusion_input = torch.cat([fusion_input, x_bbox], dim=-1)
            if self.cfg.MODEL.WITH_EGO:
                enc_h_ego = self.ego_enc_cell(x_ego, enc_h_ego)
                fusion_input = torch.cat((fusion_input, enc_h_ego))
            if self.cfg.MODEL.WITH_TRAFFIC:
                fusion_input = torch.cat((fusion_input, x_traffic), dim=-1)
            # run GRU 
            h_fused = self.enc_fused_cell(self.enc_drop(fusion_input), 
                                          self.recurrent_drop(h_fused))
            enc_hx = [h, c, h_fused]
            enc_act_score = self.action_classifier(self.enc_drop(h_fused))
            enc_int_score = self.intent_classifier(self.enc_drop(h_fused))
        elif 'gru' in self.cfg.MODEL.INTENT_NET:
            # avg pool visual feature and concat with bbox input
            if self.cfg.MODEL.INPUT_LAYER == 'attention':
                if 'trn' in self.cfg.MODEL.INTENT_NET:
                    x_visual = self.x_visual_extractor(x_visual, future_inputs)

                else:
                    x_visual = self.x_visual_extractor(x_visual, enc_hx)
            else:
                x_visual = self.x_visual_extractor(x_visual)
            fusion_input = torch.cat((x_visual, x_bbox), dim=1)
            if self.cfg.MODEL.WITH_EGO:
                enc_h_ego = self.ego_enc_cell(x_ego, enc_h_ego)
                fusion_input = torch.cat((fusion_input, enc_h_ego), dim=-1)
            if self.cfg.MODEL.WITH_TRAFFIC:
                fusion_input = torch.cat((fusion_input, x_traffic), dim=-1)
            if future_inputs is not None:
                # add input collected from action decoder
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            enc_hx = self.enc_cell(self.enc_drop(fusion_input), 
                                   self.recurrent_drop(enc_hx))
            enc_act_score = self.action_classifier(self.enc_drop(enc_hx))
            enc_int_score = self.intent_classifier(self.enc_drop(enc_hx))
        else:
            raise NameError(self.cfg.MODEL.INTENT_NET)
            
        return enc_hx, enc_act_score, enc_int_score, enc_h_ego
      
    def decoder(self, enc_hx, dec_inputs=None):
        '''
        Run decoder for pred_len step to predict future actions
            enc_hx: last hidden state of encoder
            dec_inputs: decoder inputs
        '''
        dec_hx = enc_hx[-1] if isinstance(enc_hx, list) else enc_hx
        dec_scores = []
        future_inputs = dec_hx.new_zeros(dec_hx.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.INTENT_NET else None
        for t in range(self.pred_len):
            dec_hx = self.dec_cell(self.dec_drop(dec_inputs), 
                                   self.recurrent_drop(dec_hx))
            dec_score = self.action_classifier(self.dec_drop(dec_hx))
            dec_scores.append(dec_score)
            dec_inputs = self.dec_input_linear(dec_score)        
            future_inputs = future_inputs + self.future_linear(dec_hx) if future_inputs is not None else None
        future_inputs = future_inputs / self.pred_len if future_inputs is not None else None
        return torch.stack(dec_scores, dim=1), future_inputs

    def forward(self, x_visual, x_bbox=None, x_ego=None, x_traffic=None, dec_inputs=None):
        '''
        For training only!
        Params:
            x_visual: visual feature as the encoder inputs (batch, SEG_LEN, 512, 7, 7)
            x_bbox: bounding boxes as the encoder inputs (batch, SEG_LEN, 4), optional
            x_ego: ego car motion as extra encoder inputs, (batch, SEG_LEN, 4), optional
            x_traffic: pre-extracted traffic feature, (batch, SEG_LEN, TRAFFIC_FEATURE_DIM), optional
            dec_inputs: other inputs to the decoder, (batch, SEG_LEN, PRED_LEN, ?), optional
        Returns:
            all_enc_scores: (batch, SEG_LEN, num_classes)
            all_dec_scores: (batch, SEG_LEN, PRED_LEN, num_classes)
        '''
        future_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.INTENT_NET else None
        enc_hx = x_visual.new_zeros(x_visual.shape[0], self.hidden_size)
        enc_h_ego = x_visual.new_zeros(x_visual.shape[0], 32) if self.cfg.MODEL.WITH_EGO else None
        all_enc_act_scores, all_enc_int_scores, all_dec_act_scores = [], [], []
        for t in range(self.cfg.MODEL.SEG_LEN):
            # Run one step of action detector/predictor
            x_ego_input = x_ego[:, t] if x_ego is not None else None
            x_traffic_input = x_traffic[:, t] if x_traffic is not None else None
            ret = self.step(x_visual[:, t], enc_hx,
                            x_bbox=x_bbox[:, t], 
                            x_ego=x_ego_input, 
                            x_traffic=x_traffic_input,
                            enc_h_ego=enc_h_ego, 
                            future_inputs=future_inputs, 
                            dec_inputs=dec_inputs)
            enc_act_scores, enc_int_scores, enc_hx, dec_act_scores, future_inputs, enc_h_ego = ret
            all_enc_act_scores.append(enc_act_scores)
            all_enc_int_scores.append(enc_int_scores)
            if dec_act_scores is not None:
                all_dec_act_scores.append(dec_act_scores)
            
        all_enc_act_scores = torch.stack(all_enc_act_scores, dim=1)
        all_enc_int_scores = torch.stack(all_enc_int_scores, dim=1)
        all_dec_act_scores = torch.stack(all_dec_act_scores, dim=1)
        return all_enc_act_scores, all_enc_int_scores, all_dec_act_scores
    
    def step(self, x_visual, enc_hx, x_bbox=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None, dec_inputs=None):
        '''
        Directly call step when run inferencing.
        x_visual: (batch, 512, 7, 7)
        enc_hx: (batch, hidden_size)
        '''
        # 1. encoder
        enc_hx, enc_act_scores, enc_int_scores, enc_h_ego = self.enc_step(x_visual, enc_hx, 
                                                                            x_bbox=x_bbox, 
                                                                            x_ego=x_ego, 
                                                                            x_traffic=x_traffic,
                                                                            enc_h_ego=enc_h_ego, 
                                                                            future_inputs=future_inputs)

        # 2. decoder
        dec_scores = None
        if 'trn' in self.cfg.MODEL.INTENT_NET:
            if dec_inputs is None:
                dec_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size)
            dec_scores, future_inputs = self.decoder(enc_hx, dec_inputs=dec_inputs)
            
        return enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego, 

