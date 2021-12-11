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

class ActionNet(nn.Module):
    def __init__(self, cfg, x_visual_extractor=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = self.cfg.MODEL.PRED_LEN
        self.num_classes = self.cfg.DATASET.NUM_ACTION
        if self.num_classes == 2 and self.cfg.MODEL.ACTION_LOSS=='bce':
            self.num_classes = 1
        # The encoder RNN to encode observed image features
        # NOTE: there are two ways to encode the feature
        self.enc_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)
        self.recurrent_drop = nn.Dropout(self.cfg.MODEL.RECURRENT_DROPOUT)
        if 'convlstm' in self.cfg.MODEL.ACTION_NET:
            # a. use ConvLSTM then ,ax/avg pool or flatten the hidden feature.
            self.enc_cell = ConvLSTMCell((7, 7), 
                                         512, self.cfg.MODEL.CONVLSTM_HIDDEN, #self.hidden_size, 
                                         kernel_size=(2,2),
                                         input_dropout=0.4,
                                         recurrent_dropout=0.2,
                                         attended=self.cfg.MODEL.INPUT_LAYER=='attention')
            enc_input_size = 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN + self.hidden_size if 'trn' in self.cfg.MODEL.ACTION_NET else 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN
            self.enc_fused_cell = nn.GRUCell(enc_input_size, self.hidden_size)                             
        elif 'gru' in self.cfg.MODEL.ACTION_NET:
            if self.cfg.MODEL.INPUT_LAYER == 'conv2d':
                enc_input_size = 6*6*64 + 16 + self.hidden_size if 'trn' in self.cfg.MODEL.ACTION_NET else 6*6*64 + 16
            else:
                enc_input_size = 128 + 16 + self.hidden_size if 'trn' in self.cfg.MODEL.ACTION_NET else 128 + 16 
            # a. use max/avg pooling to get 1d vector then use regular GRU
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
            self.enc_cell = nn.GRUCell(enc_input_size, self.hidden_size)
        else:
            raise NameError(self.cfg.MODEL.ACTION_NET)
        
        # The decoder RNN to predict future actions
        self.dec_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)
        self.dec_input_linear = nn.Sequential(nn.Linear(self.num_classes, self.hidden_size),
                                             nn.ReLU())
        self.future_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                             nn.ReLU())
        self.dec_cell = nn.GRUCell(self.hidden_size, self.hidden_size)

        # The classifier layer
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def enc_step(self, x_visual, enc_hx, x_bbox=None, future_inputs=None):
        '''
        Run one step of the encoder
        x_visual: visual feature as the encoder inputs
        x_bbox: bounding boxes as the encoder inputs
        future_inputs: encoder inputs from the decoder end (TRN)
        '''
        batch_size = x_visual.shape[0]
        if 'convlstm' in self.cfg.MODEL.ACTION_NET:
            h_fused = enc_hx[2]
            # run ConvLSTM
            h, c = self.enc_cell(x_visual, enc_hx[:2], future_inputs)
            # get input for GRU
            fusion_input = h.view(batch_size, -1)
            if future_inputs is not None:
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            fusion_input = torch.cat([fusion_input, x_bbox], dim=-1)
            # run GRU 
            h_fused = self.enc_fused_cell(self.enc_drop(fusion_input), 
                                          self.recurrent_drop(h_fused))
            enc_hx = [h, c, h_fused]
            enc_score = self.classifier(self.enc_drop(h_fused))
        elif 'gru' in self.cfg.MODEL.ACTION_NET:
            # avg pool visual feature and concat with bbox input
            if self.cfg.MODEL.INPUT_LAYER == 'attention':
                if 'trn' in self.cfg.MODEL.ACTION_NET:
                    x_visual, attentions = self.x_visual_extractor(x_visual, future_inputs)
                else:
                    x_visual, attentions = self.x_visual_extractor(x_visual, enc_hx)
            else:
                x_visual = self.x_visual_extractor(x_visual)
            fusion_input = torch.cat((x_visual, x_bbox), dim=1)
            if future_inputs is not None:
                # add input collected from action decoder
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            enc_hx = self.enc_cell(self.enc_drop(fusion_input), 
                                   self.recurrent_drop(enc_hx))
            enc_score = self.classifier(self.enc_drop(enc_hx))
        else:
            raise NameError(self.cfg.MODEL.ACTION_NET)
            
        return enc_hx, enc_score
      
    def decoder(self, enc_hx, dec_inputs=None):
        '''
        Run decoder for pred_len step to predict future actions
            enc_hx: last hidden state of encoder
            dec_inputs: decoder inputs
        '''
        dec_hx = enc_hx[-1] if isinstance(enc_hx, list) else enc_hx
        dec_scores = []
        future_inputs = dec_hx.new_zeros(dec_hx.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.ACTION_NET else None
        for t in range(self.pred_len):
            dec_hx = self.dec_cell(self.dec_drop(dec_inputs), 
                                   self.recurrent_drop(dec_hx))
            dec_score = self.classifier(self.dec_drop(dec_hx))
            dec_scores.append(dec_score)
            dec_inputs = self.dec_input_linear(dec_score)        
            future_inputs = future_inputs + self.future_linear(dec_hx) if future_inputs is not None else None
        future_inputs = future_inputs / self.pred_len if future_inputs is not None else None
        return torch.stack(dec_scores, dim=1), future_inputs

    def forward(self, x_visual, x_bbox=None, dec_inputs=None):
        '''
        For training only!
        Params:
            x_visual: visual feature as the encoder inputs (batch, SEG_LEN, 512, 7, 7)
            x_bbox: bounding boxes as the encoder inputs (batch, SEG_LEN, ?)
            dec_inputs: other inputs to the decoder, (batch, SEG_LEN, PRED_LEN, ?)
        Returns:
            all_enc_scores: (batch, SEG_LEN, num_classes)
            all_dec_scores: (batch, SEG_LEN, PRED_LEN, num_classes)
        '''
        future_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.ACTION_NET else None
        enc_hx = x_visual.new_zeros(x_visual.shape[0], self.hidden_size)
        all_enc_scores = []
        all_dec_scores = []
        for t in range(self.cfg.MODEL.SEG_LEN):
            # Run one step of action detector/predictor
            enc_scores, enc_hx, dec_scores, future_inputs = self.step(x_visual[:, t], enc_hx, x_bbox[:, t], future_inputs, dec_inputs)
            all_enc_scores.append(enc_scores)
            if dec_scores is not None:
                all_dec_scores.append(dec_scores)
        all_enc_scores = torch.stack(all_enc_scores, dim=1)
        all_dec_scores = torch.stack(all_dec_scores, dim=1)
        return all_enc_scores, all_dec_scores
    
    def step(self, x_visual, enc_hx, x_bbox=None, future_inputs=None, dec_inputs=None):
        '''
        Directly call step when run inferencing.
        x_visual: (batch, 512, 7, 7)
        enc_hx: (batch, hidden_size)
        '''
        # 1. encoder
        enc_hx, enc_scores = self.enc_step(x_visual, enc_hx, x_bbox=x_bbox, future_inputs=future_inputs)

        # 2. decoder
        dec_scores = None
        if 'trn' in self.cfg.MODEL.ACTION_NET:
            if dec_inputs is None:
                dec_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size)
            dec_scores, future_inputs = self.decoder(enc_hx, dec_inputs=dec_inputs)

        return enc_scores, enc_hx, dec_scores, future_inputs

