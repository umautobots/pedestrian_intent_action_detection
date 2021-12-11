'''
we need to make it generalize to any 3D Conv network
'''
import torch
import torch.nn as nn
from lib.modeling.layers.convlstm import ConvLSTMCell

class IntentNet(nn.Module):
    def __init__(self, cfg, x_visual_extractor=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = self.cfg.MODEL.PRED_LEN
        self.num_classes = self.cfg.DATASET.NUM_INTENT
        if self.num_classes == 2 and self.cfg.MODEL.INTENT_LOSS=='bce':
            self.num_classes = 1
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
            
            enc_input_size = 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN + self.hidden_size if 'action' in self.cfg.MODEL.TASK else 16 + 6*6*self.cfg.MODEL.CONVLSTM_HIDDEN
            self.enc_fused_cell = nn.GRUCell(enc_input_size, self.hidden_size)
        elif 'gru' in self.cfg.MODEL.INTENT_NET:
            # use avg pooling/conv2d to get 1d vector then use regular GRU
            if self.cfg.MODEL.INPUT_LAYER == 'conv2d':
                enc_input_size = 6*6*64 + 16 + self.hidden_size if 'action' in self.cfg.MODEL.TASK else 6*6*64 + 16
            elif self.cfg.MODEL.INPUT_LAYER == 'attention':
                enc_input_size = 7*7*64 + 16 + self.hidden_size if 'action' in self.cfg.MODEL.TASK else 7*7*64 + 16
            else:
                enc_input_size = 128 + 16 + self.hidden_size if 'action' in self.cfg.MODEL.TASK else 128 + 16 
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
            raise NameError(self.cfg.MODEL.INTENT_NET)
        
        # The classifier layer
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def step(self, x_visual, enc_hx, x_bbox=None, future_inputs=None):
        '''
        Run one step of the encoder
        x_visual: visual feature as the encoder inputs (batch, 512, 7, 7)
        enc_hx: (batch, hidden_size)
        x_bbox: bounding boxes embeddings as the encoder inputs (batch, ?)
        future_inputs: encoder inputs from the decoder end (TRN)
        '''
        batch_size = x_visual.shape[0]
        if 'convlstm' in self.cfg.MODEL.INTENT_NET:
            h_fused = enc_hx[2]
            # run ConvLSTM
            if isinstance(future_inputs, list):
                # for convlstm action_net, act_hx is [h_map, c_map, h_fused]
                h, c = self.enc_cell(x_visual, enc_hx[:2], future_inputs[-1])
            else:
                h, c = self.enc_cell(x_visual, enc_hx[:2], future_inputs)
            
            # get input for GRU
            fusion_input = h.view(batch_size, -1)
            if isinstance(future_inputs, list):
                fusion_input = torch.cat([fusion_input, future_inputs[-1]], dim=1)
            elif isinstance(future_inputs, torch.Tensor):
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            fusion_input = torch.cat([fusion_input, x_bbox], dim=-1)

            # run GRU 
            h_fused = self.enc_fused_cell(self.enc_drop(fusion_input), 
                                          self.recurrent_drop(h_fused))
            enc_hx = [h, c, h_fused]
            enc_score = self.classifier(self.enc_drop(h_fused))
        elif 'gru' in self.cfg.MODEL.INTENT_NET:
            # avg pool visual feature and concat with bbox input
            # or we can run a 7x7 kenel CNN for the same purpose also with ability of dimension reduction.
            if self.cfg.MODEL.INPUT_LAYER == 'attention':
                if 'trn' in self.cfg.MODEL.INTENT_NET:
                    x_visual, attentions = self.x_visual_extractor(x_visual, future_inputs)
                else:
                    x_visual, attentions = self.x_visual_extractor(x_visual, enc_hx)
            else:
                x_visual = self.x_visual_extractor(x_visual)
            fusion_input = torch.cat((x_visual, x_bbox), dim=-1)
            if future_inputs is not None:
                # add input collected from action decoder
                fusion_input = torch.cat([fusion_input, future_inputs], dim=1)
            enc_hx = self.enc_cell(self.enc_drop(fusion_input), 
                                   self.recurrent_drop(enc_hx))
            enc_score = self.classifier(self.enc_drop(enc_hx))
        else:
            raise NameError(self.cfg.MODEL.INTENT_NET)
        
        return enc_hx, enc_score

    def forward(self, x_visual, x_bbox=None, future_inputs=None):
        '''
        For training only!
        Params:
            x_visual: visual feature as the encoder inputs (batch, SEG_LEN, 512, 7, 7)
            x_bbox: bounding boxes as the encoder inputs (batch, SEG_LEN, 4)
            dec_inputs: other inputs to the decoder, (batch, SEG_LEN, PRED_LEN, ?)
        Returns:
            all_enc_scores: (batch, SEG_LEN, num_classes)
            all_dec_scores: (batch, SEG_LEN, PRED_LEN, num_classes)
        '''
        all_enc_scores = []
        for t in range(self.enc_steps):
            # Run one step of intention detector
            enc_hx, enc_scores = self.step(x_visual[:, t], enc_hx, x_bbox[:, t], future_inputs)
            all_enc_scores.append(enc_scores)
        return torch.stack(all_enc_scores, dim=1)
    
    
        

        
        