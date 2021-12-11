import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))
    def get_score_vec(self, encoder_states, decoder_state):
        return torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = self.get_score_vec(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states)
        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs


class AdditiveAttention2D(nn.Module):
    '''
    Given feature map and hidden state, 
    compute an attention map
    '''
    def __init__(self, cfg):
        super(AdditiveAttention2D, self).__init__()
        self.input_drop = nn.Dropout(0.4)
        self.hidden_drop = nn.Dropout(0.2)
        # self.enc_net = nn.Conv2d(512, 128, kernel_size=[2, 2], padding=1, bias=False)
        # self.dec_net = nn.Linear(128, 128, bias=False)
        # self.score_net = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=[2, 2], bias=False)
        self.enc_net = nn.Linear(512, 128, bias=True)
        self.dec_net = nn.Linear(128, 128, bias=False)
        self.score_net = nn.Linear(128, 1, bias=True)
        self.output_linear = nn.Sequential(
                                        #    nn.Linear(512, 128),
                                           nn.Linear(512, 64),
                                           nn.ReLU()
                                           )

    def forward(self, input_x, hidden_states):
        '''
        The implementation is similar to Eq(5) in 
        https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf
        Params:
            x: feature map (inputs) or hidden state map (enc_h)
            future_inputs: the input feature from the decoder
        NOTE: in literatures, spatial attention was applied in deep-cnn, if we only use it on final 7*7 map, would it be problematic?
        '''
        # NOTE: Oct 26, old implementation of attention based on Conv2d.
        # x_map = self.enc_net(self.input_drop(input_x)) # Bx512x7x7 -> Bx128x8x8
        # state_map = self.dec_net(self.hidden_drop(hidden_states))
        # score_map = self.score_net(torch.tanh(x_map + state_map[..., None, None])) # BxChx8x8 -> BxChx7x7
        # attention_probs = F.softmax(score_map.view(score_map.shape[0], -1), dim=-1).view(score_map.shape[0], 1, 7, 7)
        # final_context_vec = torch.sum(attention_probs * input_x, dim=(2,3))
        # final_context_vec = self.output_linear(final_context_vec)
        
        # NOTE: Oct 27, new implementation of attention based on linear.
        batch, ch, width, height = input_x.shape
        input_x = input_x.view(batch, ch, -1).permute(0,2,1)
        x_map = self.enc_net(self.input_drop(input_x)) #  Bx49x128
        state_map = self.dec_net(self.hidden_drop(hidden_states))
        
        score_map = self.score_net(torch.tanh(x_map + state_map[:, None, :])) # Bx49xCh -> Bx49x1
        
        # NOTE: first attention type is softmax + weighted sum
        # attention_probs = F.softmax(score_map, dim=1)
        # final_context_vec = torch.sum(attention_probs * input_x, dim=1)
        # NOTE: second attention type is sigmoid + weighted mean
        # attention_probs = score_map.sigmoid()
        # final_context_vec = torch.mean(attention_probs * input_x, dim=1)
        # final_context_vec = self.output_linear(final_context_vec)
        # NOTE: third attention type is sigmoid + fc + flatten
        attention_probs = score_map.sigmoid()
        final_context_vec = torch.reshape(attention_probs * self.output_linear(input_x), (batch, -1))
        return final_context_vec, attention_probs