import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pdb

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding=0, bias=True, input_dropout=0.0, recurrent_dropout=0.0, attended=False):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        input_dropout: float
            dropout probability of inputs x
        recurrent_dropout: float
            dropout probability of hiddent states h. NOTE: do not apply dropout to memory cell c 
        attended: bool
            whether apply attention layer to the input feature map
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        self.padding = padding
        self.attended = attended
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.input_dropout = nn.Dropout2d(input_dropout)
        self.recurrent_dropout = nn.Dropout2d(recurrent_dropout)
        
        if self.attended:
            self.input_att_net = nn.Linear(512, 64, bias=True)
            self.hidden_att_net = nn.Linear(64, 64, bias=False)
            self.future_att_net = nn.Linear(128, 64, bias=False)
            self.score_net = nn.Linear(64, 1, bias=True)

    def forward(self, input_tensor, cur_state, future_inputs=None):
        '''
        input_tensor: the input to the convlstm model
        cur_state: the hidden state map of the convlstm model from previou recurrency
        future_inputs: the hidden state map from decoder or another convlstm stream.
        '''
        # NOTE: apply dropout to input x and hiddent state h
        h_cur, c_cur = cur_state
        # pad_size = self.width - h_cur.shape[-1]
        # h_cur  = F.pad(h_cur, (pad_size, 0, pad_size, 0)) # if padding=(1,0,1,0), pad 0 only on top and left of the input map.
        h_cur = F.upsample(h_cur, size=(7,7), mode='bilinear')
        
        # dropout 
        input_tensor = self.input_dropout(input_tensor)
        h_cur = self.recurrent_dropout(h_cur)

        if self.attended:
            # NOTE: this is an implementation of the spatial attention in SCA-CNN
            input_tensor = self.attention_layer(input_tensor, h_cur, future_inputs)

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())

    def attention_layer(self, input_tensor, hidden_states, future_inputs):
        batch, ch_x, height, width = input_tensor.shape
        ch_h = hidden_states.shape[1]
        input_vec = self.input_att_net(input_tensor.view(batch, ch_x, height*width).permute(0,2,1)) #  Bx49x128
        state_vec = self.hidden_att_net(hidden_states.view(batch, ch_h, height*width).permute(0,2,1))
        if future_inputs is not None:
            # Use the future input to compute attention if it's given
            score_vec = self.score_net(torch.tanh(input_vec + state_vec + self.future_att_net(future_inputs).unsqueeze(1)))
        else:
            score_vec = self.score_net(torch.tanh(input_vec + state_vec)) # Bx49xCh -> Bx49x1
        attention_probs = F.softmax(score_vec, dim=1)

        attention_probs = attention_probs.view(batch, 1, height, width)
        return input_tensor * attention_probs

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param