import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from highway import Highway
from ..utils.functional import parameters_allocation_check


class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()

        self.options = options

        self.hw1 = Highway(self.options.input_size, 2, F.relu)

        self.rnn = nn.LSTM(input_size=self.options.input_size,
                           hidden_size=self.options.encoder_rnn_size,
                           num_layers=self.options.encoder_num_layers,
                           batch_first=True,  # ?
                           bidirectional=True)  # ?

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: [batch_size, latent_variable_size] tensor
        """

        [batch_size, seq_len, embed_size] = input.size()

        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        _, (_, final_state) = self.rnn(input)

        final_state = final_state.view[-1]
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = torch.cat([h_1, h_2], 1)

        return final_state