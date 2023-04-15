import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.highway import Highway
from utils.functional import parameters_allocation_check


class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()

        self.options = options

        self.hw1 = Highway(self.options.input_size, 2, F.relu)

        self.rnn = nn.LSTM(input_size=self.options.input_size,
                           hidden_size=self.options.encoder_rnn_size,
                           num_layers=self.options.encoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, input_size] tensor
        :return: [batch_size, latent_variable_size] tensor
        """

        [batch_size, seq_len, input_size] = x.size()

        x = x.view(-1, input_size)
        x = self.hw1(x)
        x = x.view(batch_size, seq_len, input_size)

        _, (hidden_state, cell) = self.rnn(x)

        # hidden_state = torch.swapaxes(hidden_state, 0, 1)
        # cell = torch.swapaxes(cell, 0, 1)

        return hidden_state, cell
