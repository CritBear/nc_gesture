import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn

from nc_gesture.style_transfer.modules import blocks as B


class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()

        self.options = options

        # self.hw1 = Highway(self.options.input_size, 2, F.relu)

        self.rnn = nn.LSTM(input_size=self.options.input_size,
                           hidden_size=self.options.encoder_rnn_size,
                           num_layers=self.options.encoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, input_size] tensor
        :return: [batch_size, latent_variable_size] tensor
        """

        _, (hidden_state, cell) = self.rnn(input)

        # hidden_state = torch.swapaxes(hidden_state, 0, 1)
        # cell = torch.swapaxes(cell, 0, 1)

        return (hidden_state, cell)

