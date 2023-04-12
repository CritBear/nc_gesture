import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()

        self.options = options

        self.rnn = nn.LSTM(input_size=self.options.latent_variable_size,
                           hidden_size=self.options.decoder_rnn_size,
                           num_layers=self.options.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.fc = nn.Linear(self.options.decoder_rnn_size, self.options.output_size)

    def forward(self, z, hidden):
        """
        :param z: [batch_size, seq_len, latent_size] tensor
        :return:
        """

        # print('z')
        # print(z.size())
        # [batch_size, _] = z.size()

        output, (hidden, cell) = self.rnn(z, hidden)
        result = self.fc(output)

        return result, (hidden, cell)