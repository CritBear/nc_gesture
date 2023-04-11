import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()

        self.options = options

        self.rnn = nn.LSTM(input_size=self.options.latent_variable_size,
                           hidden_size=self.options.decoder_rnn_size,
                           num_layers=self.options.decoder_num_layers,
                           batch_first=True)

        self.fc = nn.Linear(self.options.decoder_rnn_size, self.options.output_size)

    def forward(self, decoder_input, z, drop_prob, initial_state=None):
        """
        :param decoder_input: [batch_size, seq_len, embed_size] tensor
        :param z: [batch_size, latent_variable_size] tensor (sequence context)
        :param drop_prob:
        :param initial_state:
        :return:
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        decoder_input = F.dropout(decoder_input, drop_prob)

        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.options.latent_variable_size)
        decoder_input = torch.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.options.output_size)

        return result, final_state