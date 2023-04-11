import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.encoder import Encoder
from modules.decoder import Decoder

from utils.functional import kld_coef, parameters_allocation_check, fold


class RVAE(nn.Module):
    def __init__(self, options):
        super(RVAE, self).__init__()

        self.options = options

        self.encoder = Encoder(self.options)

        self.context_to_mu = nn.Linear(self.options.encoder_rnn_size * 2, self.options.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.options.encoder_rnn_size * 2, self.options.latent_variable_size)

        self.decoder = Decoder(self.options)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):

        if z is None:
            # Get context from encoder and sample z ~ N(mu, std)
            [batch_size, _] = encoder_input.size()

            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = torch.exp(0.5 * logvar)

            z = Variable(torch.randn([batch_size, self.options.latent_variable_size]))
            if self.options.use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        out, final_state = self.decoder(z, drop_prob, initial_state)

        return out, final_state, kld
