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

        self.context_to_mu = nn.Linear(
            self.options.encoder_rnn_size * self.options.encoder_num_layers,
            self.options.latent_variable_size
        )
        self.context_to_logvar = nn.Linear(
            self.options.encoder_rnn_size * self.options.encoder_num_layers,
            self.options.latent_variable_size
        )

        self.decoder = Decoder(self.options)

    def forward(self, input, z=None, initial_state=None):

        if z is None:
            [batch_size, seq_len, feature_dim] = input.size()

            encoder_hidden = self.encoder(input)

            encoder_h = encoder_hidden[0].view(batch_size, self.options.encoder_rnn_size * self.options.encoder_num_layers).to(self.options.device)

            mu = self.context_to_mu(encoder_h)
            logvar = self.context_to_logvar(encoder_h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(self.options.device)

            z = mu + eps * std

            z = z.repeat(1, seq_len, 1)
            z = z.view(batch_size, seq_len, self.options.latent_variable_size).to(self.options.device)
            recon_output, hidden = self.decoder(z, encoder_hidden)

            losses = self.loss_function(recon_output, input, mu, logvar)
            loss, recon_loss, kld_loss = (
                losses['loss'],
                losses['recon_loss'],
                losses['kld_loss']
            )

        return loss, recon_output, (recon_loss, kld_loss)

    def loss_function(self, recon, input, mu, logvar):
        kld_weight = 0.05
        recon_loss = F.mse_loss(recon, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recon_loss + kld_weight * kld_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss.detach(),
            'kld_loss': -kld_loss.detach()
        }
