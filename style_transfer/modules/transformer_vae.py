import torch
import torch.nn as nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
from nc_gesture import nc_attention_encoder as AE
from nc_gesture.style_transfer.modules.transformer import *

class TVAE(nn.Module):
    def __init__(self, config):
        super(TVAE, self).__init__()
        self.config = config
        self.encoder = Encoder_TRANSFORMER("TVAE",config.num_joints,config.num_feats,config.num_channel,
                                           config.num_action_classes,config.num_style_classes)
        self.decoder = Decoder_TRANSFORMER("TVAE",config.num_joints,config.num_feats,config.num_channel,
                                           config.num_action_classes,config.num_style_classes)
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        #encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        return batch
