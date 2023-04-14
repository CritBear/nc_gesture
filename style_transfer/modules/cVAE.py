import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nc_gesture.style_transfer.modules.networks as ns

class CVAE(nn.Module):
    def __init__(self,options):
        super(CVAE,self).__init__()

        self.options = options
        #encoder
        self.fc1 = nn.Linear(options.input_size+options.condition_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc31 = nn.Linear(1024, options.latent_variable_size)
        self.fc32 = nn.Linear(1024, options.latent_variable_size)

    def encoder(self,x,c):
        concat_input = torch.cat([x,c],1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return self.fc31(h),self.fc32(h)

    def decoder(selfself,z_s,z_c,c):
        concat_input = torch.cat([z_s,z_c,c],1)

class Generator(nn.Module):
    def __init__(self,config):
        super(Generator,self).__init__()
        self.config = config
        self.style_encoder = ns.StyleEncoder(config)
        self.content_encoder = ns.ContentEncoder(config)
        self.decoder = ns.ContentDecoder(config)
        self.mlp = ns.MLP(config,ns.get_num_adain_params(self.decoder))

    def forward(self,motion):
        #print(motion.reshape(self.config.batch_size,-1).shape)
        z_s = self.style_encoder(motion)
        print("z_s:",z_s.shape)
        z_c = self.content_encoder(motion)
        print("z_c:",z_c.shape)
        out = self.decode(motion, z_s)
        print("out:",out.shape)
        return out

    def decode(self,content,model_code):
        adain_params = self.mlp(model_code)
        ns.assign_adain_params(adain_params,self.decoder)
        out = self.decoder(content)
        return out

    def get_lagent_codes(self,data):
        codes ={}
        codes["content_code"] = self.content_encoder(data["content"])
        codes["style_code"] = self.style_encoder(data["style"])
        return codes

