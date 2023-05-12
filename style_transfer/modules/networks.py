import torch
import torch.nn as nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
from nc_gesture.style_transfer.modules.blocks import ConvBlock, ResBlock, LinearBlock, \
    BottleNeckResBlock, Upsample, ConvLayers, ActiFirstResBlock, \
    get_conv_pad, get_norm_layer
from nc_gesture import nc_attention_encoder as AE

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.attention_encoder = AE.AttentionEncoder(100,config.num_joints,6,6)
        self.style_encoder = StyleEncoder(config)
        self.content_encoder = ContentEncoder(config)
        self.decoder = ContentDecoder(config)
        self.mlp = MLP(config, get_num_adain_params(self.decoder))
        self.disc = Discriminator(config)

    def forward(self, content, style):
        content = self.attention_encoder(content.transpose(1,2))
        z_s = self.style_encoder(style)
        z_c = self.content_encoder(content.transpose(1,2))
        out = self.decode(z_c, z_s)
        return out

    def decode(self, content, model_code):
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.decoder)
        out = self.decoder(content)
        return out

    def get_lagent_codes(self,data):
        codes ={}
        codes["content_code"] = self.content_encoder(data["content"])
        codes["style_code"] = self.style_encoder(data["style"])
        return codes

    def get_style_code(self, motion):
        style_code = self.style_encoder(motion)
        return style_code


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[: , : m.num_features]
            std = adain_params[: , m.num_features: 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[: , 2 * m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

class MLP(nn.Module):
    def __init__(self, config, out_dim):
        super(MLP, self).__init__()
        dims = config.mlp_dims
        n_blk = len(dims)
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim,
                                   norm='none', acti='none')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class ContentEncoder(nn.Module):
    def __init__(self,config):
        super(ContentEncoder, self).__init__()
        channels = config.enc_co_channels
        kernel_size = config.enc_co_kernel_size
        stride = config.enc_co_stride

        layers = []
        n_convs = config.enc_co_down_n
        n_resblk = config.enc_co_resblks
        acti = 'lrelu'

        assert n_convs + 1 == len(channels)

        for i in range(n_convs):
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1],
                                stride=stride, norm='in', acti=acti)

        for i in range(n_resblk):
            layers.append(ResBlock(kernel_size, channels[-1], stride=1,
                                   pad_type='reflect', norm='in', acti=acti))

        self.conv_model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):
        x = self.conv_model(x)
        return x

class StyleEncoder(nn.Module):
    def __init__(self, config):
        super(StyleEncoder, self).__init__()
        channels = config.enc_cl_channels
        channels[0] = config.style_channel_3d

        kernel_size = config.enc_cl_kernel_size
        stride = config.enc_cl_stride

        self.global_pool = F.max_pool1d

        layers = []
        n_convs = config.enc_cl_down_n

        for i in range(n_convs):
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1],
                                stride=stride, norm='none', acti='lrelu')


        self.conv_model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):
        x = self.conv_model(x)
        kernel_size = x.shape[-1]
        x = self.global_pool(x, kernel_size)
        return x

class ContentDecoder(nn.Module):
    def __init__(self, config):
        super(ContentDecoder, self).__init__()

        channels = config.dec_channels
        kernel_size = config.dec_kernel_size
        stride = config.dec_stride

        res_norm = 'none' # no adain in res
        norm = 'none'
        pad_type = 'reflect'
        acti = 'lrelu'

        layers = []
        n_resblk = config.dec_resblks
        n_conv = config.dec_up_n
        bt_channel = config.dec_bt_channel # #channels at the bottleneck

        layers += get_norm_layer('adain', channels[0]) # adain before everything

        for i in range(n_resblk):
            layers.append(BottleNeckResBlock(kernel_size, channels[0], bt_channel, channels[0],
                                             pad_type=pad_type, norm=res_norm, acti=acti))

        for i in range(n_conv):
            layers.append(Upsample(scale_factor=2, mode='nearest'))
            cur_acti = 'none' if i == n_conv - 1 else acti
            cur_norm = 'none' if i == n_conv - 1 else norm
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1], stride=stride,
                                pad_type=pad_type, norm=cur_norm, acti=cur_acti)

        self.model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()

        channels = config.disc_channels
        down_n = config.disc_down_n
        ks = config.disc_kernel_size
        stride = config.disc_stride
        pool_ks = config.disc_pool_size
        pool_stride = config.disc_pool_stride
        out_dim = config.num_style_classes

        assert down_n + 1 == len(channels)

        cnn_f = ConvLayers(kernel_size=ks, in_channels=channels[0], out_channels=channels[0])

        for i in range(down_n):
            cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i], stride=stride, acti='lrelu', norm='none')]
            cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[i], out_channels=channels[i + 1], stride=stride, acti='lrelu', norm='none')]
            cnn_f += [get_conv_pad(pool_ks, pool_stride)]
            cnn_f += [nn.AvgPool1d(kernel_size=pool_ks, stride=pool_stride)]

        cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]
        cnn_f += [ActiFirstResBlock(kernel_size=ks, in_channels=channels[-1], out_channels=channels[-1], stride=stride, acti='lrelu', norm='none')]

        cnn_c = ConvBlock(kernel_size=ks, in_channels=channels[-1], out_channels = out_dim,
                          stride=1, norm='none', acti='lrelu', acti_first=True)

        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.device = config.device
    def forward(self,x,y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x) # 마지막 레이어 전
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).to(self.device)
        out = out[index,y, :] # y번째 요소 값만 가져옴.
        return out, feat

    '''
    fake input에 대한 loss 계산.
    '''
    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).to(self.device)  # 가짜라 판별한 갯수를 계산.
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    '''
    real input에 대한 loss 계산.
    '''
    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).to(self.device)
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),#prod array 내부 요소들의 곱.prod([128,128,39]) = 128 * 128 * 39
                                   dtype=torch.float).to(self.device)
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg