import torch
import torch.nn as nn
import torch.nn.functional as F
from nc_gesture.style_transfer.modules.blocks import ConvBlock, ResBlock, LinearBlock, \
    BottleNeckResBlock, Upsample, ConvLayers, ActiFirstResBlock, \
    get_conv_pad, get_norm_layer



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
