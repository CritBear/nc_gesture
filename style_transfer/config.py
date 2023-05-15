import os
import sys
import torch
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)


class Config:
    cuda_id = 0

    # hyyyper params
    # data paths
    data_dir = pjoin(BASEPATH, 'datasets\data')
    expr_dir = BASEPATH
    data_file_name = "fixed_200_all.pkl"

    # model paths
    main_dir = None
    model_dir = None
    tb_dir = None
    info_dir = None
    output_dir = None

    vis_freq = 100
    log_freq = 100
    save_freq = 50000
    mt_save_iter = 50000       # How often do you want to save output images during training
    mt_display_iter = 5000       # How often do you want to display output images during training
    mt_batch_n = 1  # number of batches to save in training

    # optimization options
    num_epochs = 100000              # maximum number of training iterations
    weight_decay = 0.0001          # weight decay
    lr_gen = 0.0001 #0.0001                # learning rate for the generator
    lr_dis = 0.0001                # learning rate for the discriminator
    weight_init = 'kaiming'                 # initialization [gaussian/kaiming/xavier/orthogonal]
    lr_policy = None

    triplet_margin = 1
    # Training
    batch_size = 40 # 128

    # Testing
    test_batch_n = 56  # number of test clips

    # dataset
    dataset_norm_config = {  # specify the prefix of mean/std
        "train":
            {"content": None, "style3d": None, "style2d": None},  # will be named automatically as "train_content", etc.
        "test":
            {"content": "train", "style3d": "train", "style2d": "train"},
        "trainfull":
            {"content": "train", "style3d": "train", "style2d": "train"}
    }

    # input: T * 64
    num_feats = 6

    latent_dim = 64
    num_heads = 4
    rot_channels = 100  # 128
    pos3d_channels = 100  # 64

    num_channel = rot_channels
    num_joints = 26  # 21

    style_channel_3d = pos3d_channels

    """
    encoder for class
    [down_n] stride=[enc_cl_stride], dim=[enc_cl_channels] convs, 
    followed by [enc_cl_global_pool]

    """
    enc_cl_down_n = 2  # 100 -> 128 -> 160
    enc_cl_channels = [0, 128, 160] # 64, 96, 144
    enc_cl_kernel_size = 8
    enc_cl_stride = 2

    """
    encoder for content
    [down_n] stride=[enc_co_stride], dim=[enc_co_channels] convs (with IN)
    followed by [enc_co_resblks] resblks with IN
    """
    enc_co_down_n = 1  # 100 -> 160
    enc_co_channels = [num_channel, 160]
    enc_co_kernel_size = 8
    enc_co_stride = 2
    enc_co_resblks = 1


    """
    mlp
    map from class output [enc_cl_channels[-1] * 1]
    to AdaIN params (dim calculated at runtime)
    """
    mlp_dims = [enc_cl_channels[-1], 192, 256]

    """
    decoder
    [dec_resblks] resblks with AdaIN
    [dec_up_n] Upsampling followed by stride=[dec_stride] convs
    """

    dec_bt_channel = 144 # 144
    dec_resblks = enc_co_resblks
    dec_channels = enc_co_channels.copy()
    dec_channels.reverse()
    dec_channels[-1] = 100  # 156
    dec_up_n = enc_co_down_n
    dec_kernel_size = 8
    dec_stride = 1

    """
    discriminator
    1) conv w/o acti or norm, keeps dims
    2) [disc_down_n] *
            (ActiFirstResBlk(channel[i], channel[i])
            + ActiFirstResBlk(channel[i], channel[i + 1])
            + AvgPool(pool_size, pool_stride))
    3) 2 ActiFirstResBlks that keep dims(channel[-1])
    4) conv, [channel[-1] -> num_classes]

    """
    disc_channels = [pos3d_channels, 96, 144]
    disc_down_n = 2  # 64 -> 32 -> 16 -> 8 -> 4
    disc_kernel_size = 6
    disc_stride = 1
    disc_pool_size = 3
    disc_pool_stride = 2

    num_action_classes = 100
    num_style_classes = 4       # de,di,me,mi

    trans_weight = 0.5

    style_loss_weight = 1
    content_loss_weight = 1
    triplet_loss_weight = 0.1
    adv_loss_weight = 1
    ft_loss_weight = 0.5

    device = None
    gpus = 1
