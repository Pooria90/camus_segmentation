import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, Unet
from monai.networks.layers.factories import Act, Norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_unet_small(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=[32,32,64,128,128,128],
        strides=[2,2,2,2,2],
        num_res_units=2,
        act='RELU',
        norm=None,
):
    model = Unet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=None,
    )
    return model

def get_unet_large(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=[48,96,192,384,768],
        strides=[2,2,2,2],
        num_res_units=2,
        act='RELU',
        norm='BATCH',
        bias=False,
):
    model = Unet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        bias=bias,
    )
    return model

def get_attention_unet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=[64, 128, 256, 512, 1024],
        strides=[2,2,2,2]
):
    model = AttentionUnet( 
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
    )
    return model
    