"""
@Project: PICR_Net
@File: modules/transformer_decoder.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from .swin_transformer import Mlp, WindowAttention, SwinTransformerBlock
from timm.models.layers import trunc_normal_
from modules.cross_transformer import *

class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, bias=True, norm_layer=False):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential()
        self.basicconv.add_module(
            'conv', nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias))
        if norm_layer:
            self.basicconv.add_module('bn', nn.BatchNorm2d(out_planes))
        self.basicconv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        input: [B, H * W, C]
        output: [B, H*2 * W*2, C/2]
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x



class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer decoder layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch expanding layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x



class conv_base(nn.Module):
    def __init__(self,in_res,in_dim,out_res,out_dim):
        super(conv_base, self).__init__()
        # print(in_res, in_dim, out_res, out_dim)
        self.in_res=in_res
        self.in_dim=in_dim
        self.out_res=out_res
        self.out_dim=out_dim
        self.layer1 = BaseConv2d(in_dim, out_dim)
        # self.layer2 = BaseConv2d(out_dim, out_dim)
        # print(in_res,in_dim,out_res,out_dim)
    def forward(self,x):
        B, _, _ = x.shape
        x = x.view(B, self.in_res, self.in_res, -1).permute(0, 3, 1, 2)
        x=self.layer1(x)
        x = F.interpolate(x, (self.out_res,self.out_res), mode='bilinear', align_corners=True)
        x = x.permute(0, 2,3,1).view(B, self.out_res*self.out_res, self.out_dim)
        return x


class TransformerDecoder_side(nn.Module):

    def __init__(self, dim_list=[768, 768, 384, 192], decoder_depths=[2, 6, 2, 2], num_heads=[24, 12, 6, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, ):
        super().__init__()

        self.dim_list = dim_list
        self.num_layers = len(decoder_depths)
        self.patches_resolution = [(7, 7), (7, 7), (14, 14), (28, 28)]
        self.embed_dim = 96
        self.patch_norm = patch_norm
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dims = [96 * 2, 96 * 4, 96 * 8, 96 * 8]
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]  # stochastic depth decay rule

        # build decoder layers
        self.layers = nn.ModuleList()
        self.concat_layer = nn.ModuleList()
        for i_layer in range(self.num_layers):

            concat_layer = nn.Linear(2 * self.dim_list[i_layer], self.dim_list[i_layer]) if i_layer > 0 else nn.Identity()
            layer = BasicLayer_up(dim=self.dim_list[i_layer],
                                  input_resolution=self.patches_resolution[i_layer],
                                  depth=decoder_depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  upsample=PatchExpand if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)
            self.concat_layer.append(concat_layer)
            self.layers.append(layer)
        self.norm = norm_layer(self.embed_dim)
        self.apply(self._init_weights)
        self.fusion=nn.ModuleList([])
        for item in self.dims[:-1]:
            self.fusion.append(PointFusion_side(dim=item, depth=1, heads=3, dim_head=item // 3))
        self.fusion.append(PointFusion(dim= self.dims[-1], depth=1, heads=3, dim_head=self.dims[-1] // 3))
        self.conv_map4 = nn.Sequential(BaseConv2d(768, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map3 = nn.Sequential(BaseConv2d(384, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map2 = nn.Sequential(BaseConv2d(192, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.conv_map1 = nn.Sequential(BaseConv2d(96, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.out=[self.conv_map4,self.conv_map3,self.conv_map2,self.conv_map1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_features,depth_features):




        B, _, _ = rgb_features[0].shape
        assert len(self.concat_layer) == len(self.layers)
        sides=[]
        l=len(self.layers)
        # for i in range(len(self.layers)):
            # print(rgb_features[l-i].shape, depth_features[l-i].shape)
        for i in range(len(self.layers)):
            # print(rgb_features[l-i].shape, depth_features[l-i].shape)
            if i == 0:
                rgbd_i = self.fusion[l-i-1](rgb_features[l-i], depth_features[l-i])
                x = rgbd_i
            else:
                maps=F.interpolate(sides[i-1], self.patches_resolution[i], mode='bilinear', align_corners=True)
                rgbd_i = self.fusion[l-i-1](rgb_features[l-i], depth_features[l-i],maps.sigmoid())
                x = self.concat_layer[i](torch.cat([x, rgbd_i], dim=-1))
            # print('concat', x.shape)
            x = self.layers[i](x)
            # print('layer', x.shape)
            fea = x.view(B, 7 * (2 ** i), 7 * (2 ** i), -1).permute(0, 3, 1, 2)
            # print('fea', fea.shape)
            sides.append(self.out[i](fea))

        x = self.norm(x)  # B L C  # [48, 3136, 96]

        return x,sides



if __name__ == "__main__":
    x_list = []
    x1 = torch.randn(1, 784, 192)
    x_list.append(x1)
    x2 = torch.randn(1, 196, 384)
    x_list.append(x2)
    x3 = torch.randn(1, 49, 768)
    x_list.append(x3)
    x4 = torch.randn(1, 49, 768)
    x_list.append(x4)

    model = TransformerDecoder_side()
    # print(model)

