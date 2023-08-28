"""
@Project: PICR_Net
@File: model/build_model.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from modules.swin_transformer import Swin_T, Mlp
from modules.cross_transformer import PointFusion_side,PointFusion
from modules.transformer_decoder import *
from modules.VGG import LowFeatureExtract


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



def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class PICR_Net(nn.Module):

    def __init__(self):
        super(PICR_Net, self).__init__()

        self.backbone = Swin_T(pretrained=True)
        self.low_feature_extract = LowFeatureExtract()

        self.dims = [96 * 2, 96 * 4, 96 * 8, 96 * 8]
        self.fusion = nn.ModuleList([])


        self.transformer_decoder = TransformerDecoder_side()

        # self.conv_input = nn.Conv2d(4, 3, kernel_size=1, padding=0)
        self.conv_112 = BaseConv2d(96, 128)
        self.conv_224 = nn.Sequential(BaseConv2d(256, 64), BaseConv2d(64, 64))
        self.conv_atten = conv1x1(256, 256)
        self.conv_atten1 = conv1x1(128, 128)
        self.conv_map = nn.Sequential(BaseConv2d(128, 32), nn.Conv2d(32, 1, kernel_size=3, padding=1))

    def forward(self, image, depth):
        """
        Args:
            image: The input of RGB images, three channels.
            depth: The input of Depth images, single channels.

        Returns: The final saliency maps.

        """
        b, c, h, w = image.size()
        # rgb_depth = torch.cat([image, depth], dim=1)
        depth = torch.cat([depth, depth, depth], dim=1)

        # Shared backbone to extract multi-level features
        rgb_features = self.backbone(image)  # list, length=5
        depth_features = self.backbone(depth)

        # Point Fusion Layer
        # rgbd_features = []
        # rgbd_i = self.fusion[0](rgb_features[1], depth_features[1])
        #
        # rgbd_features.append(rgbd_i)
        # for i in range(2, len(rgb_features)):
        #     rgbd_i = self.fusion[i - 1](rgb_features[i], depth_features[i])
        #
        #     rgbd_features.append(rgbd_i)

        # decoder
        x,sides = self.transformer_decoder(rgb_features,depth_features)  # [b, 3136, 96]
        # x = torch.cat([x, rgb_features[0]], dim=-1)  # [b, 3136, 192]
        x = x.view(b, 56, 56, 96).permute(0, 3, 1, 2)  # [b, 192, 56, 56]

        # CNN Based Saliency Maps Refinement Unit
        feature_224, feature_112 = self.low_feature_extract(image)
        x = self.conv_112(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # print(x.shape, feature_112.shape)
        x = torch.cat([x, feature_112], dim=1)#torch.Size([1, 128, 112, 112]) torch.Size([1, 128, 112, 112])
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        x = torch.mul(x, atten) + x
        x = self.conv_224(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # print(x.shape, feature_224.shape)
        x = torch.cat([x, feature_224], dim=1)#torch.Size([1, 64, 224, 224]) torch.Size([1, 64, 224, 224])
        atten1 = F.avg_pool2d(x, x.size()[2:])
        atten1 = torch.sigmoid(self.conv_atten1(atten1))
        x = torch.mul(x, atten1) + x
        smap = self.conv_map(x)
        # print(len(sides))
        # exit()

        return smap,sides



if __name__ == '__main__':
    rgb = torch.randn([1, 3, 224, 224])
    depth = torch.randn([1, 1, 224, 224])
    # model = C2TNet()
    # model(rgb, depth)
