"""
@Project: PICR_Net
@File: modules/VGG.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""
import torch.nn as nn
import torchvision
import torch


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU(inplace=True))
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU(inplace=True))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU(inplace=True))
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU(inplace=True))
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU(inplace=True))
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3', nn.AvgPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU(inplace=True))
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU(inplace=True))
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU(inplace=True))
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.AvgPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU(inplace=True))
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU(inplace=True))
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU(inplace=True))
        self.conv5 = conv5

        # vgg_16 = torchvision.models.vgg16(pretrained=True)
        # self._initialize_weights(vgg_16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _initialize_weights(self, vgg_16):
        features = [
            self.conv1.conv1_1, self.conv1.relu1_1,
            self.conv1.conv1_2, self.conv1.relu1_2,
            self.conv2.pool1,
            self.conv2.conv2_1, self.conv2.relu2_1,
            self.conv2.conv2_2, self.conv2.relu2_2,
            self.conv3.pool2,
            self.conv3.conv3_1, self.conv3.relu3_1,
            self.conv3.conv3_2, self.conv3.relu3_2,
            self.conv3.conv3_3, self.conv3.relu3_3,
            self.conv4.pool3,
            self.conv4.conv4_1, self.conv4.relu4_1,
            self.conv4.conv4_2, self.conv4.relu4_2,
            self.conv4.conv4_3, self.conv4.relu4_3,
            self.conv5.pool4,
            self.conv5.conv5_1, self.conv5.relu5_1,
            self.conv5.conv5_2, self.conv5.relu5_2,
            self.conv5.conv5_3, self.conv5.relu5_3,
        ]
        for l1, l2 in zip(vgg_16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


def VGG_2(pretrained=True):
    net = VGG16()
    if pretrained:
        print("The vgg model loads the pretrained parameters...")
        model_dict = net.state_dict()
        pretrained_dict = torch.load("./pretrain/vgg16_bn-6c64b313.pth")
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        net.load_state_dict(model_dict)

    low_feature_extract_224 = nn.Sequential(*list(net.children())[0])
    low_feature_extract_112 = nn.Sequential(*list(net.children())[1])

    return low_feature_extract_224, low_feature_extract_112


class LowFeatureExtract(nn.Module):
    def __init__(self):
        super(LowFeatureExtract, self).__init__()
        (
            self.low_feature_extract_224,
            self.low_feature_extract_112
        ) = VGG_2(pretrained=True)

    def forward(self, x):
        feature_224 = self.low_feature_extract_224(x)
        feature_112 = self.low_feature_extract_112(feature_224)

        return feature_224, feature_112


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    backbone = LowFeatureExtract()
    out = backbone(x)