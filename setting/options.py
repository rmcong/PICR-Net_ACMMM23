"""
@Project: PICR_Net
@File: setting/options.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""


import argparse

parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)
# training set
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=24, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training image size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--backbone', type=str, default='vgg16', help='backbone')
parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='4', help='the gpu id')
# training dataset
parser.add_argument('--rgb_root', type=str, default='./RGBD_dataset/train/RGB/')
parser.add_argument('--depth_root', type=str, default='./RGBD_dataset/train/depth/')
parser.add_argument('--gt_root', type=str, default='./RGBD_dataset/train/GT/')

# validation dataset
parser.add_argument('--val_rgb_root', type=str, default='./RGBD_dataset/val/RGB/')
parser.add_argument('--val_depth_root', type=str, default='./RGBD_dataset/val/depth/')
parser.add_argument('--val_gt_root', type=str, default='./RGBD_dataset/val/GT/')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='the path to save models and logs')
# testing set
parser.add_argument('--testsize', type=int, default=224, help='testing image size')
parser.add_argument('--test_path', type=str, default='/data/zhangchen/RGBD_dataset/test/', help='test dataset path')
parser.add_argument('--test_model', type=str, default='./checkpoints/PICR-Net.pth')
parser.add_argument('--smap_save', type=str, default='./test_maps/PICR-Net/', help='the save path of predictions')

opt = parser.parse_args()
