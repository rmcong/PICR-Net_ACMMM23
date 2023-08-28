"""
@Project: PICR_Net
@File: modules/cross_transformer.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce, repeat

from modules.swin_transformer import Mlp
from timm.models.layers import trunc_normal_
import math




class RelationModel_double(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super(RelationModel_double, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv_z = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_z = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_qkv_z1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_z1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        b, p, d = x.shape# [1, 784, 192] [1, 196, 384] [1, 49, 768] [1, 49, 768]
        h = self.heads

        # print(x.shape,y.shape)

        rgb_guidence = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=p)
        rgb_guidence = rgb_guidence.permute(0, 2, 1).expand(b, p, d)

        depth_guidence = F.avg_pool1d(y.permute(0, 2, 1), kernel_size=p)
        depth_guidence = depth_guidence.permute(0, 2, 1).expand(b, p, d)

        z = torch.stack([x, y, rgb_guidence, depth_guidence], dim=2)  # [b, p, 4, d]
        z_qkv = self.to_qkv_z(z).chunk(3, dim=-1)  # List: 3 * [b, p, 4, d']

        z_q, z_k, z_v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv)
        dots_z = einsum('b p h i d, b p h j d -> b p h i j', z_q, z_k) * self.scale
        atten_mask = torch.tensor([[0., 0., 0., -100.],
                                   [0., 0., -100., 0.],
                                   [0., -100., 0., 0.],
                                   [-100., 0., 0., 0.]], requires_grad=False).cuda()
        attn_z = F.softmax(dots_z + atten_mask, dim=-1)  # affinity matrix

        z_out = einsum('b p h i j, b p h j d -> b p h i d', attn_z, z_v)
        z_out = rearrange(z_out, 'b p h n d -> b p n (h d)')
        z_out = self.to_out_z(z_out)
        x, y = z_out[:, :, :2, :].chunk(2, dim=2)
        z1 = z_out  # [b, p, 4, d]
        z_qkv1 = self.to_qkv_z1(z1).chunk(3, dim=-1)  # List: 3 * [b, p, 4, d']

        z_q1, z_k1, z_v1 = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv1)
        dots_z1 = einsum('b p h i d, b p h j d -> b p h i j', z_q1, z_k1) * self.scale
        atten_mask1 = torch.tensor([[0., -100., 0., -100.],
                                    [-100., 0., -100., 0.],
                                    [-100., -100., -100., -100.],
                                    [-100., -100.,-100., -100.]], requires_grad=False).cuda()
        attn_z1 = F.softmax(dots_z1 + atten_mask1, dim=-1)  # affinity matrix

        z_out1 = einsum('b p h i j, b p h j d -> b p h i d', attn_z1, z_v1)
        z_out1 = rearrange(z_out1, 'b p h n d -> b p n (h d)')
        z_out1 = self.to_out_z1(z_out1)
        x1, y1 = z_out1[:, :, :2, :].chunk(2, dim=2)
        x1, y1 = x1.squeeze(dim=-2), y1.squeeze(dim=-2)

        return x1, y1





def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = windows.shape[0]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class RelationModel_side_double(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super(RelationModel_side_double, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv_z = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_z = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.to_qkv_z1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_z1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, s):
        b, p, d = x.shape
        h = self.heads
        s = s.detach()
        # B, C, H, W = f.shape
        s[s > 0.5] = 1
        s[s <= 0.5] = 1e-8
        v_s_sum = repeat(torch.sum(s.reshape(b, p), dim=1, keepdim=True), 'b () -> b d', d=d)
        # print('*******************')
        # for pp in range(p):
        #     print(v_s_sum[0,pp])
        # print('*******************')

        rgb_guidence = torch.sum(torch.mul(x, s.reshape(b, p,1)).reshape(b, p, d), dim=1) / (v_s_sum + 1e-8)
        # v_ns = torch.sum(torch.mul(f, m2).reshape(B, C, H * W), dim=2) / (v_ns_sum + 1e-8)
        # print(v_s_sum)
        # exit(1)
        # rgb_guidence = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=p)

        rgb_guidence = rgb_guidence.reshape(b, 1, d).expand(b, p, d)

        depth_guidence = torch.sum(torch.mul(y, s.reshape(b, p,1)).reshape(b, p, d), dim=1) / (v_s_sum + 1e-8)
        depth_guidence = depth_guidence.reshape(b, 1, d).expand(b, p, d)

        z = torch.stack([x, y, rgb_guidence, depth_guidence], dim=2)  # [b, p, 4, d]
        z_qkv = self.to_qkv_z(z).chunk(3, dim=-1)  # List: 3 * [b, p, 4, d']

        z_q, z_k, z_v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv)
        dots_z = einsum('b p h i d, b p h j d -> b p h i j', z_q, z_k) * self.scale
        atten_mask = torch.tensor([[0., 0., 0., -100.],
                                   [0., 0., -100., 0.],
                                   [0., -100., 0., 0.],
                                   [-100., 0., 0., 0.]], requires_grad=False).cuda()
        attn_z = F.softmax(dots_z + atten_mask, dim=-1)  # affinity matrix

        z_out = einsum('b p h i j, b p h j d -> b p h i d', attn_z, z_v)
        z_out = rearrange(z_out, 'b p h n d -> b p n (h d)')
        z_out = self.to_out_z(z_out)
        x, y = z_out[:, :, :2, :].chunk(2, dim=2)
        x, y = x.squeeze(dim=-2), y.squeeze(dim=-2)

        z1 = z_out  # [b, p, 4, d]
        z_qkv1 = self.to_qkv_z1(z1).chunk(3, dim=-1)  # List: 3 * [b, p, 4, d']

        z_q1, z_k1, z_v1 = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv1)
        dots_z1 = einsum('b p h i d, b p h j d -> b p h i j', z_q1, z_k1) * self.scale
        atten_mask1 = torch.tensor([[0., -100., 0., -100.],
                                   [-100., 0., -100., 0.],
                                    [-100., -100., -100., -100.],
                                    [-100., -100., -100., -100.]], requires_grad=False).cuda()
        attn_z1 = F.softmax(dots_z1 + atten_mask1, dim=-1)  # affinity matrix

        z_out1 = einsum('b p h i j, b p h j d -> b p h i d', attn_z1, z_v1)
        z_out1 = rearrange(z_out1, 'b p h n d -> b p n (h d)')
        z_out1 = self.to_out_z1(z_out1)
        x1, y1 = z_out1[:, :, :2, :].chunk(2, dim=2)
        x1, y1 = x1.squeeze(dim=-2), y1.squeeze(dim=-2)

        return x1, y1


class PreNorm(nn.Module):
    def __init__(self, dim, fn, dual=False):
        super().__init__()
        self.dual = dual
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if dual else None
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        if self.dual:
            y = args[0]
            return self.fn(self.norm1(x), self.norm2(y), *args[1:], **kwargs)

        return self.fn(self.norm1(x), *args, **kwargs)



class PointFusion(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, RelationModel_double(dim, heads, dim_head, dropout=dropout), dual=True),
                PreNorm(dim, Mlp(dim, dim)),
                PreNorm(dim, Mlp(dim, dim))
            ]))
        self.fusion = PreNorm(dim * 2, nn.Linear(dim * 2, dim))

    def forward(self, x, y):
        for rm, mlp1, mlp2 in self.layers:
            rm_x, rm_y = rm(x, y)
            rm_x, rm_y = x + rm_x, y + rm_y
            mlp_x = mlp1(rm_x) + rm_x
            mlp_y = mlp2(rm_y) + rm_y
        out = torch.cat([mlp_x, mlp_y], dim=-1)
        out = self.fusion(out)

        return out






class PointFusion_side(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, RelationModel_side_double(dim, heads, dim_head, dropout=dropout), dual=True),
                PreNorm(dim, Mlp(dim, dim)),
                PreNorm(dim, Mlp(dim, dim))
            ]))
        self.fusion = PreNorm(dim * 2, nn.Linear(dim * 2, dim))

    def forward(self, x, y, s):
        for i,(rm, mlp1, mlp2) in enumerate(self.layers):

            rm_x, rm_y = rm(x, y,s)
            rm_x, rm_y = x + rm_x, y + rm_y
            mlp_x = mlp1(rm_x) + rm_x
            mlp_y = mlp2(rm_y) + rm_y
        out = torch.cat([mlp_x, mlp_y], dim=-1)
        out = self.fusion(out)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 128)
    y = torch.randn(1, 3, 128)
    # model = CrossTransformer(128, 2, 4, 64, 64, 128)
    # out = model(x, y)
