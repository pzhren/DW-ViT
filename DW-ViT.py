# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if len(x.shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    elif len(x.shape) == 5:
        _, B, H, W, C = x.shape
        # print(x.shape, window_size) #torch.Size([3, 42, 56, 56, 32])
        x = x.view(3, B, H // window_size, window_size, W // window_size, window_size, C) 
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(3, -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SKConv(nn.Module):
    def __init__(self, dim, M, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        self.dim = dim
        self.channel = dim // M  
        assert dim == self.channel * M
        self.d = self.channel // r  
        self.M = M
        self.proj = nn.Linear(dim,dim) 

        self.act_layer = act_layer()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(dim,self.d) 
        self.fc2 = nn.Linear(self.d, self.M*self.channel)
        self.softmax = nn.Softmax(dim=1)
        self.proj_head = nn.Linear(self.channel, dim)

    def forward(self, input_feats):
        bs, H, W, _ = input_feats.shape
        input_groups = input_feats.permute(0,3,1,2).reshape(bs,self.M, self.channel, H, W)
        feats = self.proj(input_feats.view(bs,H*W,-1)) #[bs,H*W,dim]
        feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,H,W)
        feats = self.act_layer(feats)
        feats = feats.permute(0,2,1).reshape(bs,self.dim,H,W)
        feats_S = self.gap(feats) #[bs,dim,1,1]
        feats_Z = self.fc1(feats_S.squeeze()) #[bs,d]
        feats_Z = self.act_layer(feats_Z) #[bs,d]
        attention_vectors = self.fc2(feats_Z) #[bs,M*channel]
        attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(input_groups * attention_vectors, dim=1) #[bs,channel,H,W]
        feats_V = self.proj_head(feats_V.reshape(bs,self.channel,H*W).permute(0,2,1)) #[bs,H*W,dim]
        feats_V = feats_V.permute(0,2,1).reshape(bs,self.dim,H,W)
        output = feats_proj + feats_V #[bs,dim,H,W]
        return output

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        flops = 0
        # self.proj
        flops += H*W * self.dim * self.dim
        # self.fc1
        flops += self.dim * self.d
        # self.fc2
        flops += self.d * self.channel*self.M
        # feats_V
        flops += self.M * self.channel * H * W
        # self.proj_head
        flops += H * W * self.channel * self.dim
        return flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, shift_size, num_heads, act_layer,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww

        self.n_group = len(self.window_size)
        self.channel = self.dim // self.n_group  
        assert self.dim == self.channel * self.n_group
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.gnum_heads = num_heads // len(self.window_size)  
        assert num_heads == self.gnum_heads * len(self.window_size)
        self.gchannel = self.channel // self.gnum_heads  
        assert self.channel == self.gchannel * self.gnum_heads
        self.qk_scale = qk_scale

        # define a parameter table of relative position bias
        # print(self.window_size)

        self.relative_position_bias_table = []
        self.relative_position_index = []
        for i, window_s in enumerate(self.window_size):
            relative_position_bias_params = nn.Parameter(
                torch.zeros((2 * window_s - 1) * (2 * window_s - 1), self.gnum_heads))
            trunc_normal_(relative_position_bias_params, std=.02)
            setattr(self,'relative_position_bias_params_{}'.format(i),relative_position_bias_params) 
            self.relative_position_bias_table.append(getattr(self,'relative_position_bias_params_{}'.format(i)))

            # get pair-wise relative position index for each token inside the window
            Window_size = to_2tuple(window_s)
            coords_h = torch.arange(Window_size[0]) #[0,1,2,3,..,window_s]
            coords_w = torch.arange(Window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += Window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += Window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * Window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # print(relative_position_index.shape)
            self.register_buffer("relative_position_index_{}".format(i), relative_position_index)
            self.relative_position_index.append(getattr(self, "relative_position_index_{}".format(i)))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.sknet = SKConv(dim=dim, M=self.n_group, act_layer=act_layer)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print('----------',self.window_size)
        # exit()
        # print(x.shape) #torch.Size([64, 56, 56, 96])
        B, H, W, C = x.shape
        x = x.view(B, -1, C)  # [B,H*W,C]
        # print(x.shape) #torch.Size([64, 3136, 96])
        B, HW, C = x.shape
        qkv = self.qkv(x).reshape(B, HW, 3, C).permute(2, 0, 1, 3)  # [3,B,HW,C]
        # print(qkv.shape) #torch.Size([3, 64, 3136, 96])
        qkv = qkv.reshape(3, B, H, W, C)

        qkv_groups = qkv.chunk(len(self.window_size), -1)  
        x_groups = []
        for i, qkv_group in enumerate(qkv_groups):
            # print(qkv_group.shape) #torch.Size([3, 42, 56, 56, 32])
            window_s = self.window_size[i]
            # print(i)
            # print(q.shape) #[64, 56, 56, 32]
            # cyclic shift

            # padding
            pad_l = pad_t = 0
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            qkv_group = F.pad(qkv_group, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, _, Hp, Wp, _ = qkv_group.shape

            # cyclic shift
            if self.shift_size[i] > 0:
                shifted_qkv_group = torch.roll(qkv_group, shifts=(-self.shift_size[i], -self.shift_size[i]),
                                               dims=(2, 3))
            else:
                shifted_qkv_group = qkv_group
            # print(shifted_qkv_group.shape) #[3, 64, 56, 56, 32]

            # partition windows
            qkv_windows = window_partition(shifted_qkv_group, window_s)  # nW*B, window_size, window_size, C
            # print(qkv_windows.shape) #torch.Size([3, 4096, 7, 7, 32])
            qkv_windows = qkv_windows.view(3, -1, window_s * window_s,
                                           self.channel)  # nW*B, window_size*window_size, C//n_group
            _, B_, N, _ = qkv_windows.shape  # [3, 9216, 25, 32]
  
            # pad_c = (self.num_heads - self.channel % self.num_heads) % self.num_heads
            qkv = qkv_windows.reshape(3, B_, N, self.gnum_heads, self.gchannel).permute(0, 1, 3, 2,
                                                                                        4)  # [3,B_,self.gnum_heads,N,self.gchannel]

            head_dim = qkv.shape[-1]
            [q, k, v] = [x for x in qkv]
            # print(q.shape) #torch.Size([B_, self.gnum_heads, N, self.gchannel])
            self.scale = self.qk_scale or head_dim ** -0.5
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            # print(attn.shape) #torch.Size([4608, 1, 25, 25])
            window_size = to_2tuple(window_s)
            # print(window_size, self.relative_position_index[i].view(-1).shape,self.relative_position_bias_table[i][self.relative_position_index[i].view(-1)].shape)
            relative_position_bias = self.relative_position_bias_table[i][
                self.relative_position_index[i].view(-1)].view(
                window_size[0] * window_size[1], window_size[0] * window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            # print(relative_position_bias.shape) #torch.Size([25, 25, 3])
            # exit()
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().cuda()  # nH, Wh*Ww, Wh*Ww

            # print(attn.shape, relative_position_bias.shape) #torch.Size([9216, gnum_heads, 25, 25]) torch.Size([gnum_heads, 25, 25])
            attn = attn + relative_position_bias.unsqueeze(0)  ##torch.Size([4608, gnum_heads, N, N])

            if mask[i] is not None:
                nW = mask[i].shape[0]
                attn = attn.view(B_ // nW, nW, self.gnum_heads, N, N) + mask[i].unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.gnum_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)
            # print(attn.shape, v.shape)
            # print((attn @ v).transpose(1, 2).shape, B_, N, self.gchannel*self.gnum_heads)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, self.channel)

            # merge windows
            x = x.view(B_, window_s, window_s, self.channel)
            shifted_x = window_reverse(x, window_s, Hp, Wp)  # B H W C

            # reverse cyclic shift
            if self.shift_size[i] > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size[i], self.shift_size[i]), dims=(1, 2))
            else:
                x = shifted_x
            # print(x.shape) #torch.Size([64, 72, 72, 32])
            # x = x.reshape(B, -1, self.channel)

            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)  # [B,H,W,C]
        x = self.sknet(x) #[B,C,H,W]
        x = x.view(B, self.dim, HW).permute(0, 2, 1) #[B,HW,C]
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += H * W * self.dim * 3 * self.dim
        for i in range(len(self.window_size)):
            window_s = self.window_size[i]
            N = window_s * window_s
            nW = math.ceil(H / window_s) * math.ceil(W / window_s)
            # attn = (q @ k.transpose(-2, -1))
            attn_flop = self.gnum_heads * N * self.gchannel * N
            #  x = (attn @ v)
            attn_v_flop = self.gnum_heads * N * N * self.gchannel
            flops += nW * (attn_flop + attn_v_flop)
        # x = self.sknet(x_groups)
        flops += self.sknet.flops(H, W)
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size.copy()
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.attn_mask = []
        for i in range(len(self.window_size)):
            if min(self.input_resolution) <= self.window_size[i]:
                # if window size is larger than input resolution, we don't partition windows
                self.shift_size[i] = 0
                self.window_size[i] = min(self.input_resolution)
            assert 0 <= self.shift_size[i] < self.window_size[i], "shift_size must in 0-window_size"
            window_s = self.window_size[i]
            if self.shift_size[i] > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                Hp = int(np.ceil(H / window_s)) * window_s
                Wp = int(np.ceil(W / window_s)) * window_s
                img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 H W 1
                h_slices = (slice(0, -window_s),
                            slice(-window_s, -self.shift_size[i]),
                            slice(-self.shift_size[i], None))
                w_slices = (slice(0, -window_s),
                            slice(-window_s, -self.shift_size[i]),
                            slice(-self.shift_size[i], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, window_s)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, window_s * window_s)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                             float(0.0)).cuda()
            else:
                attn_mask = None
            self.register_buffer("attn_mask_{}".format(i), attn_mask)
            self.attn_mask.append(getattr(self, "attn_mask_{}".format(i)))

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, shift_size=self.shift_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # W-MSA/SW-MSA
        x = self.attn(x, mask=self.attn_mask)  # B, H * W, C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"Window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

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
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 # shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 shift_size=np.zeros(len(window_size)) if (i % 2 == 0) else np.array(window_size) // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DW-ViT(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=3,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        relative_position_bias_table = set()
        for i in range(len(self.window_size)):
            relative_position_bias_table.add('relative_position_bias_table_{}'.format(i))
        return relative_position_bias_table

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
