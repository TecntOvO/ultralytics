# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn as Fa

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from einops import rearrange

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    'MV2Block',
    'MobileViTBlock',
    'SEAM',
    'MultiSEAM',
    'MobileViTBlockv2',
    'MobileViTBlockv3',
    'MobileViTBlockv4',
    'MobileViTBlockv5',
)


#####################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0., use_flash_attn=False, use_pytorch_attn=False):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scale = self.head_dim ** -0.5
        self.num_heads = num_heads
        self.dropout = dropout
        assert (use_flash_attn is False and use_pytorch_attn is False) or use_flash_attn != use_pytorch_attn
        self.use_flash_attn = use_flash_attn
        self.use_pytorch_attn = use_pytorch_attn

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(p=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        if self.use_pytorch_attn:
            out, _ = F.multi_head_attention_forward(
                query=x,
                key=x,
                value=x,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.num_heads,
                in_proj_weight=torch.empty([0]),
                in_proj_bias=self.qkv_proj.bias,
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=self.attn_drop.p,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias,
                training=self.training,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                use_separate_proj_weight=True,
                q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
                k_proj_weight=self.qkv_proj.weight[
                              self.embed_dim: 2 * self.embed_dim, ...
                              ],
                v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim:, ...],
            )
            return out
        else:
            # [B*P, N, d]
            b_sz, s_len, in_channels = x.shape

            # [B*P, N, d] --> [B*P, N, 3*d] --> [B*P, N, 3, h_n, d // h_n]
            qkv = self.qkv_proj(x).reshape(b_sz, s_len, 3, self.num_heads, -1)

            if self.use_flash_attn and x.device.type != 'cpu':
                # [B*P, N, 3, h_n, d // h_n] --> [B*P, N, h_n, d // h_n] x 3
                query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                if self.training:
                    # [B * P, N, h_n, d // h_n]
                    out = Fa.flash_attn_func(query, key, value, self.dropout, self.scale)
                else:
                    # [B * P, N, h_n, d // h_n]
                    out = Fa.flash_attn_func(query, key, value, 0.0, self.scale)
            else:
                # [B*P, N, 3, h_n, d // h_n] --> [B*P, h_n, 3, N, d // h_n]
                qkv = qkv.transpose(1, 3).contiguous()
                # [B*P, h_n, 3, N, d // h_n] --> [B*P, h_n, N, d // h_n] x 3
                query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                query = query * self.scale

                # K^T
                # [B*P, h_n, N, d // h_n] --> [B*P, h_n, d // h_n, N]
                key = key.transpose(-1, -2)

                # Q*K^T
                # [B*P, h_n, N, d // h_n] x [B*P, h_n, d // h_n, N] --> [B*P, h_n, N, N]
                attn = torch.matmul(query, key)

                attn_dtype = attn.dtype
                attn_as_float = self.softmax(attn.float())
                attn = attn_as_float.to(attn_dtype)
                attn = self.attn_drop(attn)

                # weighted sum
                # [B*P, h_n, N, N] x [B*P, h_n, N, d // h_n] --> [B*P, h_n, N, d // h_n]
                out = torch.matmul(attn, value)

                # [B*P, h_n, N, d // h_n] --> [B*P, N, h_n, d // h_n]
                out = out.transpose(1, 2)

            return self.out_proj(out.reshape(b_sz, s_len, -1))


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0., use_flash_attn=False, use_pytorch_attn=False):
        super().__init__()

        self.head_dim = embed_dim
        self.embed_dim = embed_dim
        self.scale = self.embed_dim ** -0.5
        self.dropout = dropout
        assert (use_flash_attn is False and use_pytorch_attn is False) or use_flash_attn != use_pytorch_attn
        self.use_flash_attn = use_flash_attn
        self.use_pytorch_attn = use_pytorch_attn

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(p=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        if self.use_pytorch_attn:
            out, _ = F.multi_head_attention_forward(
                query=x,
                key=x,
                value=x,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.num_heads,
                in_proj_weight=torch.empty([0]),
                in_proj_bias=self.qkv_proj.bias,
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=self.attn_drop.p,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias,
                training=self.training,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                use_separate_proj_weight=True,
                q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
                k_proj_weight=self.qkv_proj.weight[
                              self.embed_dim: 2 * self.embed_dim, ...
                              ],
                v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim:, ...],
            )
            return out
        else:
            if self.use_flash_attn and x.device.type != 'cpu':
                # [B*P, N, d]
                b_sz, s_len, in_channels = x.shape
                # [B*P, N, d] --> [B*P, N, 3*d] --> [B*P, N, 3, 1, d]
                qkv = self.qkv_proj(x).reshape(b_sz, s_len, 3, 1, in_channels)
                # [B*P, N, 3, 1, d] --> [B*P, N, 1, d] x 3
                query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                if self.training:
                    d = self.dropout
                else:
                    d = 0.0
                # [B * P, N, 1, d]
                out = Fa.flash_attn_func(query, key, value, d, self.scale)
            else:
                # [B*P, N, d] --> [B*P, N, 3*d]
                qkv = self.qkv_proj(x)
                # [B*P, N, 3*d] --> [B*P, N, d] x 3
                query, key, value = torch.chunk(qkv, 3, -1)
                query = query * self.scale

                # K^T
                # [B*P, N, d] --> [B*P, d, N]
                key = key.transpose(-1, -2)

                # Q*K^T
                # [B*P, N, d] x [B*P, d, N] --> [B*P, N, N]
                attn = torch.matmul(query, key)

                attn_dtype = attn.dtype
                attn_as_float = self.softmax(attn.float())
                attn = attn_as_float.to(attn_dtype)
                attn = self.attn_drop(attn)

                # weighted sum
                # [B*P, N, N] x [B*P, N, d] --> [B*P, N, d]
                out = torch.matmul(attn, value)

            return self.out_proj(out.squeeze())


class Transformer(nn.Module):
    def __init__(self, embed_dim, hid_dim, depth, num_heads, attn_dropout=0., ffn_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(embed_dim)
        attn_unit = SingleHeadAttention(embed_dim, attn_dropout)
        if num_heads > 1:
            attn_unit = MultiHeadAttention(embed_dim, num_heads, attn_dropout)

        ffn_unit = nn.Sequential(
            nn.Linear(embed_dim, hid_dim, True),
            nn.SiLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(hid_dim, embed_dim, True),
            nn.Dropout(p=ffn_dropout)
        )
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    attn_unit
                ),
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    ffn_unit
                )
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MobileViTBlockv4(nn.Module):
    def __init__(self, embed_dim, depth, channel, hid_dim, num_heads, patch_size, kernel_size=3,
                 attn_drop=0.0, ff_dropout=0.0):
        super().__init__()
        self.ph = patch_size
        self.pw = patch_size
        self.local_rep = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
            nn.Conv2d(channel, embed_dim, 1, 1, 0, bias=False)
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * embed_dim, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU()
        )
        self.transformer = Transformer(embed_dim, hid_dim, depth, num_heads, attn_drop, ff_dropout)

        self._initialize_weights()

    def forward(self, x):
        # local rep
        # [B, C, H, W] --> [B, d, H, W]
        x_local = x_unfold = self.local_rep(x)

        self.batch_size, in_channels, self.img_h, self.img_w = x_unfold.shape

        # Resize if necessary
        do_resize = False
        new_img_h = int(math.ceil(self.img_h / self.ph) * self.ph)
        new_img_w = int(math.ceil(self.img_w / self.pw) * self.pw)
        if self.img_h != new_img_h or self.img_w != new_img_w:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x_unfold = F.interpolate(
                x_unfold,
                size=(new_img_h, new_img_w),
                mode="bilinear",
                align_corners=False
            )
            do_resize = True

        # Unfold
        # [B, d, H, W] --> [B, d, P, N]
        patches = F.unfold(
            x_unfold,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        patches = patches.reshape(
            self.batch_size, in_channels, self.ph * self.pw, -1
        )

        # [B, d, P, N] --> [B*P, N, d]
        _, _, patch_size, n_patches = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(-1, n_patches, in_channels)

        # learn global representations on all patches
        patches = self.transformer(patches)

        # [B*P, N, d] --> [B, d, P, N]
        patches = patches.reshape(self.batch_size, self.ph * self.pw, n_patches, -1).permute(0, 3, 1, 2)

        # Fold
        # [B, d, P, N] --> [B, d, H, W]
        patches = patches.reshape(self.batch_size, in_channels * patch_size, n_patches)
        x_fold = F.fold(
            patches,
            output_size=(new_img_h, new_img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )

        if do_resize:
            x_fold = F.interpolate(
                x_fold,
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            )

        # [B, d, H, W]
        x_fold = self.conv_1x1_out(x_fold)

        # Fusion + ResConnect
        return self.fusion_conv(torch.cat([x_fold, x_local], 1)) + x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m_ in m:
                    if isinstance(m_, nn.Conv2d):
                        nn.init.kaiming_normal_(m_.weight, mode="fan_out")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")


class MobileViTBlock(nn.Module):
    # CSDN 迪菲赫尔曼
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()

        def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )

        def conv_1x1_bn(inp, oup):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )

        self.ph = patch_size
        self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)

        self.conv2 = conv_1x1_bn(channel, dim)
        # self.transformer = Transformer(dim, mlp_dim, depth, 4, dropout) #todo
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # 经过nxn卷积和PW卷积获取局部特征，又称局部表征模块
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        # Unfold
        '''
        将输入的[B, H, W, C]变为[B, P, N, d]，注意，d>C，
        P为wh（每个patch的宽高，一般为2），N为patches的数目。
        就是将feature map划分为一个个的patch，假如B=C=1，patch的宽高w=h=3，
        则可以想象将图像拆分为9列，每列中的像素就为每个patch中相同位置的提取出来的，
        所以每列有num_patches个像素。我们将每一列送入到Transformer中，
        算列间的每个像素的attention，这样就能够将感受野扩大为HxW了。
        '''
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # Local Processing
        x = self.transformer(x)
        # Fold
        '''
        将[B, P, N, d]还原成[B, H, W, d]
        '''
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        # Fusion
        '''
        用来将提取到的特征融合到一起。
        先通过一个1x1卷积将channel变为C，就是[B, H, W, C]，
        再与MobileViT block的输入concat起来，
        通过nxn的卷积层将feature融合起来，输出[B, H, W, C]。
        '''
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=True)
        )

        # self._initialize_weights()

    def forward(self, x):
        return self.ffn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m_ in m:
                    if isinstance(m_, nn.Conv2d):
                        nn.init.kaiming_normal_(m_.weight, mode="fan_out")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")


class LinearSelfAttention(nn.Module):
    """
    线性可分离自注意力方法, 具体原理参照论文 `https://arxiv.org/abs/2206.02680`
    Args:
        embed_dim (int): :math:`[B, d, P, N]` 中的维度 :math:`d`，就是输入的特征维度
        attn_drop (float): 可分离自注意力中dropout的概率. 默认值: 0.0
        proj_drop (float):
        bias (bool): 线性映射中是否使用偏置. 默认值: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    """

    def __init__(
            self,
            embed_dim: int,
            attn_drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ikv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
        )
        self.drop = nn.Dropout(p=attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.context_scores = None

        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 线性映射
        # [B, d, P, N] --> [B, 1 + 2d, P, N]
        ikv = self.ikv_proj(x)

        # 分离出inp,key和value
        # inp --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        inp, key, value = torch.split(
            ikv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # 上下文分数 --> [B, 1, P, N]
        context_scores = self.drop(F.softmax(inp, dim=-1))
        if not self.training:
            self.context_scores = context_scores.detach().clone()

        # 计算上下文向量
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        return self.out_proj(F.relu(value) * context_vector.expand_as(value))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m_ in m:
                    if isinstance(m_, nn.Conv2d):
                        nn.init.kaiming_normal_(m_.weight, mode="fan_out")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")


class LinearTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hid_dim: int,
            depth: int,
            attn_drop: float = 0.0,
            ff_dropout: float = 0.0,
            dropout: float = 0.0,
            layer_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.GroupNorm(1, embed_dim)
        self.layer_dropout = layer_dropout
        self.dropout = nn.Dropout(p=dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.GroupNorm(1, embed_dim),
                LinearSelfAttention(embed_dim, attn_drop),
                nn.GroupNorm(1, embed_dim),
                FeedForward(embed_dim, hid_dim, ff_dropout)
            ]))
        self.context_scores = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.context_scores.clear()
        skip_count = 0
        for preattn_norm, attn, preff_norm, ff in self.layers:
            if self.training and torch.rand(1).item() < self.layer_dropout:
                skip_count += 1
                continue
            else:
                x = self.dropout(attn(preattn_norm(x))) + x
                if not self.training:
                    self.context_scores.append(attn.context_scores)
                x = self.dropout(ff(preff_norm(x))) + x
        if skip_count == len(self.layers):
            return x
        else:
            return self.norm(x)


class MobileViTBlockv2(nn.Module):
    '''
    Args:
        embed_dim: :math:`[B, d, P, N]` 中的维度 :math:`d`，就是局部表征张量的通道数
        depth: 可分离自注意力头的堆叠个数
        channel: 输入通道
        hid_dim: LinearTransformer中FeedForward的隐藏层的神经元数
        patch_size: LinearTransformer一个patch的边长（图像块边长）
        kernel_size: 局部表征模块卷积核大小
        attn_drop: LinearTransformer中的上下文分数context_scores的dropout概率
        ff_dropout: LinearTransformer中FeedForward的dropout概率
        dropout: LinearTransformer中经过注意力模块后以及前馈神经网络后的dropout概率
        layer_dropout: 跳过LinearTransformer的概率
    '''

    def __init__(self, embed_dim, depth, channel, hid_dim, patch_size, kernel_size=3,
                 attn_drop=0.0, ff_dropout=0.0, dropout=0.0, layer_dropout=0.):
        super().__init__()
        self.ph = patch_size
        self.pw = patch_size
        # assert self.ph == self.pw
        self.local_rep = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
            nn.Conv2d(channel, embed_dim, 1, 1, 0, bias=False)
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(embed_dim, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel)
        )
        self.transformer = LinearTransformer(embed_dim, hid_dim, depth, attn_drop=attn_drop, ff_dropout=ff_dropout,
                                             dropout=dropout, layer_dropout=layer_dropout)
        # self._initialize_weights()

    def forward(self, x):
        # Resize if necessary
        _, _, img_h, img_w = x.shape
        Resize = False
        if img_h % self.ph != 0 or img_w % self.pw != 0:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            new_img_h = int(math.ceil(img_h / self.ph) * self.ph)
            new_img_w = int(math.ceil(img_w / self.pw) * self.pw)
            x = F.interpolate(
                x,
                size=(new_img_h, new_img_w),
                mode="bilinear",
                align_corners=False
            )
            Resize = True

        # local rep
        x = self.local_rep(x)
        self.batch_size, in_channels, self.img_h, self.img_w = x.shape

        # Unfold
        # [B, d, H, W] --> [B, d, P, N]
        patches = F.unfold(
            x,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        patches = patches.reshape(
            self.batch_size, in_channels, self.ph * self.pw, -1
        )

        # learn global representations on all patches
        patches = self.transformer(patches)

        # Fold
        # [B, d, P, N] --> [B, d, H, W]
        _, _, patch_size, n_patches = patches.shape
        patches = patches.reshape(self.batch_size, in_channels * patch_size, n_patches)
        x = F.fold(
            patches,
            output_size=(self.img_h, self.img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        if Resize:
            x = F.interpolate(
                x,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False
            )
        return self.conv_1x1_out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m_ in m:
                    if isinstance(m_, nn.Conv2d):
                        nn.init.kaiming_normal_(m_.weight, mode="fan_out")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def get_score(self):
        assert self.batch_size == 1

        context_scores = torch.cat(self.transformer.context_scores, dim=0)
        # [d, 1, P, N]
        batch_size, in_dim, patch_size, n_patches = context_scores.shape
        assert in_dim == 1
        # [d, 1, P, N] --> [d, P, N]
        patches = context_scores.reshape(batch_size, in_dim * patch_size, n_patches)
        # [d, P, N] --> [d, 1, H, W]
        y = F.fold(
            patches,
            output_size=(self.img_h, self.img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        epsilon = 1e-5
        y_min = y.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        y_max = y.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        # 计算归一化，避免除零错误
        y = (y - y_min) / (y_max - y_min + epsilon)
        y = torch.split(y.squeeze(), 1, dim=0) if y.shape[0] > 1 else (y[0])
        return y


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class FlashSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            attn_drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ikv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=3 * embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.oper_q = nn.Sequential(
            SpatialOperation(embed_dim),
            ChannelOperation(embed_dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(embed_dim),
            ChannelOperation(embed_dim),
        )
        self.dwc = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.ikv_proj(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        return self.proj_drop(out)


class FlashTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hid_dim: int,
            depth: int,
            attn_drop: float = 0.0,
            ff_dropout: float = 0.0,
            dropout: float = 0.0,
            layer_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.GroupNorm(1, embed_dim)
        self.layer_dropout = layer_dropout
        self.dropout = nn.Dropout(p=dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.GroupNorm(1, embed_dim),
                FlashSelfAttention(embed_dim, attn_drop),
                nn.GroupNorm(1, embed_dim),
                FeedForward(embed_dim, hid_dim, ff_dropout)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for preattn_norm, attn, preff_norm, ff in self.layers:
            x = self.dropout(attn(preattn_norm(x))) + x
            x = self.dropout(ff(preff_norm(x))) + x
        return self.norm(x)


class MobileViTBlockv5(nn.Module):
    def __init__(self, embed_dim, depth, channel, hid_dim, patch_size, kernel_size=3,
                 attn_drop=0.0, ff_dropout=0.0, dropout=0.0, layer_dropout=0.):
        super().__init__()
        self.ph = patch_size
        self.pw = patch_size
        # assert self.ph == self.pw
        self.local_rep = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
            nn.Conv2d(channel, embed_dim, 1, 1, 0, bias=False)
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(embed_dim, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel)
        )
        self.transformer = LinearTransformer(embed_dim, hid_dim, depth, attn_drop=attn_drop, ff_dropout=ff_dropout,
                                             dropout=dropout, layer_dropout=layer_dropout)
        self._initialize_weights()

    def forward(self, x):
        # Resize if necessary
        _, _, img_h, img_w = x.shape
        resize = False
        if img_h % self.ph != 0 or img_w % self.pw != 0:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            new_img_h = int(math.ceil(img_h / self.ph) * self.ph)
            new_img_w = int(math.ceil(img_w / self.pw) * self.pw)
            x = F.interpolate(
                x,
                size=(new_img_h, new_img_w),
                mode="bilinear",
                align_corners=False
            )
            resize = True

        # local rep
        x = self.local_rep(x)
        self.batch_size, in_channels, self.img_h, self.img_w = x.shape

        # Unfold
        # [B, d, H, W] --> [B, d, P, N]
        patches = F.unfold(
            x,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        patches = patches.reshape(
            self.batch_size, in_channels, self.ph * self.pw, -1
        )

        # learn global representations on all patches
        patches = self.transformer(patches)

        # Fold
        # [B, d, P, N] --> [B, d, H, W]
        _, _, patch_size, n_patches = patches.shape
        patches = patches.reshape(self.batch_size, in_channels * patch_size, n_patches)
        x = F.fold(
            patches,
            output_size=(self.img_h, self.img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        if resize:
            x = F.interpolate(
                x,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False
            )
        return self.conv_1x1_out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m_ in m:
                    if isinstance(m_, nn.Conv2d):
                        nn.init.kaiming_normal_(m_.weight, mode="fan_out")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")


class MobileViTBlockv3(nn.Module):
    '''
    Args:
        embed_dim: :math:`[B, d, P, N]` 中的维度 :math:`d`，就是局部表征张量的通道数
        depth: 可分离自注意力头的堆叠个数
        channel: 输入通道
        hid_dim: LinearTransformer中FeedForward第一个卷积后的通道数
        patch_size: Transformer一个patch的边长
        kernel_size: 局部表征模块卷积核大小
        dropout: LinearTransformer中FeedForward的dropout概率
    '''

    def __init__(self, embed_dim, depth, channel, hid_dim, patch_size, kernel_size=3,
                 attn_drop=0.0, ff_dropout=0.0, dropout=0.1, layer_dropout=0.):
        super().__init__()
        self.ph = patch_size
        self.pw = patch_size
        self.local_rep = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False, groups=channel),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
            nn.Conv2d(channel, embed_dim, 1, 1, 0, bias=False)
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(2 * embed_dim, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU()
        )
        self.norm_out = nn.BatchNorm2d(channel)
        self.transformer = LinearTransformer(embed_dim, hid_dim, depth, attn_drop=attn_drop, ff_dropout=ff_dropout,
                                             dropout=dropout, layer_dropout=layer_dropout)

    def forward(self, x):
        # 经过nxn卷积和PW卷积获取局部特征，又称局部表征模块
        x_unfold = self.local_rep(x)

        # Unfold
        # [B, C, H, W] --> [B, d, P, N]
        self.batch_size, in_channels, self.img_h, self.img_w = x_unfold.shape
        patches = F.unfold(
            x_unfold,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        patches = patches.reshape(
            self.batch_size, in_channels, self.ph * self.pw, -1
        )

        # learn global representations on all patches
        patches = self.transformer(patches)

        # Fold
        # [B, d, P, N] --> [B, C, H, W]
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        x_flod = F.fold(
            patches,
            output_size=(self.img_h, self.img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        # Fusion block
        return self.norm_out(self.conv_1x1_out(torch.cat((x_flod, x_unfold), 1)) + x)

    def get_score(self):
        assert self.batch_size == 1

        context_scores = torch.cat(self.transformer.context_scores, dim=0)
        # [d, 1, P, N]
        batch_size, in_dim, patch_size, n_patches = context_scores.shape
        assert in_dim == 1
        # [d, 1, P, N] --> [d, P, N]
        patches = context_scores.reshape(batch_size, in_dim * patch_size, n_patches)
        # [d, P, N] --> [d, 1, H, W]
        y = F.fold(
            patches,
            output_size=(self.img_h, self.img_w),
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw)
        )
        epsilon = 1e-5
        y_min = y.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        y_max = y.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        # 计算归一化，避免除零错误
        y = (y - y_min) / (y_max - y_min + epsilon)
        y = torch.split(y.squeeze(), 1, dim=0)
        return y


class ConvMixer(nn.Module):
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=3, reduction=16):
        super(ConvMixer, self).__init__()
        if c2 != c1:
            c2 = c1
        self.DConvN = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(c2, c2, kernel_size, groups=c2, padding=1),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(c2, c1, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(depth)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DConvN(x)
        y = self.avg_pool(y).view(b, c)
        y_ = self.avg_pool(x).view(b, c)
        y = self.fc((y + y_) / 2.0).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SEAM(nn.Module):
    def __init__(self, c1, c2, n=1, reduction=16):
        super(SEAM, self).__init__()
        if c2 != c1:
            c2 = c1
        self.DCovN = nn.Sequential(
            *[nn.Sequential(
                # 深度可分离卷积
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                # 逐点卷积进行多通道融合
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)


class MultiSEAM(nn.Module):
    def __init__(self, c1, depth, kernel_size=3, patch_size=[6, 7, 8], reduction=16):
        super(MultiSEAM, self).__init__()
        c2 = c1

        def DcovN(c1, c2, depth=1, kernel_size=3, patch_size=3):
            return nn.Sequential(
                # Patch Embedding
                nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(c2),
                *[nn.Sequential(
                    # 深度可分离卷积
                    Residual(nn.Sequential(
                        nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1,
                                  groups=c2),
                        nn.GELU(),
                        nn.BatchNorm2d(c2)
                    )),
                    # 逐点卷积
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                ) for i in range(depth)]
            )

        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)

        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y3 = self.avg_pool(x).view(b, c)

        y = (y0 + y1 + y2 + y3) / 4.0
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


#####################################################################


class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride,
                                   padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)
        return base_feat


class RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n):
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        # self.cv = Conv(c2, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out


class C3RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        # self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.rfem = RFEM(c_, c_, n)
        self.m = nn.Sequential(*[RFEM(c_, c_, n=1, e=e) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class C2f_attention(C2f):
    def __init__(self, c1, c2, depth=1, e=0.5, g=1, shortcut=True, patch_size=2):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, 1, shortcut, g, e)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = MobileViTBlockv2(self.c // 2, depth, self.c, self.c, patch_size)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.m(y[-1])
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
