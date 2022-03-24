# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from cgitb import reset
import math
from numbers import Integral

import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D
from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec

import numpy as np
# from paddle.nn import Conv2D, MaxPool2D, BatchNorm, Linear
from paddle.nn import Conv2D, MaxPool2D, BatchNorm2D, Linear ,ReLU
from einops import rearrange




__all__ = ['BotNet', 'Bot5Head', 'Blocks', 'BasicBlock', 'BottleNeck']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}



def expand_dim(t, dim, k):
    """
    Expand dims for t at dim to k
    """
    t = t.unsqueeze(axis=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return paddle.expand(t, expand_shape)


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = paddle.zeros([B, Nh, L, 1])
    x = paddle.concat([x, col_pad], axis=3)
    flat_x = x.reshape([B, Nh, L * 2 * L])
    flat_pad = paddle.zeros([B, Nh, L - 1])
    flat_x = paddle.concat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements
    final_x = flat_x.reshape([B, Nh, L + 1, 2 * L - 1])
    return final_x[:, :, :L, L - 1 :]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    """
    B, Nh, H, W, _ = q.shape
    rel_logits = paddle.matmul(q, rel_k.transpose((1,0)))
    # Collapse height and heads
    rel_logits = rel_logits.reshape([-1, Nh * H, W, 2 * W - 1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = rel_logits.reshape([-1, Nh, H, W, W])
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class RelPosEmb(nn.Layer):
    '''Relative position encoding'''
    def __init__(self,
                 height,
                 width,
                 dim_head):
        super().__init__()

        scale = dim_head ** -0.5
        self.height = height
        self.width = width
        h_shape = [height * 2 - 1, dim_head]
        w_shape = [width * 2 - 1, dim_head]
        self.rel_height = paddle.create_parameter(
            shape=h_shape, dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(h_shape)*scale)
        )
        self.rel_width = paddle.create_parameter(
            shape=w_shape, dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(w_shape)*scale)
        )

    def forward(self, q):

        H = self.height
        W = self.width
        B, N, _, D = q.shape
        q = q.reshape([B, N, H, W, D]) # "B N (H W) D -> B N H W D"
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.transpose(perm=[0, 1, 2, 4, 3, 5])
        B, N, X, I, Y, J = rel_logits_w.shape
        rel_logits_w = rel_logits_w.reshape([B, N, X*Y, I*J]) # "B N X I Y J-> B N (X Y) (I J)"

        q = q.transpose(perm=[0, 1, 3, 2, 4]) # "B N H W D -> B N W H D"
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.transpose(perm=[0, 1, 4, 2, 5, 3])
        B, N, X, I, Y, J = rel_logits_h.shape
        rel_logits_h = rel_logits_h.reshape([B, N, Y*X, J*I]) # "B N X I Y J -> B N (Y X) (J I)"

        return rel_logits_w + rel_logits_h



class MHSA(nn.Layer):
    '''Multi-Head Self-Attention'''
    def __init__(self,
                 dim,
                 fmap_size,
                 heads=4,
                 dim_qk=128,
                 dim_v=128):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()

        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2D(dim, out_channels_qk * 2, 1, bias_attr=False)
        self.to_v = nn.Conv2D(dim, out_channels_v, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=-1)

        height, width = fmap_size
        self.pos_emb = RelPosEmb(height, width, dim_qk)

    def transpose_multihead(self, x):
        B, N, H, W = x.shape
        x = x.reshape([B, self.heads, -1, H, W]) # "B (h D) H W -> B h D H W"
        x = x.transpose(perm=[0, 1, 3, 4, 2])    # "B h D H W -> B h H W D"
        x = x.reshape([B, self.heads, H*W, -1])  # "B h H W D -> B h (H W) D"
        return x

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, axis=1)
        v = self.to_v(featuremap)
        q, k, v = map(self.transpose_multihead, [q, k, v])
        q *= self.scale

        logits = paddle.matmul(q, k.transpose(perm=[0, 1, 3, 2]))
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = paddle.matmul(weights, v)
        a_B, a_N, a_, a_D = attn_out.shape
        attn_out = attn_out.reshape([a_B, a_N, H, -1, a_D])  # "B N (H W) D -> B N H W D"
        attn_out = attn_out.transpose(perm=[0, 1, 4, 2, 3])  # "B N H W D -> B N D H W"
        attn_out = attn_out.reshape([a_B, a_N*a_D, H, -1]) # "B N D H W -> B (N D) H W"
        return attn_out















class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2
        if not self.dcn_v2:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=False)
        else:
            self.offset_channel = 2 * filter_size**2
            self.mask_channel = filter_size**2

            self.conv_offset = nn.Conv2D(
                in_channels=ch_in,
                out_channels=3 * filter_size**2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                weight_attr=ParamAttr(initializer=Constant(0.)),
                bias_attr=ParamAttr(initializer=Constant(0.)))
            self.conv = DeformConv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=False)

        norm_lr = 0. if freeze_norm else lr
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)

        global_stats = True if freeze_norm else False
        if norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        else:
            self.norm = nn.BatchNorm(
                ch_out,
                act=None,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset, mask = paddle.split(
                offset_mask,
                num_or_sections=[self.offset_channel, self.mask_channel],
                axis=1)
            mask = F.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


class SELayer(nn.Layer):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(
            ch,
            c_,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(
            c_,
            ch,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        scale = out * inputs
        return scale


class BasicBlock(nn.Layer):

    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BasicBlock, self).__init__()
        assert dcn_v2 is False, "Not implemented yet."
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


class BottleNeck(nn.Layer):

    expansion = 4

    def __init__(self,
                 x,
                 y,
                 z,
                 ch_in,    # 1024
                 ch_out,      #512  
                 stride,       #
                 shortcut, 
                 img_size=(640,640),
                 variant='b',
                 groups=1,
                 base_width=4,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride
        # ResNeXt
        width = int(ch_out * (base_width / 64.)) * groups
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out * self.expansion,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out * self.expansion,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width,
            filter_size=1,
            stride=stride1,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)
        
        if x and y:
            self.branch2b = paddle.nn.LayerList()
            self.branch2b.append(MHSA(512,[40, 40],4,128,128))
            self.branch2b.append(paddle.nn.AvgPool2D(2, 2))
            self.branch2b = paddle.nn.Sequential(*self.branch2b)
        elif x and not y:
            if z==0 :
                self.branch2b = MHSA(512,[20, 20],4,128,128)
            elif z==1:
                self.branch2b = ConvNormLayer(
                ch_in=width,
                ch_out=width,
                filter_size=3,
                stride=stride2,
                groups=groups,
                act='relu',
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                lr=lr,
                dcn_v2=True)
        else:   
            self.branch2b = ConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            stride=stride2,
            groups=groups,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)
            


        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * self.expansion,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)

        return out


class Blocks(nn.Layer):
    def __init__(self,
                 block,
                 x,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 img_size = (640,640),
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            if i == 0:
                y = 1
                z = 0
            else:
                if i==count-1:
                    y = 0
                    z = 1
                else:
                    y = 0
                    z = 0
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            layer = self.add_sublayer(
                conv_name,
                block(
                    x,
                    y,
                    z,
                    ch_in=ch_in,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    img_size=img_size,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=dcn_v2,
                    std_senet=std_senet))
            self.blocks.append(layer)
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out


@register
@serializable
class BotNet(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 img_size = [20,20],
                 ch_in=64,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],   # [3]
                 num_stages=4,
                 std_senet=False):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385
        
        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            ch_in (int): output channel of first stage, default 64
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model 
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            return_idx (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            num_stages (int): total num of stages
            std_senet (bool): whether use senet, default True
        """
        super(BotNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
        assert num_stages >= 1 and num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)
        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [1, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[1, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_sublayer(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))
        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        self.res_layers = []
        for i in range(num_stages):
            if i == 3 :
                x = 1
            else :
                x = 0
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    block,
                    x,
                    self.ch_in,
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    img_size = img_size,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages),
                    std_senet=std_senet))
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image'][:,0,:,:]

        # x = inputs['image']

        x = paddle.unsqueeze(x,axis=1)
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == self.freeze_at:
                x.stop_gradient = True
            if idx in self.return_idx:
                outs.append(x)
        return outs


@register
class Bot5Head(nn.Layer):
    def __init__(self, depth=50):
        super(Bot5Head, self).__init__()
        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(
            block, feat_in, feat_out, count=3, name_adapter=na, stage_num=5)
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.feat_out,
            stride=16, )]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y

