import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


class SiLU(nn.Layer):
    @staticmethod
    def forward(x):
        return x * F.sigmoid(x)


def mish(x):
    return x * paddle.tanh(F.softplus(x))

class BaseConv(nn.Layer):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=pad, groups=groups, bias_attr=bias)
        self.bn = nn.SyncBatchNorm(
            out_channels)
        self.act = mish

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Layer):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize,
                              stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels,
                              ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


# def _de_sigmoid(x, eps=1e-7):
#     x = paddle.clip(x, eps, 1. / eps)
#     x = paddle.clip(1. / x - 1., eps, 1. / eps)
#     x = -paddle.log(x)
#     return x


@register
class YOLOvXHead(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOvXHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format

        self.yolo_outputs = []
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        #self.iou        = nn.LayerList()
        act = 'RELU'
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)

            self.stems.append(BaseConv(in_channels=int(
                in_channels[i]), out_channels=int(in_channels[i]), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[
                BaseConv(in_channels=int(in_channels[i]), out_channels=int(
                    in_channels[i]), ksize=3, stride=1, act=act),
                BaseConv(in_channels=int(in_channels[i]), out_channels=int(
                    in_channels[i]), ksize=3, stride=1, act=act)
            ]))
            self.cls_preds.append(
                nn.Conv2D(in_channels=int(in_channels[i]), out_channels=(
                    1+self.num_classes)*3, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                BaseConv(in_channels=int(in_channels[i]), out_channels=int(
                    in_channels[i]), ksize=3, stride=1, act=act),
                BaseConv(in_channels=int(in_channels[i]), out_channels=int(
                    in_channels[i]), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2D(in_channels=int(in_channels[i]), out_channels=4 *
                          3, kernel_size=1, stride=1, padding=0)
            )
            if self.iou_aware:
                self.obj_preds.append(
                    nn.Conv2D(in_channels=int(in_channels[i]), out_channels=1 *
                              3, kernel_size=1, stride=1, padding=0)
                )
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        #-------------------------------------#
        #   feats 三层特征图
        #-------------------------------------#
        #-------------------------------------#
        #   targets
        #   im_id
        #   is_crowd
        #   gt_bbox
        #   curr_iter
        #   image
        #   im_shape
        #   scale_factor
        #   target0
        #   target1
        #   target2
        #-------------------------------------#
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x = self.stems[i](feat)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat = self.cls_convs[i](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output = self.cls_preds[i](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat = self.reg_convs[i](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output = self.reg_preds[i](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            if self.iou_aware:
                obj_output = self.obj_preds[i](reg_feat)

            if self.iou_aware:
                output = paddle.concat([obj_output, reg_output, cls_output], 1)
            else:
                output = paddle.concat([reg_output, cls_output], 1)

            yolo_outputs.append(output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
