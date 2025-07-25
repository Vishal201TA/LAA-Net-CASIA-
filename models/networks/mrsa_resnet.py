# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool2d
import torch.utils.model_zoo as model_zoo

from ..builder import MODELS, build_model
from .common import (
    BN_MOMENTUM,
    conv_block,
    point_wise_block,
    InceptionBlock,
)


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    @staticmethod
    def __repr__():
        return "BasicBlock"


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    @staticmethod
    def __repr__():
        return "Bottleneck"


@MODELS.register_module()
class PoseResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        heads,
        head_conv,
        dropout_prob,
        fpn=False,
        cls_based_hm=True,
        use_c2=False,
        **kwargs,
    ):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        self.fpn = fpn
        self.cls_based_hm = cls_based_hm
        self.use_c2 = use_c2

        # Convert Cls name into Cls Object
        if isinstance(block, str):
            for bl in [BasicBlock, Bottleneck]:
                if block == bl.__repr__():
                    block = bl

        for k, v in kwargs.items():
            if v is None:
                raise ValueError(
                    f"The {k} argument receive a None value, Please check!"
                )
            self.__setattr__(k, v)

        super(PoseResNet, self).__init__()
        self.num_layers = kwargs.get("num_layers", None)
        if self.num_layers is None:
            raise ValueError("num_layers must be set via config (MODEL.num_layers).")

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Custom dropout layer
        self.dropout_layer = nn.Dropout(dropout_prob)

        if self.fpn:
            # Adding sidmoid layer
            self.sigmoid_layer = nn.Sigmoid()

            # Adding pointwise block
            self.pw_block_1 = self._point_wise_block(2048, 1024)

        # used for deconv layers
        deconv_filters = [256, 128, 256] if self.fpn else [256, 256, 256]
        self.deconv_layers = self._make_deconv_layer(
            3,
            deconv_filters,
            [4, 4, 4],
        )

        # Adding inception block
        if self.fpn:
            for idx, deconv_layer in enumerate(self.deconv_layers):
                self.__setattr__(f"deconv_layer_{idx}", nn.Sequential(deconv_layer))
            self.pw_block_2 = self._point_wise_block(512, 512)
            if self.use_c2:
                self.pw_block_3 = self._point_wise_block(512, 256)
            self.pw_block_c3 = self._point_wise_block(1024, 256)
            self.pw_block_c2 = self._point_wise_block(512, 128)
            self.inception_block = InceptionBlock(256, 256, stride=1, pool_size=3)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                if head != "cls":
                    fc = nn.Sequential(
                        nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(head_conv),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            head_conv, num_output, kernel_size=1, stride=1, padding=0
                        ),
                    )
                else:
                    if self.cls_based_hm:
                        pooled_size = head_conv // 4
                        fc = nn.Sequential(
                            nn.AdaptiveMaxPool2d(pooled_size),
                            nn.Flatten(),
                            nn.Linear(
                                256
                                * (pooled_size**2),  
                                head_conv,
                                bias=True,
                            ),
                            nn.BatchNorm1d(head_conv, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                            nn.Linear(head_conv, 1, bias=True),
                            nn.Sigmoid(),
                        )
                    else:
                        fc = nn.Sequential(
                            nn.Conv2d(
                                256, head_conv, kernel_size=3, padding=1, bias=True
                            ),
                            nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(head_conv, 1, bias=True),
                        )
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            self.__setattr__(head, fc)

    def _point_wise_block(self, inplanes, outplanes):
        self.inplanes = outplanes
        module = point_wise_block(inplanes, outplanes)
        return module

    def _conv_block(self, inplanes, outplanes, kernel_size, stride=1):
        self.inplanes = outplanes
        module = conv_block(inplanes, outplanes, kernel_size=kernel_size, stride=stride)
        return module

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(
            num_filters
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            num_kernels
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.inplanes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                )
            )
            if not self.fpn:
                layers.append(nn.ReLU(inplace=True))

            self.inplanes = planes if not self.fpn else planes * 2

        if self.fpn:
            return layers
        else:
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)  # 256 x 64 x 64
        x2 = self.layer2(x1)  # 512 x 32 x 32
        x3 = self.layer3(x2)  # 1024 x 16 x 16
        x4 = self.layer4(x3)  # 2048 x 8 x 8

        # Custom dropout layer
        x = self.dropout_layer(x4)  # B x 8 x 8 x 2048
        x3 = self.dropout_layer(x3)
        x2 = self.dropout_layer(x2)
        x1 = self.dropout_layer(x1)

        # Custom FPN
        if self.fpn:
            assert isinstance(
                self.deconv_layers, list
            ), "To custom FPN, decompose deconv layers as a list!"
            x = self.pw_block_1(x)  # B x 1024 x 8 x 8
            x = self.deconv_layer_0(x)  # B x 256 x 16 x 16
            # x = self.relu(x) # B x 256 x 16 x 16

            x_weighted = self.sigmoid_layer(x)  # B x 256 x 16 x 16
            x_inverse = torch.sub(1, x_weighted, alpha=1)  # B x 256 x 16 x 16
            x3 = self.pw_block_c3(x3)  # B x 256 x 16 x 16
            x3_ = torch.multiply(x3, x_inverse)  # B x 256 x 16 x 16
            x = torch.cat((x, x3_), dim=1)  # B x 512 x 16 x 16

            x = self.pw_block_2(x)  # B x 512 x 16 x 16
            x = self.deconv_layer_1(x)  # B x 128 x 32 x 32
            # x = self.relu(x) #B x 128 x 32 x 32

            x_weighted = self.sigmoid_layer(x)  # B x 128 x 32 x 32
            x_inverse = torch.sub(1, x_weighted, alpha=1)  # B x 128 x 32 x 32
            x2 = self.pw_block_c2(x2)
            x2_ = torch.multiply(x2, x_inverse)  # B x 128 x 32 x 32
            x = torch.cat((x, x2_), dim=1)  # B x 256 x 32 x 32

            x = self.inception_block(x)  # B x 256 x 64 x 64
            x = self.deconv_layer_2(x)  # B x 256 x 64 x 64

            if self.use_c2:
                x_weighted = self.sigmoid_layer(x)
                x_inverse = torch.sub(1, x_weighted, alpha=1)
                x1_ = torch.multiply(x1, x_inverse)
                x = torch.cat((x, x1_), dim=1)
                x = self.pw_block_3(x)
            else:
                x = self.relu(x)  # B x 256 x 64 x 64
        else:
            assert isinstance(
                self.deconv_layers, nn.Module
            ), "Deconv Layer must be nn Module to compute!"
            x = self.deconv_layers(x)

        ret = {}
        x1_hm = None
        for head in self.heads:
            if self.cls_based_hm and head == "cls" and x1_hm is not None:
                x = x1_hm
            elif head == "hm":
                x1_hm = x

            ret[head] = self.__getattr__(head)(x)

        return [ret]

    def init_weights(self, pretrained=True, **kwargs):
        num_layers = getattr(self, "num_layers", None)
        if pretrained:
            if self.fpn:
                for bl in [self.pw_block_1, self.pw_block_2]:
                    for _, l in bl.named_parameters():
                        if isinstance(l, nn.Conv2d):
                            nn.init.normal_(l.weight, std=0.001)
                            nn.init.constant_(l.bias, 0)

                for _, l in self.inception_block.named_parameters():
                    if isinstance(l, nn.Conv2d):
                        nn.init.normal_(l.weight, std=0.001)
                        nn.init.constant_(l.bias, 0)

            if isinstance(self.deconv_layers, nn.Module):
                for _, m in self.deconv_layers.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        nn.init.normal_(m.weight, std=0.001)
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            else:
                for layer in [
                    self.deconv_layer_0,
                    self.deconv_layer_1,
                    self.deconv_layer_2,
                ]:
                    for _, m in layer.named_modules():
                        if isinstance(m, nn.ConvTranspose2d):
                            nn.init.normal_(m.weight, std=0.001)
                            if self.deconv_with_bias:
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.BatchNorm2d):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)

            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        if m.weight.shape[0] == self.heads[head]:
                            if "hm" in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)

            # Load pretrained model from torchvision
            if num_layers is None:
                raise ValueError(
                    "num_layers is not defined. Ensure it's set in the config and passed to the model."
                )

            resnet_key = f"resnet{num_layers}"
            if resnet_key not in model_urls:
                raise KeyError(
                    f"{resnet_key} is not a valid ResNet. Choose from: {list(model_urls.keys())}"
                )

            url = model_urls[resnet_key]
            pretrained_state_dict = model_zoo.load_url(url)
            print("=> loading pretrained model {}".format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print("=> imagenet pretrained model does not exist")
            raise ValueError("imagenet pretrained model does not exist")


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}
