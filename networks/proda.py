import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import resnet
from .classifier import Classifier_Module
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out

class DeeplabV2(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeeplabV2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)["feature"]
        x = self.classifier(x)
        return x

    def optim_parameters(self, lr):
        l1 = []
        for param in self.backbone.parameters():
            if param.requires_grad:
                l1.append(param)
        l2 = []
        for param in self.classifier.parameters():
            if param.requires_grad:
                l2.append(param)

        return [{"params": l1, "lr": lr}, {"params": l2, "lr": 10 * lr}]

def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def Res_Deeplab(cfg):
    if cfg.freeze_bn:
        bn_layer = FrozenBatchNorm2d
    else:
        #none is BN layer in torchvision
        bn_layer = None
    backbone = resnet.__dict__[cfg.backbone](
        pretrained=True, replace_stride_with_dilation=[False, True, True], norm_layer=bn_layer
    )

    return_layers = {"layer4": "feature"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = Classifier_Module2(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.dataset.num_classes)

    model = DeeplabV2(backbone, classifier)
    if cfg.freeze_bn:
        model.apply(freeze_bn_func)
    return model