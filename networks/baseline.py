import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import resnet
from .classifier import Classifier_Module
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d


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
    classifier = Classifier_Module([6, 12, 18, 24], [6, 12, 18, 24], cfg.dataset.num_classes, 2048)

    model = DeeplabV2(backbone, classifier)
    return model