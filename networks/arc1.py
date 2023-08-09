import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import resnet
from .classifier import Classifier_Module
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        # self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)
        self.flow_make = nn.Conv2d(inplane , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, m2):
        size = x.size()[2:]
        m2 = self.down(m2)
        flow = self.flow_make(m2)
        seg_flow_warp = self.flow_warp(x, flow, size)
        return seg_flow_warp, flow

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class DeeplabV2(nn.Module):
    def __init__(self, backbone, aspp, hidden_layers, num_classes):
        super(DeeplabV2, self).__init__()
        self.backbone = backbone
        self.aspp = aspp
        Norm2d = nn.BatchNorm2d
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)

        self.edge_out = nn.Sequential(
            nn.Conv2d(hidden_layers, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False)
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(hidden_layers, 128, kernel_size=3, padding=1, bias=False),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final_seg, self.edge_out)

    def forward(self, x, get_flow=False):
        x_size = x.size()
        backbone_out = self.backbone(x)
        x = backbone_out["feature"]
        m2 = backbone_out["low_level"]
        fine_size = m2.size()
        aspp_out = self.aspp(x)

        aspp_out = Upsample(aspp_out, fine_size[2:])       
        warped_feature, flow = self.squeeze_body_edge(aspp_out, m2)

        # seg_out = torch.cat([aspp_out, warped_feature],dim=1)
        seg_final = self.final_seg(warped_feature)

        seg_edge = warped_feature - aspp_out
        seg_edge_out = self.edge_out(seg_edge)
        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        
        if not get_flow:
            return seg_final, seg_edge_out
        else:
            return seg_final, seg_edge_out, flow

    def optim_parameters(self, lr):
        l1 = []
        for param in self.backbone.parameters():
            if param.requires_grad:
                l1.append(param)
        for param in self.squeeze_body_edge.parameters():
            if param.requires_grad:
                l1.append(param)
        for param in self.edge_out.parameters():
            if param.requires_grad:
                l1.append(param)
        for param in self.final_seg.parameters():
            if param.requires_grad:
                l1.append(param)

        l2 = []
        for param in self.aspp.parameters():
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

    return_layers = {"layer4": "feature", "layer1": "low_level"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    hidden_layers = 128
    aspp = Classifier_Module([6, 12, 18, 24], [6, 12, 18, 24], hidden_layers, 2048)

    model = DeeplabV2(backbone, aspp, hidden_layers, cfg.dataset.num_classes)
    return model