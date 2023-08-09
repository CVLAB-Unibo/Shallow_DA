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
            nn.Conv2d(hidden_layers*2, hidden_layers, kernel_size=3, padding=1, bias=False),
            Norm2d(hidden_layers),
            nn.ReLU(inplace=True),
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
        m2 = backbone_out["layer1"]
        fine_size = m2.size()
        aspp_out = self.aspp(x)

        aspp_out = Upsample(aspp_out, fine_size[2:])       
        warped_feature, flow = self.squeeze_body_edge(aspp_out, m2)

        # seg_out = torch.cat([m2, warped_feature],dim=1)
        seg_final = self.final_seg(warped_feature)

        # seg_edge = warped_feature - aspp_out
        seg_edge_out = self.edge_out(m2)
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

    return_layers = {"layer4": "feature", "layer1": "layer1"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if cfg.freeze_bn:
        backbone.apply(freeze_bn_func)

    hidden_layers = 128
    aspp = Classifier_Module2(2048, [6, 12, 18, 24], [6, 12, 18, 24], hidden_layers)

    model = DeeplabV2(backbone, aspp, hidden_layers, cfg.dataset.num_classes)
    return model