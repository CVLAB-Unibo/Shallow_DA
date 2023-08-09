import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, prediction, target, depth):
        # print(depth.size())
        loss = self.loss_fn(prediction, target)
        weights = torch.ones_like(depth)
        weights[depth>depth.mean()] = 10
        # print(weights.size())

        loss = torch.mean(loss * weights.squeeze())
        return loss

class WeightedCrossEntropy(nn.Module):
    def __init__(self, ignore_index, num_classes):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.max_value = 7
        self.start = True
        # self.device = "cuda:"+str(device[0])
    
    def forward(self, prediction, target):
        if self.start:
            self.class_weight = torch.FloatTensor(self.num_classes).zero_().to(target.device) + 1
            self.often_weight = torch.FloatTensor(self.num_classes).zero_().to(target.device) + 1
            self.start = False
        
        weight = torch.FloatTensor(self.num_classes).zero_().to(target.device)
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().to(target.device)
        often = torch.FloatTensor(self.num_classes).zero_().to(target.device)
        often += 1
        n, h, w = target.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(target==i)
            if count[i] < 64*64*n: #small objective
                weight[i] = self.max_value
        
        often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        return F.cross_entropy( prediction, target, weight=self.class_weight, ignore_index=self.ignore_index)

def get_loss_fn(cfg):
    if cfg.losses.loss_fn=='crossentropy':
        return nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    if cfg.losses.loss_fn=='weighted_crossentropy':
        return WeightedCrossEntropy(ignore_index=cfg.dataset.ignore_index, num_classes=cfg.dataset.num_classes) 
    elif cfg.losses.loss_fn=='masked_crossentropy':
        return MaskedCrossEntropy(ignore_index=cfg.dataset.ignore_index)        
    else:
        return nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)