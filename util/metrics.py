import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np


class L2Softmax(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super(L2Softmax, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.temp = temp
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        output = logits / self.temp
        return output


# taken and modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


# taken and modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.n_classes = out_features
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


# taken and modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.n_classes = out_features
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


# taken and modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_fetures = out_features
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
