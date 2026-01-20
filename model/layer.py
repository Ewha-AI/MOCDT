#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import math
from torch import nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
            
    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output