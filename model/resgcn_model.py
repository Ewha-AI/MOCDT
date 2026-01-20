#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from model.layer import GraphConvolution


class ResGCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.gc3 = GraphConvolution(n_hid, n_out)
        self.dropout = dropout

    def forward(self, x, adj):
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.relu(self.gc2(h1, adj))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1   

        out = self.gc3(h2, adj)
        return out
 