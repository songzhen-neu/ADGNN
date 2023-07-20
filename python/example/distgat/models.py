import torch.nn as nn
from example.distgat.layers import GraphAttentionLayer
import torch
import torch.nn.functional as F
from adgnn.util_python.remote_access import catCacheFeature
import adgnn
from adgnn.context import context
import numpy as np
import scipy.sparse as sp





class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.layerNum = len(nhid) + 1
        self.dropout = dropout
        self.parameters_collection={}

        self.attentions = [GraphAttentionLayer(nfeat,
                                               nhid[0],
                                               1,
                                               i,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for i in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.parameters_collection['w'+str(1)+'-'+str(i)]=attention.W
            self.parameters_collection['a'+str(1)+'-'+str(i)]=attention.a

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid[0] * nheads,
                                           nclass,
                                           2,
                                           0,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

        self.parameters_collection['w2-0']=self.out_att.W
        self.parameters_collection['a2-0']=self.out_att.a


    x_print_grad=None

    def forward(self,graph):
        context.glContext.graphBuild.setGraphMode(graph.graph_mode)
        x = graph.feat_data.to(context.glContext.config['device'])
        adj_1=graph.graphlayers[1].adj.tensor.to(context.glContext.config['device'])
        adj_2=graph.graphlayers[2].adj.tensor.to(context.glContext.config['device'])
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj_1,graph) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x=self.out_att(x, adj_2,graph)
        x = F.elu(x)

        return F.log_softmax(x, dim=1)
