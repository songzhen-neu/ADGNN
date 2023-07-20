import torch.nn as nn
from example.distgcn.layers import GraphConvolution
import torch
import adgnn.Function as F
from adgnn.util_python.remote_access import catCacheFeature
import adgnn
from adgnn.context import context
from adgnn.util_python.timecounter import time_counter


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        self.layerNum = len(nhid) + 1
        self.dropout = dropout
        self.gc = {}
        self.parameters_collection={}
        # 1st layer is stored in gc[0]
        for i in range(1, self.layerNum + 1):
            if i == 1:
                self.gc[i] = GraphConvolution(nfeat, nhid[0], 1)
            elif i != self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nhid[i - 1], i)
            elif i == self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nclass, i)
            self.add_module("layer_{0}".format(i),self.gc[i])
        for i in range(1,len(self.gc)+1):
            self.parameters_collection['w'+str(i)]=self.gc[i].weight.tensor
            self.parameters_collection['b'+str(i)]=self.gc[i].bias.tensor

    def forward(self, graph):
        context.glContext.graphBuild.setGraphMode(graph.graph_mode)
        x = adgnn.ECTensor(graph.feat_data) # Dorylus
        time_counter.start("to_device")
        x.tensor.to(context.glContext.config['device'])
        time_counter.end("to_device")
        # x = catCacheFeature(graph) # PipeGraph
        for i in range(1, self.layerNum + 1):
            x = self.gc[i](x, graph)
            if not i == self.layerNum:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)
