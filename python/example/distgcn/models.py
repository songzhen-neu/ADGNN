import torch.nn as nn
from example.distgcn.layers import GraphConvolution
import torch
import adgnn.Function as F
from adgnn.util_python.remote_access import catCacheFeature
import adgnn
from adgnn.context import context


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        self.layerNum = len(nhid) + 1
        self.dropout = dropout
        self.gc = {}
        # 1st layer is stored in gc[0]
        for i in range(1, self.layerNum + 1):
            if i == 1:
                self.gc[i] = GraphConvolution(nfeat, nhid[0], 1)
            elif i != self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nhid[i - 1], i)
            elif i == self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nclass, i)

    def forward(self, graph):
        context.glContext.graphBuild.setGraphMode(graph.graph_mode)
        x = adgnn.ECTensor(graph.feat_data) # Dorylus
        # x = catCacheFeature(graph) # PipeGraph
        for i in range(1, self.layerNum + 1):
            x = self.gc[i](x, graph)
            if not i == self.layerNum:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)
