import os

import psutil
import torch
import torch.nn as nn
import ecgraph
import ecgraph.Function as F
import ecgraph.util_python.remote_access as rmt
from ecgraph.util_python.timecounter import TimeCounter
from ecgraph.context.context import glContext
from ecgraph.pipeline.pipegraph import PipeGraph
from ecgraph.pipeline.pipe_dorylus import PipeDorylus
from ecgraph.util_python.timecounter import time_counter

class GraphConvolution(nn.Module):
    # 初始化层：输入feature维度，输出feature维度，权重，偏移
    def __init__(self, in_features, out_features, layer_id, bias=True):
        super(GraphConvolution, self).__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features

        self.weight = ecgraph.ECTensor(tensor=torch.FloatTensor(in_features, out_features),
                                       name='w' + str(self.layer_id), requires_grad=True)
        torch.manual_seed(1)
        nn.init.xavier_uniform_(self.weight.tensor)

        if bias:
            self.bias = ecgraph.ECTensor(tensor=torch.FloatTensor(1, out_features), name='b' + str(self.layer_id),
                                         requires_grad=True)
            nn.init.zeros_(self.bias.tensor)

    def forward(self, input, graph):
        time_counter.start("fp-getadj")
        adj = graph.getAdj(self.layer_id)
        time_counter.end("fp-getadj")
        # embs_trans = F.mm(input, self.weight)
        # embs = rmt.pushEmbs(self.layer_id, graph, embs_trans)
        # output = F.spmm(adj, embs)
        if self.in_features>self.out_features:
            time_counter.start("fp-mm")
            embs_trans = F.mm(input, self.weight)
            time_counter.end("fp-mm")
            time_counter.start("fp-push")
            embs = rmt.pushEmbs(self.layer_id, graph, embs_trans)
            time_counter.end("fp-push")
            time_counter.start("fp-spmm")
            output = F.spmm(adj, embs)
            time_counter.end("fp-spmm")
        else:
            time_counter.start("fp-push")
            embs = rmt.pushEmbs(self.layer_id, graph, input)
            time_counter.end("fp-push")
            time_counter.start("fp-spmm")
            output = F.spmm(adj, embs)
            time_counter.end("fp-spmm")
            time_counter.start("fp-mm")
            output = F.mm(output, self.weight)
            time_counter.end("fp-mm")
        # the pipeline mode of PipeGraph
        # embs_trans = F.mm(input, self.weight)
        # agg = F.spmm(adj, embs_trans)
        # if self.layer_id != glContext.config['layer_num']:
        #     output = rmt.pushEmbs(self.layer_id + 1, graph.status, agg, graph)
        # else:
        #     output = agg
        #
        # pipe_graph = PipeGraph(graph, adj=adj,input=input)
        # output=pipe_graph.startPipe()

        # time_counter.start('dorylus_pipe')
        # pipe_dorylus = PipeDorylus(graph, self.layer_id, 4, input=input, weight=self.weight)
        # output = pipe_dorylus.startPipe()
        # time_counter.start('spmm')
        # output = F.spmm(adj, output)
        # time_counter.end('spmm')
        # time_counter.end('dorylus_pipe')

        if self.bias is not None:
            return F.add(output, self.bias)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
