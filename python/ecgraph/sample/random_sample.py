from ecgraph.sample.sample import Sample
import numpy as np
import torch
import sys
from ecgraph.util_python.timecounter import time_counter
from ecgraph.context import context
import psutil,os
from ecgraph.util_python.data_trans import transGraphCppToPython

class RandomSample(Sample):
    # m = 5
    # def sample(self, graph, fan_out, epoch, batch_size,**kwargs):
    #     graph_train = graph.subgraph['train']
    #     layer_num = len(graph_train.layer_compute) - 1
    #     self.gnrtMiniBatch(graph, batch_size, layer_num)
    #     adjs_4_layer = {}
    #     train_vertices=graph_train.layer_compute[layer_num].train_vertices.numpy()
    #     for i in range(layer_num, 0, -1):
    #         adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i], graph.adj,
    #                                                fan_out[layer_num - i],train_vertices)
    #         train_vertices=self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, train_vertices)
    #         adjs_4_layer[i] = adj_new
    #     time_counter.start('updateGraph')
    #     self.updateGraph(adjs_4_layer, graph)
    #     time_counter.end('updateGraph')
    #     return graph,adjs_4_layer
    #
    # def sample4test(self, graph, fan_out, epoch, batch_size,**kwargs):
    #     if epoch % self.m == 0:
    #         self.sample(graph, [sys.maxsize, sys.maxsize], epoch, sys.maxsize)
    #     else:
    #         graph_train = graph.subgraph['train']
    #         layer_num = len(graph_train.layer_compute) - 1
    #         self.gnrtMiniBatch(graph, batch_size, layer_num)
    #         adjs_4_layer = {}
    #         for i in range(layer_num, 0, -1):
    #             adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i], graph.adj,
    #                                                    fan_out[layer_num - i])
    #             self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, graph_train)
    #             adjs_4_layer[i] = adj_new
    #         self.updateGraph(adjs_4_layer, graph)
    #
    #     return graph
    #
    # def sample4test_compromise(self, graph, fan_out, epoch, batch_size, **kwargs):
    #     if epoch % self.m == 0:
    #         self.sample(graph, [sys.maxsize, sys.maxsize], epoch, sys.maxsize)
    #         self.graph_last = graph
    #     elif epoch % self.m == 1:
    #         graph_train = graph.subgraph['train']
    #         layer_num = len(graph_train.layer_compute) - 1
    #         self.gnrtMiniBatch(graph, batch_size, layer_num)
    #         adjs_4_layer = {}
    #         for i in range(layer_num, 0, -1):
    #             adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i], graph.adj,
    #                                                    fan_out[layer_num - i])
    #             self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, graph_train)
    #             adjs_4_layer[i] = adj_new
    #         self.updateGraph(adjs_4_layer, graph)
    #         self.graph_last = graph
    #
    #     return self.graph_last
    #
    # def sampleForLayer(self, layer_id, layer_compute, adj, k,train_vertices, **kwargs):
    #     data_ad = context.glContext.dgnnClient.randomSample(train_vertices,k)
    #     adj_tmp=data_ad[0]
    #     fsthop_nei=data_ad[1]
    #
    #     # np.random.seed(1)
    #     # adj_tmp = {}
    #     # fsthop_nei = []
    #     # for id in train_vertices:
    #     #     nei_v = adj[id]
    #     #     if len(nei_v) < k:
    #     #         adj_tmp[id] = nei_v
    #     #     else:
    #     #         nei_perm = np.random.permutation(list(nei_v))
    #     #         nei_perm = nei_perm[0:k]
    #     #         adj_tmp[id] = set(nei_perm)
    #     #     fsthop_nei.extend(adj_tmp[id])
    #     # fsthop_nei=set(fsthop_nei)
    #     # print(layer_id,adj_tmp, set(fsthop_nei))
    #     return adj_tmp, fsthop_nei
    def sample(self, fan_out, epoch,  **kwargs):
        time_counter.start("randomSample")
        context.glContext.sample.randomSample(fan_out)
        # context.glContext.sample.randomSampleNoRebuild(fan_out)
        time_counter.end("randomSample")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample

random_sample = RandomSample()
