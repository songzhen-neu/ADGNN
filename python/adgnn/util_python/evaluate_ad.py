import torch

from python.adgnn.sample.agg_difference import aggDiff
from python.adgnn.sample.graphsage import random_sample
from python.adgnn.sample.bns_gcn import bns_gcn
from adgnn.util_python.comp_tree import generateAggEmb
import copy
import numpy as np
import sys
from adgnn.context import context


class EvalAD():
    def __init__(self):
        self.ad_info = ''

    def evalAD(self, model, e, subgraph):
        output_full = model(context.glContext.graph_full.subgraphs['train'])
        agg_full = generateAggEmb(output_full)

        output_sampled = model(subgraph)
        agg_sampled = generateAggEmb(output_sampled)

        emb_id = 'agg_embs' + str(context.glContext.config['layer_num'])

        num=agg_full[emb_id].shape[0]
        ad_ad = np.linalg.norm(agg_sampled[emb_id] - agg_full[emb_id], ord=2, axis=None)/num
        self.ad_info += "epoch " + str(e) + ": " + str(ad_ad) + '\n'

    def tensor2adj(self, graph):
        adj_x = graph.subgraphs['train'].graphlayers[2].adj.tensor.data.coalesce().indices()[0].numpy().tolist()
        adj_y = graph.subgraphs['train'].graphlayers[2].adj.tensor.data.coalesce().indices()[1].numpy().tolist()
        adj = {}
        for i in range(len(adj_x)):
            id = adj_x[i]
            nei = adj_y[i]
            if adj.__contains__(id):
                adj[id].add(nei)
            else:
                adj[id] = set()
                adj[id].add(nei)
        return adj

    def evalOptimalSet(self, epoch, graph):
        if epoch==0:
            return
        if epoch == 1:
            self.graph_base = copy.deepcopy(graph)
        else:
            adj = self.tensor2adj(graph)
            adj_base=self.tensor2adj(self.graph_base)

            sum=0
            for id in adj:
                intersec_size=len(adj[id].intersection(adj_base[id]))
                adj_size=len(adj[id])
                sum+=intersec_size/adj_size
            sum=sum/len(adj)

            print(sum)

    def printAD(self):
        print(self.ad_info)


eval_ad = EvalAD()
