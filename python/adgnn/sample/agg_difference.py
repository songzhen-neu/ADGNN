import os
import torch
import sys
import copy
import itertools

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH + '/../../')
from adgnn.context import context
from adgnn.sample.sample import Sample
from adgnn.sample.graphsage import RandomSample
from adgnn.probcomm.probcomm import probComm
from adgnn.util_python.comp_tree import generateAggEmb
import numpy as np
from adgnn.util_python.timecounter import time_counter
from adgnn.data_structure.graph_train import Graph_Train
import time
import psutil
from adgnn.util_python.data_trans import transGraphCppToPython
from adgnn.util_python.evaluation import evaluator


class AggDiff(Sample):
    agg_emb = {}
    adjs_layer = {}
    graph_last = None
    m = 5  # re-sample every m rounds
    beta_recomp = None  # if_argument for recomputing
    pk_last = None  # pk is for adaptive m-tuner
    benefit_m = 3  # only if m>=3, ad can benefit
    dim_itvs = [1, 1]  # hiddenN -> ... ->hidden1 -> feature
    adcomp_num = 1000
    m_sum=0
    enable_pc=None
    comm_fo=None


    full_graph = None

    def initParameter(self, epoch, fan_out, kwargs):
        if epoch == 0:
            self.enab_adap_m = kwargs['enab_adap_m']
            self.m = kwargs['m']
            self.fanout = fan_out
            self.enable_pc = kwargs['enable_pc']
            self.comm_fo = kwargs['comm_fo']
            # self.graph_last = graph  # init the graph for 0th epoch
            self.last_round_ad = 0
            self.last_count_as = 0
            self.dim_itvs = kwargs['dim_itvs']
            self.adcomp_num = kwargs['adcomp_num']
            self.model = kwargs['model']
            self.nei_prune=kwargs['nei_prune']

    def setM(self,epoch):
        if epoch==3:
            train_time_full=time_counter.time_list['train_time'][0]
            sample_time=time_counter.time_list['train_time'][1]
            train_time_sampled=time_counter.time_list['train_time'][2]
            benefit_m=int((train_time_full+sample_time)/train_time_sampled)
            # degree=context.glContext.config['edge_num']*2/context.glContext.config['data_num']
            # sample_ratio=degree/context.glContext.config['sample_num'][0]
            self.m=int(benefit_m*2)
            # if self.benefit_m<3:
            #     self.benefit_m=3
            if self.m<10:
                self.m=10
            print(self.m,self.benefit_m)
            self.m,self.benefit_m=context.glContext.dgnnServerRouter[0].setM(context.glContext.config['id'],self.m,self.benefit_m)


    def adapMTuner(self, epoch):
        if not self.pk_last:
            pk = self.model.gc[1].weight.tensor
            for i in range(2, context.glContext.config['layer_num'] + 1):
                pk = pk.mm(self.model.gc[i].weight.tensor)
            pk = np.linalg.norm(pk, ord=2)
            self.pk_last = pk
        else:
            pk = self.model.gc[1].weight.tensor
            for i in range(2, context.glContext.config['layer_num'] + 1):
                pk = pk.mm(self.model.gc[i].weight.tensor)
            pk = np.linalg.norm(pk, ord=2)
            if epoch != 0:
                if pk > 1.1 * self.pk_last:
                    if self.m > self.benefit_m:
                        self.m -= int(self.m*0.2)
                    else:
                        self.m = 1
                elif pk < self.pk_last:
                    self.m += int(self.m*0.2)

            # print(pk, self.m)
            self.pk_last = pk

    def sample(self, fan_out, epoch, batch_size, **kwargs):
        """
        :param graph: m(int), sampling interval
        :return:
        """
        self.initParameter(epoch, fan_out, kwargs)
        self.setM(epoch)

        if epoch == self.m_sum or self.m==1:
            return context.glContext.graph_full
        elif epoch==self.m_sum+1:  # last_round_ad has been updated as the 0-th epoch of the new mgroup
            time_counter.start('sample_generateAggEmb')
            self.generateAggEmb()
            time_counter.end('sample_generateAggEmb')
            context.glContext.sample.adSample(fan_out,self.dim_itvs,self.adcomp_num,self.enable_pc,self.comm_fo,self.nei_prune)
            transGraphCppToPython("train", "sample")

        if epoch==self.m_sum+self.m-1 or self.m==1:
            self.m_sum+=self.m
            evaluator.m_change.append(self.m)
            if self.enab_adap_m:
                self.adapMTuner(epoch)

        return context.glContext.graph_sample


    def sample_every(self, fan_out, epoch, batch_size, **kwargs):
        """
        :param graph: m(int), sampling interval
        :return:
        """
        self.initParameter(epoch, fan_out, kwargs)
        if epoch==0:
            return context.glContext.graph_full
        time_counter.start('sample_generateAggEmb')
        self.generateAggEmb()
        time_counter.end('sample_generateAggEmb')
        context.glContext.sample.adSample(fan_out,self.dim_itvs,self.adcomp_num,self.enable_pc,self.comm_fo)
        transGraphCppToPython("train", "sample")

        return context.glContext.graph_sample




    def generateAggEmb(self):
        layer_num = context.glContext.config['layer_num']
        layer_count = layer_num
        queue = []
        self.breadthFirstTraversal(context.glContext.compGraph, layer_count, queue)
        for id in self.agg_emb:
            context.glContext.sample.setAggEmb(id,self.agg_emb[id])

    def breadthFirstTraversal(self, comp_graph, layer_count, queue):
        if comp_graph is None:
            return
        if comp_graph.left is not None:
            queue.append(comp_graph.left)
        if comp_graph.right is not None:
            queue.append(comp_graph.right)
        if comp_graph.operator == 'spmm':
            self.agg_emb['nei_embs' + str(layer_count)] = comp_graph.right.tensor.detach().numpy()
            self.agg_emb['agg_embs' + str(layer_count)] = comp_graph.tensor.detach().numpy()
            layer_count -= 1
        if len(queue) != 0:
            next_root = queue.pop(0)
            self.breadthFirstTraversal(next_root, layer_count, queue)
        return


aggDiff = AggDiff()
