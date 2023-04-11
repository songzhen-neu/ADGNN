import os
import torch
import sys
import copy
import itertools

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH + '/../../')
from adgnn.context import context
from adgnn.sample.sample import Sample
from adgnn.sample.random_sample import RandomSample
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

        # if self.enable_probcomm:  # 概率通信模块
        #     if epoch != 0 and epoch != self.last_round_ad:
        #         self.graph_last = probComm().probComm( batch_size, self.adjs_layer, self.expect_v_num)
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

        # if self.enable_probcomm:  # 概率通信模块
        #     if epoch != 0 and epoch != self.last_round_ad:
        #         self.graph_last = probComm().probComm( batch_size, self.adjs_layer, self.expect_v_num)
        return context.glContext.graph_sample


            # the number of updating ad_sampling
            # adjs_4_layer = {}
            # graph_train = graph.subgraphs['train']
            # layer_num = len(graph_train.layer_compute) - 1
            # train_vertices = graph_train.layer_compute[layer_num].train_vertices.detach().numpy()
            # for i in range(layer_num, 0, -1):
            #     adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i],
            #                                            graph.adj, fan_out[layer_num - i], train_vertices)
            #     time_counter.start('sample_updateNextHopTargetNodes')
            #     train_vertices = self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, train_vertices)
            #     time_counter.end('sample_updateNextHopTargetNodes')
            #     adjs_4_layer[i] = adj_new
            # # print(time_counter.time_list['sample_sampleForLayer'])
            # self.adjs_layer = adjs_4_layer
            # time_counter.start('updateGraph')
            # self.updateGraph(adjs_4_layer, graph)
            # time_counter.end('updateGraph')
            # self.graph_last = graph




    # def getDiff(self, agg_v, emb_nei, adj):
    #     agg_nei = np.zeros_like(agg_v)
    #     for id in adj:
    #         agg_nei += emb_nei[id]
    #     agg_nei = agg_nei / len(adj)
    #     ad = np.linalg.norm(agg_v - agg_nei, ord=2)
    #     return ad

    # def generateAS(self, epoch):
    #     graph_train = self.graph_last.subgraph['train']
    #     adj = self.graph_last.adj
    #     if not context.glContext.config['enable_as']:
    #         for lid in range(self.graph_last.layer_num, 0, -1):
    #             self.as_v[lid] = {}
    #         return
    #     if epoch == 1:
    #         # all vertices need to re-compute AS
    #         self.lastcount_mgroup = 0
    #         time_counter.start('sample_generateAS_epoch1')
    #         for lid in range(self.graph_last.layer_num, 0, -1):
    #             train_vertices = graph_train.layer_compute[lid].train_vertices.detach().numpy().tolist()
    #             o2n_dict = graph_train.layer_compute[lid].id_old2new_dict
    #             as_v_lid = context.glContext.dgnnClient.recomputeAS(self.agg_emb['agg_embs' + str(lid)],
    #                                                                 self.agg_emb['nei_embs' + str(lid)],
    #                                                                 o2n_dict, train_vertices, adj,
    #                                                                 self.as_fanout[self.graph_last.layer_num - lid],
    #                                                                 self.alpha_select_as)
    #             self.as_v[lid] = as_v_lid
    #         time_counter.end('sample_generateAS_epoch1')
    #     elif self.count_mgroup == self.lastcount_mgroup + self.m_as:  # ensure passing self.m_as rounds
    #         # ensure only the first epoch (not 0-th or others) of m_group will update anchor set
    #         if epoch == self.last_round_ad + self.m + 1:
    #             time_counter.start('sample_generateAS')
    #             self.lastcount_mgroup = self.count_mgroup
    #             # only the vertices with high ADs need to re-compute AS
    #
    #             for lid in range(self.graph_last.layer_num, 0, -1):
    #                 train_vertices = graph_train.layer_compute[lid].train_vertices.detach().numpy().tolist()
    #                 o2n_dict = graph_train.layer_compute[lid].id_old2new_dict
    #                 as_v_lid = context.glContext.dgnnClient.updateAS(self.agg_emb['agg_embs' + str(lid)],
    #                                                                  self.agg_emb['nei_embs' + str(lid)],
    #                                                                  o2n_dict, train_vertices, adj,
    #                                                                  self.as_v[lid],
    #                                                                  self.as_fanout[self.graph_last.layer_num - lid],
    #                                                                  self.alpha_select_as)
    #
    #                 self.as_v[lid] = as_v_lid
    #             time_counter.end('sample_generateAS')

    # def getAgg4V(self, vid, adj, lid, o2n_dict):
    #     """
    #     return embeddings of vid and its neighbors
    #     :param vid: vertex id
    #     :param adj: adjacent list of vertex id
    #     :param lid: layer id
    #     :param o2n_dict: the dict that maps old (global) id to new (id for each layer) id
    #     :return: note that agg_nei is a dict with old id
    #     """
    #     agg_v = self.agg_emb['agg_embs' + str(lid)][o2n_dict[vid]]
    #     agg_nei = {}
    #     for nid in adj:
    #         nid_new = o2n_dict[nid]
    #         agg_nei[nid] = self.agg_emb['nei_embs' + str(lid)][nid_new]
    #
    #     return agg_v, agg_nei

    # def redPriorityAS(self, adj, agg_v, emb_nei, k, v_as=None):
    #     diff = {}  # 存放差异值向量
    #     nei_use = []  # 每次加入的邻居顶点
    #     v_add = -1  # the vertex selected by the current round
    #     diff_all = np.zeros_like(agg_v)
    #     diff_tmp = sys.maxsize
    #
    #     for j in adj:  # 遍历邻居顶点存储差异向量，并获取差异最小值及对应顶点
    #         diff[j] = agg_v - emb_nei[j]
    #
    #     if v_as is not None:
    #         if len(v_as) != 0:
    #             nei_use = list(v_as)
    #             for id in nei_use:
    #                 diff_all += diff[id]
    #             diff_all = diff_all / len(nei_use)
    #
    #     while len(nei_use) < k:
    #         for key in diff:
    #             if key not in nei_use:
    #                 diff_new_vec = (diff[key] + diff_all * (len(nei_use))) / (len(nei_use) + 1)
    #                 diff_new = np.linalg.norm(x=diff_new_vec, ord=2)
    #                 if diff_tmp > diff_new:
    #                     diff_tmp = diff_new
    #                     v_add = key
    #         if v_add == -1:  # 不能使差异值变小，退出
    #             break
    #         else:
    #             nei_use.append(v_add)
    #             diff_all = (diff_all * len(nei_use) + diff[v_add]) / (len(nei_use) + 1)
    #
    #     adj_new = set(nei_use)
    #     # adj_left = set()
    #     # for nid in adj:
    #     #     if not adj_new.__contains__(nid):
    #     #         adj_left.add(nid)
    #
    #     diff_all = np.linalg.norm(diff_all, ord=2)
    #     # return diff_all, adj_new, adj_left
    #     return diff_all, adj_new

    # def simPriorityAS(self, adj, agg_v, emb_nei, k):
    #     as_sim = set()
    #     # as_sim_left = set()
    #     diff = {}
    #     for nid in adj:
    #         diff[nid] = np.linalg.norm(agg_v - emb_nei[nid], ord=2)
    #     diff_item_list = sorted(diff.items(), key=lambda item: item[1])
    #     for item in diff_item_list:
    #         if len(as_sim) < k:
    #             as_sim.add(item[0])
    #         # else:
    #         #     as_sim_left.add(item[0])
    #
    #     agg_nei = np.zeros_like(agg_v)
    #     for nid in as_sim:
    #         agg_nei += emb_nei[nid]
    #     agg_nei = agg_nei / len(as_sim)
    #     ad_sim = np.linalg.norm(agg_v - agg_nei, ord=2)
    #
    #     return ad_sim, as_sim

    # def recomputeAS(self, adj, agg_v, agg_nei, lid):
    #     k = self.as_fanout[self.graph_last.layer_num - lid]
    #     ad_red, as_red = self.redPriorityAS(adj, agg_v, agg_nei, k)
    #     ad_sim, as_sim = self.simPriorityAS(adj, agg_v, agg_nei, k)
    #     if ad_red < self.alpha_select_as * ad_sim:
    #         return as_red
    #     else:
    #         return as_sim

    # def sampleForLayer(self, layer_id, layer_computer, adj, k, train_vertices, **kwargs):  # 贪婪算法
    #     # sample for each layer from L-th layer to 0-th layer
    #     # get adj of this layer and target vertices of last layer
    #     # o2n_dict = layer_computer.id_old2new_dict
    #     # adj_new= {}
    #     # nei_set=set()
    #     # for id in train_vertices:
    #     #     agg_v,emb_nei=self.getAgg4V(id,adj[id],layer_id,o2n_dict)
    #     #     _,adj_tmp=self.redPriorityAS(adj[id],agg_v,emb_nei,k)
    #     #     for nid in adj_tmp:
    #     #         nei_set.add(nid)
    #     #     adj_new[id]=adj_tmp
    #
    #     o2n_dict = layer_computer.id_old2new_dict
    #     time_counter.start('sample_sampleForLayer')
    #     dim_prune_itv = self.dim_itvs[layer_id - 1]
    #     # print("before memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #     data_ad = context.glContext.dgnnClient.sampleForLayerAd(self.agg_emb['agg_embs' + str(layer_id)],
    #                                                             self.agg_emb['nei_embs' + str(layer_id)],
    #                                                             o2n_dict, train_vertices, self.as_v[layer_id], k,
    #                                                             layer_id, dim_prune_itv, self.adcomp_num)
    #     # print("after memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #     time_counter.end('sample_sampleForLayer')
    #     adj_new = data_ad[0]
    #     nei_set = data_ad[1]
    #     return adj_new, nei_set

    # def sampleForLayer(self, layer_id, layer_computer, adj, k, **kwargs):     # 暴力求解法 len <= k
    #     adj_new = {}
    #     nei_set = []
    #     train_vertices = layer_computer.train_vertices.detach().numpy()
    #     nei_embs = self.agg_emb['nei_embs' + str(layer_id)]
    #     agg_embs = self.agg_emb['agg_embs' + str(layer_id)]
    #     for i in train_vertices:
    #         diff = {}  # 存放差异值向量
    #         diff_num = {}   # 存放差异值
    #         nei_use = []
    #         adj_i = copy.deepcopy(adj[i])
    #         adj_i.add(i)
    #         adj_list = list(adj_i)
    #         adj_list = [str(i) for i in adj_list]
    #         if len(adj_list) <= k:
    #             adj_new[i] = adj_i
    #             nei_set.extend(adj_new[i])
    #         else:
    #             if k <= 0:
    #                 adj_new[i] = set(nei_use)
    #                 nei_set.extend(adj_new[i])
    #             else:
    #                 for j in itertools.combinations(adj_list, 1):   # 先记录一个邻居节点的差异性向量、差异性值
    #                     j_int = int(j[0])
    #                     diff[j] = agg_embs[layer_computer.id_old2new_dict[i]] - nei_embs[
    #                         layer_computer.id_old2new_dict[j_int]]
    #                     diff_ten = torch.from_numpy(diff[j])
    #                     diff_num[j] = torch.norm(diff_ten, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    #                 for x in range(1, k):     # 依次记录2、3···k个邻居节点的差异性向量、差异性值
    #                     for tu in itertools.combinations(adj_list, x + 1):
    #                         tu_list = list(tu)
    #                         tu_list.pop()   # 先将子集的最后一个值去掉，然后计算子集剩余值对应的差异性向量加上最后一个值对应的差异性向量
    #                         diff[tu] = (diff[tuple([tu[x]])] + ((diff[tuple(tu_list)]) * x)) / (x+1)
    #                         diff_ten = torch.from_numpy(diff[tu])
    #                         diff_num[tu] = torch.norm(diff_ten, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    #                 key_min = min(diff_num.keys(), key=(lambda kk: diff_num[kk]))   # 取差异值的最小值，则作为返回的子集
    #                 key_min_list = list(key_min)
    #                 key_min_list = map(eval, key_min_list)
    #                 nei_use = list(key_min_list)
    #                 adj_new[i] = set(nei_use)
    #                 nei_set.extend(adj_new[i])
    #     return adj_new, set(nei_set), nei_set

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
