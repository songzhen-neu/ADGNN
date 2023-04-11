from collections import Counter
import copy
import random
from adgnn.context import context
from adgnn.sample.sample import Sample

class probComm:

    def probComm(self, graph, batch_size, adjs_layer, expect_v_num):
        adjs_4_layer = {}
        graph_train = graph.subgraph['train']
        layer_num = len(graph_train.layer_compute) - 1
        Sample().gnrtMiniBatch(graph, batch_size, layer_num)
        for i in range(layer_num, 0, -1):
            nei_now = []
            adj_now = copy.deepcopy(adjs_layer[i])
            for value in adj_now.values():
                nei_now.extend(value)
            adj_new, nei_set = self.probCommForLayer(expect_v_num[layer_num - i], graph.v2wk_map, adj_now, nei_now)
            Sample().updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, graph_train)
            adjs_4_layer[i] = adj_new
        Sample().updateGraph(adjs_4_layer, graph)
        work_id = context.glContext.config['id']
        # for k in range(layer_num, 0, -1):
        #     len_fsthop = len(graph.subgraph['train'].layer_compute[k].fsthop_for_worker[1 - work_id])
        #     print("procom module:  ", "layer:", k, "fsthop_num", len_fsthop)
        return graph

    def probCommForLayer(self, expect_v_num, v2wk, adj_new, nei_set_list):
        adj = copy.deepcopy(adj_new)
        adj_parmax = {}  # 记录每个顶点邻接点中概率最大的点
        work_id = context.glContext.config['id']
        far_v_pro = {}  # 记录每个远程顶点出现的概率
        v_count = Counter(nei_set_list)  # 记录每个邻接顶点出现的个数
        for i in list(v_count.keys()):
            if v2wk[i] == work_id:
                del v_count[i]

        nei_set_sum = sum(v_count.values())

        for key in v_count:
            far_v_pro[key] = (v_count[key] / nei_set_sum) * expect_v_num
        for key in adj:  # 计算每个顶点邻接点中概率最大的点
            v_max_par = 0
            v_max = -1
            for i in adj[key]:
                if v2wk[i] != work_id:
                    if far_v_pro[i] > v_max_par:
                        v_max_par = far_v_pro[i]
                        v_max = i
            if v_max != -1:
                adj_parmax[key] = v_max
        v_not_choice = set()  # 按照概率记录不要的顶点
        for key in far_v_pro:
            ran_num = random.uniform(0, 1)
            if ran_num > far_v_pro[key]:
                v_not_choice.add(key)
        nei_set_new = set(nei_set_list)  # 在所有邻居节点中移除随机选择的顶点
        nei_set_new = nei_set_new - v_not_choice
        for key in adj_new:  # 逐个移除每个训练顶点邻接表中的随机选择的顶点（如果为空，在邻接表和邻居节点中加上概率最大的顶点）
            adj_new[key] = adj_new[key] - v_not_choice
            if len(adj_new[key]) == 0:
                adj_new[key].add(adj_parmax[key])
                nei_set_new.add(adj_parmax[key])
        return adj_new, nei_set_new

ProbComm = probComm()
