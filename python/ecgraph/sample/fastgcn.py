from ecgraph.sample.sample import Sample
from collections import Counter
import random
import copy
from ecgraph.context import context
from ecgraph.util_python.timecounter import time_counter
from ecgraph.context import context
from ecgraph.util_python.data_trans import transGraphCppToPython

class FASTGCN(Sample):

    def sample(self, graph, fan_out, epoch, batch_size, **kwargs):
        # graph_train = graph.subgraph['train']
        # layer_num = graph.layer_num
        # adjs_4_layer = {}
        # train_vertices=graph_train.layer_compute[layer_num].train_vertices.numpy()
        # # sample for remote neighbor pool, update adj, encode and update map
        # for i in range(layer_num, 0, -1):
        #     adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i], graph.adj,
        #                                            fan_out[layer_num - i],train_vertices)
        #     train_vertices=self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, train_vertices)
        #     adjs_4_layer[i] = adj_new
        # # print(adjs_4_layer)
        # self.updateGraph(adjs_4_layer, graph)
        # return graph
        time_counter.start("randomSample")
        context.glContext.sample.fastgcnSample(fan_out)
        time_counter.end("randomSample")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample


    def sampleForLayer(self, i, layer_computer, adj, k,train_vertices, **kwargs):
        # nei_set=[]
        # for id in train_vertices:
        #     nei_set.extend(adj[id])
        # v_count=Counter(nei_set)
        #
        # v_pro = {}  # 记录每个邻接节点出现的概率
        # nei_set_sum = sum(v_count.values())
        # for key in v_count:
        #     v_pro[key] = v_count[key] / nei_set_sum * k
        #
        # v_not_choice = set()  # 按照概率记录舍弃的邻接节点
        # for key in v_pro:
        #     ran_num = random.uniform(0, 1)
        #     if ran_num > v_pro[key]:
        #         v_not_choice.add(key)
        #
        # adj_new={}
        # nei_set_new = set(nei_set)  # 在所有邻居节点中移除随机选择的顶点
        # nei_set_new = nei_set_new - v_not_choice
        # for key in train_vertices:
        #     adj_new[key] = adj[key] - v_not_choice

        # return adj_new, nei_set_new

        data_ad = context.glContext.dgnnClient.fastgcnSample(train_vertices,k)
        adj_new=data_ad[0]
        nei_set=data_ad[1]
        return adj_new, nei_set

fastgcn = FASTGCN()
