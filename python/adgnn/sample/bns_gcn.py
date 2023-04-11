from adgnn.context import context
from adgnn.sample.sample import Sample
import random
from adgnn.util_python.timecounter import time_counter
from adgnn.util_python.data_trans import transGraphCppToPython

random.seed(1)

class BNSGCN(Sample):

    def sample(self, graph, fan_out, epoch, batch_size,**kwargs):
        if epoch==0:
            context.glContext.sample.buildRmtAndLocAdj()
        time_counter.start("randomSample")
        context.glContext.sample.bnsSample(fan_out)
        time_counter.end("randomSample")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample
        # graph_train = graph.subgraph['train']
        # layer_num = graph.layer_num
        # adjs_4_layer = {}
        # train_vertices=graph_train.layer_compute[layer_num].train_vertices.numpy()
        # # sample for remote neighbor pool, update adj, encode and update map
        # for i in range(layer_num, 0, -1):
        #     adj_new, nei_set = self.sampleForLayer(i, graph_train.layer_compute[i], graph.adj,
        #                                            fan_out[layer_num - i], train_vertices, v2wk_map=graph.v2wk_map)
        #     train_vertices=self.updateNextHopTargetNodes(i, nei_set, graph.v2wk_map, train_vertices)
        #     adjs_4_layer[i] = adj_new
        # self.updateGraph(adjs_4_layer, graph)
        # return graph

    def reGnrtRmtFstHopV(self, adj, train_vertices, id, v2wk_map):
        rmt_nei_set=set()
        for v in train_vertices:
            neis=adj[v]
            for nei in neis:
                if v2wk_map[nei] != id:
                    rmt_nei_set.add(nei)
        return rmt_nei_set

    def sampleForLayer(self, i, layer_computer, adj, k, train_vertices, **kwargs):

        # adj_new = {}
        # id=context.glContext.config['id']
        # time_counter.start("reGnrtRmtFstHopV")
        # rmt_nei_set=self.reGnrtRmtFstHopV(adj, train_vertices, context.glContext.config['id'],
        #                       kwargs['v2wk_map'])
        # time_counter.end("reGnrtRmtFstHopV")
        #
        # time_counter.start("bns-contains")
        # if k >= len(rmt_nei_set):
        #     for id in train_vertices:
        #         adj_new[id] = adj[id]
        # else:
        #     time_counter.start("bns-randomsample")
        #     rmt_nei_set = random.sample(rmt_nei_set, k)
        #     time_counter.end("bns-randomsample")
        #     for id in train_vertices:
        #         adj_new[id] = []
        #         for nid in adj[id]:
        #             if rmt_nei_set.__contains__(nid) or kwargs['v2wk_map'][nid] == context.glContext.config['id']:
        #                 adj_new[id].append(nid)
        # time_counter.end("bns-contains")
        # time_counter.start("bns-set")
        # nei_set=set()
        # for v in adj_new:
        #     for nei in adj_new[v]:
        #         nei_set.add(nei)
        # time_counter.end("bns-set")
        data_ad = context.glContext.dgnnClient.bnsSample(train_vertices,k,i)
        adj_new=data_ad[0]
        nei_set=data_ad[1]
        return adj_new, nei_set


bns_gcn = BNSGCN()
