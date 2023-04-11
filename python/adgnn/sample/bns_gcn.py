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

    def reGnrtRmtFstHopV(self, adj, train_vertices, id, v2wk_map):
        rmt_nei_set=set()
        for v in train_vertices:
            neis=adj[v]
            for nei in neis:
                if v2wk_map[nei] != id:
                    rmt_nei_set.add(nei)
        return rmt_nei_set

    def sampleForLayer(self, i, layer_computer, adj, k, train_vertices, **kwargs):
        data_ad = context.glContext.dgnnClient.bnsSample(train_vertices,k,i)
        adj_new=data_ad[0]
        nei_set=data_ad[1]
        return adj_new, nei_set


bns_gcn = BNSGCN()
