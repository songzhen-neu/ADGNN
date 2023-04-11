from adgnn.sample.sample import Sample
from collections import Counter
import random
import copy
from adgnn.context import context
from adgnn.util_python.timecounter import time_counter
from adgnn.context import context
from adgnn.util_python.data_trans import transGraphCppToPython

class FASTGCN(Sample):

    def sample(self, graph, fan_out, epoch, batch_size, **kwargs):
        time_counter.start("randomSample")
        context.glContext.sample.fastgcnSample(fan_out)
        time_counter.end("randomSample")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample


    def sampleForLayer(self, i, layer_computer, adj, k,train_vertices, **kwargs):
        data_ad = context.glContext.dgnnClient.fastgcnSample(train_vertices,k)
        adj_new=data_ad[0]
        nei_set=data_ad[1]
        return adj_new, nei_set

fastgcn = FASTGCN()
