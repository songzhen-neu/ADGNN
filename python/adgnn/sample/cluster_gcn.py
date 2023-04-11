from adgnn.sample.sample import Sample
import random
import linecache
from adgnn.context import context
import torch
from adgnn.context import context as ct
from adgnn.util_python.timecounter import time_counter
from adgnn.util_python.data_trans import transGraphCppToPython


class CLUSTERGCN(Sample):
    cluster_number = None  # divide into cluster_number subgraphs
    lines=None
    def initAndReturnMetis(self,cluster_num):
        self.cluster_number = cluster_num
        fileName = ct.glContext.config['data_path']
        metisFileName = fileName + '/nodesPartition' + '.metis' + str(self.cluster_number) + '.txt'
        lines=[]
        file=open(metisFileName)
        while True:
            line=file.readline()
            if line:
                lineSplit = line.split('\t')
                lineSplit[-1] = lineSplit[-1][:-1]
                if lineSplit[0] =='':
                    lines.append(set())
                    continue
                lineSplit_set = set(lineSplit)
                lineSplit_set={int(x) for x in lineSplit_set}
                lines.append(lineSplit_set)
            else:
                break
        return lines


    def sample(self, graph, fan_out, epoch, batch_size, **kwargs):
        if epoch==0:
            self.lines=self.initAndReturnMetis(kwargs['cluster_number'])
        workerNum = ct.glContext.config['worker_num']
        worker_id = context.glContext.config['id']

        fan_out = int(fan_out / workerNum)
        cluster_choice = []
        time_counter.start("metis-read")
        while (len(cluster_choice) < fan_out):
            x = random.randint(int(self.cluster_number / workerNum * worker_id ),
                               int(self.cluster_number / workerNum * (worker_id + 1)-1))
            if x not in cluster_choice:
                cluster_choice.append(x)

        sub_node = set()
        for line_id in cluster_choice:
            # if self.lines[line_id].__contains__(0):
                # print(line_id)
            sub_node = sub_node | self.lines[line_id]
        time_counter.end("metis-read")
        # if sub_node.__contains__(0):
        #     print("0000000000")

        time_counter.start("clustergcnSample-c++")
        context.glContext.sample.clustergcnSample(sub_node)
        time_counter.end("clustergcnSample-c++")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample




cluster_gcn = CLUSTERGCN()
