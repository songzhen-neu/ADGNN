

from adgnn.context import context



class EvalClass():
    info = {}
    maxacc_test=0
    r_maxacc=0
    ad_info={}
    graoh_last_ad=None
    m_change=[]

    def evalSampledGraph(self, graph):
        """
        evaluate information of sampled graphs (numbers of edges, remote edges of each layer)
        rmt_num represents number of remote neighboring vertices
        edge_num represents number of edges
        :param graph: trained graph
        :return: Non return, formulate a dict and call printGraphInfo to show
        """
        for i in graph.layer_compute:
            layer_info = graph.layer_compute[i]
            if not self.info.__contains__('edge_num' + str(i)):
                self.info['edge_num' + str(i)] = []
            if not self.info.__contains__('rmt_num' + str(i)):
                self.info['rmt_num' + str(i)] = []

            if i != 0:
                self.info['edge_num' + str(i)].append(len(layer_info.adj.tensor._values()))

            if i != len(graph.layer_compute) - 1:
                rmt_num_tmp = 0
                for w in layer_info.push_2_worker_nodes:
                    rmt_num_tmp += len(layer_info.push_2_worker_nodes[w])
                self.info['rmt_num' + str(i)].append(rmt_num_tmp)

    def printGraphInfo(self):
        print(self.m_change)
        context.glContext.graphBuild.printGraphInfo()






evaluator = EvalClass()
