from ecgraph.util_python.data_trans import *
import torch



class Sample:
    round = 0
    round_num = 0
    train_left = []
    graph_last=None
    fanout=None


    def sampleForLayer(self, i, layer_computer, adj, k, train_vertices,**kwargs):
        """
        UDF sampling for each layer, return adj_new and nei_set
        :param i:int layer id from layer_num to 1 (e.g., 2-layer GCN, [2,1], since 0 doesn't need to be sampled)
        :param layer_computer:object contains necessary information of layer i for distributed training
        :param adj:{int,set(int)} adj of all local vertices
        :param k:int number of fan-out
        :param kwargs: passing extra object for different sampling methods (e.g., v2wk_map for BNS-GCN)
        :return: adj_new:{int,set(int)} sampled adj, nei_set:set(int): all neighbors including local and remote
        """
        pass

    def sample(self, graph, fan_out, epoch, batch_size,  **kwargs):
        pass

    def buildFsthopForWorker(self, nei_set, v2wk_map, train_vertices):
        train_set = set(train_vertices)
        nei2wk4layer, nei_loc, nei_rmt = splitNeiToWorker(nei_set, train_set, v2wk_map)
        return nei2wk4layer, nei_loc, nei_rmt

    def gnrtMiniBatch(self, graph, batch_size, layer_num):
        # generate mini-batches
        train_nodes_entire = graph.idx_train
        round_num = self.roundUpDiv(len(train_nodes_entire), batch_size)
        if self.round % round_num == 0:
            self.train_left = train_nodes_entire.copy()
        train_left_pmtt = np.random.permutation(self.train_left)
        if self.round + 1 % round_num == 0:
            train_new = self.train_left[0:]
        else:
            train_new = train_left_pmtt[0:batch_size]
            self.train_left = self.train_left[batch_size:]
        graph.subgraph['train'].layer_compute[layer_num].train_vertices = torch.LongTensor(train_new)

    def roundUpDiv(self, dividend, divisor):
        quotient = int(dividend / divisor)
        if dividend % divisor != 0:
            quotient += 1
        return quotient

    def updateNextHopTargetNodes(self, layer_id, nei_set, v2wk_map,train_vertices):
        """
        leverage l-th sampling results to generate train_vertices of (l-1)-th layer
        :param layer_id: current layer id (e.g., l)
        :param nei_set: all neighbors of local target vertices
        :param v2wk_map: indicate where vertex v is residing (map v id to worker id)
        :param graph_train: subgraph['train']
        :return: None, since (l-1)-th layer's train_vertices have been updated to graph_train
        """
        # get the target vertices of last layer
        train_vertices = train_vertices.tolist()
        nei2wk4layer, _, _ = self.buildFsthopForWorker(nei_set, v2wk_map, train_vertices)
        next_target_nodes = set()
        worker_id = context.glContext.config['id']
        next_target_nodes |= set(train_vertices)
        next_target_nodes |= set(set(nei2wk4layer[worker_id]))
        loc_neis_needed_by_otherwk = context.glContext.dgnnClientRouterForCpp.sendNodes2Wk(layer_id, nei2wk4layer)
        next_target_nodes |= set(loc_neis_needed_by_otherwk)
        # graph_train.layer_compute[layer_id - 1].train_vertices = torch.LongTensor(np.array(list(next_target_nodes)))
        return np.array(list(next_target_nodes))



    def setGraph4Cpp(self,graph):
        setLocVnum4SubGraph('train', graph)
        setSubGraphInfoForCpp('train', graph)
        # getAndSetRmtFeatForCtx(graph)


    def updateGraph(self, adjs_4_layer, graph):

        graph_train = getSubGraph(graph.labels, graph.feats, adjs_4_layer, graph.v2wk_map,
                                  graph.subgraph['train'].layer_compute[
                                      context.glContext.config['layer_num']].train_vertices.numpy().tolist(), "train",graph.subgraph['train'].rmt_nei_feat_full,graph.subgraph['train'].o2n_4rmtnei_full)
        graph.subgraph['train'] = graph_train
        self.setGraph4Cpp(graph)

