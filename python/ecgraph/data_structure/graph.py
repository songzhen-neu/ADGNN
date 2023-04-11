class Graph():
    def __init__(self, graph_train, graph_val, graph_test, idx_train, idx_test, idx_val, v2wk_map, adj, labels, feats,layer_num):
        # 这里除了agg_node_old和fsthop_for_worker是old id，其他的全部是编码好的
        # agg_node_old是需要进行聚合全部顶点
        self.subgraph = {}
        self.subgraph['train'] = graph_train
        self.subgraph['val'] = graph_val
        self.subgraph['test'] = graph_test
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        # the number of local training vertices
        self.train_num = len(idx_train)
        self.test_num = len(idx_test)
        self.val_num = len(idx_val)
        self.vnum_local = self.train_num + self.test_num + self.val_num
        self.v2wk_map = v2wk_map
        self.adj = adj
        self.labels = labels
        self.feats = feats
        self.layer_num=layer_num
