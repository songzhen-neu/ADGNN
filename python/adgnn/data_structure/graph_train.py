class Graph_Train():
    def __init__(self, status, train_vertices, feat, label, layer_compute,rmt_nei_feat_full=None,o2n_4rmtnei_full=None):
        self.train_vertices = train_vertices
        self.feat_data = feat
        self.label = label
        self.layer_compute = layer_compute
        self.status = status
        self.local_node_num = None
        self.rmt_nei_feat_full=rmt_nei_feat_full
        self.o2n_4rmtnei_full=o2n_4rmtnei_full

    def getAdj(self, layer_id):
        return self.layer_compute[layer_id].adj
