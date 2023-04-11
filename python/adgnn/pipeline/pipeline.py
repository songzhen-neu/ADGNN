class Pipeline():
    def __init__(self, graph, layer_id, batch_num, **kwargs):
        self.graph = graph
        self.layer_id = layer_id
        self.status = graph.status
        self.data = kwargs
        self.batch_num = batch_num

    def divideBatch(self, batch_size):
        pass

    def compFunc(self, **kwargs):
        pass

    def commFunc(self, batch_id, push_id, push_emb):
        pass

    def startPipe(self):
        pass

    def buildCompGraph(self,**kwargs):
        pass
