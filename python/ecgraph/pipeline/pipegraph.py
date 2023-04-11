from ecgraph.pipeline.pipeline import Pipeline
import ecgraph.util_python.remote_access as rmt


class PipeGraph(Pipeline):
    def startPipe(self):
        self.divideBatch()

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
