from ecgraph.structure.graphlayer import Graphlayer


class Subgraph():
    def __init__(self):
        self.graphlayers = {}
        self.feat_data=None
        self.status=None
        self.graph_mode=None
        self.label=None

    def getAdj(self, layer_id):
        return self.graphlayers[layer_id].adj
