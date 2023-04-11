from ecgraph.context import context
from ecgraph.structure.subgraph import Subgraph
class Graph():
    def __init__(self):
        self.subgraphs= {}
        self.subgraphs["train"]=Subgraph()
        self.subgraphs["val"]=Subgraph()
        self.subgraphs["test"]=Subgraph()
        self.graph_mode=None


