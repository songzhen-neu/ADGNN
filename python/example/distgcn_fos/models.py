import torch.nn as nn
from example.distgcn_fos.layers import GraphConvolution
import torch
import torch.nn.functional as F
from adgnn.util_python.remote_access import catCacheFeature
import adgnn
from adgnn.context import context
import dgl

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        self.layerNum = len(nhid) + 1
        self.dropout = dropout
        self.gc = {}
        self.parameters_collection={}
        self.layers=nn.ModuleList()
        self.nhid=nhid[0]
        self.nclass=nclass
        # 1st layer is stored in gc[0]
        for i in range(1, self.layerNum + 1):
            if i == 1:
                self.gc[i] = GraphConvolution(nfeat, nhid[0], 1)
            elif i != self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nhid[i - 1], i)
            elif i == self.layerNum:
                self.gc[i] = GraphConvolution(nhid[i - 2], nclass, i)
            self.layers.append(self.gc[i])

        for i in range(1,len(self.gc)+1):
            self.parameters_collection['w'+str(i)]=self.gc[i].weight
            self.parameters_collection['b'+str(i)]=self.gc[i].bias

    def forward(self, feat_data,adj):
        x=feat_data
        for i in range(1, self.layerNum + 1):
            x = self.gc[i](x, adj)
            if not i == self.layerNum:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)

    def inference_model(self,graph):
        context.glContext.graphBuild.setGraphMode(graph.graph_mode)
        x =graph.feat_data # Dorylus
        # x = catCacheFeature(graph) # PipeGraph
        for i in range(1, self.layerNum + 1):
            x = self.gc[i].inference_layer(x, graph)
            if not i == self.layerNum:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.nhid if l != len(self.layers) - 1 else self.nclass)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            #for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                block = block.int().to(device)
                adj=block.adj().t()
                h = x[input_nodes].to(device)
                h = layer(h,adj)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    # h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y