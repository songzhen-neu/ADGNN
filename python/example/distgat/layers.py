import os

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from adgnn.util_python.timecounter import TimeCounter
from adgnn.context.context import glContext
from adgnn.pipeline.pipegraph import PipeGraph
from adgnn.pipeline.pipe_dorylus import PipeDorylus
from adgnn.util_python.timecounter import time_counter
import time




class pushEmbsFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, layer_id, graph, input):
        # assert indices.requires_grad == False
        """
        push embeddings based on the (layer_id-1)-th push_2_worker_nodes
        layer_id begins from 1, so in the first layer the system needs to push 0-th embeddings
        :param layer_id:
        :param status:
        :param input:
        :param graph:
        :return:
        """
        ctx.layer_id=layer_id
        embs=glContext.dgnnClientRouterForCpp.pushEmbs(layer_id,graph.status,graph.graph_mode,input.cpu().detach().numpy())
        embs = torch.FloatTensor(embs).to(glContext.config['device'])
        return embs


    @staticmethod
    def backward(ctx, grad_output):
        layer_id=ctx.layer_id
        # print(grad_output)
        print("1111111111111111111111")

        emb_grad = glContext.dgnnClientRouterForCpp.setAndSendG(
            layer_id,
            grad_output.cpu().detach().numpy()
        )


        # print(emb_grad)
        return None,None,torch.FloatTensor(emb_grad).to(glContext.config['device'])



class PushEmbs(nn.Module):
    def forward(self, layer_id, graph, input):
        return pushEmbsFunction.apply(layer_id, graph, input)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b,layer_id):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)

        ctx.save_for_backward(a, b,torch.tensor(layer_id))
        ctx.N = shape[1]
        # ctx.N = glContext.config['data_num']
        # print(a.size())
        # print(b.size())
        # start=time.time()
        result=torch.matmul(a, b)
        # end=time.time()
        # print("time: {:.04f} s".format(end-start))
        return result

    @staticmethod
    def backward(ctx, grad_output):

        a, b,layer_id = ctx.saved_tensors
        grad_values = grad_b = None
        # print("11111111111111111")

        if ctx.needs_input_grad[1]:
            # start=time.time()
            # grad_a_dense = grad_output.matmul(b.t())
            grad_values=torch.sum(torch.mul(grad_output[a._indices()[0, :]],b[a._indices()[1, :]]),dim=1)
            # grad_a_dense=torch.mm(grad_output,b.t())
            # end=time.time()
            # print("time: {:.04f} s".format(end-start))
            # edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            # grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)

        return None, grad_values, None, grad_b,None


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b,layer_id):
        return SpecialSpmmFunction.apply(indices, values, shape, b,layer_id)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, layer_id, head_id, dropout,alpha,concat):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.head_id=head_id
        self.layer_id = layer_id
        self.pushEmbs=PushEmbs()

        # torch.manual_seed(72)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(glContext.config['device'])
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        glContext.parameters['w' + str(layer_id)+'-'+str(head_id)] = self.W.data.flatten().detach().tolist()
        self.W.retain_grad()

        self.a = nn.Parameter(torch.empty(size=(1, 2*out_features))).to(glContext.config['device'])
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        glContext.parameters['a' + str(layer_id)+'-'+str(head_id)] = self.a.data.flatten().detach().tolist()
        self.a.retain_grad()

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, x, adj,graph):
        time_counter.start("fp-getadj")
        # adj = graph.getAdj(self.layer_id)
        # adj=adj.tensor
        time_counter.end("fp-getadj")
        shape_1=adj.shape[0]
        shape_2=adj.shape[1]

        # N = input.size()[0]
        with torch.no_grad():
            edge = adj.coalesce().indices()

        if self.layer_id==1:
            x=x
        else:
            x = self.pushEmbs(self.layer_id, graph,x)

        # with torch.no_grad():
        #     self.W[self.W!=self.W]=0

        h = torch.mm(x, self.W)

        # print("max_W:{0},max_x:{1},h:{2}".format(torch.max(self.W),torch.max(x),torch.max(h)))
        assert not torch.isnan(h).any()


        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # torch.max(h)
        # edge: 2*D x E
        a_mm_edge=self.a.mm(edge_h).squeeze()
        edge_e = torch.exp(-self.leakyrelu(a_mm_edge))
        # print("max_a:{0},max_edge_e:{1},max_a_mm_edge:{2}".format(torch.max(self.a),torch.max(edge_e),torch.max(a_mm_edge)))
        assert not torch.isnan(edge_e).any()
        # edge_e: E


        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([shape_1, shape_2]), torch.ones(size=(shape_2,1), device=glContext.config['device']),self.layer_id)
        e_rowsum[e_rowsum==0]=1

        edge_e = self.dropout(edge_e)
        assert not torch.isinf(edge_e).any()
        # edge_e: E
        h_prime = self.special_spmm(edge, edge_e, torch.Size([shape_1, shape_2]), h,self.layer_id)
        # print("max_h_prime:{0}".format(torch.max(h_prime)))



        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)


        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime




    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
