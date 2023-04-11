import adgnn
import torch.nn.functional as F
import torch
from adgnn.context import context


def set_requires_grad(root, left, right):
    if left is not None:
        if left.requires_grad:
            root.requires_grad = True
    if right is not None:
        if right.requires_grad:
            root.requires_grad = True


def mm(x1, w1):
    x2 = x1.tensor.mm(w1.tensor)
    x2 = adgnn.ECTensor(x2, x1, w1, 'mm', None)
    if context.glContext.is_train:
        set_requires_grad(x2, x1, w1)
        x1.root = x2
        w1.root = x2
        context.glContext.compGraph = x2
    return x2

def mm_pipe(o1,x1,w1):
    x2 = adgnn.ECTensor(o1, x1, w1, 'mm', None)
    if context.glContext.is_train:
        set_requires_grad(x2, x1, w1)
        x1.root = x2
        w1.root = x2
        context.glContext.compGraph = x2
    return x2

def push_pipe(output,input,layer_id):
    embs = adgnn.ECTensor(output, input, None, 'push_embs', None, requires_grad=True)
    input.root = embs
    embs.layer_id = layer_id
    return embs


def spmm(adj, x1):
    x2 = torch.spmm(adj.tensor,x1.tensor)
    x2 = adgnn.ECTensor(x2, adj, x1, 'spmm', None)
    if context.glContext.is_train:
        set_requires_grad(x2, adj, x1)
        adj.root = x2
        x1.root = x2
        context.glContext.compGraph = x2
    return x2

def leaky_relu(x1, leaky_alpha):
    x2 = F.leaky_relu(x1.tensor, leaky_alpha)
    x2 = adgnn.ECTensor(x2, x1, None, 'leaky_relu', leaky_alpha)
    if context.glContext.is_train:
        set_requires_grad(x2, x1, None)
        x1.root = x2
        context.glContext.compGraph = x2
    return x2


def relu(x1):
    x2 = F.relu(x1.tensor)
    x2 = adgnn.ECTensor(x2, x1, None, 'relu', None)
    if context.glContext.is_train:
        set_requires_grad(x2, x1, None)
        x1.root = x2
        context.glContext.compGraph = x2
    return x2


def log_softmax(x1, dim):
    x2 = F.softmax(x1.tensor, dim=dim).detach()
    x3 = F.log_softmax(x1.tensor, dim=dim)
    x1.tensor.data = x2
    x3 = adgnn.ECTensor(x3, x1, None, 'log_softmax', None)
    if context.glContext.is_train:
        set_requires_grad(x3, x1, None)
        x1.root = x3
        context.glContext.compGraph = x3
    return x3


def nll_loss(x1, lab):
    loss = F.nll_loss(x1.tensor, lab.tensor)
    loss = adgnn.ECTensor(loss, x1, lab, 'nllloss', None)
    if context.glContext.is_train:
        set_requires_grad(loss, x1, None)
        x1.root = loss
        lab.root = loss
        context.glContext.compGraph = loss
    return loss


def add(x1, x2):
    x_result = x1.tensor + x2.tensor
    x_result = adgnn.ECTensor(x_result, x1, x2, 'add', None)
    if context.glContext.is_train:
        set_requires_grad(x_result, x1, x2)
        x1.root = x_result
        x2.root = x_result
        context.glContext.compGraph = x_result
    return x_result






