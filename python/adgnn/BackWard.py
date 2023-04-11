import torch
import numpy as np
from adgnn.context import context
from cmake.build.lib.pb11_ec import *
from adgnn.util_python.timecounter import time_counter

# f means former, l means latter;
# e.g., mm: xl=xf.mm(wf)

def MmBackward(xf, wf, xl_g, flag):
    if flag == 'x':
        return xl_g.mm(wf.t())
    elif flag == 'w':
        return xf.t().mm(xl_g)


def MmBackward_spmm(xf, wf, xl_g, flag):
    if flag == 'x':
        return xl_g.mm(wf.t())
    elif flag == 'w':
        return torch.spmm(xf.t(), xl_g)




def NllLossBackward(xf, lab, idx_train, train_num):
    g_x = torch.zeros_like(xf)
    g_x[idx_train,lab]=-1
    g_x = g_x / train_num
    return g_x


def LogSoftmaxBackward(xf_softmax, xl_g):
    xf_softmax = xf_softmax
    # x1_g = np.zeros_like(xf_softmax)
    # size_0 = len(xf_softmax)
    # size_1 = len(xf_softmax[0])
    index = torch.nonzero(xl_g).numpy()
    x1_g=-xf_softmax
    x1_g[index[:,0],index[:,1]]+=1

    # for m in range(len(index)):
    #     i = index[m][0]
    #     j = index[m][1]
    #     for k in range(size_1):
    #         if j == k:
    #             x1_g[i][k] += 1 - xf_softmax[i][k]
    #         else:
    #             x1_g[i][k] += -xf_softmax[i][k]

    x1_g = torch.FloatTensor(x1_g)
    x2_g = xl_g.mm(torch.ones(xl_g.size()[1], 1))
    x1_g = x1_g * x2_g
    return x1_g


def ReluBackward0(xf, xl_g):
    zero = torch.zeros_like(xf)
    one = torch.ones_like(xf)
    x_g = torch.where(xf > 0, one, zero)
    return x_g * xl_g


def LeakyReluBackward0(xf, xl_g, alpha):
    alpha_X = 0
    if alpha != 0:
        alpha_X = torch.ones_like(xf) * alpha
    else:
        print("error: alpha cannot equal to 0")
    one = torch.ones_like(xf)
    x_g = torch.where(xf > 0, one, alpha_X)
    return x_g * xl_g


def AddBackward(xl_g, ifbias):
    if ifbias:
        return xl_g.detach().numpy().sum(axis=0)
    else:
        return xl_g


def GetEmbsBackward(xl_g,layer_id):
    # transfer all data to c++ backend
    emb_grad=context.glContext.dgnnClientRouterForCpp.setAndSendG(
        layer_id,
        xl_g.detach().numpy()
    )
    return torch.FloatTensor(emb_grad)


def PushEmbsBackward(xl_g,layer_id):
    # transfer all data to c++ backend
    emb_grad=context.glContext.dgnnClientRouterForCpp.setAndSendG(
        layer_id,
        xl_g.detach().numpy()
    )
    return torch.FloatTensor(emb_grad)


def setUpdateGrad(node):
    if node.name is not None:
        context.glContext.gradients[node.name] = node.grad


def compRight(node):
    if node.right is not None:
        if not node.right.requires_grad:
            return
    if node.operator == 'mm':
        if node.right is not None:
            time_counter.start('Backward_mm')
            node.right.grad = node.backwardF(node.left.tensor, node.right.tensor, node.grad, 'w')
            setUpdateGrad(node.right)
            time_counter.end('Backward_mm')
    elif node.operator == 'spmm':
        if node.right is not None:
            time_counter.start('Backward_spmm')
            node.right.grad = node.backwardF(node.left.tensor, node.right.tensor, node.grad, 'w')
            setUpdateGrad(node.right)
            time_counter.end('Backward_spmm')
    elif node.operator == 'add':
        if node.right is not None:
            time_counter.start('Backward_add')
            if node.right.tensor.shape == node.grad.shape:
                node.right.grad = node.backwardF(node.grad, False)
                setUpdateGrad(node.right)
            else:
                node.right.grad = node.backwardF(node.grad, True)
                setUpdateGrad(node.right)
            time_counter.end('Backward_add')


def compLeft(node):
    if node.left is not None:
        if not node.left.requires_grad:
            return
    if node.operator == 'nllloss':
        # output label idx train_number
        if node.left is not None:
            time_counter.start('Backward_nllloss')
            node.left.grad = node.backwardF(node.left.tensor, node.right.tensor,
                                            [i for i in range(node.right.tensor.shape[0])],
                                            context.glContext.config['train_num'])
            setUpdateGrad(node.left)
            time_counter.end('Backward_nllloss')
    elif node.operator == 'mm':
        if node.left is not None:
            time_counter.start('Backward_mm')
            node.left.grad = node.backwardF(node.left.tensor, node.right.tensor, node.grad, 'x')
            setUpdateGrad(node.left)
            time_counter.end('Backward_mm')
    elif node.operator == 'spmm':
        if node.left is not None:
            time_counter.start('Backward_spmm')
            node.left.grad = node.backwardF(node.left.tensor, node.right.tensor, node.grad, 'x')
            setUpdateGrad(node.left)
            time_counter.end('Backward_spmm')
    elif node.operator == 'log_softmax':
        if node.left is not None:
            time_counter.start('Backward_logsoftmax')
            node.left.grad = node.backwardF(node.left.tensor, node.grad)
            setUpdateGrad(node.left)
            time_counter.end('Backward_logsoftmax')
    elif node.operator == 'relu':
        if node.left is not None:
            time_counter.start('Backward_relu')
            node.left.grad = node.backwardF(node.left.tensor, node.grad)
            setUpdateGrad(node.left)
            time_counter.end('Backward_relu')
    elif node.operator == 'leaky_relu':
        if node.left is not None:
            time_counter.start('Backward_leakyrelu')
            node.left.grad = node.backwardF(node.left.tensor, node.grad, node.leaky_alpha)
            setUpdateGrad(node.left)
            time_counter.end('Backward_leakyrelu')
    elif node.operator == 'get_embs':
        if node.left is not None:
            time_counter.start('Backward_getembs')
            node.left.grad = node.backwardF(node.grad,node.layer_id)
            setUpdateGrad(node.left)
            time_counter.end('Backward_getembs')
    elif node.operator == 'push_embs':
        if node.left is not None:
            time_counter.start('Backward_pushembs')
            node.left.grad = node.backwardF(node.grad,node.layer_id)
            setUpdateGrad(node.left)
            time_counter.end('Backward_pushembs')
    elif node.operator == 'add':
        if node.left is not None:
            time_counter.start('Backward_add')
            if node.left.tensor.shape == node.grad.shape:
                node.left.grad = node.backwardF(node.grad, False)
                setUpdateGrad(node.left)
            else:
                node.left.grad = node.backwardF(node.grad, True)
                setUpdateGrad(node.left)
            time_counter.end('Backward_add')


def preOrderTraversal(node):
    if node.operator is None:
        return
    else:
        compLeft(node)
        if node.left is not None:
            preOrderTraversal(node.left)
        compRight(node)
        if node.right is not None:
            preOrderTraversal(node.right)
