from adgnn import BackWard
from adgnn.context import context
import copy

class ECTensor(object):

    def __init__(self, tensor=None, left=None, right=None, operator=None, leaky_alpha=None, name=None,
                 requires_grad=False):
        self.tensor = tensor
        self.left = left
        self.right = right
        self.operator = operator
        self.requires_grad = requires_grad
        self.isLeaf = False
        self.grad = None
        self.backwardF = None
        self.root = None
        self.layer_id = None  # to be deleted
        self.name = name
        if leaky_alpha is not None:
            self.leaky_alpha = leaky_alpha
        if operator == 'mm':
            self.backwardF = BackWard.MmBackward
        elif operator == 'spmm':
            self.backwardF = BackWard.MmBackward_spmm
        elif operator == 'nllloss':
            self.backwardF = BackWard.NllLossBackward
        elif operator == 'log_softmax':
            self.backwardF = BackWard.LogSoftmaxBackward
        elif operator == 'relu':
            self.backwardF = BackWard.ReluBackward0
        elif operator == 'leaky_relu':
            self.backwardF = BackWard.LeakyReluBackward0
        elif operator == 'add':
            self.backwardF = BackWard.AddBackward
        elif operator == 'get_embs':
            self.backwardF = BackWard.GetEmbsBackward
        elif operator == 'push_embs':
            self.backwardF = BackWard.PushEmbsBackward
        elif operator=='elu':
            self.backwardF = BackWard.EluBackward
        elif operator is None:
            self.isLeaf = True

    # def __call__(self, *args, **kwargs):
    #     print("nanguo:{0}".format(self.tensor))
    #     return self.tensor


    def backward(self):
        # context.glContext.dgnnServerRouter[0].server_Barrier()
        BackWard.preOrderTraversal(self)
