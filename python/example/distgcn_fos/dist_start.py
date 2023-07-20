import sys, os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
sys.path.insert(2, BASE_PATH + '/../../')
sys.path.insert(3, BASE_PATH + '/../../../')
print(BASE_PATH)
import adgnn.util_python.param_parser as pp
import psutil
import adgnn.util_python.param_util as pu
import torch as torch
from sklearn.metrics import accuracy_score
from cmake.build.lib.pb11_ec import *
from adgnn.context import context
from example.distgcn_fos.models import GCN
from adgnn.util_python import data_trans as dt
from adgnn.distributed.engine import Engine
from multiprocessing import cpu_count
from adgnn.util_python.timecounter import time_counter
from adgnn.sample.agg_difference import aggDiff
from adgnn.sample.bns_gcn import bns_gcn
from adgnn.sample.graphsage import random_sample
from adgnn.sample.fastgcn import fastgcn
from adgnn.sample.cluster_gcn import cluster_gcn
import numpy as np
from python.adgnn.util_python.evaluation import *
import copy
from adgnn.util_python.evaluation import evaluator
import gc
import time
from adgnn.sample.agl import fixed_sample
from adgnn.util_python.evaluate_ad import eval_ad
import dgl
import torch.nn.functional as F
import torch.optim as optim
from example.distgcn_fos.mylog import get_logger
from sklearn.metrics import f1_score

mlog = get_logger()

cpu_num = cpu_count()
# print("cpu num:{0}".format(cpu_num))
# os.environ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(int(cpu_num / 2))

torch.manual_seed(1)

time_str = time.strftime('%Y%m%d%H%M%S')
# log_file = "/mnt/data/output/" + "Output" + time_str


# class Logger(object):
#     def __init__(self, filename=log_file):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass


# print(os.path.dirname(__file__))
# print('------------------')
# var1='hello'
# var2='world'
# print('var1:%s\nvar2: %s\n'%(var1,var2))
def combine_range(lis):
    """
    given several ranges in the format of [(st1, end1), (st2, end2)...]
    return combined ranges
    """
    lis.sort(key=lambda x: x[0])
    res = [lis[0]]
    for st, end in lis[1:]:
        if st < res[-1][1]:
            # cur_start < previous end, overlap
            if end <= res[-1][1]:
                # cur_range covered by prev_range
                continue
            else:
                prev = list(res.pop())
                prev[1] = end
                res.append(tuple(prev))
        elif st == res[-1][1]:
            # concat
            prev = list(res.pop())
            prev[1] = end
            res.append(tuple(prev))
        else:
            res.append((st, end))
    return res


def buildDGLGraph():
    root_path = context.glContext.config['data_path']
    patition_method = context.glContext.config['partitionMethod']
    worker_num = context.glContext.config['worker_num']
    id = context.glContext.config['id']
    train_num = int(context.glContext.config['train_num'] / worker_num)
    val_num = int(context.glContext.config['val_num'] / worker_num)
    test_num = int(context.glContext.config['test_num'] / worker_num)

    if root_path != '/':
        root_path += '/'
    path = root_path + patition_method + str(worker_num) + '/part' + str(id) + '/'

    nodes = []
    labels = []
    with open(path + 'label.txt', 'r') as file:
        label = file.readlines()
        for elem in label:
            id_label = elem.strip().split('\t')
            nodes.append(int(id_label[0]))
            labels.append(int(id_label[1]))

    nodes_set = set(nodes)
    from_nodes = []
    to_nodes = []
    with open(path + 'adj.txt', 'r') as file:
        adjs = file.readlines()
        for elem in adjs:
            id_neis = elem.strip().split('\t')
            id = int(id_neis[0])
            neis = [int(i) for i in id_neis[1:]]
            to_nodes.append(id)
            from_nodes.append(id)
            for nei in neis:
                if nodes_set.__contains__(nei):
                    to_nodes.append(id)
                    from_nodes.append(int(nei))

    encoding_dict = {elem: i for i, elem in enumerate(nodes)}
    from_nodes = [encoding_dict[elem] for elem in from_nodes]
    to_nodes = [encoding_dict[elem] for elem in to_nodes]

    features = [[] for i in range(len(nodes))]
    with open(path + 'feat.txt', 'r') as file:
        feat_data = file.readlines()
        for elem in feat_data:
            id_feat = elem.strip().split('\t')
            id = int(id_feat[0])
            feat = [float(i) for i in id_feat[1:]]
            features[encoding_dict[id]] = feat

    u = torch.tensor(np.array(from_nodes))
    v = torch.tensor(np.array(to_nodes))
    g = dgl.graph((u, v))
    g.ndata['feat'] = torch.tensor(np.array(features)).to(torch.float32)
    g.ndata['label'] = torch.tensor(np.array(labels))

    rand_indices = len(nodes)
    rand_indices = np.array([i for i in range(rand_indices)])
    idx_train = rand_indices[0:train_num]
    idx_val = rand_indices[train_num:train_num + val_num]
    idx_test = rand_indices[train_num + val_num:train_num + val_num + test_num]

    train_mask = [False] * len(nodes)
    val_mask = [False] * len(nodes)
    test_mask = [False] * len(nodes)
    for idx in idx_train: train_mask[idx] = True
    for idx in idx_val: val_mask[idx] = True
    for idx in idx_test: test_mask[idx] = True

    g.ndata['train_mask'] = torch.tensor(train_mask)
    g.ndata['val_mask'] = torch.tensor(val_mask)
    g.ndata['test_mask'] = torch.tensor(test_mask)

    return g


class MOSNeighborSamplerInductive(dgl.dataloading.Sampler):
    def __init__(self, node_budget, num_layers, g, train_mask, bulk_decomp=1):
        super().__init__()
        # direct arguments
        self.node_budget = node_budget
        self.num_layers = num_layers
        self.g = g
        self.train_mask = train_mask
        self.bulk_decomp = bulk_decomp

        # infered arguments
        assert self.train_mask is None
        self.total_nodes = self.g.num_nodes()
        self.num_batches = self.total_nodes // self.node_budget + 1

        # utilities
        self.N = 20 * self.num_batches * self.bulk_decomp  # num of small blocks
        self.start_idxs = np.random.choice(range(self.g.num_nodes()), size=(self.N,))
        self.start_idxs = torch.from_numpy(self.start_idxs)

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        """
        return three elements: final_ranges, input_nodes, subg
        final_ranges means the ranges of all sampled nodes, in Inductive also seeds,
        it means input_nodes, which we use to induce the subg
        """
        if self.n < self.num_batches:

            # first get the node_ranges of the composed subgraph
            st_list = self.start_idxs[self.n * self.bulk_decomp:(self.n + 1) * self.bulk_decomp]
            range_list = []
            for st in st_list:
                end = st + self.node_budget // self.bulk_decomp
                if end > self.total_nodes:
                    range_list.append((st, self.total_nodes))
                    range_list.append((0, end - self.total_nodes))
                else:
                    range_list.append((st, end))
            final_ranges = combine_range(range_list)

            # then prepare the node ids and get subgraph
            input_nodes = torch.cat([torch.arange(st, end) for st, end in final_ranges])
            subg = self.g.subgraph(input_nodes)

            self.n += 1
        else:
            rp = torch.randperm(len(self.start_idxs))
            self.start_idxs = self.start_idxs[rp]
            raise StopIteration()

        return final_ranges, None, subg


def load_subtensor_mos(nfeat, labels, final_ranges, device, seeds=None):
    batch_inputs = torch.cat([nfeat[st:end].to(device) for st, end in final_ranges])
    if seeds is None:
        batch_labels = torch.cat([labels[st:end].to(device) for st, end in final_ranges])
    else:
        batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def setGradients(model):
    with torch.no_grad():
        for id in context.glContext.parameters.keys():
            context.glContext.gradients[id] = model.parameters_collection[id].grad / context.glContext.config[
                'worker_num']


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    # val_g = g.subgraph(g.ndata['val_mask'])
    # test_g = g.subgraph(g.ndata['test_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g


def compute_acc(pred, labels, multilabel=False):
    """
    Compute the f1-micro of prediction given the labels.
    """
    y_pred = pred.cpu()
    labels = labels.cpu()
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = torch.argmax(y_pred, dim=1)

    return f1_score(labels, y_pred, average="micro")


def evaluate(model, g, nfeat, labels, nids, device, multilabel):
    """
    Evaluate the model on the validation/test set specified by nids.
    g : The entire graph.
    nfeat : The features of all the nodes.
    labels : The labels of all the nodes.
    nids : list of node Ids for validation/test
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device,
                               int(context.glContext.config['val_num'] / context.glContext.config['worker_num']), 1)
    model.train()
    if isinstance(nids, list):
        accs = [compute_acc(pred[nid], labels[nid], multilabel) for nid in nids]
    else:
        assert isinstance(nids, torch.Tensor)
        accs = compute_acc(pred[nids], labels[nids], multilabel)
    return accs


class GCN_Engine(Engine):
    def run_gnn(self):
        # time_counter.start("load_data")
        epoch = context.glContext.config['iterNum']
        # graph=dt.load_data()
        graph_sample = buildDGLGraph()
        # context.glContext.sample.initSampledGraph()
        device = 'cpu'
        train_g, val_g, test_g = inductive_split(graph_sample)
        train_nfeat = train_g.ndata.pop('feat')
        val_nfeat = val_g.ndata.pop('feat')
        test_nfeat = test_g.ndata.pop('feat')
        train_labels = train_g.ndata.pop('label')
        val_labels = val_g.ndata.pop('label')
        test_labels = test_g.ndata.pop('label')

        # train_g = graph_sample.subgraph(graph_sample.ndata['train_mask'])
        train_mask = train_g.ndata['train_mask']

        train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
        val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = torch.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]
        # dataloader
        sample_fanout=min(len(train_labels),context.glContext.config['sample_num'][0])
        dataloader = MOSNeighborSamplerInductive(sample_fanout, context.glContext.config['layer_num'], train_g, None, 8)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        # best
        best_eval = 0
        best_test = 0

        # Training loop
        for e in range(1, epoch + 1):
            tic = time.time()

            for step, (final_ranges, seeds, subgs) in enumerate(dataloader):
                # Load the input features as well as output labels
                ts_sample = time.time()
                batch_inputs, batch_labels = load_subtensor_mos(train_nfeat, train_labels, final_ranges, device, seeds)
                subgs = subgs.int().to(device)
                adj = subgs.adj()
                if e==2:
                    print(adj)
                te_sample = time.time()
                batch_pred = model(batch_inputs, adj)
                loss = F.nll_loss(batch_pred, batch_labels)
                # optimizer.zero_grad()
                loss.backward()
                # optimizer.step()
                setGradients(model)
                pu.updateParam(model)

            toc = time.time()
            mlog('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Total Time(s) {:.4f} | Sampling Time {:.4f}'.format(
                e, step, loss.item(), toc - tic, te_sample - ts_sample))

            if epoch % 1 == 0:
                eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device, False)
                test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, False)
                acc_avrg = self.getAccAvrg([1, 1, 1], 0, eval_acc, test_acc, is_avrg=True)
                mlog('Eval Acc: {:.4f}, Test Acc: {:.4f}'.format(acc_avrg['val'], acc_avrg['test']))
                if eval_acc > best_eval:
                    mlog('new best eval: {:.4f}'.format(eval_acc))
                    best_eval = eval_acc
                    best_test = test_acc




if __name__ == "__main__":
    pp.parserInit()
    # if context.glContext.config['id'] == 0:
    #     sys.stdout = Logger(log_file)
    model = GCN(nfeat=context.glContext.config['feature_dim'],
                nhid=context.glContext.config['hidden'],
                nclass=context.glContext.config['class_num'],
                dropout=0)
    gcn_engine = GCN_Engine(model)
    gcn_engine()
