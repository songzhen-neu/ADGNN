import sys, os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
sys.path.insert(2, BASE_PATH + '/../../')
sys.path.insert(3, BASE_PATH + '/../../../')
print(BASE_PATH)
import adgnn.util_python.param_parser as pp
import psutil
import adgnn.Function as F
import adgnn.util_python.param_util as pu
import torch as torch
from sklearn.metrics import accuracy_score
from cmake.build.lib.pb11_ec import *
from adgnn.context import context
from example.distgcn.models import GCN
from adgnn.util_python import data_trans as dt
from adgnn.distributed.engine import Engine
from multiprocessing import cpu_count
from adgnn.util_python.timecounter import time_counter
from adgnn.sample.agg_difference import aggDiff
from adgnn.sample.bns_gcn import bns_gcn
from adgnn.sample.graphsage import random_sample
from adgnn.sample.fastgcn import fastgcn
from adgnn.sample.cluster_gcn import cluster_gcn
from adgnn.sample.fos import fos
import numpy as np
from python.adgnn.util_python.evaluation import *
import copy
from adgnn.util_python.evaluation import evaluator
import gc
import time
from adgnn.sample.agl import fixed_sample
from adgnn.util_python.evaluate_ad import eval_ad

cpu_num = cpu_count()
# print("cpu num:{0}".format(cpu_num))
# os.environ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(int(cpu_num/2))

torch.manual_seed(1)

time_str = time.strftime('%Y%m%d%H%M%S')
log_file = "/mnt/data/output/" + "Output" + time_str


class Logger(object):
    def __init__(self, filename=log_file):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# print(os.path.dirname(__file__))
# print('------------------')
# var1='hello'
# var2='world'
# print('var1:%s\nvar2: %s\n'%(var1,var2))


class GCN_Engine(Engine):
    def run_gnn(self):
        time_counter.start("load_data")
        graph = dt.load_data()
        time_counter.end("load_data")
        epoch = context.glContext.config['iterNum']
        samp_num = context.glContext.config['sample_num']
        batch_size = context.glContext.config['batch_size']
        comm_fo = context.glContext.config['comm_fo']
        enable_pc = context.glContext.config['enable_pc']
        m_ad = context.glContext.config['m_ad']
        enab_adap_m = context.glContext.config['enab_adap_m']
        sample_method = context.glContext.config['sample_method']
        dim_itvs = context.glContext.config['dim_itvs']
        adcomp_num = context.glContext.config['adcomp_num']
        nei_prune = context.glContext.config['nei_prune']
        # context.glContext.sample.initSampledGraph()
        for e in range(epoch):
            self.train()
            time_counter.start('train_time')
            time_counter.start('sample')
            if sample_method != 'none':
                if sample_method == 'ad':
                    graph = aggDiff.sample(samp_num, e, batch_size, comm_fo=comm_fo, enable_pc=enable_pc, m=m_ad,
                                           enab_adap_m=enab_adap_m, dim_itvs=dim_itvs,
                                           adcomp_num=adcomp_num,nei_prune=nei_prune,
                                           model=self.model,model_type='GCN')
                elif sample_method == 'ad_every':
                    graph = aggDiff.sample_every(samp_num, e, batch_size, comm_fo=comm_fo, enable_pc=enable_pc, m=m_ad,
                                                 enab_adap_m=enab_adap_m, dim_itvs=dim_itvs,
                                                 adcomp_num=adcomp_num,
                                                 model=self.model,model_type='GCN')
                elif sample_method == 'random':
                    graph = random_sample.sample(samp_num, e)
                elif sample_method == 'fixed':
                    graph = fixed_sample.sample(graph, samp_num, e, batch_size)
                elif sample_method == 'bns':
                    graph = bns_gcn.sample(graph, samp_num, e, batch_size)
                elif sample_method == 'fastgcn':
                    graph = fastgcn.sample(graph, samp_num, e, batch_size)
                # elif sample_method == 'fos':
                #     graph = fos.sample(graph, samp_num, e, batch_size)
                elif sample_method == 'clustergcn':
                    graph = cluster_gcn.sample(graph, samp_num[0], e, batch_size, cluster_number=samp_num[1])
                elif sample_method == 'ad_local':
                    graph = aggDiff.sample_local(samp_num, e, batch_size, comm_fo=comm_fo, enable_pc=enable_pc, m=m_ad,
                                       enab_adap_m=enab_adap_m, dim_itvs=dim_itvs,
                                       adcomp_num=adcomp_num,nei_prune=nei_prune,
                                       model=self.model,model_type='GCN')
                else:
                    print("no such sample method!")
                    exit(-1)
                if e == 2:
                    for i in range(1, context.glContext.config['layer_num'] + 1):
                        print(graph.subgraphs['train'].graphlayers[i].adj.tensor)
            # eval_ad.evalOptimalSet(e,graph)
            #eval_ad.evalAD(model,e,graph.subgraphs['train'])
            time_counter.end('sample')
            # time_counter.start('eval_graph_time')
            # evaluator.evalSampledGraph(graph.subgraph['train'])
            # time_counter.end('eval_graph_time')
            time_counter.start('fp')
            output = model(graph.subgraphs['train'])
            time_counter.end('fp')
            loss_train = F.nll_loss(output, graph.subgraphs['train'].label)
            acc_train = accuracy_score(graph.subgraphs['train'].label.tensor.cpu().detach().numpy(),
                                       output.tensor.cpu().detach().numpy().argmax(axis=1))
            time_counter.start('bp')
            loss_train.backward()
            time_counter.end('bp')

            time_counter.start('update_model')
            pu.updateParam(model)
            time_counter.end('update_model')
            time_counter.end('train_time')

            self.eval()

            if e % context.glContext.config['print_result_interval'] == 0:
                gc.collect()
                output = model(context.glContext.graph_full.subgraphs['val'])

                acc_val = accuracy_score(context.glContext.graph_full.subgraphs['val'].label.tensor.cpu().detach().numpy(),
                                         output.tensor.cpu().detach().numpy().argmax(axis=1))

                acc_test = 0


                output = model(context.glContext.graph_full.subgraphs['test'])
                acc_test = accuracy_score(
                    context.glContext.graph_full.subgraphs['test'].label.tensor.cpu().detach().numpy(),
                    output.tensor.cpu().detach().numpy().argmax(axis=1))

                train_num=len(context.glContext.graph_full.subgraphs['train'].label.tensor)
                val_num=len(context.glContext.graph_full.subgraphs['val'].label.tensor)
                test_num=len(context.glContext.graph_full.subgraphs['test'].label.tensor)

                acc_avrg = self.getAccAvrg([train_num,val_num,test_num], acc_train, acc_val, acc_test)

                print('Epoch: {:04d}'.format(e + 1),
                      'loss_train: {:.4f}'.format(loss_train.tensor),
                      'acc_train: {:.4f}'.format(acc_avrg['train']),
                      'acc_val: {:.4f}'.format(acc_avrg['val']),
                      'acc_test: {:.4f}'.format(acc_avrg['test']),
                      'train_time: {:.4f}'.format(time_counter.time_list['train_time'][-1]),
                      'fp_time: {:.4f}'.format(time_counter.time_list['fp'][-1]),
                      'bp_time: {:.4f}'.format(time_counter.time_list['bp'][-1]),
                      'update_model_time: {:.4f}'.format(time_counter.time_list['update_model'][-1]),
                      'sample_time: {:.4f}'.format(time_counter.time_list['sample'][-1]),
                      "memory:{:.4f}G".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

                if acc_avrg['test'] > evaluator.maxacc_test:
                    evaluator.maxacc_test = acc_avrg['test']
                    evaluator.r_maxacc = e

                if e >= evaluator.r_maxacc + 10000000:
                    print('GNN model has been converged!')
                    break

        time_counter.printAvrgTime()
        time_counter.printTotalTime()
        evaluator.printGraphInfo()
        eval_ad.printAD()


if __name__ == "__main__":
    pp.parserInit()
    if context.glContext.config['id'] == 0:
        sys.stdout = Logger(log_file)
    model = GCN(nfeat=context.glContext.config['feature_dim'],
                nhid=context.glContext.config['hidden'],
                nclass=context.glContext.config['class_num'],
                dropout=0)
    gcn_engine = GCN_Engine(model)
    gcn_engine()
