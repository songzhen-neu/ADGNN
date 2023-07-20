# import adgnn.context.context as context
import adgnn.util_python.param_util as pu
from cmake.build.lib.pb11_ec import *
from adgnn.context import context


class Engine:
    def __init__(self, model):
        self.model = model
        self.dgnnClient = None
        print("construct engine")

    def __call__(self):
        self.run()

    def run(self):
        if context.glContext.config['role'] == 'server':
            context.glContext.worker_id = context.glContext.config['id']
            ServiceImpl.RunServerByPy(context.glContext.config['server_address'][context.glContext.config['id']],
                                      context.glContext.worker_id)

        elif context.glContext.config['role'] == 'worker':
            context.glContext.initCluster()
            context.glContext.setWorkerContext()
            self.dgnnClient = context.glContext.dgnnClient

            # assign parameter

            pu.assignParam(self.model)
            context.glContext.dgnnServerRouter[0].server_Barrier()

            self.run_gnn()

        # elif context.glContext.config['role'] == 'master':
        #     context.glContext.worker_id = context.glContext.config['id']
        #     ServiceImpl.RunServerByPy(context.glContext.config['master_address'], 0)

    def run_gnn(self):
        pass

    def train(self):
        context.glContext.is_train = True

    def eval(self):
        context.glContext.is_train = False

    def getAccAvrg(self, split_num, acc_train=0, acc_val=0, acc_test=0,is_avrg=False):
        train_num_all_worker = context.glContext.config['train_num']
        val_num_all_worker = context.glContext.config['val_num']
        test_num_all_worker = context.glContext.config['test_num']
        train_num=split_num[0]
        val_num=split_num[1]
        test_num=split_num[2]
        acc_avrg = {}

        if is_avrg:
            acc_entire = context.glContext.dgnnServerRouter[0].sendAccuracy(acc_val,
                                                                            acc_train,
                                                                            acc_test)
            acc_avrg['train']=acc_entire['train']/context.glContext.config['worker_num']
            acc_avrg['val']=acc_entire['val']/context.glContext.config['worker_num']
            acc_avrg['test']=acc_entire['test']/context.glContext.config['worker_num']
            return acc_avrg

        acc_entire = context.glContext.dgnnServerRouter[0].sendAccuracy(acc_val * val_num,
                                                                        acc_train * train_num,
                                                                        acc_test * test_num)
        context.glContext.dgnnServerRouter[0].server_Barrier()

        # for key in acc_entire:
        acc_avrg['train'] = acc_entire['train'] / (float(train_num_all_worker))
        acc_avrg['val'] = acc_entire['val'] / (float(val_num_all_worker))
        acc_avrg['test'] = acc_entire['test'] / (float(test_num_all_worker))
        return acc_avrg
