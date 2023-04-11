from cmake.build.lib.pb11_ec import *
from ecgraph.structure.graphlayer import Graphlayer
from ecgraph.structure.graph import Graph
import time

class Context(object):
    def foo(self):
        pass

    ifctx = True
    is_train = False

    config = {
        'server_address': {},
        'worker_address': {},
        'role': 'default',
        'id': -1,
        'server_num': 2,
        'worker_num': 2,


        # for sampling
        'sample_num':[3,3], # [sampled_metispart,all_metispart] for clustergcn
        'batch_size':100000000,
        'sample_method':'ad', # none,bns,random,ad,fastgcn,fixed,clustergcn
        # 'partitionMethod': 'clustergcn1200-',  # hash,metis
        'partitionMethod': 'hash',  # hash,metis

        # ad-sampling configuration
        'dim_itvs':[1,1], # hiddenN -> ... ->hidden1 -> feature
        'adcomp_num':1,
        'nei_prune':222,
        'comm_fo':[1100,900],
        'enable_pc':False,
        'm_ad':10,
        'enab_adap_m':False,
        'num_threads':1,


        'layer_num': 2,
        'emb_dims': [],
        'iterNum': 50,
        'lr': 0.01,
        'print_result_interval': 1,




        # 'data_path': '/mnt/data/reddit-small',
        # 'raw_data_path': '/mnt/data/reddit-small',
        # 'data_num': 232965,  # 231443
        # 'hidden': [128],
        # 'feature_dim': 602,
        # 'class_num': 41,
        # 'edge_num': 57307946,  # 11606919
        # 'train_num':153932, #153932
        # 'val_num':23699,
        # 'test_num':55334, #55334

        # 'data_path': '/mnt/data/cora/',
        # 'raw_data_path': '/mnt/data_raw/cora/',
        # 'hidden': [16],
        # 'data_num': 2708,
        # 'feature_dim': 1433,
        # 'class_num': 7,
        # 'edge_num': 5278,
        # 'train_num': 140,  # 140
        # 'val_num': 300,
        # 'test_num': 1000,


        # 'data_path': '/mnt/data/ogbn-papers100M',
        # 'raw_data_path':'/mnt/data_raw/ogbn-papers100M',
        # 'hidden': [16,16],
        # 'data_num': 1546782,
        # 'feature_dim': 128,
        # 'class_num': 172,
        # 'edge_num':13649351,
        # 'train_num':1207179, #140
        # 'val_num':125265,
        # 'test_num':214338,

        # 'data_path': '/mnt/data/ogbn-products',
        # 'raw_data_path':'/mnt/data_raw/ogbn-products',
        # 'hidden': [16],
        # 'data_num': 2449029,
        # 'feature_dim': 100,
        # 'class_num': 47,
        # 'edge_num':61859012,
        # 'train_num':196615,
        # 'val_num':39323, # 39323
        # 'test_num':2213091, #2213091

        'data_path': '/mnt/data/pubmed',
        'raw_data_path': '/mnt/data/pubmed',
        'hidden': [16],
        'data_num': 19717,
        'feature_dim': 500,
        'class_num': 3,
        'edge_num': 44324,
        'train_num': 12816,  # 12816,60
        'val_num': 1971,  # 1971,500
        'test_num': 4930,  # 4930,1000

        # 'data_path': '/mnt/data/ogbn-arxiv',
        # 'raw_data_path': '/mnt/data/ogbn-arxiv',
        # 'hidden': [128],
        # 'data_num': 169343,
        # 'feature_dim': 128,
        # 'class_num': 40,
        # 'edge_num': 1166243,
        # 'train_num': 90941,  # 12816,60
        # 'val_num': 29799,  # 1971,500
        # 'test_num': 48603,  # 4930,1000

        # 'data_path': '/mnt/data/test',
        # 'raw_data_path': '/mnt/data/test',
        # 'hidden': [4],
        # 'data_num': 16,
        # 'feature_dim': 4,
        # 'class_num': 16,
        # 'edge_num': 30,
        # 'train_num': 3,
        # 'val_num': 10,
        # 'test_num': 3,


    }

    graph_full=None
    graph_sample=None
    dgnnServerRouter = None
    dgnnClient = None
    # dgnnMasterRouter = None
    # since the explanation lock, program cannot get satisfactory parallel performance
    # dgnnClientRouterForCpp transfers the data to c++ for parallelism
    dgnnClientRouterForCpp = None
    # bypass the router to get worker router directly
    # suitable to requesting for a certain machine
    dgnnWorkerRouter = None
    graph = None
    remoteFeature = {}

    parameters = {}
    parametersForServer = {}
    gradients = {}
    gradientsForServer = {}
    train_node_num = None
    compGraph = None

    def initGraph(self):
        self.graph_full=Graph()
        self.graph_sample=Graph()
        self.graph_full.graph_mode='full'
        self.graph_sample.graph_mode='sample'
        self.graph_full.subgraphs['train'].graph_mode='full'
        self.graph_full.subgraphs['val'].graph_mode='full'
        self.graph_full.subgraphs['test'].graph_mode='full'
        self.graph_sample.subgraphs['train'].graph_mode='sample'
        for i in range(self.config['layer_num']+1):
            self.graph_full.subgraphs['train'].graphlayers[i]=Graphlayer()
            self.graph_full.subgraphs['train'].status='train'
            # self.graph_full.subgraphs['train'].graph_mode='full'
            self.graph_full.subgraphs['val'].graphlayers[i]=Graphlayer()
            self.graph_full.subgraphs['val'].status='val'
            # self.graph_full.subgraphs['val'].graph_mode='full'
            self.graph_full.subgraphs['test'].graphlayers[i]=Graphlayer()
            self.graph_full.subgraphs['test'].status='test'
            # self.graph_full.subgraphs['test'].graph_mode='full'
            self.graph_sample.subgraphs['train'].graphlayers[i]=Graphlayer()
            self.graph_sample.subgraphs['train'].status='train'




    # server 2001 worker 3001 master 4001
    def ipInit(self, servers, workers):
        worker_num = glContext.config['worker_num']
        server_num = glContext.config['server_num']
        # if self.config['mode'] == 'code':
        #     self.server_ip = self.code_ip
        #     self.master_ip = self.code_ip
        #     for i in range(worker_num):
        #         self.worker_address[i] = self.code_ip + ":300" + str(i + 1)
        #     for i in range(server_num):
        #         self.server_address[i] = self.code_ip + ":200" + str(i + 1)
        #     self.config['master_address'] = self.master_ip + ":4001"
        # elif self.config['mode'] == 'test':
        workers = str.split(workers, ',')
        servers = str.split(servers, ',')
        for i in range(worker_num):
            self.config['worker_address'][i] = workers[i]
            self.config['server_address'][i] = servers[i]
            # self.config['master_address'] = master

    def setWorkerContext(self):
        self.dgnnClient.layerNum = self.config['layer_num']

    def initCluster(self):
        self.dgnnServerRouter = []
        self.dgnnWorkerRouter = []
        self.dgnnClient = DGNNClient()
        # self.dgnnMasterRouter = DGNNClient()
        self.dgnnClientRouterForCpp = Router()
        self.graphBuild=GraphBuild()
        self.sample=Sample()
        self.worker_id = self.config['id']
        id = self.config['id']
        # 当前机器的客户端，需要启动server，以保证不同机器间中间表征向量传输

        self.dgnnClient.serverAddress = self.config['worker_address'][id]

        self.dgnnClient.startClientServer()
        for i in range(self.config['server_num']):
            self.dgnnServerRouter.insert(i, DGNNClient())
            self.dgnnServerRouter[i].init_by_address(self.config['server_address'][i])
        for i in range(self.config['worker_num']):
            self.dgnnWorkerRouter.insert(i, DGNNClient())
            self.dgnnWorkerRouter[i].init_by_address(self.config['worker_address'][i])

        # self.dgnnMasterRouter.init_by_address(self.config['master_address'])

        # 在c++端初始化dgnnWorkerRouter
        self.dgnnClientRouterForCpp.initWorkerRouter(self.config['worker_address'])
        self.dgnnClientRouterForCpp.initServerRouter(self.config['server_address'])

        # 所有创建的类都在一个进程里，通过c++对静态变量操作，在所有类中都可见
        # print(dgnnClient.testString)
        # print(dgnnMasterRouter.testString)
        # print(dgnnServerRouter[0].testString)

        # start=time.time()
        # self.dgnnMasterRouter.pullDataFromMasterGeneral(
        #     id, self.config['worker_num'],
        #     self.config['data_num'],
        #     self.config['data_path'],
        #     self.config['feature_dim'],
        #     self.config['class_num'],
        #     self.config['partitionMethod'],
        #     self.config['edge_num'])
        # end=time.time()
        # print("pullDataFromMasterGeneral time:{0}".format(end-start))




glContext = Context()
