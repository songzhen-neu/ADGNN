import torch
import numpy as np
import scipy.sparse as sp
from cmake.build.lib.pb11_ec import *
from adgnn.context import context
from adgnn.data_structure.graph import Graph
from adgnn.data_structure.layer_compute import Layer_Compute
from adgnn.data_structure.graph_train import Graph_Train
from adgnn.ECTensor import ECTensor
from adgnn.util_python.timecounter import time_counter


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_mean(mx):
    """Row-normalize sparse matrix"""
    np.seterr(divide='ignore')
    rowsum = np.array(mx.sum(1), dtype=np.float)  # 对每一行求和
    # rowsum=np.array(degs)
    # rowsum[np.where(rowsum == 0)] = 1
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def normalize_gcn(mx, degs):
    """Row-normalize sparse matrix"""

    # rowsum = np.array(mx.sum(1), dtype=np.float)  # 对每一行求和
    rowsum = np.array(degs)
    # rowsum[np.where(rowsum == 0)] = 1
    r_inv = np.power(rowsum, -0.5).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1), dtype=np.float)  # 对每一行求和
#     r_inv = np.power(rowsum, -1).flatten()  # 求倒数
#     r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
#     r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
#     mx = r_mat_inv.dot(mx)
#     # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
#     return mx


# def normalize_feature(mx):
#     rowsum = np.array(mx.sum(1))  # 对每一行求和
#     r_inv = np.power(rowsum, -1).flatten()  # 求倒数
#     r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
#     r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
#     mx = r_mat_inv.dot(mx)
#     # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
#     return mx


def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}  # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot


"""
add
"""


def get_adj_func(adjs_from_server, adj_map):
    map = {}
    for i in adjs_from_server:
        map[i] = {i}
        for j in adjs_from_server[i]:
            for n in adj_map[j]:
                map[i].add(n)
    return map


def get_adj_nodes(adjs_from_server, nodes):
    arr = set()
    for i in nodes:
        arr.add(i)
        for j in adjs_from_server[i]:
            arr.add(j)
    return list(arr)


def split_dataset(v_from_server):
    """
    Splitting the datasets into train/val/test, where id-code follows the master id-map.
    :param vnum_l: the number of vertices in the local
    :return: global id lists of train/val/test datasets
    """

    train_num = context.glContext.config['train_num']
    val_num = context.glContext.config['val_num']
    test_num = context.glContext.config['test_num']
    # vertex_num=context.glContext.config['data_num']
    # rand_v=np.random.permutation(vertex_num)

    # train_set=set(rand_v[0:train_num])
    # val_set=set(rand_v[train_num:train_num+val_num])
    # test_set=set(rand_v[train_num+val_num:train_num+val_num+test_num])

    train_set = {i for i in range(train_num)}
    val_set = {i for i in range(train_num, train_num + val_num)}
    test_set = {i for i in range(train_num + val_num, train_num + val_num + test_num)}

    idx_train = []
    idx_val = []
    idx_test = []

    for id in v_from_server:
        if train_set.__contains__(id):
            idx_train.append(id)
        elif val_set.__contains__(id):
            idx_val.append(id)
        elif test_set.__contains__(id):
            idx_test.append(id)
    return idx_train, idx_val, idx_test


def parseAA2Dict(features_cpp):
    id_arr = features_cpp[0]
    data_num = len(id_arr)
    dim_size = len(features_cpp[1]) / data_num
    feat_arr = features_cpp[1].reshape((data_num, int(dim_size)))
    feat = {}
    for i in range(data_num):
        id = id_arr[i]
        feat[id] = feat_arr[i]
    return feat


def getInfoFromMaster(dgnnClient):
    """
    build v2wk (a map that indicates what the worker a vertex belongs to)
    and wk2v (a map that indicates what the vertices a worker contains )
    get subgraph information of the local
    :param dgnnClient:  c++ backend of the local worker
    :return: the local vertices, features, labels and adjs
    wk2v: vertices that belongs to worker i
    v2wk: the worker that contains vertex i
    """
    v2wk = {}
    wk2v = dgnnClient.nodesForEachWorker
    for i in range(context.glContext.config['worker_num']):
        for j in range(len(wk2v[i])):
            v2wk[wk2v[i][j]] = i
    features_cpp = dgnnClient.features
    feat = parseAA2Dict(features_cpp)
    degree_map = dgnnClient.degree_map

    return dgnnClient.nodes, feat, \
           dgnnClient.labels, dgnnClient.adjs, \
           wk2v, v2wk, degree_map


def getInfoFromFile():
    """
    build v2wk (a map that indicates what the worker a vertex belongs to)
    and wk2v (a map that indicates what the vertices a worker contains )
    get subgraph information of the local
    :param dgnnClient:  c++ backend of the local worker
    :return: the local vertices, features, labels and adjs
    wk2v: vertices that belongs to worker i
    v2wk: the worker that contains vertex i
    """
    context.glContext.dgnnServerRouter[0].server_Barrier()
    context.glContext.dgnnClient.readDataAndInit()
    context.glContext.dgnnServerRouter[0].server_Barrier()


def splitNeiToWorker(neighbor_nodes, train_nodes, v2wk):
    nodes_tmp = {}
    train_local_neighbor = set()
    train_remote_neighbor = set()
    worker_id = context.glContext.config['id']
    for j in range(context.glContext.config['worker_num']): nodes_tmp[j] = set()
    for n in neighbor_nodes:
        if n not in train_nodes:
            nodes_tmp[v2wk[n]].add(n)
            if v2wk[n] == worker_id:
                train_local_neighbor.add(n)
            else:
                train_remote_neighbor.add(n)

    for j in range(context.glContext.config['worker_num']): nodes_tmp[j] = list(nodes_tmp[j])
    return nodes_tmp, list(train_local_neighbor), list(train_remote_neighbor)


def getOneHopNeiSet(train_nodes, adjs, v2wk):
    worker_num = context.glContext.config['worker_num']
    neis = set()
    to_remote_nei = {i: set() for i in range(worker_num)}  # add

    for id in train_nodes:  # add
        for nei_id in adjs[id]:
            neis.add(nei_id)
            to_remote_nei[v2wk[nei_id]].add(nei_id)

    for j in range(context.glContext.config['worker_num']): to_remote_nei[j] = list(to_remote_nei[j])

    return neis, to_remote_nei


def getV2WkPush(push_2_worker_nodes):
    v2wk_push = {}
    for i in range(len(push_2_worker_nodes)):
        push2wk4lay_node = push_2_worker_nodes[i]
        v2wk4lay_push = {}
        for wid in push2wk4lay_node.keys():
            for vid in push2wk4lay_node[wid]:
                if not v2wk4lay_push.__contains__(vid):
                    v2wk4lay_push[vid] = []
                    v2wk4lay_push[vid].append(wid)
                else:
                    v2wk4lay_push[vid].append(wid)
        v2wk_push[i] = v2wk4lay_push
    return v2wk_push


def getTrainInfo(adjs_4_layer, v2wk, idx_train):
    """
    get the information of the pruned graph
    :param nodes: local vertices
    :param adjs: adjacent lists of local vertices
    :param v2wk: mapping vertices to its residing worker
    :param idx_train: vertex ids of training set
    :return: train_nodes:[[1,3],[1,3],[1,3,5]], training vertices of each layer
    :return: layer_node_size:[[2,0],[2,3],[3,3]], [[local,remote]], size of local and remote vertices of each layer
    :return: train_remote_nodes: [[],[0,2,4],[0,2,4]], remote vertex ids
    :return: worker_remote_nodes:{layer:{worker:[vertex]}}, indicate the worker of the required vertices in each layer
    """
    layer_num = context.glContext.config['layer_num']
    worker_id = context.glContext.config['id']
    worker_num = context.glContext.config['worker_num']

    train_nodes = {i: set() for i in range(layer_num + 1)}
    train_nodes[layer_num] = set(idx_train)
    nei_2_wk = {}
    train_local_neighbor = {}
    train_remote_neighbor = {}
    push_2_worker_nodes = {}
    for lay_id in range(layer_num, 0, -1):
        train_nodes_tmp = train_nodes[lay_id]
        neis_4lay, to_remote_nei = getOneHopNeiSet(train_nodes_tmp, adjs_4_layer[lay_id],
                                                   v2wk)  # TODO tmp pull nodes need send these nodes to other worker
        nei2wk_4lay, loc_nei_tmp, rmt_nei_tmp = splitNeiToWorker(neis_4lay, train_nodes_tmp, v2wk)
        nei_2_wk[lay_id] = nei2wk_4lay
        train_local_neighbor[lay_id] = loc_nei_tmp
        train_remote_neighbor[lay_id] = rmt_nei_tmp
        train_nodes[lay_id - 1] |= train_nodes_tmp
        train_nodes[lay_id - 1] |= (set(nei2wk_4lay[worker_id]))
        loc_nodes_needed_by_otherwk = context.glContext.dgnnClientRouterForCpp.sendNodes2Wk(lay_id, nei2wk_4lay)
        # push_2_worker_nodes[lay_id - 1] = context.glContext.dgnnClientRouterForCpp.sendInNodes2WK(lay_id, to_remote_nei)
        train_nodes[lay_id - 1] |= set(loc_nodes_needed_by_otherwk)
    push_2_worker_nodes[layer_num] = {i: [] for i in range(context.glContext.config['worker_num'])}
    train_local_neighbor[0] = []
    train_remote_neighbor[0] = []
    nei_2_wk[0] = {i: [] for i in range(worker_num)}
    train_nodes = {i: list(train_nodes[i]) for i in train_nodes.keys()}
    v2wk_push = getV2WkPush(push_2_worker_nodes)
    return train_nodes, nei_2_wk, train_local_neighbor, train_remote_neighbor, push_2_worker_nodes, v2wk_push


def encodeLocalVertex(train_nodes, train_local_neighbor, train_remote_neighbor):
    map_old_2_new = {}
    map_new_2_old = {}
    layer_num = context.glContext.config['layer_num']
    for i in range(0, layer_num + 1):
        old_2_new = {}
        new_2_old = {}
        for j in train_nodes[i]:
            l = len(old_2_new)
            old_2_new[j] = l
            new_2_old[l] = j

        for j in train_local_neighbor[i]:
            l = len(old_2_new)
            old_2_new[j] = l
            new_2_old[l] = j

        for j in train_remote_neighbor[i]:
            l = len(old_2_new)
            old_2_new[j] = l
            new_2_old[l] = j

        map_old_2_new[i] = old_2_new
        map_new_2_old[i] = new_2_old
    return map_new_2_old, map_old_2_new


def getTrainLabelAndFeat(train_nodes, labels, feats):
    """
    get required features and labels of training vertices, and cache the remote 1-hop neighbors
    """
    labels_train = []
    layer_num = context.glContext.config["layer_num"]
    for idx in train_nodes[layer_num]:
        labels_train.append(labels[idx])
    # labels_train = np.array(labels_train)

    feats_train = []
    for item in train_nodes[0]:
        feats_train.append(feats[item])

    # get remote future
    context.glContext.dgnnServerRouter[0].server_Barrier()
    # time_counter.start('csr_matrix')
    # feats_train = sp.csr_matrix(feats_train, dtype=np.float32)
    # time_counter.end('csr_matrix')

    time_counter.start('feats_train')
    feats_train = np.array(feats_train)
    feats_train = torch.FloatTensor(feats_train)
    labels_train = torch.LongTensor(labels_train)
    time_counter.end('feats_train')
    return feats_train, labels_train


def getTrainEdges(train_nodes, map_old2new, map_new2old, train_local_neighbor, train_remote_neighbor,
                  worker_neighbor_nodes, adjs_4_layer, push_2_worker_nodes, v2wk_push):
    Layer_Com = {}
    layer_num = context.glContext.config['layer_num']
    for lay in range(layer_num, -1, -1):
        if lay != 0:
            # edges = []
            # 从adj中解析出edge
            # dis = layer_node_size[0][0] - layer_node_size[lay-1][0]
            edges = context.glContext.dgnnClient.transEdgeToNewID(train_nodes[lay], map_old2new[lay], adjs_4_layer[lay])
            # for i in train_nodes[lay]:
            #     from_id = map_old2new[lay][i]
            #     for j in adjs_4_layer[lay][i]:
            #         to_id = map_old2new[lay][j]
            #         edges.append([from_id, to_id])

            # degs=[0 for i in range(len(map_old2new[lay]))]
            # for i in map_old2new[lay]:
            #     new_id=map_old2new[lay][i]
            #     degs[new_id]=len(adjs_4_layer[lay][i])

            # list->tensor->ndarray is more efficient
            # edges=torch.LongTensor(edges)
            # edges = np.array(edges)

            matrix_len = len(train_nodes[lay]) + len(train_local_neighbor[lay]) + len(train_remote_neighbor[lay])
            adjs_train = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                       shape=(matrix_len, matrix_len),
                                       dtype=np.int)

            adjs_train = normalize_mean(adjs_train)  # eye创建单位矩阵，第一个参数为行数，第二个为列数
            adjs_train = adjs_train[range(len(train_nodes[lay]))]
            adjs_train = sparse_mx_to_torch_sparse_tensor(adjs_train)  # 邻接矩阵转为tensor处理
            adjs_train = ECTensor(adjs_train)

            layer_c = Layer_Compute(torch.LongTensor(train_nodes[lay]), adjs_train, map_old2new[lay], map_new2old[lay],
                                    worker_neighbor_nodes[lay], len(train_remote_neighbor[lay]),
                                    push_2_worker_nodes[lay], v2wk_push[lay])
            Layer_Com[lay] = layer_c
        else:
            layer_c = Layer_Compute(torch.LongTensor(train_nodes[lay]), None, map_old2new[lay], map_new2old[lay],
                                    worker_neighbor_nodes[lay], len(train_remote_neighbor[lay]),
                                    push_2_worker_nodes[lay], v2wk_push[lay])
            Layer_Com[lay] = layer_c
    return Layer_Com


def map_nei2worker(v2wk, adj_bp, nodes):
    # worker_neighbor_nodes_bp={i:{} for i in range(context.glContext.config['worker_num'])}
    worker_neighbor_nodes_bp = {i: {j: set() for j in range(context.glContext.config['worker_num'])} for i in
                                range(len(adj_bp))}
    # train_local_neighbor_bp={i:set() for i in range(len(adj_bp))}
    train_remote_neighbor_bp = {i: set() for i in range(len(adj_bp))}
    for i in range(len(adj_bp) - 2, -1, -1):
        for node_id in adj_bp[i]:
            neis = adj_bp[i][node_id]
            for nid in neis:
                if not nodes.__contains__(nid):
                    train_remote_neighbor_bp[i].add(nid)
                worker_neighbor_nodes_bp[i][v2wk[nid]].add(nid)

    worker_neighbor_nodes_bp = {
        i: {j: list(worker_neighbor_nodes_bp[i][j]) for j in range(context.glContext.config['worker_num'])} for i in
        range(len(adj_bp))}
    print("map_nei2worker end")
    return worker_neighbor_nodes_bp


# def get_edges(adj_bp, o2n_bp, train_nodes):
#     edges_map = {i: None for i in range(len(adj_bp))}
#     for i in range(len(adj_bp) - 2, -1, -1):
#         edges_tmp = []
#         for nodeid in adj_bp[i]:
#             nodeid_new = o2n_bp[i][nodeid]
#             for neiid in adj_bp[i][nodeid]:
#                 neiid_new = o2n_bp[i][neiid]
#                 edges_tmp.append([nodeid_new, neiid_new])
#         # edges_map[i]=edges_tmp
#         edges_tmp = np.array(edges_tmp)
#
#         adjs_matrix = sp.coo_matrix((np.ones(edges_tmp.shape[0]), (edges_tmp[:, 0], edges_tmp[:, 1])),
#                                     shape=(len(train_nodes[i]), len(o2n_bp[i])),
#                                     dtype=np.int)
#         adjs_matrix = normalize_gcn(adjs_matrix)
#         adjs_matrix = sparse_mx_to_torch_sparse_tensor(adjs_matrix)  # 邻接矩阵转为tensor处理
#         edges_map[i] = adjs_matrix
#     return edges_map


# def getSubGraph(labels, feats, adjs_4_layer, v2wk, idx_train, for_stage,rmt_nei_feat_full=None,o2n_4rmtnei_full=None):
#     time_counter.start('getSubGraph')
#     time_counter.start('getTrainInfo')
#     train_nodes, nei_2_wk, train_local_neighbor, train_remote_neighbor, push_2_worker_nodes, v2wk_push = getTrainInfo(
#         adjs_4_layer,
#         v2wk,
#         idx_train)
#     time_counter.end('getTrainInfo')
#     time_counter.start('encodeLocalVertex')
#     map_new2old, map_old2new = encodeLocalVertex(train_nodes, train_local_neighbor, train_remote_neighbor)
#     time_counter.end('encodeLocalVertex')
#     time_counter.start('getTrainLabelAndFeat')
#     feat_train, label_train = getTrainLabelAndFeat(train_nodes, labels, feats)
#     time_counter.end('getTrainLabelAndFeat')
#     time_counter.start('getTrainEdges')
#     edges_train = getTrainEdges(train_nodes, map_old2new, map_new2old, train_local_neighbor, train_remote_neighbor,
#                                 nei_2_wk, adjs_4_layer, push_2_worker_nodes, v2wk_push)
#     time_counter.end('getTrainEdges')
#     label_train = ECTensor(label_train)
#
#     graph_train = Graph_Train(for_stage, train_nodes[0], feat_train, label_train, edges_train,rmt_nei_feat_full,o2n_4rmtnei_full)
#     time_counter.end('getSubGraph')
#     return graph_train


def getSubGraph(labels, feats, adjs_4_layer, v2wk, idx_train, for_stage, rmt_nei_feat_full=None, o2n_4rmtnei_full=None):
    time_counter.start('getSubGraph')
    time_counter.start('getTrainInfo')
    train_nodes, nei_2_wk, train_local_neighbor, train_remote_neighbor, push_2_worker_nodes, v2wk_push = getTrainInfo(
        adjs_4_layer,
        v2wk,
        idx_train)
    time_counter.end('getTrainInfo')
    time_counter.start('encodeLocalVertex')
    map_new2old, map_old2new = encodeLocalVertex(train_nodes, train_local_neighbor, train_remote_neighbor)
    time_counter.end('encodeLocalVertex')
    time_counter.start('getTrainLabelAndFeat')
    feat_train, label_train = getTrainLabelAndFeat(train_nodes, labels, feats)
    time_counter.end('getTrainLabelAndFeat')
    time_counter.start('getTrainEdges')
    edges_train = getTrainEdges(train_nodes, map_old2new, map_new2old, train_local_neighbor, train_remote_neighbor,
                                nei_2_wk, adjs_4_layer, push_2_worker_nodes, v2wk_push)
    time_counter.end('getTrainEdges')
    label_train = ECTensor(label_train)

    graph_train = Graph_Train(for_stage, train_nodes[0], feat_train, label_train, edges_train, rmt_nei_feat_full,
                              o2n_4rmtnei_full)
    time_counter.end('getSubGraph')
    return graph_train


def setLocVnum4SubGraph(stage, graph):
    wid = context.glContext.config['id']
    graph.subgraph[stage].local_node_num = {i: (len(graph.subgraph[stage].layer_compute[i].train_vertices)
                                                + len(graph.subgraph[stage].layer_compute[i].fsthop_for_worker[wid]))
                                            for i in range(len(graph.subgraph[stage].layer_compute))}


def setLocVnum4Graph(graph):
    setLocVnum4SubGraph('train', graph)
    setLocVnum4SubGraph('val', graph)
    setLocVnum4SubGraph('test', graph)


def setSubGraphInfoForCpp(stage, graph):
    context.glContext. \
        dgnnClient.setGraphInfoForCpp(stage,
                                      {i: graph.subgraph[stage].layer_compute[i].fsthop_for_worker for i in
                                       range(len(graph.subgraph[stage].layer_compute))},
                                      {i: graph.subgraph[stage].layer_compute[i].id_old2new_dict for i in
                                       range(len(graph.subgraph[stage].layer_compute))},
                                      {i: graph.subgraph[stage].layer_compute[i].id_new2old_dict for i in
                                       range(len(graph.subgraph[stage].layer_compute))},
                                      graph.v2wk_map,
                                      graph.subgraph[stage].local_node_num,
                                      {i: (len(graph.subgraph[stage].layer_compute[i].train_vertices))
                                       for i in range(len(graph.subgraph[stage].layer_compute))},
                                      {i: graph.subgraph[stage].layer_compute[i].rmt_nei_num for i in
                                       range(len(graph.subgraph[stage].layer_compute))},
                                      {i: graph.subgraph[stage].layer_compute[i].push_2_worker_nodes for i in
                                       range(len(graph.subgraph[stage].layer_compute))},
                                      {i: graph.subgraph[stage].layer_compute[i].v2wk_push for i in
                                       range(len(graph.subgraph[stage].layer_compute))})


def setGraphInfoForCpp(graph):
    setSubGraphInfoForCpp('train', graph)
    setSubGraphInfoForCpp('val', graph)
    setSubGraphInfoForCpp('test', graph)


def setCtxForCpp():
    cluster_config = {
        'worker_num': str(context.glContext.config['worker_num']),
        'worker_id': str(context.glContext.config['id']),
        'layer_num': str(context.glContext.config['layer_num']),
        'num_threads': str(context.glContext.config['num_threads']),
        'vertex_num': str(context.glContext.config['data_num']),
        'train_num': str(context.glContext.config['train_num']),
        'val_num': str(context.glContext.config['val_num']),
        'test_num': str(context.glContext.config['test_num']),
        'feat_size': str(context.glContext.config['feature_dim']),
        'filename': context.glContext.config['data_path'],
        'sample_method': context.glContext.config['sample_method'],
        'partition_method': context.glContext.config['partitionMethod'],
    }
    context.glContext.dgnnClient.setCtxForCpp(cluster_config)


def getAndSetRmtFeatForCtx(graph):
    # get remote nei_future
    context.glContext.dgnnServerRouter[0].server_Barrier()
    rmt_train_feat = torch.FloatTensor(context.glContext.dgnnClientRouterForCpp.getRmtFeat('train'))
    graph.subgraph['train'].rmt_nei_feat_full = rmt_train_feat
    graph.subgraph['train'].o2n_4rmtnei_full = {
        i: graph.subgraph['train'].layer_compute[1].id_old2new_dict[i] - graph.subgraph['train'].local_node_num[1] for i
        in graph.subgraph['train'].layer_compute[1].id_old2new_dict}
    graph.subgraph['train'].o2n_4rmtnei_full = {i: graph.subgraph['train'].o2n_4rmtnei_full[i] for i in
                                                graph.subgraph['train'].o2n_4rmtnei_full if
                                                graph.subgraph['train'].o2n_4rmtnei_full[i] >= 0}
    rmt_val_feat = torch.FloatTensor(context.glContext.dgnnClientRouterForCpp.getRmtFeat('val'))
    graph.subgraph['val'].rmt_nei_feat_full = rmt_val_feat
    graph.subgraph['val'].o2n_4rmtnei_full = {
        i: graph.subgraph['val'].layer_compute[1].id_old2new_dict[i] - graph.subgraph['val'].local_node_num[1] for i in
        graph.subgraph['val'].layer_compute[1].id_old2new_dict}
    graph.subgraph['val'].o2n_4rmtnei_full = {i: graph.subgraph['val'].o2n_4rmtnei_full[i] for i in
                                              graph.subgraph['val'].o2n_4rmtnei_full if
                                              graph.subgraph['val'].o2n_4rmtnei_full[i] >= 0}
    rmt_test_feat = torch.FloatTensor(context.glContext.dgnnClientRouterForCpp.getRmtFeat('test'))
    graph.subgraph['test'].rmt_nei_feat_full = rmt_test_feat
    graph.subgraph['test'].o2n_4rmtnei_full = {
        i: graph.subgraph['test'].layer_compute[1].id_old2new_dict[i] - graph.subgraph['test'].local_node_num[1] for i
        in graph.subgraph['test'].layer_compute[1].id_old2new_dict}
    graph.subgraph['test'].o2n_4rmtnei_full = {i: graph.subgraph['test'].o2n_4rmtnei_full[i] for i in
                                               graph.subgraph['test'].o2n_4rmtnei_full if
                                               graph.subgraph['test'].o2n_4rmtnei_full[i] >= 0}


def setStaticInfoForCpp(graph):
    rmt_nei_set = {i: [] for i in range(len(graph.subgraph['train'].layer_compute))}
    loc_nei_set = {i: [] for i in range(len(graph.subgraph['train'].layer_compute))}
    for i in range(len(graph.subgraph['train'].layer_compute)):
        for wid in range(context.glContext.config['worker_num']):
            if wid != context.glContext.config['id']:
                rmt_nei_set[i].extend(graph.subgraph['train'].layer_compute[i].fsthop_for_worker[wid])
            else:
                loc_nei_set[i].extend(graph.subgraph['train'].layer_compute[i].fsthop_for_worker[wid])
        rmt_nei_set[i] = set(rmt_nei_set[i])
        loc_nei_set[i] = set(loc_nei_set[i])
    context.glContext.dgnnClient.setStaticInfoForCpp(rmt_nei_set, loc_nei_set)


def setAdj(status, graph_mode):
    graph=getGraph(graph_mode)
    layer_num = context.glContext.config['layer_num']
    for lay in range(layer_num, -1, -1):
        if lay != 0:
            target_num, matrix_len, edges = context.glContext.graphBuild.transEdgeToNewID(status, graph_mode, lay)
            adjs_train = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                       shape=(matrix_len, matrix_len),
                                       dtype=np.int)
            adjs_train = normalize_mean(adjs_train)  # eye创建单位矩阵，第一个参数为行数，第二个为列数
            adjs_train = adjs_train[range(target_num)]
            adjs_train = sparse_mx_to_torch_sparse_tensor(adjs_train)  # 邻接矩阵转为tensor处理
            adjs_train = ECTensor(adjs_train)
            graph.subgraphs[status].graphlayers[lay].adj = adjs_train


def getGraph(graph_mode):
    if graph_mode == 'full':
        return context.glContext.graph_full
    else:
        return context.glContext.graph_sample


def setFeat(status, graph_mode):
    graph = getGraph(graph_mode)
    graph.subgraphs[status].feat_data = torch.FloatTensor(context.glContext.graphBuild.getTrainedFeat(status, graph_mode))


def setLabel(status, graph_mode):
    graph = getGraph(graph_mode)
    graph.subgraphs[status].label = ECTensor(
        torch.LongTensor(context.glContext.graphBuild.getTrainedLabel(status, graph_mode)))




def transGraphCppToPython(status, graph_mode):
    time_counter.start("setAdj")
    setAdj(status, graph_mode)
    time_counter.end("setAdj")
    time_counter.start("setFeat")
    setFeat(status, graph_mode)
    time_counter.end("setFeat")
    time_counter.start("setLabel")
    setLabel(status, graph_mode)
    time_counter.end("setLabel")


def load_data():
    # global ids encoded by master
    setCtxForCpp()
    time_counter.start("getInfoFromMaster")
    print("getInfoFromMaster start")
    getInfoFromFile()
    print("getInfoFromMaster end")
    time_counter.end("getInfoFromMaster")

    time_counter.start("buildInitGraph")
    context.glContext.graphBuild.buildInitGraph()
    context.glContext.dgnnServerRouter[0].server_Barrier()
    time_counter.end("buildInitGraph")

    time_counter.start("transGraphCppToPython")
    transGraphCppToPython("train", "full")
    transGraphCppToPython("val", "full")
    transGraphCppToPython("test", "full")
    context.glContext.graphBuild.deleteFullGraphInCpp()
    time_counter.end("transGraphCppToPython")
    print("load data end")
    return context.glContext.graph_full

