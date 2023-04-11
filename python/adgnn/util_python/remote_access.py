from adgnn.context import context
import torch
import numpy as np
from adgnn.ECTensor import ECTensor
from cmake.build.lib.pb11_ec import *
from adgnn.util_python.timecounter import time_counter


def setEmbsForCpp(input, status):
    # if layerid!=0:
    emb_temp = input.detach().numpy()
    set_embs_ptr(emb_temp, status)


def getRemoteEmb(layer_id, status, in_features):
    remoteEmb = context.glContext.dgnnClientRouterForCpp.getRemoteEmb(layer_id, status, in_features)
    return torch.FloatTensor(remoteEmb)


def getRmtEmbs(layer_id, status, x, graph,weight=None):
    if layer_id==1:
        rmt_nei_id=[]
        for i in range(graph.local_node_num[layer_id],len(graph.layer_compute[layer_id].id_new2old_dict)):
            rmt_nei_id.append(graph.o2n_4rmtnei_full[graph.layer_compute[layer_id].id_new2old_dict[i]])
        rmt_embs=graph.rmt_nei_feat_full[np.array(rmt_nei_id)]
    else:
        rmt_embs=context.glContext.dgnnClientRouterForCpp.pushEmbs(layer_id, status, x)
    if weight is not None and layer_id==1:
        rmt_embs=rmt_embs.mm(weight.tensor)
    # rmt_embs=context.glContext.dgnnClientRouterForCpp.pushEmbs(layer_id, status, x)
    return rmt_embs


# def getLocalEmb(layer_id, input, graph):
#     local_node_num_layer = graph.local_node_num[layer_id]
#     worker_id = context.glContext.config['id']
#     emb_last_layer = input.detach().numpy()
#     dim_size = emb_last_layer.shape[1]
#     emb_last_layer = emb_last_layer.tolist()
#     train_node = graph.layer_compute[layer_id].train_vertices.detach().tolist()
#     local_nei = graph.layer_compute[layer_id].fsthop_for_worker[worker_id]
#     local_emb = [[0 for _ in range(dim_size)] for _ in range(local_node_num_layer)]
#     o2n_curlay = graph.layer_compute[layer_id].id_old2new_dict
#     o2n_lstlay = graph.layer_compute[layer_id - 1].id_old2new_dict
#     for id in train_node:
#         nid_lst = o2n_lstlay[id]
#         nid_cur = o2n_curlay[id]
#         local_emb[nid_cur] = emb_last_layer[nid_lst]
#     for id in local_nei:
#         nid_lst = o2n_lstlay[id]
#         nid_cur = o2n_curlay[id]
#         local_emb[nid_cur] = emb_last_layer[nid_lst]
#     return torch.FloatTensor(local_emb)


def getLocEmbs(layer_id, graph, ids_tmp, embs_tmp):
    ids_map = {}
    for i in range(len(ids_tmp)):
        ids_map[ids_tmp[i]] = i
    loc_nodes = graph.layer_compute[layer_id].train_vertices.detach().numpy().tolist()
    loc_nei = graph.layer_compute[layer_id].fsthop_for_worker[context.glContext.config['id']]
    loc_num = len(loc_nodes) + len(loc_nei)
    n2o_map_cur = graph.layer_compute[layer_id].id_new2old_dict
    ids_last = [0 for i in range(loc_num)]
    for i in range(loc_num):
        id_in_last = ids_map[n2o_map_cur[i]]
        ids_last[i] = id_in_last
    loc_embs = embs_tmp[ids_last]
    return loc_embs


# def getEmbs(graph, input, layer_info):
#     timecounter = TimeCounter()
#     timecounter.start('other_time')
#     layer_id = layer_info.layer_id
#     context.glContext.dgnnServerRouter[0].server_Barrier()
#     setEmbsForCpp(input(), graph.status)
#     context.glContext.dgnnServerRouter[0].server_Barrier()
#     emb_dim_size = input().shape[1]
#     timecounter.end('other_time')
#     timecounter.start('rmt_time')
#     remote_embs = getRemoteEmb(layer_id, graph.status, emb_dim_size)
#     context.glContext.dgnnServerRouter[0].server_Barrier()
#     timecounter.end('rmt_time')
#     timecounter.start('loc_time')
#     local_embs = getLocalEmb(layer_id, input(), graph)
#     embs = torch.cat((local_embs, remote_embs), dim=0)
#     embs = ECTensor(embs, input, None, 'get_embs', None, requires_grad=True)
#     input.root = embs
#     embs.layer_id = layer_id
#     timecounter.end('loc_time')
#     # print("layer_id:{:2d}".format(layer_id),
#     #       "other_time: {:.4f}".format(timecounter.time_list["other_time"][-1]),
#     #       "rmt_time: {:.4f}".format(timecounter.time_list["rmt_time"][-1]),
#     #       "loc_time: {:.4f}".format(timecounter.time_list['loc_time'][-1]))
#     return embs


# def pushEmbs(layer_id, graph, input,weight=None):
#     """
#     push embeddings based on the (layer_id-1)-th push_2_worker_nodes
#     layer_id begins from 1, so in the first layer the system needs to push 0-th embeddings
#     :param layer_id:
#     :param status:
#     :param input:
#     :param graph:
#     :return:
#     """
#     time_counter.start('fp_constructEmbs'+str(context.glContext.is_train))
#     time_counter.start('fp_pushEmbs'+str(context.glContext.is_train))
#     rmt_embs = getRmtEmbs(layer_id, graph.status, input.tensor.detach().numpy(), graph,weight)
#     # context.glContext.dgnnClientRouterForCpp.pushEmbs(layer_id, graph.status, input.tensor.detach().numpy())
#     rmt_embs = torch.FloatTensor(rmt_embs)
#     time_counter.end('fp_pushEmbs'+str(context.glContext.is_train))
#     time_counter.start('fp_getLocEmbs'+str(context.glContext.is_train))
#
#     loc_embs = getLocEmbs(layer_id, graph, graph.layer_compute[layer_id - 1].train_vertices.detach().numpy(),
#                           input.tensor)
#     time_counter.end('fp_getLocEmbs'+str(context.glContext.is_train))
#     time_counter.start('fp_getLocEmbsCat'+str(context.glContext.is_train))
#     embs = torch.cat((loc_embs, rmt_embs), dim=0)
#     embs = ECTensor(embs, input, None, 'push_embs', None, requires_grad=True)
#     time_counter.end('fp_getLocEmbsCat'+str(context.glContext.is_train))
#     input.root = embs
#     embs.layer_id = layer_id
#     time_counter.end('fp_constructEmbs'+str(context.glContext.is_train))
#     return embs

def pushEmbs(layer_id, graph, input):
    """
    push embeddings based on the (layer_id-1)-th push_2_worker_nodes
    layer_id begins from 1, so in the first layer the system needs to push 0-th embeddings
    :param layer_id:
    :param status:
    :param input:
    :param graph:
    :return:
    """
    if layer_id==1:
        return input
    else:
        embs=context.glContext.dgnnClientRouterForCpp.pushEmbs(layer_id,graph.status,graph.graph_mode,input.tensor.detach().numpy())
        embs = torch.FloatTensor(embs)
        embs = ECTensor(embs, input, None, 'push_embs', None, requires_grad=True)
        embs.layer_id=layer_id
        input.root=embs
        return embs



def getLocFeatForCacheMode(graph):
    train_vertices = graph.layer_compute[1].train_vertices.detach().numpy().tolist()
    loc_nei = graph.layer_compute[1].fsthop_for_worker[context.glContext.config['id']]
    old2new_map_1 = graph.layer_compute[1].id_old2new_dict
    old2new_map_0 = graph.layer_compute[0].id_old2new_dict
    feat_0 = graph.feat_data.detach().numpy().tolist()
    feat_1 = [None for _ in range(len(train_vertices) + len(loc_nei))]
    for id in train_vertices:
        new_id_1 = old2new_map_1[id]
        new_id_0 = old2new_map_0[id]
        feat_1[new_id_1] = feat_0[new_id_0]
    for id in loc_nei:
        new_id_1 = old2new_map_1[id]
        new_id_0 = old2new_map_0[id]
        feat_1[new_id_1] = feat_0[new_id_0]

    return torch.FloatTensor(feat_1)


def catCacheFeature(graph):
    loc_feat = getLocFeatForCacheMode(graph)
    rmt_feat = graph.rmt_nei_feat
    feat = torch.cat((loc_feat, rmt_feat), dim=0)
    feat = ECTensor(feat)
    return feat
