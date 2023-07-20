

#include "router.h"
#include <cstring>

Router::Router() {}

vector<DGNNClient *> Router::dgnnWorkerRouter;
vector<DGNNClient *> Router::dgnnServerRouter;
vector<vector<ReqEmbsMetaData>> Router::metadata_pipe;
vector<float *> Router::embs_pipe;
//vector<ReqEmbsMetaData> Router::metadata_vec;



void Router::initWorkerRouter(map<int, string> &dgnnWorkerAddress) {

    for (auto &address:dgnnWorkerAddress) {
        DGNNClient *dgnnClient = new DGNNClient();
        dgnnClient->init_by_address(address.second);
        dgnnWorkerRouter.push_back(dgnnClient);
        cout << address.second << endl;
    }
//    cout<<dgnnWorkerRouter[1]->add1()<<endl;
}


void Router::initServerRouter(map<int, string> &dgnnServerAddress) {
    for (auto &address:dgnnServerAddress) {
        DGNNClient *dgnnClient = new DGNNClient();
        dgnnClient->init_by_address(address.second);
        dgnnServerRouter.push_back(dgnnClient);
        cout << address.second << endl;
    }
}


py::array_t<float> Router::getRemoteEmb(int layerId, const string &status, int feat_num, const string &graph_mode) {
    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs[status];
    int workerNum = WorkerStore::worker_num;
    int localId = WorkerStore::worker_id;
    vector<EmbGradMessage> replyVec(workerNum);

    auto nodes = subgraph.graphlayers[layerId].wk2nei_pull;
//    auto nodes = WorkerStore::fsthop_for_worker[status][layerId];
    int totalNodeNum = 0;
    auto &layer = graph->subgraphs[status].graphlayers[layerId];

    int local_vertex_num = layer.vnum_tarv_locnei_rmtnei[0] + layer.vnum_tarv_locnei_rmtnei[1];
    totalNodeNum = (int) layer.o2n_map.size() -
                   local_vertex_num; // TODO check this place
    vector<pthread_t> pthreads(WorkerStore::worker_num);
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    auto result = py::array_t<float>(totalNodeNum * feat_num);
    result.resize({totalNodeNum, feat_num});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;


    //  从远端异步获取
    for (int i = 0; i < workerNum; i++) {
        if (i != localId) {
//            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->reply = &replyVec[i];
            metaData->serverId = i;
            metaData->workerId = localId;
            metaData->nei_set = &nodes[i];
            metaData->layerId = layerId;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->ptr_result = ptr_result;
            metaData->oldToNewMap = &layer.o2n_map;
            metaData->localNodeSize = local_vertex_num;
            metaData->feat_num = feat_num;
            metaData->status = status;
            pthread_create(&pthreads[i], NULL, DGNNClient::worker_pull_needed_emb_parallel, (void *) metaData);


        }
    }


    pthread_vec_join(pthreads);


    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "reply responsing time:" << timeuse << "s" << endl;
    return result;
}

vector<vector<float>> initGradTmp(int node_num, int dim_num) {
    vector<vector<float>> grad_tmp(node_num);
    for (int i = 0; i < node_num; i++) {
        vector<float> vec_tmp(dim_num);
        grad_tmp[i] = vec_tmp;
    }
    return grad_tmp;
}

py::array_t<float> Router::setAndSendG(int layer_id, const py::array_t<float> &emb_grads) {

    auto graph_mode = WorkerStore::graph_mode;
    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs["train"];

    py::buffer_info buf = emb_grads.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    auto *ptr = (float *) buf.ptr;
    int node_num = buf.shape[0];
    int feat_size = buf.shape[1];


//    cout<<layer_id<<endl;
//    cout<<subgraph.graphlayers[2].vnum_tarv_locnei_rmtnei.size()<<endl;
//    cout<<subgraph.graphlayers[1].vnum_tarv_locnei_rmtnei.size()<<endl;
//    cout<<subgraph.graphlayers[0].vnum_tarv_locnei_rmtnei.size()<<endl;
//    cout<<subgraph.graphlayers[layer_id-1].vnum_tarv_locnei_rmtnei.size()<<endl;
//    cout<<subgraph.graphlayers[layer_id-1].vnum_tarv_locnei_rmtnei[0]<<','<<subgraph.graphlayers[layer_id-1].vnum_tarv_locnei_rmtnei[1]<<endl;
//    cout<<"aaaaaaaaaaaaaa"<<endl;
    int local_node_num = subgraph.graphlayers[layer_id].vnum_tarv_locnei_rmtnei[0] +
                         subgraph.graphlayers[layer_id].vnum_tarv_locnei_rmtnei[1];


//    cout<<subgraph.graphlayers[layer_id-1].vnum_tarv_locnei_rmtnei.size()<<endl;
    int local_train_node_num = subgraph.graphlayers[layer_id - 1].vnum_tarv_locnei_rmtnei[0];


//    int local_node_num = WorkerStore::local_vertex_num["train"][layer_id];
//    int local_train_node_num = WorkerStore::local_train_vertex_num["train"][layer_id - 1];
    auto grad_tmp = initGradTmp(local_train_node_num, feat_size);
    vector<pthread_t> pthreads(WorkerStore::worker_num);


    auto &n2o_map_cur = subgraph.graphlayers[layer_id].n2o_map;
    auto &o2n_map_last = subgraph.graphlayers[layer_id - 1].o2n_map;
//    auto n2o_map_cur=WorkerStore::new2old_map["train"][layer_id];
//    auto o2n_map_last=WorkerStore::old2new_map["train"][layer_id - 1];

    // {worker:{layer_id:vec}}
    map<int, map<int, vector<float>>> message;
    vector<EmbGradMessage> embgrad_message_vec(WorkerStore::worker_num);


    for (int i = 0; i < WorkerStore::worker_num; i++) {
        embgrad_message_vec[i].set_featsize(feat_size);
        embgrad_message_vec[i].set_layerid(layer_id);
        embgrad_message_vec[i].set_graph_mode(graph_mode);
        embgrad_message_vec[i].set_workerid(WorkerStore::worker_id);
    }


    for (int i = 0; i < node_num; i++) {
        if (i < local_node_num) {
            int old_id = n2o_map_cur[i];
            int new_id = o2n_map_last[old_id];
            float *grad_tmp_ptr = &grad_tmp[new_id][0];
            copy(ptr + i * feat_size, ptr + (i + 1) * feat_size, grad_tmp_ptr);
        } else {
            int old_id = n2o_map_cur[i];
            int wid = WorkerStore::graph.v2wk[old_id];
            embgrad_message_vec[wid].add_nodes(old_id);
            embgrad_message_vec[wid].mutable_embs()->Add(ptr + i * feat_size, ptr + (i + 1) * feat_size);
        }
    }

    WorkerStore::local_emb_grad_agg[layer_id] = grad_tmp;
    dgnnServerRouter[0]->server_Barrier();

    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (i != WorkerStore::worker_id) {
//            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->embGradMessage = &embgrad_message_vec[i];
            metaData->dgnnClient = dgnnWorkerRouter[i];
            pthread_create(&pthreads[i], NULL, DGNNClient::worker_pull_g_parallel, (void *) metaData);
        }
    }

    pthread_vec_join(pthreads);

    dgnnServerRouter[0]->server_Barrier();


    // return the aggregate of local_emb_grad
    // the order follows the old2new_map of layer_id
    auto result = py::array_t<float>(local_train_node_num * feat_size);
    result.resize({local_train_node_num, feat_size});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;
    for (int i = 0; i < local_train_node_num; i++) {
        copy(WorkerStore::local_emb_grad_agg[layer_id][i].begin(), WorkerStore::local_emb_grad_agg[layer_id][i].end(),
             ptr_result + i * feat_size);
//        for (int j = 0; j < dim_num; j++) {
//            ptr_result[i * dim_num + j] = WorkerStore::local_emb_grad_agg[layer_id][i][j];
//        }
    }


    return result;

}


py::array_t<long> Router::sendNodes2Wk(int layer_id, map<int, vector<int>> &nei2wk_4lay) {
//    cout<<WorkerStore::worker_id<<endl;
    vector<pthread_t> pthreads(WorkerStore::worker_num);
    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (WorkerStore::worker_id != i) {
//            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->layerId = layer_id;
            metaData->nodes = &nei2wk_4lay[i];
            pthread_create(&pthreads[i], NULL, DGNNClient::sendNodes2Wk, (void *) metaData);
        }
    }

    pthread_vec_join(pthreads);

    dgnnServerRouter[0]->server_Barrier();
    auto result = py::array_t<int>(WorkerStore::neis_neededby_otherwk[layer_id].size());
    py::buffer_info buf_result = result.request();
    int *ptr_result = (int *) buf_result.ptr;
    copy(WorkerStore::neis_neededby_otherwk[layer_id].begin(), WorkerStore::neis_neededby_otherwk[layer_id].end(),
         ptr_result);
    return result;
}


void
Router::getRmtFeat(const string &status) {

    auto &graphlayer1 = WorkerStore::graph.subgraphs[status].graphlayers[1];
    vector<EmbGradMessage> replyVec(WorkerStore::worker_num);
    auto &subgraph = WorkerStore::graph.subgraphs[status];
    int totalNodeNum = graphlayer1.vnum_tarv_locnei_rmtnei[2];
//    int totalNodeNum = WorkerStore::rmt_nei_num[status][1];



    int feat_size = WorkerStore::feat_size;
    int worker_num = WorkerStore::worker_num;
    int local_id = WorkerStore::worker_id;
    auto old_2_new = graphlayer1.o2n_map;

//    int local_size = WorkerStore::local_vertex_num[status][1];
    int local_size = graphlayer1.vnum_tarv_locnei_rmtnei[0] + graphlayer1.vnum_tarv_locnei_rmtnei[1];
    auto &remote = graphlayer1.wk2nei_pull;
    vector<pthread_t> pthreads(WorkerStore::worker_num);


    for (int i = 0; i < worker_num; i++) {
        if (i != local_id) {
//            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->reply = &replyVec[i];
            metaData->nei_set = &remote[i];
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->oldToNewMap = &old_2_new;
            metaData->feat_num = feat_size;
            metaData->thread_id = i;

            pthread_create(&pthreads[i], NULL, DGNNClient::getRmtFeat, (void *) metaData);
        }
    }


    pthread_vec_join(pthreads);

//    return result;

}

unordered_map<int, unordered_set<int>> &
Router::sendInNodes2WK(int layer_id, unordered_map<int, unordered_set<int>> &to_remote_nei) {
    vector<pthread_t> pthreads(WorkerStore::worker_num);
    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (i != WorkerStore::worker_id) {
            auto *metaData = new ReqEmbsMetaData;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->layerId = layer_id;
            metaData->nei_set = &to_remote_nei[i];
            metaData->workerId = WorkerStore::worker_id;
            metaData->thread_id = i;
            pthread_create(&pthreads[i], NULL, DGNNClient::sendInNodes2WK, (void *) metaData);
        }
    }

    pthread_vec_join(pthreads);
    dgnnServerRouter[0]->server_Barrier();

    return WorkerStore::neis_in_neededby_otherwk;
}

py::array_t<float>
Router::pushEmbs(int layer_id, const string &status, const string &graph_mode, py::array_t<float> &embs) {
    // layer_id starts from 1
    // embs belong to layer 0, when layer_id=1

    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs[status];

//    auto& old2new_map = WorkerStore::old2new_map[status][layer_id - 1];
    auto &o2n_map_last = subgraph.graphlayers[layer_id - 1].o2n_map;

    auto &push2wk_nodes_4lay = subgraph.graphlayers[layer_id - 1].wk2nei_push;

//    auto& push2wk_nodes_4lay = WorkerStore::push_2_worker_nodes[status][layer_id - 1];
    auto buffer = embs.request();
    auto emb_ptr = (float *) buffer.ptr;
    int emb_dim = buffer.shape[1];


    int rmt_nei_num = subgraph.graphlayers[layer_id].vnum_tarv_locnei_rmtnei[2];
    int nei_num = subgraph.graphlayers[layer_id].o2n_map.size();
//    int rmt_nei_num = WorkerStore::rmt_nei_num[status][layer_id];

//    free(WorkerStore::emb_reply_4push_ptr);
//    free(WorkerStore::emb_reply_4push);

//    WorkerStore::emb_reply_4push = py::array_t<float>(nei_num * emb_dim);

    WorkerStore::emb_reply_4push_ptr = (float *) malloc(sizeof(float) * nei_num * emb_dim);



//    auto result_buffer = WorkerStore::emb_reply_4push.request();

//    WorkerStore::emb_reply_4push_ptr = (float *) result_buffer.ptr;

    dgnnServerRouter[0]->server_Barrier();


    vector<pthread_t> pthreads(WorkerStore::worker_num);
    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (WorkerStore::worker_id != i) {
            auto *metaData = new ReqEmbsMetaData;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->nei_set_unorder = &push2wk_nodes_4lay[i];
            metaData->oldToNewMap = &o2n_map_last;
            metaData->emb_ptr = emb_ptr;
            metaData->feat_num = emb_dim;
            metaData->layerId = layer_id;
            metaData->status = status;
            metaData->graph_mode = graph_mode;
            pthread_create(&pthreads[i], NULL, DGNNClient::pushEmbsParallel, (void *) metaData);
        }
    }


    pthread_vec_join(pthreads);

    Router::dgnnServerRouter[0]->server_Barrier();


    // fill local neighbor and target_v
    auto &target_v = subgraph.graphlayers[layer_id].target_v;
    auto &loc_nei = subgraph.graphlayers[layer_id].wk2nei_pull[WorkerStore::worker_id];

    auto &o2n_map_cur = subgraph.graphlayers[layer_id].o2n_map;


    auto result = py::array_t<float>(nei_num * emb_dim);
    auto *result_ptr = (float *) result.request().ptr;
    copy(WorkerStore::emb_reply_4push_ptr, WorkerStore::emb_reply_4push_ptr + emb_dim * nei_num, result_ptr);


    for (auto id : target_v) {
        auto nid = o2n_map_cur[id];
        auto nid_last = o2n_map_last[id];
        copy(emb_ptr + emb_dim * nid_last, emb_ptr + emb_dim * (nid_last + 1), result_ptr + emb_dim * nid);
    }


    for (auto id : loc_nei) {
        auto nid = o2n_map_cur[id];
        auto nid_last = o2n_map_last[id];
        copy(emb_ptr + emb_dim * nid_last, emb_ptr + emb_dim * (nid_last + 1), result_ptr + emb_dim * nid);
    }


    result.resize({nei_num, emb_dim});
    free(WorkerStore::emb_reply_4push_ptr);
    return result;

}


void Router::pthread_vec_join(const vector<pthread_t> &pthreads) {
    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (WorkerStore::worker_id != i) {
            pthread_join(pthreads[i], NULL);
        }
    }
}

map<int, vector<int>> divideNodesToWk4Push(const vector<int> &nodes, const string &status, int layer_id) {
    map<int, vector<int>> push2eachwk_nodes;
    for (int i = 0; i < WorkerStore::worker_num; i++) {
        vector<int> tmp;
        push2eachwk_nodes.insert(pair<int, vector<int>>(i, tmp));
    }

    auto v2wk_map = WorkerStore::v2wk_push[status][layer_id - 1];
    for (auto id : nodes) {
        auto worker_ids = v2wk_map[id];
        for (auto wid:worker_ids) {
            push2eachwk_nodes[wid].push_back(id);
        }
    }
    return push2eachwk_nodes;

}

unordered_map<int, int> convertVec2Map(const vector<int> &nodes) {
    unordered_map<int, int> old2new_map;
    for (int i = 0; i < nodes.size(); i++) {
        old2new_map.insert(pair<int, int>(nodes[i], i));
    }
    return old2new_map;
}

//ReqEmbsMetaData* Router::buildStaticMetadata(int worker_id, ReqEmbsMetaData metaData){
//    if(metadata_map.count(worker_id)==0){
//        metadata_map.insert(pair<int,ReqEmbsMetaData>(worker_id,metaData));
//    }else{
//        metadata_map[worker_id]=metaData;
//    }
//}

//ReqEmbsMetaData* Router::getMetaData(int worker_id) {
//    if(metadata_vec.empty()){
//        for(int i=0;i<WorkerStore::worker_num;i++){
//            static ReqEmbsMetaData data_tmp;
//            metadata_vec.push_back(data_tmp);
//        }
//    }
//    return &metadata_vec[worker_id];
//}


void Router::pushEmbsByIds(int layer_id, int batch_id, const string &status, const vector<int> &nodes,
                           py::array_t<float> &embs) {
    // layer_id starts from 1
    // embs belong to layer 0, when layer_id=1pushEmbs
    map<int, vector<int>> push2wk_nodes_4lay = divideNodesToWk4Push(nodes, status, layer_id);
    unordered_map<int, int> old2new_map = convertVec2Map(nodes);
    auto buffer = embs.request();
    auto emb_ptr = (float *) buffer.ptr;
    int emb_dim = buffer.shape[1];

    for (int i = 0; i < WorkerStore::worker_num; i++) {
        if (WorkerStore::worker_id != i) {
            metadata_pipe[batch_id][i].dgnnClient = dgnnWorkerRouter[i];
            metadata_pipe[batch_id][i].nodes = new vector<int>(push2wk_nodes_4lay[i].size());
            *metadata_pipe[batch_id][i].nodes = push2wk_nodes_4lay[i];
            metadata_pipe[batch_id][i].oldToNewMap = new unordered_map<int, int>();
            *metadata_pipe[batch_id][i].oldToNewMap = old2new_map;
            metadata_pipe[batch_id][i].emb_ptr = (float *) malloc(sizeof(float) * buffer.shape[0] * buffer.shape[1]);
            copy(emb_ptr, emb_ptr + buffer.shape[0] * buffer.shape[1], metadata_pipe[batch_id][i].emb_ptr);
            metadata_pipe[batch_id][i].feat_num = emb_dim;
            metadata_pipe[batch_id][i].layerId = layer_id;
            metadata_pipe[batch_id][i].status = status;
            pthread_create(&ThreadUtil::pthread_vec[batch_id][i], NULL, DGNNClient::pushEmbsParallel,
                           (void *) &metadata_pipe[batch_id][i]);
        }
    }
    // ensure all thread have been started
    while (ThreadUtil::count_pushEmbsParallel != WorkerStore::worker_num - 1) {}
    ThreadUtil::count_pushEmbsParallel = 0;


}


void Router::initReplyEmbs(const string &status, int layer_id, int emb_dim, int batch_num, const string &graph_mode) {
    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs[status];
    int rmt_nei_num = subgraph.graphlayers[layer_id].vnum_tarv_locnei_rmtnei[2];
    WorkerStore::emb_reply_4push_ptr = new float[rmt_nei_num * emb_dim];

    static vector<vector<pthread_t>> pthread_tmp(batch_num);
    ThreadUtil::pthread_vec = pthread_tmp;

    static vector<vector<ReqEmbsMetaData>> req_tmp(batch_num);
    metadata_pipe = req_tmp;

    for (int i = 0; i < batch_num; i++) {
        auto meta_tmp = new vector<ReqEmbsMetaData>(WorkerStore::worker_num);
        metadata_pipe[i] = *meta_tmp;

        auto pthread_tmp = new vector<pthread_t>(WorkerStore::worker_num);
        ThreadUtil::pthread_vec[i] = *pthread_tmp;
    }

}


//py::array_t<float> Router::getEntireEmbs(const string &status, int layer_id, int emb_dim, const string& graph_mode) {
////    cout<<"getEntireEmbs 0"<<endl;
//    Graph *graph;
//    if(graph_mode=="full"){
//        graph=&WorkerStore::graph;
//    }else{
//        graph=&WorkerStore::graph_sampled;
//    }
//    auto& subgraph=graph->subgraphs[status];
//
//    for (const auto& p_batch:ThreadUtil::pthread_vec) {
//        for(auto p:p_batch){
//            pthread_join(p, NULL);
//        }
//    }
////    sleep(3);
//    dgnnServerRouter[0]->server_Barrier();
//    int rmt_nei_num = subgraph.graphlayers[layer_id].vnum_tarv_locnei_rmtnei[2];
//    WorkerStore::emb_reply_4push = py::array_t<float>(rmt_nei_num * emb_dim);
//    auto result_buffer = WorkerStore::emb_reply_4push.request();
//    auto result_ptr = (float *) result_buffer.ptr;
//    copy(WorkerStore::emb_reply_4push_ptr, WorkerStore::emb_reply_4push_ptr + rmt_nei_num * emb_dim, result_ptr);
//    WorkerStore::emb_reply_4push.resize({rmt_nei_num, emb_dim});
//    return WorkerStore::emb_reply_4push;
//
//}