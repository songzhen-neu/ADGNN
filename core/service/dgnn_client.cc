



#include "dgnn_client.h"



DGNNClient::DGNNClient(std::shared_ptr<Channel> channel) : stub_(DgnnProtoService::NewStub(channel)) {}

DGNNClient::DGNNClient() = default;

void DGNNClient::init(std::shared_ptr<Channel> channel) {
    stub_ = (DgnnProtoService::NewStub(channel));
}


void *DGNNClient::RunServer(void *address_tmp) {
    string address = *((string *) address_tmp);
    ServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(2147483647);
    builder.SetMaxSendMessageSize(2147483647);
    builder.SetMaxMessageSize(2147483647);


    std::unique_ptr<Server> server(builder.BuildAndStart());


    std::cout << "Server Listening on" << address << std::endl;
    server->Wait();
}


void DGNNClient::startClientServer() {

    pthread_t serverThread;
    pthread_create(&serverThread, NULL, DGNNClient::RunServer, (void *) &this->serverAddress);
//        ServiceImpl::RunServerByPy(address);

}


//void DGNNClient::sendNode(int layid, py::array_t<int> list) {
//    ClientContext context;
//    NodeMessage request;
//    NullMessage reply;
//    request.set_layid(layid);
//    for (int i = 0; i < list.size(); i++) {
//        request.add_nodes((int) list.at(i));
//    }
//    Status status = stub_->workerSendNode(&context, request, &reply);
//    if (status.ok()) {
////        cout << "okokokokok" << endl;
//    } else {
//        cout << "worker send node error" << endl;
//        cout << "error detail:" << status.error_details() << endl;
//        cout << "error message:" << status.error_message() << endl;
//        cout << "error code:" << status.error_code() << endl;
//    }
//}
//
//py::array_t<int> DGNNClient::pullNode(int layid) {
//    ClientContext context;
//    NodeMessage request;
//    NodeMessage reply;
//    request.set_layid(layid);
//    Status status = stub_->serverSendNode(&context, request, &reply);
//    vector<int> arr;
//    for (int i = 0; i < reply.nodes_size(); i++) {
//        arr.push_back(reply.nodes(i));
//    }
//    py::array_t<int> result = py::array_t<double>(arr.size());
//    py::buffer_info buf = result.request();
//    int *ptr = (int *) buf.ptr;
//    for (int i = 0; i < arr.size(); i++) {
//        ptr[i] = arr[i];
//    }
//    return result;
//}

void DGNNClient::init_by_address(std::string address) {
    grpc::ChannelArguments channel_args;
    channel_args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 2147483647);
    std::shared_ptr<Channel> channel = (grpc::CreateCustomChannel(
            address, grpc::InsecureChannelCredentials(), channel_args));
    WorkerStore::serverip = address;
    stub_ = DgnnProtoService::NewStub(channel);
}

//void DGNNClient::set_testString() {};

void
DGNNClient::initParameter(const int &worker_num, const int &server_num, const int &feat_dim,
                          const vector<int> &hid_dims, const int &class_dim,
                          const int &wid, map<string, vector<float>> &params) {
    // init workerstore some object
//    for (int i = 0; i < worker_num; i++) {
//        WorkerStore::compFlag.push_back(false);
//    }
    // 发送到server中
    NetInfoMessage request;
    request.set_featuredim(feat_dim);
    request.set_classdim(class_dim);
    request.set_servernum(server_num);
    for (auto hid_dim:hid_dims) {
        request.add_hiddendim(hid_dim);
    }
    request.set_workernum(worker_num);
    request.set_wid(wid);

    map<string, vector<float>>::iterator it;
    for (it = params.begin(); it != params.end(); it++) {
        auto *param = request.add_params();
        param->set_id(it->first);
        param->mutable_elems()->Add(it->second.begin(), it->second.end());
    }

    for (int i = 0; i < request.params_size(); i++) {
        cout << "id:" << request.params().Get(i).id() << ", size:" << request.params().Get(i).elems_size() << endl;
    }


    ClientContext context;
    NullMessage reply;
    stub_->initParameter(&context, request, &reply);
}


vector<int> DGNNClient::get_nodes() {
    return WorkerStore::graph.nodes;
}

void DGNNClient::set_nodes() {};

pair<py::array_t<int>, py::array_t<float>> DGNNClient::get_features() {
    int count = 0;
    auto &graph = WorkerStore::graph;
    int data_num = graph.features.size();
    int dim_size = graph.features.begin()->second.size();
    auto id = py::array_t<int>(data_num);
    auto id_ptr = (int *) id.request().ptr;
    auto val = py::array_t<float>(data_num * dim_size);
    val.resize({data_num, dim_size});
    auto val_ptr = (float *) val.request().ptr;
    for (auto elem:graph.features) {
        copy(elem.second.begin(), elem.second.end(), val_ptr + count * dim_size);
        id_ptr[count] = elem.first;
        count++;
    }

    return make_pair(id, val);
}

//void DGNNClient::set_nodesForEachWorker() {};
//
//vector<vector<int>> DGNNClient::get_nodesForEachWorker() {
//    return WorkerStore::nodesForEachWorker;
//}

void DGNNClient::set_features() {};

unordered_map<int, int> DGNNClient::get_labels() {
    return WorkerStore::graph.labels;
}

void DGNNClient::set_labels() {}

unordered_map<int, unordered_set<int>> DGNNClient::get_adjs() {
    return WorkerStore::graph.adjs;
}

//unordered_map<int, int> DGNNClient::get_degree_map() {
//    return WorkerStore::degree_map;
//}

void DGNNClient::set_degree_map() {}

string DGNNClient::get_serverAddress() {
    return this->serverAddress;
}

void DGNNClient::set_serverAddress(string serverAddress) {
    cout << serverAddress << endl;
    this->serverAddress = serverAddress;
}

void DGNNClient::set_adjs() {};

void DGNNClient::set_layerNum(int layerNum) {
    WorkerStore::layer_num = layerNum;
//    cout<<"layer_num"<<layerNum<<endl;
}


int DGNNClient::get_layerNum() {
    return WorkerStore::layer_num;
}


void DGNNClient::freeMaster() {
    NullMessage req;
    ClientContext context;
    NullMessage reply;
    stub_->freeMaster(&context, req, &reply);
}

unordered_map<int, int> DGNNClient::get_v2wk() {
    return WorkerStore::graph.v2wk;
}

void DGNNClient::set_v2wk() {}

vector<vector<int>> DGNNClient::get_wk2v() {
    return WorkerStore::graph.wk_contain_v;
}

void DGNNClient::set_wk2v() {}


void DGNNClient::freeSpace() {
    unordered_map<int, vector<float>>().swap(WorkerStore::graph.features);
    unordered_map<int, unordered_set<int>>().swap(WorkerStore::graph.adjs);
}

//void DGNNClient::pullDataFromMasterGeneral(
//        int worker_id, int worker_num, int data_num,
//        const string &data_path, int feature_dim, int class_num, const string &partitionMethod, int edgeNum) {
//    ContextMessage m;
//    ClientContext context;
//    DataMessage reply;
//    m.set_workerid(worker_id);
//    m.set_workernum(worker_num);
//    auto *partitionMessage = m.partition().New();
//    partitionMessage->set_workernum(worker_num);
//    partitionMessage->set_datanum(data_num);
//    partitionMessage->set_datapath(data_path);
//    partitionMessage->set_classnum(class_num);
//    partitionMessage->set_featuredim(feature_dim);
//    partitionMessage->set_partitionmethod(partitionMethod);
//    partitionMessage->set_edgenum(edgeNum);
//
//    m.set_allocated_partition(partitionMessage);
//    Status status = stub_->pullDataFromMasterGeneral(&context, m, &reply);
//    if (status.ok()) {
//        cout << "pullDataFromMaster completed" << endl;
//    } else {
//        std::cout << status.error_code() << ": " << status.error_message()
//                  << std::endl;
//    }
//    // 将获取到的数据set到自己本地的store中
//    cout << "***************local worker store****************" << endl;
//    WorkerStore::set_adjs(&reply.adjlist());
//    WorkerStore::set_labels(&reply.labellist());
//    WorkerStore::set_features(&reply.featurelist());
//    WorkerStore::set_nodes(&reply.nodelist());
//    WorkerStore::set_nodes_for_each_worker(&reply);
//    WorkerStore::set_degree_map(&reply);
//
//
//}


map<string, float>
DGNNClient::sendAccuracy(float val_acc, float train_acc, float test_acc) {
    ClientContext context;
    AccuracyMessage request;
    AccuracyMessage reply;


    request.set_val_acc(val_acc);
    request.set_train_acc(train_acc);
    request.set_test_acc(test_acc);

    Status status = stub_->sendAccuracy(&context, request, &reply);

    map<string, float> map_acc;
    map_acc.insert(pair<string, float>("val", reply.val_acc_entire()));
    map_acc.insert(pair<string, float>("train", reply.train_acc_entire()));
    map_acc.insert(pair<string, float>("test", reply.test_acc_entire()));

    return map_acc;
}


vector<int> DGNNClient::aggregateNodes(vector<int> nodes) {
    ClientContext context;
    NodeMessage request;
    NodeMessage reply;

    for (int i = 0; i < nodes.size(); i++) {
        request.add_nodes(nodes[i]);
    }

    Status status = stub_->aggregateNodes(&context, request, &reply);

    vector<int> agg_nodes;
    agg_nodes.assign(reply.nodes().begin(), reply.nodes().end());
    return agg_nodes;
}

void DGNNClient::setGraphInfoForCpp(const string &status,
                                    const map<int, map<int, vector<int>>> &fsthop4wk,
                                    const unordered_map<int, unordered_map<int, int>> &old2new,
                                    const unordered_map<int, unordered_map<int, int>> &new2old,
                                    const unordered_map<int, int> &v2wk_map,
                                    const map<int, int> &local_vertex_num,
                                    const map<int, int> &local_train_vertex_num,
                                    const map<int, int> &rmt_nei_num,
                                    const map<int, map<int, vector<int>>> &push_2_worker_nodes,
                                    const map<int, map<int, vector<int>>> &v2wk_push) {
//
//
//    int layer_num = WorkerStore::layer_num;
//    if (WorkerStore::fsthop_for_worker.count(status) == 0) {
//        WorkerStore::fsthop_for_worker.insert(pair<string, map<int, map<int, vector<int>>>>(status, fsthop4wk));
//        // add
//        WorkerStore::old2new_map.insert(pair<string, unordered_map<int, unordered_map<int, int>>>(status, old2new));
//        WorkerStore::new2old_map.insert(pair<string, unordered_map<int, unordered_map<int, int>>>(status, new2old));
//        // add
//        WorkerStore::local_vertex_num.insert(pair<string, map<int, int>>(status, local_vertex_num));
//        WorkerStore::local_train_vertex_num.insert(pair<string, map<int, int>>(status, local_train_vertex_num));
//        // add
//        if (status == "val") {
//            WorkerStore::v2wk_map = v2wk_map;
//        }
//        float *tmp = nullptr;
//        WorkerStore::embs_ptr_map.insert(pair<string, float *>(status, tmp));
//        for (int i = 0; i < layer_num; i++) {
//            vector<vector<float>> tmp1;
//            WorkerStore::local_emb_grad_agg.insert(pair<int, vector<vector<float>>>(i + 1, tmp1));
//        }
//
//        WorkerStore::rmt_nei_num.insert(pair<string, map<int, int>>(status, rmt_nei_num));
//        WorkerStore::push_2_worker_nodes.insert(
//                pair<string, map<int, map<int, vector<int>>>>(status, push_2_worker_nodes));
//        WorkerStore::v2wk_push.insert(
//                pair<string, map<int, map<int, vector<int>>>>(status, v2wk_push));
//
//    } else {
//        WorkerStore::fsthop_for_worker[status] = fsthop4wk;
//        WorkerStore::old2new_map[status] = old2new;
//        WorkerStore::new2old_map[status] = new2old;
//        WorkerStore::local_vertex_num[status] = local_vertex_num;
//        WorkerStore::local_train_vertex_num[status] = local_train_vertex_num;
//        WorkerStore::rmt_nei_num[status] = rmt_nei_num;
//        WorkerStore::push_2_worker_nodes[status] = push_2_worker_nodes;
//        WorkerStore::v2wk_push[status] = v2wk_push;
//    }
//
//
}


void DGNNClient::setCtxForCpp(map<string, string> &cluster_config) {
    WorkerStore::worker_num = stoi(cluster_config["worker_num"]);
    WorkerStore::worker_id = stoi(cluster_config["worker_id"]);
    WorkerStore::layer_num = stoi(cluster_config["layer_num"]);
    WorkerStore::num_threads = stoi(cluster_config["num_threads"]);
    WorkerStore::vertex_num = stoi(cluster_config["vertex_num"]);
    WorkerStore::train_num = stoi(cluster_config["train_num"]);
    WorkerStore::val_num = stoi(cluster_config["val_num"]);
    WorkerStore::test_num = stoi(cluster_config["test_num"]);
    WorkerStore::feat_size = stoi(cluster_config["feat_size"]);
    WorkerStore::filename = cluster_config["filename"];
    WorkerStore::sample_method = cluster_config["sample_method"];
    WorkerStore::partition_method = cluster_config["partition_method"];

    auto &graph = WorkerStore::graph;
    SubGraph subGraph_train;
    graph.subgraphs.insert(make_pair("train", subGraph_train));
    SubGraph subGraph_val;
    graph.subgraphs.insert(make_pair("val", subGraph_val));
    SubGraph subGraph_test;
    graph.subgraphs.insert(make_pair("test", subGraph_test));

    auto &graph_sampled = WorkerStore::graph_sampled;
    SubGraph subGraph_sampled_train;
    graph_sampled.subgraphs.insert(make_pair("train", subGraph_sampled_train));

    for (int i = 0; i < WorkerStore::layer_num + 1; i++) {
        GraphLayer gl_train;
        graph.subgraphs["train"].graphlayers.emplace_back(gl_train);
        GraphLayer gl_val;
        graph.subgraphs["val"].graphlayers.emplace_back(gl_val);
        GraphLayer gl_test;
        graph.subgraphs["test"].graphlayers.emplace_back(gl_test);
        GraphLayer gl_sampled_train;
        graph_sampled.subgraphs["train"].graphlayers.emplace_back(gl_sampled_train);
    }

    unordered_set<int> idx_train;
    graph.idx.insert(make_pair("train", idx_train));
    unordered_set<int> idx_val;
    graph.idx.insert(make_pair("val", idx_val));
    unordered_set<int> idx_test;
    graph.idx.insert(make_pair("test", idx_test));



}


void DGNNClient::setStaticInfoForCpp(const unordered_map<int, unordered_set<int>> &rmt_nei_set,
                                     const unordered_map<int, unordered_set<int>> &loc_nei_set) {
//    WorkerStore::rmt_nei_set = rmt_nei_set;
//    WorkerStore::loc_nei_set = loc_nei_set;
}

void *DGNNClient::worker_pull_needed_emb_parallel(void *metaData_void) {
    auto metaData = (ReqEmbsMetaData *) metaData_void;
    unordered_set<int> &nodes = *metaData->nei_set;
    int layerId = metaData->layerId;
    int workerId = metaData->workerId;
    int serverId = metaData->serverId;
    EmbGradMessage &reply = *metaData->reply;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    float *ptr_result = metaData->ptr_result;
    string train_status = metaData->status;
    auto oldToNewMap = *metaData->oldToNewMap;
    int localNodeSize = metaData->localNodeSize;
    int feat_num = metaData->feat_num;

    int nodeNum = nodes.size();

    //    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;

    // 去服务器中获取嵌入,这里建立的是每个worker的channel

    ClientContext context;
    EmbGradMessage request;
//    RespEmbSparseMessage reply;
    request.set_layerid(layerId);
    request.set_workerid(workerId);
    request.set_featsize(feat_num);
    request.set_status(train_status);


    // 构建request
    request.mutable_nodes()->Add(nodes.begin(), nodes.end());
//    for (int i = 0; i < nodeNum; i++) {
//        request.add_nodes(nodes[i]);
//    }


    Status status = dgnnClient->stub_->workerPullEmb(&context, request, &reply);

    if (status.ok()) {
        auto &embConcat = reply.embs();
        if (embConcat.size() != nodes.size() * feat_num) {
            cout << "worker_pull_needed_emb_parallel: get remote feat error:" << embConcat.size() << endl;
        }
        int flag=0;
        for (auto nid:nodes) {
            int new_id = oldToNewMap[nid] - localNodeSize;
            for (int k = 0; k < feat_num; k++) {
                ptr_result[new_id * feat_num + k] = embConcat.Get(flag * feat_num + k);
            }
            flag++;
        }

        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
        ThreadUtil::count_respWorkerNumForEmbs++;
        lck.unlock();
    } else {
        cout << "pull needed embeddings false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }


}


void DGNNClient::server_Barrier() {
    ClientContext clientContext;
    NullMessage request;
    NullMessage reply;
    Status status = stub_->barrier(&clientContext, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "server_Barrier false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}


void *DGNNClient::worker_pull_g_parallel(void *metaData_void) {
    auto metaData = (ReqEmbsMetaData *) metaData_void;
    ClientContext context;
    NullMessage reply;
    EmbGradMessage request = *metaData->embGradMessage;
    auto *dgnnClient = metaData->dgnnClient;

    Status status = dgnnClient->stub_->setAndSendG(&context, request, &reply);
    if (status.ok()) {
//        unique_lock<mutex> lck(ThreadUtil::mtx_setAndSendG_for_count);
//        ThreadUtil::count_setAndSendG++;
//        lck.unlock();
    } else {
        cout << "worker_pull_g_parallel false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }

}


py::array_t<float> DGNNClient::server_PullParams(const string &param_id) {
    ClientContext context;
    ParamGrad request;
    request.set_id(param_id);

    ParamGrad reply;
    Status status = stub_->server_PullParams(&context, request, &reply);
    auto result = py::array_t<float>(reply.elems_size());

    if (status.ok()) {
        py::buffer_info buf_result = result.request();
        auto *ptr_result = (float *) buf_result.ptr;
        for (int i = 0; i < reply.elems_size(); i++) {
            ptr_result[i] = reply.elems(i);
        }
//        cout<<"result size:"<<result.size()<<endl;
    } else {
        cout << "pull parameters error" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;

    }
    return result;

}


void
DGNNClient::server_updateParam(int worker_id, int server_id, float lr, const string &key, py::array_t<double> &grad) {
    // 发送收到的梯度到参数服务器中
    ClientContext context;
    ParamGrad request;
    NullMessage reply;

    request.set_wid(worker_id);
    request.set_sid(server_id);
    request.set_lr(lr);

    request.set_id(key);

    py::buffer_info grad_buf = grad.request();
    auto *ptr1 = (double *) grad_buf.ptr;
    request.mutable_elems()->Add(ptr1, ptr1 + grad.size());


    Status status = stub_->server_updateParam(&context, request, &reply);


    if (status.ok()) {

    } else {
        cout << "update parameters false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }

}


void *DGNNClient::sendNodes2Wk(void *metaData_void) {
    auto metaData = (ReqEmbsMetaData *) metaData_void;
    auto nodes = *metaData->nodes;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    int layid = metaData->layerId;
    ClientContext context;
    NodeMessage request;
    NullMessage reply;
    request.set_layid(layid);
    request.set_wid(WorkerStore::worker_id);
    request.mutable_nodes()->Add(nodes.begin(), nodes.end());
    Status status = dgnnClient->stub_->sendNodes2Wk(&context, request, &reply);
    if (status.ok()) {
        delete metaData;
    } else {
        cout << "error sendNodes2wk" << endl;
    }
//    ThreadUtil::count_sendNodes2Wk_threads++;
}


void *DGNNClient::getRmtFeat(void *metaData_void) {


    auto metaData = (ReqEmbsMetaData *) metaData_void;
    unordered_set<int> &nodes_set = *metaData->nei_set;
    auto nodes=SetVecTrans::set2vec(nodes_set);


    EmbGradMessage &reply = *metaData->reply;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    auto oldToNewMap = *metaData->oldToNewMap;
    int feat_num = metaData->feat_num;
    auto thread_id=metaData->thread_id;
    ClientContext context;
    EmbGradMessage request;
    request.set_featsize(feat_num);
    request.set_workerid(WorkerStore::worker_id);
    request.mutable_nodes()->Add(nodes.begin(), nodes.end());


    Status status = dgnnClient->stub_->workerPullRmtTrainFeat(&context, request, &reply);


    if (status.ok()) {

        auto &embConcat = reply.embs();
        int flag=0;
        unique_lock<mutex> lck_feat(ThreadUtil::mtx_rmtfeature_insert);
        for (auto nid:nodes) {
            vector<float> feat_tmp(feat_num);
            for (int k = 0; k < feat_num; k++) {
                feat_tmp[k]=embConcat.Get(flag*feat_num+k);
            }
            WorkerStore::graph.features.insert(make_pair(nid,feat_tmp));

//            int new_id = oldToNewMap[nid] - localNodeSize;
//            for (int k = 0; k < feat_num; k++) {
//                ptr_result[new_id * feat_num + k] = embConcat.Get(flag * feat_num + k);
//            }
            flag++;
        }
//        cout<<"7777777777777"<<endl;
        lck_feat.unlock();
//        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
//        ThreadUtil::count_respWorkerNumForEmbs++;
//        lck.unlock();
    } else {
        cout << "pull needed embeddings false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }
}

void *DGNNClient::sendInNodes2WK(void *metaData_void) {
    auto metaData = (ReqEmbsMetaData *) metaData_void;
    auto nodes = *metaData->nei_set;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    int layid = metaData->layerId;
    int thread_id=metaData->thread_id;
    int workerId = metaData->workerId;
    ClientContext context;
    NodeMessage request;
    NullMessage reply;
    request.set_layid(layid);
    request.set_wid(WorkerStore::worker_id);
    request.mutable_nodes()->Add(nodes.begin(), nodes.end());
    Status status = dgnnClient->stub_->sendInNodes2Wk(&context, request, &reply);
    if (status.ok()) {

    } else {
        cout << "error sendNodes2wk" << endl;
    }
//    ThreadUtil::count_sendNodes2Wk_threads++;
}

void *DGNNClient::pushEmbsParallel(void *metaData) {

//    unique_lock<mutex> lck(ThreadUtil::mtx_pushEmbsParallel);
//    ThreadUtil::count_pushEmbsParallel++;
//    lck.unlock();

    auto data = (ReqEmbsMetaData *) metaData;
    auto old2new_map = *data->oldToNewMap;
    auto nodes = *data->nei_set_unorder;
    auto dgnnClient = data->dgnnClient;
    auto emb_ptr = data->emb_ptr;
    auto emb_dim = data->feat_num;
    int layer_id = data->layerId;
    auto graph_mode=data->graph_mode;
    string status = data->status;
//    for(auto id :nodes){
//        cout<<id<<",";
//    }
//    cout<<endl;
    ClientContext context;
    EmbGradMessage request;
    NullMessage reply;

//    cout<<"nei_set_unorder:";
//    for(auto id :nodes){
//        cout<<id<<",";
//    }
//    cout<<endl;

    request.mutable_nodes()->Add(nodes.begin(), nodes.end());
    request.set_layerid(layer_id);
    request.set_status(status);
    request.set_featsize(emb_dim);
    request.set_graph_mode(graph_mode);
    for (auto id:nodes) {
        int new_id = old2new_map[id];
        request.mutable_embs()->Add(emb_ptr + new_id * emb_dim, emb_ptr + (new_id + 1) * emb_dim); // To check this
    }

    Status status_grpc = dgnnClient->stub_->pushEmbs(&context, request, &reply);
//    sleep(0.25);

    if (status_grpc.ok()) {
    } else {
        cout << status_grpc.error_code() << "," << status_grpc.error_details() << "," << status_grpc.error_message()
             << endl;
        exit(1);
    }


}





void DGNNClient::readDataAndInit() {
//    char *buffer;
//    buffer = getcwd(NULL, 0);
//    string pwd = buffer;
//    cout << "pwd:" << pwd << ",buffer:" << buffer << endl;
////    string pwd =buffer;
//    if (pwd[pwd.length() - 1] != '/') {
//        pwd += '/';
//    }
    auto worker_id = WorkerStore::worker_id;
    auto filename = WorkerStore::filename;
    auto worker_num = WorkerStore::worker_num;
    auto feat_size = WorkerStore::feat_size;
    auto &graph = WorkerStore::graph;

    string partitionFile =
            filename + "/nodesPartition" + "." + WorkerStore::partition_method + to_string(worker_num) + ".txt";
    string fn_prefix =
            filename + "/" + WorkerStore::partition_method + to_string(worker_num) + "/part" + to_string(worker_id);
    string adj_fn = fn_prefix + "/adj.txt";
    string feat_fn = fn_prefix + "/feat.txt";
    string label_fn = fn_prefix + "/label.txt";

    cout << "fn_prefix: " << fn_prefix << endl;
    cout << "adj_fn: " << adj_fn << endl;
    cout << "feat_fn: " << feat_fn << endl;
    cout << "label_fn: " << label_fn << endl;
    cout << "partitionFile: " << partitionFile << endl;

    ifstream partitionInFile(partitionFile);
    if (!partitionInFile.is_open()) {
        cout << "partitionInFile open unsuccessfully" << endl;
    }
    int count_worker = 0;
    string temp;
    while (getline(partitionInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vSize = v.size();
        vector<int> v_tmp(vSize);
        graph.wk_contain_v.push_back(v_tmp);
        auto &nodesWorker = graph.wk_contain_v[count_worker];
        for (int i = 0; i < vSize; i++) {
            nodesWorker[i] = atoi(v[i].c_str());
//            cout<<nodesWorker[i]<<",";
        }
//        cout<<endl;

        cout << "nodes num for worker " << count_worker << " :" << graph.wk_contain_v[count_worker].size() << endl;
        count_worker++;
    }

    partitionInFile.close();
    graph.nodes = graph.wk_contain_v[worker_id];



    ifstream featInFile(feat_fn);
    if (!featInFile.is_open()) {
        cout << "featInFile open unsuccessfully" << endl;
    }

    int count_flag = 0;
    cout << "processing data " << endl;

    while (true) {
        getline(featInFile, temp);
        if (temp.empty()) {
            break;
        }
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        vector<float> vec_feat(feat_size);
        for (int i = 1; i < feat_size + 1; i++) {
            vec_feat[i - 1] = atof(v[i].c_str());
//            if(vec_feat[i - 1]>10000){
//                cout<<"aaaaaaaaaaa:"<<vec_feat[i-1]<<endl;
//            }
        }
        graph.features.insert(pair<int, vector<float>>(vertex_id, vec_feat));

        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "have processed " << count_flag << " features" << endl;
        }
    }



    count_flag = 0;
    featInFile.close();


    cout << "processing adj" << endl;
    ifstream adjInFile(adj_fn);
    if (!adjInFile.is_open()) {
        cout << "adjInFile open unsuccessfully" << endl;
    }
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        unordered_set<int> nei_set;
        for (int i = 1; i < v.size(); i++) {
            int nid = atoi(v[i].c_str());
            nei_set.insert(nid);
            count_flag++;
        }
        graph.adjs.insert(make_pair(vertex_id, nei_set));
        if (count_flag % (10000) == 0) {
            cout << "have processed " << count_flag << " edges" << endl;
        }
    }
    int edge_num = count_flag;
    count_flag = 0;
    adjInFile.close();


    ifstream labelInFile(label_fn);
    if (!labelInFile.is_open()) {
        cout << "labelInFile open unsuccessfully" << endl;
    }
    while (getline(labelInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int label = atoi(v[1].c_str());
        graph.labels.insert(make_pair(vertex_id, label));
        if (count_flag % (10000) == 0) {
            cout << "have processed " << count_flag << " labels" << endl;
        }
        count_flag++;
    }
    count_flag = 0;
    labelInFile.close();


    // 开始划分，邻接表、顶点map、属性、标签
    // 这里顶点按照哈希（取余数）的方式进行划分，因此不需要建立map
    // 邻接表：map<int, map<int,set>>

    cout << "边数:" << edge_num << endl;


    for (int i = 0; i < graph.wk_contain_v.size(); i++) {
        auto &nodes_wk = graph.wk_contain_v[i];
        for (int j = 0; j < nodes_wk.size(); j++) {
            WorkerStore::graph.v2wk.insert(make_pair(nodes_wk[j], i));
        }
    }

    cout<<"data:"<<WorkerStore::graph.features.size()<<","<<WorkerStore::graph.nodes.size()<<","<<WorkerStore::graph.v2wk.size()<<endl;


}


pair<int,int> DGNNClient::setM(int id, int m, int benefit_m){
    ClientContext ctx;
    MMessageForAD req;
    MMessageForAD res;

    req.set_wid(id);
    req.set_m(m);
    req.set_m_benefit(benefit_m);
    Status status=stub_->setM(&ctx,req,&res);

    if(status.ok()){
        return make_pair(res.m(),res.m_benefit());
    }else{
        cout << "set M error" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}





