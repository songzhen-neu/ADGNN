
#include <sys/time.h>
#include "dgnn_server.h"
// define the final ServiceImpl class (it cannot be inherited); and it extends the class Service by 'public' manner,
// it can access any public member in the parent class, expect private members.



GeneralPartition generalPartition;

Status ServiceImpl::pullDataFromMasterGeneral(
        ServerContext *context, const ContextMessage *request,
        DataMessage *reply) {

    cout << "worker " << request->workerid() << " has arrived!" << endl;
    int workerid = request->workerid();
    int workerNum = request->workernum();

    if (request->workerid() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx);

        Check::check_partition_pass(
                request->workernum(),
                request->partition().datanum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());
        //int data_num, int worker_num, string filename, int feature_size, int label_size;

        generalPartition.init(
                request->partition().datanum(),
                request->workernum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());

        generalPartition.startPartition(workerNum, request->partition().partitionmethod(),
                                        request->partition().datanum(), request->partition().edgenum());
        ThreadUtil::ready = true;
        ThreadUtil::cv.notify_all();


    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx);

        while (!ThreadUtil::ready) {
            ThreadUtil::cv.wait(lck);
        }
    }



    NodeMessage *nodeMessage = reply->nodelist().New();
    for (int id:GeneralPartition::nodes[workerid]) {
        nodeMessage->add_nodes(id);
    }
    reply->set_allocated_nodelist(nodeMessage);


    DataMessage_FeatureMessage *featureMessage = reply->featurelist().New();
    for (auto &id_feature : GeneralPartition::features[workerid]) {
        DataMessage_FeatureMessage_FeatureItem *item = featureMessage->add_features();
        item->set_vid(id_feature.first);
        for (float feat_dim:id_feature.second) {
            item->add_feature(feat_dim);
        }
    }
    reply->set_allocated_featurelist(featureMessage);


    DataMessage_LabelMessage *labelMessage = reply->labellist().New();
    for (auto &id_label:GeneralPartition::labels[workerid]) {
        DataMessage_LabelMessage_LabelItem *item = labelMessage->add_labels();
        item->set_vid(id_label.first);
        item->set_label(id_label.second);
    }
    reply->set_allocated_labellist(labelMessage);


    DataMessage_AdjMessage *adjMessage = reply->adjlist().New();
    for (const auto &id_neibors:GeneralPartition::adjs[workerid]) {
        DataMessage_AdjMessage_AdjItem *adjItem = adjMessage->add_adjs();
        adjItem->set_vid(id_neibors.first);
        for (auto neibor:id_neibors.second) {
            adjItem->add_neibors(neibor);
        }
    }
    reply->set_allocated_adjlist(adjMessage);

    for (int i = 0; i < workerNum; i++) {
        auto &nodes_worker = GeneralPartition::nodes[i];
        auto *nodelist = reply->add_nodesforeachworker();
        int nodeNum = nodes_worker.size();
        for (int j = 0; j < nodeNum; j++) {
            nodelist->add_nodes(nodes_worker[j]);
        }
    }

    // build degree map
    for (auto &item:GeneralPartition::degree_map) {
        IntIntPair *intIntPair = reply->add_degreemap();
        intIntPair->set_key(item.first);
        intIntPair->set_value(item.second);

//        cout<<item.first<<","<<item.second<<"   ";
    }
//    cout<<endl;
//    for(auto &item:reply->degreemap()){
//        cout<<item.key()<<","<<item.value()<<"   ";
//    }
//    cout<<endl;
//    cout<<"server degree_map message size: "<<GeneralPartition::degree_map.size()<<endl;

    return Status::OK;

//     vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
}

Status ServiceImpl::freeMaster(ServerContext *context, const NullMessage *request, NullMessage *reply) {
    vector<vector<int>>().swap(GeneralPartition::nodes);
    vector<map<int, vector<float>>>().swap(GeneralPartition::features);
    vector<map<int, int>>().swap(GeneralPartition::labels);
    vector<map<int, unordered_set<int>>>().swap(GeneralPartition::adjs);
    return Status::OK;
}


Status ServiceImpl::workerPullEmb(
        ServerContext *context, const EmbGradMessage *request, EmbGradMessage *reply) {

//    clock_t start = clock();
    string mode = "none"; // mom mv none


    int feat_size = request->featsize();
    int layerId = request->layerid();
    int nodeNum = request->nodes_size();
    const string &status = request->status();
    float *embs = WorkerStore::embs_ptr_map[status];
    const string &graph_mode = request->graph_mode();
    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs[status];

    reply->set_featsize(feat_size);
    reply->set_nodesize(nodeNum);


    auto *mutable_emb_reply = reply->mutable_embs();
    mutable_emb_reply->Reserve(nodeNum * feat_size);
    auto &o2n_map_last = subgraph.graphlayers[layerId - 1].o2n_map;
//    cout << "size = "<<WorkerStore::train_oidnid_embs[layerId].size() << " feat_size = " << feat_size<<" lay_id = "<< layerId<<endl;
    for (int i = 0; i < nodeNum; i++) {
        int oid = request->nodes(i);
        int nid = o2n_map_last[oid]; //- WorkerStore::train_local_num[layerId+1];
        mutable_emb_reply->Add(embs + nid * feat_size, embs + (nid + 1) * feat_size);
    }


    return Status::OK;
}


Status ServiceImpl::initParameter(ServerContext *context, const NetInfoMessage *request, NullMessage *reply) {
    int serverId = ServerStore::serverId;
    cout << "Server " << serverId << ": initParameters begining!" << endl;

    int worker_num = request->workernum();
    int feat_dim = request->featuredim();
    int server_num = request->servernum();
    vector<int> hid_dims;
    int wid = request->wid();

    for (auto hid_dim:request->hiddendim()) {
        hid_dims.push_back(hid_dim);
    }

    int class_dim = request->classdim();
    cout << "Server revceived:" << endl;
    if (wid == 0) {
        string hid = "hidden size:";
        for (auto dim:hid_dims) {
            hid.append(to_string(dim) + ",");
        }
        hid.pop_back();
        cout << "worker number:" << worker_num << ", server number: " << server_num << ",feature dimensions:"
             << feat_dim << ",class dimensions:"
             << class_dim << ",hidden size:" << hid << endl;
    }


    if (wid == 0) {
        unique_lock<mutex> mutex(ThreadUtil::mtx_initParameter);
        // 初始化神经网络参数

        ServerStore::worker_num = worker_num;
        ServerStore::server_num = server_num;
        ServerStore::feat_dim = feat_dim;
        ServerStore::class_dim = class_dim;
        ServerStore::hid_dims = hid_dims;
        for (int i = 0; i < request->params_size(); i++) {
            int size_elem = request->params(i).elems_size();
            vector<double> tmp(size_elem);
            vector<double> tmp2(size_elem);
            vector<double> tmp3(size_elem);
            vector<double> tmp4(size_elem);
            auto &param_i = request->params(i);
            for (int j = 0; j < size_elem; j++) {
                tmp[j] = param_i.elems(j);
            }
            ServerStore::params.insert(pair<string, vector<double>>(param_i.id(), tmp));
            ServerStore::grads_agg.insert(pair<string, vector<double>>(param_i.id(), tmp2));
            ServerStore::m_grads_t.insert(pair<string, vector<double>>(param_i.id(), tmp3));
            ServerStore::v_grads_t.insert(pair<string, vector<double>>(param_i.id(), tmp4));
            ServerStore::t.insert(pair<string, int>(param_i.id(), 0));
        }


        Check::check_initParameter_ServerStore();


//        ServerStore::initParams(worker_num, server_num, feat_dim, class_dim, hid_dims);
        ThreadUtil::ready_initParameter = true;
        ThreadUtil::cv.notify_all();

    } else {
        unique_lock<mutex> mutex(ThreadUtil::mtx_initParameter);
        while (!ThreadUtil::ready_initParameter) {
            ThreadUtil::cv.wait(mutex);
        }
    }

    cout << "initParameters ending!" << endl;
    return Status::OK;

}


Status ServiceImpl::barrier(
        ServerContext *context, const NullMessage *request, NullMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_barrier);
    ThreadUtil::count_worker_for_barrier++;
    if (ThreadUtil::count_worker_for_barrier == ServerStore::worker_num) {
        ThreadUtil::count_worker_for_barrier = 0;
        ThreadUtil::cv_barrier.notify_all();

    } else {
        ThreadUtil::cv_barrier.wait(lck);
    }
    return Status::OK;
}


void ServiceImpl::RunServerByPy(const string &address, int serverId) {
    ServiceImpl service;

    ServerStore::serverId = serverId;
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(2147483647);
    builder.SetMaxSendMessageSize(2147483647);
    builder.SetMaxMessageSize(2147483647);


    std::unique_ptr<Server> server(builder.BuildAndStart());


    std::cout << "Server Listening on " << address << std::endl;
    std::cout << "if signal 11, please check ip :" << address << std::endl;
    server->Wait();

}


Status ServiceImpl::sendAccuracy(ServerContext *context, const AccuracyMessage *request,
                                 AccuracyMessage *reply) {


    unique_lock<mutex> lck(ThreadUtil::mtx_accuracy);
    if (ThreadUtil::count_accuracy == 0) {
        ServerStore::val_accuracy = 0;
        ServerStore::train_accuracy = 0;
        ServerStore::test_accuracy = 0;

    }
    ServerStore::val_accuracy += request->val_acc();
    ServerStore::train_accuracy += request->train_acc();
    ServerStore::test_accuracy += request->test_acc();


    ThreadUtil::count_accuracy++;
    if (ThreadUtil::count_accuracy == ServerStore::worker_num) {
        ThreadUtil::count_accuracy = 0;
        ThreadUtil::cv_accuracy.notify_all();

    } else {
        ThreadUtil::cv_accuracy.wait(lck);
    }


    reply->set_val_acc_entire(ServerStore::val_accuracy);
    reply->set_train_acc_entire(ServerStore::train_accuracy);
    reply->set_test_acc_entire(ServerStore::test_accuracy);

    return Status::OK;

}

Status ServiceImpl::aggregateNodes(ServerContext *context, const NodeMessage *request,
                                   NodeMessage *reply) {

    unique_lock<mutex> lck(ThreadUtil::mtx_nodes);
    if (ThreadUtil::count_nodes == 0) {
        ServerStore::agg_nodes.clear();
    }

    ServerStore::agg_nodes.insert(ServerStore::agg_nodes.end(), request->nodes().begin(), request->nodes().end());

    ThreadUtil::count_nodes++;
    if (ThreadUtil::count_nodes == ServerStore::worker_num) {
        ThreadUtil::count_nodes = 0;
        ThreadUtil::cv_nodes.notify_all();
    } else {
        ThreadUtil::cv_nodes.wait(lck);
    }

    for (int i = 0; i < ServerStore::agg_nodes.size(); i++) {
        reply->add_nodes(ServerStore::agg_nodes[i]);
    }

    return Status::OK;

}


Status ServiceImpl::server_PullParams(ServerContext *context, const ParamGrad *request, ParamGrad *reply) {
    const string &lay_id = request->id();
    reply->mutable_elems()->Add(ServerStore::params[lay_id].begin(), ServerStore::params[lay_id].end());
    return Status::OK;
}




void barrier_update() {
    unique_lock<mutex> lck_barrier(ThreadUtil::mtx_barrier);
    ThreadUtil::count_worker_for_barrier++;
    if (ThreadUtil::count_worker_for_barrier == ServerStore::worker_num) {
        ThreadUtil::count_worker_for_barrier = 0;
        ThreadUtil::cv_barrier.notify_all();
    } else {
        ThreadUtil::cv_barrier.wait(lck_barrier);
    }
}


Status ServiceImpl::server_updateParam(ServerContext *context, const ParamGrad *request, NullMessage *reply) {
    if (request->wid() == 0) {
        ServerStore::grads_agg[request->id()].clear();
        // vector is initialized as 0 by default
        vector<double> tmp(request->elems_size());
        ServerStore::grads_agg[request->id()] = tmp;
        cout << "********server_updateModels-clear gradient aggregations******" << endl;

    }


    barrier_update();


    int grad_size = request->elems_size();
    string grad_id = request->id();
    float alpha = request->lr();
    int wid = request->wid();


//    pthread_mutex_lock(&ThreadUtil::mtx_updateModels_addGrad);
    unique_lock<mutex> lck_update(ThreadUtil::mtx_updateModels);

    auto &grad_agg = ServerStore::grads_agg[grad_id];
    // add gradients to grads_agg
    for (int i = 0; i < grad_size; i++) {
        grad_agg[i] = grad_agg[i] + request->elems(i);
    }
    cout << "********server_updateModels----gradient aggregating end******" << endl;
    lck_update.unlock();



    barrier_update();


    if (wid == 0) {
        cout << ThreadUtil::count_worker_for_updateModels << " workers have been added into the gradient aggregations!"
             << endl;
        cout << "param id:" << grad_id << "," << "grad size:" << grad_agg.size() << endl;
    }



    if (wid == 0) {

        ServerStore::t[grad_id]++;

        float beta_1 = 0.9;
        float beta_2 = 0.999;
        float epsilon = 1e-8;
        bool isAdam = true;
        auto &m_grads_t = ServerStore::m_grads_t[grad_id];
        auto &v_grads_t = ServerStore::v_grads_t[grad_id];
        auto &param = ServerStore::params[grad_id];
        auto bias_correction1 = 1 - pow(beta_1, ServerStore::t[grad_id]);
        auto bias_correction2 = 1 - pow(beta_2, ServerStore::t[grad_id]);
        for (int i = 0; i < grad_size; i++) {
            double g_t = grad_agg[i];
            if (isAdam) {
                m_grads_t[i] = beta_1 * m_grads_t[i] + (1 - beta_1) * g_t;
                v_grads_t[i] = beta_2 * v_grads_t[i] + (1 - beta_2) * g_t * g_t;
                auto denom = sqrt(v_grads_t[i]) / sqrt(bias_correction2) + epsilon;
                auto step_size = alpha / bias_correction1;

                param[i] -= m_grads_t[i] / denom * step_size;
            } else {
                param[i] -= alpha * g_t;
            }
        }


    }

    barrier_update();

    return Status::OK;
}


Status ServiceImpl::setAndSendG(ServerContext *context, const EmbGradMessage *request, NullMessage *reply) {
    int dim_num = request->featsize();
    int layer_id = request->layerid();
    const string &graph_mode = request->graph_mode();
    auto wid = request->workerid();


    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs["train"];
    auto &o2n_last = subgraph.graphlayers[layer_id - 1].o2n_map;
    unique_lock<mutex> lck(ThreadUtil::mtx_setAndSendG);
    auto &loc_embgrad_agg_lay = WorkerStore::local_emb_grad_agg[layer_id];
    for (int i = 0; i < request->nodes_size(); i++) {
        int oid = request->nodes(i);
        int nid = o2n_last[oid];
        auto &loc_embgrad_agg_node = loc_embgrad_agg_lay[nid];
        for (int j = 0; j < dim_num; j++) {
            loc_embgrad_agg_node[j] += request->embs(i * dim_num + j);
        }
    }
    lck.unlock();
    return Status::OK;
}

Status ServiceImpl::sendNodes2Wk(ServerContext *context, const NodeMessage *request, NullMessage *reply) {

    unique_lock<mutex> lck(ThreadUtil::mtx_sendNodes2Wk);
    ThreadUtil::count_sendNodes2Wk++;
    if (ThreadUtil::count_sendNodes2Wk == WorkerStore::worker_num - 1) {
        WorkerStore::neis_neededby_otherwk[request->layid()].clear();
        ThreadUtil::count_sendNodes2Wk = 0;
        ThreadUtil::cv_sendNodes2Wk.notify_all();

    } else {
        ThreadUtil::cv_sendNodes2Wk.wait(lck);
    }
    lck.unlock();

    unique_lock<mutex> lck_merge(ThreadUtil::mtx_sendNodes2Wk_merge);
    for (auto id : request->nodes()) {
        WorkerStore::neis_neededby_otherwk[request->layid()].insert(id);
    }
    lck_merge.unlock();

    lck.lock();
    ThreadUtil::count_sendNodes2Wk++;
    if (ThreadUtil::count_sendNodes2Wk == WorkerStore::worker_num - 1) {
        ThreadUtil::cv_sendNodes2Wk.notify_all();
        ThreadUtil::count_sendNodes2Wk = 0;
    } else {
        ThreadUtil::cv_sendNodes2Wk.wait(lck);
    }
    lck.unlock();

    return Status::OK;


}


Status
ServiceImpl::workerPullRmtTrainFeat(ServerContext *context, const EmbGradMessage *request, EmbGradMessage *reply) {
    int feat_size = request->featsize();
    int nodeNum = request->nodes_size();
    int wid = request->workerid();

    reply->set_featsize(feat_size);
    reply->set_nodesize(nodeNum);

    auto *mutable_emb_reply = reply->mutable_embs();
    mutable_emb_reply->Reserve(nodeNum * feat_size);

    unique_lock<mutex> lck(ThreadUtil::mtx_rmtfeature_insert);
    for (int i = 0; i < nodeNum; i++) {
        int oid = request->nodes(i);
        mutable_emb_reply->Add(WorkerStore::graph.features[oid].begin(),
                               WorkerStore::graph.features[oid].end());
    }
    lck.unlock();
    return Status::OK;
}

Status ServiceImpl::sendInNodes2Wk(ServerContext *context, const NodeMessage *request, NullMessage *reply) {
//    if(!WorkerStore::neis_in_neededby_otherwk.count(request->wid())){
//        unordered_set<int> set_tmp;
//        WorkerStore::neis_in_neededby_otherwk.insert(make_pair(request->wid(),set_tmp));
//    }


    unique_lock<mutex> lck(ThreadUtil::mtx_sendNodes2Wk);
    ThreadUtil::count_sendNodes2Wk++;
    if (ThreadUtil::count_sendNodes2Wk == WorkerStore::worker_num - 1) {
        ThreadUtil::count_sendNodes2Wk = 0;
        ThreadUtil::cv_sendNodes2Wk.notify_all();
    } else {
        ThreadUtil::cv_sendNodes2Wk.wait(lck);
    }
    lck.unlock();


    unique_lock<mutex> lck_merge(ThreadUtil::mtx_sendNodes2Wk_merge);
    if (!WorkerStore::neis_in_neededby_otherwk[request->wid()].empty()) {
        WorkerStore::neis_in_neededby_otherwk[request->wid()].clear();
    }
    for (auto id : request->nodes()) {
        WorkerStore::neis_in_neededby_otherwk[request->wid()].insert(id);
    }


    lck_merge.unlock();


    unique_lock<mutex> lck_2(ThreadUtil::mtx_sendNodes2Wk_2);
    ThreadUtil::count_sendNodes2Wk_2++;
    if (ThreadUtil::count_sendNodes2Wk_2 == WorkerStore::worker_num - 1) {
        ThreadUtil::count_sendNodes2Wk_2 = 0;
        ThreadUtil::cv_sendNodes2Wk_2.notify_all();
    } else {
        ThreadUtil::cv_sendNodes2Wk_2.wait(lck_2);
    }


    return Status::OK;

}


Status ServiceImpl::pushEmbs(ServerContext *context, const EmbGradMessage *request, NullMessage *reply) {
    int layer_id = request->layerid();
    const string &status = request->status();
    const string &graph_mode = request->graph_mode();
    Graph *graph;
    if (graph_mode == "full") {
        graph = &WorkerStore::graph;
    } else {
        graph = &WorkerStore::graph_sampled;
    }
    auto &subgraph = graph->subgraphs[status];


    auto* ptr_result=WorkerStore::emb_reply_4push_ptr;


    // this is used to build the neighboring embeddings for layer_id
    auto old2new_map = subgraph.graphlayers[layer_id].o2n_map;
    int emb_dim = request->featsize();

    unique_lock<mutex> lck(ThreadUtil::mtx_pushembs);

    for (int i = 0; i < request->nodes_size(); i++) {
        int id = request->nodes(i);
        int new_id = old2new_map[id];

        copy(request->embs().begin() + i * emb_dim, request->embs().begin() + (i + 1) * emb_dim,
             ptr_result + new_id * emb_dim);
    }


    lck.unlock();


    return Status::OK;
}


Status ServiceImpl::setM(ServerContext *context, const MMessageForAD *request, MMessageForAD *reply){
    if(request->wid()==0){
        ServerStore::m=request->m();
        ServerStore::m_benefit=request->m_benefit();
    }

    unique_lock<mutex> lck(ThreadUtil::mtx_barrier);
    ThreadUtil::count_worker_for_barrier++;
    if (ThreadUtil::count_worker_for_barrier == ServerStore::worker_num) {
        ThreadUtil::count_worker_for_barrier = 0;
        ThreadUtil::cv_barrier.notify_all();

    } else {
        ThreadUtil::cv_barrier.wait(lck);
    }

    reply->set_m(ServerStore::m);
    reply->set_m_benefit(ServerStore::m_benefit);
    return Status::OK;


}