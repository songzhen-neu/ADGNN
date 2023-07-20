

#include "WorkerStore.h"


int WorkerStore::layer_num;
//vector<vector<int>> WorkerStore::nodesForEachWorker;
int WorkerStore::worker_id;
int WorkerStore::worker_num;
string WorkerStore::serverip;
map<string, float *> WorkerStore::embs_ptr_map;
//map<string, map<int, map<int, vector<int>>>> WorkerStore::fsthop_for_worker;
//unordered_map<string, unordered_map<int, unordered_map<int, int>>> WorkerStore::old2new_map;
//unordered_map<string, unordered_map<int, unordered_map<int, int>>> WorkerStore::new2old_map;
//map<string, map<int, int>> WorkerStore::local_vertex_num;
//map<string, map<int, int>> WorkerStore::local_train_vertex_num;
//unordered_map<int, int> WorkerStore::v2wk_map;
map<int, set<int>> WorkerStore::neis_neededby_otherwk;
map<int, vector<vector<float>>> WorkerStore::local_emb_grad_agg;
//map<string, map<int, int>> WorkerStore::rmt_nei_num;
unordered_map<int, unordered_set<int>> WorkerStore::neis_in_neededby_otherwk;
map<string, map<int, map<int, vector<int>>>> WorkerStore::push_2_worker_nodes;
float *WorkerStore::emb_reply_4push_ptr;
map<string, map<int, map<int, vector<int>>>> WorkerStore::v2wk_push;
int WorkerStore::num_threads;
//unordered_map<int, int> WorkerStore::degree_map;
//unordered_map<int, unordered_set<int>> WorkerStore::rmt_nei_set;
//unordered_map<int, unordered_set<int>> WorkerStore::loc_nei_set;
string WorkerStore::graph_mode;

Graph WorkerStore::graph;
Graph WorkerStore::graph_sampled;

// FOS
unordered_map<int,vector<int>> WorkerStore::start_indices;
unordered_map<int,int> WorkerStore::range_size;

unordered_map<string,unordered_map<int,unordered_set<int>>> WorkerStore::loc_rmt_adj;

 double WorkerStore::sample_time=0;
double WorkerStore::construction_time=0;





int WorkerStore::vertex_num;
int WorkerStore::train_num;
int WorkerStore::val_num;
int WorkerStore::test_num;
int WorkerStore::feat_size;
int WorkerStore::sample_num;
string WorkerStore::filename;
string WorkerStore::sample_method;
string WorkerStore::partition_method;

void WorkerStore::set_nodes(const NodeMessage *nodeMessage) {
    // 将nodeMessage解析成nodes
    for (auto i:nodeMessage->nodes()) {
        WorkerStore::graph.nodes.push_back(i);
    }
    cout << "server:node number:" << WorkerStore::graph.nodes.size() << endl;
}

//void WorkerStore::set_nodes_for_each_worker(const DataMessage *reply) {
//    auto &nodes_tmp = reply->nodesforeachworker();
//    int workerNum = nodes_tmp.size();
//    for (int i = 0; i < workerNum; i++) {
//        auto &nodes_worker_i = nodes_tmp.Get(i);
//        int nodesNum = nodes_worker_i.nodes_size();
//        vector<int> vec_tmp;
//        WorkerStore::nodesForEachWorker.push_back(vec_tmp);
//        auto &nodesWorkerI_ws = WorkerStore::nodesForEachWorker[i];
//        for (int j = 0; j < nodesNum; j++) {
//            nodesWorkerI_ws.push_back(nodes_worker_i.nodes(j));
//        }
//    }
//}

//void WorkerStore::set_degree_map(const DataMessage *reply) {
//
//    for (int i = 0; i < reply->degreemap_size(); i++) {
//        auto key = reply->degreemap().Get(i).key();
//        auto val = reply->degreemap().Get(i).value();
////        WorkerStore::degree_map.insert(make_pair(key, val));
////        cout<<key<<","<<val<<"   ";
//    }
////    cout<<endl;
//    cout << "reply degree_map size:" << reply->degreemap().size() << endl;
//    cout << "workerstore degree_map size:" << WorkerStore::degree_map.size() << endl;
//}


void WorkerStore::set_features(const DataMessage_FeatureMessage *featureMessage) {
    int dim_size = featureMessage->features().begin()->feature_size();
//    cout<<"dim_size:"<<dim_size<<endl;
    for (const auto &featureItem:featureMessage->features()) {
        int vid = featureItem.vid();
        vector<float> feature(dim_size);
//        copy(featureItem.feature().begin(),featureItem.feature().end(),feature);
        for (int i = 0; i < dim_size; i++) {
            feature[i] = featureItem.feature(i);
        }
        WorkerStore::graph.features.insert(pair<int, vector<float>>(vid, feature));
    }
    Check::check_features(WorkerStore::graph.features);

}

void WorkerStore::set_labels(const DataMessage_LabelMessage *labelMessage) {
    for (const auto &labelItem:labelMessage->labels()) {
        int vid = labelItem.vid();
        int label = labelItem.label();
//        cout<<"label::setlabel::"<<label<<endl;
        WorkerStore::graph.labels.insert(pair<int, int>(vid, label));
    }
    Check::check_labels(WorkerStore::graph.labels);
}

void WorkerStore::set_adjs(const DataMessage_AdjMessage *adjMessage) {
    for (const auto &adjItem:adjMessage->adjs()) {
        int vid = adjItem.vid();
        unordered_set<int> neibors;
        for (auto neiborId:adjItem.neibors()) {
            neibors.insert(neiborId);
        }
        WorkerStore::graph.adjs.insert(pair<int, unordered_set<int>>(vid, neibors));
    }
    Check::check_adjs(WorkerStore::graph.adjs);
}


