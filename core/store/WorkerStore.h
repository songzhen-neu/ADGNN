

#ifndef DGNN_TEST_WORKERSTORE_H
#define DGNN_TEST_WORKERSTORE_H

#include <vector>
#include <map>
#include <set>
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../util/check.h"
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"
#include <unordered_set>
#include <unordered_map>
#include "../structure/Graph.h"

namespace py = pybind11;

// 不要导入<>这种grpc编译的库，不然生不成新的message
using namespace std;
using dgnn_test::NodeMessage;
using dgnn_test::DataMessage;
using dgnn_test::DataMessage_FeatureMessage;
using dgnn_test::DataMessage_LabelMessage;
using dgnn_test::DataMessage_AdjMessage;
using dgnn_test::DataMessage_FeatureMessage_FeatureItem;
using dgnn_test::DataMessage_LabelMessage_LabelItem;
using dgnn_test::DataMessage_AdjMessage_AdjItem;
using dgnn_test::ContextMessage;

// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
// node;feature;label;adj
class WorkerStore {
public:

    static string serverip;

    // 用来记录每轮迭代的神经网络权重
    static int layer_num;
    static int worker_id;
    static int worker_num;
    static int vertex_num;
    static int train_num;
    static int val_num;
    static int test_num;
    static int feat_size;
    static int sample_num;
    static string filename;
    static string sample_method;
    static string partition_method;




    static void set_nodes(const NodeMessage *nodeMessage);
//    static void set_nodes_for_each_worker(const DataMessage *reply);

    static void set_features(const DataMessage_FeatureMessage *featureMessage);

    static void set_labels(const DataMessage_LabelMessage *labelMessage);

    static void set_adjs(const DataMessage_AdjMessage *adjMessage);
//    static void set_degree_map(const DataMessage *reply);


    // {status:[vertex*features]}
    static map<string,float*> embs_ptr_map;
    // {status:{layer:{oldid:newid}}}
//    static unordered_map<string,unordered_map<int,unordered_map<int,int>>> old2new_map;
//    static unordered_map<string,unordered_map<int,unordered_map<int,int>>> new2old_map;
//    static unordered_map<int,int> v2wk_map;
    // {status:{layer:{worker:[vertex]}}}
//    static map<string,map<int, map<int, vector<int>>>> fsthop_for_worker;
//    static map<string,map<int,int>> local_vertex_num;
//    static map<string,map<int,int>> local_train_vertex_num;
    // {layer_id:[node_id:[feat]]}
    static map<int,vector<vector<float>>> local_emb_grad_agg;
    static map<int,set<int>> neis_neededby_otherwk;
//    static map<string,map<int,int>> rmt_nei_num;
    static unordered_map<int,unordered_set<int>> neis_in_neededby_otherwk;
    // {status:{layer_id:{worker_id:push_nodes}}}
    static map<string, map<int,map<int,vector<int>>>> push_2_worker_nodes;
    static float* emb_reply_4push_ptr;
    static map<string,map<int,map<int,vector<int>>>> v2wk_push;
//    static vector<>
    static int num_threads;
    static string graph_mode;
//    static unordered_map<int,int> degree_map;
//    static unordered_map<int,unordered_set<int>> rmt_nei_set;
//    static unordered_map<int,unordered_set<int>> loc_nei_set;

    static Graph graph;
    static Graph graph_sampled;

    static unordered_map<string,unordered_map<int,unordered_set<int>>> loc_rmt_adj;



};



#endif //DGNN_TEST_WORKERSTORE_H
