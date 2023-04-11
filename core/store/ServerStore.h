
#ifndef DGNN_TEST_SERVERSTORE_H
#define DGNN_TEST_SERVERSTORE_H
#include <iostream>
#include <vector>
#include <map>
#include "../util/check.h"
using namespace std;
class ServerStore {
public:
    // 神经网络每层参数
    static int feat_dim;
    static int worker_num;
    static int server_num;
    static vector<int> hid_dims;
    static int class_dim;
    static int serverId;

    // for adam optimizer
    static map<string,int> t;
    static map<string,vector<double>> params;
    static map<string,vector<double>> grads_agg;
    static map<string,vector<double>> m_grads_t;
    static map<string,vector<double>> v_grads_t;

    static float val_accuracy;
    static float train_accuracy;
    static float test_accuracy;

    // for building layer_compute
    static map<int, vector<int>> need_nodes;
    static vector<int> processed_need_nodes;
    static map<int,vector<int>> nodes;
    static map<int,set<int>> merge_nodes_bp;

    static vector<int> agg_nodes;

    static int m;
    static int m_benefit;

};


#endif //DGNN_TEST_SERVERSTORE_H
