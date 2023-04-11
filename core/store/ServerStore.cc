

#include "ServerStore.h"


int ServerStore::feat_dim;
int ServerStore::class_dim;
vector<int> ServerStore::hid_dims;
int ServerStore::worker_num;
int ServerStore::server_num;
int ServerStore::serverId;
float ServerStore::val_accuracy=0;
float ServerStore::train_accuracy=0;
float ServerStore::test_accuracy=0;

// for adam optimizer
map<string,int> ServerStore::t;
map<string,vector<double>> ServerStore::params;
map<string,vector<double>> ServerStore::grads_agg;
map<string,vector<double>> ServerStore::m_grads_t;
map<string,vector<double>> ServerStore::v_grads_t;

// for building layer_compute
map<int,vector<int>> ServerStore::nodes;
map<int, vector<int>> ServerStore::need_nodes;
vector<int> ServerStore::processed_need_nodes;
map<int,set<int>> ServerStore::merge_nodes_bp;

vector<int> ServerStore::agg_nodes;

int ServerStore::m;
int ServerStore::m_benefit;

