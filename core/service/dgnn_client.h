

#ifndef DGNN_TEST_DGNN_CLIENT_H
#define DGNN_TEST_DGNN_CLIENT_H




#include <iostream>
#include <grpcpp/grpcpp.h>
//#include "../../cmake/build/dgnn_test.grpc.pb.h"

#include <vector>
#include "../store/WorkerStore.h"
#include <pthread.h>
#include "../structure/Graph.h"
#include "dgnn_server.h"
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../../cmake/build/dgnn_test.pb.h"
#include <time.h>
#include <math.h>
#include "../util/SetVecTrans.h"


using namespace std;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;

using dgnn_test::DgnnProtoService;
using dgnn_test::DataMessage;
using dgnn_test::NodeMessage;
using dgnn_test::ContextMessage;
using dgnn_test::NetInfoMessage;
using dgnn_test::EmbGradMessage;
using dgnn_test::ParamGrad;
using dgnn_test::LayerNodeListMessage;
using dgnn_test::NullMessage;
using dgnn_test::MMessageForAD;


class DGNNClient {
//private:

public:
    std::unique_ptr<DgnnProtoService::Stub> stub_;
    ServiceImpl serverImpl;
    string serverAddress;

    explicit DGNNClient(std::shared_ptr<Channel> channel);

    DGNNClient();

    void init(std::shared_ptr<Channel> channel);

    static void *RunServer(void *address_tmp);

    void startClientServer();

    void freeSpace();

    void freeMaster();

    void init_by_address(std::string address);


    vector<int> get_nodes();

    void set_nodes();

    pair<py::array_t<int>,py::array_t<float>> get_features();

    void set_features();

    unordered_map<int, int> get_labels();

    void set_labels();

//    unordered_map<int,int> get_degree_map();
    void set_degree_map();

    unordered_map<int, unordered_set<int>> get_adjs();

    void set_adjs();

    unordered_map<int, int> get_v2wk();
    void set_v2wk();

    vector<vector<int>> get_wk2v();
    void set_wk2v();

    string get_serverAddress();

    void set_serverAddress(string serverAddress);

    vector<vector<int>> get_nodesForEachWorker();

    void set_nodesForEachWorker();

    int get_layerNum();

    void set_layerNum(int layerNum);


//    void pullDataFromMasterGeneral(int worker_id, int worker_num, int data_num,
//                                   const string &data_path, int feature_dim, int class_num,
//                                   const string &partitionMethod, int edgeNum);

//    static void split_dataset();
    void server_Barrier();


    static void *worker_pull_needed_emb_parallel(void *metaData);

    static void *worker_pull_g_parallel(void *metaData_void);


    map<string, float> sendAccuracy(float val_acc, float train_acc, float acc_test);

    vector<int> aggregateNodes(vector<int> nodes);


    void initParameter(const int &worker_num, const int &server_num, const int &feat_dim,
                       const vector<int> &hid_dims, const int &class_dim, const int &wid,
                       map<string, vector<float>> &weights);

    py::array_t<float> server_PullParams(const string &param_id);

    void server_updateParam(int worker_id, int server_id, float lr, const string &key, py::array_t<double> &grad);

    void setGraphInfoForCpp(const string &status,
                            const map<int, map<int, vector<int>>> &fsthop4wk,
                            const unordered_map<int, unordered_map<int, int>> &old2new,
                            const unordered_map<int, unordered_map<int, int>> &new2old,
                            const unordered_map<int, int> &v2wk_map,
                            const map<int, int> &local_vertex_num,
                            const map<int, int> &local_train_vertex_num,
                            const map<int, int> &rmt_nei_num,
                            const map<int, map<int, vector<int>>> &push_2_worker_nodes,
                            const map<int, map<int, vector<int>>> &v2wk_push
    );

//    tuple<unordered_map<int, unordered_set<int>>, unordered_set<int>> sampleForLayerAd(const py::array_t<float> *agg_embs,
//                                                         const py::array_t<float> *nei_embs,
//                                                         unordered_map<int, int> &o2n_dict,
//                                                         vector<int> &train_vertices,
//                                                         unordered_map<int, unordered_set<int>> &as_v,
//                                                         int k, int lid, int dim_prune_itv, int adcomp_num);
//
//    tuple<unordered_map<int,unordered_set<int>>,unordered_set<int>> randomSample(vector<int> &train_vertices,int k);
//
//    tuple<unordered_map<int,unordered_set<int>>,unordered_set<int>> bnsSample(vector<int> &train_vertices,int k, int layer_id);
//
//    unordered_map<int, unordered_set<int>> recomputeAS(const py::array_t<float> *agg_embs,
//                                   const py::array_t<float> *nei_embs,
//                                   unordered_map<int, int> &o2n_dict,
//                                   vector<int> &train_vertices,
//                                   unordered_map<int, unordered_set<int>> &adj, int k, float alpha);
//
//    unordered_map<int, unordered_set<int>> updateAS(const py::array_t<float> *agg_embs,
//                                const py::array_t<float> *nei_embs,
//                                unordered_map<int, int> &o2n_dict,
//                                vector<int> &train_vertices,
//                                unordered_map<int, unordered_set<int>> &adj,
//                                unordered_map<int, unordered_set<int>> &as_v,
//                                int k, float alpha);
//
//    tuple<unordered_map<int,unordered_set<int>>,unordered_set<int>> fastgcnSample(vector<int> &train_vertices,int k);
//
//    tuple<unordered_map<int,unordered_set<int>>,unordered_set<int>> clustergcnSample(vector<int> &train_vertices,unordered_set<int> &sub_node, int k);

    static void *getRmtFeat(void *metaData_void);

    void setCtxForCpp(map<string, string> &cluster_config);

    static void *sendNodes2Wk(void *metaData);

    static void *sendInNodes2WK(void *metaData);

    static void *pushEmbsParallel(void *metaData);

    void setStaticInfoForCpp(const unordered_map<int,unordered_set<int>> &rmt_nei_set,const unordered_map<int, unordered_set<int>> &loc_nei_set);

//    py::array_t<int> transEdgeToNewID(const vector<int> &nodes,  unordered_map<int,int> &o2n_map,  unordered_map<int,unordered_set<int>> &adj);

    void readDataAndInit();

    pair<int,int> setM(int id, int m, int benefit_m);





};

struct ReqEmbsMetaData {
    vector<int> *nodes{};
    unordered_set<int> *nei_set{};
    unordered_set<int> *nei_set_unorder{};
    int epoch{};
    int layerId{};
    int workerId{};
    int serverId{};
    int localNodeSize{};
    int feat_num{};
    float *ptr_result{};
    unordered_map<int, int> *oldToNewMap{};
    EmbGradMessage *reply{};
    DGNNClient *dgnnClient{};
    EmbGradMessage *embGradMessage{};
    string status{};
    float *emb_ptr{};
    string graph_mode{};
    int thread_id{};
};


#endif //DGNN_TEST_DGNN_CLIENT_H
