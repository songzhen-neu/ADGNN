

#ifndef DGNN_TEST_ROUTER_H
#define DGNN_TEST_ROUTER_H
//#include <vector>
#include "dgnn_client.h"
#include <map>
#include <string>
#include <vector>
#include <time.h>


using namespace std;

class Router {
public:
    static vector<DGNNClient*> dgnnWorkerRouter;
    static vector<DGNNClient*> dgnnServerRouter;
    static vector<vector<ReqEmbsMetaData>> metadata_pipe;
    static vector<float *> embs_pipe;



    Router();
    void initWorkerRouter(map<int,string> &dgnnWorkerAddress);
    void initServerRouter(map<int, string> &dgnnServerAddress);

    py::array_t<float> getRemoteEmb( int layerId, const string &status,int feat_num, const string& graph_mode);
    py::array_t<float> setAndSendG(int layer_id,const py::array_t<float>& emb_grads);
    py::array_t<long> sendNodes2Wk(int layer_id, map<int,vector<int>> &nei2wk_4lay);

    static void getRmtFeat(const string &status);
    static unordered_map<int, unordered_set<int>>& sendInNodes2WK(int layer_id, unordered_map<int,unordered_set<int>> &to_remote_nei);

    py::array_t<float> pushEmbs(int layer_id, const string& status,const string& graph_mode, py::array_t<float> &embs);
    void pushEmbsByIds(int layer_id,int batch_id, const string& status,const vector<int>& nodes,py::array_t<float> &embs);
    static void pthread_vec_join(const vector<pthread_t> &pthreads);
    static void initReplyEmbs(const string &status,int layer_id, int emb_dim,int batch_num, const string& graph_mode);
//    py::array_t<float> getEntireEmbs(const string &status, int layer_id, int emb_dim, const string& graph_mode);
//    ReqEmbsMetaData* getMetaData(int worker_id);
//    ReqEmbsMetaData* buildStaticMetadata(int worker_id, ReqEmbsMetaData metaData);

//private:
//    static vector<ReqEmbsMetaData> metadata_vec;

};


#endif //DGNN_TEST_ROUTER_H
