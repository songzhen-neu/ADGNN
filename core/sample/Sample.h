

#ifndef EC_GRAPH_SAMPLE_H
#define EC_GRAPH_SAMPLE_H

#include <iostream>
#include <vector>
#include "../graph_build/GraphBuild.h"
#include "../store/WorkerStore.h"
#include "../service/router.h"
#include "../util/SetVecTrans.h"
using namespace std;
class Sample {
public:
    static unordered_map<string,py::array_t<float>> aggembs_neiembs;
    static unordered_map<int,float> adimportance_v;
    static unordered_set<int> nei_set_pc;


    void initSampledGraph();
    void randomSample(vector<int>& fanout);
    void randomSampleNoRebuild(vector<int>& fanout);
    void adSample(vector<int>& fanout,vector<int>& dim_itvs,int adcomp_num,bool enable_pc,vector<int>& comm_fo,int nei_prune);
    void bnsSample(vector<int>& fanout);
    void clustergcnSample(unordered_set<int>& nei_set_sampled);
    void fastgcnSample(vector<int>& fanout);

    void buildRmtAndLocAdj();


    void setAggEmb(const string& id,py::array_t<float>& embs);
};


#endif //EC_GRAPH_SAMPLE_H
