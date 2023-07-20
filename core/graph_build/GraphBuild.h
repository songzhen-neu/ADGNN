

#ifndef EC_GRAPH_GRAPHBUILD_H
#define EC_GRAPH_GRAPHBUILD_H

#include "../service/router.h"
#include<time.h>

class GraphBuild {
public:
    static unordered_map<string,string> graphinfo;

    void buildInitGraph();
    tuple<int, int, py::array_t<int>> transEdgeToNewID(const string &status,const string &graph_mode, int lid);
    py::array_t<float> getTrainedFeat(const string& status,const  string& graph_mode);
    py::array_t<float> getTrainedLabel(const string& status,const  string& graph_mode);
    void setGraphMode( string graph_mode);
    void printGraphInfo();
    static void updateGraphLayer(SubGraph& subgraph,int lid);
    static void buildGraphForSample();

    static void checkGraph(Graph &graph);
    static void evalSubGraph(SubGraph &graph,const string& graph_mode);
    void deleteFullGraphInCpp();

};


#endif //EC_GRAPH_GRAPHBUILD_H
