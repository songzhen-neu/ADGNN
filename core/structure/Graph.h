

#ifndef EC_GRAPH_GRAPH_H
#define EC_GRAPH_GRAPH_H

#include <iostream>
#include<unordered_map>
#include<unordered_set>
#include "SubGraph.h"

using namespace std;

class Graph {
public:
    unordered_map<string, SubGraph> subgraphs;
    vector<int> nodes; // local nodes
    unordered_map<int, vector<float>> features;
    unordered_map<int, int> labels;
    unordered_map<int, unordered_set<int>> adjs;
    vector<vector<int>> wk_contain_v;
    unordered_map<int,int> v2wk;
    unordered_map<string, unordered_set<int>> idx;
};


#endif //EC_GRAPH_GRAPH_H
