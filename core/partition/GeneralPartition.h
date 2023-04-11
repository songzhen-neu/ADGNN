

#ifndef DGNN_TEST_GENERALPARTITION_H
#define DGNN_TEST_GENERALPARTITION_H

#include "Partitioner.h"
#include <map>
#include<set>
#include<fstream>
#include<vector>
#include<iostream>
#include<string>
#include<sstream>
#include "../util/string_split.h"
#include <zconf.h>
#include <unordered_set>
#include <unordered_map>

using namespace std;
class GeneralPartition : public Partitioner{
public:
    GeneralPartition();
    void init (const int & data_num, const int & worker_num,const string &filename,const int & feature_size,const int & label_size);
    int startPartition(int worker_num,string partitionMethod,int nodeNum,int edgeNum) override;
    // vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
    static vector<vector<int>> nodes;
    static vector<map<int,vector<float>>>  features;
    static vector<map<int,int>> labels;
    static vector<map<int,unordered_set<int>>> adjs;
    static unordered_map<int,int> degree_map;
};


#endif //DGNN_TEST_GENERALPARTITION_H
