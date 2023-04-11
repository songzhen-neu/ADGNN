

#ifndef DGNN_TEST_CHECK_H
#define DGNN_TEST_CHECK_H

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include "../store/ServerStore.h"
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Check {
public:
    static void check_features(unordered_map<int, vector<float>> &features);

    static void check_labels(unordered_map<int, int> &labels);

    static void check_adjs(unordered_map<int, unordered_set<int>> &adjs);

    static void check_partition_pass(
            const int &workerNum, const int &dataNum,
            const string &dataPath, const int &feature_dim,
            const int &class_num);

    static void check_initParameter_ServerStore();
};

#endif //DGNN_TEST_CHECK_H
