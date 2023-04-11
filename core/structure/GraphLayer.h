

#ifndef EC_GRAPH_GRAPHLAYER_H
#define EC_GRAPH_GRAPHLAYER_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"
using namespace std;
namespace py = pybind11;
class GraphLayer {
public:

    // set by @encodeLocalV
    unordered_map<int,int> o2n_map;
    unordered_map<int,int> n2o_map;

    // set by @buildLabel
    py::array_t<int> label;

    py::array_t<float> feat;
    unordered_map<int,unordered_set<int>> adj;

    // set by @buildTargetV
    unordered_set<int> target_v;
    // layer needs to pull wk2nei_pull of the current layer
    // layer needs to push wk2nei_push of the last layer
    unordered_map<int,unordered_set<int>> wk2nei_pull; // not contain train_v
    unordered_map<int,unordered_set<int>> wk2nei_push; // push embs of lid-1 when computing at lid
    vector<int> vnum_tarv_locnei_rmtnei;
    unordered_set<int> loc_nei_set;
    unordered_set<int> rmt_nei_set;

};


#endif //EC_GRAPH_GRAPHLAYER_H
