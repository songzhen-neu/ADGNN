

#include "GraphBuild.h"

#include <utility>

unordered_map<string,string> GraphBuild::graphinfo;

Graph &getGraph(const string &graph_mode) {
    if (graph_mode == "full") {
        return WorkerStore::graph;
    } else {
        return WorkerStore::graph_sampled;
    }
}


set<int> get_set_difference(set<int> &s1, set<int> &s2) {
    set<int> result;
    for (auto id:s1) {
        if (!s2.count(id)) {
            result.insert(id);
        }
    }
    return result;
}

unordered_set<int> get_set_difference(unordered_set<int> &s1, unordered_set<int> &s2) {
    unordered_set<int> result;
    for (auto id:s1) {
        if (!s2.count(id)) {
            result.insert(id);
        }
    }
    return result;
}


unordered_map<int, unordered_set<int>>
getSplitedNeiSet(unordered_set<int> &target_v_set, unordered_map<int, unordered_set<int>> &adj, GraphLayer &gl) {
    auto &v2wk = WorkerStore::graph.v2wk;
    unordered_map<int, unordered_set<int>> splited_nei_set;
    auto &loc_nei_set = gl.loc_nei_set;
    auto &rmt_nei_set = gl.rmt_nei_set;

    for (int i = 0; i < WorkerStore::worker_num; i++) {
        unordered_set<int> tmp;
        splited_nei_set.insert(make_pair(i, tmp));
    }


    for (auto id : target_v_set) {
        auto &nei_set = adj[id];
        for (auto nid:nei_set) {
            auto wk_id = v2wk[nid];
            splited_nei_set[wk_id].insert(nid);
            if (wk_id == WorkerStore::worker_id) {
                loc_nei_set.insert(nid);
            } else {
                rmt_nei_set.insert(nid);
            }
        }

    }

//
    auto tmp = get_set_difference(splited_nei_set[WorkerStore::worker_id], target_v_set);
    splited_nei_set[WorkerStore::worker_id] = tmp;

//    for(int i=0;i<WorkerStore::worker_num;i++){
//        if(WorkerStore::worker_id!=i){
//            for(auto id:splited_nei_set[i]){
//                cout<<"aaaaaa"<<id<<endl;
//            }
//        }
//    }

    return splited_nei_set;
}

void setVertexNum(int tv_size, unordered_map<int, unordered_set<int>> &nei, int lid, SubGraph &subgraph) {
    subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei.push_back(tv_size);
    subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei.push_back(nei[WorkerStore::worker_id].size());
    int count = 0;
    for (auto &nei_set:nei) {
        if (nei_set.first != WorkerStore::worker_id) {
            count += nei_set.second.size();


        }
    }
    subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei.push_back(count);
}

void GraphBuild::updateGraphLayer(SubGraph &subgraph, int lid) {
//    clock_t start = clock();
    auto &target_v_tmp = subgraph.graphlayers[lid].target_v;
    auto &adj = subgraph.graphlayers[lid].adj;
    auto wk2nei_map = getSplitedNeiSet(target_v_tmp, adj, subgraph.graphlayers[lid]);
//    clock_t end = clock();
//    cout << "!!!!!!!!!!!!!!-1111111111111 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start = clock();
    setVertexNum(target_v_tmp.size(), wk2nei_map, lid, subgraph);
//    end = clock();
//    cout << "!!!!!!!!!!!!!!-22222222222 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start = clock();
    set_union(target_v_tmp.begin(), target_v_tmp.end(), wk2nei_map[WorkerStore::worker_id].begin(),
              wk2nei_map[WorkerStore::worker_id].end(),
              inserter(subgraph.graphlayers[lid - 1].target_v, subgraph.graphlayers[lid - 1].target_v.begin()));
//    end = clock();
//    cout << "!!!!!!!!!!!!!!-33333333333 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start = clock();
    auto& locv_by_otherwk = Router::sendInNodes2WK(lid, wk2nei_map);
//    end = clock();
//    cout << "!!!!!!!!!!!!!!-444444444444 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start = clock();

    for (auto &v_wk:locv_by_otherwk) {
        if (v_wk.first != WorkerStore::worker_id) {
            subgraph.graphlayers[lid - 1].target_v.insert(v_wk.second.begin(), v_wk.second.end());
        }
    }
//    end = clock();
//    cout << "!!!!!!!!!!!!!!-5555555555555 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start = clock();
    subgraph.graphlayers[lid].wk2nei_pull = wk2nei_map;
    subgraph.graphlayers[lid - 1].wk2nei_push = locv_by_otherwk;

//    end = clock();
//    cout << "!!!!!!!!!!!!!!-6666666666666 time:  time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;
}

void
buildTargetV(const string &status, Graph &graph) {
    int layer_num = WorkerStore::layer_num;
    // must be ordered set, otherwise get a wrong logic (leverage order to get local neighbors)
    auto &subgraph = graph.subgraphs[status];
    auto idx_copy = graph.idx[status];
    subgraph.graphlayers[WorkerStore::layer_num].target_v = idx_copy;

    for (int lid = layer_num; lid > 0; lid--) {
        GraphBuild::updateGraphLayer(subgraph, lid);
    }

}


void GraphBuild::checkGraph(Graph &graph) {
    int flag = 0;

    cout << "****************idx-graph*************************" << endl;
    for (auto &idx: graph.idx) {
        cout << idx.first << ": ";
        for (auto id:idx.second) {
            cout << id << ",";
        }
        cout << endl;
    }


    cout << "****************v2wk-graph*************************" << endl;
    for (auto &v_wk: graph.v2wk) {
        cout << v_wk.first << "," << v_wk.second << "    ";
    }
    cout << endl;
    flag = 0;


    cout << "****************nodes-graph*************************" << endl;
    for (auto &id: graph.nodes) {
        cout << id << ",";
    }
    cout << endl;
    flag = 0;


    cout << "****************worker_contain_v-graph*************************" << endl;
    for (auto &wk_v:graph.wk_contain_v) {
        cout << "worker " << flag << ":";
        for (auto v:wk_v) {
            cout << v << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************features-graph*************************" << endl;
    for (auto &id_feat:graph.features) {
        cout << "node " << id_feat.first << ":";
        for (auto e:id_feat.second) {
            cout << e << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;


    cout << "****************adjs-graph*************************" << endl;
    for (auto &id_feat:graph.adjs) {
        cout << "node " << id_feat.first << ":";
        for (auto e:id_feat.second) {
            cout << e << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************labels-graph*************************" << endl;
    for (auto &label:graph.labels) {
        cout << "node " << label.first << ":";
        cout << label.second << ",";
        cout << endl;
        flag++;
    }
    flag = 0;


    cout << "****************target_v-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.target_v) {
            cout << v << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************rmt_nei_set-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.rmt_nei_set) {
            cout << v << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************loc_nei_set-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.loc_nei_set) {
            cout << v << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;


    cout << "****************vnum_tarv_locnei_rmtnei-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.vnum_tarv_locnei_rmtnei) {
            cout << v << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************o2n_map-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.o2n_map) {
            cout << v.first << "," << v.second << "    ";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************adj-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {

        cout << "layer " << flag << endl;
        cout << "gl.adj size:" << gl.adj.size() << endl;
        for (auto &v:gl.adj) {
            cout << v.first << ": ";
            for (auto id : v.second) {
                cout << id << ",";
            }
            cout << endl;
        }
        cout << endl;
        flag++;


    }
    flag = 0;

    cout << "****************n2o_map-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto v:gl.n2o_map) {
            cout << v.first << "," << v.second << "    ";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************wk2nei_pull-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto &wk_v:gl.wk2nei_pull) {
            cout << wk_v.first << ":";
            for (auto id:wk_v.second) {
                cout << id << ",";
            }
            cout << endl;
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************wk2nei_push-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        for (auto &wk_v:gl.wk2nei_push) {
            cout << wk_v.first << ":";
            for (auto id:wk_v.second) {
                cout << id << ",";
            }
            cout << endl;
        }
        cout << endl;
        flag++;
    }
    flag = 0;


    cout << "****************label-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        auto *ptr = (int *) gl.label.request().ptr;
        for (int i = 0; i < gl.label.size(); i++) {

            cout << ptr[i] << ",";
        }
        cout << endl;
        flag++;
    }
    flag = 0;

    cout << "****************feat-layer*************************" << endl;
    for (auto &gl:graph.subgraphs["train"].graphlayers) {
        cout << "layer " << flag << endl;
        auto *ptr = (float *) gl.feat.request().ptr;
        for (int i = 0; i < gl.feat.size(); i++) {
            cout << ptr[i] << ",";
            if (i % WorkerStore::feat_size == WorkerStore::feat_size - 1) {
                cout << endl;
            }
        }
        cout << endl;
        flag++;
    }
    flag = 0;


}


void encodeLocalV(const string &status, Graph &graph) {

    auto &subgraph = graph.subgraphs[status];
    // train_nodes, loc_nei, rmt_nei
    for (int lid = WorkerStore::layer_num; lid > -1; lid--) {
        int flag = 0;
        auto &layer = subgraph.graphlayers[lid];
        auto &n2o_map = layer.n2o_map;
        auto &o2n_map = layer.o2n_map;
        auto &target_v = layer.target_v;
        for (auto id:target_v) {
            n2o_map.insert(make_pair(flag, id));
            o2n_map.insert(make_pair(id, flag));
            flag++;
        }
        auto &wk2nei_lid = layer.wk2nei_pull;
        auto &loc_wk2nei = wk2nei_lid[WorkerStore::worker_id];
        for (auto id:loc_wk2nei) {
            n2o_map.insert(make_pair(flag, id));
            o2n_map.insert(make_pair(id, flag));
            flag++;
        }

        for (auto &rmt_wk2nei:wk2nei_lid) {
            if (rmt_wk2nei.first != WorkerStore::worker_id) {
                for (auto id:rmt_wk2nei.second) {
                    n2o_map.insert(make_pair(flag, id));
                    o2n_map.insert(make_pair(id, flag));
                    flag++;
                }
            }
        }
    }

}

void buildLabel(const string &status, Graph &graph) {
    auto &subgraph = graph.subgraphs[status];
    auto &label = subgraph.graphlayers[WorkerStore::layer_num].label;
//    auto& feat=subgraph.graphlayers[0].feat;
    auto &target_v = subgraph.graphlayers[WorkerStore::layer_num].target_v;
    label = py::array_t<int>(target_v.size());
    auto *label_ptr = (int *) label.request().ptr;
//    feat=py::array_t<float>(target_v.size()*WorkerStore::feat_size);
//    auto* feat_ptr=(float *) feat.request().ptr;
    int flag = 0;
    for (auto id:target_v) {
        label_ptr[flag] = WorkerStore::graph.labels[id];
//        cout<<"id and label:"<<id<<","<<label_ptr[flag]<<endl;
//        auto& feat_v=graph.features[id];
//        for(int i=0;i<WorkerStore::feat_size;i++){
//            feat_ptr[flag*WorkerStore::feat_size+i]=feat_v[i];
//        }
        flag++;
    }


}


tuple<int, int, py::array_t<int>>
GraphBuild::transEdgeToNewID(const string &status, const string &graph_mode, int lid) {
    vector<vector<int>> vec;
    auto &graph = getGraph(graph_mode);
    auto &subgraph = graph.subgraphs[status];
    auto &nodes = subgraph.graphlayers[lid].target_v;
    auto &o2n_map = subgraph.graphlayers[lid].o2n_map;
    auto &adj = subgraph.graphlayers[lid].adj;


    for (auto id:nodes) {
        int from_id = o2n_map[id];
        for (auto nei:adj[id]) {
            int to_id = o2n_map[nei];
            vector<int> vec_2(2);
            vec_2[0] = from_id;
            vec_2[1] = to_id;
            vec.push_back(vec_2);
        }
    }
    int size = vec.size();


    auto result = py::array_t<float>(size * 2);
    result.resize({size, 2});
    py::buffer_info buf_result = result.request();
    auto *ptr_result = (float *) buf_result.ptr;

    for (int i = 0; i < size; i++) {
        ptr_result[i * 2 + 0] = vec[i][0];
        ptr_result[i * 2 + 1] = vec[i][1];
    }
//    cout<<"o2n_map size:"<<o2n_map.size()<<endl;
    auto target_num = subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei[0];
    auto nei_num = subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei[0] +
                   subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei[1] +
                   subgraph.graphlayers[lid].vnum_tarv_locnei_rmtnei[2];

    return tuple<int, int, py::array_t<int>>(target_num, nei_num, result);

}


void buildSubGraph(const string &status, Graph &graph) {
    // graphlayer: o2n_map, n2o_map
    encodeLocalV(status, graph);
    // graphlayer: label
    buildLabel(status, graph);

}


void split_dataset() {
    auto &graph = WorkerStore::graph;
    auto &nodes = graph.nodes;
    int vertex_num = WorkerStore::vertex_num;
    int train_num = WorkerStore::train_num;
    int val_num = WorkerStore::val_num;
    int test_num = WorkerStore::test_num;


    unordered_set<int> train_set;
    unordered_set<int> val_set;
    unordered_set<int> test_set;

    for (int i = 0; i < vertex_num; i++) {
        if (i < train_num) {
            train_set.insert(i);
        } else if (i < val_num + train_num) {
            val_set.insert(i);
        } else if (i < val_num + train_num + test_num) {
            test_set.insert(i);
        }
    }


    auto &idx = graph.idx;
    for (auto id : graph.nodes) {
        if (train_set.count(id)) {
            idx["train"].insert(id);
        } else if (val_set.count(id)) {
            idx["val"].insert(id);
        } else if (test_set.count(id)) {
            idx["test"].insert(id);
        }
    }




}

void getRmtFeat(const string &status) {
    Router::dgnnServerRouter[0]->server_Barrier();
    Router::getRmtFeat(status);
    Router::dgnnServerRouter[0]->server_Barrier();
}

void buildSubGFeat(const string &status, Graph &graph) {
    auto &gl1 = graph.subgraphs[status].graphlayers[1];
    auto &o2n_map = gl1.o2n_map;
    auto feat_size = WorkerStore::feat_size;
    gl1.feat = py::array_t<float>(o2n_map.size() * WorkerStore::feat_size);
    gl1.feat.resize({(int) o2n_map.size(), WorkerStore::feat_size});
    auto &feats = WorkerStore::graph.features;
    auto *feat_ptr = (float *) gl1.feat.request().ptr;
    for (auto &oid_nid:o2n_map) {
        int oid = oid_nid.first;
        int nid = oid_nid.second;
        copy(feats[oid].begin(), feats[oid].end(), feat_ptr + nid * feat_size);
    }
}

void setLayerAdjForFull(const string &status) {
    for (int i = 0; i < WorkerStore::layer_num + 1; i++) {
        WorkerStore::graph.subgraphs[status].graphlayers[i].adj = WorkerStore::graph.adjs;
    }
}

void GraphBuild::evalSubGraph(SubGraph &graph,const string& graph_mode) {
    if(!GraphBuild::graphinfo.count(graph_mode+"_nei_num")){
        GraphBuild::graphinfo.insert(make_pair(graph_mode+"_nei_num",""));
    }
    if(!GraphBuild::graphinfo.count(graph_mode+"_rmtnei_num")){
        GraphBuild::graphinfo.insert(make_pair(graph_mode+"_rmtnei_num",""));
    }

    string& str_nei=GraphBuild::graphinfo[graph_mode+"_nei_num"];
    str_nei+= "[";
    for (int i = WorkerStore::layer_num; i > 0; i--) {
        auto &gl = graph.graphlayers[i];
        str_nei+= to_string(gl.o2n_map.size()) + ",";
    }
    str_nei+= "], " ;

    string& str_rmtnei=GraphBuild::graphinfo[graph_mode+"_rmtnei_num"];

    str_rmtnei+= "[";
    for (int i = WorkerStore::layer_num; i > 0; i--) {
        auto &gl = graph.graphlayers[i];
        if (!gl.vnum_tarv_locnei_rmtnei.empty()) {
            str_rmtnei+=  to_string(gl.vnum_tarv_locnei_rmtnei[2]) + ",";
        }

    }
    str_rmtnei+=  "], ";



}

void buildGraphForFull(const string &status) {
    cout << "**********processing subgraph " << status << "**************" << endl;
    setLayerAdjForFull(status);
    // graphlayer: loc_nei_set, rmt_nei_set, vnum_tarv_locnei_rmtnei, wk2nei_pull, wk2nei_push, target_v
    clock_t clk_start = clock();
    buildTargetV(status, WorkerStore::graph);
    clock_t clk_end = clock();
    cout << "buildTargetV time: " << (double) (clk_end - clk_start) / CLOCKS_PER_SEC << " s" << endl;

    clk_start = clock();
    buildSubGraph(status, WorkerStore::graph);
    clk_end = clock();
    cout << "buildSubGraph time: " << (double) (clk_end - clk_start) / CLOCKS_PER_SEC << " s" << endl;

    clk_start = clock();
    getRmtFeat(status);
    clk_end = clock();
    cout << "getRmtFeat time: " << (double) (clk_end - clk_start) / CLOCKS_PER_SEC << " s" << endl;

    clk_start = clock();
    buildSubGFeat(status, WorkerStore::graph);
    clk_end = clock();
    cout << "buildSubGFeat time: " << (double) (clk_end - clk_start) / CLOCKS_PER_SEC << " s" << endl;


}

void GraphBuild::buildGraphForSample() {
    buildSubGraph("train", WorkerStore::graph_sampled);
    buildSubGFeat("train", WorkerStore::graph_sampled);
    evalSubGraph(WorkerStore::graph_sampled.subgraphs["train"],"sample");
}

void GraphBuild::buildInitGraph() {
    // adj4layers is for sampling mode
//    for (int i = 0; i < WorkerStore::layer_num + 1; i++) {
//        WorkerStore::graph.subgraphs["train"].graphlayers[i].adj = WorkerStore::graph.adjs;
//    }
    split_dataset();
    buildGraphForFull("train");
    buildGraphForFull("val");
    buildGraphForFull("test");
    evalSubGraph(WorkerStore::graph.subgraphs["train"],"full");

//    checkGraph(WorkerStore::graph);
//    Router::dgnnServerRouter[0]->server_Barrier();
}


py::array_t<float> GraphBuild::getTrainedFeat(const string &status, const string &graph_mode) {
    auto &graph = getGraph(graph_mode);
    return graph.subgraphs[status].graphlayers[1].feat;
}

py::array_t<float> GraphBuild::getTrainedLabel(const string &status, const string &graph_mode) {
    auto &graph = getGraph(graph_mode);
    auto &label = graph.subgraphs[status].graphlayers[WorkerStore::layer_num].label;
    return label;
}

void GraphBuild::setGraphMode( string graph_mode) {
    WorkerStore::graph_mode = std::move(graph_mode);
}

void GraphBuild::printGraphInfo(){
    for(auto& id : GraphBuild::graphinfo){
        cout<<"***********************"<<id.first<<"***************"<<endl;
        cout<<id.second<<endl;
    }
}


void GraphBuild::deleteFullGraphInCpp(){
    for(int i=0;i<WorkerStore::layer_num+1;i++){
        WorkerStore::graph.subgraphs["train"].graphlayers[i].adj.clear();
        WorkerStore::graph.subgraphs["val"].graphlayers[i].adj.clear();
//        WorkerStore::graph.subgraphs["test"].graphlayers[i].adj.clear();
    }


}