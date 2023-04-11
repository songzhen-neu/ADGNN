

#include "Sample.h"

bool cmp(const pair<int, float> &p1, const pair<int, float> &p2);

unordered_map<string, py::array_t<float>> Sample::aggembs_neiembs;
unordered_map<int, float> Sample::adimportance_v;
unordered_set<int> Sample::nei_set_pc;

void *fastgcnSampleParallel(void *data);

struct RedPirorityData {
    vector<int> *train_vertices;
    unordered_set<int> *target_v;
    int thread_id;
    int lid;
    unordered_map<int, int> *o2n_dict;
    float *agg_embs_ptr;
    float *nei_embs_ptr;
    unordered_map<int, unordered_set<int>> *adj;
    int dim_size;
    unordered_map<int, unordered_set<int>> *as_v;
    int k;
    unordered_map<int, unordered_set<int>> *adj_sampled;
    unordered_set<int> *nei_set;
    unordered_map<int, unordered_set<int>> *anchor_set;
    unordered_set<int> *rmt_nei_set;
    unordered_set<int> *nei_set_sampled;
    float alpha;
    int dim_prune_itv;
    int adcomp_num;
    int nei_prune;

};

float *getAggEmb(int v, unordered_map<int, int> &o2n_dict, float *agg_embs, int dim_size) {
    int id = o2n_dict[v];
    float *agg_v = agg_embs + id * dim_size;
    return agg_v;
}

unordered_map<int, float *>
getNeiEmbs(unordered_set<int> &adj, unordered_map<int, int> &o2n_dict, float *nei_embs, int dim_size) {
    unordered_map<int, float *> nei_return;
    for (auto nid:adj) {
        int new_nid = o2n_dict[nid];
        float *nei = nei_embs + new_nid * dim_size;
        nei_return.insert(pair<int, float *>(nid, nei));
    }
    return nei_return;
}


unordered_map<int, float *>
getAddedNeiAgg(unordered_set<int> &nei_use, vector<float> &agg_nei_use, unordered_map<int, int> &o2n_map, int dim_size,
               float *nei_embs_ptr, unordered_set<int> &adj, int k_itv, int dim_prune_itv, int rand_dim,
               int sample_num, int nei_prune) {
    unordered_map<int, float *> agg_map;
    int adj_size = adj.size();
    int maintain_nei_num = adj_size;
    if (adj_size > nei_prune * sample_num * k_itv) {
        maintain_nei_num = nei_prune * k_itv;
    } else {
        maintain_nei_num = adj_size;
    }

    for (auto nid:adj) {
        auto rd = rand() % adj_size;
        if (rd < maintain_nei_num) {
            if (!nei_use.count(nid)) {
                int nid_new = o2n_map[nid];
                float *emb_nid = nei_embs_ptr + nid_new * dim_size;
                auto *agg_tmp = (float *) malloc(sizeof(float) * dim_size);
                for (int i = 0; i < dim_size; i += dim_prune_itv) {
                    int dim = i + rand_dim;
                    agg_tmp[dim] = agg_nei_use[dim] + emb_nid[dim];
                }
                agg_map.insert(make_pair(nid, agg_tmp));
            }
        }
    }
    return agg_map;


}

unordered_map<int, float>
getAggDiff(const float *agg_v, int dim_size, unordered_map<int, float *> &nei_agg_map, unordered_set<int> &nei_use,
           float diff_cur, int dim_prune_itv, int rand_dim) {
    unordered_map<int, float> agg_diff_map;
    int nei_use_size = nei_use.size() + 1;
    for (auto &elem:nei_agg_map) {
        float sum = 0;
        for (int i = 0; i < dim_size; i += dim_prune_itv) {
            int dim = i + rand_dim;
            float diff = agg_v[dim] - elem.second[dim] / (float) nei_use_size;
            sum += diff * diff;
        }
//        if (sum < diff_cur) {
        agg_diff_map.insert(make_pair(elem.first, sum));
//        }
    }
    return agg_diff_map;
}

float sampleNeiWithKMinAD(int k, unordered_map<int, float> &agg_diff_map, unordered_set<int> &nei_use,
                          vector<float> &agg_nei_use, int dim_size, unordered_map<int, int> &o2n_map,
                          float *nei_embs_ptr, const float *agg_v, int dim_prune_itv, int rand_dim, float diff_last) {
    vector<pair<int, float>> arr(agg_diff_map.size());
    unordered_set<int> nei_use_new;
    float ad = 0;
    int count = 0;
    for (auto &elem: agg_diff_map) {
        arr[count] = make_pair(elem.first, elem.second);
        count++;
    }

    sort(arr.begin(), arr.end(), cmp);
    for (int i = 0; i < k && i < agg_diff_map.size(); i++) {
        nei_use.insert(arr[i].first);
        nei_use_new.insert(arr[i].first);
        auto nid_new = o2n_map[arr[i].first];
        auto *nei_emb = nei_embs_ptr + nid_new * dim_size;
        for (int j = 0; j < dim_size; j += dim_prune_itv) {
            int dim = j + rand_dim;
            agg_nei_use[dim] += nei_emb[dim];
        }
    }


    for (int i = 0; i < dim_size; i += dim_prune_itv) {
        int dim = i + rand_dim;
        float diff = agg_v[dim] - agg_nei_use[dim];
        ad += diff * diff;
    }


    float adimp_v = diff_last - ad;
    auto &v2wk_map = WorkerStore::graph.v2wk;
    unique_lock<mutex> lck(ThreadUtil::mtx_pc);
    for (int id:nei_use_new) {
        if (v2wk_map[id] != WorkerStore::worker_id) {
            if (Sample::adimportance_v.count(id)) {
                Sample::adimportance_v[id] += adimp_v;
            } else {
                Sample::adimportance_v.insert(make_pair(id, adimp_v));
            }
        } else {
            Sample::nei_set_pc.insert(id);
        }

    }
    lck.unlock();


//    if(nei_use.empty()){
//        cout<<"qqqqqq:"<<k<<","<<agg_diff_map.size()<<"nei_use empty"<<endl;
//    }

    return ad;
}


int getKItv(int sample_num, int k, int i) {
    int k_itv;
    if (k < sample_num) {
        k_itv = 1;
    } else {
        int itv = k / sample_num;
        if (i == sample_num - 1) {
            k_itv = k - (sample_num - 1) * itv;
        } else {
            k_itv = itv;
        }
    }
    return k_itv;
}

double time1 = 0, time2 = 0, time3 = 0, time_total;


tuple<float, unordered_set<int>>
redPriorityAS_v(unordered_set<int> &adj, int k, int dim_size,
                int v, unordered_map<int, int> &o2n_dict, float *agg_embs_ptr, float *nei_embs_ptr, int lid,
                int dim_prune_itv, int adcomp_num, int nei_prune) {
    if (0.8 * adj.size() <= k) {
        return make_tuple(0, adj);
    }

    int v_new = o2n_dict[v];
    auto *agg_v = agg_embs_ptr + v_new * dim_size;
    unordered_set<int> nei_use;
    nei_use.insert(v);

    int sample_num = adcomp_num;
    if (k < sample_num) {
        sample_num = k;
    }

    vector<float> agg_nei_use(dim_size);
    float diff = 10000000;

    for (int i = 0; i < sample_num; i++) {
        int rand_dim = i % dim_prune_itv;
        int k_itv = getKItv(sample_num, k, i);

        auto nei_agg_map = getAddedNeiAgg(nei_use, agg_nei_use, o2n_dict, dim_size, nei_embs_ptr, adj, k_itv,
                                          dim_prune_itv, rand_dim, sample_num,nei_prune);

        auto agg_diff_map = getAggDiff(agg_v, dim_size, nei_agg_map, nei_use, diff, dim_prune_itv, rand_dim);
        if (agg_diff_map.empty()) {
            break;
        }

        diff = sampleNeiWithKMinAD(k_itv, agg_diff_map, nei_use, agg_nei_use, dim_size, o2n_dict, nei_embs_ptr, agg_v,
                                   dim_prune_itv, rand_dim, diff);
        for (auto &elem:nei_agg_map) {
            free(elem.second);
        }
        nei_agg_map.clear();

    }


    tuple<float, unordered_set<int>> data = make_tuple(diff, nei_use);
    return data;


//    // compute diff between v and neis
//    unordered_map<int, vector<float>> diff;
//    vector<float> diff_sum(dim_size);
//    unordered_set<int> nei_use;
//    float diff_tmp = 5000000;
//
//
//    for (auto nid:adj) {
//        vector<float> diff_nid(dim_size);
//        int nid_new = o2n_dict[nid];
//        int v_new = o2n_dict[v];
//        float *emb_nid = nei_embs_ptr + nid_new * dim_size;
//        float *agg_v = agg_embs_ptr + v_new * dim_size;
//        for (int i = 0; i < dim_size; i++) {
//            diff_nid[i] = agg_v[i] - emb_nid[i];
//        }
//        diff.insert(pair<int, vector<float>>(nid, diff_nid));
//    }
//
//
//    if (!as_v.empty()) {
//        for (auto nid:as_v) {
//            nei_use.insert(nid);
//            auto diff_nid = diff[nid];
//            for (int i = 0; i < dim_size; i++) {
//                diff_sum[i] += diff_nid[i];
//            }
//        }
//        for (int i = 0; i < dim_size; i++) {
//            diff_sum[i] /= as_v.size();
//        }
//    }
//
//
//    while (nei_use.size() < k) {
//        int v_add = -1;
//        vector<float> diff_new_add;
//        for (auto nid:adj) {
//            if (!nei_use.count(nid)) {
//                vector<float> diff_new(dim_size);
//                auto diff_nid = diff[nid];
//                float square_sum = 0;
//                for (int i = 0; i < dim_size; i++) {
//                    diff_new[i] = (diff_nid[i] + diff_sum[i] * nei_use.size()) / (nei_use.size() + 1);
//                    square_sum += diff_new[i] * diff_new[i];
//                }
////                square_sum = sqrt(square_sum);
//                if (diff_tmp > square_sum) {
//                    diff_tmp = square_sum;
//                    v_add = nid;
//                    diff_new_add = diff_new_add;
//                }
//            }
//        }
//        if (v_add == -1) {
//            break;
//        } else {
//            nei_use.insert(v_add);
//            diff_sum = diff_new_add;
//        }
//    }
//
//    float diff_sum_val = 0;
//    for (int i = 0; i < diff_sum.size(); i++) {
//        diff_sum_val += pow(diff_sum[i], 2);
//    }
//    diff_sum_val = sqrt(diff_sum_val);
//    tuple<float, unordered_set<int>> data = make_tuple(diff_sum_val, nei_use);
//    return data;


}

bool cmp(const pair<int, float> &p1, const pair<int, float> &p2) {
    return p1.second < p2.second;
}

tuple<float, unordered_set<int>>
simPriorityAS_v(unordered_set<int> &adj, float *agg_v, unordered_map<int, float *> &embs_nei, int k,
                unordered_set<int> &as_v, int dim_size) {
    if (adj.size() <= k) {
        return make_tuple(0, adj);
    }
    // compute diff between v and neis
    map<int, vector<float>> diff_vec;
    vector<float> diff_sum(dim_size);
    unordered_set<int> nei_use;
    float diff_tmp = 5000000;
    unordered_map<int, float> diff_val;


    for (auto nid:adj) {
        vector<float> diff_nid(dim_size);
        auto emb_nid = embs_nei[nid];
        float diff_val_tmp = 0;
        for (int i = 0; i < dim_size; i++) {
            diff_nid[i] = agg_v[i] - emb_nid[i];
            diff_val_tmp += pow(diff_nid[i], 2);
        }
        diff_val_tmp = sqrt(diff_val_tmp);
        diff_val.insert(pair<int, float>(nid, diff_val_tmp));
        diff_vec.insert(pair<int, vector<float>>(nid, diff_nid));
    }


    vector<pair<int, float>> arr;
    arr.reserve(diff_val.size());
    for (auto elem:diff_val) {
        arr.emplace_back(make_pair(elem.first, elem.second));
    }
    sort(arr.begin(), arr.end(), cmp);

    for (int i = 0; i < k && i < arr.size(); i++) {
        nei_use.insert(arr[i].first);
    }
    for (auto nid:nei_use) {
        auto diff_nid = diff_vec[nid];
        for (int i = 0; i < dim_size; i++) {
            diff_sum[i] += diff_nid[i];
        }

    }

    float diff_sum_val = 0;
    for (int i = 0; i < diff_sum.size(); i++) {
        diff_sum_val += pow(diff_sum[i] / nei_use.size(), 2);
    }
    diff_sum_val = sqrt(diff_sum_val);
    tuple<float, unordered_set<int>> data = make_tuple(diff_sum_val, nei_use);
    return data;


}


void *redPirorityASParallel(void *data) {
//    clock_t startTime = clock();
    auto data_parse = (RedPirorityData *) data;
    auto target_v = SetVecTrans::set2vec(*data_parse->target_v);
    auto agg_embs_ptr = data_parse->agg_embs_ptr;
    auto nei_embs_ptr = data_parse->nei_embs_ptr;
    auto dim_size = data_parse->dim_size;
    auto &o2n_dict = *data_parse->o2n_dict;
    auto &adj = *data_parse->adj;
    auto thread_id = data_parse->thread_id;
    auto k = data_parse->k;
    auto lid = data_parse->lid;
    auto dim_prune_itv = data_parse->dim_prune_itv;
    auto adcomp_num = data_parse->adcomp_num;
    auto nei_prune=data_parse->nei_prune;


    int itv = int(target_v.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;
    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = target_v.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }
    unordered_map<int, unordered_set<int>> adj_sampled;
    unordered_set<int> nei_set;

    for (int i = start_index; i < end_index; i++) {
        int v = target_v[i];
        auto data_return = redPriorityAS_v(adj[v], k, dim_size, v, o2n_dict, agg_embs_ptr, nei_embs_ptr, lid,
                                           dim_prune_itv, adcomp_num,nei_prune);
        float diff_val;
        unordered_set<int> nei_set_v;
        tie(diff_val, nei_set_v) = data_return;
        nei_set.insert(nei_set_v.begin(), nei_set_v.end());
        adj_sampled.insert(pair<int, unordered_set<int>>(v, nei_set_v));

    }

    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->adj_sampled->insert(adj_sampled.begin(), adj_sampled.end());
    data_parse->nei_set->insert(nei_set.begin(), nei_set.end());
    lck.unlock();
//    endTime = clock();
//    cout << "The add adj,set main time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
//
//    cout << "**************************" << endl;
//    cout << "The use1 time is: " << time_use1 / 1000000 << "s" << endl;
//    cout << "The use2 time is: " << time_use2 / 1000000 << "s" << endl;
//    cout << "The use3 time is: " << time_use3 / 1000000 << "s" << endl;
//    cout << "The total time is: " << time_total / 1000000 << "s" << endl;
//    cout << "The inner1 time is: " << time1 / 1000000 << "s" << endl;
//    cout << "The inner2 time is: " << time2 / 1000000 << "s" << endl;
//    cout << "The inner3 time is: " << time3 / 1000000 << "s" << endl;
//    cout << "**************************" << endl;


}


unordered_set<int>
recomputeAS_v(int v, unordered_map<int, unordered_set<int>> &adj, float *agg_v, unordered_map<int, float *> &embs_nei,
              unordered_set<int> &as_v, int k, int dim_size, float alpha, unordered_map<int, int> &o2n_dict,
              float *agg_embs_ptr, float *nei_embs_ptr, int lid, int dim_prune_itv, int adcomp_num, int nei_prune) {
    auto data_red = redPriorityAS_v(adj[v], k, dim_size, v, o2n_dict, agg_embs_ptr, nei_embs_ptr, lid,
                                    dim_prune_itv, adcomp_num,nei_prune);
    float diff_red;
    unordered_set<int> as_red;
    tie(diff_red, as_red) = data_red;
    auto data_sim = simPriorityAS_v(adj[v], agg_v, embs_nei, k, as_v, dim_size);
    float diff_sim;
    unordered_set<int> as_sim;
    tie(diff_sim, as_sim) = data_sim;
    if (diff_red < alpha * diff_sim) {
        return as_red;
    } else {
        return as_sim;
    }
}

void *recomputeASParallel(void *data) {
    auto data_parse = (RedPirorityData *) data;
    auto &train_vertices = *data_parse->train_vertices;
    auto agg_embs_ptr = data_parse->agg_embs_ptr;
    auto nei_embs_ptr = data_parse->nei_embs_ptr;
    auto dim_size = data_parse->dim_size;
    auto &o2n_dict = *data_parse->o2n_dict;
    auto &adj = *data_parse->adj;
    auto thread_id = data_parse->thread_id;
    auto k = data_parse->k;
    auto alpha = data_parse->alpha;
    auto lid = data_parse->lid;
    auto dim_prune_itv = data_parse->dim_prune_itv;
    auto adcomp_num = data_parse->adcomp_num;
    auto nei_prune=data_parse->nei_prune;

    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;
    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }
    unordered_map<int, unordered_set<int>> anchor_set;
    unordered_set<int> as_v;
    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto agg_v = getAggEmb(v, o2n_dict, agg_embs_ptr, dim_size);
        auto embs_nei = getNeiEmbs(adj[v], o2n_dict, nei_embs_ptr, dim_size);
        auto anchor_set_v = recomputeAS_v(v, adj, agg_v, embs_nei, as_v, k, dim_size, alpha, o2n_dict, agg_embs_ptr,
                                          nei_embs_ptr, lid, dim_prune_itv, adcomp_num,nei_prune);
        anchor_set.insert(pair<int, unordered_set<int>>(v, anchor_set_v));
    }

    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->anchor_set->insert(anchor_set.begin(), anchor_set.end());
    lck.unlock();

}


unordered_map<int, unordered_set<int>> recomputeAS(const py::array_t<float> *agg_embs,
                                                   const py::array_t<float> *nei_embs,
                                                   unordered_map<int, int> &o2n_dict,
                                                   vector<int> &train_vertices,
                                                   unordered_map<int, unordered_set<int>> &adj, int k,
                                                   float alpha) {
    unordered_map<int, unordered_set<int>> anchor_set;
    unordered_set<int> nei_set;

    auto agg_embs_ptr = (float *) agg_embs->request().ptr;
    auto nei_embs_ptr = (float *) nei_embs->request().ptr;
    auto dim_size = agg_embs->shape()[1];
    vector<pthread_t> pthreads(WorkerStore::num_threads);

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->train_vertices = &train_vertices;
        data->agg_embs_ptr = agg_embs_ptr;
        data->nei_embs_ptr = nei_embs_ptr;
        data->dim_size = dim_size;
        data->o2n_dict = &o2n_dict;
        data->adj = &adj;
        data->thread_id = i;
        data->k = k;
        data->anchor_set = &anchor_set;
        data->nei_set = &nei_set;
        data->alpha = alpha;
        pthread_create(&pthreads[i], NULL, recomputeASParallel, (void *) data);
        pthread_join(pthreads[i], NULL);
    }

    return anchor_set;
}

float norm2(float *data, int dim_size) {
    float l2 = 0;
    for (int i = 0; i < dim_size; i++) {
        l2 += pow(data[i], 2);
    }
    l2 = sqrt(l2);
    return l2;
}

float *getDiffPtr(float *agg_v, unordered_map<int, float *> &embs_nei, unordered_set<int> &as_v, int dim_size) {
    auto sum = (float *) malloc(sizeof(float) * dim_size);
    for (auto nid:as_v) {
        auto emb_nid = embs_nei[nid];
        for (int i = 0; i < dim_size; i++) {
            sum[i] += emb_nid[i];
        }
    }

    int as_size = as_v.size();

    for (int i = 0; i < dim_size; i++) {
        sum[i] = agg_v[i] - sum[i] / (float) as_size;
    }

    return sum;
}


void *updateASParallel(void *data) {
    auto data_parse = (RedPirorityData *) data;
    auto &train_vertices = *data_parse->train_vertices;
    auto agg_embs_ptr = data_parse->agg_embs_ptr;
    auto nei_embs_ptr = data_parse->nei_embs_ptr;
    auto dim_size = data_parse->dim_size;
    auto &o2n_dict = *data_parse->o2n_dict;
    auto &adj = *data_parse->adj;
    auto thread_id = data_parse->thread_id;
    auto k = data_parse->k;
    auto alpha = data_parse->alpha;
    auto &as_v = *data_parse->as_v;
    auto lid = data_parse->lid;
    auto dim_prune_itv = data_parse->dim_prune_itv;
    auto adcomp_num = data_parse->adcomp_num;
    auto nei_prune=data_parse->nei_prune;

    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;
    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }
    unordered_map<int, unordered_set<int>> anchor_set;

    unordered_set<int> as_v_empty;
    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto agg_v = getAggEmb(v, o2n_dict, agg_embs_ptr, dim_size);
        auto embs_nei = getNeiEmbs(adj[v], o2n_dict, nei_embs_ptr, dim_size);

        auto diff_nei_ptr = getDiffPtr(agg_v, embs_nei, as_v[v], dim_size);
        auto diff_nei_val = norm2(diff_nei_ptr, dim_size);
        auto diff_agg_val = norm2(agg_v, dim_size);
        if (diff_nei_val > alpha * diff_agg_val) {
            auto anchor_set_v = recomputeAS_v(v, adj, agg_v, embs_nei, as_v_empty, k, dim_size, alpha, o2n_dict,
                                              agg_embs_ptr, nei_embs_ptr, lid, dim_prune_itv, adcomp_num,nei_prune);
            anchor_set.insert(pair<int, unordered_set<int>>(v, anchor_set_v));
        } else {
            anchor_set.insert(pair<int, unordered_set<int>>(v, as_v[v]));
        }


    }


    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->anchor_set->insert(anchor_set.begin(), anchor_set.end());
    lck.unlock();

}


unordered_map<int, unordered_set<int>> updateAS(const py::array_t<float> *agg_embs,
                                                const py::array_t<float> *nei_embs,
                                                unordered_map<int, int> &o2n_dict,
                                                vector<int> &train_vertices,
                                                unordered_map<int, unordered_set<int>> &adj,
                                                unordered_map<int, unordered_set<int>> &as_v,
                                                int k, float alpha) {
    unordered_map<int, unordered_set<int>> anchor_set;
    unordered_set<int> nei_set;

    auto agg_embs_ptr = (float *) agg_embs->request().ptr;
    auto nei_embs_ptr = (float *) nei_embs->request().ptr;
    auto dim_size = agg_embs->shape()[1];
    vector<pthread_t> pthreads(WorkerStore::num_threads);

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->train_vertices = &train_vertices;
        data->agg_embs_ptr = agg_embs_ptr;
        data->nei_embs_ptr = nei_embs_ptr;
        data->dim_size = dim_size;
        data->o2n_dict = &o2n_dict;
        data->adj = &adj;
        data->thread_id = i;
        data->k = k;
        data->as_v = &as_v;
        data->anchor_set = &anchor_set;
        data->nei_set = &nei_set;
        data->alpha = alpha;
        pthread_create(&pthreads[i], NULL, updateASParallel, (void *) data);
        pthread_join(pthreads[i], NULL);
    }

    return anchor_set;
}


void *randomSampleParallel(void *data) {
    auto *data_parse = (RedPirorityData *) data;
    auto &adj = *data_parse->adj;
    auto &train_vertices = *data_parse->train_vertices;
    auto thread_id = data_parse->thread_id;
    auto k = data_parse->k;

    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;
    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }

    unordered_map<int, unordered_set<int>> adj_sampled;
//    unordered_set<int> nei_set;
//    srand(1);
    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto &adj_v = adj[v];
//        vector<int> adj_shuffle(adj_v.size());
        unordered_set<int> nei_set_v;


        // add self-loop
//        nei_set_v.insert(v);

//        if (0.8 * adj_v.size() <= k) {
//            adj_sampled.insert(pair<int, unordered_set<int>>(v, adj_v));
//            continue;
//        }

        if (adj_v.size() <= k) {
            adj_sampled.insert(pair<int, unordered_set<int>>(v, adj_v));
            continue;
        }

        auto adj_shuffle=SetVecTrans::set2vec(adj_v);

        // 1
//        unordered_set<int> index_set;
//        while(index_set.size()<k){
//            index_set.insert(rand()%adj_v.size());
//        }
//        for(auto id:index_set){
//            nei_set_v.insert(adj_shuffle[id]);
//        }

        // 2 may < k
//        for(int tmp=0;tmp<k;tmp++){
//            nei_set_v.insert(adj_shuffle[rand()%adj_v.size()]);
//        }

        // 3
        for (int j = 0; j < adj_shuffle.size(); j++) {
            swap(adj_shuffle[j], adj_shuffle[rand() % adj_shuffle.size()]);
        }

        for (int j = 0; j < k; j++) {
            nei_set_v.insert(adj_shuffle[j]);
        }

//        nei_set_v.insert(adj_shuffle[0]);


//        nei_set.insert(nei_set_v.begin(), nei_set_v.end());
        adj_sampled.insert(pair<int, unordered_set<int>>(v, nei_set_v));


    }

    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->adj_sampled->insert(adj_sampled.begin(), adj_sampled.end());
    lck.unlock();

}


void printadj(const string &info, unordered_map<int, unordered_set<int>> &adj) {
    cout << info << endl;
    for (auto &id_nei:adj) {
        cout << id_nei.first << ":";
        for (auto nei:id_nei.second) {
            cout << nei << ",";
        }
        cout << endl;
    }
}

void
randomSampleLayer(int k, int layer_id) {
//    clock_t start = clock();
    auto &adj = WorkerStore::graph.adjs;
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    unordered_map<int, unordered_set<int>> adj_sampled;
//    unordered_set<int> nei_set;
    auto target_v = SetVecTrans::set2vec(WorkerStore::graph_sampled.subgraphs["train"].graphlayers[layer_id].target_v);


    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->adj = &adj;
        data->train_vertices = &target_v;
        data->k = k;
        data->thread_id = i;
        data->adj_sampled = &adj_sampled;
//        data->nei_set = &nei_set;
        pthread_create(&pthreads[i], NULL, randomSampleParallel, (void *) data);

    }

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

//    clock_t end = clock();
//    cout << "!!!!!!!!!!!!!!!-1111111111111 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    start=clock();
    WorkerStore::graph_sampled.subgraphs["train"].graphlayers[layer_id].adj = adj_sampled;
//     end = clock();
//    cout << "!!!!!!!!!!!!!!!-222222222222222 time: " << (double) (end - start) / CLOCKS_PER_SEC << " s" << endl;

//    printadj("random in randomSampleLayer layerid "+to_string(layer_id),WorkerStore::graph_sampled.subgraphs["train"].graphlayers[layer_id].adj);




}

void clearSampleGraph() {
    for (int i = 0; i < WorkerStore::layer_num + 1; i++) {
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].target_v.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].adj.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].o2n_map.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].n2o_map.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].wk2nei_pull.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].wk2nei_push.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].loc_nei_set.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].rmt_nei_set.clear();
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[i].vnum_tarv_locnei_rmtnei.clear();
    }
}

void Sample::randomSample(vector<int> &fanout) {
    clearSampleGraph();
    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    auto target_v_tmp = WorkerStore::graph.idx["train"];
    subgraph_sampled.graphlayers[layer_num].target_v = target_v_tmp;
    double tu_sample=0;
    double tu_update=0;
    for (int i = layer_num; i > 0; i--) {
        int k = fanout[layer_num - i];
        clock_t start=clock();
        randomSampleLayer(k, i);
        clock_t end = clock();
        tu_sample+=(double)(end-start)/CLOCKS_PER_SEC;
         start=clock();
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
        end = clock();
        tu_update=(double)(end-start)/CLOCKS_PER_SEC;

    }

    clock_t start=clock();
    GraphBuild::buildGraphForSample();
    clock_t end = clock();
//    cout << "################-1111111111111 time: " << tu_sample<<","<< tu_update<<","<<(double)(end-start)/CLOCKS_PER_SEC<<" s" << endl;
//    GraphBuild::checkGraph(WorkerStore::graph_sampled);
}

void Sample::randomSampleNoRebuild(vector<int> &fanout) {
//    clearSampleGraph();
    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    auto target_v_tmp = WorkerStore::graph.idx["train"];
    subgraph_sampled.graphlayers[layer_num].target_v = target_v_tmp;

    for (int i = layer_num; i > 0; i--) {
        int k = fanout[layer_num - i];
        randomSampleLayer(k, i);
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
    }
//    GraphBuild::buildGraphForSample();
//    GraphBuild::checkGraph(WorkerStore::graph_sampled);
}


void Sample::initSampledGraph() {
    // value passing
    WorkerStore::graph_sampled.subgraphs["train"] = WorkerStore::graph.subgraphs["train"];
    Router::dgnnServerRouter[0]->server_Barrier();
}





unordered_map<int, unordered_set<int>>
getAdjComm(int comm_fo, unordered_set<int> &target_v, unordered_map<int, unordered_set<int>> &adj) {
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    unordered_map<int, unordered_set<int>> adj_sampled;
    auto &adimp_map = Sample::adimportance_v;
    vector<pair<int, float>> arr(adimp_map.size());
    auto &comm_set = Sample::nei_set_pc;

    int count = 0;

    for (auto &elem: adimp_map) {
        arr[count] = make_pair(elem.first, elem.second);
        count++;
    }
    sort(arr.begin(), arr.end(), cmp);

    for (int i = 0; i < comm_fo && i < arr.size(); i++) {
        comm_set.insert(arr[i].first);
    }

    auto target_vec_v = SetVecTrans::set2vec(target_v);

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->adj = &adj;
        data->train_vertices = &target_vec_v;
        data->thread_id = i;
        data->adj_sampled = &adj_sampled;
        data->nei_set_sampled = &comm_set;
        pthread_create(&pthreads[i], NULL, fastgcnSampleParallel, (void *) data);

    }

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }
    return adj_sampled;
}

void
adSampleForLayer(int k, int lid, int dim_itv, int adcomp_num, int comm_fo, bool enable_pc,int nei_prune) {
    unordered_map<int, unordered_set<int>> adj_sampled;
    unordered_set<int> nei_set;
//
    auto &adj = WorkerStore::graph.adjs;
    auto *agg_embs_ptr = (float *) Sample::aggembs_neiembs["agg_embs" + to_string(lid)].request().ptr;
    auto *nei_embs_ptr = (float *) Sample::aggembs_neiembs["nei_embs" + to_string(lid)].request().ptr;
    auto dim_size = Sample::aggembs_neiembs["agg_embs" + to_string(lid)].shape()[1];
    auto &target_v = WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid].target_v;
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    auto &o2n_map_full = WorkerStore::graph.subgraphs["train"].graphlayers[lid].o2n_map;


    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->target_v = &target_v;
        data->agg_embs_ptr = agg_embs_ptr;
        data->nei_embs_ptr = nei_embs_ptr;
        data->dim_size = dim_size;
        data->o2n_dict = &o2n_map_full;
        data->adj = &adj;
        data->thread_id = i;
        data->k = k;
        data->adj_sampled = &adj_sampled;
        data->nei_set = &nei_set;
        data->lid = lid;
        data->dim_prune_itv = dim_itv;
        data->adcomp_num = adcomp_num;
        data->nei_prune=nei_prune;
        pthread_create(&pthreads[i], NULL, redPirorityASParallel, (void *) data);
    }

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    if (enable_pc) {
        auto adj_comm = getAdjComm(comm_fo, target_v, adj_sampled);
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid].adj = adj_comm;
    } else {
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid].adj = adj_sampled;
    }


}

void
Sample::adSample(vector<int> &fanout, vector<int> &dim_itvs, int adcomp_num, bool enable_pc, vector<int> &comm_fo,int nei_prune) {
    clearSampleGraph();
    Sample::adimportance_v.clear();
    Sample::nei_set_pc.clear();


    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    auto target_v_tmp = WorkerStore::graph.idx["train"];

    subgraph_sampled.graphlayers[layer_num].target_v = target_v_tmp;
//    cout<<"1111111111111:"<<subgraph_sampled.graphlayers[layer_num].target_v.size()<<endl;

    for (int i = layer_num; i > 0; i--) {
        int k = fanout[layer_num - i];
        int dim_itv = dim_itvs[layer_num - i];
        int comm_fo_lid = comm_fo[layer_num - i];
        adSampleForLayer(k, i, dim_itv, adcomp_num, comm_fo_lid, enable_pc,nei_prune);
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
    }
    GraphBuild::buildGraphForSample();
}

void Sample::setAggEmb(const string &id, py::array_t<float> &embs) {
    if (Sample::aggembs_neiembs.count(id)) {
        Sample::aggembs_neiembs[id] = embs;
    } else {
        Sample::aggembs_neiembs.insert(make_pair(id, embs));
    }

}


void *bnsSampleParallel(void *data) {

    auto *data_parse = (RedPirorityData *) data;
    auto &adj = *data_parse->adj;
    auto &train_vertices = *data_parse->train_vertices;
    auto thread_id = data_parse->thread_id;
    auto k = data_parse->k;
    auto &rmt_nei_set = *data_parse->rmt_nei_set;
    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    auto adj_sampled=WorkerStore::loc_rmt_adj["loc"];
    auto &rmt_adj=WorkerStore::loc_rmt_adj["rmt"];
    int end_index;

    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }

    srand(1);

    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto &adj_v = rmt_adj[v];
        unordered_set<int> nei_set_v;
        auto& adj_sampled_v=adj_sampled[v];

        for (auto nei:adj_v) {
            if (rmt_nei_set.count(nei)) {
                adj_sampled_v.insert(nei);
            }
        }

//        adj_sampled.insert(pair<int, unordered_set<int>>(v, nei_set_v));

    }


    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->adj_sampled->insert(adj_sampled.begin(), adj_sampled.end());
    lck.unlock();

}


void
bnsSampleLayer(int k, int layer_id) {
    auto &adj = WorkerStore::graph.adjs;
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    unordered_map<int, unordered_set<int>> adj_sampled;
    unordered_set<int> nei_set;
    auto &rmt_nei_set = WorkerStore::graph.subgraphs["train"].graphlayers[layer_id].rmt_nei_set;
    auto &loc_nei_set = WorkerStore::graph.subgraphs["train"].graphlayers[layer_id].loc_nei_set;
    auto &v2wk = WorkerStore::graph.v2wk;
    auto wid = WorkerStore::worker_id;
    unordered_set<int> rmt_nei_set_sampled;
    auto target_v = SetVecTrans::set2vec(WorkerStore::graph.subgraphs["train"].graphlayers[layer_id].target_v);

    srand(1);

    clock_t start_time, end_time;


    if (rmt_nei_set.size() <= k) {
        for (auto id : target_v) {
            adj_sampled.insert(make_pair(id, adj[id]));
        }
        WorkerStore::graph_sampled.subgraphs["train"].graphlayers[layer_id].adj = adj_sampled;
        return;
    } else {
        vector<int> adj_shuffle(rmt_nei_set.size());
        int index = 0;
        for (auto nid:rmt_nei_set) {
            adj_shuffle[index] = nid;
            index++;
        }
        for (int j = 0; j < adj_shuffle.size(); j++) {
            swap(adj_shuffle[j], adj_shuffle[rand() % adj_shuffle.size()]);
        }

        for (int j = 0; j < k; j++) {
            rmt_nei_set_sampled.insert(adj_shuffle[j]);
        }
    }


//    rmt_nei_set_sampled.insert(loc_nei_set.begin(), loc_nei_set.end());



    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->adj = &adj;
        data->train_vertices = &target_v;
        data->k = k;
        data->thread_id = i;
        data->adj_sampled = &adj_sampled;
        data->nei_set = &nei_set;
        data->rmt_nei_set = &rmt_nei_set_sampled;
        pthread_create(&pthreads[i], NULL, bnsSampleParallel, (void *) data);

    }


    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }


    WorkerStore::graph_sampled.subgraphs["train"].graphlayers[layer_id].adj = adj_sampled;
    return;

}



void Sample::buildRmtAndLocAdj(){
    auto &loc_rmt_adj=WorkerStore::loc_rmt_adj;
    auto &adj=WorkerStore::graph.adjs;
    auto &v2wk=WorkerStore::graph.v2wk;
    unordered_map<int,unordered_set<int>> loc_adj;
    unordered_map<int,unordered_set<int>> rmt_adj;

    for(auto& nei_set:adj){
        auto target_id=nei_set.first;
        unordered_set<int> rmt_adj_v;
        unordered_set<int> loc_adj_v;
        for(auto nid:nei_set.second){
            auto wid=v2wk[nid];
            if(wid != WorkerStore::worker_id){
                rmt_adj_v.insert(nid);
            }else{
                loc_adj_v.insert(nid);
            }
        }
        rmt_adj.insert(make_pair(target_id,rmt_adj_v));
        loc_adj.insert(make_pair(target_id,loc_adj_v));
    }

    loc_rmt_adj.insert(make_pair("loc",loc_adj));
    loc_rmt_adj.insert(make_pair("rmt",rmt_adj));

    cout<<"rmt and loc adjs build success"<<endl;
    cout<<"adj lengths comparison adj,loc,rmt: "<<WorkerStore::graph.adjs.size()<<","<<WorkerStore::loc_rmt_adj["loc"].size()<<","<<WorkerStore::loc_rmt_adj["rmt"].size()<<endl;
}

void Sample::bnsSample(vector<int> &fanout) {
    clearSampleGraph();
    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    subgraph_sampled.graphlayers[layer_num].target_v = WorkerStore::graph.idx["train"];

    for (int i = layer_num; i > 0; i--) {
        int k = fanout[layer_num - i];
        bnsSampleLayer(k, i);
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
    }
    GraphBuild::buildGraphForSample();
}


void *clustergcnSampleParallel(void *data) {
    auto *data_parse = (RedPirorityData *) data;
    auto &adj = *data_parse->adj;
    auto &train_vertices = *data_parse->train_vertices;
    auto thread_id = data_parse->thread_id;
    auto &nei_set_sampled = *data_parse->nei_set_sampled;

    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;

    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }

    unordered_map<int, unordered_set<int>> adj_sampled;


    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto &adj_v = adj[v];
        unordered_set<int> nei_set_v;

        for (auto nei:adj_v) {
            if (nei_set_sampled.count(nei)) {
                nei_set_v.insert(nei);
            }
        }

        adj_sampled.insert(pair<int, unordered_set<int>>(v, nei_set_v));

    }

    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->adj_sampled->insert(adj_sampled.begin(), adj_sampled.end());
    lck.unlock();


}

void
clustergcnSampleLayer(int lid, unordered_set<int> &sub_node) {
    auto &adj = WorkerStore::graph.adjs;
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    unordered_map<int, unordered_set<int>> adj_sampled;
    auto &v2wk = WorkerStore::graph.v2wk;
    auto wid = WorkerStore::worker_id;
    auto target_v = SetVecTrans::set2vec(WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid].target_v);


    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->adj = &adj;
        data->train_vertices = &target_v;
        data->thread_id = i;
        data->adj_sampled = &adj_sampled;
        data->nei_set_sampled = &sub_node;
        pthread_create(&pthreads[i], NULL, clustergcnSampleParallel, (void *) data);

    }

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid].adj = adj_sampled;


}


void Sample::clustergcnSample(unordered_set<int> &nei_set_sampled) {
    clearSampleGraph();
    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    auto target_v_tmp = WorkerStore::graph.idx["train"];
    subgraph_sampled.graphlayers[layer_num].target_v = target_v_tmp;
//    if(nei_set_sampled.count(0)){
//        cout<<"00000000000000000"<<endl;
//    }

    for (int i = layer_num; i > 0; i--) {
        clustergcnSampleLayer(i, nei_set_sampled);
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
    }
    GraphBuild::buildGraphForSample();
}


void *fastgcnSampleParallel(void *data) {
    auto *data_parse = (RedPirorityData *) data;
    auto &adj = *data_parse->adj;
    auto &train_vertices = *data_parse->train_vertices;
    auto thread_id = data_parse->thread_id;
    auto &nei_set_sampled = *data_parse->nei_set_sampled;

    int itv = int(train_vertices.size() / WorkerStore::num_threads);
    int start_index = thread_id * itv;
    int end_index;

    if (thread_id == WorkerStore::num_threads - 1) {
        end_index = train_vertices.size();
    } else {
        end_index = (thread_id + 1) * itv;
    }

    unordered_map<int, unordered_set<int>> adj_sampled;


    for (int i = start_index; i < end_index; i++) {
        int v = train_vertices[i];
        auto &adj_v = adj[v];
        unordered_set<int> nei_set_v;

        for (auto nei:adj_v) {
            if (nei_set_sampled.count(nei)) {
                nei_set_v.insert(nei);
            }
        }

        adj_sampled.insert(pair<int, unordered_set<int>>(v, nei_set_v));

    }

    unique_lock<mutex> lck(ThreadUtil::mtx_ad);
    data_parse->adj_sampled->insert(adj_sampled.begin(), adj_sampled.end());
    lck.unlock();

}


void
fastgcnSampleLayer(int k, int lid) {
    auto &adj = WorkerStore::graph.adjs;
    vector<pthread_t> pthreads(WorkerStore::num_threads);
    unordered_map<int, unordered_set<int>> adj_sampled;
    unordered_map<int, double> nei_count;
    unordered_set<int> nei_set_sampled;
    auto &v2wk = WorkerStore::graph.v2wk;
    auto wid = WorkerStore::worker_id;
    auto &graphlayer = WorkerStore::graph_sampled.subgraphs["train"].graphlayers[lid];
    auto target_v = SetVecTrans::set2vec(graphlayer.target_v);

    int sum = 0;
    for (int id : target_v) {
        auto &adj_id = adj[id];
        for (auto nei:adj_id) {
            if (!nei_count.count(nei)) {
                nei_count.insert(make_pair(nei, 1));
                sum++;
            } else {
                nei_count[nei]++;
                sum++;
            }
        }
    }


    if (nei_count.size() <= k) {
        for (auto id : target_v) {
            adj_sampled.insert(make_pair(id, adj[id]));
        }
        graphlayer.adj = adj_sampled;
        return;
    } else {
        for (auto &elem:nei_count) {
            elem.second = (double) elem.second / (double) sum * k;
        }
    }

    for (auto &elem:nei_count) {
        float rand_num = rand() % 1000 / (float) 1000;
        if (rand_num <= elem.second) {
            nei_set_sampled.insert(elem.first);
        }
    }


    for (int i = 0; i < WorkerStore::num_threads; i++) {
        auto* data = new RedPirorityData;
        data->adj = &adj;
        data->train_vertices = &target_v;
        data->k = k;
        data->thread_id = i;
        data->adj_sampled = &adj_sampled;
        data->nei_set_sampled = &nei_set_sampled;
        pthread_create(&pthreads[i], NULL, fastgcnSampleParallel, (void *) data);

    }

    for (int i = 0; i < WorkerStore::num_threads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    graphlayer.adj = adj_sampled;
}


void Sample::fastgcnSample(vector<int> &fanout) {
    clearSampleGraph();
    auto &subgraph_sampled = WorkerStore::graph_sampled.subgraphs["train"];
    auto layer_num = WorkerStore::layer_num;
    auto target_v_tmp = WorkerStore::graph.idx["train"];
    subgraph_sampled.graphlayers[layer_num].target_v = target_v_tmp;

    for (int i = layer_num; i > 0; i--) {
        int k = fanout[layer_num - i];
        fastgcnSampleLayer(k, i);
        GraphBuild::updateGraphLayer(subgraph_sampled, i);
    }
//    GraphBuild::evalSubGraph(WorkerStore::graph_sampled.subgraphs["train"],"sample");
    GraphBuild::buildGraphForSample();
}
