

#ifndef EC_GRAPH_SETVECTRANS_H
#define EC_GRAPH_SETVECTRANS_H
#include <iostream>
#include <set>
#include <vector>
#include <unordered_set>
using namespace std;

class SetVecTrans {
public:
    template<typename T1>
    static set<T1> vec2set(vector<T1> &vec){
        set<T1> set_tmp;
        for (auto &e:vec) {
            set_tmp.insert(e);
        }
        return set_tmp;};
    template<typename T1>
    static vector<T1> set2vec(set<T1> &s){
        vector<T1> vec_tmp(s.size());
        int flag = 0;
        for (auto &e:s) {
            vec_tmp[flag] = e;
            flag++;
        }
        return vec_tmp;
    };
    template<typename T1>
    static unordered_set<T1> vec2unodset(vector<T1> &vec){
        unordered_set<T1> set_tmp;
        for (auto &e:vec) {
            set_tmp.insert(e);
        }
        return set_tmp;};
    template<typename T1>
    static vector<T1> set2vec(unordered_set<T1> &s){
        vector<T1> vec_tmp(s.size());
        int flag = 0;
        for (auto &e:s) {
            vec_tmp[flag] = e;
            flag++;
        }
        return vec_tmp;
    };
};


#endif //EC_GRAPH_SETVECTRANS_H
