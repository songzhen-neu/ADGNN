

#include "check.h"


void Check::check_features(unordered_map<int, vector<float>> &features) {
    cout << "server: feature size:" << features.size() << endl;
    cout << "server: feature dims:" << features.begin()->second.size() << endl;
}

void Check::check_labels(unordered_map<int, int> &labels) {
    cout << "server: label size:" << labels.size() << endl;
}

void Check::check_adjs(unordered_map<int, unordered_set<int>> &adjs) {
    cout << "server: adjs size:" << adjs.size() << endl;
}

void Check::check_partition_pass(
        const int &workerNum, const int &dataNum, const string &dataPath, const int &feature_dim,
        const int &class_num) {
    cout << "worker number:" << workerNum << endl;
    cout << "dataNum:" << dataNum << endl;
    cout << "dataPath:" << dataPath << endl;
    cout << "feature_dim:" << feature_dim << endl;
    cout << "class_num:" << class_num << endl;
}

void Check::check_initParameter_ServerStore() {
    cout << "*******check initParameter start***********" << endl;
    map<string, vector<double>>::iterator it;
    for (it = ServerStore::params.begin(); it != ServerStore::params.end(); it++) {
        cout<<"layer "<<it->first<<" : "<<it->second.size()<<endl;
    }
    cout << "*******check initParameter end***********" << endl;
}