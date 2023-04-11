
#include <iostream>
#include <zconf.h>
#include <map>
#include<set>
#include<fstream>
#include<vector>
#include<string>
#include "../util/string_split.h"
#include <unordered_set>
#include <sys/stat.h>

using namespace std;

void startPartition(int worker_num, string partitionMethod, int nodeNum, string filename, int feature_size) {
    vector<vector<int>> nodes;
    vector<map<int, vector<float>>> features;
    vector<map<int, int>> labels;
    vector<map<int, unordered_set<int>>> adjs;

    char *buffer;
    buffer = getcwd(NULL, 0);
    string pwd = buffer;
    cout << "pwd:" << pwd << ",buffer:" << buffer << endl;
//    string pwd =buffer;
    if (pwd[pwd.length() - 1] != '/') {
        pwd += '/';
    }
    // 开始处理数据集
//    string adjFile = pwd+"data_raw/cora/edges.txt";
//    string featFile = pwd+"data_raw/cora/featsClass.txt";

//    string adjFile = pwd + this->filename + "/edges.txt";
//    string featFile = pwd + this->filename + "/featsClass.txt";
//    string partitionFile =
//            pwd + this->filename + "/nodesPartition" + "." + partitionMethod + to_string(worker_num) + ".txt";
    string adjFile = filename + "/edges.txt";
    string featFile = filename + "/featsClass.txt";
    string partitionFile =
            filename + "/nodesPartition" + "." + partitionMethod + to_string(worker_num) + ".txt";

    cout << "adjfile:" << adjFile << endl;
    cout << "featFle:" << featFile << endl;
    cout << "partition file path:" << partitionFile << endl;


    ifstream partitionInFile(partitionFile);
    string temp;
    if (!partitionInFile.is_open()) {
        cout << "partitionInFile 未成功打开文件" << endl;
    }
    int count_worker = 0;
    while (getline(partitionInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vSize = v.size();
        vector<int> v_tmp(vSize);
        nodes.push_back(v_tmp);
        auto &nodesWorker = nodes[count_worker];
        for (int i = 0; i < vSize; i++) {
            nodesWorker[i] = atoi(v[i].c_str());
//            cout<<nodesWorker[i]<<",";
        }
//        cout<<endl;

        cout << "nodes num for worker " << count_worker << " :" << nodes[count_worker].size() << endl;
        count_worker++;
    }
    partitionInFile.close();


    ifstream adjInFile(adjFile);

    if (!adjInFile.is_open()) {
        cout << "adjInFile 未成功打开文件" << endl;
    }


    vector<unordered_set<int>> adj_map(nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        unordered_set<int> set_tmp;
        adj_map[i] = set_tmp;
    }
//    vector<vector<int>> adj_vec(2);
//    for(int i=0;i<adj_vec.size();i++){
//        vector<int> vec_tmp(edgeNum);
//        adj_vec[i]=vec_tmp;
//    }

    int count = 0;
    int count_flag = 0;
    cout << "正在处理邻接表数据" << endl;
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int neibor_id = atoi(v[1].c_str());

        // 开始构造邻接表
        adj_map[vertex_id].insert(neibor_id);
        adj_map[neibor_id].insert(vertex_id);
        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }
    }
    int edge_num = count_flag;
    adjInFile.close();




    // 开始处理feature和label,同样使用in file stream



    map<int, vector<float>> feature;
    vector<int> label_array(nodeNum); // 如果需要获取length，那么这块只能赋值常量
//    map<string, int> label_map;
    int count_label = 0;

    ifstream featInFile(featFile);
    if (!featInFile.is_open()) {
        cout << "未成功打开文件" << endl;
    }

    count = 0;
    count_flag = 0;
    cout << "正在处理特征数据 " << endl;

    while (true) {
        getline(featInFile, temp);
        if (temp.empty()) {
            break;
        }
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        vector<float> vec_feat(feature_size);
        for (int i = 1; i < feature_size + 1; i++) {
            vec_feat[i - 1] = atof(v[i].c_str());
        }
        feature.insert(pair<int, vector<float>>(vertex_id, vec_feat));

//        cout<<"label:"<<label_new<<endl;
        label_array[vertex_id] = atoi(v[feature_size + 1].c_str());
        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }
    }


    featInFile.close();

    // 开始划分，邻接表、顶点map、属性、标签
    // 这里顶点按照哈希（取余数）的方式进行划分，因此不需要建立map
    // 邻接表：map<int, map<int,set>>

    cout << "adj_map size:" << adj_map.size() << endl;
    cout << "边数:" << edge_num << endl;


    for (int i = 0; i < worker_num; i++) {
        auto &node_worker_i = nodes[i];
        int nodeSize = node_worker_i.size();
        map<int, unordered_set<int> > adjForWorkerI_tmp;
        adjs.push_back(adjForWorkerI_tmp);
        auto &adjForWorkerI = adjs[i];

        map<int, int> label_tmp;
        labels.push_back(label_tmp);
        auto &labelWorkerI = labels[i];

        map<int, vector<float>> feat_tmp;
        features.push_back(feat_tmp);
        auto &featWorkerI = features[i];

        for (int j = 0; j < nodeSize; j++) {
            int nodeId = node_worker_i[j];
            auto &neiborVecForNode = adj_map[nodeId];
            auto &featVecForNode = feature[nodeId];
            adjForWorkerI.insert(pair<int, unordered_set<int>>(nodeId, neiborVecForNode));
            labelWorkerI.insert(pair<int, int>(nodeId, label_array[nodeId]));
            featWorkerI.insert(pair<int, vector<float>>(nodeId, featVecForNode));
        }
    }

    for (int i = 0; i < worker_num; i++) {
//        string filename_w= filename + "/" + partitionMethod+to_string(worker_num)+"/"+"part"+to_string(i)+"/";
        vector<string> f_write_vec;
        f_write_vec.push_back(filename + "/" + partitionMethod + to_string(worker_num) + "/");
        f_write_vec.push_back(
                filename + "/" + partitionMethod + to_string(worker_num) + "/" + "part" + to_string(i) + "/");

        for (auto &f:f_write_vec) {
            if (access(f.c_str(), 0) == -1) {
                mkdir(f.c_str(), 0777);
            }
        }
        ofstream ofs;
        string f_feat =
                filename + "/" + partitionMethod + to_string(worker_num) + "/" + "part" + to_string(i) + "/feat.txt";
        ofs.open(f_feat, ios::out);
        for (auto &elem:features[i]) {
            string str_tmp = "";
            int id = elem.first;
            vector<float> &feat = elem.second;
            str_tmp += to_string(id) + "\t";
            for (int k = 0; k < feat.size(); k++) {
                if (k == feat.size() - 1) {
                    str_tmp += to_string(feat[k]) + "\n";
                } else {
                    str_tmp += to_string(feat[k]) + "\t";
                }
            }
            ofs << str_tmp;
        }
        ofs.close();

        string f_adj =
                filename + "/" + partitionMethod + to_string(worker_num) + "/" + "part" + to_string(i) + "/adj.txt";
        ofs.open(f_adj, ios::out);
        for (auto &elem:adjs[i]) {
            string str_tmp = "";
            int id = elem.first;
            unordered_set<int> &adj_v = elem.second;
            str_tmp += to_string(id);
            for (auto nei_id:adj_v) {
                str_tmp += "\t" + to_string(nei_id);
            }
            str_tmp += "\n";
            ofs << str_tmp;
        }
        ofs.close();


        string f_label =
                filename + "/" + partitionMethod + to_string(worker_num) + "/" + "part" + to_string(i) + "/label.txt";
        ofs.open(f_label, ios::out);
        for (auto &elem:labels[i]) {
            string str_tmp = "";
            int id = elem.first;
            int lab_v = elem.second;
            str_tmp += to_string(id) + "\t" + to_string(lab_v) + "\n";

            ofs << str_tmp;
        }
        ofs.close();

//        string f_node =filename + "/" + partitionMethod + to_string(worker_num) + "/" + "part" + to_string(i) + "/node.txt";
//        ofs.open(f_node, ios::out);
//        for (int k = 0; k < nodes[i].size(); k++) {
//            string str_tmp = "";
//
//            if (k == nodes[i].size() - 1) {
//                str_tmp += to_string(nodes[i][k]) + "\n";
//            } else {
//                str_tmp += to_string(nodes[i][k]) + "\t";
//            }
//
//            ofs << str_tmp;
//        }
//        ofs.close();
        cout<<"worker "<<i<< " write success"<<endl;

    }
    cout<<"partition success"<<endl;


};



//(int worker_num, string partitionMethod, int nodeNum, int edgeNum) {
//int worker_num=2;
//string partMethod="hash";
//int nodeNum=2708;
//int feature_size=1433;
//string filename="/mnt/data/cora";


int worker_num = 2;
string partMethod = "hash";
int nodeNum = 16;
int feature_size = 4;
string filename = "/mnt/data/test";

int main() {
    startPartition(worker_num, partMethod, nodeNum, filename, feature_size);
}


