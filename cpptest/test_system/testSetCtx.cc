

#include <iostream>
#include "../../core/service/dgnn_client.h"

using namespace std;
int main(){
    DGNNClient dgnnClient;
    int layernum=2;
    int workernum=3;
    map<int,map<int,vector<int>>> fsthop4wk_train;
    map<int, map<int, int>> old2new_train;
    map<int, int> local_vertex_num;
    map<string, string> cluster_config;
    int elem=5;
    cluster_config.insert(pair<string,string>("worker_num","3"));
    cluster_config.insert(pair<string,string>("worker_id","0"));
    for(int i=0;i<layernum;i++){
        map<int,vector<int>> ft_layer;
        map<int,int> o2n_layer;

        for(int j=0;j<elem;j++){
            o2n_layer.insert(pair<int,int>(j,j+1));
        }
        local_vertex_num.insert(pair<int,int>(i,i));
        cluster_config.insert(pair<string,string>(to_string(i),to_string(i)));
        for(int j=0;j<workernum;j++){
            vector<int> ftl_worker(elem);
            for(int k=0;k<elem;k++){
                ftl_worker.push_back(k);
            }
            ft_layer.insert(pair<int,vector<int>>(j,ftl_worker));
        }
        fsthop4wk_train.insert(pair<int,map<int,vector<int>>>(i,ft_layer));
        old2new_train.insert(pair<int,map<int,int>>(i,o2n_layer));
    }

    cout<<fsthop4wk_train[0][0][0]<<endl;


//    dgnnClient.setCtxForCpp(layernum,fsthop4wk_train,old2new_train,local_vertex_num,cluster_config);
    cout<<"aaa"<<endl;
}