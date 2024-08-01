

#include <iostream>
#include <string>

#include "../../core/store/WorkerStore.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


using namespace std;
namespace py=pybind11;

int set_embs_ptr(py::array_t<float>& embs,string status);
//
int set_embs_ptr(py::array_t<float>& embs,string status){
//    WorkerStore::embs.clear();
//    DGNNClient dgnnClient;
    py::buffer_info buf=embs.request();
    if( buf.ndim!=2){
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    auto* ptr=(float*) buf.ptr;

    WorkerStore::embs_ptr_map[status]=ptr;
    cout<<"embs ptr size:"<<buf.shape[0]<<","<<buf.shape[1]<<endl;
    return 0;
}

int main(){
    float *a=new float[20];

    for(int i=0;i<20;i++){
        a[i]=atof(to_string(i).c_str());
    }


//    set_embs_ptr(nullptr, "train");
}