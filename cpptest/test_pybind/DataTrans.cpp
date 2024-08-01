
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include <vector>
#include <pybind11/stl_bind.h>
#include <map>

//PYBIND11_MAKE_OPAQUE(std::vector<int>);
//PYBIND11_MAKE_OPAQUE(std::map<int, std::vector<int>>);
//PYBIND11_MAKE_OPAQUE(std::map<int,int>);

using namespace std;
namespace py = pybind11;


// ***********************test***********************
void test_numpy(py::array_t<float> *embs) {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    py::buffer_info buf = embs->request();
    if (buf.ndim != 2) {
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    auto *ptr = (float *) buf.ptr;
    double sum = 0;
    cout << "float size:" << sizeof(float) << ",buffer size :" << buf.size << endl;

    for (int i = 0; i < buf.size; i++) {
//        if(ptr[i]!=1){
//            cout<<ptr[i]<<endl;
//        }
        sum += ptr[i];
    }
    cout << "sum:" << sum << endl;

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "traverse" << timeuse << endl;
}

void test_numpy_vec(vector<vector<float>> &embs) {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    int sum = 0;
    for (int i = 0; i < embs.size(); i++) {
        auto emb_i = embs[i];
        for (int j = 0; j < embs[i].size(); j++) {
            sum += emb_i[j];
        }

    }
    cout << "sum:" << sum << endl;

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "traverse:" << timeuse << endl;
}


void test_dict(map<int, vector<int>> *dict) {
    cout << "aaa" << endl;
}

void test_intintdict(map<int, int> &dict) {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    int sum = 0;
    for (auto elem:dict) {
        sum += elem.second;
    }
    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << timeuse << endl;
}

void test_IIM(map<int, int> &dict) {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    int sum = 0;
    for (auto elem:dict) {
        sum += elem.second;
    }
    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << timeuse << endl;
}

void test_vec(vector<int> &data) {
    cout << "aaa" << endl;
}

map<int, vector<int>> test_dict_r() {
    map<int, vector<int>> data_r;
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    for (int i = 0; i < 200000; i++) {
        int osc = i % 20;
        vector<int> vec_tmp(100 + osc);
        for (int j = 0; j < 100 + osc; j++) {
            vec_tmp[j] = j;
        }
        data_r.insert(pair<int, vector<int>>(i, vec_tmp));
    }
    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "traverse:" << timeuse << endl;
    return data_r;
}

void test_dict_int_vec(map<int, vector<int>> &data) {
    cout << "aaa" << endl;
}

void resize_vector(vector<int> &vec, int size) {
    vec.resize(size);
}

// ***********************test***********************



map<int, py::array_t<float>> test_map_arrayt() {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    map<int,py::array_t<float>> map_return;
    for (int i = 0; i < 1000000; i++) {
        auto result = py::array_t<float>(100);
        py::buffer_info buf_result = result.request();
        auto *ptr_result = (float *) buf_result.ptr;
        for(int j=0;j<100;j++){
            ptr_result[j]=j;
        }
        map_return.insert(make_pair(i,result));
    }

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "build:" << timeuse << endl;
    return map_return;
}

map<int, vector<float>> test_map_vector() {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    map<int,vector<float>> map_return;
    for (int i = 0; i < 1000000; i++) {
       vector<float> result(100);
        for(int j=0;j<100;j++){
            result[j]=j;
        }
        map_return.insert(make_pair(i,result));
    }

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "build:" << timeuse << endl;
    return map_return;
}

pair<py::array_t<int>,py::array_t<float>> test_id_arrayt() {
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    auto id_arr=py::array_t<int>(1000000);
    auto data_arr=py::array_t<float>(100000000);
    auto id_req=id_arr.request();
    auto id_ptr=(int *) id_req.ptr;
    auto data_arr_request=data_arr.request();
    auto data_arr_ptr=(float *) data_arr_request.ptr;
    for (int i = 0; i < 1000000; i++) {
        id_ptr[i]=i;
    }
    for (int i = 0; i < 100000000; i++) {
        data_arr_ptr[i]=i;
    }

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << "build:" << timeuse << endl;
    return make_pair(id_arr,data_arr);
}


PYBIND11_MODULE(datatrans, m) {
    m.def("test_numpy", &test_numpy, "test numpy");
    m.def("test_numpy_vec", &test_numpy_vec, "test numpy");
    m.def("test_dict", &test_dict, "test dict");
    m.def("test_intintdict", &test_intintdict, "test dict");
    m.def("test_dict_r", &test_dict_r, "test dict");
    m.def("test_vec", &test_vec, "test vec");
    m.def("test_dict_int_vec", &test_dict_int_vec, "test vec");
    m.def("test_IIM", &test_IIM, "test vec");
    m.def("test_map_arrayt", &test_map_arrayt, "test vec");
    m.def("test_map_vector", &test_map_vector, "test vec");
    m.def("test_id_arrayt", &test_id_arrayt, "test vec");
//    py::bind_map<map<int,int>>(m,"IIMap");
//    py::bind_vector<std::vector<int>>(m, "VectorInt");
//    py::bind_map<std::map<int, std::vector<int>>>(m, "MapIntVec");
//    py::class_<std::vector<int>>(m, "VectorInt")
//            .def(py::init<>())
//            .def("clear", &std::vector<int>::clear)
//            .def("pop_back", &std::vector<int>::pop_back)
//            .def("set",[](vector<int> &vec,int index,int value){vec[index]=value;})
//            .def("resize",&resize_vector)
//            .def("__len__", [](const std::vector<int> &v) { return v.size(); })
//            .def("__iter__", [](std::vector<int> &v) {
//                return py::make_iterator(v.begin(), v.end());
//            }, py::keep_alive<0, 1>()) ;

}