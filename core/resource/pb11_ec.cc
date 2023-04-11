

#include <pybind11/pybind11.h>
#include "../../cpptest/Animal.cc"
#include <pybind11/stl.h>
#include "../service/dgnn_server.h"
#include "../service/dgnn_client.h"
#include "pybind11/numpy.h"
#include "../service/router.h"
#include "../data_preprocess/partition_graph.cpp"
#include "../graph_build/GraphBuild.h"
#include "../sample/Sample.h"


using namespace std;
namespace py = pybind11;





void set_embs_ptr(py::array_t<float>* embs,string status){
//    WorkerStore::embs.clear();
//    DGNNClient dgnnClient;
    py::buffer_info buf=embs->request();
    if( buf.ndim!=2){
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    auto* ptr=(float*) buf.ptr;
    WorkerStore::embs_ptr_map[status]=ptr;
}



PYBIND11_MODULE(pb11_ec, m) {

    m.doc() = "pybind11 example plugin";
    m.def("set_embs_ptr",&set_embs_ptr,"set local embs");
    m.def("startPartition",&startPartition,"partition graph");
//    m.def("set_g_ptr",&set_g_ptr,"set local g");



    py::class_<ServiceImpl>(m,"ServiceImpl")
            .def(py::init<>())
            .def("RunServerByPy",&ServiceImpl::RunServerByPy);

    py::class_<Sample>(m,"Sample")
            .def(py::init<>())
            .def("initSampledGraph",&Sample::initSampledGraph)
            .def("randomSample",&Sample::randomSample)
            .def("randomSampleNoRebuild",&Sample::randomSampleNoRebuild)
            .def("adSample",&Sample::adSample)
            .def("bnsSample",&Sample::bnsSample)
            .def("buildRmtAndLocAdj",&Sample::buildRmtAndLocAdj)
            .def("clustergcnSample",&Sample::clustergcnSample)
            .def("fastgcnSample",&Sample::fastgcnSample)
            .def("setAggEmb",&Sample::setAggEmb);


    py::class_<Router>(m,"Router")
            .def(py::init<>())
            .def("getRemoteEmb",&Router::getRemoteEmb)
            .def("setAndSendG",&Router::setAndSendG)
            .def("initWorkerRouter",&Router::initWorkerRouter)
            .def("initServerRouter",&Router::initServerRouter)
            .def("sendNodes2Wk",&Router::sendNodes2Wk)
            .def("getRmtFeat",&Router::getRmtFeat)
            .def("sendInNodes2WK",&Router::sendInNodes2WK)
            .def("pushEmbs",&Router::pushEmbs)
            .def("pushEmbsByIds",&Router::pushEmbsByIds)
//            .def("getEntireEmbs",&Router::getEntireEmbs)
            .def("initReplyEmbs",&Router::initReplyEmbs);


    py::class_<GraphBuild>(m,"GraphBuild")
            .def(py::init<>())
            .def("buildInitGraph",&GraphBuild::buildInitGraph)
            .def("transEdgeToNewID",&GraphBuild::transEdgeToNewID)
            .def("getTrainedFeat",&GraphBuild::getTrainedFeat)
            .def("getTrainedLabel",&GraphBuild::getTrainedLabel)
            .def("setGraphMode",&GraphBuild::setGraphMode)
            .def("deleteFullGraphInCpp",&GraphBuild::deleteFullGraphInCpp)
            .def("printGraphInfo",&GraphBuild::printGraphInfo);


    // 创建client
    py::class_<DGNNClient>(m,"DGNNClient")
            .def(py::init<>())
            .def("init_by_address",&DGNNClient::init_by_address)
            .def("startClientServer",&DGNNClient::startClientServer)
//            .def("pullDataFromMasterGeneral",&DGNNClient::pullDataFromMasterGeneral)
            .def("initParameter",&DGNNClient::initParameter)
            .def("server_Barrier",&DGNNClient::server_Barrier)
            .def("sendAccuracy",&DGNNClient::sendAccuracy)
            .def("aggregateNodes",&DGNNClient::aggregateNodes)
            .def("freeMaster",&DGNNClient::freeMaster)
            .def("freeSpace",&DGNNClient::freeSpace)
            .def("server_PullParams",&DGNNClient::server_PullParams)
            .def("server_updateParam",&DGNNClient::server_updateParam)
            .def("setGraphInfoForCpp",&DGNNClient::setGraphInfoForCpp)
            .def("setCtxForCpp",&DGNNClient::setCtxForCpp)
            .def("setStaticInfoForCpp",&DGNNClient::setStaticInfoForCpp)
//            .def("transEdgeToNewID",&DGNNClient::transEdgeToNewID)
            .def("readDataAndInit",&DGNNClient::readDataAndInit)
            .def("setM",&DGNNClient::setM)
//            .def_property("nodesForEachWorker",&DGNNClient::get_nodesForEachWorker,&DGNNClient::set_nodesForEachWorker)
            .def_property("nodes",&DGNNClient::get_nodes,&DGNNClient::set_nodes)
            .def_property("features",&DGNNClient::get_features,&DGNNClient::set_features)
            .def_property("labels",&DGNNClient::get_labels,&DGNNClient::set_labels)
            .def_property("adjs",&DGNNClient::get_adjs,&DGNNClient::set_adjs)
            .def_property("v2wk",&DGNNClient::get_v2wk,&DGNNClient::set_v2wk)
            .def_property("wk2v",&DGNNClient::get_wk2v,&DGNNClient::set_wk2v)
            .def_property("serverAddress",&DGNNClient::get_serverAddress,&DGNNClient::set_serverAddress)
//            .def_property("degree_map",&DGNNClient::get_degree_map,&DGNNClient::set_degree_map)
            .def_property("layerNum",&DGNNClient::get_layerNum,&DGNNClient::set_layerNum);


}



