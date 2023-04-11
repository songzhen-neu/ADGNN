

#ifndef DGNN_TEST_DGNN_SERVER_H
#define DGNN_TEST_DGNN_SERVER_H

#include <iostream>
#include <grpcpp/grpcpp.h>
#include<grpcpp/health_check_service_interface.h>
#include<grpcpp/ext/proto_server_reflection_plugin.h>
#include "../store/WorkerStore.h"
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../../cmake/build/dgnn_test.pb.h"
#include "../partition/GeneralPartition.h"
#include "../util/threadUtil.h"
#include "../store/ServerStore.h"
#include <cmath>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <google/protobuf/repeated_field.h>
using grpc::Server;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerBuilder;


// here is proto defining package name. and the classes belong to this package
using dgnn_test::DgnnProtoService;
using dgnn_test::DataMessage;
using dgnn_test::ContextMessage;
using dgnn_test::NetInfoMessage;
using dgnn_test::EmbGradMessage;
using dgnn_test::AccuracyMessage;
using dgnn_test::NodeMessage;
using dgnn_test::ParamGrad;
using dgnn_test::LayerNodeListMessage;
using dgnn_test::NullMessage;
using dgnn_test::IntIntPair;
using dgnn_test::MMessageForAD;

class ServiceImpl final:public DgnnProtoService::Service{
public:

    Status pullDataFromMasterGeneral(
            ServerContext* context,const ContextMessage* request,
            DataMessage* reply) override;
    Status initParameter(
            ServerContext* context,const NetInfoMessage* request,
            NullMessage* reply) override;

    Status barrier(
            ServerContext* context,const NullMessage* request,
            NullMessage* reply) override;
    Status workerPullEmb(
            ServerContext* context,const EmbGradMessage* request,
            EmbGradMessage* reply) override;


//    Status workerPullG(
//            ServerContext* context,const EmbGradMessage* request,
//            EmbGradMessage* reply) override;


    Status sendAccuracy(ServerContext *context,const AccuracyMessage *request,
                        AccuracyMessage *reply) override;

    Status aggregateNodes(ServerContext *context,const NodeMessage *request,
                          NodeMessage *reply) override;

    Status freeMaster(ServerContext *context,const NullMessage *request,NullMessage *reply) override;

    static void RunServerByPy(const string& address,int serverId);


    Status server_PullParams(ServerContext *context, const ParamGrad *request, ParamGrad *reply) override;
    Status server_updateParam(ServerContext *context,const ParamGrad *request, NullMessage *reply) override;

    Status setAndSendG(ServerContext *context,const EmbGradMessage *request,NullMessage *reply) override;
    Status sendNodes2Wk(ServerContext *context,const NodeMessage *request,NullMessage *reply) override;

    Status workerPullRmtTrainFeat(ServerContext *context,const EmbGradMessage *request,EmbGradMessage *reply) override;


    Status sendInNodes2Wk(ServerContext *context,const NodeMessage *request,NullMessage *reply) override;
    Status pushEmbs(ServerContext *context, const EmbGradMessage *request, NullMessage *reply) override;

    Status setM(ServerContext *context, const MMessageForAD *request, MMessageForAD *reply) override;

};




#endif //DGNN_TEST_DGNN_SERVER_H







