

#ifndef DGNN_TEST_THREADUTIL_H
#define DGNN_TEST_THREADUTIL_H
#include<pthread.h>
#include <condition_variable>
#include <unistd.h>
#include <mutex>
#include <vector>
#include <atomic>
using namespace std;

class ThreadUtil {
public:
    static mutex mtx;
    static bool ready;
    static mutex mtx_initParameter;
    static bool ready_initParameter;
    static condition_variable cv;
    static void addone(int &count);
    static int arrived_worker_num;

    static mutex mtx_sendNode;

    static mutex mtx_barrier;
    static condition_variable cv_barrier;
    static int count_worker_for_barrier;


    static mutex mtx_updateModels;
    static pthread_mutex_t mtx_updateModels_addGrad;
    static mutex mtx_updateModels_barrier;
    static condition_variable cv_updateModels;
    static int count_worker_for_updateModels;
    static bool ready_updateModels;
    static bool ready_updateModels_2;

    static int count_respWorkerNumForEmbs;
    static mutex mtx_respWorkerNumForEmbs;

    static int count_accuracy;
    static mutex mtx_accuracy;
    static condition_variable cv_accuracy;

    static int count_nodes;
    static mutex mtx_nodes;
    static condition_variable cv_nodes;

    static mutex mtx_gcompensate;

    static int count_compress_thread[30];
    static mutex mtx_thread;
//    static mutex mtx_compress_thread[100];

    static int count_merge_nodes_bp;
    static mutex mtx_merge_nodes_bp;
    static condition_variable cv_merge_nodes_bp;

    static mutex mtx_setAndSendG;
    static int count_setAndSendG;
    static mutex mtx_setAndSendG_for_count;

    static mutex mtx_sendNodes2Wk;
    static int count_sendNodes2Wk;
    static condition_variable cv_sendNodes2Wk;

    static mutex mtx_sendNodes2Wk_2;
    static int count_sendNodes2Wk_2;
    static condition_variable cv_sendNodes2Wk_2;

    static mutex mtx_sendNodes2Wk_merge;

    static int count_sendNodes2Wk_threads;

    static int count_pushEmbsParallel;
    static mutex mtx_pushEmbsParallel;

    static vector<vector<pthread_t>> pthread_vec;

    static mutex mtx_ad;


    static mutex mtx_rmtfeature_insert;
    static mutex mtx_pc;

    static mutex mtx_pushembs;



};


#endif //DGNN_TEST_THREADUTIL_H
