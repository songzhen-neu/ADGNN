

#include "threadUtil.h"

// 只在head中声明，不要在head中定义，否则会报multi definition错误
mutex ThreadUtil::mtx; // 全局互斥锁.
bool ThreadUtil::ready = false;
condition_variable ThreadUtil::cv;
int ThreadUtil::arrived_worker_num = 0;

mutex ThreadUtil::mtx_sendNode;

mutex ThreadUtil::mtx_gcompensate;

int ThreadUtil::count_compress_thread[30];
//mutex ThreadUtil::mtx_compress_thread[100];
mutex ThreadUtil::mtx_thread;
mutex ThreadUtil::mtx_initParameter;
bool ThreadUtil::ready_initParameter = false;

void ThreadUtil::addone(int &count) {
    unique_lock<mutex> lock(mtx);
    count++;
}

int ThreadUtil::count_accuracy = 0;
mutex ThreadUtil::mtx_accuracy;
condition_variable ThreadUtil::cv_accuracy;

mutex ThreadUtil::mtx_barrier;
condition_variable ThreadUtil::cv_barrier;

int ThreadUtil::count_worker_for_barrier;

mutex ThreadUtil::mtx_updateModels;
condition_variable ThreadUtil::cv_updateModels;
int ThreadUtil::count_worker_for_updateModels;
bool ThreadUtil::ready_updateModels = false;
bool ThreadUtil::ready_updateModels_2 = false;

pthread_mutex_t ThreadUtil::mtx_updateModels_addGrad;
mutex ThreadUtil::mtx_updateModels_barrier;

int ThreadUtil::count_respWorkerNumForEmbs = 0;
mutex ThreadUtil::mtx_respWorkerNumForEmbs;

int ThreadUtil::count_merge_nodes_bp = 0;
mutex ThreadUtil::mtx_merge_nodes_bp;
condition_variable ThreadUtil::cv_merge_nodes_bp;


mutex ThreadUtil::mtx_setAndSendG;
int ThreadUtil::count_setAndSendG = 0;
mutex ThreadUtil::mtx_setAndSendG_for_count;

mutex ThreadUtil::mtx_sendNodes2Wk;
int ThreadUtil::count_sendNodes2Wk = 0;
condition_variable ThreadUtil::cv_sendNodes2Wk;

 mutex ThreadUtil::mtx_sendNodes2Wk_2;
 int ThreadUtil::count_sendNodes2Wk_2=0;
 condition_variable ThreadUtil::cv_sendNodes2Wk_2;


mutex ThreadUtil::mtx_sendNodes2Wk_merge;

int ThreadUtil::count_sendNodes2Wk_threads = 0;

int ThreadUtil::count_pushEmbsParallel = 0;
mutex ThreadUtil::mtx_pushEmbsParallel;

vector<vector<pthread_t>> ThreadUtil::pthread_vec;

mutex ThreadUtil::mtx_ad;


int ThreadUtil::count_nodes;
mutex ThreadUtil::mtx_nodes;
condition_variable ThreadUtil::cv_nodes; // 全局条件变量.

mutex ThreadUtil::mtx_rmtfeature_insert;

mutex ThreadUtil::mtx_pc;

 mutex ThreadUtil::mtx_pushembs;






