
#include <iostream>
#include <atomic>
#include <thread>
#include <pthread.h>
#include <vector>
#include <mutex>

using namespace std;

//static atomic<int> a(0);
static atomic<int> a(0);
mutex mtx;
static int count_thread;

void *addone(void *data) {
    unique_lock<mutex> lck(mtx);
    a++;
    count_thread++;

}


int main() {
    int thread_num = 20000;
    vector<pthread_t> p_threads(thread_num);
    for (int i = 0; i < thread_num; i++) {
//        pthread_t p_tmp;
//        p_threads[i]=p_tmp;

        int *i_ptr = &i;
        pthread_create(&p_threads[i], NULL, addone, (void *) i_ptr);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(p_threads[i], NULL);
    }
//    while (count_thread!=thread_num){cout<<count_thread<<endl;}
    cout << "int a:" << count_thread << endl;
    cout << "atomic<int> a:" << a << endl;
}
