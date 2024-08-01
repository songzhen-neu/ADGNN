

#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;


struct TestData {
    vector<int>* train_vertices;
    unordered_map<int, int>* o2n_dict;
};

void* thread01(void* data){
    clock_t startTime=clock();
    auto data_parse=(TestData *) data;
    auto& a=*data_parse->train_vertices;
    auto& b=*data_parse->o2n_dict;
    clock_t endTime = clock();
    cout << "parse time: " << (double) (endTime - startTime)/1000000 << "s" << endl;

}

void test01(){
    clock_t startTime=clock();
    vector<int> train_vertices(10000000);
    unordered_map<int, int> o2n_dict;
    for(int i=0;i<10000000;i++){
        train_vertices[i]=i;
        o2n_dict[i]=i;
    }
    clock_t endTime = clock();
    cout << "build time: " << (double) (endTime - startTime)/1000000 << "s" << endl;

    startTime=clock();
    TestData* testData=new TestData;
    testData->train_vertices=&train_vertices;
    testData->o2n_dict=&o2n_dict;
    pthread_t pthread;
    pthread_create(&pthread,NULL,thread01,(void *) testData);
    pthread_join(pthread,NULL);
    endTime = clock();
    cout << "thread time: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}



int main(){

}