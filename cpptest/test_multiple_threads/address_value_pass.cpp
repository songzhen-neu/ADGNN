
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;


void test01() {
    clock_t startTime = clock();
    vector<int> train_vertices(100000000);
    clock_t endTime = clock();
    cout << "build time: " << (double) (endTime - startTime) / 1000000 << "s" << endl;

    startTime = clock();
    auto tv_copy = train_vertices;
    endTime = clock();
    cout << "build2 time: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

vector<int> test02() {
    clock_t startTime = clock();
    vector<int> train_vertices(100000000);
    clock_t endTime = clock();
    cout << "build time 1: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
    return train_vertices;
}


void test03(vector<int> a) {

}

int main() {

    vector<int> train_vertices(100000000);

    clock_t startTime = clock();
    test03(train_vertices);
    clock_t endTime = clock();
    cout << "all time 1: " << (double) (endTime - startTime) / 1000000 << "s" << endl;


    startTime = clock();
    auto a = test02();
    endTime = clock();
    cout << "all time 2: " << (double) (endTime - startTime) / 1000000 << "s" << endl;


}