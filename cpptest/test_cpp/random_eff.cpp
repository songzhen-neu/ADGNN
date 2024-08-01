

#include <iostream>
#include <cstdlib> // 标准库
#include <set>
#include <unordered_set>
#include <vector>

using namespace std;

void test01() {
    uint seed = 1;
    unordered_set<int> set_tmp;
//    set_tmp.rehash(1);
//    cout<<set_tmp.bucket_count()<<endl;
    for (int i = 0; i < 60; i++) {
        set_tmp.insert(i);
    }
    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (auto id : set_tmp) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test02() {
    uint seed = 1;

    set<int> set_tmp;
    for (int i = 0; i < 60; i++) {
        set_tmp.insert(i);
    }

    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (auto id : set_tmp) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test03() {
    uint seed = 1;

    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (int j = 0; j < 60; j++) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test04() {
    uint seed = 1;

    vector<int> vec_tmp(60);
    for (int i = 0; i < 60; i++) {
        vec_tmp[i] = (i);
    }

    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (auto id:vec_tmp) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test05() {
    uint seed = 1;

    vector<int> vec_tmp(60);
    for (int i = 0; i < 60; i++) {
        vec_tmp[i] = (i);
    }

    clock_t startTime = clock();
    unordered_set<int> set_new;
    for (int i = 0; i < 10000000; i++) {
        for (int j = 0; j < 60; j++) {
            auto a = 1 + 1;
        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test06() {
    uint seed = 1;

    vector<int> vec_tmp(60);
    for (int i = 0; i < 60; i++) {
        vec_tmp[i] = i;
    }

    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (int j = 0; j < 60; j++) {
            auto m = vec_tmp[j];

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test07() {
    uint seed = 1;

    vector<int> vec_tmp(60);
    for (int i = 0; i < 60; i++) {
        vec_tmp[i] = (i);
    }

    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        for (auto &id:vec_tmp) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test08() {
    uint seed = 1;
    unordered_set<int> set_tmp;
    for (int i = 0; i < 60; i++) {
        set_tmp.insert(i);
    }
    clock_t startTime = clock();
    unordered_set<int>::iterator it;
    for (int i = 0; i < 10000000; i++) {
        for (it = set_tmp.begin(); it != set_tmp.end(); it++) {

        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test09() {
    uint seed = 1;
    unordered_set<int> set_tmp;
    for (int i = 0; i < 60; i++) {
        set_tmp.insert(i);
    }
    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        vector<int> vec(60);
        copy(set_tmp.begin(), set_tmp.end(), back_inserter(vec));
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}

void test10() {
    uint seed = 1;
    vector<int> vec(60);

    for (int i = 0; i < 60; i++) {
        vec[i]=i;
    }
    clock_t startTime = clock();
    for (int i = 0; i < 10000000; i++) {
        unordered_set<int> set_tmp;
        for(int j=0;j<60;j++){
            set_tmp.insert(vec[j]);
        }
    }
    clock_t endTime = clock();
    cout << "Time:" << (double) (endTime - startTime) / 1000000 << endl;
}


int main() {
//    test01();
//    test02();
//    test03();
//    test04();
//    test05();
//    test06();
//    test07();
//    test08();
//    test09();
    test10();
}
