


//#include "../core/structure/hashset.h"
#include "../../core/structure/hashset.cpp"
#include "../../core/structure/hashmap.cpp"
#include <iostream>
#include <tr1/unordered_set>
#include <tr1/unordered_map>
#include <map>

using namespace std::tr1;

void test01() {
    clock_t startTime = clock();
    hashset<int> hs(1000000);
    for (int i = 0; i < 1000000; i++) {
        hs.insert(i);
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test02() {
    clock_t startTime = clock();
    unordered_set<int> hs;
    for (int i = 0; i < 1000000; i++) {
        hs.insert(i);
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test03() {
    clock_t startTime = clock();
    unordered_map<int, int> hm;
    for (int i = 0; i < 1000000; i++) {
        hm.insert(pair<int, int>(i, i));
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test04() {
    clock_t startTime = clock();
    hashmap<int, int> hm(1000000);
    for (int i = 0; i < 1000000; i++) {
        hm.insert(i, i);
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test05() {
    clock_t startTime = clock();
    map<int, int> m;
    for (int i = 0; i < 1000000; i++) {
        m.insert(pair<int, int>(i, i));
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}


void test06() {
    clock_t startTime = clock();
    unordered_set<int> hs;
    for (int i = 0; i < 1000000; i++) {
        hs.insert(i);
    }
    clock_t endTime = clock();
    cout << "The build time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;

    startTime = clock();
    for (int i = 0; i < 1000000; i++) {
        if(hs.count(i)){

        }
    }
    endTime = clock();
    cout << "The judgement time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;

    startTime = clock();
    for (int i = 0; i < 1000000; i++) {
        hs.erase(i);
    }
    endTime = clock();
    cout << "The erase time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;

}


int main() {
//    test01();
//    test02();
//    test03();
//    test04();
//    test05();
    test06();
}



