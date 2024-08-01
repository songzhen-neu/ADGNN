

#include<set>
#include<time.h>
#include <iostream>
#include <vector>
#include <map>

using namespace std;

int set_num = 100;
int set_size = 10000;

//1、声明变量，使用pushback
void test01() {
    set<int> s;
    vector<set<int>> vec(set_num);
    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        set<int> s_i;
        for(int j=0;j<set_size;j++){
            s_i.insert(j);
        }
        vec[i]=s_i;
    }
    clock_t endTime = clock();
    cout << "The test01 building time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test02() {
    set<int> s;
    vector<set<int>> vec;
    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        set<int> s_i;
        for(int j=0;j<set_size;j++){
            s_i.insert(j);
        }
        vec.push_back(s_i);
    }
    clock_t endTime = clock();
    cout << "The test02 building time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test03() {
    set<int> s;
    vector<vector<int>> vec;
    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        vector<int> s_i;
        for(int j=0;j<set_size;j++){
            s_i.push_back(j);
        }
        vec.push_back(s_i);
    }
    clock_t endTime = clock();
    cout << "The test03 building time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test04() {
    set<int> s;
    vector<vector<int>> vec;
    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        vector<int> s_i(set_size);
        for(int j=0;j<set_size;j++){
            s_i[j]=(j);
        }
        vec.push_back(s_i);
    }
    clock_t endTime = clock();
    cout << "The test04 building time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}


void test05() {
    set<int> s;
    vector<set<int>> vec;

    for(int i=0;i<set_num;i++){
        set<int> s_i;
        for(int j=0;j<set_size;j++){
            s_i.insert(j);
        }
        vec.push_back(s_i);
    }

    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        for(auto elem:vec[i]){
            s.insert(elem);
        }
    }
    clock_t endTime = clock();
    cout << "The test05 copy time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test06() {
    set<int> s;
    vector<set<int>> vec;

    for(int i=0;i<set_num;i++){
        set<int> s_i;
        for(int j=0;j<set_size;j++){
            s_i.insert(j);
        }
        vec.push_back(s_i);
    }

    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        s.insert(vec[i].begin(),vec[i].end());
    }
    clock_t endTime = clock();
    cout << "The test06 copy time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test07() {
    vector<int> s;
    vector<vector<int>> vec;
    for(int i=0;i<set_num;i++){
        vector<int> s_i(set_size);
        for(int j=0;j<set_size;j++){
            s_i[j]=j;
        }
        vec.push_back(s_i);
    }

    clock_t startTime = clock();
    for(int i=0;i<set_num;i++){
        s.insert(s.end(),vec[i].begin(),vec[i].end());
    }
    set<int> set_tmp;
    set_tmp.insert(s.begin(),s.end());
    clock_t endTime = clock();
    cout << "The test07 copy time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}


void test08() {
    // test map
    map<int,int> map_tmp;
    clock_t startTime = clock();

    for(int i=0;i<1000000;i++){
        map_tmp.insert(pair<int,int>(i,i));
    }

    clock_t endTime = clock();
    cout << "The test08 build time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test09() {
    // test map
    map<int,int> map_tmp;
    clock_t startTime = clock();

    for(int i=0;i<1000000;i++){
        map_tmp[i]=i;
    }

    clock_t endTime = clock();
    cout << "The test09 build time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test10() {
    // test map
    set<int> set_tmp;
    clock_t startTime = clock();

    for(int i=0;i<1000000;i++){
        set_tmp.insert(i);
    }

    clock_t endTime = clock();
    cout << "The test10 build time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}

void test11() {
    // test map
    vector<int> vec_tmp(1000000);
    clock_t startTime = clock();

    for(int i=0;i<1000000;i++){
        vec_tmp[i]=i;
    }

    clock_t endTime = clock();
    cout << "The test11 build time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
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
//    test10();
    test11();
}



