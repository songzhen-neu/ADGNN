

#include<vector>
#include<time.h>
#include <iostream>

using namespace std;

int max_val = 10000000;

//1、声明变量，使用pushback
void test01() {
    vector<int> vec;
    clock_t startTime = clock();
    for (int i = 0; i < max_val; i++) {
        vec.push_back(i);
    }
    clock_t endTime = clock();
    cout << "The test01 time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}



//	4、声明变量同时分配空间，使用数组下标
void test02() {
    vector<int> vec(max_val);
    clock_t startTime = clock();
    for (int i = 0; i < max_val; i++) {
        vec[i] = i;
    }
    clock_t endTime = clock();
    cout << "The test02 time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}


void test03() {
    auto vec_ptr=(int *)malloc(sizeof(int)*max_val);
    clock_t startTime = clock();
    for (int i = 0; i < max_val; i++) {
        vec_ptr[i] = i;
    }
    clock_t endTime = clock();
    cout << "The test03 time is: " << (double) (endTime - startTime)/1000000 << "s" << endl;
}


void test04() {
    vector<int> vec(10);
    for(int i=0;i<5;i++){
        vec.emplace_back(i);
    }
    cout<<vec.size()<<endl;
}

void test05() {
    vector<int> vec(10);
    for(int i=0;i<5;i++){
        vec.push_back(i);
    }
    cout<<vec.size()<<endl;
}




int main() {
//    test01();
//    test02();
//    test03();
test04();
}



