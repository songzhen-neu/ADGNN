
#include <iostream>
#include <math.h>
#define COUNT 1000000000
using namespace std;



void test01(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=sqrt(i);
    }
    clock_t endTime = clock();
    cout << "The sqrt time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test02(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=i+i;
    }
    clock_t endTime = clock();
    cout << "The + time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test03(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=i-i;
    }
    clock_t endTime = clock();
    cout << "The - time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test04(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=i*i;
    }
    clock_t endTime = clock();
    cout << "The * time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test05(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=(double)i/ (double)i;
    }
    clock_t endTime = clock();
    cout << "The / time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test06(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=pow(i,2);
    }
    clock_t endTime = clock();
    cout << "The pow2 time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test07(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=pow(i,6);
    }
    clock_t endTime = clock();
    cout << "The pow6 time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test08(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=i;
        for(int j=0;j<5;j++){
            tmp*=i;
        }
    }
    clock_t endTime = clock();
    cout << "The pow6 time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}

void test09(){
    clock_t startTime=clock();
    for(int i=0;i<COUNT;i++){
        double tmp=abs(i);
    }
    clock_t endTime = clock();
    cout << "The pow6 time is: " << (double) (endTime - startTime) / 1000000 << "s" << endl;
}




int main(){
//    test01();
//    test02();
//    test03();
//    test04();
//    test05();
//    test06();
//    test07();
//    test08();
    test09();
}