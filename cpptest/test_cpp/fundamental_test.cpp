

#include<vector>
#include<time.h>
#include <iostream>

using namespace std;

static int max_size = 10000000;

void test01() {
    vector<float> tmp(max_size);
    for (int i = 0; i < tmp.size(); i++) {
        if (tmp[i] != 0) {
            cout << tmp[i] << endl;
        }
    }
}

void test02() {
    float tmp[1000000];
    for (int i = 0; i < sizeof(tmp) / sizeof(float); i++) {
        if (tmp[i] != 0) {
            cout << tmp[i] << endl;
        }
    }
}

void test03() {
    vector<float> tmp(max_size);
    vector<float> tmp2;
    vector<float> *tmp3 = nullptr;

    cout << tmp.empty() << "," << tmp2.empty() << "," << tmp3 << endl;
}

int main() {
//    test01();
//    test02();
    test03();
    return 0;
}