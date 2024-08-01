
#include<iostream>
#include <vector>

using namespace std;

int main() {
    vector<vector<float>> a(5);
    for (int i = 0; i < 5; i++) {
        vector<float> tmp(5);
        for (int j = 0; j < 5; j++) {
            tmp[j] = j;
        }
        a[i] = tmp;
    }

//    for(int i=0;i<5;i++){
//        for(int j=0;j<5;j++){
//            cout<<&a[i][j]<<endl;
//        }
//    }

    cout << a[4][600] << endl;


//    cout<<*((a.begin()+3)->end())<<endl;
//    for(auto elem=a.begin();elem<a.end();elem++){
//        a[i]=i;
//        cout<<*elem<<endl;
//    }
}