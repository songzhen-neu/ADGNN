
#include <iostream>
#include <vector>
using namespace std;

int main(){
    vector<vector<int>> a(10);
    for(int i=0;i<10;i++){
        vector<int> a_tmp(100);
        a[i]=a_tmp;
    }
    int* a_ptr=&a[1][0];
    vector<int> b(100);
    for(int i=0;i<b.size();i++){
        b[i]=i;
    }
    copy(b.begin(),b.end(),a_ptr);
    cout<<"a"<<endl;
}
