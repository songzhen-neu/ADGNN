
#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main() {
//    vector<float> a(1000000);
//    for(auto i:a){
//        if(i!=0){
//            cout<<i<<",";
//        }
//    }

    // c++ map you get a key which is not in map, the map will return 0 and insert <id,0> into the map
    map<int, int> b;
    cout<<b[2]<<endl;
    cout<<b.size()<<endl;
}