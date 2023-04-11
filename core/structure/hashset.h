

#ifndef EC_GRAPH_HASHSET_H
#define EC_GRAPH_HASHSET_H


#include<iostream>
#include <vector>
using namespace std;
template<class hash_type>
class hashset
{
private:
    vector<hash_type> array;
    int MAX_LENGTH=1000000;
    int hash_fun(hash_type original);
public:
    hashset();
    hashset(int size);
    void insert(hash_type value);
    void erase(hash_type target);
    bool contain(hash_type query);
};


#endif //EC_GRAPH_HASHSET_H
