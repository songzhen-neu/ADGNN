

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
    hashset();//构造函数
    hashset(int size);//构造函数
    void insert(hash_type value);//插入一个元素
    void erase(hash_type target);//删除一个元素
    bool contain(hash_type query);//判断一个元素是否在集合中
};


#endif //EC_GRAPH_HASHSET_H
