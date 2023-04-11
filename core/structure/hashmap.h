
#ifndef HASHMAP_HASHMAP_H
#define HASHMAP_HASHMAP_H

#include <iostream>
using namespace std;

template<class Key, class Value>
class HashNode
{
public:
    Key    _key;
    Value  _value;
    HashNode *next;

    HashNode(Key key, Value value)
    {
        _key = key;
        _value = value;
        next = NULL;
    }
    ~HashNode()
    {

    }
    HashNode& operator=(const HashNode& node)
    {
        _key  = node._key;
        _value = node._value;
        next = node.next;
        return *this;
    }
};

template <class Key, class Value>
class hashmap
{
public:
    hashmap();
    hashmap(int size);
    ~hashmap();
    bool insert(const Key& key, const Value& value);
    bool del(const Key& key);
    Value& find(const Key& key);
    Value& operator [](const Key& key);

private:
    HashNode<Key, Value> **table;
    unsigned int _size;
    Value ValueNULL;
};




#endif //HASHMAP_HASHMAP_H
