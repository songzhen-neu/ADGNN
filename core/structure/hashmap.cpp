

#include "hashmap.h"


template <class Key, class Value>
hashmap<Key, Value>::hashmap() : _size(1000000)
{
    table = new HashNode<Key, Value>*[_size];
    for (unsigned i = 0; i < _size; i++)
        table[i] = NULL;
}

template <class Key, class Value>
hashmap<Key, Value>::hashmap(int size) : _size(size)
{
    table = new HashNode<Key, Value>*[_size];
    for (unsigned i = 0; i < _size; i++)
        table[i] = NULL;
}



template <class Key, class Value>
hashmap<Key, Value>::~hashmap()
{
    for (unsigned i = 0; i < _size; i++)
    {
        HashNode<Key, Value> *currentNode = table[i];
        while (currentNode)
        {
            HashNode<Key, Value> *temp = currentNode;
            currentNode = currentNode->next;
            delete temp;
        }
    }
    delete table;
}


template <class Key, class Value>
bool hashmap<Key, Value>::insert(const Key& key, const Value& value)
{

    int index = key%_size;
    HashNode<Key, Value> * node = new HashNode<Key, Value>(key,value);
    node->next = table[index];
    table[index] = node;
    return true;
}


template <class Key, class Value>
bool hashmap<Key, Value>::del(const Key& key)
{
    unsigned index = key % _size;
    HashNode<Key, Value> * node = table[index];
    HashNode<Key, Value> * prev = NULL;
    while (node)
    {
        if (node->_key == key)
        {
            if (prev == NULL)
            {
                table[index] = node->next;
            }
            else
            {
                prev->next = node->next;
            }
            delete node;
            return true;
        }
        prev = node;
        node = node->next;
    }
    return false;
}

template <class Key, class Value>
Value& hashmap<Key, Value>::find(const Key& key)
{
    unsigned  index = key % _size;
    if (table[index] == NULL)
        return ValueNULL;
    else
    {
        HashNode<Key, Value> * node = table[index];
        while (node)
        {
            if (node->_key == key)
                return node->_value;
            node = node->next;
        }
    }
}


template <class Key, class Value>
Value& hashmap<Key, Value>::operator [](const Key& key)
{
    return find(key);
}

