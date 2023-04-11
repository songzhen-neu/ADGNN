

#include "hashset.h"
template<class hash_type>

int hashset<hash_type>::hash_fun(hash_type original)
{
    return ((int)original) % MAX_LENGTH;
}
template<class hash_type>
hashset<hash_type>::hashset()
{
    array.reserve(MAX_LENGTH);
    for(uint i = 0; i < MAX_LENGTH; i++)
        array[i] = NULL;
}

template<class hash_type>
hashset<hash_type>::hashset(int size)
{
    MAX_LENGTH=size;
    array.reserve(MAX_LENGTH);
    for(uint i = 0; i < MAX_LENGTH; i++)
        array[i] = NULL;
}

template<class hash_type>
bool hashset<hash_type>::contain(hash_type query)
{
    int hash_value = hash_fun(query);
    while(array[hash_value] != NULL)
    {
        if(array[hash_value] == query)
            return true;
        hash_value++;
        if(hash_value >= MAX_LENGTH)
            hash_value = 0;
    }
    return false;
}
template<class hash_type>
void hashset<hash_type>::insert(hash_type value)
{
    if(contain(value))
    {
//        cout << "The value exists.\n";
        return;
    }
    int hash_value = hash_fun(value);
    while(array[hash_value] != NULL)
    {
        hash_value++;
        if(hash_value >= MAX_LENGTH)
            hash_value = 0;
    }
    array[hash_value] = value;
}
template<class hash_type>
void hashset<hash_type>::erase(hash_type target) {
    int hash_value = hash_fun(target);
    while (array[hash_value] != NULL) {
        if (array[hash_value] == target)
            break;
        hash_value++;
        if (hash_value >= MAX_LENGTH)
            hash_value = 0;
    }
    if (array[hash_value] == NULL)
        cout << "The value doesn't exist.\n";
    else
        array[hash_value] = NULL;

}