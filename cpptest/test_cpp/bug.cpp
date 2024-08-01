

#include <iostream>
#include <unistd.h>

int main(){
    while(1) {
        char *p = new char();
        delete p;
        p = nullptr;
        *p = '2';
        sleep(1);
    }

    return 0;
}