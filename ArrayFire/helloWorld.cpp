#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv){
    try {
        af::array rand1 = af::randu(1, 4);
        af_print(rand1);
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}