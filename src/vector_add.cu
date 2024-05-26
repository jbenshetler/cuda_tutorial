#include <stdio.h>
#include <vector>

#include "HostDevice.hpp"

constexpr int N = 1'000'000;

void vector_add_cpu(float* out, float* a, float* b, int n) {
    for (int i =0; i<n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_add(float* out, float* a, float* b, int n) {
    for (int i =0; i<n; i++) {
        out[i] = a[i] + b[i];
    }
}



int main(int argc, char* argv[]) {
    using namespace std;

    HostDevice a(N);
    HostDevice b(N);
    HostDevice out(N);

    a.fill(1.0f);
    b.fill(2.0f);

    a.copyHostToDevice();
    b.copyHostToDevice();


    vector_add<<<1,1>>>(a.dev(), b.dev(), out.dev(), N);

    out.copyDeviceToHost();

    return 0;
}


