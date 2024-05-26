#include <stdio.h>
#include <vector>

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


struct HostDevice {
    using T = float;
    using value_type = T;

    HostDevice(int const N)
    :
    N_{N}
    {
        host_ = reinterpret_cast<T*>( malloc( bytes() ) );
        cudaMalloc(&dev_, bytes() );
    }

    ~HostDevice() {
        free(host_);
        host_ = nullptr;
        cudaFree(dev_);
        dev_ = nullptr;
    }

    void copyHostToDevice() {
        cudaMemcpy(dev_, host_, bytes(), cudaMemcpyHostToDevice );
    }

    void copyDeviceToHost() {
        cudaMemcpy(host_, dev_, bytes(), cudaMemcpyDeviceToHost );
    }

    size_t bytes() const { return N_ * sizeof(value_type); }

    void fill(value_type const& value) {
        std::fill(host_, host_ + N, value);
    }

    int N_{0};
    T* host_{nullptr};
    T* dev_{nullptr};
};



int main(int argc, char* argv[]) {
    using namespace std;

    HostDevice a(N);
    HostDevice b(N);
    HostDevice out(N);

    a.fill(1.0f);
    b.fill(2.0f);

    a.copyHostToDevice();
    b.copyHostToDevice();


    vector_add<<<1,1>>>(a.dev_, b.dev_, out.dev_, N);

    out.copyDeviceToHost();

    return 0;
}


