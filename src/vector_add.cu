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
    //vector<float> a(N, 1.0f);
    vector<float> b(N, 2.0f);
    vector<float> out(N, 0.0f);
    float* a = reinterpret_cast<float*>( malloc( sizeof(float) * N ) );

    float* dev_a;
    float* dev_b;
    float* dev_out;

    cudaMalloc(&dev_a, sizeof(float) * N);
    cudaMemcpy(dev_a, &a[0], sizeof(float)*N, ::cudaMemcpyHostToDevice);

    cudaMalloc(&dev_b, sizeof(float) * N);
    cudaMemcpy(dev_b, &a[0], sizeof(float)*N, ::cudaMemcpyHostToDevice);

    cudaMalloc(&dev_out, sizeof(float) * N);
    cudaMemcpy(dev_out, &a[0], sizeof(float)*N, ::cudaMemcpyHostToDevice);


    vector_add<<<1,1>>>(&dev_out[0], &dev_a[0], &dev_b[0], N);

    cudaMemcpy(&out[0], dev_out, sizeof(float)*N, ::cudaMemcpyDeviceToHost);


    return 0;
}


