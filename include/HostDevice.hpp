
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


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
        std::fill(host(), host() + N(), value);
    }

    int N() const { return N_; }
    value_type* host() { return host_; }
    value_type* dev() { return dev_; }

protected:

    int N_{0};
    T* host_{nullptr};
    T* dev_{nullptr};
};