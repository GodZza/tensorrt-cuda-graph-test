#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                 \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

namespace yolo {

inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

inline dim3 get_grid_size(int total_threads, int block_size) {
    return dim3(div_up(total_threads, block_size));
}

inline dim3 get_grid_size_2d(int width, int height, int block_x, int block_y) {
    return dim3(div_up(width, block_x), div_up(height, block_y));
}

class PinnedMemory {
public:
    PinnedMemory() : ptr_(nullptr), size_(0) {}
    ~PinnedMemory() { release(); }
    
    void* allocate(size_t size) {
        release();
        CUDA_CHECK(cudaMallocHost(&ptr_, size));
        size_ = size;
        return ptr_;
    }
    
    void release() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    void* ptr_;
    size_t size_;
};

class GpuMemory {
public:
    GpuMemory() : ptr_(nullptr), size_(0) {}
    ~GpuMemory() { release(); }
    
    void* allocate(size_t size) {
        release();
        CUDA_CHECK(cudaMalloc(&ptr_, size));
        size_ = size;
        return ptr_;
    }
    
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    void* ptr_;
    size_t size_;
};

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    
    cudaStream_t get() const { return stream_; }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    
private:
    cudaStream_t stream_;
};

}
