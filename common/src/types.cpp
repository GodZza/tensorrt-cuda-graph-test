#include "types.h"
#include "cuda_utils.h"

namespace yolo {

GpuTimer::GpuTimer() : start(nullptr), stop(nullptr) {
    CUDA_CHECK(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&start)));
    CUDA_CHECK(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&stop)));
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(start));
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(stop));
}

void GpuTimer::start_timer() {
    CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(start)));
}

void GpuTimer::stop_timer() {
    CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(stop)));
    CUDA_CHECK(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(stop)));
}

float GpuTimer::elapsed_ms() {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, 
        reinterpret_cast<cudaEvent_t>(start), 
        reinterpret_cast<cudaEvent_t>(stop)));
    return ms;
}

}
