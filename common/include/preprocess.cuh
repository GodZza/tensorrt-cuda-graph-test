#pragma once
#include <cuda_runtime.h>
#include "types.h"

namespace yolo {

void preprocess_gpu(
    const uint8_t* d_src_images,
    float* d_dst_tensor,
    int batch_size,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    ImageInfo* d_image_infos,
    cudaStream_t stream
);

void preprocess_gpu_with_infos(
    const uint8_t* d_src_images,
    float* d_dst_tensor,
    int batch_size,
    int dst_width,
    int dst_height,
    ImageInfo* d_image_infos,
    cudaStream_t stream
);

void preprocess_single_gpu(
    const uint8_t* d_src_image,
    float* d_dst_tensor,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    ImageInfo& image_info,
    cudaStream_t stream
);

void cuda_preprocess_init();
void cuda_preprocess_cleanup();

}
