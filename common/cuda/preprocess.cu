#include "preprocess.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>

namespace yolo {

__constant__ float c_mean[3] = {0.0f, 0.0f, 0.0f};
__constant__ float c_std[3] = {255.0f, 255.0f, 255.0f};

__device__ __forceinline__ float bilinear_interpolate(
    const uint8_t* src, int src_width, int src_height,
    float x, float y, int c) {
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    x0 = max(0, min(x0, src_width - 1));
    x1 = max(0, min(x1, src_width - 1));
    y0 = max(0, min(y0, src_height - 1));
    y1 = max(0, min(y1, src_height - 1));
    
    float dx = x - x0;
    float dy = y - y0;
    
    float v00 = static_cast<float>(src[(y0 * src_width + x0) * 3 + c]);
    float v01 = static_cast<float>(src[(y0 * src_width + x1) * 3 + c]);
    float v10 = static_cast<float>(src[(y1 * src_width + x0) * 3 + c]);
    float v11 = static_cast<float>(src[(y1 * src_width + x1) * 3 + c]);
    
    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    
    return v0 * (1.0f - dy) + v1 * dy;
}

__global__ void preprocess_kernel(
    const uint8_t* __restrict__ d_src,
    float* __restrict__ d_dst,
    int src_width, int src_height,
    int dst_width, int dst_height,
    int batch_idx,
    ImageInfo* image_info) {
    
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dx >= dst_width || dy >= dst_height) return;
    
    float scale = min(
        static_cast<float>(dst_width) / src_width,
        static_cast<float>(dst_height) / src_height
    );
    
    int new_width = __float2int_rn(src_width * scale);
    int new_height = __float2int_rn(src_height * scale);
    
    int pad_x = (dst_width - new_width) / 2;
    int pad_y = (dst_height - new_height) / 2;
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && dx == 0 && dy == 0) {
        image_info[batch_idx].orig_width = src_width;
        image_info[batch_idx].orig_height = src_height;
        image_info[batch_idx].scale_x = 1.0f / scale;
        image_info[batch_idx].scale_y = 1.0f / scale;
        image_info[batch_idx].pad_x = pad_x;
        image_info[batch_idx].pad_y = pad_y;
    }
    
    int dst_idx = batch_idx * dst_width * dst_height * 3;
    
    if (dx >= pad_x && dx < pad_x + new_width &&
        dy >= pad_y && dy < pad_y + new_height) {
        
        float src_x = (dx - pad_x) / scale;
        float src_y = (dy - pad_y) / scale;
        
        for (int c = 0; c < 3; c++) {
            float val = bilinear_interpolate(d_src, src_width, src_height, src_x, src_y, c);
            int out_c = 2 - c;
            d_dst[dst_idx + out_c * dst_width * dst_height + dy * dst_width + dx] = 
                (val - c_mean[c]) / c_std[c];
        }
    } else {
        for (int c = 0; c < 3; c++) {
            d_dst[dst_idx + c * dst_width * dst_height + dy * dst_width + dx] = 0.0f;
        }
    }
}

__global__ void preprocess_batch_kernel(
    const uint8_t* __restrict__ d_src_images,
    float* __restrict__ d_dst_tensor,
    int batch_size,
    int src_width, int src_height,
    int dst_width, int dst_height,
    ImageInfo* __restrict__ d_image_infos) {
    
    int batch_idx = blockIdx.z;
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || dx >= dst_width || dy >= dst_height) return;
    
    const uint8_t* d_src = d_src_images + batch_idx * src_width * src_height * 3;
    
    float scale = min(
        static_cast<float>(dst_width) / src_width,
        static_cast<float>(dst_height) / src_height
    );
    
    int new_width = __float2int_rn(src_width * scale);
    int new_height = __float2int_rn(src_height * scale);
    
    int pad_x = (dst_width - new_width) / 2;
    int pad_y = (dst_height - new_height) / 2;
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && dx == 0 && dy == 0) {
        d_image_infos[batch_idx].orig_width = src_width;
        d_image_infos[batch_idx].orig_height = src_height;
        d_image_infos[batch_idx].scale_x = 1.0f / scale;
        d_image_infos[batch_idx].scale_y = 1.0f / scale;
        d_image_infos[batch_idx].pad_x = pad_x;
        d_image_infos[batch_idx].pad_y = pad_y;
    }
    
    int dst_idx = batch_idx * dst_width * dst_height * 3;
    
    if (dx >= pad_x && dx < pad_x + new_width &&
        dy >= pad_y && dy < pad_y + new_height) {
        
        float src_x = (dx - pad_x) / scale;
        float src_y = (dy - pad_y) / scale;
        
        for (int c = 0; c < 3; c++) {
            float val = bilinear_interpolate(d_src, src_width, src_height, src_x, src_y, c);
            int out_c = 2 - c;
            d_dst_tensor[dst_idx + out_c * dst_width * dst_height + dy * dst_width + dx] = 
                (val - c_mean[c]) / c_std[c];
        }
    } else {
        for (int c = 0; c < 3; c++) {
            d_dst_tensor[dst_idx + c * dst_width * dst_height + dy * dst_width + dx] = 0.0f;
        }
    }
}

void preprocess_gpu(
    const uint8_t* d_src_images,
    float* d_dst_tensor,
    int batch_size,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    ImageInfo* d_image_infos,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid(div_up(dst_width, block.x), div_up(dst_height, block.y), batch_size);
    
    preprocess_batch_kernel<<<grid, block, 0, stream>>>(
        d_src_images, d_dst_tensor, batch_size,
        src_width, src_height, dst_width, dst_height, d_image_infos);
}

void preprocess_single_gpu(
    const uint8_t* d_src_image,
    float* d_dst_tensor,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    ImageInfo& image_info,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid(div_up(dst_width, block.x), div_up(dst_height, block.y));
    
    ImageInfo* d_image_info;
    CUDA_CHECK(cudaMalloc(&d_image_info, sizeof(ImageInfo)));
    
    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_src_image, d_dst_tensor, src_width, src_height,
        dst_width, dst_height, 0, d_image_info);
    
    CUDA_CHECK(cudaMemcpyAsync(&image_info, d_image_info, sizeof(ImageInfo),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_image_info));
}

void cuda_preprocess_init() {}

void cuda_preprocess_cleanup() {}

}
