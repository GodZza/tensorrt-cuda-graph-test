#pragma once
#include <cuda_runtime.h>
#include "types.h"

namespace yolo {

void postprocess_gpu(
    const float* d_output,
    PoseResult* d_results,
    int* d_num_detections,
    int batch_size,
    int output_size,
    int input_width,
    int input_height,
    const ImageInfo* d_image_infos,
    float conf_threshold,
    float nms_threshold,
    int max_detections,
    cudaStream_t stream
);

void cuda_postprocess_init(int max_detections, int max_batch_size);
void cuda_postprocess_cleanup();

}
