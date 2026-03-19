#include "postprocess.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace yolo {

static int g_max_detections = 100;
static int g_max_batch_size = 16;
static float* g_d_boxes = nullptr;
static float* g_d_scores = nullptr;
static int* g_d_class_ids = nullptr;
static float* g_d_keypoints = nullptr;
static int* g_d_valid_count = nullptr;
static int* g_d_keep_indices = nullptr;
static int* g_d_num_keep = nullptr;
static float* g_d_sorted_scores = nullptr;
static int* g_d_sorted_indices = nullptr;

__constant__ int c_max_detections = 100;

__device__ float iou_device(const float* box1, const float* box2) {
    float x1 = max(box1[0], box2[0]);
    float y1 = max(box1[1], box2[1]);
    float x2 = min(box1[2], box2[2]);
    float y2 = min(box1[3], box2[3]);
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;
    
    return intersection / (union_area + 1e-6f);
}

__global__ void decode_output_kernel(
    const float* __restrict__ d_output,
    float* __restrict__ d_boxes,
    float* __restrict__ d_scores,
    int* __restrict__ d_class_ids,
    float* __restrict__ d_keypoints,
    int* __restrict__ d_valid_count,
    int batch_size,
    int num_preds,
    int num_attrs,
    int num_keypoints,
    float conf_threshold) {
    
    int batch_idx = blockIdx.y;
    int pred_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || pred_idx >= num_preds) return;
    
    const float* output = d_output + batch_idx * num_preds * num_attrs;
    
    float obj_score = output[4 * num_preds + pred_idx];
    
    if (obj_score < conf_threshold) return;
    
    int valid_idx = atomicAdd(&d_valid_count[batch_idx], 1);
    
    if (valid_idx >= c_max_detections) return;
    
    int out_base = batch_idx * c_max_detections + valid_idx;
    
    float cx = output[0 * num_preds + pred_idx];
    float cy = output[1 * num_preds + pred_idx];
    float w = output[2 * num_preds + pred_idx];
    float h = output[3 * num_preds + pred_idx];
    
    d_boxes[out_base * 4 + 0] = cx - w / 2;
    d_boxes[out_base * 4 + 1] = cy - h / 2;
    d_boxes[out_base * 4 + 2] = cx + w / 2;
    d_boxes[out_base * 4 + 3] = cy + h / 2;
    
    d_scores[out_base] = obj_score;
    d_class_ids[out_base] = 0;
    
    for (int k = 0; k < num_keypoints; k++) {
        int kp_base = 5 + k * 3;
        d_keypoints[out_base * num_keypoints * 3 + k * 3 + 0] = output[kp_base * num_preds + pred_idx];
        d_keypoints[out_base * num_keypoints * 3 + k * 3 + 1] = output[(kp_base + 1) * num_preds + pred_idx];
        d_keypoints[out_base * num_keypoints * 3 + k * 3 + 2] = output[(kp_base + 2) * num_preds + pred_idx];
    }
}

__global__ void nms_kernel(
    float* __restrict__ d_boxes,
    float* __restrict__ d_scores,
    int* __restrict__ d_keep_indices,
    int* __restrict__ d_num_keep,
    int batch_size,
    int num_boxes,
    float nms_threshold) {
    
    int batch_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || i >= num_boxes) return;
    
    if (d_scores[batch_idx * c_max_detections + i] <= 0.0f) return;
    
    float* box_i = &d_boxes[(batch_idx * c_max_detections + i) * 4];
    float score_i = d_scores[batch_idx * c_max_detections + i];
    
    for (int j = 0; j < num_boxes; j++) {
        if (i == j) continue;
        
        float score_j = d_scores[batch_idx * c_max_detections + j];
        if (score_j <= 0.0f) continue;
        
        if (score_j > score_i) {
            float* box_j = &d_boxes[(batch_idx * c_max_detections + j) * 4];
            float iou = iou_device(box_i, box_j);
            if (iou > nms_threshold) {
                d_scores[batch_idx * c_max_detections + i] = -1.0f;
                return;
            }
        }
    }
    
    int keep_idx = atomicAdd(&d_num_keep[batch_idx], 1);
    d_keep_indices[batch_idx * c_max_detections + keep_idx] = i;
}

__global__ void gather_results_kernel(
    const float* __restrict__ d_boxes,
    const float* __restrict__ d_scores,
    const int* __restrict__ d_class_ids,
    const float* __restrict__ d_keypoints,
    const int* __restrict__ d_keep_indices,
    const int* __restrict__ d_num_keep,
    PoseResult* __restrict__ d_results,
    int* __restrict__ d_num_detections,
    int batch_size,
    int num_keypoints,
    int max_detections) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int num_keep = min(d_num_keep[batch_idx], max_detections);
    d_num_detections[batch_idx] = num_keep;
    
    for (int i = 0; i < num_keep; i++) {
        int idx = d_keep_indices[batch_idx * c_max_detections + i];
        
        PoseResult& result = d_results[batch_idx * max_detections + i];
        
        result.bbox.x1 = d_boxes[(batch_idx * c_max_detections + idx) * 4 + 0];
        result.bbox.y1 = d_boxes[(batch_idx * c_max_detections + idx) * 4 + 1];
        result.bbox.x2 = d_boxes[(batch_idx * c_max_detections + idx) * 4 + 2];
        result.bbox.y2 = d_boxes[(batch_idx * c_max_detections + idx) * 4 + 3];
        result.bbox.conf = d_scores[batch_idx * c_max_detections + idx];
        result.bbox.class_id = d_class_ids[batch_idx * c_max_detections + idx];
        
        for (int k = 0; k < num_keypoints; k++) {
            result.keypoints[k].x = d_keypoints[(batch_idx * c_max_detections + idx) * num_keypoints * 3 + k * 3 + 0];
            result.keypoints[k].y = d_keypoints[(batch_idx * c_max_detections + idx) * num_keypoints * 3 + k * 3 + 1];
            result.keypoints[k].conf = d_keypoints[(batch_idx * c_max_detections + idx) * num_keypoints * 3 + k * 3 + 2];
        }
    }
}

__global__ void scale_coords_kernel(
    PoseResult* __restrict__ d_results,
    int* __restrict__ d_num_detections,
    const ImageInfo* __restrict__ d_image_infos,
    int batch_size,
    int max_detections) {
    
    int batch_idx = blockIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || det_idx >= max_detections) return;
    
    int num_det = d_num_detections[batch_idx];
    if (det_idx >= num_det) return;
    
    const ImageInfo& info = d_image_infos[batch_idx];
    PoseResult& result = d_results[batch_idx * max_detections + det_idx];
    
    result.bbox.x1 = (result.bbox.x1 - info.pad_x) * info.scale_x;
    result.bbox.y1 = (result.bbox.y1 - info.pad_y) * info.scale_y;
    result.bbox.x2 = (result.bbox.x2 - info.pad_x) * info.scale_x;
    result.bbox.y2 = (result.bbox.y2 - info.pad_y) * info.scale_y;
    
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
        result.keypoints[k].x = (result.keypoints[k].x - info.pad_x) * info.scale_x;
        result.keypoints[k].y = (result.keypoints[k].y - info.pad_y) * info.scale_y;
    }
}

void cuda_postprocess_init(int max_detections, int max_batch_size) {
    g_max_detections = max_detections;
    g_max_batch_size = max_batch_size;
    
    CUDA_CHECK(cudaMalloc(&g_d_boxes, max_batch_size * max_detections * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_scores, max_batch_size * max_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_class_ids, max_batch_size * max_detections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_keypoints, max_batch_size * max_detections * NUM_KEYPOINTS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_valid_count, max_batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_keep_indices, max_batch_size * max_detections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_num_keep, max_batch_size * sizeof(int)));
}

void cuda_postprocess_cleanup() {
    if (g_d_boxes) { cudaFree(g_d_boxes); g_d_boxes = nullptr; }
    if (g_d_scores) { cudaFree(g_d_scores); g_d_scores = nullptr; }
    if (g_d_class_ids) { cudaFree(g_d_class_ids); g_d_class_ids = nullptr; }
    if (g_d_keypoints) { cudaFree(g_d_keypoints); g_d_keypoints = nullptr; }
    if (g_d_valid_count) { cudaFree(g_d_valid_count); g_d_valid_count = nullptr; }
    if (g_d_keep_indices) { cudaFree(g_d_keep_indices); g_d_keep_indices = nullptr; }
    if (g_d_num_keep) { cudaFree(g_d_num_keep); g_d_num_keep = nullptr; }
}

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
    cudaStream_t stream) {
    
    int num_preds = 8400;
    int num_attrs = 56;
    
    CUDA_CHECK(cudaMemsetAsync(g_d_valid_count, 0, batch_size * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(g_d_num_keep, 0, batch_size * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(g_d_scores, 0, batch_size * max_detections * sizeof(float), stream));
    
    dim3 decode_block(256);
    dim3 decode_grid(div_up(num_preds, 256), batch_size);
    
    decode_output_kernel<<<decode_grid, decode_block, 0, stream>>>(
        d_output, g_d_boxes, g_d_scores, g_d_class_ids, g_d_keypoints,
        g_d_valid_count, batch_size, num_preds, num_attrs, NUM_KEYPOINTS,
        conf_threshold);
    
    dim3 nms_block(256);
    dim3 nms_grid(div_up(max_detections, 256), batch_size);
    
    nms_kernel<<<nms_grid, nms_block, 0, stream>>>(
        g_d_boxes, g_d_scores, g_d_keep_indices, g_d_num_keep,
        batch_size, max_detections, nms_threshold);
    
    dim3 gather_block(1);
    dim3 gather_grid(batch_size);
    
    gather_results_kernel<<<gather_grid, gather_block, 0, stream>>>(
        g_d_boxes, g_d_scores, g_d_class_ids, g_d_keypoints,
        g_d_keep_indices, g_d_num_keep, d_results, d_num_detections,
        batch_size, NUM_KEYPOINTS, max_detections);
    
    dim3 scale_block(32);
    dim3 scale_grid(div_up(max_detections, 32), batch_size);
    
    scale_coords_kernel<<<scale_grid, scale_block, 0, stream>>>(
        d_results, d_num_detections, d_image_infos,
        batch_size, max_detections);
}

}
