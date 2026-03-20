#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace yolo {

constexpr int NUM_KEYPOINTS = 17;
constexpr int KEYPOINT_DIM = 3;
constexpr int MAX_DETECTIONS = 100;

struct BBox {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

struct KeyPoint {
    float x, y;
    float conf;
};

struct PoseResult {
    BBox bbox;
    KeyPoint keypoints[NUM_KEYPOINTS];
};

struct InferConfig {
    int input_width = 640;
    int input_height = 640;
    int max_batch_size = 16;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = MAX_DETECTIONS;
    int max_detections_to_copy = 15;
    bool use_fp16 = true;
    bool use_zero_copy = false;
    std::string engine_path;
};

struct ImageInfo {
    int src_width;
    int src_height;
    int orig_width;
    int orig_height;
    float scale_x;
    float scale_y;
    int pad_x;
    int pad_y;
    size_t data_offset;
};

struct GpuTimer {
    void* start;
    void* stop;
    
    GpuTimer();
    ~GpuTimer();
    void start_timer();
    void stop_timer();
    float elapsed_ms();
};

}
