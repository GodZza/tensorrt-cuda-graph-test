#pragma once
#include <vector>
#include <string>
#include <memory>
#include "../../common/include/types.h"
#include "../../common/include/cuda_utils.h"
#include "../../common/include/trt_engine.h"

namespace yolo {

class YoloPoseDetector {
public:
    YoloPoseDetector();
    ~YoloPoseDetector();
    
    bool init(const InferConfig& config);
    
    std::vector<std::vector<PoseResult>> infer(
        const std::vector<std::vector<uint8_t>>& images,
        int src_width,
        int src_height);
    
    void infer_async(
        const std::vector<std::vector<uint8_t>>& images,
        int src_width,
        int src_height,
        std::vector<std::vector<PoseResult>>& results);
    
    void benchmark(
        const std::vector<std::vector<uint8_t>>& images,
        int src_width,
        int src_height,
        int iterations = 100);
    
private:
    bool allocate_buffers(int max_batch_size);
    
    InferConfig config_;
    std::unique_ptr<TrtEngine> engine_;
    std::unique_ptr<CudaStream> stream_;
    
    PinnedMemory pinned_input_;
    GpuMemory d_input_images_;
    GpuMemory d_image_infos_;
    GpuMemory d_results_;
    GpuMemory d_num_detections_;
    
    std::vector<ImageInfo> h_image_infos_;
    std::vector<PoseResult> h_results_;
    std::vector<int> h_num_detections_;
};

}
