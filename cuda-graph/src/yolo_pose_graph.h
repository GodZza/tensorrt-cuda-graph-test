#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <future>
#include <mutex>
#include "../../common/include/types.h"
#include "../../common/include/cuda_utils.h"
#include "../../common/include/trt_engine.h"

namespace yolo {

struct CudaGraphInstance {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool initialized = false;
};

struct BufferHandle {
    int id;
    PinnedMemory pinned_input;
    GpuMemory d_input_images;
    GpuMemory d_image_infos;
    GpuMemory d_results;
    GpuMemory d_num_detections;
    GpuMemory d_input_tensor;
    GpuMemory d_output_tensor;
    nvinfer1::IExecutionContext* context = nullptr;
    std::vector<ImageInfo> h_image_infos;
    std::vector<PoseResult> h_results;
    std::vector<int> h_num_detections;
    std::unique_ptr<CudaStream> stream;
    bool in_use = false;
};

class YoloPoseDetectorGraph {
public:
    YoloPoseDetectorGraph();
    ~YoloPoseDetectorGraph();
    
    bool init(const InferConfig& config);
    
    std::vector<std::vector<PoseResult>> infer(
        const std::vector<std::vector<uint8_t>>& images,
        int src_width,
        int src_height);
    
    std::vector<std::vector<PoseResult>> infer_batch(
        const std::vector<std::vector<uint8_t>>& images,
        const std::vector<std::pair<int, int>>& image_sizes);
    
    std::future<std::vector<std::vector<PoseResult>>> infer_batch_async(
        const std::vector<std::vector<uint8_t>>& images,
        const std::vector<std::pair<int, int>>& image_sizes);
    
    // Pipeline API
    std::shared_ptr<BufferHandle> create_buffer();
    void prepare_async(
        const std::vector<std::vector<uint8_t>>& images,
        const std::vector<std::pair<int, int>>& image_sizes,
        std::shared_ptr<BufferHandle> buffer);
    std::vector<std::vector<PoseResult>> wait_and_get_results(
        std::shared_ptr<BufferHandle> buffer);
    
    void benchmark(
        const std::vector<std::vector<uint8_t>>& images,
        int src_width,
        int src_height,
        int iterations = 100);
    
private:
    bool allocate_buffers(int max_batch_size);
    bool build_cuda_graph(int batch_size);
    int get_graph_batch_size(int actual_batch_size);
    
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
    
    std::unordered_map<int, CudaGraphInstance> graph_pool_;
    std::vector<int> supported_batch_sizes_;
    
    std::mutex infer_mutex_;
    int next_buffer_id_ = 0;
    
    static constexpr int BATCH_SIZES[] = {1, 2, 3, 4};
};

}
