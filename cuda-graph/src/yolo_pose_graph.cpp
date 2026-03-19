#include "yolo_pose_graph.h"
#include "../../common/include/preprocess.cuh"
#include "../../common/include/postprocess.cuh"
#include <iostream>
#include <algorithm>

namespace yolo {

constexpr int YoloPoseDetectorGraph::BATCH_SIZES[];

YoloPoseDetectorGraph::YoloPoseDetectorGraph() {}

YoloPoseDetectorGraph::~YoloPoseDetectorGraph() {
    for (auto& pair : graph_pool_) {
        if (pair.second.exec) {
            cudaGraphExecDestroy(pair.second.exec);
        }
        if (pair.second.graph) {
            cudaGraphDestroy(pair.second.graph);
        }
    }
    
    cuda_preprocess_cleanup();
    cuda_postprocess_cleanup();
}

bool YoloPoseDetectorGraph::init(const InferConfig& config) {
    config_ = config;
    
    engine_ = std::make_unique<TrtEngine>();
    if (!engine_->load_engine(config.engine_path, config.max_batch_size)) {
        std::cerr << "Failed to load engine: " << config.engine_path << std::endl;
        return false;
    }
    
    stream_ = std::make_unique<CudaStream>();
    
    if (!allocate_buffers(config.max_batch_size)) {
        return false;
    }
    
    cuda_preprocess_init();
    cuda_postprocess_init(config.max_detections, config.max_batch_size);
    
    for (int batch_size : BATCH_SIZES) {
        if (batch_size <= config.max_batch_size) {
            supported_batch_sizes_.push_back(batch_size);
            if (!build_cuda_graph(batch_size)) {
                std::cerr << "Warning: Failed to build CUDA graph for batch " << batch_size << std::endl;
            }
        }
    }
    
    std::cout << "YoloPoseDetector (CUDA Graph) initialized successfully" << std::endl;
    std::cout << "Supported batch sizes with CUDA Graph: ";
    for (int bs : supported_batch_sizes_) {
        if (graph_pool_[bs].initialized) {
            std::cout << bs << " ";
        }
    }
    std::cout << std::endl;
    
    return true;
}

bool YoloPoseDetectorGraph::allocate_buffers(int max_batch_size) {
    int input_image_size = max_batch_size * config_.input_width * config_.input_height * 3;
    
    pinned_input_.allocate(input_image_size * sizeof(uint8_t));
    d_input_images_.allocate(input_image_size * sizeof(uint8_t));
    d_image_infos_.allocate(max_batch_size * sizeof(ImageInfo));
    d_results_.allocate(max_batch_size * config_.max_detections * sizeof(PoseResult));
    d_num_detections_.allocate(max_batch_size * sizeof(int));
    
    h_image_infos_.resize(max_batch_size);
    h_results_.resize(max_batch_size * config_.max_detections);
    h_num_detections_.resize(max_batch_size);
    
    return true;
}

int YoloPoseDetectorGraph::get_graph_batch_size(int actual_batch_size) {
    for (int bs : supported_batch_sizes_) {
        if (bs >= actual_batch_size && graph_pool_[bs].initialized) {
            return bs;
        }
    }
    return actual_batch_size;
}

bool YoloPoseDetectorGraph::build_cuda_graph(int batch_size) {
    std::cout << "Building CUDA Graph for batch size " << batch_size << "..." << std::endl;
    
    CudaGraphInstance& instance = graph_pool_[batch_size];
    
    stream_->synchronize();
    
    engine_->setup_inference(batch_size);
    
    preprocess_gpu_with_infos(
        static_cast<const uint8_t*>(d_input_images_.get()),
        static_cast<float*>(engine_->get_input_buffer()),
        batch_size,
        config_.input_width, config_.input_height,
        static_cast<ImageInfo*>(d_image_infos_.get()),
        stream_->get());
    
    engine_->enqueue_async(stream_->get());
    
    postprocess_gpu(
        static_cast<const float*>(engine_->get_output_buffer()),
        static_cast<PoseResult*>(d_results_.get()),
        static_cast<int*>(d_num_detections_.get()),
        batch_size,
        engine_->get_output_size(),
        config_.input_width,
        config_.input_height,
        static_cast<const ImageInfo*>(d_image_infos_.get()),
        config_.conf_threshold,
        config_.nms_threshold,
        config_.max_detections,
        stream_->get());
    
    stream_->synchronize();
    
    CUDA_CHECK(cudaStreamBeginCapture(stream_->get(), cudaStreamCaptureModeGlobal));
    
    preprocess_gpu_with_infos(
        static_cast<const uint8_t*>(d_input_images_.get()),
        static_cast<float*>(engine_->get_input_buffer()),
        batch_size,
        config_.input_width, config_.input_height,
        static_cast<ImageInfo*>(d_image_infos_.get()),
        stream_->get());
    
    engine_->enqueue_async(stream_->get());
    
    postprocess_gpu(
        static_cast<const float*>(engine_->get_output_buffer()),
        static_cast<PoseResult*>(d_results_.get()),
        static_cast<int*>(d_num_detections_.get()),
        batch_size,
        engine_->get_output_size(),
        config_.input_width,
        config_.input_height,
        static_cast<const ImageInfo*>(d_image_infos_.get()),
        config_.conf_threshold,
        config_.nms_threshold,
        config_.max_detections,
        stream_->get());
    
    CUDA_CHECK(cudaStreamEndCapture(stream_->get(), &instance.graph));
    
    CUDA_CHECK(cudaGraphInstantiate(&instance.exec, instance.graph, NULL, NULL, 0));
    
    instance.initialized = true;
    
    std::cout << "CUDA Graph built for batch size " << batch_size << std::endl;
    return true;
}

std::vector<std::vector<PoseResult>> YoloPoseDetectorGraph::infer(
    const std::vector<std::vector<uint8_t>>& images,
    int src_width,
    int src_height) {
    
    std::vector<std::pair<int, int>> image_sizes(images.size(), {src_width, src_height});
    return infer_batch(images, image_sizes);
}

std::vector<std::vector<PoseResult>> YoloPoseDetectorGraph::infer_batch(
    const std::vector<std::vector<uint8_t>>& images,
    const std::vector<std::pair<int, int>>& image_sizes) {
    
    int actual_batch_size = static_cast<int>(images.size());
    if (actual_batch_size == 0 || actual_batch_size > config_.max_batch_size) {
        return {};
    }
    
    int graph_batch_size = get_graph_batch_size(actual_batch_size);
    bool use_graph = graph_pool_[graph_batch_size].initialized;
    
    size_t total_image_size = 0;
    for (int i = 0; i < actual_batch_size; i++) {
        total_image_size += images[i].size();
    }
    
    uint8_t* pinned_ptr = static_cast<uint8_t*>(pinned_input_.get());
    size_t offset = 0;
    for (int i = 0; i < actual_batch_size; i++) {
        memcpy(pinned_ptr + offset, images[i].data(), images[i].size());
        offset += images[i].size();
    }
    
    for (int i = 0; i < actual_batch_size; i++) {
        h_image_infos_[i].src_width = image_sizes[i].first;
        h_image_infos_[i].src_height = image_sizes[i].second;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_input_images_.get(), pinned_input_.get(),
        total_image_size, cudaMemcpyHostToDevice, stream_->get()));
    
    CUDA_CHECK(cudaMemcpyAsync(d_image_infos_.get(), h_image_infos_.data(),
        actual_batch_size * sizeof(ImageInfo), cudaMemcpyHostToDevice, stream_->get()));
    
    if (use_graph) {
        CUDA_CHECK(cudaGraphLaunch(graph_pool_[graph_batch_size].exec, stream_->get()));
    } else {
        engine_->setup_inference(actual_batch_size);
        
        preprocess_gpu_with_infos(
            static_cast<const uint8_t*>(d_input_images_.get()),
            static_cast<float*>(engine_->get_input_buffer()),
            actual_batch_size,
            config_.input_width, config_.input_height,
            static_cast<ImageInfo*>(d_image_infos_.get()),
            stream_->get());
        
        engine_->enqueue_async(stream_->get());
        
        postprocess_gpu(
            static_cast<const float*>(engine_->get_output_buffer()),
            static_cast<PoseResult*>(d_results_.get()),
            static_cast<int*>(d_num_detections_.get()),
            actual_batch_size,
            engine_->get_output_size(),
            config_.input_width,
            config_.input_height,
            static_cast<const ImageInfo*>(d_image_infos_.get()),
            config_.conf_threshold,
            config_.nms_threshold,
            config_.max_detections,
            stream_->get());
    }
    
    CUDA_CHECK(cudaMemcpyAsync(h_num_detections_.data(), d_num_detections_.get(),
        actual_batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_->get()));
    
    CUDA_CHECK(cudaMemcpyAsync(h_results_.data(), d_results_.get(),
        actual_batch_size * config_.max_detections * sizeof(PoseResult),
        cudaMemcpyDeviceToHost, stream_->get()));
    
    stream_->synchronize();
    
    std::vector<std::vector<PoseResult>> results(actual_batch_size);
    for (int i = 0; i < actual_batch_size; i++) {
        int num_det = h_num_detections_[i];
        results[i].reserve(num_det);
        for (int j = 0; j < num_det; j++) {
            results[i].push_back(h_results_[i * config_.max_detections + j]);
        }
    }
    
    return results;
}

void YoloPoseDetectorGraph::benchmark(
    const std::vector<std::vector<uint8_t>>& images,
    int src_width,
    int src_height,
    int iterations) {
    
    int batch_size = static_cast<int>(images.size());
    int graph_batch_size = get_graph_batch_size(batch_size);
    bool use_graph = graph_pool_[graph_batch_size].initialized;
    
    std::cout << "\n=== Benchmark (CUDA Graph) ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Graph batch size: " << graph_batch_size << std::endl;
    std::cout << "Using CUDA Graph: " << (use_graph ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_h2d = 0.0f;
    float total_graph = 0.0f;
    float total_d2h = 0.0f;
    float total_time = 0.0f;
    
    size_t total_image_size = 0;
    for (int i = 0; i < batch_size; i++) {
        total_image_size += images[i].size();
    }
    
    uint8_t* pinned_ptr = static_cast<uint8_t*>(pinned_input_.get());
    size_t offset = 0;
    for (int i = 0; i < batch_size; i++) {
        memcpy(pinned_ptr + offset, images[i].data(), images[i].size());
        offset += images[i].size();
    }
    
    for (int i = 0; i < batch_size; i++) {
        h_image_infos_[i].src_width = src_width;
        h_image_infos_[i].src_height = src_height;
    }
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        CUDA_CHECK(cudaMemcpyAsync(d_input_images_.get(), pinned_input_.get(),
            total_image_size, cudaMemcpyHostToDevice, stream_->get()));
        
        CUDA_CHECK(cudaMemcpyAsync(d_image_infos_.get(), h_image_infos_.data(),
            batch_size * sizeof(ImageInfo), cudaMemcpyHostToDevice, stream_->get()));
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float h2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start, stop));
        total_h2d += h2d_time;
        
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        if (use_graph) {
            CUDA_CHECK(cudaGraphLaunch(graph_pool_[graph_batch_size].exec, stream_->get()));
        } else {
            engine_->setup_inference(batch_size);
            
            preprocess_gpu_with_infos(
                static_cast<const uint8_t*>(d_input_images_.get()),
                static_cast<float*>(engine_->get_input_buffer()),
                batch_size,
                config_.input_width, config_.input_height,
                static_cast<ImageInfo*>(d_image_infos_.get()),
                stream_->get());
            
            engine_->enqueue_async(stream_->get());
            
            postprocess_gpu(
                static_cast<const float*>(engine_->get_output_buffer()),
                static_cast<PoseResult*>(d_results_.get()),
                static_cast<int*>(d_num_detections_.get()),
                batch_size,
                engine_->get_output_size(),
                config_.input_width,
                config_.input_height,
                static_cast<const ImageInfo*>(d_image_infos_.get()),
                config_.conf_threshold,
                config_.nms_threshold,
                config_.max_detections,
                stream_->get());
        }
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float graph_time;
        CUDA_CHECK(cudaEventElapsedTime(&graph_time, start, stop));
        total_graph += graph_time;
        
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        CUDA_CHECK(cudaMemcpyAsync(h_num_detections_.data(), d_num_detections_.get(),
            batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_->get()));
        
        CUDA_CHECK(cudaMemcpyAsync(h_results_.data(), d_results_.get(),
            batch_size * config_.max_detections * sizeof(PoseResult),
            cudaMemcpyDeviceToHost, stream_->get()));
        
        stream_->synchronize();
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float d2h_time;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_time, start, stop));
        total_d2h += d2h_time;
        
        total_time += h2d_time + graph_time + d2h_time;
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "\n--- Timing Results ---" << std::endl;
    std::cout << "H2D Transfer:    " << total_h2d / iterations << " ms" << std::endl;
    std::cout << "GPU Pipeline:    " << total_graph / iterations << " ms" << std::endl;
    std::cout << "D2H Transfer:    " << total_d2h / iterations << " ms" << std::endl;
    std::cout << "Total (avg):     " << total_time / iterations << " ms" << std::endl;
    std::cout << "Throughput:      " << (batch_size * 1000.0f) / (total_time / iterations) 
              << " images/sec" << std::endl;
}

}
