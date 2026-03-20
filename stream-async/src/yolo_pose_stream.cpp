#include "yolo_pose_stream.h"
#include "../../common/include/preprocess.cuh"
#include "../../common/include/postprocess.cuh"
#include <iostream>
#include <chrono>

namespace yolo {

YoloPoseDetector::YoloPoseDetector() {}

YoloPoseDetector::~YoloPoseDetector() {
    cuda_preprocess_cleanup();
    cuda_postprocess_cleanup();
}

bool YoloPoseDetector::init(const InferConfig& config) {
    config_ = config;
    
    engine_ = std::make_unique<TrtEngine>();
    if (!engine_->load_engine(config.engine_path)) {
        std::cerr << "Failed to load engine: " << config.engine_path << std::endl;
        return false;
    }
    
    stream_ = std::make_unique<CudaStream>();
    
    if (!allocate_buffers(config.max_batch_size)) {
        return false;
    }
    
    cuda_preprocess_init();
    cuda_postprocess_init(config.max_detections, config.max_batch_size);
    
    std::cout << "YoloPoseDetector (Stream Async) initialized successfully" << std::endl;
    return true;
}

bool YoloPoseDetector::allocate_buffers(int max_batch_size) {
    int input_image_size = max_batch_size * config_.input_width * config_.input_height * 3;
    int tensor_size = max_batch_size * 3 * config_.input_width * config_.input_height;
    
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

std::vector<std::vector<PoseResult>> YoloPoseDetector::infer(
    const std::vector<std::vector<uint8_t>>& images,
    int src_width,
    int src_height) {
    
    int batch_size = static_cast<int>(images.size());
    if (batch_size == 0 || batch_size > config_.max_batch_size) {
        return {};
    }
    
    size_t single_image_size = src_width * src_height * 3;
    
    uint8_t* pinned_ptr = static_cast<uint8_t*>(pinned_input_.get());
    for (int i = 0; i < batch_size; i++) {
        memcpy(pinned_ptr + i * single_image_size, 
               images[i].data(), single_image_size);
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_input_images_.get(), pinned_input_.get(),
        batch_size * single_image_size, cudaMemcpyHostToDevice, stream_->get()));
    
    preprocess_gpu(
        static_cast<const uint8_t*>(d_input_images_.get()),
        static_cast<float*>(engine_->get_input_buffer()),
        batch_size,
        src_width, src_height,
        config_.input_width, config_.input_height,
        static_cast<ImageInfo*>(d_image_infos_.get()),
        stream_->get());
    
    engine_->infer_async(batch_size, stream_->get());
    
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
        config_.max_detections,
        stream_->get());
    
    CUDA_CHECK(cudaMemcpyAsync(h_num_detections_.data(), d_num_detections_.get(),
        batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_->get()));
    
    CUDA_CHECK(cudaMemcpyAsync(h_results_.data(), d_results_.get(),
        batch_size * config_.max_detections * sizeof(PoseResult),
        cudaMemcpyDeviceToHost, stream_->get()));
    
    stream_->synchronize();
    
    std::vector<std::vector<PoseResult>> results(batch_size);
    for (int i = 0; i < batch_size; i++) {
        int num_det = h_num_detections_[i];
        results[i].reserve(num_det);
        for (int j = 0; j < num_det; j++) {
            results[i].push_back(h_results_[i * config_.max_detections + j]);
        }
    }
    
    return results;
}

void YoloPoseDetector::infer_async(
    const std::vector<std::vector<uint8_t>>& images,
    int src_width,
    int src_height,
    std::vector<std::vector<PoseResult>>& results) {
    
    results = infer(images, src_width, src_height);
}

void YoloPoseDetector::benchmark(
    const std::vector<std::vector<uint8_t>>& images,
    int src_width,
    int src_height,
    int iterations) {
    
    int batch_size = static_cast<int>(images.size());
    
    std::cout << "\n=== Benchmark (CUDA Stream Async) ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_preprocess = 0.0f;
    float total_infer = 0.0f;
    float total_postprocess = 0.0f;
    float total_time = 0.0f;
    
    size_t single_image_size = src_width * src_height * 3;
    uint8_t* pinned_ptr = static_cast<uint8_t*>(pinned_input_.get());
    for (int i = 0; i < batch_size; i++) {
        memcpy(pinned_ptr + i * single_image_size, 
               images[i].data(), single_image_size);
    }
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        CUDA_CHECK(cudaMemcpyAsync(d_input_images_.get(), pinned_input_.get(),
            batch_size * single_image_size, cudaMemcpyHostToDevice, stream_->get()));
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float h2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start, stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        preprocess_gpu(
            static_cast<const uint8_t*>(d_input_images_.get()),
            static_cast<float*>(engine_->get_input_buffer()),
            batch_size,
            src_width, src_height,
            config_.input_width, config_.input_height,
            static_cast<ImageInfo*>(d_image_infos_.get()),
            stream_->get());
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float preprocess_time;
        CUDA_CHECK(cudaEventElapsedTime(&preprocess_time, start, stop));
        total_preprocess += preprocess_time;
        
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
        engine_->infer_async(batch_size, stream_->get());
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float infer_time;
        CUDA_CHECK(cudaEventElapsedTime(&infer_time, start, stop));
        total_infer += infer_time;
        
        CUDA_CHECK(cudaEventRecord(start, stream_->get()));
        
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
            config_.max_detections,
            stream_->get());
        
        CUDA_CHECK(cudaEventRecord(stop, stream_->get()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float postprocess_time;
        CUDA_CHECK(cudaEventElapsedTime(&postprocess_time, start, stop));
        total_postprocess += postprocess_time;
        
        total_time += h2d_time + preprocess_time + infer_time + postprocess_time;
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "\n--- Timing Results ---" << std::endl;
    std::cout << "H2D Transfer:    " << total_time / iterations << " ms" << std::endl;
    std::cout << "Preprocess:      " << total_preprocess / iterations << " ms" << std::endl;
    std::cout << "Inference:       " << total_infer / iterations << " ms" << std::endl;
    std::cout << "Postprocess:     " << total_postprocess / iterations << " ms" << std::endl;
    std::cout << "Total (avg):     " << total_time / iterations << " ms" << std::endl;
    std::cout << "Throughput:      " << (batch_size * 1000.0f) / (total_time / iterations) 
              << " images/sec" << std::endl;
}

}
