#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>

namespace yolo {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            printf("[TensorRT] %s\n", msg);
        }
    }
};

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();
    
    bool load_engine(const std::string& engine_path, int max_batch_size = 16);
    bool build_engine_from_onnx(const std::string& onnx_path, 
                                 const std::string& engine_path,
                                 int max_batch_size,
                                 bool use_fp16);
    
    void* get_input_buffer() const;
    void* get_output_buffer() const;
    
    int get_input_size() const;
    int get_output_size() const;
    int get_max_batch_size() const;
    
    void infer_async(int batch_size, cudaStream_t stream);
    void infer_sync(int batch_size);
    
    void setup_inference(int batch_size);
    void enqueue_async(cudaStream_t stream);
    
    nvinfer1::ICudaEngine* get_engine() const;
    nvinfer1::IExecutionContext* get_context() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool build_engine(
    const std::string& onnx_path,
    const std::string& engine_path,
    int max_batch_size,
    bool use_fp16);

}
