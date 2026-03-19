#include "trt_engine.h"
#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <memory>

namespace yolo {

struct TrtEngine::Impl {
    Logger logger;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    void* d_input = nullptr;
    void* d_output = nullptr;
    int input_size = 0;
    int output_size = 0;
    int max_batch_size = 1;
    int configured_max_batch = 16;
    
    ~Impl() {
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        delete context;
        delete engine;
    }
};

TrtEngine::TrtEngine() : impl_(new Impl()) {}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::load_engine(const std::string& engine_path, int max_batch_size) {
    impl_->configured_max_batch = max_batch_size;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    
    auto runtime = nvinfer1::createInferRuntime(impl_->logger);
    if (!runtime) return false;
    
    impl_->engine = runtime->deserializeCudaEngine(buffer.data(), size);
    delete runtime;
    
    if (!impl_->engine) return false;
    
    impl_->context = impl_->engine->createExecutionContext();
    if (!impl_->context) return false;
    
    auto input_name = impl_->engine->getIOTensorName(0);
    auto output_name = impl_->engine->getIOTensorName(1);
    
    auto input_dims = impl_->engine->getTensorShape(input_name);
    auto output_dims = impl_->engine->getTensorShape(output_name);
    
    impl_->max_batch_size = input_dims.d[0];
    if (impl_->max_batch_size == -1) {
        impl_->max_batch_size = impl_->configured_max_batch;
    }
    
    impl_->input_size = 1;
    for (int i = 1; i < input_dims.nbDims; i++) {
        int dim = input_dims.d[i];
        if (dim == -1) dim = 640;
        impl_->input_size *= dim;
    }
    
    impl_->output_size = 1;
    for (int i = 1; i < output_dims.nbDims; i++) {
        int dim = output_dims.d[i];
        if (dim == -1) dim = 8400;
        impl_->output_size *= dim;
    }
    
    CUDA_CHECK(cudaMalloc(&impl_->d_input, impl_->max_batch_size * impl_->input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_output, impl_->max_batch_size * impl_->output_size * sizeof(float)));
    
    std::cout << "Engine loaded: " << engine_path << std::endl;
    std::cout << "  Max batch: " << impl_->max_batch_size << std::endl;
    std::cout << "  Input size: " << impl_->input_size << std::endl;
    std::cout << "  Output size: " << impl_->output_size << std::endl;
    
    return true;
}

bool TrtEngine::build_engine_from_onnx(
    const std::string& onnx_path,
    const std::string& engine_path,
    int max_batch_size,
    bool use_fp16) {
    
    auto builder = nvinfer1::createInferBuilder(impl_->logger);
    if (!builder) return false;
    
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network) { delete builder; return false; }
    
    auto config = builder->createBuilderConfig();
    if (!config) { delete network; delete builder; return false; }
    
    auto parser = nvonnxparser::createParser(*network, impl_->logger);
    if (!parser) { delete config; delete network; delete builder; return false; }
    
    if (!parser->parseFromFile(onnx_path.c_str(), 
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        delete parser;
        delete config;
        delete network;
        delete builder;
        return false;
    }
    
    auto profile = builder->createOptimizationProfile();
    auto input_name = network->getInput(0)->getName();
    auto input_dims = network->getInput(0)->getDimensions();
    
    int c = input_dims.d[1];
    int h = input_dims.d[2];
    int w = input_dims.d[3];
    
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, 
        nvinfer1::Dims4{1, c, h, w});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, 
        nvinfer1::Dims4{max_batch_size / 2, c, h, w});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, 
        nvinfer1::Dims4{max_batch_size, c, h, w});
    
    config->addOptimizationProfile(profile);
    
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
    
    if (use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }
    
    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        delete parser;
        delete config;
        delete network;
        delete builder;
        return false;
    }
    
    std::ofstream file(engine_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    
    delete plan;
    delete parser;
    delete config;
    delete network;
    delete builder;
    
    std::cout << "Engine built and saved to: " << engine_path << std::endl;
    return true;
}

void* TrtEngine::get_input_buffer() const { return impl_->d_input; }
void* TrtEngine::get_output_buffer() const { return impl_->d_output; }
int TrtEngine::get_input_size() const { return impl_->input_size; }
int TrtEngine::get_output_size() const { return impl_->output_size; }
int TrtEngine::get_max_batch_size() const { return impl_->max_batch_size; }

void TrtEngine::setup_inference(int batch_size) {
    auto input_name = impl_->engine->getIOTensorName(0);
    auto output_name = impl_->engine->getIOTensorName(1);
    
    impl_->context->setInputShape(input_name, nvinfer1::Dims4{batch_size, 3, 640, 640});
    impl_->context->setTensorAddress(input_name, impl_->d_input);
    impl_->context->setTensorAddress(output_name, impl_->d_output);
}

void TrtEngine::enqueue_async(cudaStream_t stream) {
    impl_->context->enqueueV3(stream);
}

void TrtEngine::infer_async(int batch_size, cudaStream_t stream) {
    setup_inference(batch_size);
    enqueue_async(stream);
}

void TrtEngine::infer_sync(int batch_size) {
    auto input_name = impl_->engine->getIOTensorName(0);
    auto output_name = impl_->engine->getIOTensorName(1);
    
    impl_->context->setInputShape(input_name, nvinfer1::Dims4{batch_size, 3, 640, 640});
    impl_->context->setTensorAddress(input_name, impl_->d_input);
    impl_->context->setTensorAddress(output_name, impl_->d_output);
    
    impl_->context->executeV2(nullptr);
}

nvinfer1::ICudaEngine* TrtEngine::get_engine() const { return impl_->engine; }
nvinfer1::IExecutionContext* TrtEngine::get_context() const { return impl_->context; }

bool build_engine(
    const std::string& onnx_path,
    const std::string& engine_path,
    int max_batch_size,
    bool use_fp16) {
    
    TrtEngine engine;
    return engine.build_engine_from_onnx(onnx_path, engine_path, max_batch_size, use_fp16);
}

}
