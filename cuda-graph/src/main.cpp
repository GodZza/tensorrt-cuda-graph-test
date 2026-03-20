#include "yolo_pose_graph.h"
#include "../../common/include/stb_image_utils.h"
#include "../../common/include/win32_display.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void draw_results_on_image(
    std::vector<uint8_t>& img,
    int width, int height,
    const std::vector<yolo::PoseResult>& results) {
    
    static const uint8_t colors[17][3] = {
        {255, 0, 0}, {255, 85, 0}, {255, 170, 0},
        {255, 255, 0}, {170, 255, 0}, {85, 255, 0},
        {0, 255, 0}, {0, 255, 85}, {0, 255, 170},
        {0, 255, 255}, {0, 170, 255}, {0, 85, 255},
        {0, 0, 255}, {85, 0, 255}, {170, 0, 255},
        {255, 0, 255}, {255, 0, 170}
    };
    
    static const int skeleton[][2] = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };
    
    for (const auto& result : results) {
        int x1 = static_cast<int>(result.bbox.x1);
        int y1 = static_cast<int>(result.bbox.y1);
        int x2 = static_cast<int>(result.bbox.x2);
        int y2 = static_cast<int>(result.bbox.y2);
        
        image_utils::draw_box(img.data(), width, height, x1, y1, x2, y2, 0, 255, 0, 2);
        
        for (int i = 0; i < 17; i++) {
            if (result.keypoints[i].conf > 0.5f) {
                int kx = static_cast<int>(result.keypoints[i].x);
                int ky = static_cast<int>(result.keypoints[i].y);
                image_utils::draw_circle(img.data(), width, height, kx, ky, 3,
                    colors[i][0], colors[i][1], colors[i][2]);
            }
        }
        
        for (int i = 0; i < 16; i++) {
            int a = skeleton[i][0];
            int b = skeleton[i][1];
            if (result.keypoints[a].conf > 0.5f && result.keypoints[b].conf > 0.5f) {
                int ax = static_cast<int>(result.keypoints[a].x);
                int ay = static_cast<int>(result.keypoints[a].y);
                int bx = static_cast<int>(result.keypoints[b].x);
                int by = static_cast<int>(result.keypoints[b].y);
                image_utils::draw_line(img.data(), width, height, ax, ay, bx, by, 255, 255, 255, 2);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <engine_path> <image_path> [iterations]" << std::endl;
        std::cout << "Example: " << argv[0] << " models/yolo11n-pose.engine test-img/bus.jpg" << std::endl;
        std::cout << "  Multiple images: " << argv[0] << " models/yolo11n-pose.engine test-img/bus.jpg test-img/zidane.jpg" << std::endl;
        return -1;
    }
    
    std::string engine_path = argv[1];
    std::vector<std::string> image_paths;
    for (int i = 2; i < argc && argv[i][0] != '-'; i++) {
        image_paths.push_back(argv[i]);
    }
    int iterations = 0;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--iter" && i + 1 < argc) {
            iterations = std::atoi(argv[i + 1]);
        }
    }
    
    yolo::InferConfig config;
    config.engine_path = engine_path;
    config.input_width = 640;
    config.input_height = 640;
    config.max_batch_size = 16;
    config.conf_threshold = 0.25f;
    config.nms_threshold = 0.45f;
    //config.max_detections = 25;
    //config.max_detections_to_copy = 15;
    config.use_fp16 = true;
    
    yolo::YoloPoseDetectorGraph detector;
    if (!detector.init(config)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }
    
    std::vector<std::vector<uint8_t>> images;
    std::vector<std::pair<int, int>> image_sizes;
    std::vector<int> widths, heights;
    
    for (const auto& path : image_paths) {
        int w, h, c;
        auto img = image_utils::load_image(path, w, h, c);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }
        images.push_back(img);
        image_sizes.push_back({w, h});
        widths.push_back(w);
        heights.push_back(h);
        std::cout << "Loaded: " << path << " (" << w << "x" << h << ")" << std::endl;
    }
    
    if (images.empty()) {
        std::cerr << "No valid images loaded" << std::endl;
        return -1;
    }
    
    auto results = detector.infer_batch(images, image_sizes);
    
    std::cout << "\n=== Sync inference done ===" << std::endl;
    
    // Test async inference
    std::cout << "\n=== Test async inference ===" << std::endl;
    auto future = detector.infer_batch_async(images, image_sizes);
    
    // Can do other work while waiting
    std::cout << "Async inference in progress... CPU can do other tasks" << std::endl;
    
    // Get async results
    auto async_results = future.get();
    std::cout << "Async inference done!" << std::endl;
    
    // Verify sync and async results match
    bool results_match = true;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].size() != async_results[i].size()) {
            results_match = false;
            break;
        }
        for (size_t j = 0; j < results[i].size(); j++) {
            if (std::abs(results[i][j].bbox.conf - async_results[i][j].bbox.conf) > 0.001f) {
                results_match = false;
                break;
            }
        }
    }
    std::cout << "Sync/Async results match: " << (results_match ? "YES" : "NO") << std::endl;
    
    // Test double-buffer pipeline
    std::cout << "\n=== Test Double-Buffer Pipeline ===" << std::endl;
    
    auto buffer_a = detector.create_buffer();
    auto buffer_b = detector.create_buffer();
    
    std::vector<std::vector<std::vector<yolo::PoseResult>>> pipeline_results;
    
    // Frame 1: prepare to buffer_a
    detector.prepare_async(images, image_sizes, buffer_a);
    
    // Frame 2: prepare to buffer_b, wait for buffer_a
    detector.prepare_async(images, image_sizes, buffer_b);
    auto results_a = detector.wait_and_get_results(buffer_a);
    pipeline_results.push_back(results_a);
    
    // Frame 3: prepare to buffer_a, wait for buffer_b
    detector.prepare_async(images, image_sizes, buffer_a);
    auto results_b = detector.wait_and_get_results(buffer_b);
    pipeline_results.push_back(results_b);
    
    // Get last result
    auto results_a2 = detector.wait_and_get_results(buffer_a);
    pipeline_results.push_back(results_a2);
    
    std::cout << "Pipeline completed! Processed " << pipeline_results.size() << " frames" << std::endl;
    
    // Verify pipeline results match sync results
    bool pipeline_match = true;
    for (size_t f = 0; f < pipeline_results.size(); f++) {
        for (size_t i = 0; i < results.size(); i++) {
            if (pipeline_results[f][i].size() != results[i].size()) {
                pipeline_match = false;
                break;
            }
            for (size_t j = 0; j < results[i].size(); j++) {
                if (std::abs(pipeline_results[f][i][j].bbox.conf - results[i][j].bbox.conf) > 0.001f) {
                    pipeline_match = false;
                    break;
                }
            }
        }
    }
    std::cout << "Pipeline results match sync: " << (pipeline_match ? "YES" : "NO") << std::endl;
    
    // Performance comparison: single vs double buffer
    if (iterations > 0) {
        std::cout << "\n=== Pipeline Performance Comparison ===" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Single buffer (no pipeline)
        std::cout << "\n--- Single Buffer (No Pipeline) ---" << std::endl;
        auto single_buffer = detector.create_buffer();
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            detector.prepare_async(images, image_sizes, single_buffer);
            detector.wait_and_get_results(single_buffer);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float single_time;
        CUDA_CHECK(cudaEventElapsedTime(&single_time, start, stop));
        float single_fps = (iterations * images.size()) / (single_time / 1000.0f);
        std::cout << "Total time: " << single_time << " ms" << std::endl;
        std::cout << "Avg per frame: " << single_time / iterations << " ms" << std::endl;
        std::cout << "Throughput: " << single_fps << " images/sec" << std::endl;
        
        // Double buffer (pipeline)
        std::cout << "\n--- Double Buffer (Pipeline) ---" << std::endl;
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // Frame 1: prepare to buffer_a
        detector.prepare_async(images, image_sizes, buffer_a);
        
        for (int i = 1; i < iterations - 1; i++) {
            auto& cur_buffer = (i % 2 == 0) ? buffer_a : buffer_b;
            auto& prev_buffer = (i % 2 == 0) ? buffer_b : buffer_a;
            
            detector.prepare_async(images, image_sizes, cur_buffer);
            detector.wait_and_get_results(prev_buffer);
        }
        
        // Last frame
        detector.wait_and_get_results((iterations % 2 == 0) ? buffer_a : buffer_b);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float double_time;
        CUDA_CHECK(cudaEventElapsedTime(&double_time, start, stop));
        float double_fps = (iterations * images.size()) / (double_time / 1000.0f);
        std::cout << "Total time: " << double_time << " ms" << std::endl;
        std::cout << "Avg per frame: " << double_time / iterations << " ms" << std::endl;
        std::cout << "Throughput: " << double_fps << " images/sec" << std::endl;
        
        // Comparison
        std::cout << "\n--- Comparison ---" << std::endl;
        float speedup = single_time / double_time;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "Time saved: " << ((single_time - double_time) / single_time * 100) << "%" << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "\nDetection results for " << image_paths[i] << ":" << std::endl;
        for (size_t j = 0; j < results[i].size(); j++) {
            const auto& r = results[i][j];
            std::cout << "  Person " << j + 1 << ": conf=" << r.bbox.conf
                      << " bbox=[" << r.bbox.x1 << "," << r.bbox.y1 
                      << "," << r.bbox.x2 << "," << r.bbox.y2 << "]" << std::endl;
        }
        
        draw_results_on_image(images[i], widths[i], heights[i], results[i]);
        
        size_t dot_pos = image_paths[i].find_last_of('.');
        std::string output_path = image_paths[i].substr(0, dot_pos) + "_result_graph.jpg";
        stbi_write_jpg(output_path.c_str(), widths[i], heights[i], 3, images[i].data(), 90);
        std::cout << "Result saved to: " << output_path << std::endl;
    }
    
    if (iterations > 0 && images.size() > 0) {
        detector.benchmark(images, widths[0], heights[0], iterations);
    }
    
    std::cout << "\nPress ESC to exit..." << std::endl;
    
    if (images.size() == 1) {
        win32_display::show_image("YOLO11 Pose Detection Result (CUDA Graph)", images[0].data(), widths[0], heights[0]);
    } else {
        std::vector<uint8_t*> img_ptrs;
        for (auto& img : images) {
            img_ptrs.push_back(img.data());
        }
        win32_display::show_images("YOLO11 Pose Detection Result (CUDA Graph - Batch)", img_ptrs, widths, heights);
    }
    
    return 0;
}
