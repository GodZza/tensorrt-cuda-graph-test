#include "yolo_pose_graph.h"
#include "../../common/include/stb_image_utils.h"
#include "../../common/include/win32_display.h"
#include <iostream>
#include <vector>
#include <string>

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
    config.max_detections = 100;
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
    win32_display::show_image("YOLO11 Pose Detection Result (CUDA Graph)", images[0].data(), widths[0], heights[0]);
    
    return 0;
}
