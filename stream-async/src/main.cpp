#include "yolo_pose_stream.h"
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
        return -1;
    }
    
    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    int iterations = argc > 3 ? std::atoi(argv[3]) : 0;
    
    yolo::InferConfig config;
    config.engine_path = engine_path;
    config.input_width = 640;
    config.input_height = 640;
    config.max_batch_size = 16;
    config.conf_threshold = 0.25f;
    config.nms_threshold = 0.45f;
    config.max_detections = 100;
    config.use_fp16 = true;
    
    yolo::YoloPoseDetector detector;
    if (!detector.init(config)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }
    
    int width, height, channels;
    auto image_data = image_utils::load_image(image_path, width, height, channels);
    if (image_data.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    
    std::vector<std::vector<uint8_t>> images = {image_data};
    
    auto results = detector.infer(images, width, height);
    
    std::cout << "\nDetection results for " << image_path << ":" << std::endl;
    for (size_t i = 0; i < results[0].size(); i++) {
        const auto& r = results[0][i];
        std::cout << "  Person " << i + 1 << ": conf=" << r.bbox.conf
                  << " bbox=[" << r.bbox.x1 << "," << r.bbox.y1 
                  << "," << r.bbox.x2 << "," << r.bbox.y2 << "]" << std::endl;
    }
    
    draw_results_on_image(image_data, width, height, results[0]);
    
    size_t dot_pos = image_path.find_last_of('.');
    std::string output_path = image_path.substr(0, dot_pos) + "_result_stream.jpg";
    stbi_write_jpg(output_path.c_str(), width, height, 3, image_data.data(), 90);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    if (iterations > 0) {
        detector.benchmark(images, width, height, iterations);
    }
    
    std::cout << "\nPress ESC to exit..." << std::endl;
    win32_display::show_image("YOLO11 Pose Detection Result", image_data.data(), width, height);
    
    return 0;
}
