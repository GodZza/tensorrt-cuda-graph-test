#include "yolo_pose_stream.h"
#include "../../common/include/stb_image_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

static std::vector<uint8_t> g_display_image_bgr;
static int g_img_width = 0;
static int g_img_height = 0;
static BITMAPINFO g_bmi = {};

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        
        if (!g_display_image_bgr.empty()) {
            StretchDIBits(hdc,
                0, 0, g_img_width, g_img_height,
                0, 0, g_img_width, g_img_height,
                g_display_image_bgr.data(), &g_bmi,
                DIB_RGB_COLORS, SRCCOPY);
        }
        
        EndPaint(hWnd, &ps);
        break;
    }
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            PostQuitMessage(0);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

void show_image_win32(const std::string& title, const uint8_t* img_data, int width, int height) {
    g_img_width = width;
    g_img_height = height;
    
    g_display_image_bgr.assign(img_data, img_data + width * height * 3);
    image_utils::rgb_to_bgr(g_display_image_bgr.data(), width, height);
    
    g_bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    g_bmi.bmiHeader.biWidth = width;
    g_bmi.bmiHeader.biHeight = -height;
    g_bmi.bmiHeader.biPlanes = 1;
    g_bmi.bmiHeader.biBitCount = 24;
    g_bmi.bmiHeader.biCompression = BI_RGB;
    g_bmi.bmiHeader.biSizeImage = 0;
    g_bmi.bmiHeader.biXPelsPerMeter = 0;
    g_bmi.bmiHeader.biYPelsPerMeter = 0;
    g_bmi.bmiHeader.biClrUsed = 0;
    g_bmi.bmiHeader.biClrImportant = 0;
    
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    
    WNDCLASSEXW wcex = {};
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszClassName = L"YoloPoseWindow";
    RegisterClassExW(&wcex);
    
    int screen_width = GetSystemMetrics(SM_CXSCREEN);
    int screen_height = GetSystemMetrics(SM_CYSCREEN);
    
    int window_width = min(width, screen_width - 100);
    int window_height = min(height, screen_height - 100);
    int x = (screen_width - window_width) / 2;
    int y = (screen_height - window_height) / 5;
    
    HWND hWnd = CreateWindowW(
        L"YoloPoseWindow",
        std::wstring(title.begin(), title.end()).c_str(),
        WS_OVERLAPPEDWINDOW,
        x, y, window_width, window_height,
        nullptr, nullptr, hInstance, nullptr);
    
    if (!hWnd) {
        std::cerr << "Failed to create window" << std::endl;
        return;
    }
    
    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);
    
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

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
    
    std::cout << "\nPress ESC or close window to exit..." << std::endl;
    show_image_win32("YOLO11 Pose Detection Result", image_data.data(), width, height);
    
    if (iterations > 0) {
        detector.benchmark(images, width, height, iterations);
    }
    
    return 0;
}
