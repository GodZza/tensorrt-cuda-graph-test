#pragma once

#include <windows.h>
#include <vector>
#include <string>
#include <cstdint>

namespace win32_display {

class ImageWindow {
public:
    ImageWindow() : hWnd_(nullptr), img_width_(0), img_height_(0) {}
    
    ~ImageWindow() {
        if (hWnd_) {
            DestroyWindow(hWnd_);
        }
    }
    
    bool show(const std::string& title, const uint8_t* rgb_data, int width, int height) {
        if (!rgb_data || width <= 0 || height <= 0) return false;
        
        img_width_ = width;
        img_height_ = height;
        
        int row_size = ((width * 3 + 3) / 4) * 4;
        bgr_image_.resize(row_size * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_idx = (y * width + x) * 3;
                int dst_idx = y * row_size + x * 3;
                bgr_image_[dst_idx + 0] = rgb_data[src_idx + 2];
                bgr_image_[dst_idx + 1] = rgb_data[src_idx + 1];
                bgr_image_[dst_idx + 2] = rgb_data[src_idx + 0];
            }
        }
        
        bmi_.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi_.bmiHeader.biWidth = width;
        bmi_.bmiHeader.biHeight = -height;
        bmi_.bmiHeader.biPlanes = 1;
        bmi_.bmiHeader.biBitCount = 24;
        bmi_.bmiHeader.biCompression = BI_RGB;
        bmi_.bmiHeader.biSizeImage = row_size * height;
        bmi_.bmiHeader.biXPelsPerMeter = 0;
        bmi_.bmiHeader.biYPelsPerMeter = 0;
        bmi_.bmiHeader.biClrUsed = 0;
        bmi_.bmiHeader.biClrImportant = 0;
        
        if (!hWnd_) {
            HINSTANCE hInstance = GetModuleHandle(nullptr);
            
            static bool registered = false;
            if (!registered) {
                WNDCLASSEXW wcex = {};
                wcex.cbSize = sizeof(WNDCLASSEX);
                wcex.style = CS_HREDRAW | CS_VREDRAW;
                wcex.lpfnWndProc = StaticWndProc;
                wcex.hInstance = hInstance;
                wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
                wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
                wcex.lpszClassName = L"ImageWindowClass";
                RegisterClassExW(&wcex);
                registered = true;
            }
            
            int screen_width = GetSystemMetrics(SM_CXSCREEN);
            int screen_height = GetSystemMetrics(SM_CYSCREEN);
            
            int window_width = min(width + 16, screen_width - 100);
            int window_height = min(height + 39, screen_height - 100);
            int x = (screen_width - window_width) / 2;
            int y = (screen_height - window_height) / 5;
            
            hWnd_ = CreateWindowW(
                L"ImageWindowClass",
                std::wstring(title.begin(), title.end()).c_str(),
                WS_OVERLAPPEDWINDOW,
                x, y, window_width, window_height,
                nullptr, nullptr, hInstance, this);
            
            if (!hWnd_) return false;
            
            ShowWindow(hWnd_, SW_SHOW);
            UpdateWindow(hWnd_);
        } else {
            SetWindowTextW(hWnd_, std::wstring(title.begin(), title.end()).c_str());
            InvalidateRect(hWnd_, nullptr, TRUE);
        }
        
        return true;
    }
    
    int messageLoop() {
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        return static_cast<int>(msg.wParam);
    }
    
private:
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        ImageWindow* pThis = nullptr;
        
        if (message == WM_NCCREATE) {
            CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
            pThis = reinterpret_cast<ImageWindow*>(pCreate->lpCreateParams);
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        } else {
            pThis = reinterpret_cast<ImageWindow*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        }
        
        if (pThis) {
            return pThis->WndProc(hWnd, message, wParam, lParam);
        }
        
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    
    LRESULT WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            
            if (!bgr_image_.empty()) {
                int row_size = ((img_width_ * 3 + 3) / 4) * 4;
                
                RECT clientRect;
                GetClientRect(hWnd, &clientRect);
                
                StretchDIBits(hdc,
                    0, 0, clientRect.right, clientRect.bottom,
                    0, 0, img_width_, img_height_,
                    bgr_image_.data(), &bmi_,
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
    
    HWND hWnd_;
    int img_width_;
    int img_height_;
    std::vector<uint8_t> bgr_image_;
    BITMAPINFO bmi_ = {};
};

inline int show_image(const std::string& title, const uint8_t* rgb_data, int width, int height) {
    ImageWindow window;
    if (!window.show(title, rgb_data, width, height)) {
        return -1;
    }
    return window.messageLoop();
}

inline int show_images(
    const std::string& title,
    const std::vector<uint8_t*>& rgb_datas,
    const std::vector<int>& widths,
    const std::vector<int>& heights) {
    
    if (rgb_datas.empty()) return -1;
    
    int num_images = static_cast<int>(rgb_datas.size());
    
    int cols = (num_images <= 2) ? num_images : 2;
    int rows = (num_images + cols - 1) / cols;
    
    int max_w = 0, max_h = 0;
    for (int i = 0; i < num_images; i++) {
        if (widths[i] > max_w) max_w = widths[i];
        if (heights[i] > max_h) max_h = heights[i];
    }
    
    int canvas_width = max_w * cols;
    int canvas_height = max_h * rows;
    
    std::vector<uint8_t> canvas(canvas_width * canvas_height * 3, 128);
    
    for (int i = 0; i < num_images; i++) {
        int col = i % cols;
        int row = i / cols;
        int offset_x = col * max_w + (max_w - widths[i]) / 2;
        int offset_y = row * max_h + (max_h - heights[i]) / 2;
        
        for (int y = 0; y < heights[i]; y++) {
            for (int x = 0; x < widths[i]; x++) {
                int src_idx = (y * widths[i] + x) * 3;
                int dst_y = offset_y + y;
                int dst_x = offset_x + x;
                int dst_idx = (dst_y * canvas_width + dst_x) * 3;
                canvas[dst_idx + 0] = rgb_datas[i][src_idx + 0];
                canvas[dst_idx + 1] = rgb_datas[i][src_idx + 1];
                canvas[dst_idx + 2] = rgb_datas[i][src_idx + 2];
            }
        }
    }
    
    return show_image(title, canvas.data(), canvas_width, canvas_height);
}

}
