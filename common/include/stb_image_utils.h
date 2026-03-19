#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>
#include <algorithm>

namespace image_utils {

inline std::vector<uint8_t> load_image(const std::string& path, int& width, int& height, int& channels) {
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        return {};
    }
    
    std::vector<uint8_t> result(width * height * 3);
    memcpy(result.data(), data, result.size());
    stbi_image_free(data);
    
    return result;
}

inline bool save_image(const std::string& path, const uint8_t* data, int width, int height, int channels = 3) {
    return stbi_write_jpg(path.c_str(), width, height, channels, data, 90) != 0;
}

inline void draw_box(uint8_t* img, int img_w, int img_h,
                     int x1, int y1, int x2, int y2,
                     uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    for (int t = 0; t < thickness; t++) {
        for (int x = std::max(0, x1 + t); x <= std::min(img_w - 1, x2 - t); x++) {
            if (y1 + t >= 0 && y1 + t < img_h) {
                int idx = ((y1 + t) * img_w + x) * 3;
                img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
            }
            if (y2 - t >= 0 && y2 - t < img_h) {
                int idx = ((y2 - t) * img_w + x) * 3;
                img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
            }
        }
        for (int y = std::max(0, y1 + t); y <= std::min(img_h - 1, y2 - t); y++) {
            if (x1 + t >= 0 && x1 + t < img_w) {
                int idx = (y * img_w + x1 + t) * 3;
                img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
            }
            if (x2 - t >= 0 && x2 - t < img_w) {
                int idx = (y * img_w + x2 - t) * 3;
                img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
            }
        }
    }
}

inline void draw_circle(uint8_t* img, int img_w, int img_h,
                        int cx, int cy, int radius,
                        uint8_t r, uint8_t g, uint8_t b) {
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy <= radius * radius) {
                int x = cx + dx;
                int y = cy + dy;
                if (x >= 0 && x < img_w && y >= 0 && y < img_h) {
                    int idx = (y * img_w + x) * 3;
                    img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
                }
            }
        }
    }
}

inline void draw_line(uint8_t* img, int img_w, int img_h,
                      int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    
    int px1 = x1, py1 = y1;
    
    while (true) {
        for (int tx = -thickness / 2; tx <= thickness / 2; tx++) {
            for (int ty = -thickness / 2; ty <= thickness / 2; ty++) {
                int px = px1 + tx;
                int py = py1 + ty;
                if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                    int idx = (py * img_w + px) * 3;
                    img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
                }
            }
        }
        
        if (px1 == x2 && py1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; px1 += sx; }
        if (e2 < dx) { err += dx; py1 += sy; }
    }
}

inline void rgb_to_bgr(uint8_t* data, int width, int height) {
    int size = width * height;
    for (int i = 0; i < size; i++) {
        int idx = i * 3;
        std::swap(data[idx], data[idx + 2]);
    }
}

}
