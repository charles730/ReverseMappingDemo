#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

struct PixelResult {
    int local_idx;
    float u;
    float v;
};

// 纯C++版本的逆映射计算函数
std::vector<PixelResult> compute_reverse_mapping_pure(
    int canvas_width, int canvas_height,
    int global_width, int global_height,
    const std::vector<std::vector<std::vector<double>>>& H_invs,
    const std::vector<std::vector<int>>& local_sizes
) {
    // 获取局部图像数量
    int num_locals = H_invs.size();
    
    // 创建输出数组
    std::vector<PixelResult> result(canvas_height * canvas_width);
    
    // 计算缩放因子
    double scale_x = static_cast<double>(global_width) / canvas_width;
    double scale_y = static_cast<double>(global_height) / canvas_height;
    
    // 遍历画布上的每个像素
    for (int y = 0; y < canvas_height; ++y) {
        for (int x = 0; x < canvas_width; ++x) {
            // 将画布坐标映射到全局图像坐标
            double global_x = x * scale_x;
            double global_y = y * scale_y;
            
            PixelResult best_result = {-1, -1.0f, -1.0f};
            double min_dist = std::numeric_limits<double>::max();
            
            // 检查每个局部图像
            for (int idx = 0; idx < num_locals; ++idx) {
                // 获取当前局部图像的逆矩阵
                const auto& H_inv = H_invs[idx];
                
                // 应用逆变换
                double u = H_inv[0][0] * global_x + H_inv[0][1] * global_y + H_inv[0][2];
                double v = H_inv[1][0] * global_x + H_inv[1][1] * global_y + H_inv[1][2];
                double w = H_inv[2][0] * global_x + H_inv[2][1] * global_y + H_inv[2][2];
                
                if (w == 0) continue;
                u /= w;
                v /= w;
                
                // 检查是否在局部图像范围内
                if (u >= 0 && u < local_sizes[idx][0] && v >= 0 && v < local_sizes[idx][1]) {
                    // 计算到图像中心的距离
                    double center_x = local_sizes[idx][0] / 2.0;
                    double center_y = local_sizes[idx][1] / 2.0;
                    double dist = std::sqrt((u - center_x) * (u - center_x) + 
                                           (v - center_y) * (v - center_y));
                    
                    // 更新最佳匹配
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_result = {idx, static_cast<float>(u), static_cast<float>(v)};
                    }
                }
            }
            
            // 存储结果
            result[y * canvas_width + x] = best_result;
        }
    }
    
    return result;
}

// 测试函数
int main() {
    // 创建测试数据
    int canvas_width = 100;
    int canvas_height = 100;
    int global_width = 200;
    int global_height = 200;
    
    // 创建简单的3x3单位矩阵作为测试
    std::vector<std::vector<std::vector<double>>> H_invs = {
        {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        {{{1, 0, 50}, {0, 1, 50}, {0, 0, 1}}}
    };
    
    std::vector<std::vector<int>> local_sizes = {
        {100, 100},
        {100, 100}
    };
    
    // 计算逆映射
    auto result = compute_reverse_mapping_pure(canvas_width, canvas_height, 
                                              global_width, global_height,
                                              H_invs, local_sizes);
    
    // 输出前几个结果
    std::cout << "前10个像素的映射结果:" << std::endl;
    for (int i = 0; i < 10 && i < result.size(); ++i) {
        std::cout << "像素 " << i << ": local_idx=" << result[i].local_idx 
                  << ", u=" << result[i].u << ", v=" << result[i].v << std::endl;
    }
    
    return 0;
}