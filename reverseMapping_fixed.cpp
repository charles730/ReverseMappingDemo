#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct PixelResult {
    int local_idx;
    float u;
    float v;
};

py::array_t<PixelResult> compute_reverse_mapping(
    int canvas_width, int canvas_height,
    int global_width, int global_height,
    py::array_t<double> H_invs_array,
    py::array_t<int> local_sizes_array
) {
    // 获取输入数组的指针
    auto H_invs = H_invs_array.unchecked<3>();
    auto local_sizes = local_sizes_array.unchecked<2>();
    
    // 获取局部图像数量
    int num_locals = H_invs.shape(0);
    
    // 创建输出数组
    auto result = py::array_t<PixelResult>(canvas_height * canvas_width);
    auto result_ptr = result.mutable_data();
    
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
                // 获取当前局部图像的逆矩阵 - 修复数组访问语法
                auto H_inv = H_invs[idx];
                
                // 应用逆变换
                double u = H_inv(0, 0) * global_x + H_inv(0, 1) * global_y + H_inv(0, 2);
                double v = H_inv(1, 0) * global_x + H_inv(1, 1) * global_y + H_inv(1, 2);
                double w = H_inv(2, 0) * global_x + H_inv(2, 1) * global_y + H_inv(2, 2);
                
                if (w == 0) continue;
                u /= w;
                v /= w;
                
                // 检查是否在局部图像范围内
                if (u >= 0 && u < local_sizes(idx, 0) && v >= 0 && v < local_sizes(idx, 1)) {
                    // 计算到图像中心的距离
                    double center_x = local_sizes(idx, 0) / 2.0;
                    double center_y = local_sizes(idx, 1) / 2.0;
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
            result_ptr[y * canvas_width + x] = best_result;
        }
    }
    
    return result;
}

PYBIND11_MODULE(reverse_mapping_core, m) {
    m.def("compute_reverse_mapping", &compute_reverse_mapping, 
          "Compute reverse mapping for image stitching demo");
    
    py::class_<PixelResult>(m, "PixelResult")
        .def_readwrite("local_idx", &PixelResult::local_idx)
        .def_readwrite("u", &PixelResult::u)
        .def_readwrite("v", &PixelResult::v);
}