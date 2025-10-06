#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

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
    py::array_t<double, py::array::c_style | py::array::forcecast> H_invs_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> local_sizes_array
) {
    auto H_invs = H_invs_array.unchecked<3>();
    auto local_sizes = local_sizes_array.unchecked<2>();
    
    int num_locals = H_invs.shape(0);
    
    auto result = py::array_t<PixelResult>({canvas_height, canvas_width});
    auto result_ptr = result.mutable_data();
    
    double scale_x = static_cast<double>(global_width) / canvas_width;
    double scale_y = static_cast<double>(global_height) / canvas_height;
    
    // 修正: 移除 collapse(2) 以消除警告 C4849
    #pragma omp parallel for
    for (int y = 0; y < canvas_height; ++y) {
        for (int x = 0; x < canvas_width; ++x) {
            double global_x = x * scale_x;
            double global_y = y * scale_y;
            
            PixelResult best_result = {-1, -1.0f, -1.0f};
            double min_dist = std::numeric_limits<double>::max();
            
            for (int idx = 0; idx < num_locals; ++idx) {
                // 关键修正: 直接使用 H_invs(idx, row, col) 访问数据
                double u_prime = H_invs(idx, 0, 0) * global_x + H_invs(idx, 0, 1) * global_y + H_invs(idx, 0, 2);
                double v_prime = H_invs(idx, 1, 0) * global_x + H_invs(idx, 1, 1) * global_y + H_invs(idx, 1, 2);
                double w_prime = H_invs(idx, 2, 0) * global_x + H_invs(idx, 2, 1) * global_y + H_invs(idx, 2, 2);
                
                if (w_prime == 0) continue;
                
                double u = u_prime / w_prime;
                double v = v_prime / w_prime;
                
                if (u >= 0 && u < local_sizes(idx, 0) && v >= 0 && v < local_sizes(idx, 1)) {
                    double center_x = local_sizes(idx, 0) / 2.0;
                    double center_y = local_sizes(idx, 1) / 2.0;
                    double dist = std::sqrt((u - center_x) * (u - center_x) + (v - center_y) * (v - center_y));
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_result = {idx, static_cast<float>(u), static_cast<float>(v)};
                    }
                }
            }
            result_ptr[y * canvas_width + x] = best_result;
        }
    }
    
    return result;
}

PYBIND11_MODULE(reverse_mapping_core, m) {
    m.doc() = "A high-performance reverse mapping module for image stitching using pybind11 and OpenMP";

    PYBIND11_NUMPY_DTYPE(PixelResult, local_idx, u, v);

    m.def("compute_reverse_mapping", &compute_reverse_mapping, 
          "Compute reverse mapping from canvas to local images.",
          py::arg("canvas_width"),
          py::arg("canvas_height"),
          py::arg("global_width"),
          py::arg("global_height"),
          py::arg("H_invs_array"),
          py::arg("local_sizes_array")
    );
    
    py::class_<PixelResult>(m, "PixelResult")
        .def(py::init<>())
        .def_readwrite("local_idx", &PixelResult::local_idx)
        .def_readwrite("u", &PixelResult::u)
        .def_readwrite("v", &PixelResult::v)
        .def("__repr__",
            [](const PixelResult &p) {
                return "<PixelResult: local_idx=" + std::to_string(p.local_idx) +
                       ", u=" + std::to_string(p.u) + ", v=" + std::to_string(p.v) + ">";
            }
        );
}