import cv2
import numpy as np
import time
import os
import json
import threading
from queue import Queue
from collections import deque

# 导入我们已编译好的C++模块
import reverse_mapping_core 

# --- 函数部分保持不变 ---
def load_homographies(h_dir, num_cameras):
    # (此函数无变化)
    homographies = []
    h_filenames = []
    prefixes = ['1', '2', '3']
    for prefix in prefixes:
        for suffix in range(1, 6):
            filename = f"{prefix}{suffix}_H_init.json"
            h_filenames.append(filename)
    print(f"Loading homographies from {h_dir}...")
    for filename in h_filenames:
        h_path = os.path.join(h_dir, filename)
        try:
            with open(h_path, 'r') as f:
                data = json.load(f)
                h_matrix = np.array(data["H_further_optimized"])
                homographies.append(h_matrix)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading {h_path}: {e}")
            return None
    print(f"Successfully loaded {len(homographies)} homography matrices.")
    return homographies

def precompute_stitching_maps(canvas_size, global_size, local_image_shapes, H_invs_array):
    # (此函数无变化)
    print("Pre-computing static stitching maps... (This may take a moment)")
    canvas_width, canvas_height = canvas_size
    global_width, global_height = global_size
    local_sizes = [[shape[1], shape[0]] for shape in local_image_shapes]
    local_sizes_array = np.array(local_sizes, dtype=np.int32)
    
    # --- 关键诊断问题 #1 ---
    # 我们需要知道这个C++核心计算到底花了多长时间
    cpp_start_time = time.time()
    mapping = reverse_mapping_core.compute_reverse_mapping(
        canvas_width=canvas_width, canvas_height=canvas_height,
        global_width=global_width, global_height=global_height,
        H_invs_array=H_invs_array, local_sizes_array=local_sizes_array
    )
    cpp_end_time = time.time()
    print(f"!!! DIAGNOSTIC: C++ mapping computation took: {cpp_end_time - cpp_start_time:.4f} seconds.")
    
    map_x = mapping['u'].astype(np.float32)
    map_y = mapping['v'].astype(np.float32)
    masks = [(mapping['local_idx'] == idx) for idx in range(len(local_image_shapes))]
    print("Static maps pre-computation complete.")
    return map_x, map_y, masks

# --- 新增的带计时的渲染函数 ---
def render_canvas_with_profiling(canvas, local_images, map_x, map_y, masks):
    """
    带内部计时的渲染函数，用于诊断 remap 和 mask assignment 的耗时
    """
    total_remap_time = 0
    total_masking_time = 0

    for idx, local_img in enumerate(local_images):
        if not np.any(masks[idx]):
            continue
        
        # 计时 remap
        remap_start = time.perf_counter()
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        total_remap_time += (time.perf_counter() - remap_start)

        # 计时掩码赋值
        masking_start = time.perf_counter()
        canvas[masks[idx]] = remapped_img[masks[idx]]
        total_masking_time += (time.perf_counter() - masking_start)
    
    return total_remap_time, total_masking_time

# --- 新增的带计时的多线程加载器 ---
def image_loader_thread_with_profiling(frame_queue, local_cam_dirs, num_frames, num_cameras, timings):
    """
    带内部计时的加载器线程
    """
    for frame_idx in range(1, num_frames + 1):
        io_start = time.perf_counter()
        image_bundle = []
        for i in range(num_cameras):
            img_path = os.path.join(local_cam_dirs[i], f"{frame_idx:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                frame_queue.put(None) 
                return
            image_bundle.append(img)
        
        # 记录I/O耗时
        timings['io'].append(time.perf_counter() - io_start)
        frame_queue.put((frame_idx, image_bundle))
    
    frame_queue.put(None)

def main():
    # --- 参数配置 ---
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"
    NUM_CAMERAS = 15
    NUM_FRAMES = 100
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000)

    # ... 省略路径构建和H矩阵加载 ...
    local_cam_dirs = [os.path.join(BASE_DIR, f"local_cam_{i}") for i in range(1, NUM_CAMERAS + 1)]
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if not homographies: return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)
    first_frame_images = [cv2.imread(os.path.join(local_cam_dirs[i], "001.jpg")) for i in range(NUM_CAMERAS)]
    if any(img is None for img in first_frame_images): return
    local_image_shapes = [img.shape for img in first_frame_images]
    map_x, map_y, masks = precompute_stitching_maps(CANVAS_SIZE, GLOBAL_SIZE, local_image_shapes, H_invs_array)

    # --- 性能剖析数据结构 ---
    profiling_timings = {
        'io': deque(),
        'render': deque(),
        'remap': deque(),
        'masking': deque(),
        'full_loop': deque()
    }
    
    # --- 启动带计时的加载器 ---
    frame_queue = Queue(maxsize=10) 
    loader = threading.Thread(target=image_loader_thread_with_profiling, 
                              args=(frame_queue, local_cam_dirs, NUM_FRAMES, NUM_CAMERAS, profiling_timings))
    loader.daemon = True
    loader.start()

    cv2.namedWindow("Profiling Real-time Stitching", cv2.WINDOW_NORMAL)
    
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    frame_count = 0
    start_time = time.time()
    
    while frame_count < NUM_FRAMES:
        loop_start_time = time.perf_counter()
        bundle = frame_queue.get()
        if bundle is None: break
        frame_idx, local_images_current_frame = bundle
        
        # 计时渲染部分
        render_start_time = time.perf_counter()
        remap_time, masking_time = render_canvas_with_profiling(canvas, local_images_current_frame, map_x, map_y, masks)
        profiling_timings['render'].append(time.perf_counter() - render_start_time)
        profiling_timings['remap'].append(remap_time)
        profiling_timings['masking'].append(masking_time)

        frame_count += 1
        # ... 省略putText和imshow ...
        cv2.imshow("Profiling Real-time Stitching", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        profiling_timings['full_loop'].append(time.perf_counter() - loop_start_time)

    # --- 5. 结束时打印性能剖析报告 ---
    cv2.destroyAllWindows()
    loader.join(timeout=1)

    print("\n--- PERFORMANCE PROFILING REPORT ---")
    if profiling_timings['io']:
        avg_io = np.mean(profiling_timings['io']) * 1000
        print(f"Avg. I/O Time (15 reads): {avg_io:.2f} ms")
    if profiling_timings['remap']:
        avg_remap = np.mean(profiling_timings['remap']) * 1000
        print(f"Avg. Remap Time (15 calls): {avg_remap:.2f} ms")
    if profiling_timings['masking']:
        avg_masking = np.mean(profiling_timings['masking']) * 1000
        print(f"Avg. Masking Time (15 assignments): {avg_masking:.2f} ms")
    if profiling_timings['render']:
        avg_render = np.mean(profiling_timings['render']) * 1000
        print(f"Avg. Total Render Function Time: {avg_render:.2f} ms")
    if profiling_timings['full_loop']:
        avg_loop = np.mean(profiling_timings['full_loop']) * 1000
        total_fps = 1000 / avg_loop if avg_loop > 0 else 0
        print(f"Avg. Full Consumer Loop Time: {avg_loop:.2f} ms  (Implies FPS: {total_fps:.2f})")
    print("------------------------------------")

if __name__ == '__main__':
    main()