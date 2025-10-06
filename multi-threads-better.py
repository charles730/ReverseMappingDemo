import cv2
import numpy as np
import time
import os
import json
from collections import deque

# 导入我们已编译好的C++模块
import reverse_mapping_core 

# --- 函数区 ---
def load_homographies(h_dir, num_cameras):
    # (此函数无变化)
    homographies = []
    h_filenames = [f"{p}{s}_H_init.json" for p in ['1','2','3'] for s in range(1, 6)]
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
    print("Pre-computing static stitching maps...")
    canvas_width, canvas_height = canvas_size
    global_width, global_height = global_size
    local_sizes = [[shape[1], shape[0]] for shape in local_image_shapes]
    local_sizes_array = np.array(local_sizes, dtype=np.int32)
    start_time = time.time()
    mapping = reverse_mapping_core.compute_reverse_mapping(
        canvas_width=canvas_width, canvas_height=canvas_height,
        global_width=global_width, global_height=global_height,
        H_invs_array=H_invs_array, local_sizes_array=local_sizes_array
    )
    print(f"C++ mapping computation finished in {time.time() - start_time:.4f} seconds.")
    map_x = mapping['u'].astype(np.float32)
    map_y = mapping['v'].astype(np.float32)
    masks = [(mapping['local_idx'] == idx) for idx in range(len(local_image_shapes))]
    print("Static maps pre-computation complete.")
    return map_x, map_y, masks

def render_canvas_with_profiling(canvas, local_images, map_x, map_y, masks):
    """
    带内部计时的渲染函数，用于诊断 remap 和 mask assignment 的耗时
    """
    total_remap_time = 0
    total_masking_time = 0

    for idx, local_img in enumerate(local_images):
        if not np.any(masks[idx]):
            continue
        
        remap_start = time.perf_counter()
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        total_remap_time += (time.perf_counter() - remap_start)

        masking_start = time.perf_counter()
        canvas[masks[idx]] = remapped_img[masks[idx]]
        total_masking_time += (time.perf_counter() - masking_start)
    
    return total_remap_time, total_masking_time

def main():
    # --- 1. 参数配置 ---
    VIDEO_SOURCE_DIR = r"D:\CTS\to_cts\video_stream\20201024\20201024084037322\main" # <--- 包含 11.mp4, 12.mp4... 的文件夹
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"
    
    NUM_CAMERAS = 15
    NUM_FRAMES = 100
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000)

    # --- 2. 加载H矩阵 & 打开视频流 ---
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if not homographies: return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)

    video_captures = []
    video_filenames = [f"{p}{s}.mp4" for p in ['1','2','3'] for s in range(1, 6)]
    
    print("Opening 15 video streams...")
    for filename in video_filenames:
        video_path = os.path.join(VIDEO_SOURCE_DIR, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}"); return
        video_captures.append(cap)
        
    # --- 3. 预计算 ---
    first_frame_images = []
    for cap in video_captures:
        ret, frame = cap.read()
        if not ret: print("Error: Could not read the first frame."); return
        first_frame_images.append(frame)
    for cap in video_captures: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    local_image_shapes = [img.shape for img in first_frame_images]
    map_x, map_y, masks = precompute_stitching_maps(CANVAS_SIZE, GLOBAL_SIZE, local_image_shapes, H_invs_array)

    # --- 4. 主循环 (带性能剖析) ---
    profiling_timings = {
        'video_read': deque(),
        'render': deque(),
        'remap': deque(),
        'masking': deque(),
        'full_loop': deque()
    }
    
    cv2.namedWindow("Profiler on High-Performance Player", cv2.WINDOW_NORMAL)
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    
    for frame_idx in range(1, NUM_FRAMES + 1):
        loop_start_time = time.perf_counter()

        # 计时: 视频帧读取
        read_start_time = time.perf_counter()
        local_images_current_frame = []
        for cap in video_captures:
            ret, frame = cap.read()
            if not ret: break
            local_images_current_frame.append(frame)
        profiling_timings['video_read'].append(time.perf_counter() - read_start_time)
        
        if len(local_images_current_frame) != NUM_CAMERAS:
            print(f"\nOne of the videos ended before frame {frame_idx}. Stopping.")
            break

        # 计时: 渲染
        render_start_time = time.perf_counter()
        remap_time, masking_time = render_canvas_with_profiling(canvas, local_images_current_frame, map_x, map_y, masks)
        profiling_timings['render'].append(time.perf_counter() - render_start_time)
        profiling_timings['remap'].append(remap_time)
        profiling_timings['masking'].append(masking_time)

        # ... (显示逻辑) ...
        elapsed_time = (time.perf_counter() - loop_start_time)
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(canvas, f"Frame: {frame_idx}/{NUM_FRAMES} FPS: {fps:.2f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Profiler on High-Performance Player", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        profiling_timings['full_loop'].append(time.perf_counter() - loop_start_time)

    # --- 5. 收尾与报告 ---
    for cap in video_captures: cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- FINAL PERFORMANCE PROFILING REPORT (Reading from Videos) ---")
    if profiling_timings['video_read']:
        avg_io = np.mean(profiling_timings['video_read']) * 1000
        print(f"Avg. Video Read Time (15 streams): {avg_io:.2f} ms")
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
        print(f"Avg. Full Loop Time: {avg_loop:.2f} ms  (Implies Avg. FPS: {total_fps:.2f})")
    print("---------------------------------------------------------------")


if __name__ == '__main__':
    main()