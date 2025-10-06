import cv2
import numpy as np
import time
import os
import json
import threading
from queue import Queue

# 导入我们已编译好的C++模块
import reverse_mapping_core 

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
    # (此函数无变化, 依然是核心优化)
    print("Pre-computing static stitching maps... (This may take a moment)")
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

def render_canvas_from_maps(canvas, local_images, map_x, map_y, masks):
    # (此函數無變化, 負責高效渲染)
    for idx, local_img in enumerate(local_images):
        if not np.any(masks[idx]):
            continue
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        canvas[masks[idx]] = remapped_img[masks[idx]]

# --- 新增的多线程图像加载器 (生产者) ---
def image_loader_thread(frame_queue, local_cam_dirs, num_frames, num_cameras):
    """
    这个函数在子线程中运行，负责读取图像并放入队列。
    """
    for frame_idx in range(1, num_frames + 1):
        image_bundle = []
        for i in range(num_cameras):
            img_path = os.path.join(local_cam_dirs[i], f"{frame_idx:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Loader Thread: Failed to load {img_path}, stopping.")
                # 发送停止信号
                frame_queue.put(None) 
                return
            image_bundle.append(img)
        
        # 将帧序号和图像数据一同放入队列
        frame_queue.put((frame_idx, image_bundle))
    
    # 所有帧都加载完毕，发送停止信号
    frame_queue.put(None)

def main():
    # --- 1. 参数配置 ---
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"
    NUM_CAMERAS = 15
    NUM_FRAMES = 100
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000)

    # --- 2. 構建文件路徑 ---
    local_cam_dirs = [os.path.join(BASE_DIR, f"local_cam_{i}") for i in range(1, NUM_CAMERAS + 1)]
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    
    # --- 3. 初始化与预计算 ---
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if not homographies: return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)
    
    first_frame_images = [cv2.imread(os.path.join(local_cam_dirs[i], "001.jpg")) for i in range(NUM_CAMERAS)]
    if any(img is None for img in first_frame_images):
        print("Error: Could not read all first frame images for pre-computation.")
        return
    local_image_shapes = [img.shape for img in first_frame_images]
    map_x, map_y, masks = precompute_stitching_maps(CANVAS_SIZE, GLOBAL_SIZE, local_image_shapes, H_invs_array)

    # --- 4. 启动多线程加载器 ---
    # 创建一个有缓冲区的队列，防止加载过快导致内存爆炸
    frame_queue = Queue(maxsize=10) 
    loader = threading.Thread(target=image_loader_thread, 
                              args=(frame_queue, local_cam_dirs, NUM_FRAMES, NUM_CAMERAS))
    loader.daemon = True # 设置为守护线程，主程序退出时子线程也退出
    loader.start()

    # --- 5. 主循环 (消费者) ---
    cv2.namedWindow("Multi-threaded Real-time Stitching", cv2.WINDOW_NORMAL)
    print("\nStarting Multi-threaded real-time stitching...")
    print("Press 'q' in the video window to quit.")
    
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    frame_count = 0
    start_time = time.time()
    
    while True:
        # 从队列中获取数据，如果队列为空，这里会阻塞等待
        bundle = frame_queue.get()
        
        # 如果收到停止信号，则退出循环
        if bundle is None:
            break
            
        frame_idx, local_images_current_frame = bundle
        frame_count += 1
        
        render_canvas_from_maps(canvas, local_images_current_frame, map_x, map_y, masks)
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(canvas, f"Frame: {frame_idx}/{NUM_FRAMES} FPS: {fps:.2f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Multi-threaded Real-time Stitching", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

    # --- 6. 收尾 ---
    cv2.destroyAllWindows()
    loader.join(timeout=1) # 等待子线程结束
    # ... (收尾信息) ...

if __name__ == '__main__':
    main()