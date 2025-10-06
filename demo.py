import cv2
import numpy as np
import time
import os
import json

# 导入我们已编译好的C++模块
import reverse_mapping_core 

def load_homographies(h_dir, num_cameras):
    """
    加载 "11_H_init.json" 命名规则的单应性矩阵。(此函数无变化)
    """
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
        except FileNotFoundError:
            print(f"Error: Homography file not found at {h_path}")
            return None
        except KeyError:
            print(f"Error: JSON file {h_path} does not contain the key 'H_further_optimized'")
            return None
    
    if len(homographies) == num_cameras:
        print(f"Successfully loaded {len(homographies)} homography matrices.")
    else:
        print(f"Warning: Expected {num_cameras} matrices, but only found {len(homographies)}.")
    return homographies

def stitch_canvas_with_reverse_mapping(canvas_size, global_size, local_images, H_invs_array):
    """
    使用反向映射和C++核心模块来为单帧生成全景图。(此函数无变化)
    """
    canvas_width, canvas_height = canvas_size
    global_width, global_height = global_size
    local_sizes = [[img.shape[1], img.shape[0]] for img in local_images]
    local_sizes_array = np.array(local_sizes, dtype=np.int32)

    mapping = reverse_mapping_core.compute_reverse_mapping(
        canvas_width=canvas_width, canvas_height=canvas_height,
        global_width=global_width, global_height=global_height,
        H_invs_array=H_invs_array, local_sizes_array=local_sizes_array
    )
    
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    map_x = mapping['u'].astype(np.float32)
    map_y = mapping['v'].astype(np.float32)

    for idx, local_img in enumerate(local_images):
        mask = (mapping['local_idx'] == idx)
        if not np.any(mask):
            continue
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        canvas[mask] = remapped_img[mask]
    return canvas

def main():
    # --- 1. 参数配置 ---
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"  # <--- 修改为你的数据根目录
    NUM_CAMERAS = 15
    NUM_FRAMES = 100 # <--- 要处理的总帧数
    
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000)

    # --- 2. 构建文件路径 ---
    local_cam_dirs = [os.path.join(BASE_DIR, f"local_cam_{i}") for i in range(1, NUM_CAMERAS + 1)]
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    
    # --- 3. 初始化 ---
    # 加载H矩阵并只计算一次逆矩阵
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if homographies is None or len(homographies) != NUM_CAMERAS:
        print("Could not load all homography matrices. Exiting.")
        return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)

    # 创建一个可调整大小的窗口用于显示
    cv2.namedWindow("Real-time Stitched Video", cv2.WINDOW_NORMAL)

    # --- 4. 主循环：逐帧处理并实时播放 ---
    print("\nStarting real-time stitching and playback...")
    print("Press 'q' in the video window to quit.")
    
    start_time = time.time()
    for frame_idx in range(1, NUM_FRAMES + 1):
        
        # 加载当前帧对应的15张局部图像
        local_images_current_frame = []
        valid_frame = True
        for i in range(NUM_CAMERAS):
            img_path = os.path.join(local_cam_dirs[i], f"{frame_idx:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Stopping playback.")
                valid_frame = False
                break
            local_images_current_frame.append(img)
        
        if not valid_frame:
            break

        # 为当前帧执行拼接
        stitched_frame = stitch_canvas_with_reverse_mapping(
            canvas_size=CANVAS_SIZE,
            global_size=GLOBAL_SIZE,
            local_images=local_images_current_frame,
            H_invs_array=H_invs_array
        )
        
        # 计算并显示实时FPS (帧率)
        elapsed_time = time.time() - start_time
        fps = frame_idx / elapsed_time
        cv2.putText(stitched_frame, f"Frame: {frame_idx}/{NUM_FRAMES} FPS: {fps:.2f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 在窗口中显示拼接好的帧
        cv2.imshow("Real-time Stitched Video", stitched_frame)
        
        # 等待1毫秒，并检查是否有按键。如果按下'q'，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Playback stopped by user.")
            break

    # --- 5. 收尾 ---
    cv2.destroyAllWindows()
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = (NUM_FRAMES-1) / total_time if total_time > 0 else 0
    print("\nPlayback finished.")
    print(f"Processed {frame_idx-1} frames in {total_time:.2f} seconds (Average FPS: {avg_fps:.2f}).")


if __name__ == '__main__':
    main()