import cv2
import numpy as np
import time
import os
import json
from tqdm import tqdm # 用于显示进度条，pip install tqdm

# 导入我们已编译好的C++模块
import reverse_mapping_core 

def load_homographies(h_dir, num_cameras):
    """
    从指定目录加载所有相机的单应性矩阵JSON文件。
    """
    homographies = []
    print(f"Loading homographies from {h_dir}...")
    for i in range(1, num_cameras + 1):
        # 假设H矩阵文件命名为 H_1.json, H_2.json ...
        h_path = os.path.join(h_dir, f"H_{i}.json")
        try:
            with open(h_path, 'r') as f:
                data = json.load(f)
                # 根据你提供的JSON结构，提取矩阵
                h_matrix = np.array(data["H_further_optimized"])
                homographies.append(h_matrix)
        except FileNotFoundError:
            print(f"Error: Homography file not found at {h_path}")
            return None
        except KeyError:
            print(f"Error: JSON file {h_path} does not contain the key 'H_further_optimized'")
            return None
    
    print(f"Successfully loaded {len(homographies)} homography matrices.")
    return homographies

def stitch_canvas_with_reverse_mapping(canvas_size, global_size, local_images, H_invs_array):
    """
    使用反向映射和C++核心模块来为单帧生成全景图。
    这个函数现在更纯粹，只负责计算。
    """
    canvas_width, canvas_height = canvas_size
    global_width, global_height = global_size

    # 1. 准备C++模块所需的输入数据
    local_sizes = [[img.shape[1], img.shape[0]] for img in local_images] # (width, height)
    local_sizes_array = np.array(local_sizes, dtype=np.int32)

    # 2. 调用C++核心模块，计算反向映射
    mapping = reverse_mapping_core.compute_reverse_mapping(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        global_width=global_width,
        global_height=global_height,
        H_invs_array=H_invs_array,
        local_sizes_array=local_sizes_array
    )

    # 3. 高性能渲染：使用cv2.remap生成画布
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 准备remap所需的映射表
    map_x = mapping['u'].astype(np.float32)
    map_y = mapping['v'].astype(np.float32)

    for idx, local_img in enumerate(local_images):
        mask = (mapping['local_idx'] == idx)
        if not np.any(mask):
            continue
        
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        canvas[mask] = remapped_img[mask]
        
    return canvas

def main():
    # --- 1. 参数配置 ---
    BASE_DIR = r"E:\your_data_root_path" # <--- 修改为你的数据根目录
    NUM_CAMERAS = 15
    NUM_FRAMES = 100
    
    # Demo画布尺寸
    CANVAS_SIZE = (1280, 720) # (width, height)
    
    # 全局坐标系尺寸，H矩阵是基于这个坐标系计算的
    GLOBAL_SIZE = (4000, 3000) # (width, height) <--- 根据你的实际情况修改

    # 输出视频设置
    OUTPUT_VIDEO_PATH = "stitched_video.mp4"
    OUTPUT_FPS = 30

    # --- 2. 构建文件路径 ---
    # 局部相机文件夹路径，假设命名为 local_cam_1, local_cam_2, ...
    local_cam_dirs = [os.path.join(BASE_DIR, f"local_cam_{i}") for i in range(1, NUM_CAMERAS + 1)]
    # 单应性矩阵文件夹路径
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    
    # --- 3. 初始化：加载H矩阵 ---
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if homographies is None:
        return
        
    # 预先计算好所有逆矩阵，避免在循环中重复计算
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)

    # --- 4. 初始化视频写入器 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_FPS, CANVAS_SIZE)
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path {OUTPUT_VIDEO_PATH}")
        return

    # --- 5. 主循环：逐帧处理 ---
    print("Starting frame-by-frame stitching process...")
    for frame_idx in tqdm(range(1, NUM_FRAMES + 1), desc="Stitching Frames"):
        # 加载当前帧对应的15张局部图像
        local_images_current_frame = []
        valid_frame = True
        for i in range(NUM_CAMERAS):
            # 假设图像命名为 001.jpg, 002.jpg ...
            img_path = os.path.join(local_cam_dirs[i], f"{frame_idx:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping frame {frame_idx}.")
                valid_frame = False
                break
            local_images_current_frame.append(img)
        
        if not valid_frame:
            continue

        # 为当前帧执行拼接
        stitched_frame = stitch_canvas_with_reverse_mapping(
            canvas_size=CANVAS_SIZE,
            global_size=GLOBAL_SIZE,
            local_images=local_images_current_frame,
            H_invs_array=H_invs_array
        )
        
        # 将合成的帧写入视频
        video_writer.write(stitched_frame)
        
        # (可选) 实时显示
        # cv2.imshow("Stitched Video", stitched_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # --- 6. 收尾 ---
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nStitching complete. Video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()