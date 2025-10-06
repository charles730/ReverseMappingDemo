import cv2
import numpy as np
import time
import os
import json

# 导入我们已编译好的C++模块
import reverse_mapping_core 

def load_homographies(h_dir, num_cameras):
    """
    从指定目录加载所有相机的单应性矩阵JSON文件。
    --- 已根据新的 "11_H_init.json" 命名规则更新 ---
    """
    homographies = []
    
    # 1. 根据命名规则，按顺序生成15个文件名
    h_filenames = []
    prefixes = ['1', '2', '3']
    for prefix in prefixes:
        for suffix in range(1, 6):
            filename = f"{prefix}{suffix}_H_init.json"
            h_filenames.append(filename)
            
    print(f"Loading homographies from {h_dir} according to the new naming convention...")
    
    # 2. 遍历生成的文件名列表并加载文件
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
    
    # 检查是否成功加载了预期数量的矩阵
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

    print("Calling C++ core to compute reverse mapping...")
    start_time = time.time()
    mapping = reverse_mapping_core.compute_reverse_mapping(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        global_width=global_width,
        global_height=global_height,
        H_invs_array=H_invs_array,
        local_sizes_array=local_sizes_array
    )
    end_time = time.time()
    print(f"Reverse mapping computation finished in {end_time - start_time:.4f} seconds.")

    print("Rendering canvas using cv2.remap...")
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
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
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"  # <--- 修改为你的数据根目录
    NUM_CAMERAS = 15
    
    TEST_FRAME_IDX = 1  # <--- 修改为你想要测试的帧，例如 1, 50, 或 100
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000) # <--- 根据你的实际情况修改
    OUTPUT_IMAGE_PATH = "D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures\output\stitched_test_frame.jpg"

    # --- 2. 构建文件路径 ---
    local_cam_dirs = [os.path.join(BASE_DIR, f"local_cam_{i}") for i in range(1, NUM_CAMERAS + 1)]
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    
    # --- 3. 加载数据 ---
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if homographies is None or len(homographies) != NUM_CAMERAS:
        print("Could not load all homography matrices. Exiting.")
        return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)

    print(f"Loading images for frame number {TEST_FRAME_IDX}...")
    local_images_for_test = []
    for i in range(NUM_CAMERAS):
        img_path = os.path.join(local_cam_dirs[i], f"{TEST_FRAME_IDX:03d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}. Please check the path and frame index.")
            return
        local_images_for_test.append(img)
    print("All images for the test frame loaded successfully.")

    # --- 4. 执行拼接 ---
    stitched_frame = stitch_canvas_with_reverse_mapping(
        canvas_size=CANVAS_SIZE,
        global_size=GLOBAL_SIZE,
        local_images=local_images_for_test,
        H_invs_array=H_invs_array
    )
    
    # --- 5. 显示与保存结果 ---
    print(f"Stitching complete. Saving result to {OUTPUT_IMAGE_PATH}")
    cv2.imwrite(OUTPUT_IMAGE_PATH, stitched_frame)
    
    cv2.namedWindow("Stitched Test Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Stitched Test Frame", stitched_frame)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()