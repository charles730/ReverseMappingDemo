import cv2
import os
# 移除了: from tqdm import tqdm

# --- 1. 参数配置 (请根据你的实际情况修改) ---

# 存放你原始视频文件的目录 (例如，包含 11.mp4, 12.mp4, ..., 100.mp4)
SOURCE_VIDEO_DIR = r"D:\CTS\to_cts\video_stream\20201024\20201024084037322\main" 

# 你希望存放提取出的图像帧的根目录
# 脚本会自动在此目录下创建 local_cam_1, local_cam_2 等子文件夹
OUTPUT_BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures" # 这应该和你的 video_stitcher.py 脚本中的 BASE_DIR 一致

# 要提取的帧数
NUM_FRAMES_TO_EXTRACT = 100

# --- 2. 自动生成视频文件映射 (根据你的命名规则) ---
# 这部分代码保持不变，它会自动创建映射关系。

def generate_video_mapping():
    """
    根据 "11-15, 21-25, 31-35" 代表局部相机，"100" 代表全局相机的规则，
    自动生成 VIDEO_MAPPING 字典。
    """
    mapping = {}
    local_cam_index = 1
    
    video_prefixes = ['1', '2', '3']
    for prefix in video_prefixes:
        for suffix in range(1, 6):
            video_filename = f"{prefix}{suffix}.mp4"
            target_dir_name = f"local_cam_{local_cam_index}"
            mapping[target_dir_name] = video_filename
            local_cam_index += 1
            
    mapping["global_cam"] = "100.mp4"
    
    print("根据文件命名规则，自动生成的视频映射关系如下:")
    for key, value in mapping.items():
        print(f"  - {key} -> {value}")
    
    return mapping

VIDEO_MAPPING = generate_video_mapping()


# --- 脚本主逻辑 (这部分无需改动) ---

def extract_frames_from_video(video_path, output_dir, num_frames):
    """
    从单个视频文件中提取指定数量的帧并保存为图片。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 -> {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 -> {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频总帧数: {total_frames}, FPS: {fps:.2f}")

    if total_frames < num_frames:
        print(f"警告：视频总帧数 ({total_frames}) 小于要提取的帧数 ({num_frames})。将仅提取所有可用帧。")
        num_frames = total_frames

    for frame_count in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"警告：视频在提取到第 {frame_count} 帧时意外结束。")
            break
            
        output_filename = f"{frame_count + 1:03d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, frame)

    cap.release()
    print(f"成功从 {os.path.basename(video_path)} 提取了 {num_frames} 帧到目录 {output_dir}")

def main():
    """
    主函数，遍历所有视频并调用提取函数。
    """
    print("\n开始视频预处理：提取帧...")
    
    total_videos = len(VIDEO_MAPPING)
    current_video_num = 1
    
    # --- 这里是主要改动 ---
    # 移除了 tqdm，换成一个标准的 for 循环，并手动打印进度
    for target_dir_name, video_filename in VIDEO_MAPPING.items():
        
        # 打印进度信息
        print(f"\n--- [进度 {current_video_num}/{total_videos}] 正在处理视频: {video_filename} ---")
        
        source_video_path = os.path.join(SOURCE_VIDEO_DIR, video_filename)
        output_frames_dir = os.path.join(OUTPUT_BASE_DIR, target_dir_name)
        
        extract_frames_from_video(source_video_path, output_frames_dir, NUM_FRAMES_TO_EXTRACT)
        
        current_video_num += 1

    print("\n所有视频处理完毕！")
    print(f"数据已准备就绪，位于根目录: {OUTPUT_BASE_DIR}")
    print("现在你可以运行 video_stitcher.py 脚本了。")
    print(f"请确保你的单应性矩阵也已放置在: {os.path.join(OUTPUT_BASE_DIR, 'homography_matrices')}")


if __name__ == "__main__":
    main()