import cv2
import numpy as np
import time
import os
import json

# 导入我们已编译好的C++模块
import reverse_mapping_core 

# --- 函数区 ---
def load_homographies(h_dir, num_cameras):
    """加载H矩阵"""
    # ... (无变化)
    homographies = []
    h_filenames = [f"{p}{s}_H_init.json" for p in ['1','2','3'] for s in range(1, 6)]
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
    """预计算映射表"""
    # ... (无变化)
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

def render_canvas_from_maps(canvas, local_images, map_x, map_y, masks):
    """从映射表高效渲染画布"""
    # ... (无变化, 移除了计时)
    for idx, local_img in enumerate(local_images):
        if not np.any(masks[idx]):
            continue
        remapped_img = cv2.remap(
            local_img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        canvas[masks[idx]] = remapped_img[masks[idx]]

# --- 新增的交互状态管理器 ---
class ViewInteractor:
    """管理缩放和平移的状态与逻辑"""
    def __init__(self, width, height, window_name):
        self.width = width
        self.height = height
        self.window_name = window_name
        self.reset()
        
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def reset(self):
        """重置视图状态"""
        self.zoom = 1.0
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        print("View reset to default.")

    def get_view(self, source_canvas):
        """根据当前状态，从原始画布中提取并返回用于显示的视图"""
        if self.zoom == 1.0:
            return source_canvas

        view_w = int(self.width / self.zoom)
        view_h = int(self.height / self.zoom)
        
        x1 = int(self.center_x - view_w / 2)
        y1 = int(self.center_y - view_h / 2)
        
        # 边界检查，防止裁切区域超出原始画布
        x1 = max(0, min(x1, self.width - view_w))
        y1 = max(0, min(y1, self.height - view_h))
        
        # 更新中心点以匹配有效区域
        self.center_x = x1 + view_w // 2
        self.center_y = y1 + view_h // 2
        
        x2, y2 = x1 + view_w, y1 + view_h
        
        roi = source_canvas[y1:y2, x1:x2]
        # 使用 INTER_NEAREST 可以保留像素感，速度最快
        zoomed_view = cv2.resize(roi, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        return zoomed_view

    def handle_key(self, key):
        """处理键盘输入"""
        pan_step = int(50 / self.zoom)
        zoom_factor = 1.1

        if key in [ord('+'), ord('=')]: self.zoom *= zoom_factor
        elif key in [ord('-'), ord('_')]: self.zoom /= zoom_factor
        elif key in [65363, 2555904]: self.center_x += pan_step # Right arrow
        elif key in [65361, 2424832]: self.center_x -= pan_step # Left arrow
        elif key in [65362, 2490368]: self.center_y -= pan_step # Up arrow
        elif key in [65364, 2621440]: self.center_y += pan_step # Down arrow
        elif key == ord('r'): self.reset()

        self.zoom = max(1.0, self.zoom) # 最小缩放为1.0
            
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.panning = True
            self.pan_start_x = x
            self.pan_start_y = y
        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx, dy = x - self.pan_start_x, y - self.pan_start_y
            self.center_x -= int(dx / self.zoom)
            self.center_y -= int(dy / self.zoom)
            self.pan_start_x, self.pan_start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1
            if flags > 0: self.zoom *= zoom_factor
            else: self.zoom /= zoom_factor
            self.zoom = max(1.0, self.zoom)

def main():
    # --- 1. 参数配置 ---
    VIDEO_SOURCE_DIR = r"D:\CTS\to_cts\video_stream\20201024\20201024084037322\main"
    BASE_DIR = r"D:\CTS\Likun_Hetu\code\ReverseMappingDemo\pictures"
    NUM_CAMERAS = 15
    NUM_FRAMES = 100
    CANVAS_SIZE = (1980, 1080)
    GLOBAL_SIZE = (4000, 3000)
    WINDOW_NAME = "Interactive CPU Stitching Player"

    # --- 2. 加载与预计算 ---
    # ... (与之前相同, 省略)
    homography_dir = os.path.join(BASE_DIR, "homography_matrices")
    homographies = load_homographies(homography_dir, NUM_CAMERAS)
    if not homographies: return
    H_invs_array = np.array([np.linalg.inv(H) for H in homographies], dtype=np.float64)
    video_captures = []
    video_filenames = [f"{p}{s}.mp4" for p in ['1','2','3'] for s in range(1, 6)]
    for filename in video_filenames:
        video_path = os.path.join(VIDEO_SOURCE_DIR, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error opening {video_path}"); return
        video_captures.append(cap)
    first_frame_images = [cap.read()[1] for cap in video_captures]
    for cap in video_captures: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    local_image_shapes = [img.shape for img in first_frame_images]
    map_x, map_y, masks = precompute_stitching_maps(CANVAS_SIZE, GLOBAL_SIZE, local_image_shapes, H_invs_array)

    # --- 3. 初始化交互 & 主循环 ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    interactor = ViewInteractor(CANVAS_SIZE[0], CANVAS_SIZE[1], WINDOW_NAME)
    source_canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    
    start_time = time.time()
    for frame_idx in range(1, NUM_FRAMES + 1):
        # 从视频流高效读取帧
        local_images_current_frame = [cap.read()[1] for cap in video_captures]
        if any(frame is None for frame in local_images_current_frame): break

        # 1. 渲染完整的1080p原始画布
        render_canvas_from_maps(source_canvas, local_images_current_frame, map_x, map_y, masks)

        # 2. 从原始画布中获取当前的交互式视图
        display_view = interactor.get_view(source_canvas)

        # 3. 在最终视图上绘制UI信息
        fps = frame_idx / (time.time() - start_time) if time.time() - start_time > 0 else 0
        cv2.putText(display_view, f"FPS: {fps:.2f} Zoom: {interactor.zoom:.2f}x", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_view, "Scroll/+-:Zoom | Drag/Arrows:Pan | R:Reset | Q:Quit", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, display_view)
        
        # 处理键盘输入 (注意：不同系统箭头键值可能不同)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
        if key != -1: # 仅在有按键时调用处理函数
            interactor.handle_key(key)
        
    # --- 4. 收尾 ---
    for cap in video_captures: cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()