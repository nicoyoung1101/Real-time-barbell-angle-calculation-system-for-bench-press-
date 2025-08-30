import cv2
import h5py
import numpy as np
import math

# 配置路径
video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
result_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5'
output_video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/Degree_display.mp4'

# 读取 HDF5 数据
with h5py.File(result_path, 'r') as h5file:
    table = h5file['df_with_missing/table'][:]
    keypoints = table['values_block_0']

# 提取关键点数据
left_x, left_y = keypoints[:, 0], keypoints[:, 1]
middle_x, middle_y = keypoints[:, 3], keypoints[:, 4]
right_x, right_y = keypoints[:, 6], keypoints[:, 7]

# 打开视频
cap = cv2.VideoCapture(video_path)

# 获取视频信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 准备视频写入器
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def calculate_angle(p1, p2, is_left=True):
    """
    计算线段与水平线的夹角
    p1, p2: 线段的两个端点坐标 (x, y)
    is_left: 是否是左侧的角度
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(-dy, dx))
    
    # 对左侧角度进行调整：取反即可
    if is_left:
        angle = -angle
            
    return angle

# 水平线的长度（像素）
horizontal_line_length = 100
# 线条颜色
line_color = (0, 255, 0)  # 绿色
text_color = (0, 0, 0)    # 黑色

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(left_x):
        break

    try:
        x = [left_x[frame_idx], middle_x[frame_idx], right_x[frame_idx]]
        y = [left_y[frame_idx], middle_y[frame_idx], right_y[frame_idx]]
        
        # 绘制骨架线
        cv2.line(frame, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), line_color, 2)
        cv2.line(frame, (int(x[1]), int(y[1])), (int(x[2]), int(y[2])), line_color, 2)
        
        # 在中点绘制水平线
        middle_x_int = int(x[1])
        middle_y_int = int(y[1])
        horizontal_start = (middle_x_int - horizontal_line_length//2, middle_y_int)
        horizontal_end = (middle_x_int + horizontal_line_length//2, middle_y_int)
        cv2.line(frame, horizontal_start, horizontal_end, (0, 255, 255), 2)  # 黄色水平线

        # 计算左侧和右侧线段与水平线的夹角
        left_angle = calculate_angle((x[0], y[0]), (x[1], y[1]), is_left=True)
        right_angle = calculate_angle((x[1], y[1]), (x[2], y[2]), is_left=False)

        # 在画面上显示角度（纯黑色文字）
        cv2.putText(frame, f"L: {left_angle:.1f}deg", 
                   (middle_x_int - 120, middle_y_int - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.putText(frame, f"R: {right_angle:.1f}deg", 
                   (middle_x_int + 20, middle_y_int - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 显示帧数信息
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
    except IndexError:
        print("帧索引超出范围，处理完成")
        break

    # 写入视频
    out.write(frame)
    frame_idx += 1

    print(f"正在处理帧 {frame_idx}/{total_frames}", end='\r')

cap.release()
out.release()

print("\n视频生成完成，保存路径为：", output_video_path)
