import cv2
import h5py
import numpy as np

# 配置路径
video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
result_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5'
output_video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/overlay_video_two_phases.mp4'

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

# 计算两个阶段的帧范围
phase1_start = int(1 * fps)  # 第一阶段开始（1秒）
phase1_end = int(5 * fps)    # 第一阶段结束（5秒）
phase2_start = int(5 * fps)  # 第二阶段开始（6秒）
phase2_end = int(8 * fps)    # 第二阶段结束（8秒）

# 准备视频写入器
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定义每个阶段的颜色
phase1_color = (255, 0, 255)    # 蓝色用于第一阶段
phase2_color = (255, 0, 255)    # 绿色用于第二阶段

# 处理两个阶段
for phase in [1, 2]:
    # 为每个阶段创建新的叠加层
    overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # 设置阶段参数
    if phase == 1:
        start_frame = phase1_start
        end_frame = phase1_end
        color = phase1_color
    else:
        start_frame = phase2_start
        end_frame = phase2_end
        color = phase2_color
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    
    # 重置上一帧的坐标
    last_x = None
    last_y = None
    
    # 处理当前阶段的帧
    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(left_x):
            break

        try:
            # 获取当前帧的关键点
            x = [left_x[frame_idx], middle_x[frame_idx], right_x[frame_idx]]
            y = [left_y[frame_idx], middle_y[frame_idx], right_y[frame_idx]]
            
            # 绘制骨架线
            cv2.line(overlay, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, 2)
            cv2.line(overlay, (int(x[1]), int(y[1])), (int(x[2]), int(y[2])), color, 2)
            
            # 如果有上一帧的坐标，绘制轨迹线
            if last_x is not None:
                for i in range(3):
                    cv2.line(overlay, 
                            (int(last_x[i]), int(last_y[i])), 
                            (int(x[i]), int(y[i])), 
                            color, 1)
            
            # 更新上一帧的坐标
            last_x = x.copy()
            last_y = y.copy()
            
        except IndexError:
            print(f"第{phase}阶段帧索引超出范围，处理完成")
            break

        # 叠加图层
        combined = cv2.addWeighted(frame, 0.6, overlay, 0.9, 0)

        # 写入视频
        out.write(combined)
        frame_idx += 1

        # 打印进度
        print(f"正在处理第{phase}阶段：帧 {frame_idx}/{end_frame}", end='\r')

cap.release()
out.release()

print("\n视频生成完成，保存路径为：", output_video_path)
