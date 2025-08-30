import cv2
import numpy as np
import os

class BarbellTracker:
    def __init__(self):
        # 初始化 ArUco 检测器
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # 存储轨迹
        self.trajectory = []
        self.max_trajectory_points = 50
        
        # 确保标记已生成
        self.generate_markers_if_needed()
        
    def generate_markers_if_needed(self):
        """生成 ArUco 标记如果它们不存在"""
        markers_dir = "markers"
        if not os.path.exists(markers_dir):
            os.makedirs(markers_dir)
            
        for i in range(3):
            marker_path = os.path.join(markers_dir, f'marker_{i}.png')
            if not os.path.exists(marker_path):
                marker = np.zeros((300, 300), dtype=np.uint8)
                marker = cv2.aruco.generateImageMarker(self.aruco_dict, i, 300, marker, 1)
                cv2.imwrite(marker_path, marker)
                print(f"Generated marker {i} at {marker_path}")

    def detect_markers(self, frame):
        """检测帧中的 ArUco 标记"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        points = []
        if ids is not None:
            for i in range(len(ids)):
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                points.append((center_x, center_y, ids[i][0]))
            
            points.sort(key=lambda x: x[2])
        
        return [(p[0], p[1]) for p in points]

    def draw_markers_and_lines(self, frame, points):
        """在帧上绘制标记点、连线和轨迹"""
        if len(points) == 3:
            # 绘制标记点
            colors = [(255,0,0), (0,255,0), (0,0,255)]  # BGR颜色
            labels = ["Left", "Middle", "Right"]
            
            for i, point in enumerate(points):
                cv2.circle(frame, point, 5, colors[i], -1)
                cv2.putText(frame, labels[i], 
                          (point[0] + 10, point[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # 绘制连接线
            cv2.line(frame, points[0], points[1], (255,0 , 255), 2)
            cv2.line(frame, points[1], points[2], (255, 0, 255), 2)
            
            # 更新并绘制轨迹
            self.trajectory.append(points[1])
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
            
            for i in range(1, len(self.trajectory)):
                cv2.line(frame, 
                        self.trajectory[i-1], 
                        self.trajectory[i], 
                        (255, 0, 0), 1)

    def process_video(self, video_path=0, output_path=None):
        """处理视频或摄像头输入"""
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 设置视频写入器
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
        
        # 用于计算实际帧率
        prev_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 计算实际帧率
            current_time = cv2.getTickCount()
            actual_fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time
            
            # 检测标记
            points = self.detect_markers(frame)
            
            # 绘制标记和连线
            self.draw_markers_and_lines(frame, points)
            
            # 显示信息
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            status_color = (0, 255, 0) if len(points) == 3 else (0, 0, 255)
            cv2.putText(frame, f"Detected: {len(points)}/3", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # 显示结果
            cv2.imshow('Barbell Tracking', frame)
            
            # 保存视频
            if out:
                out.write(frame)
            
            # 按'q'退出，按's'保存当前帧
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('frame_capture.jpg', frame)
                print("Frame saved as frame_capture.jpg")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

def print_instructions():
    """打印使用说明"""
    print("\n=== Barbell Tracking System ===")
    print("1. 标记已生成在 'markers' 文件夹中")
    print("2. 请按以下顺序安装标记：")
    print("   - marker_0.png -> 杠铃左端")
    print("   - marker_1.png -> 杠铃中间")
    print("   - marker_2.png -> 杠铃右端")
    print("\n控制键:")
    print("- 按 'q' 退出程序")
    print("- 按 's' 保存当前帧")
    print("========================\n")

if __name__ == "__main__":
    # 设置固定的视频路径
    video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
    
    # 创建跟踪器实例
    tracker = BarbellTracker()
    
    # 打印使用说明
    print_instructions()
    
    # 设置输出路径
    output_path = "/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/Outputvideo.mp4"
    
    # 开始处理
    try:
        tracker.process_video(video_path, output_path)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("\n程序已结束")
