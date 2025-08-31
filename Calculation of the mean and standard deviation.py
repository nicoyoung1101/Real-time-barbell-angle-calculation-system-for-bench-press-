import pandas as pd
import numpy as np

def analyze_barbell_angles(input_file, video_height=1342):
    """
    分析杠铃杆角度数据
    
    参数:
    input_file: h5文件路径
    video_height: 视频高度，默认1342但不影响相对角度计算
    
    返回:
    包含左右角度统计信息的字典
    """
    try:
        # 读取 h5 文件
        print(f"Reading data from {input_file}...")
        data = pd.read_hdf(input_file)

        # 检查是否为多级索引
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Input H5 file does not have a MultiIndex structure!")

        # 提取索引层次
        scorer = data.columns.levels[0][0]  # 假设只有一个 scorer
        
        # 创建DataFrame的副本以避免警告
        df = data.copy()

        # 转换Y坐标到数学坐标系统
        left_y_math = video_height - df[(scorer, 'Left', 'y')]
        middle_y_math = video_height - df[(scorer, 'Middle', 'y')]
        right_y_math = video_height - df[(scorer, 'Right', 'y')]

        # 计算相对于中点的dx和dy
        df['left_dx'] = abs(df[(scorer, 'Left', 'x')] - df[(scorer, 'Middle', 'x')])
        df['right_dx'] = abs(df[(scorer, 'Right', 'x')] - df[(scorer, 'Middle', 'x')])

        df['left_dy'] = left_y_math - middle_y_math
        df['right_dy'] = right_y_math - middle_y_math

        # 计算角度
        df['left_angle_deg'] = np.degrees(np.arctan(df['left_dy'] / df['left_dx']))
        df['right_angle_deg'] = np.degrees(np.arctan(df['right_dy'] / df['right_dx']))

        # 计算统计值
        stats = {
            'left_angle_mean': round(df['left_angle_deg'].mean(), 2),
            'left_angle_std': round(df['left_angle_deg'].std(), 2),
            'right_angle_mean': round(df['right_angle_deg'].mean(), 2),
            'right_angle_std': round(df['right_angle_deg'].std(), 2)
        }

        print("\nResults:")
        print(f"Left Angle: {stats['left_angle_mean']}° ± {stats['left_angle_std']}°")
        print(f"Right Angle: {stats['right_angle_mean']}° ± {stats['right_angle_std']}°")

        return stats

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    input_file = "/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5"  # 替换为你的h5文件路径
    results = analyze_barbell_angles(input_file)
