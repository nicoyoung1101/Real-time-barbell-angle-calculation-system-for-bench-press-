# 1. 导入必要的库
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from google.colab import files  # 导入 Colab 文件处理库

# 2. 数据处理与绘图函数

def moving_average(data, window_size):
    """应用移动平均法平滑数据"""
    if len(data) < window_size:
        raise ValueError("Data length must be greater than or equal to the window size.")
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def calculate_and_plot_angles(df, scorer, title, plot_filename, csv_filename):
    """
    一个辅助函数，用于从给定的 DataFrame 计算角度、保存 CSV 并绘图。
    :param df: 输入的 DataFrame (可以是原始的，也可以是平滑后的)
    :param scorer: DLC scorer 的名称
    :param title: 图表的标题
    :param plot_filename: 输出的图片文件名
    :param csv_filename: 输出的 CSV 文件名
    :return: 生成的 [csv_filename, plot_filename] 列表
    """
    print(f"\n--- Processing for: {title} ---")
    
    # ---- 角度计算 ----
    left_y_math = 1342 - df[(scorer, 'Left', 'y')]
    middle_y_math = 1342 - df[(scorer, 'Middle', 'y')]
    right_y_math = 1342 - df[(scorer, 'Right', 'y')]
    
    # 使用 .copy() 避免 SettingWithCopyWarning
    df_calc = df.copy()
    df_calc['left_dx'] = abs(df[(scorer, 'Left', 'x')] - df[(scorer, 'Middle', 'x')])
    df_calc['right_dx'] = abs(df[(scorer, 'Right', 'x')] - df[(scorer, 'Middle', 'x')])
    df_calc['left_dy'] = left_y_math - middle_y_math
    df_calc['right_dy'] = right_y_math - middle_y_math

    epsilon = 1e-9
    left_angle_deg = np.degrees(np.arctan(df_calc['left_dy'] / (df_calc['left_dx'] + epsilon)))
    right_angle_deg = np.degrees(np.arctan(df_calc['right_dy'] / (df_calc['right_dx'] + epsilon)))

    left_angle_deg = np.round(left_angle_deg, 2)
    right_angle_deg = np.round(right_angle_deg, 2)

    result = pd.DataFrame({
        'Frame': df.index,
        'left_angle_deg': left_angle_deg,
        'right_angle_deg': right_angle_deg
    })

    # 保存角度结果为 CSV 文件
    result.to_csv(csv_filename, index=False)
    print(f"Angle results saved to {csv_filename}")

    # --- 绘图部分 ---
    plt.figure(figsize=(12, 6))
    plt.plot(result['Frame'].values, result['left_angle_deg'].values, label="Left Angle (deg)", color="blue", linewidth=1.5)
    plt.plot(result['Frame'].values, result['right_angle_deg'].values, label="Right Angle (deg)", color="red", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Angle (deg)", fontsize=14)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    # 扩大Y轴范围以便观察噪声
    plt.ylim(-10, 10)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_filename)
    print(f"Angle plot saved to {plot_filename}")
    plt.show() # 在 Colab 中显示图片
    
    return [csv_filename, plot_filename]


def process_h5_file(input_file, output_file, window_size=21):
    """
    主函数：读取 H5，执行平滑，并调用绘图函数分别处理平滑前后的数据。
    """
    # 读取 h5 文件
    print(f"Reading data from {input_file}...")
    original_data = pd.read_hdf(input_file)

    if not isinstance(original_data.columns, pd.MultiIndex):
        raise ValueError("Input H5 file does not have a MultiIndex structure!")

    scorer = original_data.columns.levels[0][0]
    
    # --- 关键修改：在这里先处理平滑前的数据 ---
    base_name = os.path.splitext(output_file)[0]
    plot_file_before = f"{base_name}_angles_BEFORE_smoothing.png"
    csv_file_before = f"{base_name}_angles_BEFORE_smoothing.csv"
    files_before = calculate_and_plot_angles(original_data, scorer, "Angles Before Smoothing", plot_file_before, csv_file_before)

    # --- 执行平滑处理 ---
    print("\n--- Applying moving average for smoothing ---")
    keypoints = original_data.columns.levels[1]
    coordinates = original_data.columns.levels[2]
    smoothed_data_dict = {}

    for keypoint in keypoints:
        for coordinate in coordinates:
            try:
                data_col = original_data[(scorer, keypoint, coordinate)].values
                smoothed_values = moving_average(data_col, window_size)
                smoothed_data_dict[(scorer, keypoint, coordinate)] = smoothed_values
            except KeyError:
                print(f"Warning: Data for ({keypoint}, {coordinate}) not found!")
    
    smoothed_data = pd.DataFrame(smoothed_data_dict, index=original_data.index)

    print(f"\nSaving smoothed data to {output_file}...")
    smoothed_data.to_hdf(output_file, key="df", mode="w")
    print("Processing complete!")

    # --- 关键修改：在这里处理平滑后的数据 ---
    plot_file_after = f"{base_name}_angles_AFTER_smoothing.png"
    csv_file_after = f"{base_name}_angles_AFTER_smoothing.csv"
    files_after = calculate_and_plot_angles(smoothed_data, scorer, f"Angles After Smoothing (Window Size = {window_size})", plot_file_after, csv_file_after)
    
    # 返回所有生成的文件名
    return [output_file] + files_before + files_after

# -----------------------------------------------------
# 主执行流程
# -----------------------------------------------------

# 3. 上传文件
print("Please upload your H5 file...")
uploaded = files.upload()

if not uploaded:
    print("No file was uploaded. Please run the cell again.")
else:
    input_h5_file = next(iter(uploaded))
    print(f"\nSuccessfully uploaded: {input_h5_file}")

    # 4. 定义输出文件名
    base_name = os.path.splitext(input_h5_file)[0]
    output_h5_file = f"{base_name}_output.h5"

    # 5. 调用主函数进行处理和绘图
    generated_files = process_h5_file(input_h5_file, output_h5_file)

    # 6. 下载所有生成的结果文件
    print("\nDownloading your result files...")
    for f in generated_files:
        files.download(f)
    print("All files have been downloaded.")