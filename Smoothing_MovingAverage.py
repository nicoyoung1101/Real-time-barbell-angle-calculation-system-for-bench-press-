# 1. Import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from google.colab import files  # Import Colab file processing library

# 2. Data processing and plotting functions

def moving_average(data, window_size):
    """Apply moving average method to smooth data"""
    if len(data) < window_size:
        raise ValueError("Data length must be greater than or equal to the window size.")
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def calculate_and_plot_angles(df, scorer, title, plot_filename, csv_filename):
    """
    A helper function to calculate angles from given DataFrame, save CSV and create plots.
    :param df: Input DataFrame (can be original or smoothed)
    :param scorer: Name of the DLC scorer
    :param title: Title of the chart
    :param plot_filename: Output image filename
    :param csv_filename: Output CSV filename
    :return: Generated [csv_filename, plot_filename] list
    """
    print(f"\n--- Processing for: {title} ---")
    
    # ---- Angle calculation ----
    left_y_math = 1342 - df[(scorer, 'Left', 'y')]
    middle_y_math = 1342 - df[(scorer, 'Middle', 'y')]
    right_y_math = 1342 - df[(scorer, 'Right', 'y')]
    
    # Use .copy() to avoid SettingWithCopyWarning
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

    # Save angle results as CSV file
    result.to_csv(csv_filename, index=False)
    print(f"Angle results saved to {csv_filename}")

    # --- Plotting section ---
    plt.figure(figsize=(12, 6))
    plt.plot(result['Frame'].values, result['left_angle_deg'].values, label="Left Angle (deg)", color="blue", linewidth=1.5)
    plt.plot(result['Frame'].values, result['right_angle_deg'].values, label="Right Angle (deg)", color="red", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Angle (deg)", fontsize=14)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    # Expand Y-axis range to observe noise
    plt.ylim(-10, 10)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_filename)
    print(f"Angle plot saved to {plot_filename}")
    plt.show() # Display image in Colab
    
    return [csv_filename, plot_filename]


def process_h5_file(input_file, output_file, window_size=21):
    """
    Main function: Read H5, perform smoothing, and call plotting function to process data before and after smoothing.
    """
    # Read h5 file
    print(f"Reading data from {input_file}...")
    original_data = pd.read_hdf(input_file)

    if not isinstance(original_data.columns, pd.MultiIndex):
        raise ValueError("Input H5 file does not have a MultiIndex structure!")

    scorer = original_data.columns.levels[0][0]
    
    # --- Key modification: Process data before smoothing here first ---
    base_name = os.path.splitext(output_file)[0]
    plot_file_before = f"{base_name}_angles_BEFORE_smoothing.png"
    csv_file_before = f"{base_name}_angles_BEFORE_smoothing.csv"
    files_before = calculate_and_plot_angles(original_data, scorer, "Angles Before Smoothing", plot_file_before, csv_file_before)

    # --- Perform smoothing process ---
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

    # --- Key modification: Process data after smoothing here ---
    plot_file_after = f"{base_name}_angles_AFTER_smoothing.png"
    csv_file_after = f"{base_name}_angles_AFTER_smoothing.csv"
    files_after = calculate_and_plot_angles(smoothed_data, scorer, f"Angles After Smoothing (Window Size = {window_size})", plot_file_after, csv_file_after)
    
    # Return all generated filenames
    return [output_file] + files_before + files_after

# -----------------------------------------------------
# Main execution flow
# -----------------------------------------------------

# 3. Upload file
print("Please upload your H5 file...")
uploaded = files.upload()

if not uploaded:
    print("No file was uploaded. Please run the cell again.")
else:
    input_h5_file = next(iter(uploaded))
    print(f"\nSuccessfully uploaded: {input_h5_file}")

    # 4. Define output filename
    base_name = os.path.splitext(input_h5_file)[0]
    output_h5_file = f"{base_name}_output.h5"

    # 5. Call main function for processing and plotting
    generated_files = process_h5_file(input_h5_file, output_h5_file)

    # 6. Download all generated result files
    print("\nDownloading your result files...")
    for f in generated_files:
        files.download(f)
    print("All files have been downloaded.")
