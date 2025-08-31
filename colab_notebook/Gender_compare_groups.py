"""
Group Comparison Script using Independent T-test

This script compares two independent groups based on provided data in CSV files.
It performs an independent samples t-test on 'left_angle_deg' and 'right_angle_deg'
columns, visualizes the results with a bar chart including error bars (SEM),
and annotates the chart with the resulting p-values.

The script saves the plot as a PNG file and the statistical results as a TXT file.

Expected CSV Format:
    The input CSV files must contain at least two columns named:
    - left_angle_deg
    - right_angle_deg

Usage:
    python 03_compare_groups.py --group1_data path/to/group1.csv --group2_data path/to/group2.csv --output_dir path/to/results/

Example:
    python 03_compare_groups.py --group1_data data/heavy_lifters.csv --group2_data data/light_lifters.csv --output_dir results/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def add_significance_annotation(p_value, x1, x2, y, h, alpha=0.05):
    """
    Adds a significance annotation line and p-value text to the plot.
    Also adds stars for common significance levels.
    """
    # Determine significance level stars
    if p_value < 0.001:
        sig_text = f"p < 0.001 (***)"
    elif p_value < 0.01:
        sig_text = f"p = {p_value:.3f} (**)"
    elif p_value < alpha:
        sig_text = f"p = {p_value:.3f} (*)"
    else:
        sig_text = f"p = {p_value:.3f} (ns)" # ns = not significant

    # Draw the annotation line
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], 'k-', linewidth=1.5)
    # Add the text
    plt.text((x1 + x2) * .5, y + h, sig_text, ha='center', va='bottom', fontsize=14)

def main(args):
    """
    Main function to load data, run t-test, and generate outputs.
    """
    # --- 1. Load Data ---
    try:
        group1_df = pd.read_csv(args.group1_data)
        group2_df = pd.read_csv(args.group2_data)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # Extract data columns, dropping any potential missing values
    group1_left = group1_df['left_angle_deg'].dropna()
    group1_right = group1_df['right_angle_deg'].dropna()
    group2_left = group2_df['left_angle_deg'].dropna()
    group2_right = group2_df['right_angle_deg'].dropna()

    print(f"Loaded {len(group1_left)} samples for Group 1.")
    print(f"Loaded {len(group2_left)} samples for Group 2.")

    # --- 2. Perform T-tests ---
    t_stat_left, p_value_left = stats.ttest_ind(group1_left, group2_left)
    t_stat_right, p_value_right = stats.ttest_ind(group1_right, group2_right)

    # --- 3. Save Statistical Results to a Text File ---
    results_path = os.path.join(args.output_dir, 't-test_results.txt')
    with open(results_path, 'w') as f:
        f.write("T-test Statistical Results\n")
        f.write("============================\n\n")
        f.write(f"Group 1 (Heavy): {os.path.basename(args.group1_data)}\n")
        f.write(f"Group 2 (Light): {os.path.basename(args.group2_data)}\n\n")
        f.write("--- Left Angle Comparison ---\n")
        f.write(f"Mean Group 1: {np.mean(group1_left):.2f} ± {stats.sem(group1_left):.2f}\n")
        f.write(f"Mean Group 2: {np.mean(group2_left):.2f} ± {stats.sem(group2_left):.2f}\n")
        f.write(f"T-statistic: {t_stat_left:.3f}\n")
        f.write(f"P-value: {p_value_left:.5f}\n\n")
        f.write("--- Right Angle Comparison ---\n")
        f.write(f"Mean Group 1: {np.mean(group1_right):.2f} ± {stats.sem(group1_right):.2f}\n")
        f.write(f"Mean Group 2: {np.mean(group2_right):.2f} ± {stats.sem(group2_right):.2f}\n")
        f.write(f"T-statistic: {t_stat_right:.3f}\n")
        f.write(f"P-value: {p_value_right:.5f}\n")
    print(f"Statistical results saved to {results_path}")

    # --- 4. Create Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    groups = ['Left Angle', 'Right Angle']
    group1_means = [np.mean(group1_left), np.mean(group1_right)]
    group2_means = [np.mean(group2_left), np.mean(group2_right)]
    group1_sems = [stats.sem(group1_left), stats.sem(group1_right)]
    group2_sems = [stats.sem(group2_left), stats.sem(group2_right)]

    x = np.arange(len(groups))
    width = 0.35

    plt.bar(x - width/2, group1_means, width, label='Group 1 (Heavy)', yerr=group1_sems, capsize=7, color='royalblue')
    plt.bar(x + width/2, group2_means, width, label='Group 2 (Light)', yerr=group2_sems, capsize=7, color='lightcoral')
    
    plt.xlabel('Angle Type', fontsize=16)
    plt.ylabel('Mean Angle (degrees)', fontsize=16)
    plt.title('Group Comparison of Barbell Tilt Angles', fontsize=20, pad=20)
    plt.xticks(x, groups, fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=14)
    
    # Determine the top of the plot for placing annotations
    y_max = max(
        max(np.add(group1_means, group1_sems)), 
        max(np.add(group2_means, group2_sems))
    )
    y_min = min(
        min(np.subtract(group1_means, group1_sems)), 
        min(np.subtract(group2_means, group2_sems))
    )

    annotation_y = y_max + abs(y_max - y_min) * 0.1 # Position 10% above the highest bar/error
    annotation_h = abs(y_max - y_min) * 0.05       # Height of the annotation line
    
    # Add significance annotations
    add_significance_annotation(p_value_left, x[0] - width/2, x[0] + width/2, annotation_y, annotation_h, args.alpha)
    add_significance_annotation(p_value_right, x[1] - width/2, x[1] + width/2, annotation_y, annotation_h, args.alpha)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for title
    
    plot_path = os.path.join(args.output_dir, 'group_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two groups using an independent t-test and generate a plot.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--group1_data', 
        type=str, 
        required=True, 
        help="Path to the CSV file for the first group (e.g., heavy lifters)."
    )
    parser.add_argument(
        '--group2_data', 
        type=str, 
        required=True, 
        help="Path to the CSV file for the second group (e.g., light lifters)."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Directory to save the output plot and results text file."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help="Significance level (alpha) for statistical tests. Default is 0.05."
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
    print("\nAnalysis finished successfully.")