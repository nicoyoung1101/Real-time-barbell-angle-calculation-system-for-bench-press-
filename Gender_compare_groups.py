# -*- coding: utf-8 -*-
"""
Group Comparison Script (by Gender) using Independent T-test

This script compares two gender groups (Male vs. Female) based on data from CSV files.
It performs an independent samples t-test on 'left_angle_deg' and 'right_angle_deg'
columns, visualizes the results with a bar chart including error bars (SEM),
and annotates the chart with the resulting p-values.

The script saves the plot as a PNG file and the statistical results as a TXT file.

Expected CSV Format:
    The input CSV files must contain at least two columns named:
    - left_angle_deg
    - right_angle_deg

Usage:
    python 04_compare_groups_by_gender.py --male_data path/to/males.csv --female_data path/to/females.csv --output_dir path/to/results/

Example:
    python 04_compare_groups_by_gender.py --male_data data/male_lifters.csv --female_data data/female_lifters.csv --output_dir results/
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
        male_df = pd.read_csv(args.male_data)
        female_df = pd.read_csv(args.female_data)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # Extract data columns, dropping any potential missing values
    male_left = male_df['left_angle_deg'].dropna()
    male_right = male_df['right_angle_deg'].dropna()
    female_left = female_df['left_angle_deg'].dropna()
    female_right = female_df['right_angle_deg'].dropna()

    print(f"Loaded {len(male_left)} samples for Male group.")
    print(f"Loaded {len(female_left)} samples for Female group.")

    # --- 2. Perform T-tests ---
    t_stat_left, p_value_left = stats.ttest_ind(male_left, female_left)
    t_stat_right, p_value_right = stats.ttest_ind(male_right, female_right)

    # --- 3. Save Statistical Results to a Text File ---
    results_path = os.path.join(args.output_dir, 'gender_t-test_results.txt')
    with open(results_path, 'w') as f:
        f.write("T-test Statistical Results: Gender Comparison\n")
        f.write("=============================================\n\n")
        f.write(f"Male Group Data: {os.path.basename(args.male_data)}\n")
        f.write(f"Female Group Data: {os.path.basename(args.female_data)}\n\n")
        f.write("--- Left Angle Comparison ---\n")
        f.write(f"Mean Male: {np.mean(male_left):.2f} ± {stats.sem(male_left):.2f}\n")
        f.write(f"Mean Female: {np.mean(female_left):.2f} ± {stats.sem(female_left):.2f}\n")
        f.write(f"T-statistic: {t_stat_left:.3f}\n")
        f.write(f"P-value: {p_value_left:.5f}\n\n")
        f.write("--- Right Angle Comparison ---\n")
        f.write(f"Mean Male: {np.mean(male_right):.2f} ± {stats.sem(male_right):.2f}\n")
        f.write(f"Mean Female: {np.mean(female_right):.2f} ± {stats.sem(female_right):.2f}\n")
        f.write(f"T-statistic: {t_stat_right:.3f}\n")
        f.write(f"P-value: {p_value_right:.5f}\n")
    print(f"Statistical results saved to {results_path}")

    # --- 4. Create Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    groups = ['Left Angle', 'Right Angle']
    male_means = [np.mean(male_left), np.mean(male_right)]
    female_means = [np.mean(female_left), np.mean(female_right)]
    male_sems = [stats.sem(male_left), stats.sem(male_right)]
    female_sems = [stats.sem(female_left), stats.sem(female_right)]

    x = np.arange(len(groups))
    width = 0.35

    plt.bar(x - width/2, male_means, width, label='Male', yerr=male_sems, capsize=7, color='skyblue')
    plt.bar(x + width/2, female_means, width, label='Female', yerr=female_sems, capsize=7, color='salmon')
    
    plt.xlabel('Angle Type', fontsize=16)
    plt.ylabel('Mean Angle (degrees)', fontsize=16)
    plt.title('Gender Comparison of Barbell Tilt Angles', fontsize=20, pad=20)
    plt.xticks(x, groups, fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=14)
    
    # Determine the top of the plot for placing annotations
    y_max = max(max(np.add(male_means, male_sems)), max(np.add(female_means, female_sems)))
    y_min = min(min(np.subtract(male_means, male_sems)), min(np.subtract(female_means, female_sems)))

    annotation_y = y_max + abs(y_max - y_min) * 0.1
    annotation_h = abs(y_max - y_min) * 0.05
    
    # Add significance annotations
    add_significance_annotation(p_value_left, x[0] - width/2, x[0] + width/2, annotation_y, annotation_h, args.alpha)
    add_significance_annotation(p_value_right, x[1] - width/2, x[1] + width/2, annotation_y, annotation_h, args.alpha)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = os.path.join(args.output_dir, 'gender_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two gender groups using an independent t-test and generate a plot.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--male_data', 
        type=str, 
        required=True, 
        help="Path to the CSV file for the male group."
    )
    parser.add_argument(
        '--female_data', 
        type=str, 
        required=True, 
        help="Path to the CSV file for the female group."
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