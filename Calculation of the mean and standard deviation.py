import pandas as pd
import numpy as np

def analyze_barbell_angles(input_file, video_height=1342):
    """
    Analyze barbell angle data
    
    Parameters:
    input_file: h5 file path
    video_height: video height, default 1342 but doesn't affect relative angle calculation
    
    Returns:
    Dictionary containing left and right angle statistics
    """
    try:
        # Read h5 file
        print(f"Reading data from {input_file}...")
        data = pd.read_hdf(input_file)

        # Check if it has MultiIndex structure
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Input H5 file does not have a MultiIndex structure!")

        # Extract index levels
        scorer = data.columns.levels[0][0]  # Assume there's only one scorer
        
        # Create a copy of DataFrame to avoid warnings
        df = data.copy()

        # Convert Y coordinates to mathematical coordinate system
        left_y_math = video_height - df[(scorer, 'Left', 'y')]
        middle_y_math = video_height - df[(scorer, 'Middle', 'y')]
        right_y_math = video_height - df[(scorer, 'Right', 'y')]

        # Calculate dx and dy relative to the midpoint
        df['left_dx'] = abs(df[(scorer, 'Left', 'x')] - df[(scorer, 'Middle', 'x')])
        df['right_dx'] = abs(df[(scorer, 'Right', 'x')] - df[(scorer, 'Middle', 'x')])

        df['left_dy'] = left_y_math - middle_y_math
        df['right_dy'] = right_y_math - middle_y_math

        # Calculate angles
        df['left_angle_deg'] = np.degrees(np.arctan(df['left_dy'] / df['left_dx']))
        df['right_angle_deg'] = np.degrees(np.arctan(df['right_dy'] / df['right_dx']))

        # Calculate statistics
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

# Usage example
if __name__ == "__main__":
    input_file = "/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5"  # Replace with your h5 file path
    results = analyze_barbell_angles(input_file)
