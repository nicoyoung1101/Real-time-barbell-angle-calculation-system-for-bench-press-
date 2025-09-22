import cv2
import h5py
import numpy as np
import math

# Configure paths
video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
result_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5'
output_video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/Degree_display.mp4'

# Read HDF5 data
with h5py.File(result_path, 'r') as h5file:
    table = h5file['df_with_missing/table'][:]
    keypoints = table['values_block_0']

# Extract keypoint data
left_x, left_y = keypoints[:, 0], keypoints[:, 1]
middle_x, middle_y = keypoints[:, 3], keypoints[:, 4]
right_x, right_y = keypoints[:, 6], keypoints[:, 7]

# Open video
cap = cv2.VideoCapture(video_path)

# Get video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def calculate_angle(p1, p2, is_left=True):
    """
    Calculate the angle between a line segment and horizontal line
    p1, p2: coordinates of the two endpoints of the line segment (x, y)
    is_left: whether it's the left side angle
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(-dy, dx))
    
    # Adjust for left side angle: simply negate
    if is_left:
        angle = -angle
            
    return angle

# Length of horizontal line (pixels)
horizontal_line_length = 100
# Line colors
line_color = (0, 255, 0)  # Green
text_color = (0, 0, 0)    # Black

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(left_x):
        break

    try:
        x = [left_x[frame_idx], middle_x[frame_idx], right_x[frame_idx]]
        y = [left_y[frame_idx], middle_y[frame_idx], right_y[frame_idx]]
        
        # Draw skeleton lines
        cv2.line(frame, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), line_color, 2)
        cv2.line(frame, (int(x[1]), int(y[1])), (int(x[2]), int(y[2])), line_color, 2)
        
        # Draw horizontal line at the midpoint
        middle_x_int = int(x[1])
        middle_y_int = int(y[1])
        horizontal_start = (middle_x_int - horizontal_line_length//2, middle_y_int)
        horizontal_end = (middle_x_int + horizontal_line_length//2, middle_y_int)
        cv2.line(frame, horizontal_start, horizontal_end, (0, 255, 255), 2)  # Yellow horizontal line

        # Calculate angles between left and right line segments and horizontal line
        left_angle = calculate_angle((x[0], y[0]), (x[1], y[1]), is_left=True)
        right_angle = calculate_angle((x[1], y[1]), (x[2], y[2]), is_left=False)

        # Display angles on the screen (pure black text)
        cv2.putText(frame, f"L: {left_angle:.1f}deg", 
                   (middle_x_int - 120, middle_y_int - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.putText(frame, f"R: {right_angle:.1f}deg", 
                   (middle_x_int + 20, middle_y_int - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display frame information
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
    except IndexError:
        print("Frame index out of range, processing completed")
        break

    # Write to video
    out.write(frame)
    frame_idx += 1

    print(f"Processing frame {frame_idx}/{total_frames}", end='\r')

cap.release()
out.release()

print("\nVideo generation completed, saved to:", output_video_path)
