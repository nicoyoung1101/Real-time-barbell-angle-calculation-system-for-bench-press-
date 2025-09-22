import cv2
import h5py
import numpy as np

# Configure paths
video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
result_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.h5'
output_video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/overlay_video_two_phases.mp4'

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

# Calculate frame ranges for two phases
phase1_start = int(1 * fps)  # Phase 1 start (1 second)
phase1_end = int(5 * fps)    # Phase 1 end (5 seconds)
phase2_start = int(5 * fps)  # Phase 2 start (6 seconds)
phase2_end = int(8 * fps)    # Phase 2 end (8 seconds)

# Prepare video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define colors for each phase
phase1_color = (255, 0, 255)    # Blue for phase 1
phase2_color = (255, 0, 255)    # Green for phase 2

# Process two phases
for phase in [1, 2]:
    # Create new overlay layer for each phase
    overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Set phase parameters
    if phase == 1:
        start_frame = phase1_start
        end_frame = phase1_end
        color = phase1_color
    else:
        start_frame = phase2_start
        end_frame = phase2_end
        color = phase2_color
    
    # Jump to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    
    # Reset previous frame coordinates
    last_x = None
    last_y = None
    
    # Process frames in current phase
    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(left_x):
            break

        try:
            # Get keypoints for current frame
            x = [left_x[frame_idx], middle_x[frame_idx], right_x[frame_idx]]
            y = [left_y[frame_idx], middle_y[frame_idx], right_y[frame_idx]]
            
            # Draw skeleton lines
            cv2.line(overlay, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, 2)
            cv2.line(overlay, (int(x[1]), int(y[1])), (int(x[2]), int(y[2])), color, 2)
            
            # If previous frame coordinates exist, draw trajectory lines
            if last_x is not None:
                for i in range(3):
                    cv2.line(overlay, 
                            (int(last_x[i]), int(last_y[i])), 
                            (int(x[i]), int(y[i])), 
                            color, 1)
            
            # Update previous frame coordinates
            last_x = x.copy()
            last_y = y.copy()
            
        except IndexError:
            print(f"Phase {phase} frame index out of range, processing completed")
            break

        # Overlay layers
        combined = cv2.addWeighted(frame, 0.6, overlay, 0.9, 0)

        # Write to video
        out.write(combined)
        frame_idx += 1

        # Print progress
        print(f"Processing phase {phase}: frame {frame_idx}/{end_frame}", end='\r')

cap.release()
out.release()

print("\nVideo generation completed, saved to:", output_video_path)
