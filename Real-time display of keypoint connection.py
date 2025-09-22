import cv2
import numpy as np
import os

class BarbellTracker:
    def __init__(self):
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Store trajectory
        self.trajectory = []
        self.max_trajectory_points = 50
        
        # Ensure markers are generated
        self.generate_markers_if_needed()
        
    def generate_markers_if_needed(self):
        """Generate ArUco markers if they don't exist"""
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
        """Detect ArUco markers in the frame"""
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
        """Draw marker points, connecting lines, and trajectory on the frame"""
        if len(points) == 3:
            # Draw marker points
            colors = [(255,0,0), (0,255,0), (0,0,255)]  # BGR colors
            labels = ["Left", "Middle", "Right"]
            
            for i, point in enumerate(points):
                cv2.circle(frame, point, 5, colors[i], -1)
                cv2.putText(frame, labels[i], 
                          (point[0] + 10, point[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # Draw connecting lines
            cv2.line(frame, points[0], points[1], (255,0 , 255), 2)
            cv2.line(frame, points[1], points[2], (255, 0, 255), 2)
            
            # Update and draw trajectory
            self.trajectory.append(points[1])
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
            
            for i in range(1, len(self.trajectory)):
                cv2.line(frame, 
                        self.trajectory[i-1], 
                        self.trajectory[i], 
                        (255, 0, 0), 1)

    def process_video(self, video_path=0, output_path=None):
        """Process video or camera input"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set up video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
        
        # For calculating actual frame rate
        prev_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate actual frame rate
            current_time = cv2.getTickCount()
            actual_fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time
            
            # Detect markers
            points = self.detect_markers(frame)
            
            # Draw markers and lines
            self.draw_markers_and_lines(frame, points)
            
            # Display information
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            status_color = (0, 255, 0) if len(points) == 3 else (0, 0, 255)
            cv2.putText(frame, f"Detected: {len(points)}/3", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Display results
            cv2.imshow('Barbell Tracking', frame)
            
            # Save video
            if out:
                out.write(frame)
            
            # Press 'q' to quit, press 's' to save current frame
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
    """Print usage instructions"""
    print("\n=== Barbell Tracking System ===")
    print("1. Markers have been generated in the 'markers' folder")
    print("2. Please install markers in the following order:")
    print("   - marker_0.png -> Left end of barbell")
    print("   - marker_1.png -> Middle of barbell")
    print("   - marker_2.png -> Right end of barbell")
    print("\nControl keys:")
    print("- Press 'q' to quit the program")
    print("- Press 's' to save current frame")
    print("========================\n")

if __name__ == "__main__":
    # Set fixed video path
    video_path = '/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/TestObject_2.mp4'
    
    # Create tracker instance
    tracker = BarbellTracker()
    
    # Print usage instructions
    print_instructions()
    
    # Set output path
    output_path = "/Users/a/Desktop/ゼミ用Folder/BarTracking_Videos/TestVideo/Outputvideo.mp4"
    
    # Start processing
    try:
        tracker.process_video(video_path, output_path)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nProgram has ended")
