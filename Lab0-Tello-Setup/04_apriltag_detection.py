"""
Tello AprilTag Detection using djitellopy
"""
import cv2
import numpy as np
from time import sleep
from djitellopy import Tello
from pupil_apriltags import Detector
import math

def euler_from_matrix(R):
    """
    Extract Euler angles (roll, pitch, yaw) from rotation matrix
    Returns angles in degrees
    """
    # Extract Euler angles from rotation matrix
    # This follows the ZYX convention (yaw-pitch-roll)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])  # roll
        y = math.atan2(-R[2,0], sy)     # pitch
        z = math.atan2(R[1,0], R[0,0])  # yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])  # roll
        y = math.atan2(-R[2,0], sy)      # pitch
        z = 0                            # yaw
    
    # Convert to degrees
    return np.degrees([x, y, z])

def main():
    # Initialize Tello drone
    print("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    
    # Check battery level
    battery = tello.get_battery()
    print(f"Battery level: {battery}%")
    
    if battery < 10:
        print("Warning: Low battery level!")
    
    # Start video stream
    print("Starting video stream...")
    tello.streamon()
    
    # Wait a moment for video stream to initialize
    sleep(2)
    
    # Get frame reader
    frame_read = tello.get_frame_read()
    
    # Initialize AprilTag detector
    print("Initializing AprilTag detector...")
    detector = Detector(
        families='tag36h11',  # Common AprilTag family
        nthreads=1,
        quad_decimate=2.0,    # Increase speed, decrease accuracy
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    # Camera parameters for pose estimation (approximate values for Tello)
    # These should be calibrated for your specific drone
    camera_params = [
        921.170702,   # fx
        919.018377,   # fy
        459.904354,   # cx
        351.238301    # cy
    ]
    tag_size = 0.08  # Tag size in meters (adjust based on your tag size)
    
    print("Starting detection loop... Press 'q' to quit")
    framerate = 10.0  # Hz
    
    try:
        while True:
            # Get current frame
            frame = frame_read.frame
            
            if frame is None:
                print("No frame received, skipping...")
                sleep(1 / framerate)
                continue
            
            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags
            tags = detector.detect(gray, 
                                 estimate_tag_pose=True, 
                                 camera_params=camera_params, 
                                 tag_size=tag_size)
            
            # Process detected tags
            if tags:
                for tag in tags:
                    print(f"\n--- Tag ID: {tag.tag_id} ---")
                    
                    # Get tag center
                    center_x, center_y = tag.center
                    print(f"Tag center: ({center_x:.2f}, {center_y:.2f})")
                    
                    # Get pose information
                    if hasattr(tag, 'pose_t') and hasattr(tag, 'pose_R'):
                        # Translation (position)
                        position = tag.pose_t.flatten()
                        print(f"Position (x, y, z): ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) meters")
                        
                        # Rotation matrix to Euler angles
                        euler_angles = euler_from_matrix(tag.pose_R)
                        print(f"Euler angles (roll, pitch, yaw): ({euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°)")
                    
                    # Draw detection on frame for visualization
                    # Draw tag outline
                    corners = tag.corners.astype(int)
                    cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                    
                    # Draw tag ID
                    cv2.putText(frame, f'ID: {tag.tag_id}', 
                              (int(center_x) - 30, int(center_y) - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            # Display frame
            cv2.imshow('Tello AprilTag Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate
            sleep(1 / framerate)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        tello.streamoff()  # Stop video stream
        cv2.destroyAllWindows()
        tello.end()
        print("Disconnected from Tello")

if __name__ == "__main__":
    main()