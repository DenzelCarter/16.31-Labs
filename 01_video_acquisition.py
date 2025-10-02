"""
An example of using djitellopy to get video from the drone.
"""
from time import sleep
import cv2
from djitellopy import Tello

def main():
    """Video streaming example with live display"""
    
    print("Tello Video Stream Example")
    print("=========================")
    
    # Make a Tello object
    tello = Tello()
    
    try:
        # Connect to the drone
        print("Connecting to Tello...")
        tello.connect()
        
        # Check battery
        battery = tello.get_battery()
        print(f"Connected! Battery level: {battery}%")
        
        # Start video stream
        print("Starting video stream...")
        tello.streamon()
        
        # Wait a moment for stream to initialize
        sleep(2)
        
        # Get frame reader object
        frame_read = tello.get_frame_read()
        
        print("Video stream active!")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Video parameters
        framerate = 30.0  # Hz (target framerate)
        frame_count = 0
        
        # Main video loop
        while True:
            # Get current frame
            frame = frame_read.frame
            
            if frame is not None:
                frame_count += 1
                
                # Add overlay information
                height, width = frame.shape[:2]
                
                # Add text overlay with drone info
                overlay_text = [
                    f"Battery: {tello.get_battery()}%",
                    f"Frame: {frame_count}",
                    f"Size: {width}x{height}",
                    "Press 'q' to quit, 's' for screenshot"
                ]
                
                # Draw text overlay
                for i, text in enumerate(overlay_text):
                    y_pos = 30 + (i * 25)
                    cv2.putText(frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Tello Video Stream', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"tello_screenshot_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
            else:
                print("No frame received, retrying...")
                sleep(0.1)
                continue
            
            # Control frame rate
            sleep(1.0 / framerate)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        
        # Common troubleshooting tips
        print("\nTroubleshooting tips:")
        print("- Ensure you're connected to Tello's WiFi network")
        print("- Check if firewall is blocking UDP port 11111")
        print("- Try restarting the Tello drone")
        print("- Update Tello firmware using the official app")
    
    finally:
        # Cleanup
        try:
            print("Cleaning up...")
            tello.streamoff()  # Stop video stream
            cv2.destroyAllWindows()  # Close OpenCV windows
            tello.end()  # Disconnect from drone
            print("Cleanup complete")
        except:
            pass

if __name__ == "__main__":
    main()