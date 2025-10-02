"""
A minimal example of using the djitellopy library.
"""
import time
from djitellopy import Tello

def main():
    """Minimal example: connect, takeoff, hover, and land"""
    
    print("Minimal Tello Example")
    print("===================")
    
    # Make a Tello object
    tello = Tello()
    
    try:
        # Connect to the drone
        print("Connecting to Tello...")
        tello.connect()
        
        # Check connection and battery
        battery = tello.get_battery()
        print(f"Connected! Battery level: {battery}%")
        
        # Safety check
        if battery < 20:
            print("Warning: Low battery! Consider charging before flight.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Flight cancelled.")
                return
        
        # Take off
        print("Taking off...")
        tello.takeoff()
        print("Takeoff complete!")
        
        # Hover for 2 seconds
        print("Hovering for 2 seconds...")
        time.sleep(2)
        
        # Land
        print("Landing...")
        tello.land()
        print("Landing complete!")
        
        print("Flight successful! ✈️")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            # Emergency landing if something goes wrong
            print("Attempting emergency landing...")
            tello.land()
        except:
            print("Emergency landing failed - manual intervention required!")
    
    finally:
        # Clean up connection
        try:
            tello.end()
            print("Disconnected from Tello")
        except:
            pass

if __name__ == "__main__":
    main()