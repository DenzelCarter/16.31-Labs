#!/usr/bin/env python3
"""
Lab 1 - Phase 1: Reconnaissance Sensor Data Collection

This script programmatically flies the drone to specific altitude setpoints
while continuously collecting sensor data to understand sensor behavior across
different flight conditions.

Altitude setpoints: [0.5, 1.0, 1.2, 1.7, 2.0] meters
Data collection: Continuous sampling at 10 Hz throughout entire mission
Output: CSV file with global timestamps for easy analysis

Tasks:
- Automatically visit each altitude setpoint
- Continuously collect sensor data with global mission timeline
- Sample at 10 Hz throughout entire mission
- Save all sensor readings to CSV for analysis
"""

import os
import time
import csv
import numpy as np
from djitellopy import Tello

# Configuration
OUTPUT_DIR = "Lab1-Phase1"
SAMPLE_DT = 0.1  # 10 Hz sampling rate

# Mission parameters
ALTITUDE_SETPOINTS = [0.5, 1.0, 1.2, 1.7, 2.0]  # meters
# ALTITUDE_SETPOINTS = [0.1]  # meters
TIME_PER_ALTITUDE = 6.0    # seconds at each altitude
MOVEMENT_TIMEOUT = 15.0    # max seconds to reach each altitude
SETTLE_TIME = 2.0          # seconds to wait after movement

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_sensor_readings(tello):
    """
    Extract sensor readings from Tello state with proper unit conversions.
    
    TODO: Implement sensor reading logic
    1. Get the current state dictionary: tello.get_current_state()
    2. Extract acceleration values (agx, agy, agz) and convert from cm/s² to m/s²
    3. Extract velocity values (vgx, vgy, vgz) and convert from cm/s to m/s
    4. Extract attitude values (roll, pitch, yaw) - already in degrees
    5. Get sonar height: tello.get_distance_tof() and convert from cm to m
    6. Get barometer height: tello.get_barometer() - already in m (or check units!)
    7. Get battery level: tello.get_battery()
    
    Hint: Use a helper function to safely extract values from the state dictionary
    and handle None/missing values gracefully (return 0.0 as default).
    
    Returns dict with all values in SI units (meters, m/s, m/s², degrees)
    """
    # TODO: Implement sensor reading
    # Hint: The state dictionary contains 'agx', 'agy', 'agz', 'vgx', 'vgy', 'vgz', 
    # 'roll', 'pitch', 'yaw', etc.
    # Remember: djitellopy uses centimeters for most measurements!
    
    return {
        'ax': 0.0, 'ay': 0.0, 'az': 0.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'height_sonar': 0.0,
        'height_baro': 0.0,
        'battery': 0.0
    }

def move_to_altitude(tello, current_altitude, target_altitude):
    """
    Move drone from current altitude to target altitude using move_up/move_down commands.
    
    Args:
        tello: Tello drone object
        current_altitude: Current altitude in meters
        target_altitude: Target altitude in meters
    """
    altitude_diff = target_altitude - current_altitude
    
    if abs(altitude_diff) < 0.05:  # Already close enough
        print(f"  Already at target altitude ({current_altitude:.2f}m)")
        return
    
    # Convert to centimeters for Tello commands
    move_distance_cm = int(abs(altitude_diff) * 100)
    
    # Clamp to Tello's movement limits (20-500 cm)
    move_distance_cm = max(20, min(500, move_distance_cm))
    
    if altitude_diff > 0:
        print(f"  Moving up {move_distance_cm}cm to reach {target_altitude}m")
        tello.move_up(move_distance_cm)
    else:
        print(f"  Moving down {move_distance_cm}cm to reach {target_altitude}m")
        tello.move_down(move_distance_cm)
    
    # Wait for movement to complete and drone to settle
    print(f"  Settling for {SETTLE_TIME}s...")
    time.sleep(SETTLE_TIME)

def get_current_altitude(tello):
    """Get current altitude estimate, preferring sonar"""
    readings = get_sensor_readings(tello)
    
    # Use sonar if available and reasonable, otherwise barometer
    if not np.isnan(readings['height_sonar']) and readings['height_sonar'] > 0.1:
        return readings['height_sonar']
    elif not np.isnan(readings['height_baro']):
        return readings['height_baro']
    else:
        return 1.0  # Fallback estimate

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Initialize drone
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    
    try:
        # Check battery
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 25:
            print("ERROR: Battery too low. Please charge to at least 25%")
            return
        
        print("\nTaking off...")
        tello.takeoff()
        time.sleep(3)  # Allow stabilization after takeoff
        
        # Get initial altitude
        initial_altitude = get_current_altitude(tello)
        print(f"Initial altitude after takeoff: {initial_altitude:.2f}m")
        
        print(f"\n{'='*60}")
        print("PHASE 1: SENSOR RECONNAISSANCE DATA COLLECTION")
        print(f"{'='*60}")
        print(f"Altitude setpoints: {ALTITUDE_SETPOINTS}")
        print(f"Data collection: Continuous at 10 Hz throughout mission")
        print(f"Output: CSV with global timestamps")
        print()
        
        # Start mission timer
        mission_start_time = time.time()
        print(f"Mission start time: {mission_start_time}")
        
        # TODO: Implement the main data collection loop
        # 1. Open CSV file for writing with appropriate fieldnames
        # 2. Write CSV header
        # 3. Loop through each altitude setpoint:
        #    a. Move to the altitude
        #    b. Collect data at 10 Hz for TIME_PER_ALTITUDE seconds
        #    c. Write each sample to CSV with mission_time, target_altitude, sensor readings
        # 4. Continue collecting for a few more seconds at final altitude
        
        # Hint: Use mission_start_time to calculate mission_time for each sample
        # Hint: Use next_sample_time to maintain consistent 10 Hz sampling
        # Hint: CSV fieldnames should include: mission_time, target_altitude, 
        #       current_altitude_est, ax, ay, az, vx, vy, vz, roll, pitch, yaw,
        #       height_sonar, height_baro, battery
        
        csv_filename = os.path.join(OUTPUT_DIR, "sensor_data.csv")
        
        # TODO: Implement main collection loop here
        print("TODO: Implement data collection loop")
        
        # Display summary
        print(f"\n{'='*50}")
        print("DATA COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Data saved to: {csv_filename}")
        
        # Landing
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        print("Phase 1 reconnaissance complete!")
        print("Next: Run analyze_sensor_data.py to analyze the collected data")
        
    except KeyboardInterrupt:
        print("\nMission interrupted by user")
        try:
            tello.land()
        except:
            pass
    except Exception as e:
        print(f"Error during mission: {e}")
        try:
            tello.land()
        except:
            pass
    finally:
        try:
            tello.end()
        except:
            pass
        print("Tello connection closed")

if __name__ == "__main__":
    main()