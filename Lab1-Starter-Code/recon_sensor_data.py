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

    Returns a dict (all SI units unless noted):
        ax, ay, az         : linear acceleration [m/s^2]
        vx, vy, vz         : linear velocity [m/s]
        roll, pitch, yaw   : attitude [deg]
        height_sonar       : ToF range [m]
        height_baro        : barometric altitude [m]
        battery            : battery percent [%]
    """
        # Helper: safely pull a numeric field from the state dict;
    # return np.nan if missing or not parseable.
    def _get_numeric(d, key):
        try:
            val = d.get(key, np.nan)
            # Some fields may arrive as strings; coerce to float when possible.
            return float(val) if val is not None else np.nan
        except Exception:
            return np.nan

    # Helper: convert centimeters → meters, handling NaN gracefully.
    def _cm_to_m(x_cm):
        return x_cm / 100.0 if np.isfinite(x_cm) else np.nan

    # 1) Snapshot the full state dict (includes 'agx', 'vgx', 'roll', etc.)
    # djitellopy keeps a cached state; this is non-blocking.
    state = tello.get_current_state() or {}

    # 2) Acceleration (SDK gives cm/s^2) → [m/s^2]
    ax = _cm_to_m(_get_numeric(state, "agx"))
    ay = _cm_to_m(_get_numeric(state, "agy"))
    az = _cm_to_m(_get_numeric(state, "agz"))

    # 3) Velocity (SDK gives cm/s) → [m/s]
    vx = _cm_to_m(_get_numeric(state, "vgx"))
    vy = _cm_to_m(_get_numeric(state, "vgy"))
    vz = _cm_to_m(_get_numeric(state, "vgz"))

    # 4) Attitude (SDK gives degrees already)
    roll  = _get_numeric(state, "roll")
    pitch = _get_numeric(state, "pitch")
    yaw   = _get_numeric(state, "yaw")

    # 5) Sonar/ToF height: get_distance_tof() returns centimeters (int) → meters
    try:
        tof_cm = tello.get_distance_tof()
        height_sonar = float(tof_cm) / 100.0 if tof_cm is not None else np.nan
    except Exception:
        height_sonar = np.nan

    # 6) Barometer: djitellopy.get_barometer() is usually meters as float.
    # Add a defensive check: if value is implausibly large for meters,
    # assume it was centimeters and convert.
    try:
        baro_val = tello.get_barometer()  # expected meters (float)
        if baro_val is None:
            height_baro = np.nan
        else:
            height_baro = float(baro_val)
            if height_baro > 25.0:  # larger than any indoor flight in meters
                height_baro = height_baro / 100.0  # treat as centimeters
    except Exception:
        height_baro = np.nan

    # 7) Battery percent
    try:
        battery = float(tello.get_battery())
    except Exception:
        battery = np.nan

    return {
        "ax": ax, "ay": ay, "az": az,
        "vx": vx, "vy": vy, "vz": vz,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "height_sonar": height_sonar,
        "height_baro": height_baro,
        "battery": battery,
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
        
        # ---------------- MAIN DATA COLLECTION LOOP (10 Hz) ----------------
        # Open the CSV and stream samples for the entire mission timeline.
        csv_filename = os.path.join(OUTPUT_DIR, "sensor_data.csv")
        fieldnames = [
            # Global timeline
            "mission_time",          # seconds since mission_start_time
            "target_altitude",       # setpoint [m]
            "current_altitude_est",  # fused estimate (sonar preferred) [m]
            # IMU/kinematics
            "ax", "ay", "az",        # [m/s^2]
            "vx", "vy", "vz",        # [m/s]
            "roll", "pitch", "yaw",  # [deg]
            # Heights
            "height_sonar",          # [m]
            "height_baro",           # [m]
            # Utilities
            "battery",               # [%]
        ]

        # We'll enforce 10 Hz using a "next_sample_time" scheduler so timing stays stable
        # regardless of loop body duration (sleep the remainder of each 0.1 s frame).
        with open(csv_filename, "w", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Helper to sample once (used during holds and at the end)
            def _sample_and_write(current_target):
                now = time.time()
                mission_time = now - mission_start_time

                # Read all raw channels
                readings = get_sensor_readings(tello)

                # Fused altitude estimate (prefer sonar if reasonable, else baro)
                current_altitude_est = get_current_altitude(tello)

                row = {
                    "mission_time": float(mission_time),
                    "target_altitude": float(current_target),
                    "current_altitude_est": float(current_altitude_est),
                    "ax": readings["ax"], "ay": readings["ay"], "az": readings["az"],
                    "vx": readings["vx"], "vy": readings["vy"], "vz": readings["vz"],
                    "roll": readings["roll"], "pitch": readings["pitch"], "yaw": readings["yaw"],
                    "height_sonar": readings["height_sonar"],
                    "height_baro": readings["height_baro"],
                    "battery": readings["battery"],
                }
                writer.writerow(row)
                return now  # return the wall-clock time we sampled at

            # Iterate through each altitude setpoint
            for target_altitude in ALTITUDE_SETPOINTS:
                print(f"\n→ Target altitude: {target_altitude:.2f} m")

                # Move to the setpoint (discrete move_up/down in centimeters)
                current_altitude = get_current_altitude(tello)
                move_to_altitude(tello, current_altitude, target_altitude)

                # Hold at the setpoint and record for TIME_PER_ALTITUDE seconds at 10 Hz
                hold_start = time.time()
                next_sample_time = hold_start  # start sampling immediately
                while (time.time() - hold_start) < TIME_PER_ALTITUDE:
                    # Take one sample and write row
                    t_sample = _sample_and_write(target_altitude)

                    # Schedule next 10 Hz tick; sleep the remainder if any
                    next_sample_time += SAMPLE_DT
                    sleep_time = max(0.0, next_sample_time - time.time())
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # After each setpoint hold, let the drone settle briefly (already done in move)
                # but take one more sample to capture the settled state
                _sample_and_write(target_altitude)

            # After the final setpoint, linger a couple seconds to round out the mission log
            final_target = ALTITUDE_SETPOINTS[-1]
            end_linger_start = time.time()
            next_sample_time = end_linger_start
            while (time.time() - end_linger_start) < max(SETTLE_TIME, 2.0):
                _sample_and_write(final_target)
                next_sample_time += SAMPLE_DT
                sleep_time = max(0.0, next_sample_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"\nSaved Phase 1 data → {csv_filename}")

        
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