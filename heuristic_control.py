#!/usr/bin/env python3
"""
Lab 1 - Phase 3: Heuristic Control Approach

This script implements a simple if-else controller for altitude control.
This serves as the baseline performance that the PID controller needs to beat.

Target: 1.5 meters altitude (from 0.5m starting position)
Tolerance: ±0.1 meters
Duration: 30 seconds
"""

import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello

# Configuration
OUTPUT_DIR = "Lab1-Phase3"
SAMPLE_DT = 0.05  # 20 Hz sampling rate

# Control parameters - Use these values, do not modify
INITIAL_ALTITUDE = 0.5  # meters - starting position (matching Phase 4)
TARGET_ALTITUDE = 1.5   # meters
TOLERANCE = 0.1         # ±0.1 meters
TEST_DURATION = 30.0    # seconds

# Heuristic control velocities - Use these values, do not modify
VELOCITY_UP = 1.0      # m/s when climbing
VELOCITY_DOWN = -0.8   # m/s when descending
VELOCITY_HOLD = 0.0    # m/s when within tolerance

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def velocity_to_rc(velocity_mps):
    """Convert velocity in m/s to RC throttle command (-100 to +100)"""
    rc_cmd = int(velocity_mps * 100)
    return max(-100, min(100, rc_cmd))

# ============================================================================
# PROVIDED: Sensor reading function (reuse from Phase 1/2)
# ============================================================================
def get_sensor_data(tello):
    """
    Get sensor data from Tello using direct API methods.
    Returns values in SI units with proper error handling.
    """
    try:
        # Use direct API methods for reliable measurements
        height_sonar = tello.get_distance_tof() / 100.0  # cm -> m
        height_baro = tello.get_barometer() / 100.0       # cm -> m  
        velocity_z = tello.get_speed_z() / 100.0          # cm/s -> m/s
        battery = tello.get_battery()                     # already in %
        
        # For acceleration, still need to use state dict as no direct methods
        state = tello.get_current_state()
        def safe_get(key, default=0.0):
            try:
                val = state.get(key, default)
                if val is None:
                    return default
                return float(val) / 100.0  # cm/s^2 -> m/s^2
            except (ValueError, TypeError):
                return default
        
        ax = safe_get('agx', 0.0)
        ay = safe_get('agy', 0.0)
        az = safe_get('agz', 0.0)
        
        return {
            'height_sonar': height_sonar,
            'height_baro': height_baro,
            'velocity_z': velocity_z,
            'ax': ax,
            'ay': ay, 
            'az': az,
            'battery': battery
        }
        
    except Exception as e:
        print(f"Error reading sensors: {e}")
        return {
            'height_sonar': 0.0,
            'height_baro': 0.0,
            'velocity_z': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'az': 0.0,
            'battery': 0.0
        }

def get_current_altitude(sensor_data):
    """
    Get current altitude estimate, preferring sonar if valid.
    Based on Phase 2 calibration results, choose the better sensor.
    """
    # Use sonar if available and reasonable, otherwise barometer
    if not np.isnan(sensor_data['height_sonar']) and sensor_data['height_sonar'] > 0.1:
        return sensor_data['height_sonar']
    elif not np.isnan(sensor_data['height_baro']):
        return sensor_data['height_baro']
    else:
        return TARGET_ALTITUDE  # Fallback estimate

# ============================================================================
# PROVIDED: Positioning function
# ============================================================================
def move_to_initial_position(tello, target_altitude, duration):
    """
    Move drone to initial position (0.5m) using simple velocity commands.
    This positioning phase uses basic control, not heuristic control.
    """
    print(f"Positioning to initial altitude ({target_altitude:.1f}m)...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get current altitude
        sensor_data = get_sensor_data(tello)
        current_altitude = get_current_altitude(sensor_data)
        error = target_altitude - current_altitude
        
        # Simple velocity control for positioning
        if abs(error) < 0.05:  # Close enough - hover
            velocity_cmd = 0.0
        elif error > 0:  # Need to go up
            velocity_cmd = 0.5  # Gentle upward velocity
        else:  # Need to go down
            velocity_cmd = -0.4  # Gentle downward velocity
        
        # Send command
        rc_cmd = velocity_to_rc(velocity_cmd)
        tello.send_rc_control(0, 0, rc_cmd, 0)
        
        time.sleep(0.05)
    
    # Stop motion
    tello.send_rc_control(0, 0, 0, 0)
    final_altitude = get_current_altitude(get_sensor_data(tello))
    print(f"Positioned at {final_altitude:.2f}m")

# ============================================================================
# TODO: Implement the if-else controller
# This is the core learning objective for Phase 3
# ============================================================================

def heuristic_controller(current_altitude, target_altitude, tolerance):
    """
    Simple if-else controller logic.
    
    TODO: Implement the if-else control logic:
    1. Calculate error = target_altitude - current_altitude
    2. If error > tolerance: return VELOCITY_UP and "CLIMB"
    3. If error < -tolerance: return VELOCITY_DOWN and "DESCEND"  
    4. Otherwise: return VELOCITY_HOLD and "HOLD"
    
    Use the provided VELOCITY_UP, VELOCITY_DOWN, and VELOCITY_HOLD constants.
    
    Args:
        current_altitude: Current measured altitude [m]
        target_altitude: Desired altitude [m]
        tolerance: Acceptable error band [m]
    
    Returns:
        velocity_command: Commanded vertical velocity [m/s]
        control_action: String describing the action ("CLIMB", "DESCEND", or "HOLD")
    """
    # TODO: Implement if-else control logic here (should be ~5-10 lines)
    error = 0.0  # TODO: Calculate error
    velocity_command = 0.0  # TODO: Determine velocity command based on error
    control_action = "HOLD"  # TODO: Determine action string
    
    return velocity_command, control_action

# ============================================================================
# PROVIDED: Performance metrics calculation
# This implements the formulas from the lab document
# ============================================================================

def calculate_performance_metrics(data):
    """
    Calculate performance metrics for the heuristic controller.
    
    Args:
        data: Dictionary containing time series data
    
    Returns:
        metrics: Dictionary of calculated performance metrics
    """
    timestamps = np.array(data['mission_time'])
    altitudes = np.array(data['current_altitude'])
    target = TARGET_ALTITUDE
    errors = target - altitudes
    velocities = np.array(data['velocity_cmd'])
    
    # Time parameters
    t0 = timestamps[0]
    T = timestamps[-1]
    dt = np.mean(np.diff(timestamps))
    N = len(timestamps)
    
    # Settling band (±5% of setpoint)
    delta_band = 0.05 * abs(target)
    
    # 1. Settling time
    abs_errors = np.abs(errors)
    settling_time = None
    
    for i in range(len(timestamps)):
        # Check if all subsequent errors are within band
        if all(abs_errors[i:] <= delta_band):
            settling_time = timestamps[i] - t0
            break
    
    if settling_time is None:
        settling_time = float('inf')  # Never settles
    
    # 2. Steady-state analysis
    if settling_time != float('inf'):
        T_ss = T - (t0 + settling_time)
    else:
        T_ss = min(15.0, T - t0)  # Use last 15 seconds if never settles
    
    # Find steady-state window
    ss_start_time = T - T_ss
    ss_mask = timestamps >= ss_start_time
    ss_errors = errors[ss_mask]
    
    # Steady-state error
    steady_state_error = np.mean(np.abs(ss_errors)) if len(ss_errors) > 0 else float('inf')
    
    # 3. Rise time (10% to 90%)
    h0 = altitudes[0]
    A = abs(target - h0)
    
    if A > 0.01:  # Meaningful change required
        h_10 = h0 + 0.1 * (target - h0)
        h_90 = h0 + 0.9 * (target - h0)
        
        # Find crossing times
        t_10 = None
        t_90 = None
        
        if target > h0:  # Moving up
            for i, h in enumerate(altitudes):
                if t_10 is None and h >= h_10:
                    t_10 = timestamps[i]
                if t_90 is None and h >= h_90:
                    t_90 = timestamps[i]
                    break
        else:  # Moving down
            for i, h in enumerate(altitudes):
                if t_10 is None and h <= h_10:
                    t_10 = timestamps[i]
                if t_90 is None and h <= h_90:
                    t_90 = timestamps[i]
                    break
        
        if t_10 is not None and t_90 is not None:
            rise_time = t_90 - t_10
        else:
            rise_time = float('inf')
    else:
        rise_time = 0.0  # Already at target
    
    # 4. Maximum overshoot
    s = np.sign(target - h0)
    overshoot_values = s * (altitudes - target)
    max_overshoot_abs = np.max(overshoot_values.clip(min=0))
    
    if A > 0.01:
        max_overshoot_percent = 100 * max_overshoot_abs / A
    else:
        max_overshoot_percent = 0.0
    
    # 5. Control chattering (sign changes per minute)
    sign_changes = 0
    deadband = 0.01  # Small deadband to ignore noise
    
    for i in range(1, len(velocities)):
        v_prev = velocities[i-1]
        v_curr = velocities[i]
        
        # Only count significant sign changes
        if abs(v_prev) > deadband and abs(v_curr) > deadband:
            if np.sign(v_prev) != np.sign(v_curr):
                sign_changes += 1
    
    test_duration_minutes = (T - t0) / 60.0
    chattering = sign_changes / test_duration_minutes if test_duration_minutes > 0 else 0
    
    # 6. RMS error
    rms_error = np.sqrt(np.mean(errors**2))
    
    return {
        'settling_time': settling_time,
        'steady_state_error': steady_state_error,
        'rise_time': rise_time,
        'max_overshoot_percent': max_overshoot_percent,
        'control_chattering': chattering,
        'rms_error': rms_error,
        'final_error': abs(errors[-1]),
        'mean_absolute_error': np.mean(np.abs(errors))
    }

# ============================================================================
# PROVIDED: Main control loop and data collection
# Study this to understand the control system structure
# ============================================================================

def run_heuristic_control():
    """Execute the heuristic control test"""
    ensure_dir(OUTPUT_DIR)
    
    # Initialize drone
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    
    try:
        # Check battery
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 30:
            print("ERROR: Battery too low for test. Need at least 30%")
            return None
        
        print("Taking off...")
        tello.takeoff()
        time.sleep(3)  # Allow stabilization
        
        # Move to initial position (0.5m) first
        print(f"Moving to initial position ({INITIAL_ALTITUDE:.1f}m)...")
        move_to_initial_position(tello, INITIAL_ALTITUDE, 10.0)
        time.sleep(2)  # Brief pause before starting test
        
        # Get initial readings (should be close to 0.5m now)
        initial_data = get_sensor_data(tello)
        initial_altitude = get_current_altitude(initial_data)
        
        print(f"Initial altitude: {initial_altitude:.2f}m")
        print(f"Target altitude: {TARGET_ALTITUDE:.2f}m")
        print(f"Tolerance: ±{TOLERANCE:.2f}m")
        print(f"Starting heuristic control test: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m for {TEST_DURATION:.0f} seconds...")
        
        # Start mission and open CSV
        mission_start_time = time.time()
        csv_filename = os.path.join(OUTPUT_DIR, "heuristic_control_data.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'mission_time', 'target_altitude', 'current_altitude', 'error',
                'velocity_cmd', 'control_action', 'height_sonar', 'height_baro',
                'velocity_z_measured', 'battery'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Control loop
            next_sample_time = mission_start_time
            samples_collected = 0
            
            while time.time() - mission_start_time < TEST_DURATION:
                current_time = time.time()
                
                # Sample at regular intervals
                if current_time >= next_sample_time:
                    mission_time = current_time - mission_start_time
                    
                    # Get sensor readings
                    sensor_data = get_sensor_data(tello)
                    current_altitude = get_current_altitude(sensor_data)
                    error = TARGET_ALTITUDE - current_altitude
                    
                    # Heuristic control decision
                    velocity_cmd, control_action = heuristic_controller(
                        current_altitude, TARGET_ALTITUDE, TOLERANCE
                    )
                    
                    # Send control command
                    rc_cmd = velocity_to_rc(velocity_cmd)
                    tello.send_rc_control(0, 0, rc_cmd, 0)
                    
                    # Log data
                    row = {
                        'mission_time': mission_time,
                        'target_altitude': TARGET_ALTITUDE,
                        'current_altitude': current_altitude,
                        'error': error,
                        'velocity_cmd': velocity_cmd,
                        'control_action': control_action,
                        'height_sonar': sensor_data['height_sonar'],
                        'height_baro': sensor_data['height_baro'],
                        'velocity_z_measured': sensor_data['velocity_z'],
                        'battery': sensor_data['battery']
                    }
                    
                    writer.writerow(row)
                    samples_collected += 1
                    
                    # Progress update every 5 seconds
                    if samples_collected % 100 == 0:  # Every 5 seconds at 20Hz
                        print(f"  t={mission_time:.1f}s: Alt={current_altitude:.2f}m, "
                              f"Error={error:+.2f}m, Action={control_action}, "
                              f"Bat={sensor_data['battery']:.0f}%")
                    
                    next_sample_time += SAMPLE_DT
                
                time.sleep(0.01)
        
        # Stop all motion
        tello.send_rc_control(0, 0, 0, 0)
        print("Heuristic control test complete!")
        
        # Land
        print("Landing...")
        tello.land()
        time.sleep(2)
        
        print(f"Data saved to {csv_filename}")
        print(f"Total samples collected: {samples_collected}")
        
        return csv_filename
        
    except Exception as e:
        print(f"Error during test: {e}")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except:
            pass
        return None
    finally:
        try:
            tello.end()
        except:
            pass
        print("Tello connection closed")

# ============================================================================
# PROVIDED: Analysis and plotting
# ============================================================================

def analyze_heuristic_performance(csv_filename):
    """Analyze the heuristic controller performance and generate plots"""
    
    print(f"\nAnalyzing heuristic control data from {csv_filename}...")
    
    # Load data
    import pandas as pd
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # Extract data for analysis
    data = {
        'mission_time': df['mission_time'].values,
        'current_altitude': df['current_altitude'].values,
        'velocity_cmd': df['velocity_cmd'].values,
        'error': df['error'].values,
        'control_action': df['control_action'].values
    }
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(data)
    
    # ================= PLOTTING =================
    
    # Plot 1: Height tracking
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Altitude vs time
    ax1.plot(data['mission_time'], data['current_altitude'], 'b-', linewidth=2, label='Measured Altitude')
    ax1.axhline(TARGET_ALTITUDE, color='r', linestyle='--', linewidth=2, label=f'Target ({TARGET_ALTITUDE}m)')
    ax1.axhline(INITIAL_ALTITUDE, color='g', linestyle=':', linewidth=2, label=f'Initial ({INITIAL_ALTITUDE}m)')
    ax1.axhline(TARGET_ALTITUDE + TOLERANCE, color='r', linestyle=':', alpha=0.7, label=f'Tolerance (±{TOLERANCE}m)')
    ax1.axhline(TARGET_ALTITUDE - TOLERANCE, color='r', linestyle=':', alpha=0.7)
    ax1.fill_between(data['mission_time'], TARGET_ALTITUDE - TOLERANCE, TARGET_ALTITUDE + TOLERANCE, 
                     alpha=0.2, color='green', label='Acceptable Band')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title(f'Heuristic Controller: {INITIAL_ALTITUDE}m → {TARGET_ALTITUDE}m Altitude Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error vs time
    ax2.plot(data['mission_time'], data['error'], 'r-', linewidth=1.5, label='Tracking Error')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(TOLERANCE, color='r', linestyle=':', alpha=0.7)
    ax2.axhline(-TOLERANCE, color='r', linestyle=':', alpha=0.7)
    ax2.fill_between(data['mission_time'], -TOLERANCE, TOLERANCE, alpha=0.2, color='green')
    ax2.set_ylabel('Error [m]')
    ax2.set_title('Tracking Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Control output vs time
    ax3.plot(data['mission_time'], data['velocity_cmd'], 'g-', linewidth=2, label='Velocity Command')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity Command [m/s]')
    ax3.set_title('Control Output (Bang-Bang Control)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heuristic_performance.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Control action timeline
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create numerical representation of control actions
    action_map = {'HOLD': 0, 'CLIMB': 1, 'DESCEND': -1}
    action_values = [action_map.get(action, 0) for action in data['control_action']]
    
    ax.plot(data['mission_time'], action_values, 'k-', linewidth=2, marker='o', markersize=3)
    ax.set_ylabel('Control Action')
    ax.set_xlabel('Time [s]')
    ax.set_title('Heuristic Controller: Control Action Timeline')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['DESCEND', 'HOLD', 'CLIMB'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'control_actions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # ================= PERFORMANCE SUMMARY =================
    
    print(f"\n{'='*60}")
    print("HEURISTIC CONTROLLER PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Test trajectory: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m")
    
    print(f"\n--- PERFORMANCE METRICS ---")
    print(f"Settling time:           {metrics['settling_time']:.2f} s" if metrics['settling_time'] != float('inf') else "Settling time:           Never settles (inf)")
    print(f"Steady-state error:      {metrics['steady_state_error']:.3f} m")
    print(f"Rise time:               {metrics['rise_time']:.2f} s" if metrics['rise_time'] != float('inf') else "Rise time:               Not achieved (inf)")
    print(f"Maximum overshoot:       {metrics['max_overshoot_percent']:.1f} %")
    print(f"Control chattering:      {metrics['control_chattering']:.1f} switches/min")
    print(f"RMS error:               {metrics['rms_error']:.3f} m")
    print(f"Final error:             {metrics['final_error']:.3f} m")
    print(f"Mean absolute error:     {metrics['mean_absolute_error']:.3f} m")
    
    # ================= TODO: Add your analysis ====================
    print(f"\n--- TODO: ANALYZE THE CONTROLLER BEHAVIOR ---")
    print(f"Examine the plots and metrics above, then answer:")
    print(f"  1. Does the system reach and maintain the target altitude?")
    print(f"  2. Is the behavior smooth or jerky?")
    print(f"  3. Is it stable or oscillatory?")
    print(f"  4. What causes any oscillations you observe?")
    print(f"  5. What are the fundamental limitations of this approach?")
    print(f"\nWrite your observations in a text file or comments.")
    
    # Save metrics to file
    import json
    metrics_filename = os.path.join(OUTPUT_DIR, 'heuristic_metrics.json')
    
    # Convert any inf values to string for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if value == float('inf'):
            metrics_json[key] = "inf"
        elif np.isnan(value):
            metrics_json[key] = "nan"
        else:
            metrics_json[key] = float(value)
    
    with open(metrics_filename, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_filename}")
    print(f"Plots saved to {OUTPUT_DIR}/")
    
    return metrics

def main():
    print("Phase 3: Heuristic Control Approach")
    print("=" * 50)
    print(f"Initial altitude: {INITIAL_ALTITUDE}m")
    print(f"Target altitude: {TARGET_ALTITUDE}m")
    print(f"Tolerance: ±{TOLERANCE}m")
    print(f"Test duration: {TEST_DURATION}s")
    
    # Run the heuristic control test
    csv_filename = run_heuristic_control()
    
    if csv_filename:
        # Analyze the collected data
        metrics = analyze_heuristic_performance(csv_filename)
        print(f"\nPhase 3 complete! Files saved in {OUTPUT_DIR}/")
        print("Next: Run pid_altitude_control.py for Phase 4")
    else:
        print("Heuristic control test failed - no data to analyze")

if __name__ == "__main__":
    main()