#!/usr/bin/env python3
"""
Lab 1 - Phase 4: PID Altitude Control with Systematic Tuning

This script implements a PID controller for altitude control and performs
systematic tuning to find optimal gains.

Test: 0.5m -> 1.5m -> 0.5m for each gain configuration
Ensure heuristic baseline trajectory and PID have fair comparison.
"""

import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello

# Configuration
OUTPUT_DIR = "Lab1-Phase4-Simple"
SAMPLE_DT = 0.05  # 20 Hz sampling rate

# Test parameters
INITIAL_ALTITUDE = 0.5  # meters - fixed starting position
TARGET_ALTITUDE = 1.5   # meters - matching heuristic baseline
TEST_DURATION = 15.0    # seconds per test phase

# Tuning parameters - You can modify these to test different gain values
KP_VALUES = [0.3, 0.5, 1.0]      # Test these Kp values
KI_VALUES = [0.01, 0.05, 0.1]    # Test these Ki values
KD_VALUES = [0.01, 0.05, 0.1]    # Test these Kd values

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def velocity_to_rc(velocity_mps):
    """Convert velocity in m/s to RC throttle command (-100 to +100)"""
    velocity_mps = max(-1.0, min(1.0, velocity_mps))
    rc_cmd = int(velocity_mps * 100)
    return max(-100, min(100, rc_cmd))

# ============================================================================
# PROVIDED: Sensor reading function (reuse from previous phases)
# ============================================================================
def get_sensor_data(tello):
    """Get sensor data from Tello"""
    try:
        height_sonar = tello.get_distance_tof() / 100.0  # cm -> m
        height_baro = tello.get_barometer() / 100.0       # cm -> m  
        velocity_z = tello.get_speed_z() / 100.0          # cm/s -> m/s
        battery = tello.get_battery()                     # already in %
        
        return {
            'height_sonar': height_sonar,
            'height_baro': height_baro,
            'velocity_z': velocity_z,
            'battery': battery
        }
        
    except Exception as e:
        print(f"Error reading sensors: {e}")
        return {
            'height_sonar': 0.0,
            'height_baro': 0.0,
            'velocity_z': 0.0,
            'battery': 0.0
        }

def get_current_altitude(sensor_data):
    """Get current altitude estimate, preferring sonar if valid"""
    if not np.isnan(sensor_data['height_sonar']) and sensor_data['height_sonar'] > 0.1:
        return sensor_data['height_sonar']
    elif not np.isnan(sensor_data['height_baro']):
        return sensor_data['height_baro']
    else:
        return 1.0  # Fallback estimate

# ============================================================================
# TODO: Implement PID Controller
# This is the core learning objective for Phase 4
# ============================================================================

class PIDController:
    """Simple PID Controller for altitude control"""
    
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True
        
    def update(self, setpoint, measured_value):
        """
        Update PID controller with new measurement.
        
        TODO: Implement PID calculation
        1. Calculate error = setpoint - measured_value
        2. Calculate proportional term: P = kp * error
        3. Update and calculate integral term:
           - self.integral += error * self.dt  (accumulate error over time)
           - I = ki * self.integral
        4. Calculate derivative term:
           - If first run: D = 0 (no previous error to compare)
           - Otherwise: D = kd * (error - self.previous_error) / self.dt
           - Set self.first_run = False after first calculation
        5. Calculate total output = P + I + D
        6. Clamp output to [-1.0, 1.0] for safety
        7. Store current error as self.previous_error for next iteration
        8. Return: (output, P, I, D) for logging/plotting
        
        Args:
            setpoint: Desired altitude [m]
            measured_value: Current altitude [m]
        
        Returns:
            output: Control output (velocity command) [m/s]
            proportional: P term value
            integral: I term value  
            derivative: D term value
        """
        # TODO: Implement PID calculation here (~15-20 lines)
        error = 0.0  # TODO
        proportional = 0.0  # TODO
        integral = 0.0  # TODO
        derivative = 0.0  # TODO
        output = 0.0  # TODO
        
        return output, proportional, integral, derivative
    
    def reset(self):
        """Reset PID internal state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True

# ============================================================================
# PROVIDED: Test execution framework
# Study this to understand systematic tuning methodology
# ============================================================================

def move_to_altitude(tello, target_altitude, pid_controller, duration):
    """
    Move drone to target altitude using PID control for specified duration.
    """
    print(f"    Moving to {target_altitude:.1f}m for {duration:.0f}s...")
    
    data = {
        'mission_time': [],
        'target_altitude': [],
        'current_altitude': [],
        'error': [],
        'velocity_cmd': [],
        'proportional': [],
        'integral': [],
        'derivative': []
    }
    
    start_time = time.time()
    next_sample_time = start_time
    
    while time.time() - start_time < duration:
        current_time = time.time()
        
        if current_time >= next_sample_time:
            mission_time = current_time - start_time
            
            # Get sensor readings
            sensor_data = get_sensor_data(tello)
            current_altitude = get_current_altitude(sensor_data)
            
            # PID control
            velocity_cmd, p_term, i_term, d_term = pid_controller.update(target_altitude, current_altitude)
            
            # Send control command
            rc_cmd = velocity_to_rc(velocity_cmd)
            tello.send_rc_control(0, 0, rc_cmd, 0)
            
            # Log data
            data['mission_time'].append(mission_time)
            data['target_altitude'].append(target_altitude)
            data['current_altitude'].append(current_altitude)
            data['error'].append(target_altitude - current_altitude)
            data['velocity_cmd'].append(velocity_cmd)
            data['proportional'].append(p_term)
            data['integral'].append(i_term)
            data['derivative'].append(d_term)
            
            next_sample_time += SAMPLE_DT
        
        time.sleep(0.01)
    
    # Stop motion
    tello.send_rc_control(0, 0, 0, 0)
    final_altitude = get_current_altitude(get_sensor_data(tello))
    print(f"    Reached {final_altitude:.2f}m")
    
    return data

# ============================================================================
# PROVIDED: Performance metrics calculation (reuse from Phase 3)
# ============================================================================

def calculate_performance_metrics(timestamps, altitudes, target, velocities):
    """Calculate performance metrics for PID controller"""
    
    errors = target - altitudes
    t0 = timestamps[0]
    T = timestamps[-1]
    
    # Settling band (±5% of setpoint)
    delta_band = 0.05 * abs(target)
    
    # 1. Settling time
    abs_errors = np.abs(errors)
    settling_time = None
    
    for i in range(len(timestamps)):
        if all(abs_errors[i:] <= delta_band):
            settling_time = timestamps[i] - t0
            break
    
    if settling_time is None:
        settling_time = float('inf')
    
    # 2. Steady-state analysis
    if settling_time != float('inf'):
        T_ss = T - (t0 + settling_time)
    else:
        T_ss = min(10.0, T - t0)
    
    ss_start_time = T - T_ss
    ss_mask = timestamps >= ss_start_time
    ss_errors = errors[ss_mask]
    
    steady_state_error = np.mean(np.abs(ss_errors)) if len(ss_errors) > 0 else float('inf')
    
    # 3. Rise time (10% to 90%)
    h0 = altitudes[0]
    A = abs(target - h0)
    
    if A > 0.01:
        h_10 = h0 + 0.1 * (target - h0)
        h_90 = h0 + 0.9 * (target - h0)
        
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
        rise_time = 0.0
    
    # 4. Maximum overshoot
    s = np.sign(target - h0)
    overshoot_values = s * (altitudes - target)
    max_overshoot_abs = np.max(overshoot_values.clip(min=0))
    
    if A > 0.01:
        max_overshoot_percent = 100 * max_overshoot_abs / A
    else:
        max_overshoot_percent = 0.0
    
    # 5. Control chattering
    sign_changes = 0
    deadband = 0.01
    
    for i in range(1, len(velocities)):
        v_prev = velocities[i-1]
        v_curr = velocities[i]
        
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
# PROVIDED: Plotting function
# ============================================================================

def plot_pid_test(data, kp, ki, kd, test_name, save_path):
    """Generate plots for a PID test"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    times = np.array(data['mission_time'])
    altitudes = np.array(data['current_altitude'])
    targets = np.array(data['target_altitude'])
    errors = np.array(data['error'])
    velocities = np.array(data['velocity_cmd'])
    p_terms = np.array(data['proportional'])
    i_terms = np.array(data['integral'])
    d_terms = np.array(data['derivative'])
    
    # Plot 1: Height tracking
    ax1.plot(times, altitudes, 'b-', linewidth=2, label='Measured Altitude')
    ax1.plot(times, targets, 'r--', linewidth=2, label='Target Altitude')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title(f'{test_name}: Height Tracking (Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2.plot(times, errors, 'r-', linewidth=1.5)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Error [m]')
    ax2.set_title('Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Control output
    ax3.plot(times, velocities, 'g-', linewidth=1.5)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity Command [m/s]')
    ax3.set_title('Control Output')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PID terms
    ax4.plot(times, p_terms, 'r-', linewidth=1, alpha=0.8, label=f'P (Kp={kp:.2f})')
    ax4.plot(times, i_terms, 'g-', linewidth=1, alpha=0.8, label=f'I (Ki={ki:.3f})')
    ax4.plot(times, d_terms, 'b-', linewidth=1, alpha=0.8, label=f'D (Kd={kd:.3f})')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('PID Terms [m/s]')
    ax4.set_title('PID Term Contributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def run_gain_test(tello, kp, ki, kd, test_name):
    """
    Run one complete gain test: 0.5m → 1.5m → 0.5m
    """
    print(f"\n  Running {test_name}")
    print(f"    Gains: Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"    Test sequence: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m → {INITIAL_ALTITUDE:.1f}m")
    
    # Create PID controller
    pid = PIDController(kp=kp, ki=ki, kd=kd, dt=SAMPLE_DT)
    
    # Phase 1: Go to initial altitude (0.5m)
    print(f"    Phase 1: Moving to initial position ({INITIAL_ALTITUDE:.1f}m)")
    pid.reset()
    move_to_altitude(tello, INITIAL_ALTITUDE, pid, TEST_DURATION)
    time.sleep(1)
    
    # Phase 2: Go to target altitude (1.5m) - TEST DATA
    print(f"    Phase 2: Testing gain - moving to target ({TARGET_ALTITUDE:.1f}m)")
    pid.reset()
    test_data = move_to_altitude(tello, TARGET_ALTITUDE, pid, TEST_DURATION)
    time.sleep(1)
    
    # Phase 3: Return to initial (0.5m)
    print(f"    Phase 3: Returning to initial position ({INITIAL_ALTITUDE:.1f}m)")
    pid.reset()
    move_to_altitude(tello, INITIAL_ALTITUDE, pid, TEST_DURATION)
    time.sleep(1)
    
    return test_data

# ============================================================================
# PROVIDED: Systematic tuning procedure
# ============================================================================

def systematic_pid_tuning():
    """Execute systematic PID tuning with 0.5m → 1.5m → 0.5m for each gain"""
    
    ensure_dir(OUTPUT_DIR)
    
    # Initialize drone
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    
    try:
        # Check battery
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 60:
            print("ERROR: Battery too low for extensive tuning. Need at least 60%")
            return None, None, None
        
        print("Taking off...")
        tello.takeoff()
        time.sleep(3)
        
        print(f"\n{'='*70}")
        print("SYSTEMATIC PID TUNING")
        print(f"{'='*70}")
        print(f"Test pattern for each gain: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m → {INITIAL_ALTITUDE:.1f}m")
        print(f"Analysis focuses on {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m phase (matching heuristic)")
        
        tuning_results = {}
        
        # STEP 1: P-only control
        print(f"\nSTEP 1: P-ONLY CONTROL (Ki=0, Kd=0)")
        print("=" * 50)
        
        best_kp = KP_VALUES[0]  # Default
        best_p_metrics = None
        p_results = {}
        
        for kp in KP_VALUES:
            test_name = f"P_only_Kp_{kp:.1f}"
            
            test_data = run_gain_test(tello, kp, 0.0, 0.0, test_name)
            
            metrics = calculate_performance_metrics(
                np.array(test_data['mission_time']),
                np.array(test_data['current_altitude']),
                TARGET_ALTITUDE,
                np.array(test_data['velocity_cmd'])
            )
            
            p_results[kp] = {'test_data': test_data, 'metrics': metrics}
            
            plot_path = os.path.join(OUTPUT_DIR, f'{test_name}.png')
            plot_pid_test(test_data, kp, 0.0, 0.0, test_name, plot_path)
            
            print(f"    Kp={kp:.1f}: Settling={metrics['settling_time']:.1f}s, "
                  f"Overshoot={metrics['max_overshoot_percent']:.1f}%, "
                  f"RMS={metrics['rms_error']:.3f}m")
            
            # TODO: Select best Kp based on reasonable performance
            # Compare metrics and choose the one with lowest RMS error
            # that also has settling_time < 15s and overshoot < 30%
            if (metrics['settling_time'] < 15.0 and 
                metrics['max_overshoot_percent'] < 30.0):
                if best_p_metrics is None or metrics['rms_error'] < best_p_metrics['rms_error']:
                    best_kp = kp
                    best_p_metrics = metrics
        
        print(f"\n  Selected best Kp: {best_kp:.2f}")
        tuning_results['p_only'] = p_results
        
        # STEP 2: PI control
        print(f"\nSTEP 2: PI CONTROL (Kp={best_kp:.2f}, Kd=0)")
        print("=" * 50)
        
        best_ki = KI_VALUES[0]  # Default
        best_pi_metrics = None
        pi_results = {}
        
        for ki in KI_VALUES:
            test_name = f"PI_Kp_{best_kp:.1f}_Ki_{ki:.2f}"
            
            test_data = run_gain_test(tello, best_kp, ki, 0.0, test_name)
            
            metrics = calculate_performance_metrics(
                np.array(test_data['mission_time']),
                np.array(test_data['current_altitude']),
                TARGET_ALTITUDE,
                np.array(test_data['velocity_cmd'])
            )
            
            pi_results[ki] = {'test_data': test_data, 'metrics': metrics}
            
            plot_path = os.path.join(OUTPUT_DIR, f'{test_name}.png')
            plot_pid_test(test_data, best_kp, ki, 0.0, test_name, plot_path)
            
            print(f"    Ki={ki:.2f}: Settling={metrics['settling_time']:.1f}s, "
                  f"SS_Error={metrics['steady_state_error']:.3f}m, "
                  f"RMS={metrics['rms_error']:.3f}m")
            
            # TODO: Select best Ki based on steady-state error reduction
            if metrics['settling_time'] < 20.0:
                if best_pi_metrics is None or metrics['steady_state_error'] < best_pi_metrics['steady_state_error']:
                    best_ki = ki
                    best_pi_metrics = metrics
        
        print(f"\n  Selected best Ki: {best_ki:.3f}")
        tuning_results['pi'] = pi_results
        
        # STEP 3: PID control
        print(f"\nSTEP 3: PID CONTROL (Kp={best_kp:.2f}, Ki={best_ki:.3f})")
        print("=" * 50)
        
        best_kd = KD_VALUES[0]  # Default
        best_pid_metrics = None
        pid_results = {}
        
        for kd in KD_VALUES:
            test_name = f"PID_Kp_{best_kp:.1f}_Ki_{best_ki:.2f}_Kd_{kd:.2f}"
            
            test_data = run_gain_test(tello, best_kp, best_ki, kd, test_name)
            
            metrics = calculate_performance_metrics(
                np.array(test_data['mission_time']),
                np.array(test_data['current_altitude']),
                TARGET_ALTITUDE,
                np.array(test_data['velocity_cmd'])
            )
            
            pid_results[kd] = {'test_data': test_data, 'metrics': metrics}
            
            plot_path = os.path.join(OUTPUT_DIR, f'{test_name}.png')
            plot_pid_test(test_data, best_kp, best_ki, kd, test_name, plot_path)
            
            print(f"    Kd={kd:.2f}: Settling={metrics['settling_time']:.1f}s, "
                  f"Overshoot={metrics['max_overshoot_percent']:.1f}%, "
                  f"RMS={metrics['rms_error']:.3f}m")
            
            # TODO: Select best Kd based on overshoot reduction and settling time
            if metrics['settling_time'] < 15.0:
                if best_pid_metrics is None or metrics['rms_error'] < best_pid_metrics['rms_error']:
                    best_kd = kd
                    best_pid_metrics = metrics
        
        print(f"\n  Selected best Kd: {best_kd:.3f}")
        tuning_results['pid'] = pid_results
        
        print(f"\n{'='*70}")
        print("FINAL OPTIMIZED PID GAINS")
        print(f"{'='*70}")
        print(f"Kp = {best_kp:.2f}")
        print(f"Ki = {best_ki:.3f}")
        print(f"Kd = {best_kd:.3f}")
        
        # Save tuning results
        tuning_summary = {
            'experimental_design': {
                'initial_altitude': INITIAL_ALTITUDE,
                'target_altitude': TARGET_ALTITUDE,
                'test_sequence': f'{INITIAL_ALTITUDE}m → {TARGET_ALTITUDE}m → {INITIAL_ALTITUDE}m for each gain',
                'analysis_phase': f'{INITIAL_ALTITUDE}m → {TARGET_ALTITUDE}m (matches heuristic baseline)'
            },
            'best_gains': {'kp': best_kp, 'ki': best_ki, 'kd': best_kd},
            'tuning_process': {}
        }
        
        # Store metrics
        for step_name, step_results in tuning_results.items():
            tuning_summary['tuning_process'][step_name] = {}
            for gain_value, result in step_results.items():
                metrics = result['metrics']
                serializable_metrics = {}
                for key, value in metrics.items():
                    if value == float('inf'):
                        serializable_metrics[key] = "inf"
                    elif np.isnan(value):
                        serializable_metrics[key] = "nan"
                    else:
                        serializable_metrics[key] = float(value)
                tuning_summary['tuning_process'][step_name][str(gain_value)] = serializable_metrics
        
        tuning_file = os.path.join(OUTPUT_DIR, 'simple_pid_tuning_results.json')
        with open(tuning_file, 'w') as f:
            json.dump(tuning_summary, f, indent=2)
        
        print(f"\nTuning results saved to {tuning_file}")
        
        # Land
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        tello.end()
        
        return best_kp, best_ki, best_kd
        
    except Exception as e:
        print(f"Error during tuning: {e}")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            tello.end()
        except:
            pass
        return None, None, None

def run_final_comparison_test(kp, ki, kd):
    """Run final test using optimized gains matching heuristic baseline exactly"""
    
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TEST")
    print(f"{'='*70}")
    print(f"PID Gains: Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"Test: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m (matching heuristic)")
    
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    
    try:
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 30:
            print("ERROR: Battery too low. Need at least 30%")
            return None
        
        print("Taking off...")
        tello.takeoff()
        time.sleep(3)
        
        # Position to initial altitude
        print(f"Positioning to initial altitude ({INITIAL_ALTITUDE:.1f}m)...")
        pid = PIDController(kp=0.3, ki=0.01, kd=0.01, dt=SAMPLE_DT)
        pid.reset()
        move_to_altitude(tello, INITIAL_ALTITUDE, pid, 10.0)
        time.sleep(2)
        
        # Run comparison test: 0.5m → 1.5m for 30s
        print(f"Running PID comparison test: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m")
        pid = PIDController(kp=kp, ki=ki, kd=kd, dt=SAMPLE_DT)
        test_data = move_to_altitude(tello, TARGET_ALTITUDE, pid, 30.0)
        
        # Save to CSV
        csv_filename = os.path.join(OUTPUT_DIR, "pid_vs_heuristic_comparison.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'mission_time', 'target_altitude', 'current_altitude', 'error',
                'velocity_cmd', 'proportional', 'integral', 'derivative'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(len(test_data['mission_time'])):
                row = {field: test_data[field][i] for field in fieldnames}
                writer.writerow(row)
        
        # Analyze
        analyze_final_test(test_data, kp, ki, kd)
        
        # Land
        print("Landing...")
        tello.land()
        time.sleep(2)
        tello.end()
        
        print(f"Comparison test data saved to {csv_filename}")
        return csv_filename
        
    except Exception as e:
        print(f"Error during comparison test: {e}")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            tello.end()
        except:
            pass
        return None

def analyze_final_test(data, kp, ki, kd):
    """Analyze and plot the final comparison test"""
    
    times = np.array(data['mission_time'])
    altitudes = np.array(data['current_altitude'])
    errors = np.array(data['error'])
    velocities = np.array(data['velocity_cmd'])
    
    metrics = calculate_performance_metrics(times, altitudes, TARGET_ALTITUDE, velocities)
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1.plot(times, altitudes, 'b-', linewidth=2, label='PID Altitude')
    ax1.axhline(TARGET_ALTITUDE, color='r', linestyle='--', linewidth=2, label=f'Target ({TARGET_ALTITUDE}m)')
    ax1.axhline(INITIAL_ALTITUDE, color='g', linestyle=':', linewidth=2, label=f'Initial ({INITIAL_ALTITUDE}m)')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title(f'PID vs Heuristic: {INITIAL_ALTITUDE}m → {TARGET_ALTITUDE}m\n(Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, errors, 'r-', linewidth=1.5)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Error [m]')
    ax2.set_title('Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(times, velocities, 'g-', linewidth=1.5, label='PID (Continuous)')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='Heuristic UP (1.0 m/s)')
    ax3.axhline(-0.8, color='orange', linestyle='--', alpha=0.7, label='Heuristic DOWN (-0.8 m/s)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity Command [m/s]')
    ax3.set_title('Control Output: PID vs Heuristic')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.text(0.1, 0.9, f'Settling Time: {metrics["settling_time"]:.2f} s', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.8, f'Steady-State Error: {metrics["steady_state_error"]:.3f} m', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, f'Rise Time: {metrics["rise_time"]:.2f} s', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f'Max Overshoot: {metrics["max_overshoot_percent"]:.1f} %', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, f'RMS Error: {metrics["rms_error"]:.3f} m', transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.4, f'Control Chattering: {metrics["control_chattering"]:.1f} switches/min', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Performance Metrics Summary')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pid_vs_heuristic_final.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("PID CONTROLLER PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Settling time:           {metrics['settling_time']:.2f} s" if metrics['settling_time'] != float('inf') else "Settling time:           Never settles (inf)")
    print(f"Steady-state error:      {metrics['steady_state_error']:.3f} m")
    print(f"Rise time:               {metrics['rise_time']:.2f} s" if metrics['rise_time'] != float('inf') else "Rise time:               Not achieved (inf)")
    print(f"Maximum overshoot:       {metrics['max_overshoot_percent']:.1f} %")
    print(f"Control chattering:      {metrics['control_chattering']:.1f} switches/min")
    print(f"RMS error:               {metrics['rms_error']:.3f} m")
    print(f"Final error:             {metrics['final_error']:.3f} m")
    print(f"Mean absolute error:     {metrics['mean_absolute_error']:.3f} m")
    
    # Save results
    results = {
        'test_type': 'pid_vs_heuristic_comparison',
        'trajectory': f'{INITIAL_ALTITUDE}m → {TARGET_ALTITUDE}m',
        'pid_gains': {'kp': kp, 'ki': ki, 'kd': kd},
        'performance_metrics': {
            key: float(value) if value != float('inf') and not np.isnan(value) else str(value)
            for key, value in metrics.items()
        }
    }
    
    results_file = os.path.join(OUTPUT_DIR, 'final_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

def main():
    """Main function to execute the PID tuning procedure"""
    
    print("Lab 1 - Phase 4: PID Tuning with Systematic Gain Selection")
    print("=" * 80)
    print(f"Test pattern: {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m → {INITIAL_ALTITUDE:.1f}m for each gain")
    print(f"Analysis focuses on {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m (matches heuristic baseline)")
    print("Systematic progression: P → PI → PID")
    
    try:
        # Step 1: Systematic PID tuning
        best_kp, best_ki, best_kd = systematic_pid_tuning()
        
        if best_kp is None:
            print("Tuning failed - aborting")
            return
        
        # Step 2: Final comparison test
        csv_filename = run_final_comparison_test(best_kp, best_ki, best_kd)
        
        if csv_filename:
            print(f"\n{'='*80}")
            print("PHASE 4 COMPLETE!")
            print(f"{'='*80}")
            print(f"Optimized PID gains: Kp={best_kp:.2f}, Ki={best_ki:.3f}, Kd={best_kd:.3f}")
            print(f"Files created:")
            print(f"  - {csv_filename}")
            print(f"  - {OUTPUT_DIR}/simple_pid_tuning_results.json")
            print(f"  - {OUTPUT_DIR}/final_comparison_results.json")
            print(f"  - Individual plots for each gain test")
            print(f"  - Final comparison plot")
            print(f"\nAll results saved in {OUTPUT_DIR}/")
            print(f"\nReady for comparison with Phase 3 heuristic results!")
            print(f"Both controllers tested on same {INITIAL_ALTITUDE:.1f}m → {TARGET_ALTITUDE:.1f}m trajectory")
            
            return best_kp, best_ki, best_kd
        else:
            print("Final test failed")
            return None, None, None
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return None, None, None
    except Exception as e:
        print(f"Error during PID testing: {e}")
        return None, None, None

if __name__ == "__main__":
    main()