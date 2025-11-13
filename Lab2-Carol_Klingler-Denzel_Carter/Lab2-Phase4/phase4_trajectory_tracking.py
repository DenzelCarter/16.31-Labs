#!/usr/bin/env python3
"""
Lab 2 - Phase 4: Trajectory Tracking Comparison

Objectives:
1. Implement helix trajectory generation
2. Test both PID and pole placement controllers on trajectory tracking
3. Compare performance and demonstrate control design advantages

Implement:
- Helix trajectory generation equations
- Trajectory tracking metrics calculation
- Performance comparison analysis
"""

import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from djitellopy import Tello
import cv2
from pupil_apriltags import Detector

# Configuration
OUTPUT_DIR = "Lab2-Phase4"
SAMPLE_DT = 0.05  # 50ms = 20 Hz sampling

# AprilTag parameters
TAG_SIZE = 0.162  # meters
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]

# Helix trajectory parameters (from lab specifications)
HELIX_CENTER = [0.0, 0.0, 1.5]    # x=0 (centered), y=0 (tag height), z=1.5m (distance)
HELIX_RADIUS = 0.3                 # 30cm lateral movement (y-direction)
HELIX_AMPLITUDE = 0.4              # 40cm vertical movement (z-direction)
CYCLE_TIME = 20.0                  # 20 seconds per cycle

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def generate_helix_trajectory(t, center, radius, amplitude, cycle_time):
    """
    Generate vertical helix (ellipse) trajectory reference.
    
    TODO: Implement helix trajectory equations.
    
    The trajectory should:
    - Keep x constant (no forward/backward motion)
    - Oscillate in y (lateral, left-right)
    - Oscillate in z (vertical, up-down)
    - Complete one cycle in cycle_time seconds
    
    Consider:
    - What is the angular frequency omega for the given cycle time?
    - Which trigonometric function (sin/cos) should be used for each axis?
    - At t=0, where should the trajectory start?
    
    Args:
        t: Current time [s]
        center: [x_c, y_c, z_c] center position in tag frame
        radius: Lateral oscillation radius [m]
        amplitude: Vertical oscillation amplitude [m]
        cycle_time: Time for one complete cycle [s]
    
    Returns:
        x_ref, y_ref, z_ref: Reference positions in tag frame
    """
    x_c, y_c, z_c = center
    
    # TODO: Calculate angular frequency
    # Hint: omega = 2*pi / T where T is the cycle time
    omega = 2.0 * np.pi / float(cycle_time)  # REPLACE THIS
    
    # TODO: Generate helix trajectory
    # Hint: x stays constant, y and z oscillate with sin and cos
    x_ref = float(x_c)  # REPLACE THIS
    y_ref = float(y_c + radius * np.cos(omega * t))  # REPLACE THIS
    z_ref = float(z_c + amplitude * np.sin(omega * t))
    
    return x_ref, y_ref, z_ref

# Controllers from previous phases
class PIDController:
    """Single-axis PID controller from Lab 1"""
    
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True
    
    def update(self, setpoint, measured_value):
        """Calculate PID control output"""
        error = setpoint - measured_value
        
        p_term = self.kp * error
        
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        if self.first_run:
            d_term = 0.0
            self.first_run = False
        else:
            d_term = self.kd * (error - self.previous_error) / self.dt
        
        self.previous_error = error
        
        output = p_term + i_term + d_term
        
        return output, p_term, i_term, d_term
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True

class MultiAxisPIDController:
    """Multi-axis PID controller from Phase 2"""
    
    def __init__(self, kp_x=0.5, ki_x=0.01, kd_x=0.05,
                       kp_y=0.5, ki_y=0.01, kd_y=0.05,
                       kp_z=0.5, ki_z=0.01, kd_z=0.05,
                       dt=0.05):
        self.pid_x = PIDController(kp_x, ki_x, kd_x, dt)
        self.pid_y = PIDController(kp_y, ki_y, kd_y, dt)
        self.pid_z = PIDController(kp_z, ki_z, kd_z, dt)
        self.dt = dt
    
    def update(self, x_ref, y_ref, z_ref, x_meas, y_meas, z_meas):
        """Calculate control velocities for all three axes"""
        vx, _, _, _ = self.pid_x.update(x_ref, x_meas)
        vy, _, _, _ = self.pid_y.update(y_ref, y_meas)
        vz, _, _, _ = self.pid_z.update(z_ref, z_meas)
        
        max_vel = 0.3
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)
        vz = np.clip(vz, -max_vel, max_vel)
        
        return vx, vy, vz
    
    def reset(self):
        """Reset all controller states"""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

class PolePlacementController:
    """Pole placement controller from Phase 3"""
    
    def __init__(self, K, dt=0.05):
        self.K = K
        self.dt = dt
    
    def update(self, x_ref, y_ref, z_ref, x_meas, y_meas, z_meas):
        """Calculate control velocities using state feedback"""
        x = np.array([x_meas, y_meas, z_meas])
        x_ref_vec = np.array([x_ref, y_ref, z_ref])
        error = x - x_ref_vec
        
        u = -self.K @ error
        
        max_vel = 0.3
        vx = np.clip(u[0], -max_vel, max_vel)
        vy = np.clip(u[1], -max_vel, max_vel)
        vz = np.clip(u[2], -max_vel, max_vel)
        
        return vx, vy, vz

def get_apriltag_pose(frame, detector):
    """Detect AprilTag and return pose in tag coordinate frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE
    )
    
    if len(detections) > 0:
        detection = detections[0]
        pose = detection.pose_t.flatten()
        
        return {
            'detected': True,
            'x': pose[0],
            'y': pose[1],
            'z': pose[2]
        }
    else:
        return {
            'detected': False,
            'x': np.nan,
            'y': np.nan,
            'z': np.nan
        }

def tag_to_drone_velocity(vx_tag, vy_tag, vz_tag):
    """Transform velocity commands from tag frame to drone RC commands"""
    rc_left_right = int(-vx_tag * 100)
    rc_forward_backward = int(-vz_tag * 100)
    rc_up_down = int(vy_tag * 100)
    rc_yaw = 0
    
    rc_left_right = int(np.clip(rc_left_right, -100, 100))
    rc_forward_backward = int(np.clip(rc_forward_backward, -100, 100))
    rc_up_down = int(np.clip(rc_up_down, -100, 100))
    
    return rc_left_right, rc_forward_backward, rc_up_down, rc_yaw

def position_to_target(tello, detector, x_target, y_target, z_target, timeout=15.0):
    """Position drone to target location using simple proportional control"""
    start_time = time.time()
    settled_count = 0
    required_settled = 20
    
    print(f"  Target: x={x_target:.2f}, y={y_target:.2f}, z={z_target:.2f}")
    
    while time.time() - start_time < timeout:
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector)
        
        if pose['detected']:
            x_error = x_target - pose['x']
            y_error = y_target - pose['y']
            z_error = z_target - pose['z']
            
            if abs(x_error) < 0.1 and abs(y_error) < 0.1 and abs(z_error) < 0.1:
                settled_count += 1
                if settled_count >= required_settled:
                    print(f"  Positioned! Final: x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}")
                    break
            else:
                settled_count = 0
            
            vx = np.clip(0.3 * x_error, -0.2, 0.2)
            vy = np.clip(0.3 * y_error, -0.2, 0.2)
            vz = np.clip(0.3 * z_error, -0.2, 0.2)
            
            rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx, vy, vz)
            tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)
        
        time.sleep(0.05)
    
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

def track_helix_trajectory(tello, detector, controller, n_cycles, test_name):
    """
    Track helix trajectory and record performance data.
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector
        controller: Control object (PID or Pole Placement)
        n_cycles: Number of helix cycles to execute
        test_name: Name for saving results
    
    Returns:
        data: Dictionary of trajectory tracking data
        metrics: Dictionary of performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Tracking helix: {n_cycles} cycles ({n_cycles * CYCLE_TIME:.0f} seconds)")
    print(f"{'='*60}")
    
    ensure_dir(OUTPUT_DIR)
    
    print("Positioning to helix center...")
    position_to_target(tello, detector, *HELIX_CENTER, timeout=15.0)
    time.sleep(2)
    
    if hasattr(controller, 'reset'):
        controller.reset()
    
    print("Starting helix trajectory...")
    
    total_duration = n_cycles * CYCLE_TIME
    start_time = time.time()
    next_sample = start_time
    
    data = {
        'time': [],
        'cycle': [],
        'x_ref': [], 'y_ref': [], 'z_ref': [],
        'x_meas': [], 'y_meas': [], 'z_meas': [],
        'vx_cmd': [], 'vy_cmd': [], 'vz_cmd': []
    }
    
    consecutive_losses = 0
    max_consecutive_losses = 10
    
    while time.time() - start_time < total_duration:
        if time.time() >= next_sample:
            t = time.time() - start_time
            cycle_num = int(t / CYCLE_TIME)
            
            x_ref, y_ref, z_ref = generate_helix_trajectory(
                t, HELIX_CENTER, HELIX_RADIUS, HELIX_AMPLITUDE, CYCLE_TIME
            )
            
            frame = tello.get_frame_read().frame
            pose = get_apriltag_pose(frame, detector)
            
            if pose['detected']:
                consecutive_losses = 0
                
                vx, vy, vz = controller.update(
                    x_ref, y_ref, z_ref,
                    pose['x'], pose['y'], pose['z']
                )
                
                rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx, vy, vz)
                tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)
                
                data['time'].append(t)
                data['cycle'].append(cycle_num)
                data['x_ref'].append(x_ref)
                data['y_ref'].append(y_ref)
                data['z_ref'].append(z_ref)
                data['x_meas'].append(pose['x'])
                data['y_meas'].append(pose['y'])
                data['z_meas'].append(pose['z'])
                data['vx_cmd'].append(vx)
                data['vy_cmd'].append(vy)
                data['vz_cmd'].append(vz)
                
                if len(data['time']) % 40 == 0:
                    mean_error = calculate_tracking_error(data)
                    print(f"  t={t:.1f}s (cycle {cycle_num+1}/{n_cycles}), mean error={mean_error*100:.1f}cm")
            else:
                consecutive_losses += 1
                if consecutive_losses >= max_consecutive_losses:
                    print(f"  WARNING: Tag lost for {consecutive_losses * SAMPLE_DT:.1f}s, stopping trajectory")
                    break
            
            next_sample += SAMPLE_DT
        
        time.sleep(0.01)
    
    tello.send_rc_control(0, 0, 0, 0)
    
    print("Trajectory complete!")
    
    csv_filename = os.path.join(OUTPUT_DIR, f"{test_name}_trajectory.csv")
    save_test_data(data, csv_filename)
    
    metrics = calculate_trajectory_metrics(data, n_cycles)
    plot_trajectory_tracking(data, metrics, test_name)
    
    return data, metrics

def calculate_tracking_error(data):
    """Calculate current mean tracking error"""
    if len(data['x_meas']) == 0:
        return 0.0
    
    x_err = np.array(data['x_ref']) - np.array(data['x_meas'])
    y_err = np.array(data['y_ref']) - np.array(data['y_meas'])
    z_err = np.array(data['z_ref']) - np.array(data['z_meas'])
    
    euclidean_err = np.sqrt(x_err**2 + y_err**2 + z_err**2)
    
    return np.mean(euclidean_err)

def calculate_trajectory_metrics(data, n_cycles):
    """
    Calculate comprehensive trajectory tracking metrics.
    
    TODO: Implement trajectory performance metrics.
    
    Calculate:
    - Mean tracking error (Euclidean distance from reference)
    - Maximum deviation from reference
    - Per-cycle tracking errors
    - Error growth ratio (compare first and last cycle)
    - Control effort (RMS of velocity commands)
    
    Args:
        data: Dictionary of trajectory tracking data
        n_cycles: Number of cycles executed
    
    Returns:
        metrics: Dictionary of performance metrics
    """
    times = np.array(data['time'])
    cycles = np.array(data['cycle'])
    
    x_ref = np.array(data['x_ref'])
    y_ref = np.array(data['y_ref'])
    z_ref = np.array(data['z_ref'])
    
    x_meas = np.array(data['x_meas'])
    y_meas = np.array(data['y_meas'])
    z_meas = np.array(data['z_meas'])
    
    # TODO: Calculate tracking errors
    # Hint: Euclidean distance = sqrt(x_err^2 + y_err^2 + z_err^2)
    x_err = x_ref - x_meas
    y_err = y_ref - y_meas
    z_err = z_ref - z_meas
    euclidean_err = np.sqrt(x_err**2 + y_err**2 + z_err**2)
    
    # TODO: Calculate overall metrics
    mean_error = float(np.nanmean(euclidean_err)) if euclidean_err.size else 0.0
    max_error = float(np.nanmax(euclidean_err)) if euclidean_err.size else 0.0
    
    # TODO: Calculate per-cycle metrics
    # Hint: Loop through each cycle, calculate mean error for that cycle
    cycle_errors = []
    for c in range(int(n_cycles)):
        mask = (cycles == c)
        if np.any(mask):
            cycle_errors.append(float(np.nanmean(euclidean_err[mask])))
        else:
            cycle_errors.append(float('nan'))
    
    # TODO: Calculate error growth
    # Hint: Compare last cycle error to first cycle error
    if len(cycle_errors) >= 2 and np.isfinite(cycle_errors[0]) and cycle_errors[0] > 1e-9 and np.isfinite(cycle_errors[-1]):
        error_growth = float((cycle_errors[-1] - cycle_errors[0]) / cycle_errors[0] * 100.0)
    else:
        error_growth = 0.0
    
    # TODO: Calculate control effort
    # Hint: RMS of velocity commands
    vx = np.array(data['vx_cmd'])
    vy = np.array(data['vy_cmd'])
    vz = np.array(data['vz_cmd'])
    if vx.size:
        control_rms = float(np.sqrt(np.nanmean(vx**2 + vy**2 + vz**2)))
    else:
        control_rms = 0.0
    
    
    metrics = {
        'mean_tracking_error': mean_error,
        'max_deviation': max_error,
        'error_growth_percent': error_growth,
        'control_effort_rms': control_rms,
        'cycle_errors': cycle_errors
    }
    
    return metrics

def plot_trajectory_tracking(data, metrics, test_name):
    """Generate comprehensive trajectory tracking plots"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    x_ref = np.array(data['x_ref'])
    y_ref = np.array(data['y_ref'])
    z_ref = np.array(data['z_ref'])
    
    x_meas = np.array(data['x_meas'])
    y_meas = np.array(data['y_meas'])
    z_meas = np.array(data['z_meas'])
    
    ax1.plot(x_ref, y_ref, z_ref, 'r--', linewidth=2, label='Reference', alpha=0.8)
    ax1.plot(x_meas, y_meas, z_meas, 'b-', linewidth=1, label='Actual', alpha=0.7)
    ax1.scatter(x_ref[0], y_ref[0], z_ref[0], c='g', s=100, marker='o', label='Start')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title(f'{test_name}: 3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tracking error vs time
    ax2 = fig.add_subplot(2, 2, 2)
    
    times = np.array(data['time'])
    errors = np.sqrt((x_ref - x_meas)**2 + (y_ref - y_meas)**2 + (z_ref - z_meas)**2)
    
    ax2.plot(times, errors * 100, 'r-', linewidth=1)
    ax2.axhline(10, color='g', linestyle='--', linewidth=2, label='10cm requirement')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Tracking Error [cm]')
    ax2.set_title('Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-cycle performance
    ax3 = fig.add_subplot(2, 2, 3)
    
    if 'cycle_errors' in metrics and len(metrics['cycle_errors']) > 0:
        cycle_nums = range(1, len(metrics['cycle_errors']) + 1)
        cycle_errs = [e * 100 for e in metrics['cycle_errors']]
        
        ax3.bar(cycle_nums, cycle_errs, alpha=0.7, color='steelblue')
        ax3.axhline(10, color='r', linestyle='--', linewidth=2, label='10cm requirement')
        ax3.set_xlabel('Cycle Number')
        ax3.set_ylabel('Mean Error [cm]')
        ax3.set_title('Per-Cycle Performance')
        ax3.set_xticks(cycle_nums)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Performance metrics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"Performance Metrics:\n\n"
    summary_text += f"Mean tracking error: {metrics['mean_tracking_error']*100:.2f} cm\n"
    summary_text += f"Max deviation: {metrics['max_deviation']*100:.2f} cm\n"
    summary_text += f"Error growth: {metrics['error_growth_percent']:+.1f}%\n"
    summary_text += f"Control RMS: {metrics['control_effort_rms']:.3f} m/s\n\n"
    
    summary_text += "Requirements Check:\n"
    summary_text += f"Mean error < 10cm: {'PASS' if metrics['mean_tracking_error'] < 0.1 else 'FAIL'}\n"
    summary_text += f"Max dev < 20cm: {'PASS' if metrics['max_deviation'] < 0.2 else 'FAIL'}\n"
    summary_text += f"Error growth < 5%: {'PASS' if abs(metrics['error_growth_percent']) < 5 else 'FAIL'}\n"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{test_name}_tracking.png'), 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: {test_name}_tracking.png")

def save_test_data(data, filename):
    """Save trajectory data to CSV"""
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"  Data saved: {os.path.basename(filename)}")

def create_comparison_table(pid_metrics, pole_metrics):
    """Create and display performance comparison table"""
    
    print(f"\n{'='*60}")
    print("TRAJECTORY TRACKING COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<35} {'PID':<15} {'Pole Placement':<15}")
    print("-" * 65)
    
    print(f"{'Mean tracking error [cm]':<35} {pid_metrics['mean_tracking_error']*100:>14.2f} "
          f"{pole_metrics['mean_tracking_error']*100:>14.2f}")
    
    print(f"{'Max deviation [cm]':<35} {pid_metrics['max_deviation']*100:>14.2f} "
          f"{pole_metrics['max_deviation']*100:>14.2f}")
    
    print(f"{'Error growth [%]':<35} {pid_metrics['error_growth_percent']:>14.1f} "
          f"{pole_metrics['error_growth_percent']:>14.1f}")
    
    print(f"{'Control effort RMS [m/s]':<35} {pid_metrics['control_effort_rms']:>14.3f} "
          f"{pole_metrics['control_effort_rms']:>14.3f}")
    
    # Determine winner
    pid_wins = 0
    pole_wins = 0
    
    if pid_metrics['mean_tracking_error'] < pole_metrics['mean_tracking_error']:
        pid_wins += 1
    else:
        pole_wins += 1
    
    if pid_metrics['max_deviation'] < pole_metrics['max_deviation']:
        pid_wins += 1
    else:
        pole_wins += 1
    
    if abs(pid_metrics['error_growth_percent']) < abs(pole_metrics['error_growth_percent']):
        pid_wins += 1
    else:
        pole_wins += 1
    
    print(f"\n{'='*60}")
    if pole_wins > pid_wins:
        print("WINNER: POLE PLACEMENT CONTROLLER")
        print(f"  Superior performance in {pole_wins}/3 key metrics")
    elif pid_wins > pole_wins:
        print("WINNER: PID CONTROLLER")
        print(f"  Superior performance in {pid_wins}/3 key metrics")
    else:
        print("RESULT: TIE")
        print("  Both controllers achieved comparable performance")
    print(f"{'='*60}")

def main():
    """Main function - Phase 4 trajectory tracking comparison"""
    
    print("="*60)
    print("Lab 2 - Phase 4: Trajectory Tracking Comparison")
    print("="*60)
    
    ensure_dir(OUTPUT_DIR)
    
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    tello.streamon()
    
    detector = Detector(families='tag36h11')
    
    try:
        battery = tello.get_battery()
        print(f"Battery: {battery}%")
        
        if battery < 40:
            print("WARNING: Battery too low for trajectory tracking!")
            return
        
        print("\nTaking off...")
        tello.takeoff()
        time.sleep(3)
        
        # Test 1: PID Controller
        print("\n" + "="*60)
        print("TEST 1: PID CONTROLLER")
        print("="*60)
        
        pid_controller = MultiAxisPIDController(
            kp_x=0.5, ki_x=0.01, kd_x=0.05,
            kp_y=0.5, ki_y=0.01, kd_y=0.05,
            kp_z=0.5, ki_z=0.01, kd_z=0.05
        )
        
        pid_data, pid_metrics = track_helix_trajectory(
            tello, detector, pid_controller, n_cycles=3, test_name="pid_3cycle"
        )
        
        print("\nPID Results:")
        print(f"  Mean error: {pid_metrics['mean_tracking_error']*100:.2f} cm")
        print(f"  Max deviation: {pid_metrics['max_deviation']*100:.2f} cm")
        print(f"  Error growth: {pid_metrics['error_growth_percent']:+.1f}%")
        
        time.sleep(3)
        
        # Test 2: Pole Placement
        print("\n" + "="*60)
        print("TEST 2: POLE PLACEMENT CONTROLLER")
        print("="*60)
        
        K = np.diag([1.5, 1.5, 1.5])
        pole_controller = PolePlacementController(K, dt=SAMPLE_DT)
        
        pole_data, pole_metrics = track_helix_trajectory(
            tello, detector, pole_controller, n_cycles=3, test_name="pole_3cycle"
        )
        
        print("\nPole Placement Results:")
        print(f"  Mean error: {pole_metrics['mean_tracking_error']*100:.2f} cm")
        print(f"  Max deviation: {pole_metrics['max_deviation']*100:.2f} cm")
        print(f"  Error growth: {pole_metrics['error_growth_percent']:+.1f}%")
        
        create_comparison_table(pid_metrics, pole_metrics)
        
        comparison = {
            'pid': {k: float(v) if not isinstance(v, list) else v 
                   for k, v in pid_metrics.items()},
            'pole_placement': {k: float(v) if not isinstance(v, list) else v 
                              for k, v in pole_metrics.items()}
        }
        
        results_file = os.path.join(OUTPUT_DIR, 'trajectory_comparison.json')
        with open(results_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison saved to {results_file}")
        
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        print(f"\n{'='*60}")
        print("LAB 2 PHASE 4 COMPLETE!")
        print(f"{'='*60}")
        print(f"\nAll results saved in {OUTPUT_DIR}/")
        
    except KeyboardInterrupt:
        print("\n\nTrajectory interrupted by user")
        tello.send_rc_control(0, 0, 0, 0)
        try:
            tello.land()
        except:
            pass
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except:
            pass
    finally:
        try:
            tello.streamoff()
            tello.end()
        except:
            pass

if __name__ == "__main__":
    main()