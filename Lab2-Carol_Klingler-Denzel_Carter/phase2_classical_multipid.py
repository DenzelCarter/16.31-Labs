#!/usr/bin/env python3
"""
Lab 2 - Phase 2: Multi-Axis PID Control

Objectives:
1. Extend Lab 1 PID controller to three independent axes
2. Tune PID gains for multi-axis coordination
3. Evaluate performance against Navy requirements

Implement:
- Multi-axis PID controller using three Lab 1 PID instances
- Multi-axis performance metrics calculation
- Requirement validation
"""

import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello
import cv2
from pupil_apriltags import Detector

# Configuration
OUTPUT_DIR = "Lab2-Phase2"
SAMPLE_DT = 0.05  # 20 Hz
TAG_SIZE = 0.162
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]

# Navy Requirements
SETTLING_TIME_REQ = 4.0
OVERSHOOT_REQ = 15.0
STEADY_STATE_REQ = 0.10

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class PIDController:
    """
    Single-axis PID controller from Lab 1.
    
    This is the same PID implementation from Lab 1 Phase 4.
    """
    
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True
    
    def update(self, setpoint, measured_value):
        """
        Calculate PID control output.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
        
        Returns:
            output: Control output
            p_term: Proportional term (for debugging)
            i_term: Integral term (for debugging)
            d_term: Derivative term (for debugging)
        """
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
    """
    Multi-axis PID controller using three independent single-axis PIDs.
    
    TODO: Implement this class to control x, y, and z axes independently.
    """
    
    def __init__(self, kp_x=0.5, ki_x=0.01, kd_x=0.05,
                       kp_y=0.5, ki_y=0.01, kd_y=0.05,
                       kp_z=0.5, ki_z=0.01, kd_z=0.05,
                       dt=0.05):
        """
        Initialize three independent PID controllers.
        
        TODO: Create three PIDController instances (one for each axis).
        
        Args:
            kp_x, ki_x, kd_x: X-axis PID gains
            kp_y, ki_y, kd_y: Y-axis PID gains
            kp_z, ki_z, kd_z: Z-axis PID gains
            dt: Sampling time [s]
        """
        # TODO: Create three PID controller instances
        # Hint: Use the PIDController class from Lab 1
        self.pid_x = PIDController(kp=kp_x, ki=ki_x, kd=kd_x, dt=dt)
        self.pid_y = PIDController(kp=kp_y, ki=ki_y, kd=kd_y, dt=dt)
        self.pid_z = PIDController(kp=kp_z, ki=ki_z, kd=kd_z, dt=dt)
        self.dt = dt
        self.max_vel = 0.15  # m/s safety limit

    def update(self, x_ref, y_ref, z_ref, x_meas, y_meas, z_meas):
        """
        Calculate control velocities for all three axes.
        
        TODO: Call each PID controller and return velocity commands.
        
        Args:
            x_ref, y_ref, z_ref: Reference positions [m]
            x_meas, y_meas, z_meas: Measured positions [m]
        
        Returns:
            vx, vy, vz: Velocity commands [m/s]
        """
        # TODO: Update each axis controller
        # Hint: Call self.pid_x.update(x_ref, x_meas) and similar for y, z
        # Use Lab 1 PIDController.update(setpoint, measurement)
        vx, _, _, _ = self.pid_x.update(x_ref, x_meas)
        vy, _, _, _ = self.pid_y.update(y_ref, y_meas)
        vz, _, _, _ = self.pid_z.update(z_ref, z_meas)
        
        # Apply velocity limits for safety
        # Safety clamp
        vx = float(np.clip(vx, -self.max_vel, self.max_vel))
        vy = float(np.clip(vy, -self.max_vel, self.max_vel))
        vz = float(np.clip(vz, -self.max_vel, self.max_vel))
        return vx, vy, vz
        
    
    def reset(self):
        """Reset all controller states"""
        # TODO: Reset each PID controller        
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

def get_apriltag_pose(frame, detector):
    """
    Detect AprilTag and return pose in tag coordinate frame.
    
    Returns:
        dict with 'detected' (bool) and 'x', 'y', 'z' (float or nan)
    """
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
    """
    Transform velocity commands from tag frame to drone RC commands.
    
    Tag frame velocities:
    - vx_tag: lateral velocity (+ = right)
    - vy_tag: vertical velocity (+ = up)
    - vz_tag: distance velocity (+ = away from tag)
    
    Drone RC commands:
    - left_right_velocity: -100 to 100 (+ = right)
    - forward_backward_velocity: -100 to 100 (+ = forward)
    - up_down_velocity: -100 to 100 (+ = up)
    - yaw_velocity: -100 to 100 (+ = clockwise)
    """
    rc_left_right =  int(-vx_tag * 100)
    rc_forward_backward = int(-vz_tag * 100)
    rc_up_down = int(vy_tag * 100)
    rc_yaw = 0
    
    rc_left_right = int(np.clip(rc_left_right, -100, 100))
    rc_forward_backward = int(np.clip(rc_forward_backward, -100, 100))
    rc_up_down = int(np.clip(rc_up_down, -100, 100))
    
    return rc_left_right, rc_forward_backward, rc_up_down, rc_yaw

def test_step_response(tello, detector, controller, test_name):
    """
    Test multi-axis step response.
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector
        controller: MultiAxisPIDController instance
        test_name: Name for saving results
    
    Returns:
        data: Dictionary of test data
        metrics: Dictionary of performance metrics
    """
    print(f"\nRunning test: {test_name}")
    
    ensure_dir(OUTPUT_DIR)
    
    # Test positions
    x_initial, y_initial, z_initial = 0.0, 0.0, 1.2
    x_target, y_target, z_target = 0.15, 0.15, 1.4
    
    print(f"Initial: [{x_initial:.2f}, {y_initial:.2f}, {z_initial:.2f}]")
    print(f"Target:  [{x_target:.2f}, {y_target:.2f}, {z_target:.2f}]")
    
    print("Positioning to initial location...")
    position_to_target(tello, detector, x_initial, y_initial, z_initial, timeout=15.0)
    time.sleep(2)
    
    controller.reset()
    
    print("Running step response...")
    
    test_duration = 15.0
    start_time = time.time()
    next_sample = start_time
    
    data = {
        'time': [],
        'x_ref': [], 'y_ref': [], 'z_ref': [],
        'x_meas': [], 'y_meas': [], 'z_meas': [],
        'vx_cmd': [], 'vy_cmd': [], 'vz_cmd': []
    }
    
    while time.time() - start_time < test_duration:
        if time.time() >= next_sample:
            frame = tello.get_frame_read().frame
            pose = get_apriltag_pose(frame, detector)
            
            if pose['detected']:
                vx, vy, vz = controller.update(
                    x_target, y_target, z_target,
                    pose['x'], pose['y'], pose['z']
                )
                
                rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx, vy, vz)
                tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)
                
                t = time.time() - start_time
                data['time'].append(t)
                data['x_ref'].append(x_target)
                data['y_ref'].append(y_target)
                data['z_ref'].append(z_target)
                data['x_meas'].append(pose['x'])
                data['y_meas'].append(pose['y'])
                data['z_meas'].append(pose['z'])
                data['vx_cmd'].append(vx)
                data['vy_cmd'].append(vy)
                data['vz_cmd'].append(vz)
                
                if len(data['time']) % 20 == 0:
                    print(f"  t={t:.1f}s: pos=[{pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}]")
                
            else:
                tello.send_rc_control(0, 0, 0, 0)
            next_sample += SAMPLE_DT
        
        time.sleep(0.01)
    
    tello.send_rc_control(0, 0, 0, 0)
    
    csv_filename = os.path.join(OUTPUT_DIR, f"{test_name}_data.csv")
    save_test_data(data, csv_filename)
    
    metrics = calculate_multi_axis_metrics(data)
    plot_multi_axis_response(data, metrics, test_name)
    
    return data, metrics

def position_to_target(tello, detector, x_target, y_target, z_target, timeout=15.0):
    """Simple positioning using proportional control"""
    start_time = time.time()
    settled_count = 0
    
    while time.time() - start_time < timeout:
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector)
        
        if pose['detected']:
            x_error = x_target - pose['x']
            y_error = y_target - pose['y']
            z_error = z_target - pose['z']
            
            if abs(x_error) < 0.1 and abs(y_error) < 0.1 and abs(z_error) < 0.1:
                settled_count += 1
                if settled_count > 20:
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
def calculate_multi_axis_metrics(data):
    """
    Calculate performance metrics for multi-axis step response.

    Metrics per axis:
      - Settling time (±5% of step size; first time after which it stays inside)
      - Overshoot (% of step size)
      - Steady-state error (mean |error| after settling; or last 2 s if never settles)

    Multi-axis:
      - Multi-axis settling time = max of per-axis settling times
      - Coordination spread = max(ts) - min(ts) among axes that settled
    """
    times = np.asarray(data['time'], dtype=float)
    T_end = float(times[-1]) if times.size else np.nan

    def settling_time(t, y, target, band):
        """First t[i] such that |y - target| <= band for all j >= i."""
        if t.size == 0 or y.size == 0:
            return np.nan
        err = np.abs(y - target)
        inside = err <= band
        # find earliest index after which it's always inside
        for i in range(len(inside)):
            if np.all(inside[i:]):
                return float(t[i])
        return np.nan

    def steady_state_error(t, err, ts):
        """Mean |error| after settling; if no settle, use last 2 s (or last 10 samples)."""
        err = np.asarray(err, dtype=float)
        valid = np.isfinite(err)
        if not np.any(valid):
            return np.nan
        if np.isfinite(ts):
            mask = t >= ts
        else:
            if t.size == 0:
                return np.nan
            t_tail = max(0.0, t[-1] - 2.0)
            mask = t >= t_tail
            # if super short run, fallback to last 10 samples
            if not np.any(mask):
                mask = np.zeros_like(t, dtype=bool)
                mask[-min(10, len(t)):] = True
        sel = valid & mask
        return float(np.nanmean(np.abs(err[sel]))) if np.any(sel) else np.nan

    def overshoot_pct(y, target, step):
        """Max deviation beyond target as % of |step| (0 if no overshoot)."""
        if not np.isfinite(step) or abs(step) < 1e-9 or y.size == 0:
            return 0.0
        if step >= 0:
            peak = np.nanmax(y)
            os = max(0.0, peak - target)
        else:
            trough = np.nanmin(y)
            os = max(0.0, target - trough)
        return float((os / abs(step)) * 100.0)

    metrics = {}
    per_axis_ts = []

    for axis in ['x', 'y', 'z']:
        ref = np.asarray(data[f'{axis}_ref'], dtype=float)
        meas = np.asarray(data[f'{axis}_meas'], dtype=float)

        # Assume step to a constant target; use final reference value as target.
        target = float(ref[-1]) if ref.size else np.nan
        y0 = float(meas[0]) if meas.size else np.nan
        step = target - y0 if np.isfinite(target) and np.isfinite(y0) else np.nan

        # 5% band of step magnitude; fallback absolute band = 0.05 m if step is tiny/undefined
        band = 0.05 * abs(step) if np.isfinite(step) and abs(step) > 1e-6 else 0.05
       # band = max(0.05 * abs(step), 0.04)   # never tighter than 4 cm

        # Settling time
        ts_axis = settling_time(times, meas, target, band)
        per_axis_ts.append(ts_axis)

        # Steady-state error (mean |error| after ts or last 2 s)
        err = (ref - meas) if (ref.size == meas.size and ref.size > 0) else (target - meas)
        ss_err = steady_state_error(times, err, ts_axis)

        # Overshoot (%)
        os_pct = overshoot_pct(meas, target, step)

        metrics[f'{axis}_settling_time'] = ts_axis if np.isfinite(ts_axis) else float('inf')
        metrics[f'{axis}_steady_state_error'] = ss_err
        metrics[f'{axis}_overshoot'] = os_pct

    # Multi-axis metrics
    finite_ts = [ts for ts in per_axis_ts if np.isfinite(ts)]
    if finite_ts:
        multi_settle = max(finite_ts)
        coord_spread = max(finite_ts) - min(finite_ts) if len(finite_ts) > 1 else 0.0
    else:
        multi_settle = float('inf')
        coord_spread = float('nan')

    metrics['multi_axis_settling'] = multi_settle
    metrics['coordination_spread'] = coord_spread

    return metrics


def plot_multi_axis_response(data, metrics, test_name):
    """Generate plots for multi-axis response"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    times = np.array(data['time'])
    
    for i, axis in enumerate(['x', 'y', 'z']):
        ref = np.array(data[f'{axis}_ref'])
        meas = np.array(data[f'{axis}_meas'])
        vel = np.array(data[f'v{axis}_cmd'])
        
        ax = axes[i, 0]
        ax.plot(times, meas, 'b-', linewidth=2, label=f'{axis} measured')
        ax.plot(times, ref, 'r--', linewidth=2, label=f'{axis} reference')
        ax.set_ylabel(f'{axis.upper()} Position [m]')
        ax.set_title(f'{axis.upper()}-axis Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        ax.plot(times, vel, 'g-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.set_ylabel(f'v_{axis} [m/s]')
        ax.set_title(f'{axis.upper()}-axis Control')
        ax.grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    
    metrics_text = f"Multi-axis settling: {metrics['multi_axis_settling']:.2f}s\n"
    metrics_text += f"Coordination spread: {metrics['coordination_spread']:.2f}s\n"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{test_name}_response.png'), 
                dpi=200, bbox_inches='tight')
    plt.close()

def save_test_data(data, filename):
    """Save test data to CSV"""
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def main():
    """Main function"""
    print("Lab 2 - Phase 2: Multi-Axis PID Control")
    print("="*60)
    
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    tello.streamon()
    
    detector = Detector(families='tag36h11')
    
    try:
        battery = tello.get_battery()
        print(f"Battery: {battery}%")
        
        if battery < 30:
            print("Battery too low!")
            return
        
        print("Taking off...")
        tello.takeoff()
        time.sleep(3)
        
        print("\nTesting Multi-Axis PID Controller")
        
        controller = MultiAxisPIDController(
            kp_x=0.5, ki_x=0.01, kd_x=0.05,
            kp_y=0.5, ki_y=0.01, kd_y=0.05,
            kp_z=0.5, ki_z=0.01, kd_z=0.05
        )
        
        data, metrics = test_step_response(tello, detector, controller, "test_initial")
        
        print("\nTest Results:")
        print(f"  X-axis: settling={metrics['x_settling_time']:.2f}s, "
              f"overshoot={metrics['x_overshoot']:.1f}%, "
              f"ss_error={metrics['x_steady_state_error']:.3f}m")
        print(f"  Y-axis: settling={metrics['y_settling_time']:.2f}s, "
              f"overshoot={metrics['y_overshoot']:.1f}%, "
              f"ss_error={metrics['y_steady_state_error']:.3f}m")
        print(f"  Z-axis: settling={metrics['z_settling_time']:.2f}s, "
              f"overshoot={metrics['z_overshoot']:.1f}%, "
              f"ss_error={metrics['z_steady_state_error']:.3f}m")
        print(f"  Multi-axis settling: {metrics['multi_axis_settling']:.2f}s")
        print(f"  Coordination spread: {metrics['coordination_spread']:.2f}s")
        
        print("\nNavy Requirements Check:")
        settling_ok = metrics['multi_axis_settling'] <= SETTLING_TIME_REQ
        overshoot_ok = all(metrics[f'{ax}_overshoot'] < OVERSHOOT_REQ for ax in ['x','y','z'])
        ss_error_ok = all(metrics[f'{ax}_steady_state_error'] < STEADY_STATE_REQ for ax in ['x','y','z'])
        
        print(f"  Settling time <= {SETTLING_TIME_REQ}s: {'PASS' if settling_ok else 'FAIL'}")
        print(f"  Overshoot < {OVERSHOOT_REQ}%: {'PASS' if overshoot_ok else 'FAIL'}")
        print(f"  SS error < {STEADY_STATE_REQ}m: {'PASS' if ss_error_ok else 'FAIL'}")
        
        results = {
            'test_initial': {
                'gains': {'kp_x': 0.5, 'ki_x': 0.01, 'kd_x': 0.05,
                         'kp_y': 0.5, 'ki_y': 0.01, 'kd_y': 0.05,
                         'kp_z': 0.5, 'ki_z': 0.01, 'kd_z': 0.05},
                'metrics': {k: float(v) for k, v in metrics.items()}
            }
        }
        
        results_file = os.path.join(OUTPUT_DIR, 'multi_pid_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        print(f"\nPhase 2 complete! Files saved in {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"Error: {e}")
        try:
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