#!/usr/bin/env python3
"""
Lab 2 - Phase 3: Pole Placement Control

Objectives:
1. Design state feedback controller using pole placement
2. Systematically meet Navy requirements through pole selection
3. Compare design methodology with trial-and-error PID tuning

Students will implement:
- Requirements to pole locations conversion
- State feedback gain calculation
- Pole placement control law
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
from scipy import signal

# Configuration
OUTPUT_DIR = "Lab2-Phase3"
SAMPLE_DT = 0.05
TAG_SIZE = 0.162
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]

# Navy Requirements
SETTLING_TIME_REQ = 4.0
OVERSHOOT_REQ = 15.0
STEADY_STATE_REQ = 0.10

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_system_matrices():
    """
    Define state-space model for 3-DOF position control.
    
    For integrator model (velocity commands directly affect positions):
        dx/dt = vx
        dy/dt = vy  
        dz/dt = vz
    
    State: x = [x, y, z]^T
    Input: u = [vx, vy, vz]^T
    
    Returns:
        A: State matrix (3x3)
        B: Input matrix (3x3)
    """
    # Pure integrators (no state coupling)
    A = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    # Each velocity directly affects its position
    B = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    return A, B

def requirements_to_poles(settling_time_req, overshoot_req):
    """
    Convert performance requirements to desired pole locations.
    
    TODO: Implement pole selection based on requirements.
    
    Design guidelines:
    - Settling time (2% criterion): t_s approximately 4/|Re(pole)|
    - For overshoot < 15%, need damping ratio zeta > 0.5
    - Real poles give no overshoot
    - Start conservative, then iterate if needed
    
    Consider:
    - What is the minimum real part needed to meet settling time?
    - Should you use real or complex poles?
    - What safety margin should you add?
    
    Args:
        settling_time_req: Required settling time [s] (4 seconds)
        overshoot_req: Maximum overshoot [%] (15 percent)
    
    Returns:
        poles: Array of 3 desired pole locations
    """
    # TODO: Calculate pole locations from requirements
    # Hint: Start with settling time requirement to find minimum |Re(pole)|
    # Hint: t_s = 4/|Re(pole)| so |Re(pole)| = 4/t_s
    
    min_real_part = 0.0  # REPLACE THIS
    
    # TODO: Choose pole locations
    # Hint: Add safety margin (e.g., multiply by 1.5)
    # Hint: Use real poles to avoid overshoot
    
    pole_location = 0.0  # REPLACE THIS
    
    # TODO: Assign poles for all three axes
    # You can use same pole for all axes, or differentiate them
    poles = np.array([0.0, 0.0, 0.0])  # REPLACE THIS
    
    return poles

def calculate_gains_from_poles(A, B, desired_poles):
    """
    Calculate state feedback gains K from desired poles.
    
    TODO: Implement gain calculation.
    
    For our integrator system with state feedback u = -K*(x - x_ref),
    the closed-loop dynamics become: dx/dt = -B*K*x
    Since B = I, the closed-loop poles are simply -K[i,i]
    
    Therefore: K[i,i] = |desired_pole[i]|
    
    Consider:
    - What is the relationship between poles and gains for this system?
    - How do you calculate K from the desired poles?
    
    Args:
        A: State matrix (3x3) 
        B: Input matrix (3x3)
        desired_poles: Desired poles (3,)
    
    Returns:
        K: State feedback gain matrix (3x3)
    """
    # TODO: Calculate gain matrix K
    # Hint: For integrator system, K = diag(|pole1|, |pole2|, |pole3|)
    
    K = np.zeros((3, 3))  # REPLACE THIS
    
    # Optional: Verify using scipy
    # fsf = signal.place_poles(A, B, desired_poles)
    # K_scipy = fsf.gain_matrix
    # print(f"Manual K:\n{K}")
    # print(f"Scipy K:\n{K_scipy}")
    
    return K

class PolePlacementController:
    """
    State feedback controller with pole placement.
    
    TODO: Implement control law u = -K*(x - x_ref)
    """
    
    def __init__(self, K, dt=0.05):
        self.K = K
        self.dt = dt
    
    def update(self, x_ref, y_ref, z_ref, x_meas, y_meas, z_meas):
        """
        Calculate control from state feedback law.
        
        TODO: Implement u = -K*(x - x_ref)
        
        Args:
            x_ref, y_ref, z_ref: Reference positions [m]
            x_meas, y_meas, z_meas: Measured positions [m]
        
        Returns:
            vx, vy, vz: Velocity commands [m/s]
        """
        # TODO: Form state vector
        x = np.array([0.0, 0.0, 0.0])  # REPLACE THIS
        
        # TODO: Form reference vector
        x_ref_vec = np.array([0.0, 0.0, 0.0])  # REPLACE THIS
        
        # TODO: Calculate error
        error = np.array([0.0, 0.0, 0.0])  # REPLACE THIS
        
        # TODO: Apply control law u = -K * error
        u = np.array([0.0, 0.0, 0.0])  # REPLACE THIS
        
        # Apply velocity limits
        max_vel = 0.3
        vx = np.clip(u[0], -max_vel, max_vel)
        vy = np.clip(u[1], -max_vel, max_vel)
        vz = np.clip(u[2], -max_vel, max_vel)
        
        return vx, vy, vz

def get_apriltag_pose(frame, detector):
    """Detect AprilTag and return pose"""
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
    rc_left_right = int(vx_tag * 100)
    rc_forward_backward = int(-vz_tag * 100)
    rc_up_down = int(vy_tag * 100)
    rc_yaw = 0
    
    rc_left_right = int(np.clip(rc_left_right, -100, 100))
    rc_forward_backward = int(np.clip(rc_forward_backward, -100, 100))
    rc_up_down = int(np.clip(rc_up_down, -100, 100))
    
    return rc_left_right, rc_forward_backward, rc_up_down, rc_yaw

def test_pole_placement_response(tello, detector, controller, test_name):
    """Test pole placement controller"""
    print(f"\nRunning test: {test_name}")
    
    ensure_dir(OUTPUT_DIR)
    
    # Test positions
    x_initial, y_initial, z_initial = 0.0, 0.0, 1.5
    x_target, y_target, z_target = 0.2, 0.2, 1.7
    
    print(f"Initial: [{x_initial:.2f}, {y_initial:.2f}, {z_initial:.2f}]")
    print(f"Target:  [{x_target:.2f}, {y_target:.2f}, {z_target:.2f}]")
    
    print("Positioning to initial location...")
    position_to_target(tello, detector, x_initial, y_initial, z_initial, timeout=20.0)
    time.sleep(2)
    
    print("Running pole placement controller...")
    
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
            
            next_sample += SAMPLE_DT
        
        time.sleep(0.01)
    
    tello.send_rc_control(0, 0, 0, 0)
    
    csv_filename = os.path.join(OUTPUT_DIR, f"{test_name}_data.csv")
    save_test_data(data, csv_filename)
    
    metrics = calculate_multi_axis_metrics(data)
    plot_response(data, metrics, test_name)
    
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
    """Calculate performance metrics (same as Phase 2)"""
    times = np.array(data['time'])
    metrics = {}
    settling_times = []
    
    for axis in ['x', 'y', 'z']:
        ref = np.array(data[f'{axis}_ref'])
        meas = np.array(data[f'{axis}_meas'])
        error = ref - meas
        
        target = ref[0]
        band = 0.05 * abs(target) if abs(target) > 0.1 else 0.05
        
        settled_idx = None
        for i in range(len(times)):
            if all(abs(error[i:]) <= band):
                settled_idx = i
                break
        
        if settled_idx is not None:
            settling_time = times[settled_idx]
            settling_times.append(settling_time)
            ss_error = np.mean(np.abs(error[settled_idx:]))
        else:
            settling_time = float('inf')
            settling_times.append(float('inf'))
            ss_error = np.mean(np.abs(error[-50:]))
        
        initial = meas[0]
        if abs(target - initial) > 0.01:
            s = np.sign(target - initial)
            overshoot_vals = s * (meas - target)
            max_overshoot = np.max(overshoot_vals.clip(min=0))
            overshoot_pct = 100 * max_overshoot / abs(target - initial)
        else:
            overshoot_pct = 0.0
        
        metrics[f'{axis}_settling_time'] = settling_time
        metrics[f'{axis}_steady_state_error'] = ss_error
        metrics[f'{axis}_overshoot'] = overshoot_pct
    
    if all(s != float('inf') for s in settling_times):
        metrics['multi_axis_settling'] = max(settling_times)
        metrics['coordination_spread'] = max(settling_times) - min(settling_times)
    else:
        metrics['multi_axis_settling'] = float('inf')
        metrics['coordination_spread'] = float('inf')
    
    return metrics

def plot_response(data, metrics, test_name):
    """Generate response plots"""
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
        ax.set_title(f'{axis.upper()}-axis: Pole Placement Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        ax.plot(times, vel, 'g-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        ax.set_ylabel(f'v_{axis} [m/s]')
        ax.set_title(f'{axis.upper()}-axis Control Effort')
        ax.grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    
    plt.tight_layout()
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
    print("Lab 2 - Phase 3: Pole Placement Control")
    print("="*60)
    
    # Step 1: Define system model
    print("\nStep 1: System Model")
    A, B = get_system_matrices()
    print(f"A matrix:\n{A}")
    print(f"B matrix:\n{B}")
    
    # Step 2: Design poles from requirements
    print("\nStep 2: Pole Selection")
    desired_poles = requirements_to_poles(SETTLING_TIME_REQ, OVERSHOOT_REQ)
    print(f"Desired poles: {desired_poles}")
    
    # Step 3: Calculate gains
    print("\nStep 3: Gain Calculation")
    K = calculate_gains_from_poles(A, B, desired_poles)
    print(f"Gain matrix K:\n{K}")
    
    # Verify closed-loop poles
    A_cl = A - B @ K
    poles_cl = np.linalg.eigvals(A_cl)
    print(f"Closed-loop poles (verification): {poles_cl}")
    
    # Initialize drone
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
        
        # Create controller
        controller = PolePlacementController(K, dt=SAMPLE_DT)
        
        print("\nTesting Pole Placement Controller")
        
        data, metrics = test_pole_placement_response(
            tello, detector, controller, "pole_placement"
        )
        
        print("\nPole Placement Results:")
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
        
        # Save results
        results = {
            'design': {
                'desired_poles': desired_poles.tolist(),
                'gain_matrix': K.tolist(),
                'closed_loop_poles': poles_cl.tolist()
            },
            'metrics': {k: float(v) if v != float('inf') else 'inf' 
                       for k, v in metrics.items()}
        }
        
        results_file = os.path.join(OUTPUT_DIR, 'pole_placement_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        print(f"\nPhase 3 complete! Files saved in {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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