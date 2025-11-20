#!/usr/bin/env python3
"""
Lab 3 - Phase 1: Baseline Landing System

Objectives:
1. Implement simple P-control landing using direct ToF sensor feedback
2. Demonstrate vulnerability to sensor attacks
3. Provide baseline performance comparison

Mission flow:
1. Takeoff from ~2m away
2. Detect AprilTag and approach
3. Position above landing target (1.5m forward of tag)
4. Descend straight down using ONLY ToF feedback (no horizontal control)
5. Land

Landing target: 1.5m forward of AprilTag on ground

KEY CONCEPT:
- AprilTag: Used for horizontal (x,y) positioning in Phase 1 only
- ToF sensor: Used for altitude (z) control during Phase 3 descent
- During descent, NO horizontal control - just straight down
- Demonstrates vulnerability of direct ToF feedback to sensor attacks
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello
from pupil_apriltags import Detector

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.apriltag_utils import get_apriltag_pose, tag_to_drone_velocity, position_to_target

# Configuration
OUTPUT_DIR = "Lab3-Phase1"
LANDING_TARGET = (1.5, 0.0)  # (x, y) in tag frame, on ground

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def baseline_landing_mission(tello, detector, tag_size=0.125):
    """
    Execute baseline landing mission.
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector
        tag_size: Physical tag size in meters (must match actual tag!)
    
    Returns:
        metrics: Dictionary of landing performance metrics
    """
    print("\n" + "="*60)
    print("BASELINE LANDING MISSION")
    print(f"Using tag_size = {tag_size}m")
    print("="*60)
    
    data = {
        'time': [],
        'z_tof': [],
        'vz_cmd': [],
        'x_tag': [],
        'y_tag': [],
        'tag_visible': []
    }
    
    # Phase 1: Approach landing target
    print("\n[Phase 1] Positioning above landing target...")
    
    # Get initial tag pose to determine position
    frame = tello.get_frame_read().frame
    pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
    
    if not pose['detected']:
        print("ERROR: AprilTag not detected!")
        return None
    
    z_tag_height = pose['z']  # Tag height above ground
    print(f"  AprilTag detected at height: {z_tag_height:.2f}m")
    
    # Store initial position as last known position
    last_known_x = pose['x']
    last_known_y = pose['y']
    
    # Position above landing target at tag height using AprilTag
    target_x, target_y = LANDING_TARGET
    success = position_to_target(tello, detector, target_x, target_y, z_tag_height, 
                      timeout=30.0,
                      position_threshold=0.15,
                      velocity_threshold=0.10,
                      tag_size=tag_size)
    
    # Capture final position after Phase 1
    frame = tello.get_frame_read().frame
    final_pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
    
    if final_pose['detected']:
        phase1_final_x = final_pose['x']
        phase1_final_y = final_pose['y']
        phase1_error = np.sqrt((phase1_final_x - target_x)**2 + (phase1_final_y - target_y)**2)
        print(f"  Final Phase 1 position: ({phase1_final_x:.2f}, {phase1_final_y:.2f})m")
        print(f"  Position error: {phase1_error*100:.1f} cm")
    else:
        print("  WARNING: Lost tag detection after positioning")
        phase1_final_x, phase1_final_y = last_known_x, last_known_y
        phase1_error = np.sqrt((phase1_final_x - target_x)**2 + (phase1_final_y - target_y)**2)
        print(f"  Using last known position: ({phase1_final_x:.2f}, {phase1_final_y:.2f})m")
        print(f"  Position error (from last known): {phase1_error*100:.1f} cm")
    
    if not success:
        print("  WARNING: Failed to position perfectly, but continuing with descent...")
        print("  (This is Phase 1 baseline - imperfect positioning is acceptable)")
    
    print("\n[Phase 2] Stabilizing...")
    time.sleep(2.0)
    
    # Phase 3: Descend using ToF feedback
    print("\n[Phase 3] Descending to land...")
    print("  Using DIRECT ToF feedback for altitude (no filtering)")
    print("  NO horizontal control - descending straight down")
    
    # ============================================================
    # TODO: Implement P-control for descent
    # ============================================================
    # Control parameters - you may need to tune these
    kp_z = 0.5  # Proportional gain for altitude (start with this value)
    z_target = 0.1  # Target altitude to trigger land command [m]
    
    start_time = time.time()
    
    while True:
        t = time.time() - start_time
        
        # ALTITUDE CONTROL: Use ToF sensor (this is what we care about!)
        z_tof = tello.get_distance_tof() / 100.0  # Convert cm to meters
        
        # ============================================================
        # TODO: Implement simple P-control for descent
        # ============================================================
        # 1. Calculate altitude error: z_error = z_target - z_tof
        # 2. Compute velocity command: vz_cmd = kp_z * z_error
        # 3. Clip velocity to safe range: [-0.3, 0.3] m/s
        #
        # This is VULNERABLE to sensor attacks because it blindly
        # trusts z_tof without any validation or filtering!
        
        # YOUR CODE HERE
        z_error = 0.0  # REPLACE THIS
        vz_cmd = 0.0   # REPLACE THIS
        
        
        # ============================================================
        
        # NO HORIZONTAL CONTROL - just descend straight down
        vx_cmd, vy_cmd = 0.0, 0.0
        
        # Send commands (only vertical movement)
        rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx_cmd, vy_cmd, vz_cmd)
        tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)
        
        # Log data
        data['time'].append(t)
        data['z_tof'].append(z_tof)
        data['vz_cmd'].append(vz_cmd)
        data['x_tag'].append(np.nan)
        data['y_tag'].append(np.nan)
        data['tag_visible'].append(False)
        
        # Print progress
        if len(data['time']) % 10 == 0:
            print(f"  t={t:.1f}s: z_ToF={z_tof:.2f}m, vz={vz_cmd:.2f}m/s")
        
        # Land when close to ground (based on ToF, NOT AprilTag)
        if z_tof < z_target:
            print(f"\n  [+] Altitude {z_tof:.2f}m < {z_target:.2f}m, landing!")
            break
        
        # Safety timeout
        if t > 20.0:
            print("\n  WARNING: Timeout reached, landing")
            break
        
        time.sleep(0.1)
    
    # Stop and land
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    
    print("\n[Phase 4] Landing...")
    tello.land()
    time.sleep(2.0)
    
    # Calculate final metrics
    final_x = phase1_final_x
    final_y = phase1_final_y
    landing_error = phase1_error
    touchdown_vz = abs(data['vz_cmd'][-1])
    
    metrics = {
        'landing_error': landing_error,
        'touchdown_vz': touchdown_vz,
        'time_to_land': data['time'][-1],
        'final_position': (final_x, final_y)
    }
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Landing error: {landing_error*100:.2f} cm (from Phase 1 positioning)")
    print(f"Touchdown velocity: {touchdown_vz:.3f} m/s")
    print(f"Time to land: {metrics['time_to_land']:.1f} s")
    print(f"Final position: ({final_x:.2f}, {final_y:.2f}) m")
    print(f"Target position: ({target_x:.2f}, {target_y:.2f}) m")
    
    # Save data
    ensure_dir(OUTPUT_DIR)
    
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_data.csv'), index=False)
    
    with open(os.path.join(OUTPUT_DIR, 'baseline_metrics.txt'), 'w') as f:
        f.write("BASELINE LANDING METRICS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Tag size used: {tag_size}m\n")
        f.write(f"Landing error: {landing_error*100:.2f} cm\n")
        f.write(f"Touchdown velocity: {touchdown_vz:.3f} m/s\n")
        f.write(f"Time to land: {metrics['time_to_land']:.1f} s\n")
        f.write(f"Final position: ({final_x:.2f}, {final_y:.2f}) m\n")
        f.write(f"\nNote: Phase 3 descent used ONLY altitude control.\n")
        f.write(f"No horizontal positioning was attempted during descent.\n")
    
    # Plot descent
    plot_baseline_descent(data)
    
    return metrics


def plot_baseline_descent(data):
    """Generate descent trajectory plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    times = np.array(data['time'])
    z_tof = np.array(data['z_tof'])
    vz_cmd = np.array(data['vz_cmd'])
    
    # Plot 1: Altitude vs time
    ax1.plot(times, z_tof, 'b-', linewidth=2, label='ToF altitude')
    ax1.axhline(0.1, color='r', linestyle='--', label='Landing threshold')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Baseline Landing - Altitude Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity command
    ax2.plot(times, vz_cmd, 'g-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Vertical Velocity [m/s]')
    ax2.set_title('Descent Control Command')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_descent.png'), dpi=200)
    print(f"\n[+] Plot saved: {OUTPUT_DIR}/baseline_descent.png")
    plt.close()


def write_vulnerability_analysis():
    """
    Write analysis of why baseline fails under attack.
    
    Creates baseline_analysis.txt deliverable.
    """
    # ============================================================
    # TODO: Complete vulnerability analysis
    # ============================================================
    # Write a 2-3 sentence explanation of what happens when an
    # attacker injects a +0.5m bias into the ToF sensor.
    #
    # Include a mathematical example:
    # - True altitude: z_true = 0.6 m
    # - Sensor reports: z_ToF = 1.1 m (due to +0.5m bias)
    # - Controller computes: v_z = kp * (z_target - z_ToF) = ?
    # - Result: What happens to the drone?
    
    analysis_text = """
BASELINE CONTROLLER VULNERABILITY ANALYSIS

The baseline controller uses a simple proportional control law:
    v_z = K_p * (z_target - z_ToF)

where z_ToF is the direct reading from the Time-of-Flight sensor.

ATTACK SCENARIO:
When an adversary injects a +0.5m bias into the ToF sensor:
- True altitude: z_true
- Compromised reading: z_ToF = z_true + 0.5m

TODO: Complete this analysis
- Explain what happens when the controller uses the biased reading
- Provide mathematical example with numbers
- Explain why this causes a crash

YOUR ANALYSIS HERE:


"""
    
    ensure_dir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'baseline_analysis.txt'), 'w') as f:
        f.write(analysis_text)
    
    print(f"[+] Analysis template saved: {OUTPUT_DIR}/baseline_analysis.txt")
    print("    TODO: Complete the vulnerability analysis section!")


def main():
    """Main function for Phase 1"""
    print("\n" + "="*60)
    print("LAB 3 - PHASE 1: BASELINE LANDING SYSTEM")
    print("="*60)
    
    # Initialize drone
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    
    if battery < 30:
        print("ERROR: Battery too low (need >30%)")
        return
    
    print("Starting video stream...")
    tello.streamon()
    time.sleep(2)
    
    # Use robust detector parameters
    detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    # CRITICAL: Set this to your actual physical tag size!
    # Measure your tag: 8cm = 0.08m, 12.5cm = 0.125m, 17cm = 0.17m
    TAG_SIZE = 0.125  # meters (12.5cm custom tag)
    
    try:
        input("\nPress ENTER to takeoff and begin baseline landing test...")
        
        print("\nTaking off...")
        tello.takeoff()
        
        print("Stabilizing after takeoff...")
        time.sleep(3)
        
        # Verify AprilTag detection before proceeding
        print("Verifying AprilTag detection...")
        tag_detected = False
        for attempt in range(10):
            frame = tello.get_frame_read().frame
            pose = get_apriltag_pose(frame, detector, tag_size=TAG_SIZE)
            
            if pose['detected']:
                print(f"  [+] Tag detected at ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})m")
                tag_detected = True
                break
            else:
                print(f"  Attempt {attempt+1}/10: No tag detected, retrying...")
                time.sleep(0.5)
        
        if not tag_detected:
            print("\n[-] ERROR: Cannot detect AprilTag after multiple attempts")
            print("  Possible issues:")
            print("  - Tag too far away (should be ~1-2m)")
            print("  - Tag not in camera view")
            print(f"  - Wrong tag size in code (currently set to {TAG_SIZE}m)")
            print("  - Lighting conditions")
            tello.land()
            return
        
        # Execute mission
        metrics = baseline_landing_mission(tello, detector, tag_size=TAG_SIZE)
        
        if metrics:
            print("\n[+] Mission complete!")
            write_vulnerability_analysis()
            
            print(f"\nAll files saved in: {OUTPUT_DIR}/")
            print("  - baseline_data.csv")
            print("  - baseline_metrics.txt")
            print("  - baseline_descent.png")
            print("  - baseline_analysis.txt (TODO: Complete this!)")
        else:
            print("\n[-] Mission failed")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        tello.send_rc_control(0, 0, 0, 0)
        try:
            tello.land()
        except:
            pass
    
    except Exception as e:
        print(f"\nERROR: {e}")
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