#!/usr/bin/env python3
"""
Lab 3 - Phase 3: Blind Navigation Mission

Mission: Navigate to high-altitude target while emulating sensor attack
when AprilTag is lost.

Workflow:
1. Takeoff and detect AprilTag
2. Navigate toward target (1.5, 0, 2.5) using AprilTag
3. When tag lost -> BLIND NAVIGATION:
   - Remember last known position
   - Continue to target using:
     * Horizontal (x,y): Dead reckoning
     * Vertical (z): Kalman filter prediction-only
   - ToF sensor compromised (+0.5m bias injected)
4. Reach target (1.5, 0, 2.5) -> land

Deliverables:
- landing_attack.py
- attack_mission.csv
- mission_analysis.png
- mission_report.txt
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello
from pupil_apriltags import Detector

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.apriltag_utils import get_apriltag_pose
from utils.kalman_filter import AltitudeKalmanFilter

OUTPUT_DIR = "Lab3-Phase3"

# Mission parameters
TARGET_POSITION = (1.5, 0.0, 2.5)  # (x, y, z) in tag frame

# ============================================================
# Control parameters (you can still tweak these)
# ============================================================
KP_XY = 0.3   # Horizontal control gain
KP_Z = 0.3    # Vertical control gain
XY_TOLERANCE = 0.2   # Horizontal position tolerance [m]
Z_TOLERANCE = 0.05    # Vertical position tolerance [m]
# ============================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def attack_resilience_mission(tello, detector, tag_size=0.117):
    """
    Execute complete landing mission with blind navigation capability
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector
        tag_size: Physical tag size [m]
    
    Returns:
        data: Mission telemetry dictionary
    """
    print(f"\n{'='*60}")
    print(f"ATTACK RESILIENCE MISSION")
    print(f"{'='*60}")
    print(f"Target: ({TARGET_POSITION[0]}, {TARGET_POSITION[1]}, {TARGET_POSITION[2]}) m")
    
    data = {
        'time': [],
        'phase': [],
        'z_tof_real': [],
        'z_tof_reported': [],
        'z_est': [],
        'vz_est': [],
        'vz_cmd': [],
        'x_tag': [],
        'y_tag': [],
        'x_est': [],
        'y_est': [],
        'tag_visible': [],
        'kf_mode': []
    }
    
    # Initialize Kalman filter
    kf = AltitudeKalmanFilter(dt=0.1)
    
    target_x, target_y, target_z = TARGET_POSITION
    
    # PHASE 1: Navigate toward target with AprilTag guidance
    print(f"\n[PHASE 1] Navigating toward target...")
    print(f"  Using AprilTag for guidance")
    
    # Verify initial tag detection
    print(f"  Verifying AprilTag detection...")
    tag_detected = False
    pose = None
    
    for attempt in range(10):
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
        
        if pose['detected']:
            print(f"  [+] Tag detected at ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}) m")
            tag_detected = True
            break
        else:
            print(f"  Attempt {attempt+1}/10: No tag detected, retrying...")
            time.sleep(0.5)
    
    if not tag_detected:
        print("\n[-] ERROR: Cannot detect AprilTag after multiple attempts")
        return None
    
    # Initialize KF with current altitude
    z_initial = tello.get_distance_tof() / 100.0
    kf.initialize(z_initial, 0.0)
    
    # Mission state
    mission_start = time.time()
    attack_triggered = False
    attack_start_time = None
    
    # Position tracking
    x_current = pose['x']
    y_current = pose['y']
    z_current = pose['z']
    
    x_last_known = x_current
    y_last_known = y_current
    z_last_known = z_current
    
    x_est = x_current
    y_est = y_current
    
    consecutive_lost = 0
    reached_horizontal_target = False
    reached_vertical_target = False
    
    # For KF input (approx vertical acceleration from vz_cmd history)
    prev_vz_cmd = 0.0
    prev_prev_vz_cmd = 0.0
    dt = 0.1
    
    print(f"  Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) m")
    print(f"  Distance: {np.sqrt((target_x-x_current)**2 + (target_y-y_current)**2 + (target_z-z_current)**2):.2f} m")
    
    # Main mission loop
    while True:
        t = time.time() - mission_start
        
        # Approximate vertical acceleration from previous vertical commands
        a_cmd = (prev_vz_cmd - prev_prev_vz_cmd) / dt
        
        # Get real altitude
        z_tof_real = tello.get_distance_tof() / 100.0
        
        # Try to detect AprilTag
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
        
        # ============================================================
        # Tag loss detection and mode switching
        # ============================================================
        if pose['detected'] and not attack_triggered:
            # Normal operation - tag visible
            consecutive_lost = 0
            tag_visible = True
            phase = 'approach'
            kf_mode = 'normal'
            
            # Update position from tag
            x_current = pose['x']
            y_current = pose['y']
            z_current = pose['z']
            
            x_last_known = x_current
            y_last_known = y_current
            z_last_known = z_current
            
            x_est = x_current
            y_est = y_current
            
            # Normal KF operation
            z_tof_reported = z_tof_real
            kf.predict(u=a_cmd)
            kf.update(z_tof_real)
        
        else:
            # Tag lost or attack active
            consecutive_lost += 1
            tag_visible = False
            
            # Trigger attack after 3 consecutive lost frames
            if consecutive_lost >= 3 and not attack_triggered:
                attack_triggered = True
                attack_start_time = t
                
                dx_remaining = target_x - x_last_known
                dy_remaining = target_y - y_last_known
                dz_remaining = target_z - z_last_known
                dist_remaining = np.sqrt(dx_remaining**2 + dy_remaining**2 + dz_remaining**2)
                
                print(f"\n  +---------------------------------------------------------+")
                print(f"  | TAG LOST - ATTACK INITIATED                             |")
                print(f"  | Time: {t:.1f}s                                          |")
                print(f"  | Last position: ({x_last_known:.2f}, {y_last_known:.2f}, {z_last_known:.2f}) m              |")
                print(f"  | Distance to target: {dist_remaining:.2f} m                             |")
                print(f"  |                                                         |")
                print(f"  | BLIND NAVIGATION MODE:                                  |")
                print(f"  | - ToF sensor: +0.5m bias injected                      |")
                print(f"  | - AprilTag: LOST (cannot reacquire)                    |")
                print(f"  | - Horizontal nav: Dead reckoning                       |")
                print(f"  | - Vertical nav: KF prediction-only                     |")
                print(f"  +---------------------------------------------------------+")
                
                # Switch KF to prediction-only mode (no measurement updates)
                kf.set_mode(prediction_only=True)
                
            
            if attack_triggered:
                # Attack mode - compromised sensors
                phase = 'blind_nav'
                kf_mode = 'prediction'
                
                # ============================================================
                # Emulate sensor attack
                # ============================================================
                # Inject +0.5m bias into ToF readings (compromised sensor)
                z_tof_reported = z_tof_real + 0.5
                
                
                # KF prediction only (no updates during attack)
                kf.predict(u=a_cmd)
                # NO update during attack
        
        # Get KF state AFTER prediction/update
        z_est, vz_est = kf.get_state()
        
        # ============================================================
        # Check mission phase transitions
        # ============================================================
        # 1. Check if horizontal target reached (within XY_TOLERANCE)
        # 2. Check if vertical target reached (within Z_TOLERANCE)
        # 3. When both reached, break from loop
        
        if not reached_horizontal_target:
            dx = target_x - x_est
            dy = target_y - y_est
            horizontal_error = np.sqrt(dx**2 + dy**2)
            
            if horizontal_error < XY_TOLERANCE:
                reached_horizontal_target = True
                print(f"\n  [+] Reached horizontal target region (error = {horizontal_error:.2f}m)")
        
        if reached_horizontal_target and not reached_vertical_target:
            dz_check = target_z - z_est
            if abs(dz_check) < Z_TOLERANCE:
                reached_vertical_target = True
                print(f"\n  [+] Reached target altitude (KF): {target_z:.2f}m")
                print(f"    Real altitude: {z_tof_real:.2f}m")
                print(f"    KF estimate:   {z_est:.2f}m")
                print(f"    Estimation error: {abs(z_tof_real - z_est)*100:.1f} cm")
                
                if attack_triggered:
                    blind_time = time.time() - attack_start_time
                    print(f"    Time under attack: {blind_time:.1f}s")
                
                print(f"\n  [+] Mission complete - commanding land")
                print(f"    Total time: {t:.1f}s")
                print(f"    Final position estimate: ({x_est:.2f}, {y_est:.2f}, {z_est:.2f})m")
                break
        
        # ============================================================
        # Calculate control commands (P-control)
        # ============================================================
        # Horizontal: P-control toward (target_x, target_y)
        # Vertical: P-control toward target_z
        
        if not reached_horizontal_target:
            # Navigate horizontally
            dx = target_x - x_est
            dy = target_y - y_est
            
            vx_cmd = np.clip(KP_XY * dx, -0.2, 0.2)
            vy_cmd = np.clip(KP_XY * dy, -0.2, 0.2)
            
            # Dead reckoning: when blind, assume we reached horizontal target
            if attack_triggered:
                x_est = target_x
                y_est = target_y
        else:
            vx_cmd = 0.0
            vy_cmd = 0.0
        
        if not reached_vertical_target:
            # Navigate to target altitude
            dz = target_z - z_est
            vz_cmd = np.clip(KP_Z * dz, -0.25, 0.25)
        else:
            vz_cmd = 0.0
        
        # ============================================================
        # Update vertical command history for next-step acceleration estimate
        prev_prev_vz_cmd = prev_vz_cmd
        prev_vz_cmd = vz_cmd
        
        # Send RC commands
        rc_lr = int(vy_cmd * 100)
        rc_fb = int(-vx_cmd * 100)
        rc_ud = int(vz_cmd * 100)
        tello.send_rc_control(rc_lr, rc_fb, rc_ud, 0)
        
        # Log data
        data['time'].append(t)
        data['phase'].append(phase)
        data['z_tof_real'].append(z_tof_real)
        data['z_tof_reported'].append(z_tof_reported)
        data['z_est'].append(z_est)
        data['vz_est'].append(vz_est)
        data['vz_cmd'].append(vz_cmd)
        data['x_tag'].append(x_current if tag_visible else np.nan)
        data['y_tag'].append(y_current if tag_visible else np.nan)
        data['x_est'].append(x_est)
        data['y_est'].append(y_est)
        data['tag_visible'].append(tag_visible)
        data['kf_mode'].append(kf_mode)
        
        # Progress updates
        if len(data['time']) % 20 == 0:
            if attack_triggered:
                mode = "BLIND"
                err = abs(z_tof_real - z_est) * 100
            else:
                mode = "NORMAL"
                err = abs(z_tof_real - z_est) * 100
            
            status = "APPROACH" if not reached_horizontal_target else "ASCEND"
            
            print(f"  t={t:.1f}s [{mode:6s}] {status:8s}: "
                  f"pos=({x_est:.2f},{y_est:.2f},{z_est:.2f}), "
                  f"z_err={err:.1f}cm, vz_cmd={vz_cmd:.2f}")
        
        # Safety timeout
        if t > 90.0:
            print("\n  WARNING: Mission timeout (90s) - aborting!")
            break
        
        time.sleep(0.1)
    
    # Stop and land
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    print("\n[LANDING]")
    tello.land()
    time.sleep(2.0)
    
    return data


def plot_mission_analysis(data):
    """Generate mission analysis plots"""
    ensure_dir(OUTPUT_DIR)
    
    times = np.array(data['time'])
    z_real = np.array(data['z_tof_real'])
    z_est = np.array(data['z_est'])
    z_rep = np.array(data['z_tof_reported'])
    tag_visible = np.array(data['tag_visible'])
    errors = np.abs(z_real - z_est) * 100
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Attack indices
    attack_indices = [i for i, p in enumerate(data['phase']) if p == 'blind_nav']
    
    # Plot 1: Altitude profile
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, z_real, 'b-', linewidth=2, label='Real altitude')
    ax1.plot(times, z_est, 'g--', linewidth=2, label='KF estimate')
    
    if attack_indices:
        attack_times = times[attack_indices]
        ax1.plot(attack_times, z_rep[attack_indices], 'r.', markersize=3, alpha=0.6,
                 label='Compromised sensor')
        ax1.axvspan(attack_times[0], attack_times[-1], color='red', alpha=0.15,
                    label='Attack window')
    
    ax1.axhline(TARGET_POSITION[2], color='purple', linestyle=':',
                linewidth=2, label=f'Target alt ({TARGET_POSITION[2]}m)')
    
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Altitude [m]', fontsize=11)
    ax1.set_title('Altitude Profile', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Altitude estimation error
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(times, errors, 'r-', linewidth=2)
    ax2.axhline(15, color='k', linestyle='--', linewidth=1.5,
                label='15cm threshold')
    
    if attack_indices:
        attack_errors = errors[attack_indices]
        max_error = np.max(attack_errors)
        mean_error = np.mean(attack_errors)
        ax2.text(0.05, 0.95, f'Attack Phase:\nMax: {max_error:.1f}cm\nMean: {mean_error:.1f}cm',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Error [cm]', fontsize=11)
    ax2.set_title('Altitude Estimation Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Horizontal trajectory (x-y plane)
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i in range(len(times)-1):
        color = 'blue' if data['tag_visible'][i] else 'red'
        alpha = 0.6 if data['tag_visible'][i] else 0.8
        ax3.plot(data['x_est'][i:i+2], data['y_est'][i:i+2], 
                color=color, linewidth=2, alpha=alpha)
    
    # Mark key positions
    ax3.plot(data['x_est'][0], data['y_est'][0], 'go', markersize=12, 
            label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax3.plot(TARGET_POSITION[0], TARGET_POSITION[1], 'r*', markersize=20, 
            label='Target', markeredgecolor='darkred', markeredgewidth=1)
    ax3.plot(data['x_est'][-1], data['y_est'][-1], 'bs', markersize=10,
            label='End', markeredgecolor='darkblue', markeredgewidth=2)
    
    ax3.set_xlabel('x [m] (forward)', fontsize=11)
    ax3.set_ylabel('y [m] (lateral)', fontsize=11)
    ax3.set_title('Horizontal Trajectory (x-y)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Vertical trajectory (x-z plane)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i in range(len(times)-1):
        color = 'blue' if data['tag_visible'][i] else 'red'
        alpha = 0.6 if data['tag_visible'][i] else 0.8
        ax4.plot(data['x_est'][i:i+2], data['z_est'][i:i+2],
                color=color, linewidth=2, alpha=alpha)
    
    ax4.plot(data['x_est'][0], data['z_est'][0], 'go', markersize=12,
            label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax4.plot(TARGET_POSITION[0], TARGET_POSITION[2], 'r*', markersize=20,
            label='Target', markeredgecolor='darkred', markeredgewidth=1)
    ax4.plot(data['x_est'][-1], data['z_est'][-1], 'bs', markersize=10,
            label='End', markeredgecolor='darkblue', markeredgewidth=2)
    
    ax4.set_xlabel('x [m] (forward)', fontsize=11)
    ax4.set_ylabel('z [m] (altitude)', fontsize=11)
    ax4.set_title('Vertical Trajectory (x-z)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: AprilTag visibility
    ax5 = fig.add_subplot(gs[1, 2])
    vis = np.array(data['tag_visible'], dtype=float)
    ax5.fill_between(times, 0, vis, color='green', alpha=0.3, label='Tag visible')
    ax5.fill_between(times, vis, 1, color='red', alpha=0.3, label='Tag lost')
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Status', fontsize=11)
    ax5.set_title('AprilTag Visibility', fontsize=12, fontweight='bold')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Lost', 'Visible'])
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Horizontal position vs time
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(times, data['x_est'], 'b-', linewidth=2, label='x position')
    ax6.axhline(TARGET_POSITION[0], color='blue', linestyle='--', linewidth=1.5,
                label=f'Target x ({TARGET_POSITION[0]}m)')
    
    if attack_indices:
        ax6.axvspan(times[attack_indices[0]], times[attack_indices[-1]],
                    color='red', alpha=0.15, label='Blind navigation')
    
    ax6.set_xlabel('Time [s]', fontsize=11)
    ax6.set_ylabel('Position [m]', fontsize=11)
    ax6.set_title('Horizontal Position (Dead Reckoning)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Mission summary panel
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    if attack_indices:
        attack_errors = errors[attack_indices]
        max_error = np.max(attack_errors)
        mean_error = np.mean(attack_errors)
        attack_time = times[attack_indices[-1]] - times[attack_indices[0]]
        tag_loss_time = times[attack_indices[0]]
    else:
        max_error = mean_error = attack_time = tag_loss_time = 0.0
    
    final_altitude_err = abs(data['z_tof_real'][-1] - TARGET_POSITION[2]) * 100
    
    total_time = times[-1]
    final_x = data['x_est'][-1]
    final_y = data['y_est'][-1]
    final_z = data['z_tof_real'][-1]
    
    summary = f"""MISSION SUMMARY
==============================

Total time: {total_time:.1f} s

Tag loss at: {tag_loss_time:.1f} s
Attack duration: {attack_time:.1f} s

ALTITUDE PERFORMANCE:
  Max error: {max_error:.1f} cm
  Mean error: {mean_error:.1f} cm
  Final error: {final_altitude_err:.1f} cm
  Status: {'PASS' if max_error < 15 else 'FAIL'}

FINAL POSITION:
  x: {final_x:.2f} m (target: {TARGET_POSITION[0]:.1f} m)
  y: {final_y:.2f} m (target: {TARGET_POSITION[1]:.1f} m)
  z: {final_z:.2f} m (target: {TARGET_POSITION[2]:.1f} m)

BLIND NAVIGATION:
  Horizontal: Dead reckoning
  Vertical:   KF prediction-only
  Success: {'YES' if max_error < 15 else 'NO'}
"""
    
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Attack Resilience Mission Analysis', fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'mission_analysis.png'), dpi=200, bbox_inches='tight')
    print(f"\n[+] Plots saved: {OUTPUT_DIR}/mission_analysis.png")
    plt.close()


def generate_report(data):
    """Generate text report"""
    times = np.array(data['time'])
    attack_indices = [i for i, p in enumerate(data['phase']) if p == 'blind_nav']
    errors = np.abs(np.array(data['z_tof_real']) - np.array(data['z_est'])) * 100
    
    if attack_indices:
        attack_errors = errors[attack_indices]
        max_err = np.max(attack_errors)
        mean_err = np.mean(attack_errors)
        attack_time = times[attack_indices[-1]] - times[attack_indices[0]]
        tag_loss_time = times[attack_indices[0]]
    else:
        max_err = mean_err = attack_time = tag_loss_time = 0
    
    final_altitude_err = abs(data['z_tof_real'][-1] - TARGET_POSITION[2]) * 100
    
    report = f"""
{'='*60}
ATTACK RESILIENCE MISSION REPORT
{'='*60}

MISSION PARAMETERS:
  Target position: ({TARGET_POSITION[0]}, {TARGET_POSITION[1]}, {TARGET_POSITION[2]}) m
  
TIMELINE:
  Total mission time: {times[-1]:.1f} s
  Tag lost at:        {tag_loss_time:.1f} s
  Attack duration:    {attack_time:.1f} s
  
ALTITUDE PERFORMANCE DURING ATTACK:
  Max estimation error:  {max_err:.1f} cm
  Mean estimation error: {mean_err:.1f} cm
  Threshold:             15 cm
  Status: {'PASS' if max_err < 15 else 'FAIL'}

FINAL ALTITUDE:
  Target: {TARGET_POSITION[2]:.2f} m
  Actual: {data['z_tof_real'][-1]:.2f} m
  Error:  {final_altitude_err:.1f} cm

NAVIGATION STRATEGY:
  Normal: AprilTag + ToF + KF updates
  Attack: Dead reckoning (x,y) + KF prediction-only (z)
  Sensor compromise: +0.5m bias on ToF

OVERALL RESULT:
  {'[+] MISSION SUCCESS' if max_err < 15 else '[-] MISSION DEGRADED'}

{'='*60}
"""
    
    ensure_dir(OUTPUT_DIR)
    report_path = os.path.join(OUTPUT_DIR, 'mission_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"[+] Report saved: {report_path}")


def main():
    print("\n" + "="*60)
    print("LAB 3 - PHASE 3: ATTACK RESILIENCE MISSION")
    print("="*60)
    
    ensure_dir(OUTPUT_DIR)
    
    # Connect to drone
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    if battery < 30:
        print("ERROR: Battery too low (need >30%).")
        return
    
    tello.streamon()
    time.sleep(2.0)
    
    # AprilTag detector
    detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )
    
    TAG_SIZE = 0.117  # [m]
    
    try:
        print("\nSETUP:")
        print("  1. AprilTag on wall at ~0.5–1.0 m height")
        print("  2. Drone ~2 m back from wall, centered on tag")
        print("  3. Clear 2.5 m vertical space for ascent")
        print("\nMission:")
        print("  - Navigate to (1.5, 0, 2.5) m using AprilTag")
        print("  - If tag lost -> blind navigation (dead reckoning + KF)")
        print("  - Reach target -> land")
        
        input("\nPress ENTER to start mission...")
        
        print("\nTaking off...")
        tello.takeoff()
        time.sleep(3.0)
        print("Stabilizing...")
        time.sleep(2.0)
        
        data = attack_resilience_mission(tello, detector, TAG_SIZE)
        
        if data:
            # Save data
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(OUTPUT_DIR, 'attack_mission.csv'), index=False)
            print(f"\n[+] Data saved: {OUTPUT_DIR}/attack_mission.csv")
            
            # Generate plots
            print("\nGenerating analysis plots...")
            plot_mission_analysis(data)
            
            # Generate report
            generate_report(data)
            
            print("\n" + "="*60)
            print("[+] MISSION COMPLETE!")
            print("="*60)
            print(f"\nDeliverables in {OUTPUT_DIR}/:")
            print("  [+] attack_mission.csv - Full telemetry")
            print("  [+] mission_analysis.png - Comprehensive plots")
            print("  [+] mission_report.txt - Performance summary")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        tello.send_rc_control(0, 0, 0, 0)
        try:
            tello.land()
        except:
            pass
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        tello.send_rc_control(0, 0, 0, 0)
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
