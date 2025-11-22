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
from utils.apriltag_utils import get_apriltag_pose, tag_to_drone_velocity
from utils.kalman_filter import AltitudeKalmanFilter

OUTPUT_DIR = "Lab3-Phase3"

# Mission parameters
TARGET_POSITION = (1.5, 0.0, 2.5)  # (x, y, z) in tag frame

# ============================================================
# TODO: Tune these control parameters for stable flight
# ============================================================
KP_XY = 0.3  # Horizontal control gain
KP_Z = 0.20   # Vertical control gain
XY_TOLERANCE = 0.2  # Horizontal position tolerance [m]
Z_TOLERANCE = 0.1   # Vertical position tolerance [m]
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
    
    # Initialize KF
    kf = AltitudeKalmanFilter(dt=0.1)
    
    target_x, target_y, target_z = TARGET_POSITION
    
    # PHASE 1: Navigate toward target with AprilTag guidance
    print(f"\n[PHASE 1] Navigating toward target...")
    print(f"  Using AprilTag for guidance")
    
    # Verify initial tag detection
    print("  Verifying AprilTag detection...")
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
    
    print(f"  Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) m")
    print(f"  Distance: {np.sqrt((target_x-x_current)**2 + (target_y-y_current)**2 + (target_z-z_current)**2):.2f} m")
    
    # Main mission loop
    while True:
        t = time.time() - mission_start
        
        # Get real altitude
        z_tof_real = tello.get_distance_tof() / 100.0
        
        # Try to detect AprilTag
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
        
        # ============================================================
        # TODO: Implement tag loss detection and attack emulation
        # ============================================================
        # Determine sensor status:
        # - If tag detected and no attack: normal operation
        # - If tag lost for 3+ consecutive frames: trigger attack
        
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
            kf.predict(u=0.0)
            kf.update(z_tof_real)
            
        else:
            # Tag lost or attack active
            consecutive_lost += 1
            tag_visible = False
            
            # ============================================================
            # TODO: Trigger attack after 3 consecutive lost frames
            # ============================================================
            # 1. Check if consecutive_lost >= 3 and attack not yet triggered
            # 2. Set attack_triggered = True
            # 3. Print attack message
            # 4. Switch KF to prediction-only mode
            
            if consecutive_lost >= 3 and not attack_triggered:
                attack_triggered = True
                attack_start_time = time.time()
                
                dx_remaining = target_x - x_last_known
                dy_remaining = target_y - y_last_known
                dz_remaining = target_z - z_last_known
                dist_remaining = np.sqrt(dx_remaining**2 + dy_remaining**2 + dz_remaining**2)
                
                print(f"\n  +---------------------------------------------------------+")
                print(f"  | TAG LOST - ATTACK INITIATED                             |")
                print(f"  | Time: {t:.1f}s                                            |")
                print(f"  | Last position: ({x_last_known:.2f}, {y_last_known:.2f}, {z_last_known:.2f}) m              |")
                print(f"  | Distance to target: {dist_remaining:.2f} m                             |")
                print(f"  |                                                         |")
                print(f"  | BLIND NAVIGATION MODE:                                  |")
                print(f"  | - ToF sensor: +0.5m bias injected                      |")
                print(f"  | - AprilTag: LOST (cannot reacquire)                    |")
                print(f"  | - Horizontal nav: Dead reckoning                       |")
                print(f"  | - Vertical nav: KF prediction-only                     |")
                print(f"  +---------------------------------------------------------+")
                
                # Switch KF to prediction-only mode
                kf.set_mode(prediction_only=True)
            
            if attack_triggered:
                # Attack mode - compromised sensors
                phase = 'blind_nav'
                kf_mode = 'prediction'
                
                # ============================================================
                # TODO: Emulate sensor attack
                # ============================================================
                # Inject +0.5m bias into ToF readings
                # z_tof_reported = z_tof_real + 0.5
                z_tof_reported = z_tof_real + 0.5
                
                # KF prediction only (no updates during attack)
                kf.predict(u=0.0)
                # NO update during attack
        
        # Get KF state AFTER prediction/update
        z_est, vz_est = kf.get_state()
        
        # ============================================================
        # TODO: Check mission phase transitions
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
                print(f"\n  [+] Reached horizontal target")
                print(f"    Estimated position: ({x_est:.2f}, {y_est:.2f}) m")
        
        if reached_horizontal_target and not reached_vertical_target:
            dz = target_z - z_est
            
            if abs(dz) < Z_TOLERANCE:
                reached_vertical_target = True
                print(f"\n  [+] Reached target altitude: {target_z} m")
                print(f"    Real altitude: {z_tof_real:.2f} m")
                print(f"    KF estimate: {z_est:.2f} m")
                print(f"    Estimation error: {abs(z_tof_real - z_est)*100:.1f} cm")
                
                if attack_triggered:
                    blind_time = time.time() - attack_start_time
                    print(f"    Time under attack: {blind_time:.1f}s")
                
                print(f"\n  [+] Mission complete - commanding land")
                print(f"    Total time: {t:.1f}s")
                print(f"    Final position estimate: ({x_est:.2f}, {y_est:.2f}, {z_est:.2f})m")
                break
        
        # ============================================================
        # TODO: Calculate control commands
        # ============================================================
        # Use proportional control to navigate to target
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
            print("\n  WARNING: Safety timeout (90s)")
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
    """
    Generate comprehensive mission analysis plots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    times = np.array(data['time'])
    
    # Find phase transitions
    attack_indices = [i for i, p in enumerate(data['phase']) if p == 'blind_nav']
    tag_visible = np.array(data['tag_visible'])
    
    # Plot 1: Altitude profile
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, data['z_tof_real'], 'b-', linewidth=2.5, label='Real altitude', alpha=0.8)
    ax1.plot(times, data['z_est'], 'g--', linewidth=2.5, label='KF estimate', alpha=0.8)
    
    # Show compromised sensor during attack
    if attack_indices:
        attack_times = times[attack_indices]
        reported_vals = np.array(data['z_tof_reported'])[attack_indices]
        ax1.plot(attack_times, reported_vals, 'r.', alpha=0.4, markersize=3, 
                label='Compromised sensor')
        
        # Shade attack region
        ax1.axvspan(attack_times[0], attack_times[-1], alpha=0.15, 
                   color='red', label='Attack window')
    
    # Mark target altitude
    ax1.axhline(TARGET_POSITION[2], color='purple', linestyle=':', linewidth=2, 
               label=f'Target alt ({TARGET_POSITION[2]}m)')
    
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Altitude [m]', fontsize=11)
    ax1.set_title('Altitude Profile', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Altitude error
    ax2 = fig.add_subplot(gs[0, 2])
    errors = np.abs(np.array(data['z_tof_real']) - np.array(data['z_est'])) * 100
    ax2.plot(times, errors, 'r-', linewidth=2, alpha=0.7)
    ax2.axhline(15, color='k', linestyle='--', linewidth=1.5, label='15cm threshold')
    
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
    
    # Plot 4: 3D trajectory (x-z plane)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i in range(len(times)-1):
        color = 'blue' if data['tag_visible'][i] else 'red'
        alpha = 0.6 if data['tag_visible'][i] else 0.8
        ax4.plot(data['x_est'][i:i+2], data['z_est'][i:i+2],
                color=color, linewidth=2, alpha=alpha)
    
    # Mark key positions
    ax4.plot(data['x_est'][0], data['z_est'][0], 'go', markersize=12,
            label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax4.plot(TARGET_POSITION[0], TARGET_POSITION[2], 'r*', markersize=20,
            label='Target', markeredgecolor='darkred', markeredgewidth=1)
    ax4.plot(data['x_est'][-1], data['z_est'][-1], 'bs', markersize=10,
            label='End', markeredgecolor='darkblue', markeredgewidth=2)
    
    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='AprilTag visible'),
        Line2D([0], [0], color='red', linewidth=2, label='Blind navigation')
    ]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax4.set_xlabel('x [m] (forward)', fontsize=11)
    ax4.set_ylabel('z [m] (altitude)', fontsize=11)
    ax4.set_title('Vertical Trajectory (x-z)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: AprilTag visibility
    ax5 = fig.add_subplot(gs[1, 2])
    visibility = np.array(data['tag_visible']).astype(float)
    ax5.fill_between(times, 0, visibility, alpha=0.3, color='green', label='Tag visible')
    ax5.fill_between(times, visibility, 1, alpha=0.3, color='red', label='Tag lost')
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Status', fontsize=11)
    ax5.set_title('AprilTag Visibility', fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Lost', 'Visible'])
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Horizontal position over time
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(times, data['x_est'], 'b-', linewidth=2, label='x position', alpha=0.8)
    ax6.axhline(TARGET_POSITION[0], color='blue', linestyle='--', 
               linewidth=1.5, label=f'Target x ({TARGET_POSITION[0]}m)')
    
    if attack_indices:
        ax6.axvspan(times[attack_indices[0]], times[attack_indices[-1]], 
                   alpha=0.15, color='red', label='Blind navigation')
    
    ax6.set_xlabel('Time [s]', fontsize=11)
    ax6.set_ylabel('Position [m]', fontsize=11)
    ax6.set_title('Horizontal Position (Dead Reckoning)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Mission summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Calculate statistics
    if attack_indices:
        attack_errors = errors[attack_indices]
        max_err = np.max(attack_errors)
        mean_err = np.mean(attack_errors)
        attack_time = times[attack_indices[-1]] - times[attack_indices[0]]
        tag_loss_time = times[attack_indices[0]]
    else:
        max_err = mean_err = attack_time = tag_loss_time = 0
    
    final_altitude_err = abs(data['z_tof_real'][-1] - TARGET_POSITION[2]) * 100
    total_time = times[-1]
    final_x = data['x_est'][-1]
    final_y = data['y_est'][-1]
    
    summary = f"""MISSION SUMMARY
{'='*30}

Total time: {total_time:.1f} s

Tag loss at: {tag_loss_time:.1f} s
Attack duration: {attack_time:.1f} s

ALTITUDE PERFORMANCE:
Max error: {max_err:.1f} cm
Mean error: {mean_err:.1f} cm
Final error: {final_altitude_err:.1f} cm
Status: {'PASS' if max_err < 15 else 'FAIL'}

FINAL POSITION:
x: {final_x:.2f} m (target: {TARGET_POSITION[0]:.1f} m)
y: {final_y:.2f} m (target: {TARGET_POSITION[1]:.1f} m)
z: {data['z_tof_real'][-1]:.2f} m (target: {TARGET_POSITION[2]:.1f} m)

BLIND NAVIGATION:
Horizontal: Dead reckoning
Vertical: KF prediction-only
Success: {'YES' if max_err < 15 else 'NO'}
"""
    
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Attack Resilience Mission Analysis', fontsize=14, fontweight='bold')
    
    # Save
    ensure_dir(OUTPUT_DIR)
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
  AprilTag lost at: {tag_loss_time:.1f} s
  Blind navigation duration: {attack_time:.1f} s
  
PERFORMANCE METRICS:

Altitude Control (during attack):
  Max estimation error: {max_err:.1f} cm
  Mean estimation error: {mean_err:.1f} cm
  Status: {'[+] PASS' if max_err < 15 else '[-] FAIL'} (threshold: 15cm)

Final Position Accuracy:
  Target altitude: {TARGET_POSITION[2]:.2f} m
  Actual altitude: {data['z_tof_real'][-1]:.2f} m
  Final altitude error: {final_altitude_err:.1f} cm
  
  x: {data['x_est'][-1]:.2f} m (target: {TARGET_POSITION[0]} m, error: {abs(data['x_est'][-1] - TARGET_POSITION[0]):.2f} m)
  y: {data['y_est'][-1]:.2f} m (target: {TARGET_POSITION[1]} m, error: {abs(data['y_est'][-1] - TARGET_POSITION[1]):.2f} m)
  z: {data['z_tof_real'][-1]:.2f} m (target: {TARGET_POSITION[2]} m, error: {final_altitude_err:.1f} cm)

NAVIGATION STRATEGY:
  Normal mode: AprilTag + ToF sensor + KF updates
  Attack mode: Dead reckoning (x,y) + KF prediction-only (z)
  Sensor compromise: +0.5m ToF bias during attack

OVERALL RESULT: {'[+] MISSION SUCCESS' if max_err < 15 else '[-] MISSION DEGRADED'}

{'='*60}
"""
    
    ensure_dir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'mission_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"[+] Report saved: {OUTPUT_DIR}/mission_report.txt")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("LAB 3 - PHASE 3: ATTACK RESILIENCE MISSION")
    print("="*60)
    print("\nDemonstrates blind 3D navigation after AprilTag loss")
    
    ensure_dir(OUTPUT_DIR)
    
    # Connect to drone
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    
    if battery < 30:
        print("ERROR: Battery too low (need >30%)")
        return
    
    tello.streamon()
    time.sleep(2)
    
    # Setup AprilTag detector
    detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )
    
    TAG_SIZE = 0.117  # meters
    
    try:
        print("\n" + "="*60)
        print("SETUP")
        print("="*60)
        print("1. AprilTag on wall at ~0.5-1.0m height")
        print("2. Drone ~2m BACK from wall, centered on tag")
        print("3. Clear 2.5m vertical space for ascent")
        print("\nMission:")
        print("  - Navigate to (1.5, 0, 2.5) m using AprilTag")
        print("  - If tag lost -> blind navigation (dead reckoning + KF)")
        print("  - Reach target -> land")
        
        input("\nPress ENTER to start mission...")
        
        # Takeoff
        print("\nTaking off...")
        tello.takeoff()
        time.sleep(3)
        
        # Stabilize after takeoff
        print("Stabilizing after takeoff...")
        time.sleep(2)
        
        # Run mission
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
