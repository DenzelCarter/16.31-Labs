#!/usr/bin/env python3
"""
AprilTag Utilities for Lab 3

Provides simple proportional-control based positioning using AprilTag detection.
Students use these as black-box utilities to focus on altitude control/filtering.

Coordinate system (tag frame):
- x: forward from tag (perpendicular to tag face)
- y: left from tag  
- z: up from tag
"""

import numpy as np
import time


def get_apriltag_pose(frame, detector, tag_size=0.125):
    """
    Detect AprilTag and return 3D pose in tag frame.
    
    Args:
        frame: Camera image from drone
        detector: pupil_apriltags Detector instance
        tag_size: Physical tag size in meters (default: 0.125m = 12.5cm)
    
    Returns:
        dict with:
            'detected': bool
            'x': forward distance from tag [m]
            'y': lateral distance from tag [m] 
            'z': vertical distance from tag [m]
    """
    # Convert to grayscale for detection
    import cv2
    
    if frame is None:
        print("WARNING: get_apriltag_pose received None frame!")
        return {'detected': False, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"WARNING: Failed to convert frame to grayscale: {e}")
        return {'detected': False, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    # Detect tags
    detections = detector.detect(gray, estimate_tag_pose=True, 
                                  camera_params=[921.170702, 919.018377, 459.904354, 351.238301],
                                  tag_size=tag_size)
    
    if len(detections) == 0:
        return {'detected': False, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    # Use first detected tag
    detection = detections[0]
    
    # Extract translation vector (position of tag in camera frame)
    # pupil-apriltags returns position in camera coordinates
    t = detection.pose_t.flatten()
    
    # Transform from camera frame to tag frame
    # Camera frame: +Z forward, +X right, +Y down
    # Tag frame: +X forward from tag, +Y left from tag, +Z up from tag
    x_tag = t[2]   # Camera Z -> tag X (forward)
    y_tag = t[0]  # Camera -X -> tag Y (left)
    z_tag = -t[1]  # Camera -Y -> tag Z (up)
    
    return {
        'detected': True,
        'x': float(x_tag),
        'y': float(y_tag),
        'z': float(z_tag)
    }


def tag_to_drone_velocity(vx_tag, vy_tag, vz_tag, vz_yaw=0):
    """
    Convert tag-frame velocities to Tello RC commands.
    
    Args:
        vx_tag: Forward velocity in tag frame [m/s]
        vy_tag: Lateral velocity in tag frame [m/s]
        vz_tag: Vertical velocity in tag frame [m/s]
        vz_yaw: Yaw rate [deg/s] (default: 0)
    
    Returns:
        rc_lr: Left-right RC command [-100, 100]
        rc_fb: Forward-back RC command [-100, 100]
        rc_ud: Up-down RC command [-100, 100]
        rc_yaw: Yaw RC command [-100, 100]
    """
    # Velocity scaling: ±0.3 m/s -> ±100 RC units
    velocity_scale = 100.0 / 0.3
    
    # Map tag velocities to drone body frame
    # Tag frame +X (forward) -> Drone forward-back
    # Tag frame +Y (left) -> Drone left-right  
    # Tag frame +Z (up) -> Drone up-down
    rc_fb = int(np.clip(-vx_tag * velocity_scale, -100, 100))
    rc_lr = int(np.clip(vy_tag * velocity_scale, -100, 100))
    rc_ud = int(np.clip(vz_tag * velocity_scale, -100, 100))
    rc_yaw = int(np.clip(vz_yaw, -100, 100))
    
    return rc_lr, rc_fb, rc_ud, rc_yaw


def position_to_target(tello, detector, target_x, target_y, target_z, 
                       timeout=20.0, position_threshold=0.10, velocity_threshold=0.05,
                       tag_size=0.117):
    """
    Move drone to target position using simple proportional control.
    
    Uses visual servoing with AprilTag feedback. Continues commanding
    velocities until position error is small and drone is approximately stationary.
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector
        target_x: Target forward position from tag [m]
        target_y: Target lateral position from tag [m]
        target_z: Target altitude above tag [m]
        timeout: Maximum time to attempt positioning [s]
        position_threshold: Position error tolerance [m]
        velocity_threshold: Velocity threshold for "stationary" [m/s]
        tag_size: Physical tag size in meters (default: 0.125m = 12.5cm)
    
    Returns:
        success: True if positioned successfully, False if timeout
    """
    print(f"  Positioning to ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})m from tag...")
    
    # Proportional gains (conservative for stability)
    kp_x = 0.2
    kp_y = 0.2
    kp_z = 0.2
    
    # Velocity limits [m/s]
    max_vel_xy = 0.2
    max_vel_z = 0.2
    
    start_time = time.time()
    settled_count = 0
    required_settled = 10  # Require 10 consecutive "good" measurements
    
    last_pose = None
    
    while time.time() - start_time < timeout:
        # Get current pose
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
        
        if not pose['detected']:
            # Lost tag - stop and wait
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)
            continue
        
        # Calculate errors
        x_error = target_x - pose['x']
        y_error = target_y - pose['y']
        z_error = target_z - pose['z']
        
        position_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
        
        # Estimate velocity from position change (if we have previous pose)
        if last_pose is not None and last_pose['detected']:
            dt = 0.1  # Approximate
            vx_est = (pose['x'] - last_pose['x']) / dt
            vy_est = (pose['y'] - last_pose['y']) / dt
            vz_est = (pose['z'] - last_pose['z']) / dt
            velocity_mag = np.sqrt(vx_est**2 + vy_est**2 + vz_est**2)
        else:
            velocity_mag = 999  # Unknown, assume moving
        
        last_pose = pose
        
        # Check if settled
        if position_error < position_threshold and velocity_mag < velocity_threshold:
            settled_count += 1
            if settled_count >= required_settled:
                print(f"  ✓ Positioned: error={position_error*100:.1f}cm")
                tello.send_rc_control(0, 0, 0, 0)
                return True
        else:
            settled_count = 0
        
        # Proportional control
        vx_cmd = kp_x * x_error
        vy_cmd = kp_y * y_error
        vz_cmd = kp_z * z_error
        
        # Apply limits
        vx_cmd = np.clip(vx_cmd, -max_vel_xy, max_vel_xy)
        vy_cmd = np.clip(vy_cmd, -max_vel_xy, max_vel_xy)
        vz_cmd = np.clip(vz_cmd, -max_vel_z, max_vel_z)
        
        # Send commands
        rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx_cmd, vy_cmd, vz_cmd)
        tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)
        
        # Progress update every 1s
        if int(time.time() - start_time) % 1 == 0 and int((time.time() - start_time) * 10) % 10 == 0:
            print(f"    t={time.time()-start_time:.1f}s: pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), "
                  f"err={position_error*100:.1f}cm")
        
        time.sleep(0.1)
    
    # Timeout
    print(f"  Positioning timeout after {timeout}s")
    tello.send_rc_control(0, 0, 0, 0)
    return False


def hover_at_current_position(tello, detector, duration=2.0, tag_size=0.125):
    """
    Hold current position using proportional control.
    
    Useful for stabilizing before starting a new maneuver.
    
    Args:
        tello: Tello drone object
        detector: AprilTag detector  
        duration: How long to hover [s]
        tag_size: Physical tag size in meters (default: 0.125m = 12.5cm)
    
    Returns:
        success: True if maintained position, False if lost tag
    """
    print(f"  Hovering for {duration}s...")
    
    # Get initial position
    frame = tello.get_frame_read().frame
    initial_pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
    
    if not initial_pose['detected']:
        print("  ✗ Cannot hover - no AprilTag detected")
        return False
    
    # Use current position as target
    return position_to_target(
        tello, detector,
        target_x=initial_pose['x'],
        target_y=initial_pose['y'], 
        target_z=initial_pose['z'],
        timeout=duration + 2.0,
        tag_size=tag_size
    )


def simple_xy_control(x_ref, y_ref, x_meas, y_meas, kp=0.4, max_vel=0.2):
    """
    Simple proportional control for horizontal (x,y) tracking.
    
    Used for orbit tracking and stabilization. Students focus on altitude
    control with Kalman filter; this handles horizontal positioning.
    
    Args:
        x_ref, y_ref: Reference position [m]
        x_meas, y_meas: Measured position [m]
        kp: Proportional gain (default: 0.4)
        max_vel: Maximum velocity limit [m/s] (default: 0.2)
    
    Returns:
        vx, vy: Velocity commands [m/s]
    """
    # Simple P-control
    x_error = x_ref - x_meas
    y_error = y_ref - y_meas
    
    vx = kp * x_error
    vy = kp * y_error
    
    # Apply limits
    vx = np.clip(vx, -max_vel, max_vel)
    vy = np.clip(vy, -max_vel, max_vel)
    
    return vx, vy