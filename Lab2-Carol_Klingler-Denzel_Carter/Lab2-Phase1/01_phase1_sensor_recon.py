#!/usr/bin/env python3
"""
Lab 2 - Phase 1: AprilTag Sensor Characterization with Altitude PID

- Uses AprilTag (tag25h9, 100 mm) for forward/back distance control.
- Uses sonar-based PID (from Lab 1) to hold altitude at a fixed tag height.
- Characterizes detection rate and pose noise at multiple distances.
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
import pandas as pd

# =========================
# Config / constants
# =========================

OUTPUT_DIR = "Lab2-Phase1"

SAMPLE_RATE = 10.0  # Hz for characterization logging and altitude updates
SAMPLE_DT = 1.0 / SAMPLE_RATE

TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.10  # meters (100 mm tag side length)

# >>> IMPORTANT: set this to the physical height of your printed tag above the floor <<<
TAG_HEIGHT = 0.81  # meters (e.g., 1.0 m; measure and adjust as needed)

# Camera intrinsics (from lab handout / calibration)
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]

# Distances at which we will characterize the tag
TEST_DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0]  # meters

# Number of samples to collect at each distance
SAMPLES_PER_DISTANCE = 300  # 300 samples @ 10 Hz -> 30 seconds

# Altitude PID gains (from Lab 1 tuning)
ALT_KP = 0.5
ALT_KI = 0.01
ALT_KD = 0.01

# Left right P gain
LR_KP = 0.5

# proportional gain on distance
DIST_KP = 0.5


# =========================
# Utility functions
# =========================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def velocity_to_rc(velocity_mps):
    """
    Map vertical/forward velocity in m/s to RC command [-100, 100].

    -1.0 m/s -> -100
     0.0 m/s -> 0
    +1.0 m/s -> +100
    """
    velocity_mps = max(-1.0, min(1.0, float(velocity_mps)))
    return int(velocity_mps * 100.0)


def get_sensor_data(tello):
    """Read sonar, barometer, and battery from the Tello."""
    try:
        height_sonar = tello.get_distance_tof() / 100.0  # cm -> m
    except Exception:
        height_sonar = np.nan

    try:
        height_baro = tello.get_barometer() / 100.0  # cm -> m
    except Exception:
        height_baro = np.nan

    try:
        battery = tello.get_battery()
    except Exception:
        battery = np.nan

    return {
        "height_sonar": height_sonar,
        "height_baro": height_baro,
        "battery": battery,
    }


def get_current_altitude(sensor_data):
    """
    Choose best altitude estimate, preferring sonar when valid.
    Fallback to barometer; if both are bad, assume ~1 m (rare).
    """
    h_sonar = sensor_data["height_sonar"]
    h_baro = sensor_data["height_baro"]

    if not np.isnan(h_sonar) and h_sonar > 0.1:
        return h_sonar
    if not np.isnan(h_baro):
        return h_baro
    return 1.0


# =========================
# PID controller (from Lab 1)
# =========================

class PIDController:
    """Simple discrete PID controller (same structure as Lab 1)."""

    def __init__(self, kp, ki, kd, dt):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)

        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def update(self, setpoint, measurement):
        """
        Compute PID output for current measurement.

        Returns:
            u      : total output (clamped to [-1, 1])
            p_term : proportional contribution
            i_term : integral contribution
            d_term : derivative contribution
        """
        error = float(setpoint) - float(measurement)

        # Proportional
        p_term = self.kp * error

        # Integral
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative
        if self.first_run:
            d_term = 0.0
            self.first_run = False
        else:
            d_term = self.kd * (error - self.prev_error) / self.dt

        u = p_term + i_term + d_term
        u = max(-1.0, min(1.0, u))  # clamp to velocity range

        self.prev_error = error
        return u, p_term, i_term, d_term

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True


# =========================
# AprilTag detection helpers
# =========================

def get_apriltag_detection(frame, detector):
    """
    Run AprilTag detection on a frame.

    Returns dict:
        detected : bool
        x, y, z  : pose in tag frame (meters) or NaN if not detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE,
    )

    if detections:
        pose = detections[0].pose_t.flatten()  # [x_tag, y_tag, z_tag]
        return {
            "detected": True,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
        }
    else:
        return {"detected": False, "x": np.nan, "y": np.nan, "z": np.nan}


# =========================
# High-level motion helpers
# =========================

def move_to_tag_height(tello, alt_pid, duration=5.0):
    """
    Bring the drone to the preset TAG_HEIGHT using the altitude PID.

    Runs for a fixed duration (e.g., 5–6 seconds) and then stops.
    We assume Tello's internal controller plus gravity/friction keep it near
    that height afterwards, with PID still running occasionally in other loops.
    """
    print(f"\nMoving to tag height ~{TAG_HEIGHT:.2f} m using PID...")
    start = time.time()

    while time.time() - start < duration:
        sensor = get_sensor_data(tello)
        alt = get_current_altitude(sensor)
        u_alt, _, _, _ = alt_pid.update(TAG_HEIGHT, alt)
        rc_ud = velocity_to_rc(u_alt)

        # Only adjust vertical velocity here
        tello.send_rc_control(0, 0, rc_ud, 0)

        time.sleep(SAMPLE_DT)

    tello.send_rc_control(0, 0, 0, 0)
    final_alt = get_current_altitude(get_sensor_data(tello))
    print(f"  Final altitude estimate: {final_alt:.2f} m")


def move_to_distance(tello, detector, alt_pid, target_distance, timeout=30.0):
    """
    Move the drone to a target distance from the tag (forward/back only).

    - Forward/back velocity: proportional on tag z-distance.
    - Vertical velocity: from altitude PID to hold at TAG_HEIGHT.
    """
    print(f"\nPositioning to {target_distance:.1f} m from tag...")

    start_time = time.time()
    dist_tol = 0.05       # meters
    max_speed = 0.3       # m/s limit on forward/back
    settled = 0
    required_settled = 20  # number of consecutive in-tolerance steps

    frame_reader = tello.get_frame_read()

    while time.time() - start_time < timeout:
        frame = frame_reader.frame
        detection = get_apriltag_detection(frame, detector)

        # Altitude PID (always active)
        sensor = get_sensor_data(tello)
        alt = get_current_altitude(sensor)
        u_alt, _, _, _ = alt_pid.update(TAG_HEIGHT, alt)
        rc_ud = velocity_to_rc(u_alt)

        # Forward/back control from tag distance
        rc_fb = 0
        if detection["detected"]:
            current_distance = detection["z"]
            error = target_distance - current_distance
            x_tag = detection["x"]   # lateral offset in tag frame
            v_y = LR_KP * x_tag    # move left/right to keep x_tag ≈ 0
            v_y = np.clip(v_y, -0.1, 0.1)
            rc_lr = velocity_to_rc(v_y)

            if abs(error) < dist_tol:
                # Within tolerance: count as "settled" and stop forward motion
                settled += 1
                rc_fb = 0
                if settled >= required_settled:
                    print(f"  Reached target distance: {current_distance:.2f} m")
                    break
            else:
                settled = 0
                v = np.clip(DIST_KP * error, -max_speed, max_speed)
                # Positive error (too close) -> move BACK -> negative rc_fb
                rc_fb = velocity_to_rc(-v)
        else:
            # Lost tag: hold position in x, still run altitude PID
            settled = 0
            rc_fb = 0
            rc_lr = 0

        # tello.send_rc_control(leftright, forwardback, updown, yaw)
        tello.send_rc_control(rc_lr, rc_fb, rc_ud, 0)
        time.sleep(0.1)

    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)


# =========================
# Data collection + analysis
# =========================

def characterize_apriltag_detection():
    """
    Main Phase 1 routine:
    - Connect, start video, take off.
    - Move to tag height using altitude PID.
    - For each test distance:
        - Move forward/back to that distance.
        - Collect AprilTag pose + altitude data for 30 seconds.
    """
    ensure_dir(OUTPUT_DIR)

    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    print("Connected!")

    print("Starting video stream...")
    tello.streamon()
    time.sleep(2.0)

    detector = Detector(families=TAG_FAMILY)
    alt_pid = PIDController(ALT_KP, ALT_KI, ALT_KD, dt=SAMPLE_DT)

    try:
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 30:
            print("Battery too low, need at least 30%")
            return None

        input("Press Enter to takeoff and start Phase 1...")

        print("Taking off...")
        tello.takeoff()
        time.sleep(2.0)

        # Bring drone to tag height once before starting distance sweeps
        alt_pid.reset()
        move_to_tag_height(tello, alt_pid, duration=6.0)

        csv_path = os.path.join(OUTPUT_DIR, "apriltag_characterization.csv")
        frame_reader = tello.get_frame_read()

        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "test_distance",
                "sample_num",
                "detected",
                "x_tag",
                "y_tag",
                "z_tag",
                "altitude",
                "battery",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for d_idx, test_distance in enumerate(TEST_DISTANCES):
                print(f"\n=== Distance {d_idx + 1}/{len(TEST_DISTANCES)}: {test_distance:.1f} m ===")

                # Reuse altitude PID, but reset its memory when we start a new distance
                alt_pid.reset()
                move_to_distance(tello, detector, alt_pid, test_distance)

                detection_count = 0

                for k in range(SAMPLES_PER_DISTANCE):
                    frame = frame_reader.frame
                    detection = get_apriltag_detection(frame, detector)

                    sensor = get_sensor_data(tello)
                    alt = get_current_altitude(sensor)

                    # Keep altitude near tag height while we collect data
                    u_alt, _, _, _ = alt_pid.update(TAG_HEIGHT, alt)
                    rc_ud = velocity_to_rc(u_alt)
                    tello.send_rc_control(0, 0, rc_ud, 0)

                    if detection["detected"]:
                        detection_count += 1

                    writer.writerow({
                        "test_distance": test_distance,
                        "sample_num": k,
                        "detected": 1 if detection["detected"] else 0,
                        "x_tag": detection["x"],
                        "y_tag": detection["y"],
                        "z_tag": detection["z"],
                        "altitude": alt,
                        "battery": sensor["battery"],
                    })

                    if k % 50 == 0 and k > 0:
                        rate = detection_count / (k + 1) * 100.0
                        print(f"  Sample {k}/{SAMPLES_PER_DISTANCE} | "
                              f"Detection {rate:.1f}% | Alt {alt:.2f} m")

                    time.sleep(SAMPLE_DT)

                final_rate = detection_count / SAMPLES_PER_DISTANCE * 100.0
                print(f"  Detection rate at {test_distance:.1f} m: {final_rate:.1f}%")

        print("\nLanding...")
        tello.land()
        time.sleep(2.0)

        return csv_path

    finally:
        # Cleanup regardless of success / failure
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass


def analyze_characterization_data(csv_path):
    """
    Compute detection rate and pose noise vs distance & make simple plots.
    """
    print(f"\nAnalyzing data from {csv_path}")
    df = pd.read_csv(csv_path)

    results = {}

    for d in sorted(df["test_distance"].unique()):
        sub = df[df["test_distance"] == d]
        total = len(sub)
        detected = sub["detected"].sum()
        detection_rate = detected / total if total > 0 else 0.0

        x_noise = float(np.nanstd(sub["x_tag"].values))
        y_noise = float(np.nanstd(sub["y_tag"].values))
        z_noise = float(np.nanstd(sub["z_tag"].values))
        alt_mean = float(np.nanmean(sub["altitude"].values))

        results[d] = {
            "detection_rate": detection_rate,
            "x_noise": x_noise,
            "y_noise": y_noise,
            "z_noise": z_noise,
            "altitude_mean": alt_mean,
        }

        print(f"\nDistance {d:.1f} m:")
        print(f"  Detection rate: {detection_rate * 100:.1f}%")
        print(f"  Noise (mm): X={x_noise * 1000:.1f}, "
              f"Y={y_noise * 1000:.1f}, Z={z_noise * 1000:.1f}")
        print(f"  Mean altitude: {alt_mean:.2f} m")

    # Simple plots
    distances = sorted(results.keys())
    det_rates = [results[d]["detection_rate"] * 100 for d in distances]
    x_noise = [results[d]["x_noise"] * 100 for d in distances]  # m -> cm
    y_noise = [results[d]["y_noise"] * 100 for d in distances]
    z_noise = [results[d]["z_noise"] * 100 for d in distances]

    # Detection rate vs distance
    plt.figure(figsize=(6, 4))
    plt.plot(distances, det_rates, "o-")
    plt.xlabel("Distance from tag [m]")
    plt.ylabel("Detection rate [%]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "detection_rate_vs_distance.png"), dpi=200)
    plt.close()

    # Pose noise vs distance
    plt.figure(figsize=(6, 4))
    plt.plot(distances, x_noise, "r.-", label="X noise")
    plt.plot(distances, y_noise, "g.-", label="Y noise")
    plt.plot(distances, z_noise, "b.-", label="Z noise")
    plt.xlabel("Distance from tag [m]")
    plt.ylabel("Std dev [cm]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "noise_vs_distance.png"), dpi=200)
    plt.close()

    results_path = os.path.join(OUTPUT_DIR, "characterization_results.json")
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\nSaved plots and results in {OUTPUT_DIR}/")


def main():
    print("Lab 2 - Phase 1: AprilTag Sensor Characterization with Altitude PID")
    ensure_dir(OUTPUT_DIR)

    csv_path = characterize_apriltag_detection()
    if csv_path:
        analyze_characterization_data(csv_path)
        print("\nPhase 1 complete.")
    else:
        print("\nNo data collected (battery low, error, or user abort).")


if __name__ == "__main__":
    main()
