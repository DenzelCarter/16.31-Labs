#!/usr/bin/env python3
"""
Lab 2 - Phase 1: AprilTag Sensor Characterization

Objectives:
1. Implement coordinate frame transformations between AprilTag and drone frames
2. Characterize AprilTag detection performance at various distances
3. Analyze position measurement noise and detection reliability
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
OUTPUT_DIR = "Lab2-Phase1"
SAMPLE_RATE = 10  # Hz
TEST_DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0]  # meters
SAMPLES_PER_DISTANCE = 300  # 30 seconds at 10 Hz

# AprilTag setup
TAG_SIZE = 0.162  # meters
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def get_tello_frame_opencv():
    """
    OpenCV-based capture for Tello video stream.
    This avoids PyAV errors on Windows when opening udp://@0.0.0.0:11111
    """
    cap = cv2.VideoCapture("udp://0.0.0.0:11111", cv2.CAP_FFMPEG)
    return cap


class CoordinateTransformer:
    """
    Coordinate transformations between AprilTag and drone frames.

    AprilTag Frame:
        x_tag: lateral displacement (positive = drone is to the right of tag)
        y_tag: vertical displacement (positive = drone is above tag)
        z_tag: distance from tag (positive = drone is away from tag)

    Drone Body Frame (for velocity commands):
        x_drone: forward/backward (positive = forward)
        y_drone: left/right (positive = right)
        z_drone: up/down (positive = up)
    """

    def __init__(self):
        pass

    def tag_to_drone_position(self, x_tag, y_tag, z_tag):
        """
        Transform AprilTag measurements to drone body frame.

        For this lab we assume drone is facing the tag (yaw ≈ 0), so we can map axes:

        - tag z (distance out of tag) → drone x (forward) but with sign flip
        - tag x (right) → drone y (right) (same sign)
        - tag y (up) → drone z (up) (same sign)
        """
        x_drone = -float(z_tag)   # farther from tag → more negative forward
        y_drone = float(x_tag)    # right stays right
        z_drone = float(y_tag)    # up stays up
        return x_drone, y_drone, z_drone

    def position_error_to_velocity_command(
        self,
        x_error,
        y_error,
        z_error,
        kp_x=0.5,
        kp_y=0.5,
        kp_z=0.5,
    ):
        """
        Convert position errors to velocity commands using proportional control.

        velocity = Kp * error, with |velocity| ≤ 0.3 m/s
        """
        max_velocity = 0.3

        vx = kp_x * float(x_error)  # forward
        vy = kp_y * float(y_error)  # right
        vz = kp_z * float(z_error)  # up

        vx = np.clip(vx, -max_velocity, max_velocity)
        vy = np.clip(vy, -max_velocity, max_velocity)
        vz = np.clip(vz, -max_velocity, max_velocity)

        return vx, vy, vz


def get_apriltag_detection(frame, detector):
    """
    Detect AprilTag in frame and return pose.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE,
    )

    if len(detections) > 0:
        detection = detections[0]
        pose = detection.pose_t.flatten()
        return {
            "detected": True,
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
        }
    else:
        return {
            "detected": False,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
        }


def move_to_distance(tello, detector, target_distance, cap, timeout=30.0):
    """
    Move drone to target distance from AprilTag using simple proportional control.
    Uses OpenCV capture to read frames.
    """
    print(f"\nPositioning to {target_distance:.1f}m from tag...")

    start_time = time.time()
    settled_count = 0
    required_settled = 20  # need to be within tolerance 20 cycles ~2s

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            # no frame → pause and try again
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)
            continue

        detection = get_apriltag_detection(frame, detector)

        if detection["detected"]:
            current_distance = detection["z"]
            error = target_distance - current_distance

            if abs(error) < 0.05:  # within 5 cm
                settled_count += 1
                if settled_count >= required_settled:
                    print(f"Reached target: {current_distance:.2f}m")
                    break
            else:
                settled_count = 0

            # simple P on distance
            velocity_cmd = np.clip(0.5 * error, -0.3, 0.3)
            # Tello rc: left/right, forward/back, up/down, yaw
            # we want forward/back, so that's the second arg
            rc_cmd = int(-velocity_cmd * 100)  # sign flip: +forward is negative rc
            tello.send_rc_control(0, rc_cmd, 0, 0)
        else:
            # no tag → stop
            tello.send_rc_control(0, 0, 0, 0)

        time.sleep(0.1)

    # stop at end
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1.0)


def characterize_apriltag_detection():
    """
    Main characterization function. Collects detection data at multiple distances.
    """
    ensure_dir(OUTPUT_DIR)

    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    print("Connected!")

    print("Starting video stream...")
    tello.streamon()
    time.sleep(2)

    # OpenCV capture (our fallback)
    cap = get_tello_frame_opencv()
    if not cap.isOpened():
        print("WARNING: OpenCV could not open Tello stream. Detection will fail.")

    detector = Detector(families="tag36h11")

    try:
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 30:
            print("ERROR: Battery too low. Need at least 30%")
            return None

        input("Press Enter to takeoff and begin characterization...")

        print("\nTaking off...")
        tello.takeoff()
        time.sleep(2)

        print("Adjusting altitude...")
        tello.move_down(30)
        time.sleep(2)

        print("\nStarting AprilTag characterization")
        print(f"Test distances: {TEST_DISTANCES}")
        print(f"Samples per distance: {SAMPLES_PER_DISTANCE}")

        csv_filename = os.path.join(OUTPUT_DIR, "apriltag_characterization.csv")

        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = [
                "test_distance",
                "sample_num",
                "detected",
                "x_tag",
                "y_tag",
                "z_tag",
                "battery",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for test_idx, test_distance in enumerate(TEST_DISTANCES):
                print(f"\nTest {test_idx+1}/{len(TEST_DISTANCES)}: {test_distance:.1f}m")

                move_to_distance(tello, detector, test_distance, cap)

                print(f"Collecting {SAMPLES_PER_DISTANCE} samples...")
                detection_count = 0

                for sample_num in range(SAMPLES_PER_DISTANCE):
                    ret, frame = cap.read()
                    if not ret:
                        detection = {
                            "detected": False,
                            "x": np.nan,
                            "y": np.nan,
                            "z": np.nan,
                        }
                    else:
                        detection = get_apriltag_detection(frame, detector)

                    battery = tello.get_battery()

                    if detection["detected"]:
                        detection_count += 1

                    row = {
                        "test_distance": test_distance,
                        "sample_num": sample_num,
                        "detected": 1 if detection["detected"] else 0,
                        "x_tag": detection["x"],
                        "y_tag": detection["y"],
                        "z_tag": detection["z"],
                        "battery": battery,
                    }
                    writer.writerow(row)

                    if sample_num % 50 == 0 and sample_num > 0:
                        rate = (detection_count / (sample_num + 1)) * 100
                        print(
                            f"  Sample {sample_num}/{SAMPLES_PER_DISTANCE} | "
                            f"Detection: {rate:.1f}% | Battery: {battery}%"
                        )

                    tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(1.0 / SAMPLE_RATE)

                final_rate = (detection_count / SAMPLES_PER_DISTANCE) * 100
                print(f"Completed! Detection rate: {final_rate:.1f}%")

        print("\nData collection complete!")
        print(f"Data saved to: {csv_filename}")

        print("\nLanding...")
        tello.land()
        time.sleep(2)

        return csv_filename

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        try:
            tello.land()
        except Exception:
            pass
        return None

    except Exception as e:
        print(f"\nERROR during characterization: {e}")
        import traceback

        traceback.print_exc()
        try:
            tello.land()
        except Exception:
            pass
        return None
    finally:
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass
        if cap is not None:
            cap.release()


def analyze_characterization_data(csv_filename):
    """
    Analyze collected characterization data and generate metrics:
    - Detection success rate per distance
    - Position noise per distance
    - Max consecutive dropout per distance
    """
    import pandas as pd

    print(f"\nAnalyzing characterization data from: {csv_filename}")
    df = pd.read_csv(csv_filename)

    results = {}

    for test_distance in df["test_distance"].unique():
        distance_data = df[df["test_distance"] == test_distance]

        # detection rate
        total_samples = len(distance_data)
        detected_samples = int(distance_data["detected"].sum())
        detection_rate = detected_samples / total_samples

        # helper for noise
        def noise_std(series):
            arr = series.to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return 0.0
            # remove bias
            arr = arr - np.nanmedian(arr)
            return float(np.nanstd(arr))

        x_noise = noise_std(distance_data["x_tag"])
        y_noise = noise_std(distance_data["y_tag"])
        z_noise = noise_std(distance_data["z_tag"])

        # max consecutive dropout
        detected_array = distance_data["detected"].values
        max_dropout = 0
        current = 0
        for d in detected_array:
            if d == 0:
                current += 1
                if current > max_dropout:
                    max_dropout = current
            else:
                current = 0
        max_dropout_seconds = max_dropout / float(SAMPLE_RATE)

        results[test_distance] = {
            "detection_rate": detection_rate,
            "x_noise": x_noise,
            "y_noise": y_noise,
            "z_noise": z_noise,
            "max_dropout": max_dropout_seconds,
        }

        print(f"\nDistance: {test_distance:.1f}m")
        print(f"  Detection rate:  {detection_rate*100:6.1f}%")
        print(f"  Position noise:")
        print(f"    X: {x_noise*1000:6.2f} mm")
        print(f"    Y: {y_noise*1000:6.2f} mm")
        print(f"    Z: {z_noise*1000:6.2f} mm")
        print(f"  Max dropout:     {max_dropout_seconds:6.2f} s")

    create_characterization_plots(results)

    results_file = os.path.join(OUTPUT_DIR, "characterization_results.json")
    with open(results_file, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {results_file}")


def create_characterization_plots(results):
    """
    Generate characterization plots.
    """
    distances = sorted(results.keys())
    detection_rates = [results[d]["detection_rate"] * 100 for d in distances]
    x_noise = [results[d]["x_noise"] * 100 for d in distances]
    y_noise = [results[d]["y_noise"] * 100 for d in distances]
    z_noise = [results[d]["z_noise"] * 100 for d in distances]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(distances, detection_rates, "bo-", linewidth=2, markersize=8)
    ax1.axhline(95, color="g", linestyle="--", label="95% threshold")
    ax1.set_xlabel("Distance from Tag [m]")
    ax1.set_ylabel("Detection Success Rate [%]")
    ax1.set_title("AprilTag Detection Reliability")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 105)

    ax2.plot(distances, x_noise, "r.-", label="X noise", markersize=8)
    ax2.plot(distances, y_noise, "g.-", label="Y noise", markersize=8)
    ax2.plot(distances, z_noise, "b.-", label="Z noise", markersize=8)
    ax2.set_xlabel("Distance from Tag [m]")
    ax2.set_ylabel("Position Noise Std Dev [cm]")
    ax2.set_title("AprilTag Pose Measurement Noise")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "apriltag_characterization.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}/apriltag_characterization.png")


def main():
    print("Lab 2 - Phase 1: AprilTag Sensor Characterization")
    print("=" * 60)

    csv_filename = characterize_apriltag_detection()

    if csv_filename:
        analyze_characterization_data(csv_filename)
        print("\nPhase 1 Complete!")
        print(f"Files saved in: {OUTPUT_DIR}/")
    else:
        print("\nCharacterization failed or was cancelled")


if __name__ == "__main__":
    main()
