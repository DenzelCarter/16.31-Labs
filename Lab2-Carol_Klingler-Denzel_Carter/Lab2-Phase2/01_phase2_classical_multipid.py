#!/usr/bin/env python3
"""
Lab 2 - Phase 2: Multi-Axis PID Control

- Multi-axis PID controller (x, y, z) using three Lab 1-style PID loops
- Constant yaw via attitude feedback (maintain initial yaw after takeoff)
- Step-response test and Navy requirement evaluation

Assumptions:
- You physically orient the drone on the ground facing the AprilTag before takeoff.
- After takeoff, yaw_ref is captured and held constant via yaw control.
"""

import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello
import cv2
from pupil_apriltags import Detector

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "Lab2-Phase2"
SAMPLE_DT = 0.05  # 20 Hz

# AprilTag parameters (update to match your tag)
TAG_SIZE = 0.078  # meters, e.g. 0.10 for 100 mm tag
CAMERA_PARAMS = [921.170702, 919.018377, 459.904354, 351.238301]
TAG_FAMILY = "tag25h9"  # change to 'tag36h11' if you use that family

# Navy performance requirements 
SETTLING_TIME_REQ = 4.0    # s
OVERSHOOT_REQ = 15.0        # %
STEADY_STATE_REQ = 0.10     # m

# Yaw control
YAW_KP = 4.0     # [rc units / deg] small proportional gain
YAW_RC_LIMIT = 50  # max |rc_yaw|


# =============================================================================
# Utility
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =============================================================================
# PID Controllers
# =============================================================================

class PIDController:
    """Single-axis PID controller (Lab 1 style)."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=SAMPLE_DT):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)

        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True

    def update(self, setpoint, measured_value):
        """
        Calculate PID output.

        Returns:
            output: control output
            p_term, i_term, d_term: contributions (for debugging)
        """
        error = float(setpoint) - float(measured_value)

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
            d_term = self.kd * (error - self.previous_error) / self.dt

        self.previous_error = error

        output = p_term + i_term + d_term
        return output, p_term, i_term, d_term

    def reset(self):
        """Reset PID memory."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_run = True


class MultiAxisPIDController:
    """
    Multi-axis PID using three independent single-axis PIDs.

    Conceptually:
      - X-axis PID: uses xtag error (lateral)
      - Y-axis PID: uses ytag error (vertical)
      - Z-axis PID: uses ztag error (distance)

    Outputs:
      vx, vy, vz commands (interpreted in tag/drone frame and then mapped to RC).
    """

    def __init__(
        self,
        kp_x=0.5, ki_x=0.01, kd_x=0.05,
        kp_y=0.5, ki_y=0.01, kd_y=0.05,
        kp_z=0.5, ki_z=0.01, kd_z=0.05,
        dt=SAMPLE_DT,
    ):
        self.pid_x = PIDController(kp_x, ki_x, kd_x, dt)
        self.pid_y = PIDController(kp_y, ki_y, kd_y, dt)
        self.pid_z = PIDController(kp_z, ki_z, kd_z, dt)

    def update(self, x_ref, y_ref, z_ref, x_meas, y_meas, z_meas):
        """
        Run three PID loops and return clipped velocity commands (m/s).
        """
        vx_raw, _, _, _ = self.pid_x.update(x_ref, x_meas)
        vy_raw, _, _, _ = self.pid_y.update(y_ref, y_meas)
        vz_raw, _, _, _ = self.pid_z.update(z_ref, z_meas)

        max_vel = 0.3  # |vi| ≤ 0.3 m/s per axis
        vx = float(np.clip(vx_raw, -max_vel, max_vel))
        vy = float(np.clip(vy_raw, -max_vel, max_vel))
        vz = float(np.clip(vz_raw, -max_vel, max_vel))

        return vx, vy, vz

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()


# =============================================================================
# AprilTag + yaw helpers
# =============================================================================

def get_apriltag_pose(frame, detector):
    """
    Detect AprilTag and return pose in tag frame.

    Returns dict:
        detected : bool
        x, y, z  : floats (meters or NaN)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE,
    )

    if detections:
        pose = detections[0].pose_t.flatten()
        return {
            "detected": True,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
        }
    else:
        return {"detected": False, "x": np.nan, "y": np.nan, "z": np.nan}


def tag_vel_to_rc(vx_tag, vy_tag, vz_tag):
    """
    Convert tag-frame velocity commands to Tello RC commands.

    Tag frame:
      - x_tag: lateral (right positive)
      - y_tag: vertical (up positive)
      - z_tag: distance (away from tag positive)

    RC commands:
      - left_right_velocity (lr): +right
      - forward_backward_velocity (fb): +forward
      - up_down_velocity (ud): +up
      - yaw_velocity (yaw): handled separately

    Mapping (assuming yaw ≈ 0 relative to tag):
      - vx_tag controls left/right
      - vy_tag controls up/down
      - vz_tag controls forward/back (with sign flip).
    """
    lr = int(np.clip(vx_tag * 100.0, -100, 100))
    fb = int(np.clip(-vz_tag * 100.0, -100, 100))  # move forward to decrease z
    ud = int(np.clip(vy_tag * 100.0, -100, 100))

    return lr, fb, ud


def wrap_angle_deg(angle):
    """Wrap angle to [-180, 180] degrees."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a


def compute_yaw_rc(tello, yaw_ref_deg):
    """
    Simple P controller to hold yaw at yaw_ref_deg using tello.get_yaw().

    Returns:
        rc_yaw in [-YAW_RC_LIMIT, YAW_RC_LIMIT]
    """
    try:
        yaw_now = tello.get_yaw()
    except Exception:
        # If we can't read yaw, do nothing
        return 0

    yaw_error = wrap_angle_deg(yaw_ref_deg - yaw_now)
    rc_yaw = int(np.clip(YAW_KP * yaw_error, -YAW_RC_LIMIT, YAW_RC_LIMIT))
    return rc_yaw


# =============================================================================
# Positioning + step-response
# =============================================================================

def position_to_target(tello, detector, yaw_ref, x_target, y_target, z_target, timeout=20.0):
    """
    Gentle positioning helper to bring the drone near a desired [x,y,z].

    - Low gains and low velocity limits so it moves slowly.
    - Stops if tag is lost for too long.
    - Always sends 0 velocity when exiting.
    """
    print(f"position_to_target → target = [{x_target:.2f}, {y_target:.2f}, {z_target:.2f}]")
    start = time.time()
    settled_count = 0
    max_missed = 20          # ~1 second of missed tags at 50 ms
    missed = 0

    # Very conservative gains & limits
    kp_pos = 0.15           # was 0.3
    max_vel = 0.10          # m/s (was 0.2)

    try:
        while time.time() - start < timeout:
            frame = tello.get_frame_read().frame
            pose = get_apriltag_pose(frame, detector)

            if pose["detected"]:
                missed = 0

                ex = x_target - pose["x"]
                ey = y_target - pose["y"]
                ez = z_target - pose["z"]

                # Check if close enough
                if abs(ex) < 0.07 and abs(ey) < 0.07 and abs(ez) < 0.07:
                    settled_count += 1
                    if settled_count > 30:   # stay settled for ~1.5 s
                        print("  Reached initial position.")
                        break
                else:
                    settled_count = 0

                vx = float(np.clip(kp_pos * ex, -max_vel, max_vel))
                vy = float(np.clip(kp_pos * ey, -max_vel, max_vel))
                vz = float(np.clip(kp_pos * ez, -max_vel, max_vel))

                lr, fb, ud = tag_vel_to_rc(vx, vy, vz)
                rc_yaw = compute_yaw_rc(tello, yaw_ref)
                tello.send_rc_control(lr, fb, ud, rc_yaw)

            else:
                # No tag → stop movement, count missed frames
                missed += 1
                tello.send_rc_control(0, 0, 0, 0)
                if missed > max_missed:
                    print("  Lost tag during positioning, aborting.")
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        # Make sure we stop immediately on Ctrl+C
        tello.send_rc_control(0, 0, 0, 0)
        raise

    # Extra safety stop when leaving the function
    tello.send_rc_control(0, 0, 0, 0)


def test_step_response(tello, detector, controller, yaw_ref, test_name):
    """
    Run a diagonal 3D step test and log data for metric calculation.

    Initial: approximately [0, 0, z0]
    Target:  [0.15, 0.15, 1.4] (example values, adjust to match lab spec)
    """
    print(f"\nRunning test: {test_name}")
    ensure_dir(OUTPUT_DIR)

    # Define initial and target tag-frame positions
    x_initial, y_initial, z_initial = -0.15, -0.15, 1.2
    x_target, y_target, z_target = 0.15, 0.15, 1.5

    print(f"Positioning to initial [{x_initial:.2f}, {y_initial:.2f}, {z_initial:.2f}]...")
    position_to_target(tello, detector, yaw_ref, x_initial, y_initial, z_initial, timeout=15.0)
    time.sleep(2.0)

    controller.reset()
    print("Executing step...")

    data = {
        "time": [],
        "x_ref": [], "y_ref": [], "z_ref": [],
        "x_meas": [], "y_meas": [], "z_meas": [],
        "vx_cmd": [], "vy_cmd": [], "vz_cmd": [],
    }

    test_duration = 15.0
    start = time.time()
    next_sample = start

    frame_reader = tello.get_frame_read()

    while time.time() - start < test_duration:
        if time.time() >= next_sample:
            frame = frame_reader.frame
            pose = get_apriltag_pose(frame, detector)

            if pose["detected"]:
                vx, vy, vz = controller.update(
                    x_target, y_target, z_target,
                    pose["x"], pose["y"], pose["z"],
                )

                lr, fb, ud = tag_vel_to_rc(vx, vy, vz)
                rc_yaw = compute_yaw_rc(tello, yaw_ref)

                tello.send_rc_control(lr, fb, ud, rc_yaw)

                t = time.time() - start
                data["time"].append(t)

                data["x_ref"].append(x_target)
                data["y_ref"].append(y_target)
                data["z_ref"].append(z_target)

                data["x_meas"].append(pose["x"])
                data["y_meas"].append(pose["y"])
                data["z_meas"].append(pose["z"])

                data["vx_cmd"].append(vx)
                data["vy_cmd"].append(vy)
                data["vz_cmd"].append(vz)

            next_sample += SAMPLE_DT

        time.sleep(0.01)

    tello.send_rc_control(0, 0, 0, 0)

    csv_path = os.path.join(OUTPUT_DIR, f"{test_name}_data.csv")
    save_test_data(data, csv_path)

    metrics = calculate_multi_axis_metrics(data)
    plot_multi_axis_response(data, metrics, test_name)

    return data, metrics


# =============================================================================
# Metrics + plotting
# =============================================================================

def calculate_multi_axis_metrics(data):
    """
    Compute per-axis settling time, overshoot, steady-state error,
    plus multi-axis settling and coordination spread. :contentReference[oaicite:2]{index=2}
    """
    times = np.asarray(data["time"], dtype=float)
    metrics = {}
    axis_settling_times = []

    for axis in ["x", "y", "z"]:
        ref = np.asarray(data[f"{axis}_ref"], dtype=float)
        meas = np.asarray(data[f"{axis}_meas"], dtype=float)

        if len(times) == 0:
            metrics[f"{axis}_settling_time"] = float("inf")
            metrics[f"{axis}_steady_state_error"] = float("nan")
            metrics[f"{axis}_overshoot"] = 0.0
            axis_settling_times.append(float("inf"))
            continue

        target = ref[0]
        error = ref - meas

        band = 0.05 * abs(target) if abs(target) > 0.1 else 0.05

        within_band = np.abs(error) <= band
        settling_idx = None
        for k in range(len(times)):
            if np.all(within_band[k:]):
                settling_idx = k
                break

        if settling_idx is None:
            settling_time = float("inf")
            start_ss = int(0.8 * len(times))
        else:
            settling_time = float(times[settling_idx])
            start_ss = settling_idx

        axis_settling_times.append(settling_time)

        if start_ss < len(error):
            ss_error = float(np.mean(np.abs(error[start_ss:])))
        else:
            ss_error = 0.0

        initial = meas[0]
        step = target - initial
        if abs(step) < 1e-6:
            overshoot_pct = 0.0
        else:
            if step > 0:
                peak = float(np.max(meas))
                overshoot = max(0.0, peak - target)
            else:
                trough = float(np.min(meas))
                overshoot = max(0.0, target - trough)
            overshoot_pct = float(overshoot / abs(step) * 100.0)

        metrics[f"{axis}_settling_time"] = settling_time
        metrics[f"{axis}_steady_state_error"] = ss_error
        metrics[f"{axis}_overshoot"] = overshoot_pct

    finite = [t for t in axis_settling_times if np.isfinite(t)]
    multi_axis_settling = float(max(finite)) if finite else float("inf")
    if len(finite) >= 2:
        coordination_spread = float(max(finite) - min(finite))
    else:
        coordination_spread = 0.0

    metrics["multi_axis_settling"] = multi_axis_settling
    metrics["coordination_spread"] = coordination_spread

    return metrics


def plot_multi_axis_response(data, metrics, test_name):
    """Plot position + control for x, y, z and save figure."""
    ensure_dir(OUTPUT_DIR)

    t = np.asarray(data["time"], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, axis in enumerate(["x", "y", "z"]):
        ref = np.asarray(data[f"{axis}_ref"], dtype=float)
        meas = np.asarray(data[f"{axis}_meas"], dtype=float)
        vel = np.asarray(data[f"v{axis}_cmd"], dtype=float)

        ax_pos = axes[i, 0]
        ax_pos.plot(t, meas, "b-", linewidth=2, label=f"{axis} measured")
        ax_pos.plot(t, ref, "r--", linewidth=2, label=f"{axis} reference")
        ax_pos.set_ylabel(f"{axis.upper()} [m]")
        ax_pos.set_title(f"{axis.upper()}-axis Tracking")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend()

        ax_vel = axes[i, 1]
        ax_vel.plot(t, vel, "g-", linewidth=1.5)
        ax_vel.axhline(0, color="k", linestyle="-", alpha=0.5)
        ax_vel.set_ylabel(f"v_{axis} [m/s]")
        ax_vel.set_title(f"{axis.upper()}-axis Control")
        ax_vel.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    txt = (
        f"Multi-axis settling: {metrics['multi_axis_settling']:.2f} s\n"
        f"Coordination spread: {metrics['coordination_spread']:.2f} s"
    )
    fig.text(
        0.5,
        0.02,
        txt,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    fig_path = os.path.join(OUTPUT_DIR, f"{test_name}_response.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_test_data(data, filename):
    """Save logged data to CSV."""
    ensure_dir(os.path.dirname(filename))
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Lab 2 - Phase 2: Multi-Axis PID Control")
    print("=" * 60)

    ensure_dir(OUTPUT_DIR)

    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    tello.streamon()

    detector = Detector(families=TAG_FAMILY)

    try:
        battery = tello.get_battery()
        print(f"Battery: {battery}%")
        if battery < 30:
            print("Battery too low!")
            return

        print("Taking off...")
        tello.takeoff()
        time.sleep(3.0)

        # Capture initial yaw as reference (assume drone is facing tag)
        yaw_ref = tello.get_yaw()
        print(f"Reference yaw: {yaw_ref:.1f} deg")

        print("\nTesting Multi-Axis PID Controller")

        # >>> Put your current gains here for each tuning flight <<<
        controller = MultiAxisPIDController(
            kp_x=0.3, ki_x=0.0, kd_x=0.0,
            kp_y=0.0, ki_y=0.0, kd_y=0.0,
            kp_z=0.0, ki_z=0.0, kd_z=0.0,
            dt=SAMPLE_DT,
        )

        data, metrics = test_step_response(tello, detector, controller, yaw_ref, "test_initial")

        print("\nPer-axis metrics:")
        for axis in ["x", "y", "z"]:
            print(
                f"  {axis.upper()}-axis: "
                f"settling={metrics[f'{axis}_settling_time']:.2f}s, "
                f"overshoot={metrics[f'{axis}_overshoot']:.1f}%, "
                f"ss_error={metrics[f'{axis}_steady_state_error']:.3f}m"
            )

        print(f"\nMulti-axis settling: {metrics['multi_axis_settling']:.2f}s")
        print(f"Coordination spread: {metrics['coordination_spread']:.2f}s")

        print("\nNavy Requirements Check:")
        settling_ok = metrics["multi_axis_settling"] <= SETTLING_TIME_REQ
        overshoot_ok = all(
            metrics[f"{ax}_overshoot"] < OVERSHOOT_REQ for ax in ["x", "y", "z"]
        )
        ss_error_ok = all(
            metrics[f"{ax}_steady_state_error"] < STEADY_STATE_REQ
            for ax in ["x", "y", "z"]
        )

        print(f"  Settling time <= {SETTLING_TIME_REQ}s: {'PASS' if settling_ok else 'FAIL'}")
        print(f"  Overshoot < {OVERSHOOT_REQ}%:       {'PASS' if overshoot_ok else 'FAIL'}")
        print(f"  SS error < {STEADY_STATE_REQ}m:     {'PASS' if ss_error_ok else 'FAIL'}")

        results = {
            "gains": {
                "kp_x": controller.pid_x.kp, "ki_x": controller.pid_x.ki, "kd_x": controller.pid_x.kd,
                "kp_y": controller.pid_y.kp, "ki_y": controller.pid_y.ki, "kd_y": controller.pid_y.kd,
                "kp_z": controller.pid_z.kp, "ki_z": controller.pid_z.ki, "kd_z": controller.pid_z.kd,
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        json_path = os.path.join(OUTPUT_DIR, "phase2_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {json_path}")

        print("\nLanding...")
        tello.land()
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt, landing...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception:
            pass
    except Exception as e:
        print(f"Error: {e}")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception:
            pass
    finally:
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
