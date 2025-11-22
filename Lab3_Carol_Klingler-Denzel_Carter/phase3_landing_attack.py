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
     * Horizontal (x,y): Dead reckoning (no more corrections)
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
# Control / timing parameters
# ============================================================
DT = 0.1  # control + KF timestep [s]

# Horizontal control
KP_XY = 0.3
XY_TOLERANCE = 0.2  # [m]

# Vertical control
KP_Z = 0.25
Z_TOLERANCE = 0.08  # [m]
VZ_MAX = 0.25       # max vertical speed [m/s]

# Optional: max time allowed in blind prediction-only mode
MAX_BLIND_NAV_TIME = 4.0  # seconds (set to None to disable)
# ============================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def attack_resilience_mission(tello, detector, tag_size=0.125):
    """
    Execute complete mission with blind navigation capability.
    """
    print("\n" + "="*60)
    print("ATTACK RESILIENCE MISSION")
    print("="*60)
    print(f"Target: ({TARGET_POSITION[0]}, {TARGET_POSITION[1]}, {TARGET_POSITION[2]}) m")

    data = {
        "time": [],
        "phase": [],
        "z_tof_real": [],
        "z_tof_reported": [],
        "z_est": [],
        "vz_est": [],
        "vz_cmd": [],
        "x_tag": [],
        "y_tag": [],
        "x_est": [],
        "y_est": [],
        "tag_visible": [],
        "kf_mode": [],
    }

    # Kalman filter (Q, R tuned in kalman_filter.py)
    kf = AltitudeKalmanFilter(dt=DT)

    target_x, target_y, target_z = TARGET_POSITION

    # ---- Initial tag acquisition ----
    print("\n[PHASE 1] Navigating toward target using AprilTag...")
    print("  Verifying AprilTag detection...")
    tag_detected = False
    pose = None

    for attempt in range(10):
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)
        if pose["detected"]:
            print(f"  [+] Tag detected at ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}) m")
            tag_detected = True
            break
        else:
            print(f"  Attempt {attempt+1}/10: No tag detected, retrying...")
            time.sleep(0.5)

    if not tag_detected:
        print("\n[-] ERROR: Cannot detect AprilTag after multiple attempts.")
        return None

    # Initialize KF with current altitude
    z_initial = tello.get_distance_tof() / 100.0
    kf.initialize(z_initial, 0.0)

    # Mission state
    mission_start = time.time()
    attack_triggered = False
    attack_start_time = None

    # Position tracking
    x_current = pose["x"]
    y_current = pose["y"]
    z_current = pose["z"]

    x_last_known = x_current
    y_last_known = y_current
    z_last_known = z_current

    x_est = x_current
    y_est = y_current

    consecutive_lost = 0
    reached_horizontal_target = False
    reached_vertical_target = False

    # For KF input: track last 2 vertical commands to approximate acceleration
    prev_vz_cmd = 0.0
    prev_prev_vz_cmd = 0.0

    print(
        f"  Initial distance to target: "
        f"{np.sqrt((target_x-x_current)**2 + (target_y-y_current)**2 + (target_z-z_current)**2):.2f} m"
    )

    # Main mission loop
    while True:
        t = time.time() - mission_start

        # Approximate vertical acceleration command from past RC commands
        a_cmd = (prev_vz_cmd - prev_prev_vz_cmd) / DT

        # Real (uncompromised) altitude
        z_tof_real = tello.get_distance_tof() / 100.0

        # AprilTag detection
        frame = tello.get_frame_read().frame
        pose = get_apriltag_pose(frame, detector, tag_size=tag_size)

        # ------------------ Sensor handling & KF ------------------
        if pose["detected"] and not attack_triggered:
            # Normal operation: tag visible
            consecutive_lost = 0
            tag_visible = True
            phase = "approach"
            kf_mode = "normal"

            # Update position from tag
            x_current = pose["x"]
            y_current = pose["y"]
            z_current = pose["z"]

            x_last_known = x_current
            y_last_known = y_current
            z_last_known = z_current

            x_est = x_current
            y_est = y_current

            z_tof_reported = z_tof_real

            # KF: predict + update
            kf.predict(u=a_cmd)
            kf.update(z_tof_real)

        else:
            # Tag lost OR already in attack
            consecutive_lost += 1
            tag_visible = False

            if consecutive_lost >= 3 and not attack_triggered:
                # Trigger attack
                attack_triggered = True
                attack_start_time = t

                dx_rem = target_x - x_last_known
                dy_rem = target_y - y_last_known
                dz_rem = target_z - z_last_known
                dist_rem = np.sqrt(dx_rem**2 + dy_rem**2 + dz_rem**2)

                print("\n  +---------------------------------------------------------+")
                print("  | TAG LOST - ATTACK INITIATED                             |")
                print(f"  | Time: {t:.1f}s                                          |")
                print(f"  | Last position: ({x_last_known:.2f}, {y_last_known:.2f}, {z_last_known:.2f}) m |")
                print(f"  | Distance to target: {dist_rem:.2f} m                    |")
                print("  | BLIND NAVIGATION MODE:                                  |")
                print("  |  - ToF sensor: +0.5m bias injected                      |")
                print("  |  - AprilTag: LOST (cannot reacquire)                   |")
                print("  |  - Horizontal nav: Dead reckoning (hold last x,y)      |")
                print("  |  - Vertical nav: KF prediction-only                    |")
                print("  +---------------------------------------------------------+")
                kf.set_mode(prediction_only=True)
                # Assume horizontal goal is essentially reached; freeze xy commands
                reached_horizontal_target = True

            if attack_triggered:
                phase = "blind_nav"
                kf_mode = "prediction"
                z_tof_reported = z_tof_real + 0.5  # compromised sensor (only for logging)
                # KF prediction-only with acceleration input
                kf.predict(u=a_cmd)
                # no update during attack
            else:
                # Tag lost briefly but attack not yet triggered -> still trust sensors
                phase = "search"
                kf_mode = "normal"
                z_tof_reported = z_tof_real
                kf.predict(u=a_cmd)
                kf.update(z_tof_real)

        # KF state after prediction/update
        z_est, vz_est = kf.get_state()

        # ------------------ Target checks ------------------
        if not reached_horizontal_target:
            dx = target_x - x_est
            dy = target_y - y_est
            horizontal_error = np.sqrt(dx**2 + dy**2)
            if horizontal_error < XY_TOLERANCE:
                reached_horizontal_target = True
                print(f"\n  [+] Reached horizontal target region "
                      f"(err ≈ {horizontal_error:.2f} m)")

        if reached_horizontal_target and not reached_vertical_target:
            dz = target_z - z_est
            if abs(dz) < Z_TOLERANCE:
                reached_vertical_target = True
                print(f"\n  [+] Reached target altitude (KF): {target_z:.2f} m")
                print(f"      Real altitude: {z_tof_real:.2f} m")
                print(f"      KF estimate:   {z_est:.2f} m")
                print(f"      Estimation error: {abs(z_tof_real - z_est)*100:.1f} cm")
                if attack_triggered:
                    blind_time = t - attack_start_time
                    print(f"      Time under attack: {blind_time:.1f} s")
                print("\n  [+] Mission complete - commanding land")
                break

        # ------------------ Control commands ------------------
        # Horizontal velocities (only before attack / before horizontal target)
        if not reached_horizontal_target and not attack_triggered:
            dx = target_x - x_est
            dy = target_y - y_est
            vx_cmd = np.clip(KP_XY * dx, -0.2, 0.2)
            vy_cmd = np.clip(KP_XY * dy, -0.2, 0.2)
        else:
            vx_cmd = 0.0
            vy_cmd = 0.0

        # Vertical velocity
        if not reached_vertical_target:
            dz = target_z - z_est

            # Optional safety: limit time in blind prediction-only mode
            if attack_triggered and MAX_BLIND_NAV_TIME is not None:
                blind_time = t - attack_start_time
                if blind_time > MAX_BLIND_NAV_TIME:
                    # Stop climbing in prediction-only beyond this time
                    vz_cmd = 0.0
                else:
                    vz_cmd = np.clip(KP_Z * dz, -VZ_MAX, VZ_MAX)
            else:
                vz_cmd = np.clip(KP_Z * dz, -VZ_MAX, VZ_MAX)
        else:
            vz_cmd = 0.0

        # Map velocities to RC commands
        rc_lr, rc_fb, rc_ud, rc_yaw = tag_to_drone_velocity(vx_cmd, vy_cmd, vz_cmd)
        tello.send_rc_control(rc_lr, rc_fb, rc_ud, rc_yaw)

        # ------------------ Logging ------------------
        data["time"].append(t)
        data["phase"].append(phase)
        data["z_tof_real"].append(z_tof_real)
        data["z_tof_reported"].append(z_tof_reported)
        data["z_est"].append(z_est)
        data["vz_est"].append(vz_est)
        data["vz_cmd"].append(vz_cmd)
        data["x_tag"].append(x_current if tag_visible else np.nan)
        data["y_tag"].append(y_current if tag_visible else np.nan)
        data["x_est"].append(x_est)
        data["y_est"].append(y_est)
        data["tag_visible"].append(tag_visible)
        data["kf_mode"].append(kf_mode)

        if len(data["time"]) % 20 == 0:
            err_cm = abs(z_tof_real - z_est) * 100
            mode_str = "BLIND" if attack_triggered else "NORMAL"
            status = "ASCEND" if reached_horizontal_target else "APPROACH"
            print(
                f"  t={t:.1f}s [{mode_str:6s}] {status:8s}: "
                f"z_real={z_tof_real:.2f}m, z_est={z_est:.2f}m, "
                f"err={err_cm:.1f}cm, vz_cmd={vz_cmd:.2f}m/s"
            )

        # Roll command history for next acceleration estimate
        prev_prev_vz_cmd = prev_vz_cmd
        prev_vz_cmd = vz_cmd

        # Safety timeout
        if t > 90.0:
            print("\n  [!] Safety timeout (90s) – aborting mission loop.")
            break

        time.sleep(DT)

    # Stop and land
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    print("\n[LANDING]")
    tello.land()
    time.sleep(2.0)

    return data


def plot_mission_analysis(data):
    """Generate mission analysis plots."""
    ensure_dir(OUTPUT_DIR)
    times = np.array(data["time"])
    errors = np.abs(
        np.array(data["z_tof_real"]) - np.array(data["z_est"])
    ) * 100.0  # cm

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Attack indices
    attack_indices = [i for i, ph in enumerate(data["phase"]) if ph == "blind_nav"]
    tag_visible = np.array(data["tag_visible"])

    # Altitude profile
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, data["z_tof_real"], "b-", lw=2, label="Real altitude")
    ax1.plot(times, data["z_est"], "g--", lw=2, label="KF estimate")

    if attack_indices:
        attack_times = times[attack_indices]
        reported = np.array(data["z_tof_reported"])[attack_indices]
        ax1.plot(
            attack_times,
            reported,
            "r.",
            ms=3,
            alpha=0.4,
            label="Compromised sensor",
        )
        ax1.axvspan(
            attack_times[0],
            attack_times[-1],
            color="red",
            alpha=0.15,
            label="Attack window",
        )

    ax1.axhline(
        TARGET_POSITION[2],
        color="purple",
        ls=":",
        lw=2,
        label=f"Target alt ({TARGET_POSITION[2]} m)",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Altitude [m]")
    ax1.set_title("Altitude Profile")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Altitude error
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(times, errors, "r-", lw=2)
    ax2.axhline(15, color="k", ls="--", lw=1.5, label="15cm threshold")

    if attack_indices:
        atk_err = errors[attack_indices]
        ax2.text(
            0.05,
            0.95,
            f"Attack Phase:\nMax: {np.max(atk_err):.1f}cm\nMean: {np.mean(atk_err):.1f}cm",
            transform=ax2.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [cm]")
    ax2.set_title("Altitude Estimation Error")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Horizontal trajectory (x-y)
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(len(times) - 1):
        color = "blue" if tag_visible[i] else "red"
        ax3.plot(
            data["x_est"][i : i + 2],
            data["y_est"][i : i + 2],
            color=color,
            lw=2,
            alpha=0.7,
        )

    ax3.plot(
        data["x_est"][0],
        data["y_est"][0],
        "go",
        ms=10,
        label="Start",
        markeredgecolor="darkgreen",
    )
    ax3.plot(
        TARGET_POSITION[0],
        TARGET_POSITION[1],
        "r*",
        ms=18,
        label="Target",
        markeredgecolor="darkred",
    )
    ax3.plot(
        data["x_est"][-1],
        data["y_est"][-1],
        "bs",
        ms=9,
        label="End",
        markeredgecolor="darkblue",
    )
    ax3.set_xlabel("x [m] (forward)")
    ax3.set_ylabel("y [m] (lateral)")
    ax3.set_title("Horizontal Trajectory (x-y)")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axis("equal")

    # Vertical trajectory (x-z)
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(len(times) - 1):
        color = "blue" if tag_visible[i] else "red"
        ax4.plot(
            data["x_est"][i : i + 2],
            data["z_est"][i : i + 2],
            color=color,
            lw=2,
            alpha=0.7,
        )

    ax4.plot(
        data["x_est"][0],
        data["z_est"][0],
        "go",
        ms=10,
        label="Start",
        markeredgecolor="darkgreen",
    )
    ax4.plot(
        TARGET_POSITION[0],
        TARGET_POSITION[2],
        "r*",
        ms=18,
        label="Target",
        markeredgecolor="darkred",
    )
    ax4.plot(
        data["x_est"][-1],
        data["z_est"][-1],
        "bs",
        ms=9,
        label="End",
        markeredgecolor="darkblue",
    )
    ax4.set_xlabel("x [m] (forward)")
    ax4.set_ylabel("z [m] (altitude)")
    ax4.set_title("Vertical Trajectory (x-z)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # AprilTag visibility
    ax5 = fig.add_subplot(gs[1, 2])
    vis = tag_visible.astype(float)
    ax5.fill_between(times, 0, vis, color="green", alpha=0.3, label="Tag visible")
    ax5.fill_between(times, vis, 1, color="red", alpha=0.3, label="Tag lost")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Status")
    ax5.set_title("AprilTag Visibility")
    ax5.set_ylim([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(["Lost", "Visible"])
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Horizontal position vs time
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(times, data["x_est"], "b-", lw=2, label="x position")
    ax6.axhline(
        TARGET_POSITION[0],
        color="blue",
        ls="--",
        lw=1.5,
        label=f"Target x ({TARGET_POSITION[0]} m)",
    )
    if attack_indices:
        ax6.axvspan(
            times[attack_indices[0]],
            times[attack_indices[-1]],
            color="red",
            alpha=0.15,
            label="Blind navigation",
        )
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Position [m]")
    ax6.set_title("Horizontal Position (Dead Reckoning)")
    ax6.legend()
    ax6.grid(alpha=0.3)

    # Mission summary panel
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    if attack_indices:
        atk_err = errors[attack_indices]
        max_err = np.max(atk_err)
        mean_err = np.mean(atk_err)
        attack_time = times[attack_indices[-1]] - times[attack_indices[0]]
        tag_loss_time = times[attack_indices[0]]
    else:
        max_err = mean_err = attack_time = tag_loss_time = 0.0

    final_altitude_err = abs(data["z_tof_real"][-1] - TARGET_POSITION[2]) * 100.0
    total_time = times[-1]
    final_x = data["x_est"][-1]
    final_y = data["y_est"][-1]
    final_z = data["z_tof_real"][-1]

    summary = f"""MISSION SUMMARY
==============================

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
  z: {final_z:.2f} m (target: {TARGET_POSITION[2]:.1f} m)

BLIND NAVIGATION:
  Horizontal: Dead reckoning (hold)
  Vertical:   KF prediction-only
  Success: {'YES' if max_err < 15 else 'NO'}
"""

    ax7.text(
        0.05,
        0.95,
        summary,
        transform=ax7.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.suptitle("Attack Resilience Mission Analysis", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(OUTPUT_DIR, "mission_analysis.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Plots saved: {OUTPUT_DIR}/mission_analysis.png")


def generate_report(data):
    """Generate text report."""
    ensure_dir(OUTPUT_DIR)
    times = np.array(data["time"])
    errors = np.abs(
        np.array(data["z_tof_real"]) - np.array(data["z_est"])
    ) * 100.0  # cm
    attack_indices = [i for i, ph in enumerate(data["phase"]) if ph == "blind_nav"]

    if attack_indices:
        atk_err = errors[attack_indices]
        max_err = np.max(atk_err)
        mean_err = np.mean(atk_err)
        attack_time = times[attack_indices[-1]] - times[attack_indices[0]]
        tag_loss_time = times[attack_indices[0]]
    else:
        max_err = mean_err = attack_time = tag_loss_time = 0.0

    final_altitude_err = abs(data["z_tof_real"][-1] - TARGET_POSITION[2]) * 100.0

    report = f"""
============================================================
ATTACK RESILIENCE MISSION REPORT
============================================================

MISSION PARAMETERS:
  Target position: ({TARGET_POSITION[0]}, {TARGET_POSITION[1]}, {TARGET_POSITION[2]}) m

TIMELINE:
  Total mission time: {times[-1]:.1f} s
  AprilTag lost at:   {tag_loss_time:.1f} s
  Blind navigation:   {attack_time:.1f} s

PERFORMANCE:

Altitude (during attack):
  Max estimation error:  {max_err:.1f} cm
  Mean estimation error: {mean_err:.1f} cm
  Threshold:             15 cm
  Status: {'[+] PASS' if max_err < 15 else '[-] FAIL'}

Final altitude:
  Target: {TARGET_POSITION[2]:.2f} m
  Actual: {data['z_tof_real'][-1]:.2f} m
  Error:  {final_altitude_err:.1f} cm

Navigation strategy:
  Normal: AprilTag + ToF + KF updates
  Attack: Dead reckoning (x,y) + KF prediction-only (z)
  Sensor compromise: +0.5 m bias on ToF

Overall result: {'[+] MISSION SUCCESS' if max_err < 15 else '[-] MISSION DEGRADED'}

============================================================
"""

    with open(os.path.join(OUTPUT_DIR, "mission_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"[+] Report saved: {OUTPUT_DIR}/mission_report.txt")


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
        families="tag36h11",
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    TAG_SIZE = 0.125  # [m]

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

        if data is not None:
            import pandas as pd

            df = pd.DataFrame(data)
            csv_path = os.path.join(OUTPUT_DIR, "attack_mission.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n[+] Data saved: {csv_path}")

            print("\nGenerating plots...")
            plot_mission_analysis(data)

            print("\nGenerating report...")
            generate_report(data)

            print("\n" + "="*60)
            print("MISSION COMPLETE")
            print("="*60)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        tello.send_rc_control(0, 0, 0, 0)
        try:
            tello.land()
        except Exception:
            pass
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        tello.send_rc_control(0, 0, 0, 0)
        try:
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
