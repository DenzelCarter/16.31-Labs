#!/usr/bin/env python3
"""
Lab 1 - Phase 2: Step Response Calibration

This script performs systematic step response testing to calibrate the drone's
altitude sensors and understand system dynamics. It commands specific vertical
velocities and measures the resulting altitude changes to build calibration models.
"""

import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello

# Configuration
OUTPUT_DIR = "Lab1-Phase2"
SAMPLE_DT = 0.05  # 20 Hz sampling rate

# Step velocity sequence: (velocity_cmd [m/s], duration [s])
STEP_SEQUENCE = [
    (0.0, 3.0),   # Initial hold
    (0.2, 3.5),   # Gentle up
    (0.0, 3.0),   # Hold
    (-0.15, 4.0), # Gentle down
    (0.0, 3.0),   # Hold
    (0.3, 3.0),   # Faster up
    (0.0, 3.0),   # Hold
    (-0.25, 3.0), # Faster down
    (0.0, 3.0),   # Final hold
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def velocity_to_rc(velocity_mps):
    """Convert velocity in m/s to RC throttle command (-100 to +100)"""
    rc_cmd = int(velocity_mps * 100)
    return max(-100, min(100, rc_cmd))

# ============================================================================
# PROVIDED: Sensor reading function (reuse from Phase 1)
# ============================================================================
def get_sensor_data(tello):
    """
    Get sensor data from Tello using direct API methods.
    Returns values in SI units with proper error handling.
    """
    try:
        # Use direct API methods for reliable measurements
        height_sonar = tello.get_distance_tof() / 100.0  # cm -> m
        height_baro = tello.get_barometer() / 100.0       # cm -> m  
        velocity_z = tello.get_speed_z() / 100.0          # cm/s -> m/s
        battery = tello.get_battery()                     # already in %
        
        # For acceleration, still need to use state dict as no direct methods
        state = tello.get_current_state()
        def safe_get(key, default=0.0):
            try:
                val = state.get(key, default)
                if val is None:
                    return default
                return float(val) / 100.0  # cm/s^2 -> m/s^2
            except (ValueError, TypeError):
                return default
        
        ax = safe_get('agx', 0.0)
        ay = safe_get('agy', 0.0)
        az = safe_get('agz', 0.0)
        
        return {
            'height_sonar': height_sonar,
            'height_baro': height_baro,
            'velocity_z': velocity_z,
            'ax': ax,
            'ay': ay, 
            'az': az,
            'battery': battery
        }
        
    except Exception as e:
        print(f"Error reading sensors: {e}")
        return {
            'height_sonar': 0.0,
            'height_baro': 0.0,
            'velocity_z': 0.0,
            'ax': 0.0,
            'ay': 0.0,
            'az': 0.0,
            'battery': 0.0
        }

# ============================================================================
# PROVIDED: Data collection routine
# Study this to understand step response testing methodology
# ============================================================================
def run_step_calibration():
    """Execute the step response calibration test"""
    ensure_dir(OUTPUT_DIR)
    
    # Initialize drone
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    
    try:
        # Check battery
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 40:
            print("ERROR: Battery too low for calibration. Need at least 40%")
            return None
        
        print("Taking off...")
        tello.takeoff()
        time.sleep(3)  # Allow stabilization
        
        # Get initial readings
        initial_data = get_sensor_data(tello)
        initial_altitude = initial_data['height_sonar']
        if initial_altitude < 0.1:  # If sonar seems invalid
            initial_altitude = initial_data['height_baro']
        
        print(f"Initial altitude: {initial_altitude:.2f}m")
        print(f"Starting step response calibration...")
        
        # Calculate total test duration
        total_duration = sum(duration for _, duration in STEP_SEQUENCE)
        print(f"Test sequence duration: {total_duration:.1f} seconds")
        
        # Start mission and open CSV
        mission_start_time = time.time()
        csv_filename = os.path.join(OUTPUT_DIR, "step_calibration_data.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            # Define exact CSV columns
            fieldnames = [
                'mission_time', 'step_index', 'velocity_cmd', 'step_time_remaining',
                'height_sonar', 'height_baro', 'velocity_z',
                'ax', 'ay', 'az', 'battery'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Execute step sequence
            next_sample_time = mission_start_time
            samples_collected = 0
            
            for step_index, (velocity_cmd, duration) in enumerate(STEP_SEQUENCE):
                print(f"\nStep {step_index + 1}/{len(STEP_SEQUENCE)}: {velocity_cmd:+.1f} m/s for {duration:.1f}s")
                
                step_start_time = time.time()
                step_end_time = step_start_time + duration
                rc_cmd = velocity_to_rc(velocity_cmd)
                
                # Execute this step
                while time.time() < step_end_time:
                    current_time = time.time()
                    
                    # Send RC command continuously
                    tello.send_rc_control(0, 0, rc_cmd, 0)
                    
                    # Sample at regular intervals
                    if current_time >= next_sample_time:
                        mission_time = current_time - mission_start_time
                        step_time_remaining = step_end_time - current_time
                        
                        sensor_data = get_sensor_data(tello)
                        
                        # Create row with EXACT fieldnames
                        row = {
                            'mission_time': mission_time,
                            'step_index': step_index,
                            'velocity_cmd': velocity_cmd,
                            'step_time_remaining': step_time_remaining,
                            'height_sonar': sensor_data['height_sonar'],
                            'height_baro': sensor_data['height_baro'],
                            'velocity_z': sensor_data['velocity_z'],
                            'ax': sensor_data['ax'],
                            'ay': sensor_data['ay'],
                            'az': sensor_data['az'],
                            'battery': sensor_data['battery']
                        }
                        
                        writer.writerow(row)
                        samples_collected += 1
                        
                        # Progress update every second
                        if samples_collected % 20 == 0:
                            current_alt = sensor_data['height_sonar'] if sensor_data['height_sonar'] > 0.1 else sensor_data['height_baro']
                            print(f"  t={mission_time:.1f}s: Alt={current_alt:.2f}m, Bat={sensor_data['battery']:.0f}%")
                        
                        next_sample_time += SAMPLE_DT
                    
                    time.sleep(0.01)
                
                # Stop movement between steps
                tello.send_rc_control(0, 0, 0, 0)
        
        # Stop all motion
        tello.send_rc_control(0, 0, 0, 0)
        print("Step sequence complete!")
        
        # Land
        print("Landing...")
        tello.land()
        time.sleep(2)
        
        print(f"Data saved to {csv_filename}")
        print(f"Total samples collected: {samples_collected}")
        
        return csv_filename
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except:
            pass
        return None
    finally:
        try:
            tello.end()
        except:
            pass
        print("Tello connection closed")

# ============================================================================
# TODO: Implement analysis functions
# This is where you analyze the step response data to build calibration models
# ============================================================================

def analyze_step_data(csv_filename):
    """Analyze the step response data and generate calibration results"""
    
    print(f"\nAnalyzing step calibration data from {csv_filename}...")
    
    # Local import keeps pandas optional until analysis time (collection can run w/o it).
    import pandas as pd
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # ----- Extract columns we need as NumPy arrays for speed/convenience -----
    mission_time     = df['mission_time'].values
    step_indices     = df['step_index'].values
    velocity_cmds    = df['velocity_cmd'].values
    height_sonar     = df['height_sonar'].values
    height_baro      = df['height_baro'].values
    velocity_measured= df['velocity_z'].values
    
    ensure_dir(OUTPUT_DIR)
    
    # ==================== PROVIDED: PLOTTING ====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Commanded vs measured velocity
    ax1.plot(mission_time, velocity_cmds, 'r-', linewidth=2, label='Commanded vz [m/s]')
    ax1.plot(mission_time, velocity_measured, 'b-', linewidth=1, alpha=0.8, label='Measured vz [m/s]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_title('Velocity Command vs Measured Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Height response
    ax2.plot(mission_time, height_sonar, 'g-', linewidth=1.5, label='Sonar Height [m]', alpha=0.8)
    ax2.plot(mission_time, height_baro, 'm-', linewidth=1.5, label='Barometer Height [m]', alpha=0.8)
    ax2.set_xlabel('Mission Time [s]')
    ax2.set_ylabel('Height [m]')
    ax2.set_title('Height Response from Both Sensors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'step_response.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # ==================== STEP-BY-STEP ANALYSIS ====================
    print(f"\nStep-by-step analysis:")
    
    calibration_table = []  # we will store per-step summaries here (later saved to CSV)
    
    # We iterate over each unique step in the same order they were executed.
    unique_steps = sorted(df['step_index'].unique())
    
    # For robustness we use medians of the first/last few samples in each step
    # to estimate pre/post heights; this reduces the impact of noise/outliers.
    PRE_N  = 10  # number of samples from the start of a step for "pre" statistic
    POST_N = 10  # number of samples from the end of a step for "post" statistic
    
    for step_idx in unique_steps:
        # Select the rows belonging to this step.
        step_df = df[df['step_index'] == step_idx]
        n = len(step_df)
        if n < max(PRE_N, POST_N):
            # Skip very short steps; we can't form reliable pre/post medians.
            print(f"  - Step {step_idx}: skipped (too few samples: {n})")
            continue
        
        # Velocity command for this step (constant within step by construction).
        v_cmd = float(step_df['velocity_cmd'].iloc[0])
        
        # Duration estimate: either use count * SAMPLE_DT (nominal) or true time span.
        # We prefer the measured mission_time span for better accuracy.
        t0 = float(step_df['mission_time'].iloc[0])
        t1 = float(step_df['mission_time'].iloc[-1])
        duration_s = max(0.0, t1 - t0)
        if duration_s == 0.0:
            # Fallback to nominal if timestamps are degenerate.
            duration_s = n * float(SAMPLE_DT)
        
        # Expected height change from command assuming perfect tracking:
        # Δh_expected ≈ v_cmd * duration
        expected_dh = v_cmd * duration_s
        
        # Compute robust pre/post height for each sensor using medians.
        # We explicitly use the first PRE_N and last POST_N rows within the step.
        pre_sonar  = float(step_df['height_sonar'].iloc[:PRE_N].median(skipna=True))
        post_sonar = float(step_df['height_sonar'].iloc[-POST_N:].median(skipna=True))
        pre_baro   = float(step_df['height_baro'].iloc[:PRE_N].median(skipna=True))
        post_baro  = float(step_df['height_baro'].iloc[-POST_N:].median(skipna=True))
        
        # Actual height changes measured by each sensor.
        sonar_dh = post_sonar - pre_sonar
        baro_dh  = post_baro  - pre_baro
        
        # Save a concise summary row for this step. Keep everything numeric + simple types.
        calibration_table.append({
            'step_index': step_idx,
            'velocity_cmd_mps': v_cmd,
            'samples': n,
            'duration_s': duration_s,
            'expected_dh_m': expected_dh,
            'sonar_dh_m': float(sonar_dh),
            'baro_dh_m': float(baro_dh),
            'pre_sonar_m': pre_sonar,
            'post_sonar_m': post_sonar,
            'pre_baro_m': pre_baro,
            'post_baro_m': post_baro,
        })
        
        # Human-readable one-line summary for quick terminal sanity check.
        print(f"  - Step {step_idx:02d} | v_cmd={v_cmd:+.2f} m/s | dur={duration_s:4.2f} s | "
              f"Δh_exp={expected_dh:+.2f} m | Δh_sonar={sonar_dh:+.2f} m | Δh_baro={baro_dh:+.2f} m")
    
    # If the table is empty (e.g., all steps skipped), stop here gracefully.
    if not calibration_table:
        print("No usable steps found for calibration.")
        return
    
    # ==================== CALIBRATION REGRESSION ====================
    print(f"\nCalibration analysis:")
    
    # Convert table to DataFrame for convenient filtering/analysis.
    calib_df = pd.DataFrame(calibration_table)
    
    # We calibrate using **movement** steps only (exclude holds where v_cmd == 0).
    move_df = calib_df[np.abs(calib_df['velocity_cmd_mps']) > 1e-6].copy()
    if move_df.empty:
        print("No movement steps available for regression. Aborting calibration fit.")
        return
    
    # Prepare regression data:
    # x = expected height change from command; y = actual measured change.
    x = move_df['expected_dh_m'].values.astype(float)
    y_sonar = move_df['sonar_dh_m'].values.astype(float)
    y_baro  = move_df['baro_dh_m'].values.astype(float)
    
    # Helper to compute a robust linear fit and goodness metrics.
    def fit_line(x, y):
        """
        Fit y ≈ s*x + b and compute (s, b, r2, rmse).
        Handles degenerate cases (constant y, length < 2).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Require at least 2 points to fit a line.
        if x.size < 2:
            return 1.0, 0.0, 0.0, float('nan')
        
        # Fit slope and bias via least squares.
        s, b = np.polyfit(x, y, 1)
        
        # Predictions and residuals.
        y_hat = s * x + b
        resid = y - y_hat
        
        # R^2: 1 - SS_res/SS_tot (guard against zero variance).
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # RMSE of the fit residuals.
        rmse = float(np.sqrt(np.mean(resid**2)))
        return float(s), float(b), float(max(min(r2, 1.0), 0.0)), rmse
    
    # Fit separate calibration lines for sonar and barometer.
    sonar_scale, sonar_bias, sonar_r2, sonar_rmse = fit_line(x, y_sonar)
    baro_scale,  baro_bias,  baro_r2,  baro_rmse  = fit_line(x, y_baro)
    
    # Print the calibration equations and metrics for quick review.
    print(f"  Sonar:  Δh_meas ≈ {sonar_scale:.3f} * Δh_exp + {sonar_bias:+.3f}   "
          f"(R²={sonar_r2:.3f}, RMSE={sonar_rmse:.3f} m)")
    print(f"  Baro :  Δh_meas ≈ {baro_scale:.3f} * Δh_exp + {baro_bias:+.3f}   "
          f"(R²={baro_r2:.3f}, RMSE={baro_rmse:.3f} m)")
    
    # ==================== SAVE RESULTS ====================
    # 1) Save the per-step table so you can audit which steps drove the fit.
    table_path = os.path.join(OUTPUT_DIR, 'step_calibration_table.csv')
    calib_df.to_csv(table_path, index=False)
    print(f"Saved per-step table → {table_path}")
    
    # 2) Decide which sensor to recommend for altitude control.
    #    Simple rule: prefer higher R²; if nearly tied (≤0.02 apart), pick lower RMSE.
    if (baro_r2 - sonar_r2) > 0.02:
        recommended_sensor = "BAROMETER"
    elif (sonar_r2 - baro_r2) > 0.02:
        recommended_sensor = "SONAR"
    else:
        recommended_sensor = "BAROMETER" if baro_rmse < sonar_rmse else "SONAR"
    
    # 3) Save a compact JSON with the calibration results for easy reuse by Phase 3.
    results = {
        "recommended_sensor": recommended_sensor,
        "sonar_calibration": {
            "scale": sonar_scale,
            "bias": sonar_bias,
            "r2": sonar_r2,
            "rmse": sonar_rmse
        },
        "barometer_calibration": {
            "scale": baro_scale,
            "bias": baro_bias,
            "r2": baro_r2,
            "rmse": baro_rmse
        },
        # For reproducibility/context
        "notes": {
            "pre_samples": PRE_N,
            "post_samples": POST_N,
            "sample_dt_nominal": SAMPLE_DT,
            "csv_source": os.path.basename(csv_filename)
        }
    }
    results_path = os.path.join(OUTPUT_DIR, 'step_calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration results → {results_path}")
    
    # 4) Print a summary block for the lab report.
    print(f"\n{'='*60}")
    print("PHASE 2 CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Recommended sensor: {recommended_sensor}")
    print(f"Sonar  : scale={sonar_scale:.3f}, bias={sonar_bias:+.3f}, R²={sonar_r2:.3f}, RMSE={sonar_rmse:.3f} m")
    print(f"Baro   : scale={baro_scale:.3f},  bias={baro_bias:+.3f},  R²={baro_r2:.3f},  RMSE={baro_rmse:.3f} m")

def main():
    print("Phase 2: Step Response Calibration")
    print("=" * 40)
    
    # Run the step calibration test
    csv_filename = run_step_calibration()
    
    if csv_filename:
        # Analyze the collected data
        analyze_step_data(csv_filename)
        print(f"\nPhase 2 complete! Files saved in {OUTPUT_DIR}/")
        print("Next: Run heuristic_control.py for Phase 3")
    else:
        print("Calibration test failed - no data to analyze")

if __name__ == "__main__":
    main()