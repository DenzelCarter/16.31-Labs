#!/usr/bin/env python3
"""
Lab 1 - Phase 1: Analyze Sensor Data (CSV Version)

This script analyzes the sensor data collected from the continuous mission
in recon_sensor_data.py. It reads from CSV format and focuses on basic 
sensor characterization: noise, drift, reliability, and sampling rate verification.

Tasks:
- Load CSV data with global mission timeline
- Plot all 5 sensor types with proper labels and legends
- Analyze noise characteristics across different altitudes
- Check for drift or bias in sensor readings
- Compare barometer vs sonar height sensor performance
- Verify sampling rate and data quality
- Provide educational commentary on sensor behavior
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = "Lab1-Phase1"
OUTPUT_DIR = "Lab1-Phase1"
DATA_FILE = os.path.join(INPUT_DIR, "sensor_data.csv")

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

# ============================================================================
# TODO: Implement these analysis functions
# These are the core learning objectives for Phase 1
# ============================================================================

def calculate_noise_level(signal, detrend_window=None):
    """
    Estimate noise level by calculating standard deviation after detrending.
    
    TODO: Implement noise calculation
    1. Remove NaN values from signal using np.isfinite()
    2. If signal is too short (<10 samples), just return np.std()
    3. Otherwise, remove trend using moving average:
       - Use np.convolve() with a rectangular window
       - Window size = detrend_window (or auto: len(signal)//10)
       - Pad signal edges to avoid boundary effects
    4. Calculate detrended signal = signal - trend
    5. Return standard deviation of detrended signal
    
    Hint: Moving average removes slow drifts, leaving only noise
    
    Args:
        signal: 1D array of sensor readings
        detrend_window: samples for moving average (None = auto)
    Returns:
        noise_std: standard deviation of detrended signal
    """
    # TODO: Implement noise calculation
    return 0.0

def calculate_drift_rate(timestamps, signal):
    """
    Calculate linear drift rate using least squares fit.
    
    TODO: Implement drift calculation
    1. Remove NaN values from both timestamps and signal
    2. Use np.polyfit(timestamps, signal, 1) to fit a line
       - Returns [slope, intercept]
       - Slope is the drift rate (signal units per second)
    3. Calculate R-squared to assess fit quality:
       - ss_res = sum of squared residuals
       - ss_tot = total sum of squares
       - R² = 1 - (ss_res / ss_tot)
    4. Return drift_rate and r_squared
    
    Hint: High R² (>0.7) means signal has significant linear drift
    
    Args:
        timestamps: time array in seconds
        signal: sensor readings array
    Returns:
        drift_rate: slope in signal units per second
        r_squared: coefficient of determination (0-1)
    """
    # TODO: Implement drift calculation
    return 0.0, 0.0

def analyze_sampling_rate(timestamps):
    """
    Analyze sampling rate consistency and statistics.
    
    TODO: Implement sampling rate analysis
    1. Calculate time intervals: dt = np.diff(timestamps)
    2. Remove outliers (gaps > 1 second during altitude transitions)
    3. Calculate statistics on valid intervals:
       - mean_rate = 1.0 / mean(dt)
       - rate_std = std deviation of (1.0 / dt)
       - rate_min = 1.0 / max(dt)
       - rate_max = 1.0 / min(dt)
    4. Return all statistics
    
    Hint: Target is 10 Hz (0.1s intervals)
    
    Returns:
        mean_rate: average sampling rate in Hz
        rate_std: standard deviation of sampling rate
        rate_min, rate_max: min/max instantaneous rates
    """
    # TODO: Implement sampling rate analysis
    return 10.0, 0.0, 10.0, 10.0

def compare_height_sensors(height_sonar, height_baro, target_altitudes):
    """
    Compare sonar and barometer height sensor performance.
    
    TODO: Implement sensor comparison
    1. Calculate bias (systematic offset):
       - Find samples where both sensors are valid
       - bias = mean(baro - sonar) for valid samples
    2. Calculate correlation coefficient:
       - Use np.corrcoef(sonar, baro) on valid samples
       - Returns 2x2 matrix, use [0,1] element
    3. Calculate dropout rates:
       - Fraction of samples with NaN/invalid readings
    4. Calculate RMSE vs target altitudes:
       - RMSE = sqrt(mean((sensor - target)²))
       - Do this for both sonar and baro
    5. Return all metrics
    
    Hint: Lower RMSE = better tracking accuracy
    Hint: High correlation (>0.9) = sensors agree well
    
    Returns:
        bias: mean difference (baro - sonar) in meters
        correlation: correlation coefficient (0-1)
        sonar_dropout_rate: fraction of invalid sonar readings
        baro_dropout_rate: fraction of invalid baro readings
        sonar_rmse: RMS error vs targets for sonar
        baro_rmse: RMS error vs targets for barometer
    """
    # TODO: Implement sensor comparison
    return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

# ============================================================================
# PROVIDED: Data loading and plotting code
# Study this to understand how to work with the data
# ============================================================================

def main():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        print("Run recon_sensor_data.py first to collect data.")
        return
    
    print("Loading Phase 1 sensor data from CSV...")
    
    try:
        # Load CSV data
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} samples from CSV")
        print(f"Columns: {list(df.columns)}")
        
        # Extract data arrays
        timestamps = df['mission_time'].values
        target_altitudes = df['target_altitude'].values
        
        # 3-axis sensor data
        acceleration = np.column_stack([df['ax'].values, df['ay'].values, df['az'].values])
        velocity = np.column_stack([df['vx'].values, df['vy'].values, df['vz'].values])
        attitude = np.column_stack([df['roll'].values, df['pitch'].values, df['yaw'].values])
        
        # Height sensors
        height_sonar = df['height_sonar'].values
        height_baro = df['height_baro'].values
        
        # Get unique altitude setpoints
        altitude_setpoints = sorted(df['target_altitude'].unique())
        
        print(f"Mission duration: {timestamps[0]:.1f} to {timestamps[-1]:.1f} seconds")
        print(f"Altitude setpoints: {altitude_setpoints}")
        
        # DEBUG: Check data quality
        print(f"\nDATA QUALITY CHECK:")
        print(f"Timestamp range: {timestamps[0]:.2f} - {timestamps[-1]:.2f} s")
        print(f"Sonar height range: {np.nanmin(height_sonar):.3f} - {np.nanmax(height_sonar):.3f} m")
        print(f"Baro height range: {np.nanmin(height_baro):.3f} - {np.nanmax(height_baro):.3f} m")
        print(f"Acceleration ranges: ax=[{np.nanmin(acceleration[:,0]):.2f}, {np.nanmax(acceleration[:,0]):.2f}], "
              f"ay=[{np.nanmin(acceleration[:,1]):.2f}, {np.nanmax(acceleration[:,1]):.2f}], "
              f"az=[{np.nanmin(acceleration[:,2]):.2f}, {np.nanmax(acceleration[:,2]):.2f}]")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    ensure_dir(OUTPUT_DIR)
    
    # Define colors for altitude setpoints
    colors = ['purple', 'orange', 'brown', 'pink', 'cyan']
    
    # ==================== PLOT 1: ACCELERATION ====================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, acceleration[:, 0], 'r-', label='ax [m/s²]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, acceleration[:, 1], 'g-', label='ay [m/s²]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, acceleration[:, 2], 'b-', label='az [m/s²]', linewidth=1.5, alpha=0.8)
    
    # Add altitude setpoint regions
    for i, alt in enumerate(altitude_setpoints):
        mask = np.abs(target_altitudes - alt) < 0.05
        if np.any(mask):
            t_segment = timestamps[mask]
            t_start, t_end = t_segment[0], t_segment[-1]
            color = colors[i % len(colors)]
            ax.axvspan(t_start, t_end, alpha=0.1, color=color, label=f'{alt}m altitude')
    
    ax.set_xlabel('Mission Time [s]')
    ax.set_ylabel('Acceleration [m/s²]')
    ax.set_title('3-Axis Acceleration Data Across Mission')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'acceleration_xyz.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== PLOT 2: VELOCITY ====================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, velocity[:, 0], 'r-', label='vx [m/s]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, velocity[:, 1], 'g-', label='vy [m/s]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, velocity[:, 2], 'b-', label='vz [m/s]', linewidth=1.5, alpha=0.8)
    
    # Add altitude setpoint regions
    for i, alt in enumerate(altitude_setpoints):
        mask = np.abs(target_altitudes - alt) < 0.05
        if np.any(mask):
            t_segment = timestamps[mask]
            t_start, t_end = t_segment[0], t_segment[-1]
            color = colors[i % len(colors)]
            ax.axvspan(t_start, t_end, alpha=0.1, color=color)
    
    ax.set_xlabel('Mission Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('3-Axis Velocity Data (Tello Estimates)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'velocity_xyz.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== PLOT 3: ATTITUDE ====================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, attitude[:, 0], 'r-', label='Roll [deg]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, attitude[:, 1], 'g-', label='Pitch [deg]', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, attitude[:, 2], 'b-', label='Yaw [deg]', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Mission Time [s]')
    ax.set_ylabel('Angle [degrees]')
    ax.set_title('Attitude Data (Roll, Pitch, Yaw)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attitude_rpy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== PLOT 4: HEIGHT COMPARISON ====================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, height_sonar, 'b-', label='Sonar Height [m]', linewidth=2, alpha=0.8)
    ax.plot(timestamps, height_baro, 'r-', label='Barometer Height [m]', linewidth=2, alpha=0.8)
    ax.plot(timestamps, target_altitudes, 'k--', label='Target Altitude [m]', linewidth=2, alpha=0.7)
    
    # Add setpoint region backgrounds and labels
    for i, alt in enumerate(altitude_setpoints):
        mask = np.abs(target_altitudes - alt) < 0.05
        if np.any(mask):
            t_segment = timestamps[mask]
            t_start, t_end = t_segment[0], t_segment[-1]
            color = colors[i % len(colors)]
            
            # Add colored background
            ax.axvspan(t_start, t_end, alpha=0.15, color=color)
            
            # Add altitude label
            t_center = (t_start + t_end) / 2
            ax.text(t_center, alt + 0.1, f'{alt}m', horizontalalignment='center', 
                   fontsize=11, color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Mission Time [s]')
    ax.set_ylabel('Height [m]')
    ax.set_title('Height Sensor Comparison Across Mission')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'height_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== PLOT 5: SAMPLING RATE ANALYSIS ====================
    dt_values = np.diff(timestamps)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series of sampling intervals
    ax1.plot(timestamps[1:], dt_values, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(0.1, color='r', linestyle='--', linewidth=2, label='Target (0.1s = 10Hz)')
    ax1.set_ylabel('Sample Interval [s]')
    ax1.set_title('Sampling Rate Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.5)  # Focus on normal sampling intervals
    
    # Histogram of sampling intervals (exclude large gaps)
    valid_dt = dt_values[dt_values < 0.5]  # Remove movement gaps
    ax2.hist(valid_dt, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax2.axvline(0.1, color='r', linestyle='--', linewidth=2, label='Target (10Hz)')
    if len(valid_dt) > 0:
        ax2.axvline(np.mean(valid_dt), color='g', linestyle='-', linewidth=2, 
                   label=f'Mean ({1/np.mean(valid_dt):.1f}Hz)')
    ax2.set_xlabel('Sample Interval [s]')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Sample Interval Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sampling_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== QUANTITATIVE ANALYSIS ====================
    print(f"\n{'='*60}")
    print("PHASE 1 SENSOR ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Call student-implemented analysis functions
    mean_rate, rate_std, rate_min, rate_max = analyze_sampling_rate(timestamps)
    print(f"\n--- SAMPLING RATE ANALYSIS ---")
    print(f"Target sampling rate:        10.0 Hz")
    print(f"Actual mean sampling rate:   {mean_rate:.1f} ± {rate_std:.1f} Hz")
    print(f"Sampling rate range:         {rate_min:.1f} - {rate_max:.1f} Hz")
    
    # Noise analysis for each sensor type
    print(f"\n--- NOISE ANALYSIS ---")
    acc_noise = [calculate_noise_level(acceleration[:, i]) for i in range(3)]
    vel_noise = [calculate_noise_level(velocity[:, i]) for i in range(3)]
    att_noise = [calculate_noise_level(attitude[:, i]) for i in range(3)]
    sonar_noise = calculate_noise_level(height_sonar)
    baro_noise = calculate_noise_level(height_baro)
    
    print(f"Acceleration noise (x,y,z):  {acc_noise[0]:.3f}, {acc_noise[1]:.3f}, {acc_noise[2]:.3f} m/s²")
    print(f"Velocity noise (x,y,z):      {vel_noise[0]:.3f}, {vel_noise[1]:.3f}, {vel_noise[2]:.3f} m/s")
    print(f"Attitude noise (r,p,y):      {att_noise[0]:.2f}, {att_noise[1]:.2f}, {att_noise[2]:.2f} deg")
    print(f"Sonar height noise:          {sonar_noise:.3f} m")
    print(f"Barometer height noise:      {baro_noise:.3f} m")
    
    # Drift analysis
    print(f"\n--- DRIFT ANALYSIS ---")
    sonar_drift, sonar_r2 = calculate_drift_rate(timestamps, height_sonar)
    baro_drift, baro_r2 = calculate_drift_rate(timestamps, height_baro)
    
    print(f"Sonar height drift:          {sonar_drift:.4f} m/s (R² = {sonar_r2:.3f})")
    print(f"Barometer height drift:      {baro_drift:.4f} m/s (R² = {baro_r2:.3f})")
    
    # Height sensor comparison
    print(f"\n--- HEIGHT SENSOR COMPARISON ---")
    bias, correlation, sonar_dropout, baro_dropout, sonar_rmse, baro_rmse = \
        compare_height_sensors(height_sonar, height_baro, target_altitudes)
    
    print(f"Mean bias (baro - sonar):    {bias:.3f} m")
    print(f"Correlation coefficient:     {correlation:.3f}")
    print(f"Sonar dropout rate:          {sonar_dropout*100:.1f}%")
    print(f"Barometer dropout rate:      {baro_dropout*100:.1f}%")
    print(f"Sonar RMSE vs targets:       {sonar_rmse:.3f} m")
    print(f"Barometer RMSE vs targets:   {baro_rmse:.3f} m")
    
    # ==================== TODO: Add your interpretation here ====================
    print(f"\n{'='*60}")
    print("SENSOR CHARACTERIZATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTODO: Examine the analysis results above and answer these questions:")
    print(f"  1. Which sensors are noisy? Compare noise levels.")
    print(f"  2. Do you see any drift or bias? Is it significant?")
    print(f"  3. How do barometer vs sonar compare? Which is more reliable?")
    print(f"  4. What's the sampling rate? Did we achieve 10 Hz?")
    print(f"  5. Which sensor would you recommend for altitude control?")
    print(f"\nWrite your observations in comments or a separate text file.")
    
    print(f"\nPhase 1 analysis complete! Plots saved to {OUTPUT_DIR}/")
    print(f"Next: Run step_response_calibration.py for Phase 2")

if __name__ == "__main__":
    main()