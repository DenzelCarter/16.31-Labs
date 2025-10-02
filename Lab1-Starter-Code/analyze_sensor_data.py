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
    Estimate the sensor's noise level as the standard deviation of the
    signal after removing slow trends (moving-average detrend).

    Why detrend?
      - Raw sensor streams often include slow changes from setpoint moves,
        temperature drift, or vehicle dynamics. Those aren't "noise".
      - By subtracting a moving average (low-pass estimate of the trend),
        the residual mostly reflects high-frequency noise.

    Parameters
    ----------
    signal : array-like
        1D vector of samples (e.g., height, acceleration). Can include NaNs.
    detrend_window : int or None
        Length (in samples) of the moving-average window used to estimate
        the trend. If None, we pick ~10% of the segment length (min 5).
        The window is forced odd to make the filter symmetric.

    Returns
    -------
    float
        Estimated noise standard deviation (same units as the signal).
        Returns np.nan if the input is empty; returns 0.0 for size==1.
    """
    s = np.asarray(signal, dtype=float)

    # Keep only finite values so NaNs/Infs don't propagate.
    mask = np.isfinite(s)
    s = s[mask]
    n = s.size
    if n == 0:
        return np.nan
    if n < 10:
        # With very short segments, just return the raw STD (best we can do).
        return float(np.std(s, ddof=1)) if n > 1 else 0.0

    # Choose a default detrend window if none provided:
    #  - use ~10% of the samples, clamped to [3, inf), and make it odd.
    if detrend_window is None:
        detrend_window = max(5, n // 10)
    detrend_window = int(detrend_window)
    detrend_window = max(3, detrend_window)
    if detrend_window % 2 == 0:
        detrend_window += 1

    # Half-window for edge padding size (symmetric padding reduces edge bias).
    k = detrend_window // 2

    # Build simple boxcar (moving-average) kernel.
    w = np.ones(detrend_window, dtype=float) / detrend_window

    # Pad the series so the 'valid' convolution returns exactly n samples.
    s_padded = np.pad(s, (k, k), mode="edge")

    # Trend is the moving-average of the padded signal, trimmed back to length n.
    trend = np.convolve(s_padded, w, mode="valid")

    # High-frequency residual approximates noise.
    detrended = s - trend

    # Use sample STD (ddof=1) to avoid bias on finite samples.
    return float(np.std(detrended, ddof=1))


def calculate_drift_rate(timestamps, signal):
    """
    Estimate linear drift (slope) of a sensor by fitting s ≈ m*t + b.

    Rationale
      - Some sensors exhibit slow, near-linear drift due to temperature,
        pressure, bias creep, etc. A simple least-squares line fit captures
        the average drift rate over the window.

    Parameters
    ----------
    timestamps : array-like
        Time stamps in seconds (monotonic, but minor jitter is OK). Can include NaNs.
    signal : array-like
        Sensor measurements aligned with timestamps. Can include NaNs.

    Returns
    -------
    (drift_rate, r2) : (float, float)
        drift_rate : slope m in units of (signal units)/second.
        r2         : coefficient of determination [0..1] indicating how
                     "linear" the data is (1.0 = perfectly linear).
                     Returns 0.0 if variance is zero or too little data.
    """
    t = np.asarray(timestamps, dtype=float)
    s = np.asarray(signal, dtype=float)

    # Use only pairs where both time and signal are finite.
    mask = np.isfinite(t) & np.isfinite(s)
    t = t[mask]
    s = s[mask]
    if t.size < 2:
        return 0.0, 0.0

    # Least-squares linear fit: s_hat = m*t + b
    m, b = np.polyfit(t, s, 1)

    # Goodness of fit (R^2): how much of the variance is explained by the line.
    s_hat = m * t + b
    resid = s - s_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Clamp R^2 into [0, 1] for numeric safety.
    return float(m), float(max(min(r2, 1.0), 0.0))


def analyze_sampling_rate(timestamps):
    """
    Compute sampling-rate statistics from time stamps.

    Method
      - Convert timestamps to inter-sample intervals dt = diff(t).
      - Filter to 'plausible' intervals: (0, 1.0) s to ignore big gaps
        from pauses or logging hiccups (tuned for a target ~10 Hz system).
      - Convert to instantaneous rates r = 1/dt and summarize.

    Parameters
    ----------
    timestamps : array-like
        Time stamps in seconds. Can include NaNs and occasional gaps.

    Returns
    -------
    mean_rate, rate_std, rate_min, rate_max : floats (Hz)
        mean_rate : average instantaneous sampling rate
        rate_std  : sample std dev of instantaneous rate (jitter indicator)
        rate_min  : minimum observed rate over valid intervals
        rate_max  : maximum observed rate over valid intervals

    Notes
    -----
    - If no valid intervals remain after filtering, returns NaNs.
    - For strictly uniform sampling at 10 Hz, expect:
        mean_rate ≈ 10, rate_std ≈ 0, rate_min ≈ rate_max ≈ 10.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    # First differences (time between consecutive samples).
    dt = np.diff(t)

    # Keep only finite, positive intervals smaller than 1 second.
    #  - Removes zeros/negatives (bad clocks) and large gaps (e.g., pauses).
    valid = np.isfinite(dt) & (dt > 0) & (dt < 1.0)
    dt_valid = dt[valid]
    if dt_valid.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Instantaneous sampling rate at each interval.
    rates = 1.0 / dt_valid

    mean_rate = float(np.mean(rates))
    rate_std = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
    rate_min = float(np.min(rates))
    rate_max = float(np.max(rates))
    return mean_rate, rate_std, rate_min, rate_max


def compare_height_sensors(height_sonar, height_baro, target_altitudes):
    """
    Cross-compare sonar and barometer altitude measurements and quantify
    performance relative to commanded/target altitude.

    Metrics reported
      - bias (baro - sonar): mean difference where both sensors are valid
      - correlation: linear correlation coefficient between sensors
      - sonar_dropout_rate: fraction of sonar samples that are NaN/Inf
      - baro_dropout_rate: fraction of baro samples that are NaN/Inf
      - sonar_rmse: RMSE between sonar and target altitude
      - baro_rmse: RMSE between baro and target altitude

    Parameters
    ----------
    height_sonar : array-like
        Sonar altitude (meters). Can include NaNs during out-of-range or loss.
    height_baro : array-like
        Barometer altitude (meters). Can include NaNs.
    target_altitudes : array-like
        Commanded or reference altitude (meters), aligned in time with sensors.

    Returns
    -------
    bias, corr, sonar_dropout, baro_dropout, sonar_rmse, baro_rmse : floats
        bias          : mean(baro - sonar) over samples where both are valid
        corr          : Pearson correlation between sonar and baro (NaN if <2 pairs)
        sonar_dropout : fraction of invalid sonar samples
        baro_dropout  : fraction of invalid baro samples
        sonar_rmse    : RMSE(sonar, target) using samples where both are valid
        baro_rmse     : RMSE(baro, target) using samples where both are valid

    Notes
    -----
    - Bias sign convention (baro - sonar):
        > 0  → baro reads higher than sonar on average
        < 0  → baro reads lower than sonar on average
    - RMSE is computed independently for each sensor against target, using
      only the samples valid for that sensor (to avoid penalizing dropouts
      as large errors).
    """
    sonar = np.asarray(height_sonar, dtype=float)
    baro = np.asarray(height_baro, dtype=float)
    target = np.asarray(target_altitudes, dtype=float)

    # Identify indices where both sensors are valid for bias/correlation.
    valid_both = np.isfinite(sonar) & np.isfinite(baro)
    n_both = int(np.count_nonzero(valid_both))

    if n_both >= 2:
        # Mean difference and correlation when we have at least two pairs.
        bias = float(np.mean(baro[valid_both] - sonar[valid_both]))
        corr = float(np.corrcoef(sonar[valid_both], baro[valid_both])[0, 1])
    elif n_both == 1:
        # With a single overlapping sample, correlation is undefined;
        # bias is still meaningful for that one pair.
        bias = float((baro[valid_both] - sonar[valid_both])[0])
        corr = np.nan
    else:
        # No overlap: cannot compute bias or correlation.
        bias = np.nan
        corr = np.nan

    # Dropout rates: fraction of samples that are invalid (NaN/Inf).
    sonar_dropout = float(np.mean(~np.isfinite(sonar))) if sonar.size else np.nan
    baro_dropout = float(np.mean(~np.isfinite(baro))) if baro.size else np.nan

    # RMSE vs target for each sensor computed on that sensor's valid samples.
    valid_sonar = np.isfinite(sonar) & np.isfinite(target)
    valid_baro = np.isfinite(baro) & np.isfinite(target)

    if np.any(valid_sonar):
        sonar_rmse = float(np.sqrt(np.mean((sonar[valid_sonar] - target[valid_sonar]) ** 2)))
    else:
        sonar_rmse = np.nan

    if np.any(valid_baro):
        baro_rmse = float(np.sqrt(np.mean((baro[valid_baro] - target[valid_baro]) ** 2)))
    else:
        baro_rmse = np.nan

    return bias, corr, sonar_dropout, baro_dropout, sonar_rmse, baro_rmse

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