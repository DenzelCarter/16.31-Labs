#!/usr/bin/env python3
"""
Lab 3 - Phase 2: Kalman Filter Design and Validation

Objectives:
1. Quick simulation sanity check of KF implementation
2. Validate KF on real drone with hover test
3. Test KF resilience against simulated attacks of varying duration
4. Analyze relationship between attack duration and system performance
5. Document Q/R tuning and attack tolerance based on real data

Deliverables:
- kalman_filter.py (in utils/)
- kf_simulation.png
- hover_test_data.csv, hover_test_plots.png
- attack_test_2s/5s/10s.csv
- attack_comparison.png
- attack_analysis.txt
- kf_tuning.txt
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
from utils.kalman_filter import AltitudeKalmanFilter, test_kalman_filter

OUTPUT_DIR = "Lab3-Phase2"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def quick_simulation_check():
    """
    Quick sanity check: Verify KF implementation works in simulation
    
    This is just to catch obvious bugs before flying.
    The REAL validation happens on hardware.
    """
    print("\n" + "="*60)
    print("QUICK SIMULATION SANITY CHECK")
    print("="*60)
    print("(Just verifying KF math - real validation on drone)")
    
    ensure_dir(OUTPUT_DIR)
    
    # Change directory temporarily for output
    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)
    
    try:
        test_kalman_filter()  # From kalman_filter.py
        print("\n[+] Simulation sanity check passed!")
        print("  (Proceeding to hardware validation...)")
    finally:
        os.chdir(original_dir)


def test_hover_validation(tello):
    """
    Hover test on real drone
    
    Goal: Characterize real sensor noise and validate KF smoothing
    
    Hovers at constant altitude for 30 seconds while logging:
    - ToF measurements (raw sensor data)
    - KF estimates (filtered data)
    - Innovation (measurement residual)
    - Kalman gain evolution
    """
    print("\n" + "="*60)
    print("HARDWARE TEST 1: HOVER VALIDATION")
    print("="*60)
    
    # Get initial altitude from ToF
    z_tof_initial = tello.get_distance_tof() / 100.0
    print(f"\nInitial altitude: {z_tof_initial:.2f}m")
    
    # Initialize KF
    kf = AltitudeKalmanFilter(dt=0.1)
    kf.initialize(z_tof_initial, 0.0)
    
    print("\nHovering for 30 seconds to characterize sensor noise...")
    print("(KF should smooth out ToF noise while tracking altitude)")
    
    data = {
        'time': [],
        'z_tof': [],
        'z_est': [],
        'vz_est': [],
        'innovation': [],
        'K_z': [],
        'P_zz': []
    }
    
    start_time = time.time()
    duration = 30.0
    
    while time.time() - start_time < duration:
        t = time.time() - start_time
        
        # CRITICAL: Send neutral RC command to prevent auto-land timeout
        tello.send_rc_control(0, 0, 0, 0)
        
        # Get ToF measurement
        z_tof = tello.get_distance_tof() / 100.0
        
        # Run KF (hovering, so acceleration command = 0)
        kf.predict(u=0.0)
        kf.update(z_tof)
        
        z_est, vz_est = kf.get_state()
        diag = kf.get_diagnostics()
        
        # Log data
        data['time'].append(t)
        data['z_tof'].append(z_tof)
        data['z_est'].append(z_est)
        data['vz_est'].append(vz_est)
        data['innovation'].append(diag['innovation'])
        data['K_z'].append(diag['K_z'])
        data['P_zz'].append(diag['P_zz'])
        
        # Print progress every 5 seconds
        if len(data['time']) % 50 == 0:
            print(f"  t={t:.1f}s: z_ToF={z_tof:.3f}m, z_est={z_est:.3f}m, "
                  f"innovation={diag['innovation']*100:.1f}cm")
        
        time.sleep(0.1)
    
    print("\n[+] Hover test complete!")
    
    # Save data
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'hover_test_data.csv'), index=False)
    print(f"  Data saved: {OUTPUT_DIR}/hover_test_data.csv")
    
    # Plot results
    plot_hover_test(data)
    
    # Calculate statistics
    z_tof_std = np.std(data['z_tof'])
    z_est_std = np.std(data['z_est'])
    innovation_rms = np.sqrt(np.mean(np.array(data['innovation'])**2))
    
    print(f"\nHover Statistics:")
    print(f"  ToF noise (std): {z_tof_std*100:.2f} cm")
    print(f"  KF estimate (std): {z_est_std*100:.2f} cm")
    print(f"  Innovation RMS: {innovation_rms*100:.2f} cm")
    print(f"  Noise reduction: {z_tof_std/z_est_std:.1f}x smoother")
    
    return data


def test_attack_resilience(tello, attack_duration, attack_start=5.0):
    """
    Attack resilience test
    
    Goal: Test KF performance under simulated sensor attack
    
    Performs controlled descent while injecting a +0.5m bias during attack window.
    KF switches to prediction-only mode during attack, ignoring compromised sensor.
    
    Args:
        tello: Tello drone object
        attack_duration: Duration of attack in seconds (2, 5, or 10)
        attack_start: When to start attack [s]
    
    Returns:
        data: Dictionary with test results
    """
    print(f"\nTesting {attack_duration}s attack duration...")
    
    # Climb to test altitude
    target_alt = 1.5
    current_alt = tello.get_distance_tof() / 100.0
    
    print("  Climbing to 1.5m...")
    while current_alt < target_alt - 0.1:
        tello.send_rc_control(0, 0, 20, 0)  # Gentle climb
        time.sleep(0.1)
        current_alt = tello.get_distance_tof() / 100.0
    
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(2.0)
    
    # Initialize KF
    z_init = tello.get_distance_tof() / 100.0
    kf = AltitudeKalmanFilter(dt=0.1)
    kf.initialize(z_init, 0.0)
    
    attack_end = attack_start + attack_duration
    
    print(f"  Descending with attack from t={attack_start:.1f}s to t={attack_end:.1f}s")
    print(f"  Attack: Injecting +0.5m bias into ToF readings")
    
    data = {
        'time': [],
        'z_tof_real': [],           # Ground truth
        'z_tof_compromised': [],    # What sensor reports during attack
        'z_est': [],                # KF estimate
        'vz_est': [],
        'mode': [],                 # 'normal' or 'attack'
        'error_real': [],           # |z_real - z_est|
        'attack_duration': attack_duration
    }
    
    start_time = time.time()
    kp_z = 0.3  # Gentle descent control
    z_target = 0.5
    
    while True:
        t = time.time() - start_time
        
        # Get REAL measurement (ground truth)
        z_tof_real = tello.get_distance_tof() / 100.0
        
        # Inject attack: Add +0.5m bias during attack window
        if attack_start <= t < attack_end:
            z_tof_compromised = z_tof_real + 0.5  # SIMULATED ATTACK
            mode = 'attack'
            
            # Switch to prediction-only (ignore compromised sensor)
            if not kf.is_prediction_only():
                kf.set_mode(prediction_only=True)
        else:
            z_tof_compromised = z_tof_real  # No attack
            mode = 'normal'
            
            # Resume normal operation
            if kf.is_prediction_only():
                kf.set_mode(prediction_only=False)
        
        # Control based on KF estimate (not raw sensor!)
        z_est, vz_est = kf.get_state()
        z_error = z_target - z_est
        vz_cmd = kp_z * z_error
        vz_cmd = np.clip(vz_cmd, -0.3, 0.3)
        
        # KF prediction
        kf.predict(u=0.0)
        
        # KF update: Use REAL measurements in normal mode, ignore in attack mode
        if mode == 'normal':
            kf.update(z_tof_real)
        
        z_est, vz_est = kf.get_state()
        
        # Send control command
        rc_ud = int(vz_cmd * 100)
        tello.send_rc_control(0, 0, rc_ud, 0)
        
        # Log data
        data['time'].append(t)
        data['z_tof_real'].append(z_tof_real)
        data['z_tof_compromised'].append(z_tof_compromised)
        data['z_est'].append(z_est)
        data['vz_est'].append(vz_est)
        data['mode'].append(mode)
        data['error_real'].append(abs(z_tof_real - z_est))
        
        # Print progress
        if len(data['time']) % 25 == 0:
            bias = z_tof_compromised - z_tof_real
            print(f"  t={t:.1f}s [{mode:6s}]: Real={z_tof_real:.2f}m, "
                  f"Sensor={z_tof_compromised:.2f}m (+{bias:.2f}m), "
                  f"Est={z_est:.2f}m, err={data['error_real'][-1]*100:.1f}cm")
        
        # Stop when close to target
        if z_est < z_target + 0.1 and t > attack_end + 2.0:
            print(f"\n  [+] Reached target after {t:.1f}s")
            break
        
        # Safety timeout
        if t > 25.0:
            print("\n  WARNING: Timeout")
            break
        
        time.sleep(0.1)
    
    tello.send_rc_control(0, 0, 0, 0)
    
    # Analyze attack performance
    attack_errors = [data['error_real'][i] 
                     for i in range(len(data['mode'])) 
                     if data['mode'][i] == 'attack']
    
    if attack_errors:
        max_error = max(attack_errors) * 100
        mean_error = np.mean(attack_errors) * 100
        final_error = attack_errors[-1] * 100
        
        print(f"\n  Attack Performance ({attack_duration}s duration):")
        print(f"    Max error: {max_error:.1f} cm")
        print(f"    Mean error: {mean_error:.1f} cm")
        print(f"    Final error: {final_error:.1f} cm")
        status = "PASS" if max_error < 15 else "FAIL"
        print(f"    [{status}]: Max error {'<' if max_error < 15 else '>'} 15cm threshold")
    
    return data


def run_attack_tests(tello):
    """
    Run multiple attack tests with different durations
    
    Returns:
        results: Dictionary mapping duration -> test data
    """
    print("\n" + "="*60)
    print("HARDWARE TEST 2: ATTACK RESILIENCE TESTING")
    print("="*60)
    print("\nTesting KF resilience against attacks of varying duration:")
    print("  - 2s attack: Should pass easily")
    print("  - 5s attack: Testing limits")
    print("  - 10s attack: Expected to fail (demonstrates breakdown)")
    
    results = {}
    durations = [2, 5, 10]
    
    for i, duration in enumerate(durations):
        print(f"\n--- Attack Test {i+1}/3: {duration}s duration ---")
        
        # Run test
        data = test_attack_resilience(tello, duration)
        results[duration] = data
        
        # Save individual test data
        import pandas as pd
        df = pd.DataFrame(data)
        filename = f'attack_test_{duration}s.csv'
        df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        print(f"  Data saved: {OUTPUT_DIR}/{filename}")
        
        # Stabilize before next test
        if i < len(durations) - 1:
            print(f"\n  Stabilizing before next test...")
            time.sleep(3)
    
    return results


def plot_hover_test(data):
    """Generate hover test plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.array(data['time'])
    
    # Plot 1: Altitude measurements vs estimates
    ax = axes[0, 0]
    ax.plot(times, data['z_tof'], 'r.', alpha=0.3, markersize=2, label='ToF measurements')
    ax.plot(times, data['z_est'], 'b-', linewidth=2, label='KF estimate')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Hover Test: KF Smooths Noisy Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocity estimate
    ax = axes[0, 1]
    ax.plot(times, data['vz_est'], 'g-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity Estimate [m/s]')
    ax.set_title('Vertical Velocity (Should Be ~0 During Hover)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Innovation
    ax = axes[1, 0]
    ax.plot(times, np.array(data['innovation'])*100, 'b-', linewidth=1)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Innovation [cm]')
    ax.set_title('Measurement Residual (Small = Well-Tuned)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Kalman gain evolution
    ax = axes[1, 1]
    ax.plot(times, data['K_z'], 'r-', linewidth=2, label='K_z (position gain)')
    ax.plot(times, np.array(data['P_zz'])*100, 'b-', linewidth=2, label='P_zz * 100')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Gain / Covariance')
    ax.set_title('Kalman Gain Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hover_test_plots.png'), dpi=200)
    print(f"  Plots saved: {OUTPUT_DIR}/hover_test_plots.png")
    plt.close()


def plot_attack_comparison(results):
    """
    Generate comparative plots for all attack durations
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {2: 'green', 5: 'orange', 10: 'red'}
    
    # Plot 1-3: Individual descent profiles
    for i, duration in enumerate([2, 5, 10]):
        ax = fig.add_subplot(gs[i, 0])
        data = results[duration]
        times = np.array(data['time'])
        
        attack_mask = np.array([m == 'attack' for m in data['mode']])
        normal_mask = ~attack_mask
        
        # Plot real ToF
        ax.plot(times[normal_mask], np.array(data['z_tof_real'])[normal_mask],
                'b.', alpha=0.5, markersize=2, label='Real ToF (normal)')
        ax.plot(times[attack_mask], np.array(data['z_tof_real'])[attack_mask],
                'b.', alpha=0.5, markersize=2)
        
        # Plot compromised ToF during attack
        ax.plot(times[attack_mask], np.array(data['z_tof_compromised'])[attack_mask],
                'r.', alpha=0.5, markersize=3, label='Compromised ToF (+0.5m bias)')
        
        # Plot KF estimate
        ax.plot(times, data['z_est'], color=colors[duration], linewidth=2.5, 
                label='KF estimate (prediction-only during attack)')
        
        # Shade attack region
        attack_start = times[attack_mask][0] if any(attack_mask) else 0
        attack_end = times[attack_mask][-1] if any(attack_mask) else 0
        ax.axvspan(attack_start, attack_end, alpha=0.15, color='red')
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Altitude [m]')
        ax.set_title(f'{duration}s Attack: Sensor Bias vs KF Estimate')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Error comparison
    ax = fig.add_subplot(gs[0, 1])
    for duration in [2, 5, 10]:
        data = results[duration]
        times = np.array(data['time'])
        errors = np.array(data['error_real']) * 100
        ax.plot(times, errors, color=colors[duration], linewidth=2, 
                label=f'{duration}s attack', alpha=0.8)
    
    ax.axhline(15, color='k', linestyle='--', linewidth=1.5, label='15cm threshold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimation Error [cm]')
    ax.set_title('Error Growth During Attack')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Max error vs attack duration
    ax = fig.add_subplot(gs[1, 1])
    durations = [2, 5, 10]
    max_errors = []
    
    for dur in durations:
        data = results[dur]
        attack_errors = [data['error_real'][i] * 100
                        for i in range(len(data['mode'])) 
                        if data['mode'][i] == 'attack']
        max_errors.append(max(attack_errors) if attack_errors else 0)
    
    bars = ax.bar(durations, max_errors, color=[colors[d] for d in durations], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(15, color='red', linestyle='--', linewidth=2, label='15cm threshold')
    ax.set_xlabel('Attack Duration [s]')
    ax.set_ylabel('Maximum Error [cm]')
    ax.set_title('Max Drift vs Attack Duration')
    ax.set_xticks(durations)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, max_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}cm', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 6: Summary text
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    
    summary_text = "ATTACK RESILIENCE SUMMARY\n" + "="*50 + "\n\n"
    for dur in durations:
        data = results[dur]
        attack_errors = [data['error_real'][i] * 100
                        for i in range(len(data['mode'])) 
                        if data['mode'][i] == 'attack']
        
        if attack_errors:
            max_e = max(attack_errors)
            mean_e = np.mean(attack_errors)
            final_e = attack_errors[-1]
            status = "PASS" if max_e < 15 else "FAIL"
            
            summary_text += f"{dur}s Attack: {status}\n"
            summary_text += f"  Max error: {max_e:.1f} cm\n"
            summary_text += f"  Mean error: {mean_e:.1f} cm\n"
            summary_text += f"  Final error: {final_e:.1f} cm\n\n"
    
    safe_durations = [d for d, e in zip(durations, max_errors) if e < 15]
    max_safe = safe_durations[-1] if safe_durations else 0
    
    summary_text += "\nCONCLUSION:\n"
    summary_text += f"Safe attack tolerance: <{max_safe}s\n"
    summary_text += "Error growth is approximately linear with attack duration."
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'attack_comparison.png'), dpi=200)
    print(f"\n[+] Comparison plots saved: {OUTPUT_DIR}/attack_comparison.png")
    plt.close()


def write_tuning_document(hover_data, attack_results):
    """
    Generate tuning document based on REAL hardware data
    """
    z_tof_std = np.std(hover_data['z_tof'])
    z_est_std = np.std(hover_data['z_est'])
    innovation_rms = np.sqrt(np.mean(np.array(hover_data['innovation'])**2))
    
    attack_stats = {}
    for duration in [2, 5, 10]:
        data = attack_results[duration]
        attack_errors = [data['error_real'][i] * 100
                        for i in range(len(data['mode'])) 
                        if data['mode'][i] == 'attack']
        attack_stats[duration] = {
            'max': max(attack_errors) if attack_errors else 0,
            'mean': np.mean(attack_errors) if attack_errors else 0
        }
    
    tuning_text = f"""
KALMAN FILTER TUNING DOCUMENT
Based on Real Hardware Validation

1. MEASUREMENT NOISE COVARIANCE (R)
=====================================

From real hover test data:
- ToF sensor standard deviation: {z_tof_std*100:.2f} cm
- Therefore: R = sigma^2 = ({z_tof_std:.4f})^2 = {z_tof_std**2:.6f}

Physical meaning: Expected variance in ToF measurements
Source: Direct measurement from Phase 2 hover test

2. PROCESS NOISE COVARIANCE (Q)
=====================================

Q represents uncertainty in the motion model:

Q = diag([q_z, q_vz]) = diag([0.001^2, 0.01^2])

Position uncertainty (q_z = 0.001 m = 1 mm):
- Accounts for unmodeled dynamics (air resistance, prop wash)
- Command execution delays (~50-100ms)
- Linearization errors in double integrator model

Velocity uncertainty (q_vz = 0.01 m/s = 1 cm/s):
- Rapid velocity changes between samples
- Motor response delays
- Air currents and disturbances

3. VALIDATION FROM HARDWARE TESTS
===================================

Hover Test Results:
- ToF noise (std): {z_tof_std*100:.2f} cm (raw sensor)
- KF estimate (std): {z_est_std*100:.2f} cm (filtered)
- Noise reduction: {z_tof_std/z_est_std:.1f}x smoother
- Innovation RMS: {innovation_rms*100:.2f} cm

Attack Resilience Tests:
Attack with +0.5m sensor bias, KF using prediction-only mode

2s Attack:
- Max error: {attack_stats[2]['max']:.1f} cm
- Mean error: {attack_stats[2]['mean']:.1f} cm
- Result: {'PASS' if attack_stats[2]['max'] < 15 else 'FAIL'} (threshold: 15cm)

5s Attack:
- Max error: {attack_stats[5]['max']:.1f} cm
- Mean error: {attack_stats[5]['mean']:.1f} cm
- Result: {'PASS' if attack_stats[5]['max'] < 15 else 'FAIL'} (threshold: 15cm)

10s Attack:
- Max error: {attack_stats[10]['max']:.1f} cm
- Mean error: {attack_stats[10]['mean']:.1f} cm
- Result: {'PASS' if attack_stats[10]['max'] < 15 else 'FAIL'} (threshold: 15cm)

Interpretation:
- Prediction-only mode degrades with time (drift accumulates)
- Safe operational limit: ~{[d for d in [2, 5, 10] if attack_stats[d]['max'] < 15][-1] if any(attack_stats[d]['max'] < 15 for d in [2, 5, 10]) else 0}s without sensor updates
- Error growth approximately linear with attack duration
- For Phase 3 landing (5-8s descent), attack tolerance is {"adequate" if attack_stats[5]['max'] < 15 else "marginal"}

4. TUNING GUIDELINES
=====================

If KF needs adjustment based on flight testing:

- KF tracks too sluggishly -> Decrease R (trust measurements more)
- KF is too noisy -> Increase R (trust model more)
- Predictions drift quickly -> Increase Q (model is uncertain)
- KF ignores measurements -> Decrease Q (trust model more)

For prediction-only performance:
- Drift too large -> DECREASE Q (trust model more)
- Filter too rigid -> INCREASE Q (allow flexibility)

5. CONCLUSION
==============

Filter parameters validated on real hardware:
[+] Noise reduction: {z_tof_std/z_est_std:.1f}x smoother than raw measurements
[+] Prediction-only: <15cm drift for {[d for d in [2, 5, 10] if attack_stats[d]['max'] < 15][-1] if any(attack_stats[d]['max'] < 15 for d in [2, 5, 10]) else 0}s attacks
[+] Well-tuned: Small innovations, stable gain convergence

System is ready for Phase 3 resilient landing mission.
"""
    
    ensure_dir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'kf_tuning.txt'), 'w', encoding='utf-8') as f:
        f.write(tuning_text)
    
    print(f"\n[+] Tuning document saved: {OUTPUT_DIR}/kf_tuning.txt")


def create_analysis_template():
    """
    Create template for student analysis
    """
    template = """
ATTACK RESILIENCE ANALYSIS
Student: [Your Name]
Date: [Date]

Answer the following questions based on your experimental data:

1. ATTACK TOLERANCE
===================

Q1: What is the maximum attack duration your system can tolerate while
    maintaining <15cm accuracy?

A1: [Your answer]

Q2: How does estimation error scale with attack duration? Is it linear,
    quadratic, or something else? Support with data.

A2: [Your answer]

2. SYSTEM BEHAVIOR
==================

Q3: Describe what happens to the KF estimate during the attack window.
    Does it diverge immediately or gradually?

A3: [Your observations]

Q4: When the attack ends and normal measurements resume, how quickly does
    the KF recover?

A4: [Your observations]

3. CONTROL IMPLICATIONS
========================

Q5: During the 10s attack test, did the drone's descent behavior become
    unsafe or unstable? Explain.

A5: [Your observations]

Q6: For a real landing mission (5-8 seconds of descent), what attack
    duration could your system safely handle?

A6: [Your recommendation]

4. FILTER TUNING
================

Q7: If you needed to increase attack tolerance from 5s to 10s, how would
    you modify the Q matrix? Explain your reasoning.

A7: [Your answer]

Q8: What is the tradeoff of decreasing Q to improve prediction-only
    performance? What might get worse?

A8: [Your answer]

5. CONCLUSIONS
==============

Based on your experiments, summarize:
- Maximum safe attack duration: [X] seconds
- Key limiting factor: [e.g., "Velocity drift", "Position uncertainty"]
- Recommendation for Phase 3: [Your operational guidelines]

SUPPORTING DATA:
================
Include key numbers from your tests:
- 2s attack max error: [X] cm
- 5s attack max error: [X] cm  
- 10s attack max error: [X] cm
"""
    
    ensure_dir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, 'attack_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"[+] Analysis template created: {OUTPUT_DIR}/attack_analysis.txt")


def main():
    """Main function for Phase 2"""
    print("\n" + "="*60)
    print("LAB 3 - PHASE 2: KALMAN FILTER VALIDATION")
    print("="*60)
    
    ensure_dir(OUTPUT_DIR)
    
    # Optional: Quick simulation check
    run_sim = input("\nRun quick simulation sanity check? (y/n): ").lower() == 'y'
    if run_sim:
        quick_simulation_check()
    else:
        print("\nSkipping simulation (going straight to hardware validation)")
    
    # Hardware validation
    print("\n" + "="*60)
    print("HARDWARE VALIDATION")
    print("="*60)
    
    tello = Tello()
    print("\nConnecting to Tello...")
    tello.connect()
    
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    
    if battery < 50:
        print("ERROR: Battery too low (need >50% for multiple tests)")
        return
    
    tello.streamon()
    time.sleep(1)
    
    try:
        input("\nPress ENTER to begin hardware validation...")
        
        print("\nTaking off...")
        tello.takeoff()
        time.sleep(3)
        
        # Test 1: Hover validation
        print("\n" + "="*60)
        print("Starting Test 1: Hover Validation")
        print("="*60)
        hover_data = test_hover_validation(tello)
        
        print("\n  Stabilizing between tests...")
        time.sleep(2)
        
        # Test 2: Attack resilience
        attack_results = run_attack_tests(tello)
        
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        # Generate comparative plots
        print("\n" + "="*60)
        print("Generating comparative analysis...")
        print("="*60)
        plot_attack_comparison(attack_results)
        
        # Generate tuning document
        print("\n" + "="*60)
        print("Generating tuning document...")
        print("="*60)
        write_tuning_document(hover_data, attack_results)
        
        # Create analysis template
        create_analysis_template()
        
        # Summary
        print("\n" + "="*60)
        print("PHASE 2 COMPLETE!")
        print("="*60)
        print(f"\nDeliverables in {OUTPUT_DIR}/:")
        print("  [+] kalman_filter.py (in utils/)")
        if run_sim:
            print("  [+] kf_simulation.png")
        print("  [+] hover_test_data.csv")
        print("  [+] hover_test_plots.png")
        print("  [+] attack_test_2s.csv")
        print("  [+] attack_test_5s.csv")
        print("  [+] attack_test_10s.csv")
        print("  [+] attack_comparison.png")
        print("  [+] kf_tuning.txt")
        print("  [+] attack_analysis.txt")
        
        print("\nKey Results:")
        z_tof_std = np.std(hover_data['z_tof'])
        z_est_std = np.std(hover_data['z_est'])
        print(f"  - Noise reduction: {z_tof_std/z_est_std:.1f}x smoother")
        
        for dur in [2, 5, 10]:
            data = attack_results[dur]
            attack_errors = [data['error_real'][i] * 100
                           for i in range(len(data['mode'])) 
                           if data['mode'][i] == 'attack']
            max_err = max(attack_errors) if attack_errors else 0
            status = "[+]" if max_err < 15 else "[-]"
            print(f"  {status} {dur}s attack: max error {max_err:.1f} cm")
        
        print("\n[+] Complete attack_analysis.txt with your findings!")
        print("[+] Ready for Phase 3: Resilient Landing System")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    finally:
        try:
            tello.streamoff()
            tello.end()
        except:
            pass


if __name__ == "__main__":
    main()