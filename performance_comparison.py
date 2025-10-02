#!/usr/bin/env python3
"""
Lab 1 - Phase 5: Control Edge Validation

This script compares the performance of the heuristic (if-else) controller from Phase 3
with the PID controller from Phase 4.

Both controllers tested identical trajectory: 0.5m → 1.5m for 30 seconds
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
OUTPUT_DIR = "Lab1-Phase5"
PHASE3_DIR = "Lab1-Phase3"
PHASE4_DIR = "Lab1-Phase4-Simple"

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

# ============================================================================
# PROVIDED: Data loading functions
# ============================================================================

def load_heuristic_results():
    """Load heuristic controller results from Phase 3"""
    
    csv_file = os.path.join(PHASE3_DIR, "heuristic_control_data.csv")
    metrics_file = os.path.join(PHASE3_DIR, "heuristic_metrics.json")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Heuristic data not found: {csv_file}")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Heuristic metrics not found: {metrics_file}")
    
    df = pd.read_csv(csv_file)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Convert string representations back to appropriate types
    for key, value in metrics.items():
        if value == "inf":
            metrics[key] = float('inf')
        elif value == "nan":
            metrics[key] = float('nan')
    
    print(f"Loaded heuristic results: {len(df)} samples")
    print(f"Test duration: {df['mission_time'].iloc[-1]:.1f} seconds")
    
    return df, metrics

def load_pid_results():
    """Load PID controller results from Phase 4"""
    
    csv_file = os.path.join(PHASE4_DIR, "pid_vs_heuristic_comparison.csv")
    results_file = os.path.join(PHASE4_DIR, "final_comparison_results.json")
    tuning_file = os.path.join(PHASE4_DIR, "simple_pid_tuning_results.json")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"PID data not found: {csv_file}")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"PID results not found: {results_file}")
    
    df = pd.read_csv(csv_file)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    tuning_info = None
    if os.path.exists(tuning_file):
        with open(tuning_file, 'r') as f:
            tuning_info = json.load(f)
    
    print(f"Loaded PID results: {len(df)} samples")
    print(f"Test duration: {df['mission_time'].iloc[-1]:.1f} seconds")
    print(f"PID gains: Kp={results['pid_gains']['kp']:.2f}, "
          f"Ki={results['pid_gains']['ki']:.3f}, "
          f"Kd={results['pid_gains']['kd']:.3f}")
    
    return df, results, tuning_info

# ============================================================================
# PROVIDED: Performance metrics calculation (reuse from Phase 3/4)
# ============================================================================

def calculate_performance_metrics(data_dict):
    """Calculate performance metrics from data dictionary"""
    
    timestamps = np.array(data_dict['mission_time'])
    altitudes = np.array(data_dict['current_altitude'])
    targets = np.array(data_dict['target_altitude'])
    errors = np.array(data_dict['error'])
    velocities = np.array(data_dict['velocity_cmd'])
    
    target = targets[0]
    t0 = timestamps[0]
    T = timestamps[-1]
    delta_band = 0.05 * abs(target)
    
    # Settling time
    abs_errors = np.abs(errors)
    settling_time = None
    for i in range(len(timestamps)):
        if all(abs_errors[i:] <= delta_band):
            settling_time = timestamps[i] - t0
            break
    if settling_time is None:
        settling_time = float('inf')
    
    # Steady-state error
    if settling_time != float('inf'):
        T_ss = T - (t0 + settling_time)
    else:
        T_ss = min(15.0, T - t0)
    ss_start_time = T - T_ss
    ss_mask = timestamps >= ss_start_time
    ss_errors = errors[ss_mask]
    steady_state_error = np.mean(np.abs(ss_errors)) if len(ss_errors) > 0 else float('inf')
    
    # Rise time
    h0 = altitudes[0]
    A = abs(target - h0)
    if A > 0.01:
        h_10 = h0 + 0.1 * (target - h0)
        h_90 = h0 + 0.9 * (target - h0)
        t_10 = None
        t_90 = None
        if target > h0:
            for i, h in enumerate(altitudes):
                if t_10 is None and h >= h_10:
                    t_10 = timestamps[i]
                if t_90 is None and h >= h_90:
                    t_90 = timestamps[i]
                    break
        else:
            for i, h in enumerate(altitudes):
                if t_10 is None and h <= h_10:
                    t_10 = timestamps[i]
                if t_90 is None and h <= h_90:
                    t_90 = timestamps[i]
                    break
        rise_time = (t_90 - t_10) if (t_10 is not None and t_90 is not None) else float('inf')
    else:
        rise_time = 0.0
    
    # Maximum overshoot
    s = np.sign(target - h0)
    overshoot_values = s * (altitudes - target)
    max_overshoot_abs = np.max(overshoot_values.clip(min=0))
    max_overshoot_percent = (100 * max_overshoot_abs / A) if A > 0.01 else 0.0
    
    # Control chattering
    sign_changes = 0
    deadband = 0.01
    for i in range(1, len(velocities)):
        v_prev = velocities[i-1]
        v_curr = velocities[i]
        if abs(v_prev) > deadband and abs(v_curr) > deadband:
            if np.sign(v_prev) != np.sign(v_curr):
                sign_changes += 1
    test_duration_minutes = (T - t0) / 60.0
    chattering = sign_changes / test_duration_minutes if test_duration_minutes > 0 else 0
    
    # RMS error
    rms_error = np.sqrt(np.mean(errors**2))
    
    return {
        'settling_time': settling_time,
        'steady_state_error': steady_state_error,
        'rise_time': rise_time,
        'max_overshoot_percent': max_overshoot_percent,
        'control_chattering': chattering,
        'rms_error': rms_error,
        'final_error': abs(errors[-1]),
        'mean_absolute_error': np.mean(np.abs(errors))
    }

# ============================================================================
# PROVIDED: Plotting functions
# ============================================================================

def create_comparison_plots(heuristic_df, pid_df):
    """Create side-by-side comparison plots"""
    
    ensure_dir(OUTPUT_DIR)
    
    # Main comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    h_times = heuristic_df['mission_time'].values
    h_altitudes = heuristic_df['current_altitude'].values
    h_targets = heuristic_df['target_altitude'].values
    h_errors = heuristic_df['error'].values
    h_velocities = heuristic_df['velocity_cmd'].values
    
    p_times = pid_df['mission_time'].values
    p_altitudes = pid_df['current_altitude'].values
    p_targets = pid_df['target_altitude'].values
    p_errors = pid_df['error'].values
    p_velocities = pid_df['velocity_cmd'].values
    
    # Altitude tracking comparison
    ax1.plot(h_times, h_altitudes, 'r-', linewidth=2, alpha=0.8, label='Heuristic (If-Else)')
    ax1.axhline(h_targets[0], color='k', linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({h_targets[0]:.1f}m)')
    ax1.axhline(0.5, color='g', linestyle=':', linewidth=1, alpha=0.7, label='Initial (0.5m)')
    ax1.fill_between(h_times, h_targets[0] - 0.1, h_targets[0] + 0.1, 
                     alpha=0.2, color='green', label='±0.1m tolerance')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('Altitude Tracking: Heuristic Controller (0.5m → 1.5m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1.8)
    
    ax2.plot(p_times, p_altitudes, 'b-', linewidth=2, alpha=0.8, label='PID Controller')
    ax2.axhline(p_targets[0], color='k', linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({p_targets[0]:.1f}m)')
    ax2.axhline(0.5, color='g', linestyle=':', linewidth=1, alpha=0.7, label='Initial (0.5m)')
    ax2.fill_between(p_times, p_targets[0] - 0.1, p_targets[0] + 0.1, 
                     alpha=0.2, color='green', label='±0.1m tolerance')
    ax2.set_ylabel('Altitude [m]')
    ax2.set_title('Altitude Tracking: PID Controller (0.5m → 1.5m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 1.8)
    
    # Error comparison
    ax3.plot(h_times, h_errors, 'r-', linewidth=1.5, alpha=0.8, label='Heuristic Error')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='±0.1m tolerance')
    ax3.axhline(-0.1, color='gray', linestyle=':', alpha=0.7)
    ax3.fill_between(h_times, -0.1, 0.1, alpha=0.2, color='green')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error [m]')
    ax3.set_title('Tracking Error: Heuristic')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(p_times, p_errors, 'b-', linewidth=1.5, alpha=0.8, label='PID Error')
    ax4.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax4.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='±0.1m tolerance')
    ax4.axhline(-0.1, color='gray', linestyle=':', alpha=0.7)
    ax4.fill_between(p_times, -0.1, 0.1, alpha=0.2, color='green')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Error [m]')
    ax4.set_title('Tracking Error: PID')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'controller_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Control effort comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    ax1.plot(h_times, h_velocities, 'r-', linewidth=2, alpha=0.8, label='Heuristic Control (Bang-Bang)')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax1.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='UP command (1.0 m/s)')
    ax1.axhline(-0.8, color='orange', linestyle='--', alpha=0.7, label='DOWN command (-0.8 m/s)')
    ax1.set_ylabel('Velocity Command [m/s]')
    ax1.set_title('Control Effort: Heuristic (Binary/Discontinuous)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.2, 1.2)
    
    ax2.plot(p_times, p_velocities, 'b-', linewidth=2, alpha=0.8, label='PID Control (Continuous)')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='Heuristic UP (1.0 m/s)')
    ax2.axhline(-0.8, color='orange', linestyle='--', alpha=0.7, label='Heuristic DOWN (-0.8 m/s)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity Command [m/s]')
    ax2.set_title('Control Effort: PID (Smooth/Continuous)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'control_effort_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

# ============================================================================
# PROVIDED: Table generation and winner determination
# ============================================================================

def determine_winner(metric_key, heuristic_value, pid_value):
    """Determine which controller performs better for a given metric"""
    
    if np.isnan(heuristic_value) and np.isnan(pid_value):
        return "Tie"
    elif np.isnan(heuristic_value):
        return "PID"
    elif np.isnan(pid_value):
        return "Heuristic"
    elif heuristic_value == float('inf') and pid_value == float('inf'):
        return "Tie"
    elif heuristic_value == float('inf'):
        return "PID"
    elif pid_value == float('inf'):
        return "Heuristic"
    
    # For all metrics, lower is better
    if abs(heuristic_value - pid_value) < 1e-6:
        return "Tie"
    elif pid_value < heuristic_value:
        return "PID"
    else:
        return "Heuristic"

def create_metrics_comparison_table(heuristic_metrics, pid_metrics):
    """Create and save the performance metrics comparison table"""
    
    metrics_labels = {
        'settling_time': 'Settling time [s]',
        'steady_state_error': 'Steady-state error [m]',
        'rise_time': 'Rise time [s]',
        'max_overshoot_percent': 'Maximum overshoot [%]',
        'control_chattering': 'Control chattering [switches/min]',
        'rms_error': 'RMS error [m]'
    }
    
    comparison_data = []
    
    for metric_key, metric_label in metrics_labels.items():
        h_value = heuristic_metrics.get(metric_key, float('nan'))
        p_value = pid_metrics.get(metric_key, float('nan'))
        
        # Format values
        if metric_key in ['settling_time', 'rise_time']:
            h_str = f"{h_value:.2f}" if h_value != float('inf') else "inf"
            p_str = f"{p_value:.2f}" if p_value != float('inf') else "inf"
        elif metric_key in ['steady_state_error', 'rms_error']:
            h_str = f"{h_value:.3f}" if not np.isnan(h_value) and h_value != float('inf') else "inf"
            p_str = f"{p_value:.3f}" if not np.isnan(p_value) and p_value != float('inf') else "inf"
        else:
            h_str = f"{h_value:.1f}" if not np.isnan(h_value) else "N/A"
            p_str = f"{p_value:.1f}" if not np.isnan(p_value) else "N/A"
        
        winner = determine_winner(metric_key, h_value, p_value)
        
        comparison_data.append({
            'Metric': metric_label,
            'If-Else Controller': h_str,
            'PID Controller': p_str,
            'Winner': winner
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_file = os.path.join(OUTPUT_DIR, 'performance_comparison_table.csv')
    df.to_csv(csv_file, index=False)
    
    # Create formatted text table
    table_file = os.path.join(OUTPUT_DIR, 'performance_comparison_table.txt')
    with open(table_file, 'w') as f:
        f.write("PERFORMANCE COMPARISON: HEURISTIC vs PID CONTROL\n")
        f.write("=" * 60 + "\n")
        f.write("Test Trajectory: 0.5m -> 1.5m (30 seconds)\n")
        f.write("Both controllers tested under identical conditions\n\n")
        
        col_widths = [30, 20, 15, 10]
        header = f"{'Metric':<{col_widths[0]}} {'If-Else Controller':<{col_widths[1]}} {'PID Controller':<{col_widths[2]}} {'Winner':<{col_widths[3]}}"
        f.write(header + "\n")
        f.write("-" * sum(col_widths) + "\n")
        
        for _, row in df.iterrows():
            line = f"{row['Metric']:<{col_widths[0]}} {row['If-Else Controller']:<{col_widths[1]}} {row['PID Controller']:<{col_widths[2]}} {row['Winner']:<{col_widths[3]}}"
            f.write(line + "\n")
        
        f.write("\n")
        
        pid_wins = sum(1 for _, row in df.iterrows() if row['Winner'] == 'PID')
        heuristic_wins = sum(1 for _, row in df.iterrows() if row['Winner'] == 'Heuristic')
        ties = len(df) - pid_wins - heuristic_wins
        
        f.write(f"SUMMARY:\n")
        f.write(f"PID Controller wins: {pid_wins}/{len(df)} metrics\n")
        f.write(f"If-Else Controller wins: {heuristic_wins}/{len(df)} metrics\n")
        f.write(f"Ties: {ties}/{len(df)} metrics\n")
    
    print(f"Comparison table saved to {csv_file} and {table_file}")
    
    return df

# ============================================================================
# TODO: Write executive summary (minimal - just fill in conclusions)
# ============================================================================

def generate_executive_summary(comparison_df, heuristic_metrics, pid_metrics, pid_results):
    """
    Generate the executive summary for AeroTech management.
    
    TODO: After reviewing the comparison results, write 1-2 paragraphs answering:
    1. Has the PID approach outperformed if-else control?
    2. What are the key advantages (precision, stability, efficiency)?
    3. Does this give AeroTech a competitive advantage for the Navy contract?
    
    The template below shows where to add your analysis.
    """
    
    pid_wins = sum(1 for _, row in comparison_df.iterrows() if row['Winner'] == 'PID')
    total_metrics = len(comparison_df)
    
    # Calculate improvements
    improvements = {}
    for metric in ['rms_error', 'steady_state_error', 'control_chattering']:
        h_val = heuristic_metrics.get(metric, float('nan'))
        p_val = pid_metrics.get(metric, float('nan'))
        if not np.isnan(h_val) and not np.isnan(p_val) and h_val != 0:
            improvements[metric] = ((h_val - p_val) / h_val) * 100
    
    summary_file = os.path.join(OUTPUT_DIR, 'executive_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("Executive Summary: PID vs Heuristic Control Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENTAL DESIGN:\n")
        f.write("-" * 20 + "\n")
        f.write(" Identical test trajectory: 0.5m -> 1.5m\n")
        f.write(" Same test duration: 30 seconds\n")
        f.write(" Same drone, sensors, and environmental conditions\n")
        f.write(" Fair comparison with controlled initial conditions\n\n")
        
        f.write("PERFORMANCE COMPARISON RESULTS:\n")
        f.write("-" * 35 + "\n")
        f.write(f"PID Controller superior in: {pid_wins}/{total_metrics} metrics ({pid_wins/total_metrics*100:.0f}%)\n\n")
        
        if improvements:
            f.write("KEY PERFORMANCE IMPROVEMENTS:\n")
            f.write("-" * 32 + "\n")
            for metric, improvement in improvements.items():
                f.write(f" {metric}: {improvement:+.1f}% improvement\n")
            f.write("\n")
        
        # TODO: Add your technical summary here (1-2 paragraphs)
        f.write("TODO: Write your technical summary here\n")
        f.write("Based on the experimental results:\n")
        f.write("- Has PID outperformed if-else control?\n")
        f.write("- What are the key advantages?\n")
        f.write("- Does this provide competitive advantage?\n\n")
        
        # TODO: Add your recommendation
        f.write("RECOMMENDATION:\n")
        f.write("-" * 14 + "\n")
        f.write("TODO: Should AeroTech proceed with PID-based proposal?\n")
    
    print(f"Executive summary saved to {summary_file}")
    return summary_file

def main():
    """Main function to execute the performance comparison analysis"""
    
    print("Phase 5: Control Edge Validation")
    print("=" * 45)
    print("Comparing identical 0.5m → 1.5m trajectory tests...")
    
    ensure_dir(OUTPUT_DIR)
    
    try:
        # Load results
        print("\nLoading Phase 3 (Heuristic) results...")
        heuristic_df, heuristic_saved_metrics = load_heuristic_results()
        
        print("\nLoading Phase 4 (PID) results...")
        pid_df, pid_results, tuning_info = load_pid_results()
        
        # Calculate fresh metrics
        print("\nCalculating performance metrics for fair comparison...")
        
        heuristic_data = {
            'mission_time': heuristic_df['mission_time'].values,
            'current_altitude': heuristic_df['current_altitude'].values,
            'target_altitude': heuristic_df['target_altitude'].values,
            'velocity_cmd': heuristic_df['velocity_cmd'].values,
            'error': heuristic_df['error'].values
        }
        
        pid_data = {
            'mission_time': pid_df['mission_time'].values,
            'current_altitude': pid_df['current_altitude'].values,
            'target_altitude': pid_df['target_altitude'].values,
            'velocity_cmd': pid_df['velocity_cmd'].values,
            'error': pid_df['error'].values
        }
        
        heuristic_metrics = calculate_performance_metrics(heuristic_data)
        pid_metrics = calculate_performance_metrics(pid_data)
        
        print(f"Perfect experimental match for fair comparison!")
        
        # Create comparison plots
        print("\nGenerating comparison plots...")
        create_comparison_plots(heuristic_df, pid_df)
        
        # Create metrics table
        print("\nCreating performance comparison table...")
        comparison_df = create_metrics_comparison_table(heuristic_metrics, pid_metrics)
        
        # Generate executive summary
        print("\nGenerating executive summary...")
        summary_file = generate_executive_summary(comparison_df, heuristic_metrics, pid_metrics, pid_results)
        
        # Display results
        print(f"\n{'='*60}")
        print("PHASE 5 COMPARISON ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        pid_wins = sum(1 for _, row in comparison_df.iterrows() if row['Winner'] == 'PID')
        total_metrics = len(comparison_df)
        
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"PID Controller wins: {pid_wins}/{total_metrics} metrics ({pid_wins/total_metrics*100:.0f}%)")
        
        print(f"\nFILES GENERATED:")
        print(f"• {OUTPUT_DIR}/controller_comparison.png")
        print(f"• {OUTPUT_DIR}/control_effort_comparison.png") 
        print(f"• {OUTPUT_DIR}/performance_comparison_table.txt")
        print(f"• {OUTPUT_DIR}/executive_summary.txt")
        
        print(f"\nAll analysis files saved in {OUTPUT_DIR}/")
        print("Lab 1 Complete! Ready for submission.")
        print("\nTODO: Review the executive summary and add your conclusions!")
        
        return True
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure Phase 3 and Phase 4 have been completed successfully.")
        return False
    except Exception as e:
        print(f"ERROR during comparison analysis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nPhase 5 Complete! Lab 1 finished successfully.")
    else:
        print("\nPhase 5 failed. Check previous phases and try again.")