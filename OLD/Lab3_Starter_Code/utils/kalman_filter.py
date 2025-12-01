#!/usr/bin/env python3
"""
Lab 3 - Kalman Filter for Altitude Estimation

1D Kalman filter for altitude and velocity estimation using ToF sensor.
Supports two operational modes:
- Normal: Full predict-update cycle (honest sensors)
- Prediction-only: Dead reckoning (sensor attack/failure)

State: x = [z, v_z]^T
Input: u = a_z (vertical acceleration command)
Measurement: y = z_ToF (altitude from ToF sensor)
"""

import numpy as np

class AltitudeKalmanFilter:
    """
    1D Kalman filter for altitude estimation.
    
    State space model:
        x_{k+1} = A * x_k + B * u_k + w_k
        y_k = H * x_k + v_k
    
    where:
        x = [z, v_z]^T  (altitude, velocity)
        u = a_z         (acceleration command)
        y = z_ToF       (measured altitude)
        w ~ N(0, Q)     (process noise)
        v ~ N(0, R)     (measurement noise)
    """
    
    def __init__(self, dt=0.1, Q=None, R=None):
        """
        Initialize Kalman filter.
        
        Args:
            dt: Sampling period [s] (default: 0.1s = 10Hz)
            Q: Process noise covariance (2x2 matrix)
            R: Measurement noise variance (scalar)
        """
        self.dt = dt
        
        # State transition matrix (discrete double integrator)
        self.A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        # Control input matrix
        self.B = np.array([
            [0.5 * dt**2],
            [dt]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        if Q is None:
            # Default: conservative values
            # Q[0,0]: position uncertainty
            # Q[1,1]: velocity uncertainty
            self.Q = np.diag([0.001**2, 0.01**2])
        else:
            self.Q = Q
        
        # Measurement noise variance
        if R is None:
            # Default: typical ToF sensor noise from Lab 1
            self.R = np.array([[0.01**2]])
        else:
            self.R = np.array([[R]]) if np.isscalar(R) else R
        
        # State estimate [z, v_z]
        self.x_hat = np.zeros((2, 1))
        
        # Estimation error covariance
        self.P = np.eye(2) * 1.0  # Initial uncertainty
        
        # Kalman gain (for monitoring)
        self.K = np.zeros((2, 1))
        
        # Innovation (measurement residual)
        self.innovation = 0.0
        
        # Mode flag
        self.prediction_only_mode = False
    
    def initialize(self, z_initial, vz_initial=0.0):
        """
        Initialize filter state.
        
        Args:
            z_initial: Initial altitude [m]
            vz_initial: Initial vertical velocity [m/s]
        """
        self.x_hat = np.array([[z_initial], [vz_initial]])
        self.P = np.eye(2) * 0.1  # Reset covariance with some initial uncertainty
        self.prediction_only_mode = False
    
    def predict(self, u):
        """
        Prediction step: Propagate state using motion model.
        
        This step ALWAYS runs, regardless of mode.
        
        Args:
            u: Control input (vertical acceleration) [m/s^2]
        """
        # Convert scalar input to column vector
        u_vec = np.array([[u]]) if np.isscalar(u) else u.reshape(-1, 1)
        
        # ============================================================
        # TODO: Implement prediction step
        # ============================================================
        # 1. State prediction: x_{k|k-1} = A * x_{k-1|k-1} + B * u_{k-1}
        #    Update self.x_hat using self.A, self.B, and u_vec
        
        # YOUR CODE HERE
        
        
        # 2. Covariance prediction: P_{k|k-1} = A * P_{k-1|k-1} * A^T + Q
        #    Update self.P using self.A, self.Q
        
        # YOUR CODE HERE
        
        
        # ============================================================
    
    def update(self, measurement):
        """
        Update step: Incorporate measurement.
        
        This step is SKIPPED in prediction-only mode.
        
        Args:
            measurement: Altitude measurement from ToF sensor [m]
        """
        if self.prediction_only_mode:
            # Skip measurement update in attack mode
            return
        
        # Convert scalar measurement to correct shape
        y = np.array([[measurement]]) if np.isscalar(measurement) else measurement.reshape(-1, 1)
        
        # ============================================================
        # TODO: Implement update step
        # ============================================================
        # 1. Innovation: nu = y - H * x_{k|k-1}
        #    Compute innovation and store in self.innovation
        
        # YOUR CODE HERE
        
        
        # 2. Innovation covariance: S = H * P_{k|k-1} * H^T + R
        #    Compute S using self.H, self.P, self.R
        
        # YOUR CODE HERE
        
        
        # 3. Kalman gain: K = P_{k|k-1} * H^T * S^{-1}
        #    Compute and store in self.K
        #    Hint: Use np.linalg.inv() for matrix inverse
        
        # YOUR CODE HERE
        
        
        # 4. State update: x_{k|k} = x_{k|k-1} + K * nu
        #    Update self.x_hat using Kalman gain and innovation
        
        # YOUR CODE HERE
        
        
        # 5. Covariance update: P_{k|k} = (I - K * H) * P_{k|k-1}
        #    Update self.P
        #    Hint: I = np.eye(2)
        
        # YOUR CODE HERE
        
        
        # ============================================================
    
    def get_state(self):
        """
        Get current state estimate.
        
        Returns:
            z: Estimated altitude [m]
            vz: Estimated vertical velocity [m/s]
        """
        return float(self.x_hat[0, 0]), float(self.x_hat[1, 0])
    
    def get_diagnostics(self):
        """
        Get filter diagnostics for logging/monitoring.
        
        Returns:
            dict with kalman gain, covariance diagonal, innovation
        """
        return {
            'K_z': float(self.K[0, 0]),
            'K_vz': float(self.K[1, 0]),
            'P_zz': float(self.P[0, 0]),
            'P_vzvz': float(self.P[1, 1]),
            'innovation': float(self.innovation),
            'prediction_only': self.prediction_only_mode
        }
    
    def set_mode(self, prediction_only):
        """
        Set operational mode.
        
        Args:
            prediction_only: True for attack mode (no updates), False for normal
        """
        # ============================================================
        # TODO: Implement mode switching
        # ============================================================
        # 1. Set self.prediction_only_mode to the input parameter
        # 2. Print status message indicating mode change
        #    Example: "[KF] Switched to PREDICTION-ONLY mode (sensor attack)"
        #         or: "[KF] Switched to NORMAL mode (sensor trusted)"
        
        # YOUR CODE HERE
        
        
        # ============================================================
    
    def is_prediction_only(self):
        """Check if in prediction-only mode."""
        return self.prediction_only_mode


# Simulation utilities for Phase 2 testing
def simulate_altitude_trajectory(duration=10.0, dt=0.1, with_attack=False, attack_start=5.0):
    """
    Generate simulated altitude trajectory for KF testing.
    
    Args:
        duration: Simulation duration [s]
        dt: Time step [s]
        with_attack: If True, inject +0.5m bias after attack_start
        attack_start: Time to start attack [s]
    
    Returns:
        times, true_z, measurements, commands
    """
    times = np.arange(0, duration, dt)
    n = len(times)
    
    # Generate smooth descent profile
    true_z = np.zeros(n)
    true_vz = np.zeros(n)
    commands = np.zeros(n)
    
    # Start at 1.5m, descend to 0.1m over duration
    z0 = 1.5
    zf = 0.1
    
    for i, t in enumerate(times):
        # Smooth polynomial trajectory
        progress = t / duration
        true_z[i] = z0 + (zf - z0) * (3*progress**2 - 2*progress**3)
        true_vz[i] = (zf - z0) / duration * (6*progress - 6*progress**2)
        
        # Constant descent acceleration
        commands[i] = -0.1 if t > 0.5 else 0.0
    
    # Add measurement noise
    measurement_noise = np.random.normal(0, 0.01, n)
    measurements = true_z + measurement_noise
    
    # Inject attack bias
    if with_attack:
        attack_idx = int(attack_start / dt)
        measurements[attack_idx:] += 0.5  # +0.5m bias
    
    return times, true_z, measurements, commands, true_vz


def test_kalman_filter():
    """
    Test Kalman filter on simulated data.
    
    This generates plots for Phase 2 deliverables.
    """
    import matplotlib.pyplot as plt
    
    print("Testing Kalman Filter...")
    
    # Test 1: Normal operation
    print("\n1. Testing normal operation (honest sensors)...")
    times, true_z, meas_z, cmds, true_vz = simulate_altitude_trajectory(
        duration=10.0, with_attack=False
    )
    
    kf_normal = AltitudeKalmanFilter(dt=0.1)
    kf_normal.initialize(true_z[0], 0.0)
    
    est_z_normal = []
    est_vz_normal = []
    
    for i in range(len(times)):
        kf_normal.predict(cmds[i])
        kf_normal.update(meas_z[i])
        z, vz = kf_normal.get_state()
        est_z_normal.append(z)
        est_vz_normal.append(vz)
    
    # Test 2: Attack scenario with prediction-only response
    print("2. Testing attack scenario (prediction-only after 5s)...")
    times_atk, true_z_atk, meas_z_atk, cmds_atk, true_vz_atk = simulate_altitude_trajectory(
        duration=10.0, with_attack=True, attack_start=5.0
    )
    
    kf_attack = AltitudeKalmanFilter(dt=0.1)
    kf_attack.initialize(true_z_atk[0], 0.0)
    
    est_z_attack = []
    est_vz_attack = []
    
    for i, t in enumerate(times_atk):
        kf_attack.predict(cmds_atk[i])
        
        # Switch to prediction-only at attack start
        if t >= 5.0 and not kf_attack.is_prediction_only():
            kf_attack.set_mode(prediction_only=True)
        
        # Only update if in normal mode
        if not kf_attack.is_prediction_only():
            kf_attack.update(meas_z_atk[i])
        
        z, vz = kf_attack.get_state()
        est_z_attack.append(z)
        est_vz_attack.append(vz)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Normal operation
    ax = axes[0, 0]
    ax.plot(times, true_z, 'k-', linewidth=2, label='True altitude')
    ax.plot(times, meas_z, 'r.', alpha=0.3, markersize=2, label='Measurements')
    ax.plot(times, est_z_normal, 'b-', linewidth=2, label='KF estimate')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Normal Operation (Honest Sensors)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocity estimation
    ax = axes[0, 1]
    ax.plot(times, true_vz, 'k-', linewidth=2, label='True velocity')
    ax.plot(times, est_vz_normal, 'b-', linewidth=2, label='KF estimate')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Attack scenario
    ax = axes[1, 0]
    ax.plot(times_atk, true_z_atk, 'k-', linewidth=2, label='True altitude')
    ax.plot(times_atk, meas_z_atk, 'r.', alpha=0.3, markersize=2, label='Compromised meas')
    ax.plot(times_atk, est_z_attack, 'b-', linewidth=2, label='KF estimate')
    ax.axvline(5.0, color='orange', linestyle='--', linewidth=2, label='Attack starts')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Attack Scenario (Prediction-Only After 5s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Estimation error
    ax = axes[1, 1]
    error_normal = np.abs(np.array(est_z_normal) - true_z) * 100
    error_attack = np.abs(np.array(est_z_attack) - true_z_atk) * 100
    ax.plot(times, error_normal, 'b-', linewidth=2, label='Normal mode')
    ax.plot(times_atk, error_attack, 'r-', linewidth=2, label='Prediction-only')
    ax.axvline(5.0, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimation Error [cm]')
    ax.set_title('Altitude Estimation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kf_simulation.png', dpi=200, bbox_inches='tight')
    print("\n[+] Plot saved: kf_simulation.png")
    
    # Calculate metrics
    rmse_normal = np.sqrt(np.mean((np.array(est_z_normal) - true_z)**2))
    rmse_attack = np.sqrt(np.mean((np.array(est_z_attack) - true_z_atk)**2))
    
    print(f"\nNormal mode RMSE: {rmse_normal*100:.2f} cm")
    print(f"Attack mode RMSE: {rmse_attack*100:.2f} cm")
    print(f"Drift during prediction (last 5s): {error_attack[-1]:.2f} cm")
    
    plt.show()


if __name__ == "__main__":
    # Run simulation test when executed directly
    import os
    os.makedirs("Lab3-Phase2", exist_ok=True)
    test_kalman_filter()