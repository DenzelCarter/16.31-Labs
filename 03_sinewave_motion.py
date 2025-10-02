"""
Tello Drone Control with Sine Wave Pattern using djitellopy
"""
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter, sleep
from djitellopy import Tello
from typing import List, Dict, Any

def convert_velocity_to_rc(velocity_ms: float, max_velocity_ms: float = 2.0) -> int:
    """
    Convert velocity in m/s to djitellopy RC control value (-100 to 100)
    
    Args:
        velocity_ms: Velocity in m/s
        max_velocity_ms: Maximum expected velocity in m/s (for scaling)
    
    Returns:
        RC control value between -100 and 100
    """
    # Scale velocity to -100/100 range
    rc_value = (velocity_ms / max_velocity_ms) * 100
    # Clamp to valid range
    return max(-100, min(100, int(rc_value)))

def get_sensor_readings(tello: Tello, start_time: float) -> Dict[str, Any]:
    """
    Collect all available sensor readings from the Tello drone
    
    Args:
        tello: Tello drone object
        start_time: Start time for relative timing
    
    Returns:
        Dictionary containing sensor readings
    """
    try:
        readings = {
            'timestamp': perf_counter(),
            'flight_time': start_time + (perf_counter() - start_time),  # Time since start
            
            # Acceleration data (in acceleration units, typically 0.001g)
            'acceleration_x': tello.get_acceleration_x() * 0.001 * 9.81,  # Convert to m/s²
            'acceleration_y': tello.get_acceleration_y() * 0.001 * 9.81,  # Convert to m/s²
            'acceleration_z': tello.get_acceleration_z() * 0.001 * 9.81,  # Convert to m/s²
            
            # Velocity data (cm/s to m/s)
            'speed_x': tello.get_speed_x() * 0.01,  # Convert cm/s to m/s
            'speed_y': tello.get_speed_y() * 0.01,  # Convert cm/s to m/s
            'speed_z': tello.get_speed_z() * 0.01,  # Convert cm/s to m/s
            
            # Attitude data
            'height': tello.get_height() * 0.01,  # Convert cm to m
            'battery': tello.get_battery(),
            'temperature': tello.get_temperature(),
            
            # IMU data (if available)
            'pitch': tello.get_pitch(),
            'roll': tello.get_roll(), 
            'yaw': tello.get_yaw(),
        }
        
        # Try to get additional attitude data
        try:
            attitude = tello.query_attitude()
            if isinstance(attitude, dict):
                readings.update(attitude)
        except:
            pass  # Attitude query might not always work
            
        return readings
        
    except Exception as e:
        print(f"Error reading sensors: {e}")
        # Return basic readings with current time
        return {
            'timestamp': perf_counter(),
            'flight_time': start_time + (perf_counter() - start_time),
            'acceleration_x': 0.0,
            'acceleration_y': 0.0, 
            'acceleration_z': 0.0,
            'speed_x': 0.0,
            'speed_y': 0.0,
            'speed_z': 0.0,
            'height': 0.0,
            'battery': 0,
            'temperature': 0,
            'pitch': 0,
            'roll': 0,
            'yaw': 0
        }

def aggregate_sensor_readings(readings_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert list of sensor readings to numpy arrays for plotting
    
    Args:
        readings_list: List of sensor reading dictionaries
    
    Returns:
        Dictionary with numpy arrays for each sensor type
    """
    if not readings_list:
        return {}
    
    # Get all unique keys from all readings
    all_keys = set()
    for reading in readings_list:
        all_keys.update(reading.keys())
    
    aggregated = {}
    for key in all_keys:
        values = []
        for reading in readings_list:
            values.append(reading.get(key, 0))
        
        # Convert to numpy array
        aggregated[key] = np.array(values)
    
    # Create acceleration vector
    if all(k in aggregated for k in ['acceleration_x', 'acceleration_y', 'acceleration_z']):
        aggregated['acceleration'] = np.column_stack([
            aggregated['acceleration_x'],
            aggregated['acceleration_y'], 
            aggregated['acceleration_z']
        ])
    
    # Create speed vector  
    if all(k in aggregated for k in ['speed_x', 'speed_y', 'speed_z']):
        aggregated['speed'] = np.column_stack([
            aggregated['speed_x'],
            aggregated['speed_y'],
            aggregated['speed_z']
        ])
    
    return aggregated

def main():
    """Main control loop for sine wave drone flight pattern"""
    
    print("Initializing Tello drone...")
    tello = Tello()
    
    try:
        # Connect to drone
        print("Connecting to drone...")
        tello.connect()
        
        # Check battery level
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        
        if battery < 20:
            print("Warning: Low battery level! Consider charging before flight.")
            return
        
        # Set speed (optional)
        tello.set_speed(50)  # Set to moderate speed
        
        # Take off
        print("Taking off...")
        tello.takeoff()
        sleep(3)  # Allow drone to stabilize
        
        # Flight parameters
        flight_time = 10.0  # seconds
        framerate = 20.0   # Hz (reduced from 100Hz for more realistic control)
        max_velocity = 1.0  # m/s - maximum velocity for scaling
        
        print(f"Starting {flight_time}s sine wave flight pattern...")
        
        # Data collection lists
        readings = []
        controls = []
        times = []
        
        start_time = perf_counter()
        
        # Main control loop
        for t in np.arange(0, flight_time, 1 / framerate):
            loop_start = perf_counter()
            times.append(loop_start)
            
            # Get sensor readings
            sensor_data = get_sensor_readings(tello, start_time)
            readings.append(sensor_data)
            
            # Calculate control commands (sine wave in Z axis)
            yaw_rate = 0.0  # rad/s
            xyz_velocity = np.array([
                0.0,  # x velocity (m/s)
                0.0,  # y velocity (m/s)
                0.5 * np.sin(4 * np.pi * t / flight_time),  # z velocity (m/s) - sine wave
            ])
            
            # Store control command
            controls.append(xyz_velocity.copy())
            
            # Convert to RC control values
            left_right = convert_velocity_to_rc(xyz_velocity[0], max_velocity)      # x -> left/right
            forward_backward = convert_velocity_to_rc(xyz_velocity[1], max_velocity) # y -> forward/backward  
            up_down = convert_velocity_to_rc(xyz_velocity[2], max_velocity)         # z -> up/down
            yaw_velocity = convert_velocity_to_rc(yaw_rate * 0.5, 1.0)             # yaw rate to yaw velocity
            
            # Send control command
            tello.send_rc_control(left_right, forward_backward, up_down, yaw_velocity)
            
            # Debug output (every second)
            if int(t * framerate) % int(framerate) == 0:
                print(f"t={t:.1f}s: vel_z={xyz_velocity[2]:.2f} m/s, "
                      f"rc_up_down={up_down}, height={sensor_data.get('height', 0):.2f}m, "
                      f"battery={sensor_data.get('battery', 0)}%")
            
            # Maintain frame rate
            loop_time = perf_counter() - loop_start
            sleep_time = (1 / framerate) - loop_time
            if sleep_time > 0:
                sleep(sleep_time)
        
        # Stop movement and land
        print("Landing...")
        tello.send_rc_control(0, 0, 0, 0)  # Stop all movement
        sleep(1)
        tello.land()
        
        print("Flight completed successfully!")
        
        # Convert data for plotting
        print("Processing data for plotting...")
        readings_dict = aggregate_sensor_readings(readings)
        controls_np = np.array(controls)
        times_np = np.array(times) - start_time  # Relative time
        
        # Create plots
        plt.figure(figsize=(12, 8))
        
        # Plot acceleration data
        plt.subplot(2, 1, 1)
        if 'acceleration' in readings_dict:
            plt.plot(readings_dict['flight_time'], readings_dict['acceleration'][:, 0], 
                    linestyle='-', label='accel_x', alpha=0.7)
            plt.plot(readings_dict['flight_time'], readings_dict['acceleration'][:, 1], 
                    linestyle='-', label='accel_y', alpha=0.7) 
            plt.plot(readings_dict['flight_time'], readings_dict['acceleration'][:, 2], 
                    linestyle='-', label='accel_z', alpha=0.7)
        
        # Plot control commands
        plt.plot(readings_dict['flight_time'], controls_np[:, 0], 
                linestyle='--', label='cmd_vx', linewidth=2)
        plt.plot(readings_dict['flight_time'], controls_np[:, 1], 
                linestyle='--', label='cmd_vy', linewidth=2)
        plt.plot(readings_dict['flight_time'], controls_np[:, 2], 
                linestyle='--', label='cmd_vz', linewidth=2)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²) / Velocity Commands (m/s)')
        plt.title('Tello Drone: Acceleration vs Velocity Commands')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot additional telemetry
        plt.subplot(2, 1, 2)
        if 'height' in readings_dict:
            plt.plot(readings_dict['flight_time'], readings_dict['height'], 
                    label='Height (m)', color='green')
        
        if 'battery' in readings_dict:
            plt.plot(readings_dict['flight_time'], readings_dict['battery'], 
                    label='Battery (%)', color='red')
            
        plt.xlabel('Time (s)')
        plt.ylabel('Various Units')
        plt.title('Tello Drone: Height and Battery During Flight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Flight Summary ===")
        print(f"Flight duration: {flight_time:.1f}s")
        print(f"Data points collected: {len(readings)}")
        if 'height' in readings_dict:
            print(f"Height range: {readings_dict['height'].min():.2f} - {readings_dict['height'].max():.2f} m")
        if 'battery' in readings_dict:
            print(f"Battery used: {readings_dict['battery'][0] - readings_dict['battery'][-1]:.0f}%")
        
    except Exception as e:
        print(f"Error during flight: {e}")
        try:
            # Emergency landing
            print("Attempting emergency landing...")
            tello.send_rc_control(0, 0, 0, 0)  # Stop movement
            tello.land()
        except:
            print("Could not land automatically - manual intervention required!")
    
    finally:
        # Cleanup
        try:
            tello.end()
            print("Disconnected from drone")
        except:
            pass

if __name__ == "__main__":
    main()