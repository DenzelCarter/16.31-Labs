"""
An example of using djitellopy to get sensor readings from the drone.
"""
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from djitellopy import Tello
from typing import List, Dict, Any

def get_sensor_readings(tello: Tello) -> Dict[str, Any]:
    """
    Get comprehensive sensor readings from the Tello drone
    
    Args:
        tello: Tello drone object
    
    Returns:
        Dictionary containing all available sensor readings
    """
    try:
        readings = {
            # Acceleration data (convert from 0.001g units to m/s²)
            'acceleration_x': tello.get_acceleration_x() * 0.001 * 9.81,
            'acceleration_y': tello.get_acceleration_y() * 0.001 * 9.81,
            'acceleration_z': tello.get_acceleration_z() * 0.001 * 9.81,
            
            # Velocity data (convert from cm/s to m/s)
            'speed_x': tello.get_speed_x() * 0.01,
            'speed_y': tello.get_speed_y() * 0.01,
            'speed_z': tello.get_speed_z() * 0.01,
            
            # Position and orientation
            'height': tello.get_height() * 0.01,  # Convert cm to m
            'pitch': tello.get_pitch(),  # degrees
            'roll': tello.get_roll(),    # degrees
            'yaw': tello.get_yaw(),      # degrees
            
            # Environmental and status
            'battery': tello.get_battery(),      # percentage
            'temperature': tello.get_temperature(),  # Celsius
            'flight_time': tello.get_flight_time(),  # seconds since takeoff
            
            # Distance sensor
            'distance_tof': tello.get_distance_tof() * 0.01,  # Convert cm to m
        }
        
        # Try to get barometer reading (may not be available on all models)
        try:
            readings['barometer'] = tello.get_barometer() * 0.01  # Convert cm to m
        except:
            readings['barometer'] = 0.0
            
        return readings
        
    except Exception as e:
        print(f"Error reading sensors: {e}")
        # Return zero readings if there's an error
        return {
            'acceleration_x': 0.0, 'acceleration_y': 0.0, 'acceleration_z': 0.0,
            'speed_x': 0.0, 'speed_y': 0.0, 'speed_z': 0.0,
            'height': 0.0, 'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
            'battery': 0, 'temperature': 0, 'flight_time': 0,
            'distance_tof': 0.0, 'barometer': 0.0
        }

def aggregate_sensor_readings(readings_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert list of sensor readings to numpy arrays for analysis and plotting
    
    Args:
        readings_list: List of sensor reading dictionaries
    
    Returns:
        Dictionary with numpy arrays for each sensor type
    """
    if not readings_list:
        return {}
    
    # Initialize dictionary to store aggregated data
    aggregated = {}
    
    # Get all sensor keys from the first reading
    sensor_keys = readings_list[0].keys()
    
    # Convert each sensor type to numpy array
    for key in sensor_keys:
        values = [reading[key] for reading in readings_list]
        aggregated[key] = np.array(values)
    
    # Create composite acceleration vector (3D)
    if all(k in aggregated for k in ['acceleration_x', 'acceleration_y', 'acceleration_z']):
        aggregated['acceleration'] = np.column_stack([
            aggregated['acceleration_x'],
            aggregated['acceleration_y'],
            aggregated['acceleration_z']
        ])
    
    # Create composite speed vector (3D)
    if all(k in aggregated for k in ['speed_x', 'speed_y', 'speed_z']):
        aggregated['speed'] = np.column_stack([
            aggregated['speed_x'],
            aggregated['speed_y'],
            aggregated['speed_z']
        ])
    
    # Create composite attitude vector (roll, pitch, yaw)
    if all(k in aggregated for k in ['roll', 'pitch', 'yaw']):
        aggregated['attitude'] = np.column_stack([
            aggregated['roll'],
            aggregated['pitch'],
            aggregated['yaw']
        ])
    
    return aggregated

def main():
    """Main function to collect and plot sensor readings"""
    
    print("Initializing Tello drone...")
    tello = Tello()
    
    try:
        # Connect to drone
        print("Connecting to drone...")
        tello.connect()
        
        # Check battery level
        battery = tello.get_battery()
        print(f"Battery level: {battery}%")
        
        # Flight parameters
        flight_time = 5.0   # seconds
        framerate = 10.0    # Hz
        num_readings = int(flight_time * framerate)
        
        print(f"Starting {num_readings} measurements over {flight_time} seconds!")
        print("Note: Drone will remain on the ground for safety")
        
        # Collect sensor readings
        readings = []
        for i in range(num_readings):
            reading = get_sensor_readings(tello)
            readings.append(reading)
            
            # Print progress every second
            if i % int(framerate) == 0:
                print(f"Progress: {i//int(framerate)+1}/{int(flight_time)}s - "
                      f"Accel: ({reading['acceleration_x']:.2f}, {reading['acceleration_y']:.2f}, {reading['acceleration_z']:.2f}) m/s²")
            
            sleep(1 / framerate)
        
        print("Done with measurements!")
        
        # Convert readings to numpy arrays for plotting
        readings_dict = aggregate_sensor_readings(readings)
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tello Drone Sensor Readings (Ground Static Test)', fontsize=16)
        
        # Measurement indices for x-axis (since flight_time will be zero on ground)
        measurement_indices = np.arange(len(readings))
        
        # Plot 1: Acceleration
        axes[0, 0].plot(measurement_indices, readings_dict["acceleration"][:, 0], 'r-', label="X", alpha=0.8)
        axes[0, 0].plot(measurement_indices, readings_dict["acceleration"][:, 1], 'g-', label="Y", alpha=0.8)
        axes[0, 0].plot(measurement_indices, readings_dict["acceleration"][:, 2], 'b-', label="Z", alpha=0.8)
        axes[0, 0].set_xlabel("Measurement #")
        axes[0, 0].set_ylabel("Acceleration (m/s²)")
        axes[0, 0].set_title("3-Axis Acceleration")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Attitude (Roll, Pitch, Yaw)
        axes[0, 1].plot(measurement_indices, readings_dict["roll"], 'r-', label="Roll", alpha=0.8)
        axes[0, 1].plot(measurement_indices, readings_dict["pitch"], 'g-', label="Pitch", alpha=0.8)
        axes[0, 1].plot(measurement_indices, readings_dict["yaw"], 'b-', label="Yaw", alpha=0.8)
        axes[0, 1].set_xlabel("Measurement #")
        axes[0, 1].set_ylabel("Angle (degrees)")
        axes[0, 1].set_title("Attitude (Roll, Pitch, Yaw)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Environmental Data
        ax3_temp = axes[1, 0]
        ax3_batt = ax3_temp.twinx()
        
        line1 = ax3_temp.plot(measurement_indices, readings_dict["temperature"], 'r-', label="Temperature")
        line2 = ax3_batt.plot(measurement_indices, readings_dict["battery"], 'b-', label="Battery")
        
        ax3_temp.set_xlabel("Measurement #")
        ax3_temp.set_ylabel("Temperature (°C)", color='r')
        ax3_batt.set_ylabel("Battery (%)", color='b')
        ax3_temp.set_title("Environmental & Status Data")
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3_temp.legend(lines, labels, loc='upper left')
        ax3_temp.grid(True, alpha=0.3)
        
        # Plot 4: Height and Distance Sensors
        axes[1, 1].plot(measurement_indices, readings_dict["height"], 'g-', label="Height", alpha=0.8)
        axes[1, 1].plot(measurement_indices, readings_dict["distance_tof"], 'orange', label="ToF Distance", alpha=0.8)
        if readings_dict["barometer"].max() > 0:  # Only plot if barometer data is available
            axes[1, 1].plot(measurement_indices, readings_dict["barometer"], 'purple', label="Barometer", alpha=0.8)
        axes[1, 1].set_xlabel("Measurement #")
        axes[1, 1].set_ylabel("Distance (m)")
        axes[1, 1].set_title("Distance Sensors")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Sensor Reading Summary ===")
        print(f"Total measurements: {len(readings)}")
        print(f"Acceleration ranges:")
        print(f"  X: {readings_dict['acceleration_x'].min():.3f} to {readings_dict['acceleration_x'].max():.3f} m/s²")
        print(f"  Y: {readings_dict['acceleration_y'].min():.3f} to {readings_dict['acceleration_y'].max():.3f} m/s²")
        print(f"  Z: {readings_dict['acceleration_z'].min():.3f} to {readings_dict['acceleration_z'].max():.3f} m/s²")
        print(f"Temperature range: {readings_dict['temperature'].min():.1f} to {readings_dict['temperature'].max():.1f} °C")
        print(f"Battery level: {readings_dict['battery'][-1]}%")
        
    except Exception as e:
        print(f"Error during sensor reading: {e}")
    
    finally:
        # Cleanup
        try:
            tello.end()
            print("Disconnected from drone")
        except:
            pass

if __name__ == "__main__":
    main() 