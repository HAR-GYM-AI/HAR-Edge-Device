import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def load_data(filepath):
    """Load IMU data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_quaternions(data, output_file=None):
    """Plot quaternion components for both IMUs"""
    # Get device MAC addresses
    devices = list(data['device_windows'].keys())
    device1_data = data['device_windows'][devices[0]]
    device2_data = data['device_windows'][devices[1]]
    
    # Get device names
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Extract all samples from all windows
    timestamps1 = []
    qw1, qx1, qy1, qz1 = [], [], [], []
    
    for window in device1_data:
        for sample in window['samples']:
            timestamps1.append(sample['timestamp'])
            qw1.append(sample['qw'])
            qx1.append(sample['qx'])
            qy1.append(sample['qy'])
            qz1.append(sample['qz'])
    
    timestamps2 = []
    qw2, qx2, qy2, qz2 = [], [], [], []
    
    for window in device2_data:
        for sample in window['samples']:
            timestamps2.append(sample['timestamp'])
            qw2.append(sample['qw'])
            qx2.append(sample['qx'])
            qy2.append(sample['qy'])
            qz2.append(sample['qz'])
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'IMU Quaternion Comparison - {device1_name} vs {device2_name}', fontsize=16)
    
    components = [('W', qw1, qw2), ('X', qx1, qx2), ('Y', qy1, qy2), ('Z', qz1, qz2)]
    
    for idx, (name, comp1, comp2) in enumerate(components):
        axes[idx].plot(timestamps1, comp1, label=f'{device1_name} - Q{name}', alpha=0.7, linewidth=1.5)
        axes[idx].plot(timestamps2, comp2, label=f'{device2_name} - Q{name}', alpha=0.7, linewidth=1.5)
        axes[idx].set_ylabel(f'Q{name}')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestamp (ms)')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

def plot_time_sync(data, output_file=None):
    """Check and visualize time synchronization between IMUs"""
    # Get device MAC addresses
    devices = list(data['device_windows'].keys())
    device1_data = data['device_windows'][devices[0]]
    device2_data = data['device_windows'][devices[1]]
    
    # Get device names
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Extract timestamps from all windows
    timestamps1 = []
    for window in device1_data:
        for sample in window['samples']:
            timestamps1.append(sample['timestamp'])
    
    timestamps2 = []
    for window in device2_data:
        for sample in window['samples']:
            timestamps2.append(sample['timestamp'])
    
    timestamps1 = np.array(timestamps1)
    timestamps2 = np.array(timestamps2)
    
    # Calculate time differences
    min_len = min(len(timestamps1), len(timestamps2))
    time_diff = timestamps1[:min_len] - timestamps2[:min_len]
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_diff, linewidth=1.5)
    plt.title(f'Timestamp Difference ({device1_name} - {device2_name})')
    plt.xlabel('Sample Index')
    plt.ylabel('Time Difference (ms)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Sync', linewidth=2)
    plt.legend()
    
    # Print statistics
    print("\n=== Time Synchronization Analysis ===")
    print(f"Mean time difference: {np.mean(time_diff):.2f} ms")
    print(f"Std time difference: {np.std(time_diff):.2f} ms")
    print(f"Max time difference: {np.max(np.abs(time_diff)):.2f} ms")
    print(f"Min time difference: {np.min(time_diff):.2f} ms")
    print(f"Max time difference: {np.max(time_diff):.2f} ms")
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Sync plot saved to: {output_file}")
    
    plt.show()

def plot_quaternion_magnitude(data, output_file=None):
    """Plot quaternion magnitudes to verify normalization"""
    # Get device MAC addresses
    devices = list(data['device_windows'].keys())
    device1_data = data['device_windows'][devices[0]]
    device2_data = data['device_windows'][devices[1]]
    
    # Get device names
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Extract timestamps and calculate magnitudes
    timestamps1 = []
    mag1 = []
    for window in device1_data:
        for sample in window['samples']:
            timestamps1.append(sample['timestamp'])
            mag = np.sqrt(sample['qw']**2 + sample['qx']**2 + sample['qy']**2 + sample['qz']**2)
            mag1.append(mag)
    
    timestamps2 = []
    mag2 = []
    for window in device2_data:
        for sample in window['samples']:
            timestamps2.append(sample['timestamp'])
            mag = np.sqrt(sample['qw']**2 + sample['qx']**2 + sample['qy']**2 + sample['qz']**2)
            mag2.append(mag)
    
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps1, mag1, label=f'{device1_name} Magnitude', alpha=0.7, linewidth=1.5)
    plt.plot(timestamps2, mag2, label=f'{device2_name} Magnitude', alpha=0.7, linewidth=1.5)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Expected (1.0)', linewidth=2)
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Quaternion Magnitude')
    plt.title('Quaternion Magnitude Verification (Should be ~1.0 for normalized quaternions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.95, 1.05])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Magnitude plot saved to: {output_file}")
    
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_quaternions.py <data_file.json> [--save]")
        print("Example: python visualize_quaternions.py collected_data/session_2_BICEP_CURL_20251008_125800.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    save_plots = '--save' in sys.argv
    
    print(f"Loading data from: {filepath}")
    data = load_data(filepath)
    
    # Get device information
    devices = list(data['device_windows'].keys())
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Count total samples
    samples1 = sum(len(window['samples']) for window in data['device_windows'][devices[0]])
    samples2 = sum(len(window['samples']) for window in data['device_windows'][devices[1]])
    
    print(f"\n{device1_name} ({devices[0]}) samples: {samples1}")
    print(f"{device2_name} ({devices[1]}) samples: {samples2}")
    print(f"Exercise type: {data['session_metadata']['exercise_type']}")
    print(f"Participant ID: {data['session_metadata']['participant_id']}")
    print(f"Duration: {data['session_metadata']['duration_seconds']:.2f} seconds")
    
    # Generate output filenames if saving
    base_name = filepath.replace('.json', '')
    quat_output = f"{base_name}_quaternions.png" if save_plots else None
    sync_output = f"{base_name}_sync.png" if save_plots else None
    mag_output = f"{base_name}_magnitude.png" if save_plots else None
    
    # Plot quaternions
    print("\n=== Plotting Quaternion Components ===")
    plot_quaternions(data, quat_output)
    
    # Plot time synchronization
    print("\n=== Checking Time Synchronization ===")
    plot_time_sync(data, sync_output)
    
    # Plot quaternion magnitudes
    print("\n=== Verifying Quaternion Normalization ===")
    plot_quaternion_magnitude(data, mag_output)

if __name__ == "__main__":
    main()
