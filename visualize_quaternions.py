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
    
    # Extract all samples from all windows with NaN separators between windows
    # This prevents matplotlib from drawing lines between disconnected windows
    timestamps1 = []
    qw1, qx1, qy1, qz1 = [], [], [], []
    
    for window in device1_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            # Falls back to 'timestamp_ms' for backward compatibility with old data
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps1.append(timestamp)
            qw1.append(sample['qw'])
            qx1.append(sample['qx'])
            qy1.append(sample['qy'])
            qz1.append(sample['qz'])
        # Add NaN separator between windows to create line breaks
        timestamps1.append(np.nan)
        qw1.append(np.nan)
        qx1.append(np.nan)
        qy1.append(np.nan)
        qz1.append(np.nan)
    
    timestamps2 = []
    qw2, qx2, qy2, qz2 = [], [], [], []
    
    for window in device2_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            # Falls back to 'timestamp_ms' for backward compatibility with old data
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps2.append(timestamp)
            qw2.append(sample['qw'])
            qx2.append(sample['qx'])
            qy2.append(sample['qy'])
            qz2.append(sample['qz'])
        # Add NaN separator between windows to create line breaks
        timestamps2.append(np.nan)
        qw2.append(np.nan)
        qx2.append(np.nan)
        qy2.append(np.nan)
        qz2.append(np.nan)
    
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
    """Check and visualize time synchronization between IMUs by interpolating to common timestamps"""
    # Get device MAC addresses
    devices = list(data['device_windows'].keys())
    device1_data = data['device_windows'][devices[0]]
    device2_data = data['device_windows'][devices[1]]
    
    # Get device names
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Extract timestamps from all windows
    # Note: Windows are pre-sorted chronologically at data collection time
    timestamps1 = []
    for window in device1_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps1.append(timestamp)
    
    timestamps2 = []
    for window in device2_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps2.append(timestamp)
    
    timestamps1 = np.array(timestamps1)
    timestamps2 = np.array(timestamps2)
    
    # Find overlapping time range
    common_start = max(timestamps1.min(), timestamps2.min())
    common_end = min(timestamps1.max(), timestamps2.max())
    
    # Create common time grid (every 10ms for 100Hz sampling)
    common_times = np.arange(common_start, common_end, 10)
    
    # For each common time point, find the nearest timestamp from each device
    # and calculate the difference
    time_diffs = []
    actual_times = []
    
    for t in common_times:
        # Find nearest timestamps in both devices
        idx1 = np.argmin(np.abs(timestamps1 - t))
        idx2 = np.argmin(np.abs(timestamps2 - t))
        
        # Only include if both timestamps are reasonably close to the target time
        if abs(timestamps1[idx1] - t) < 50 and abs(timestamps2[idx2] - t) < 50:
            time_diff = timestamps1[idx1] - timestamps2[idx2]
            time_diffs.append(time_diff)
            actual_times.append(t)
    
    time_diffs = np.array(time_diffs)
    actual_times = np.array(actual_times)
    
    plt.figure(figsize=(12, 4))
    plt.plot(actual_times, time_diffs, linewidth=1.5, alpha=0.7)
    plt.title(f'Timestamp Synchronization ({device1_name} - {device2_name})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Time Difference (ms)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Sync', linewidth=2)
    plt.legend()
    
    # Print statistics
    print("\n=== Time Synchronization Analysis ===")
    print(f"Analyzed {len(time_diffs)} aligned time points")
    print(f"Mean time difference: {np.mean(time_diffs):.2f} ms")
    print(f"Std time difference: {np.std(time_diffs):.2f} ms")
    print(f"Median time difference: {np.median(time_diffs):.2f} ms")
    print(f"Min time difference: {np.min(time_diffs):.2f} ms")
    print(f"Max time difference: {np.max(time_diffs):.2f} ms")
    print(f"Time difference range: {np.max(time_diffs) - np.min(time_diffs):.2f} ms")
    
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
    
    # Extract timestamps and calculate magnitudes with NaN separators between windows
    timestamps1 = []
    mag1 = []
    for window in device1_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps1.append(timestamp)
            mag = np.sqrt(sample['qw']**2 + sample['qx']**2 + sample['qy']**2 + sample['qz']**2)
            mag1.append(mag)
        # Add NaN separator between windows to create line breaks
        timestamps1.append(np.nan)
        mag1.append(np.nan)
    
    timestamps2 = []
    mag2 = []
    for window in device2_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps2.append(timestamp)
            mag = np.sqrt(sample['qw']**2 + sample['qx']**2 + sample['qy']**2 + sample['qz']**2)
            mag2.append(mag)
        # Add NaN separator between windows to create line breaks
        timestamps2.append(np.nan)
        mag2.append(np.nan)
    
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

def analyze_clock_drift(data):
    """Analyze clock drift between the two IMU sensors"""
    print("\n=== Clock Drift Analysis ===")
    
    # Get device MAC addresses
    devices = list(data['device_windows'].keys())
    device1_data = data['device_windows'][devices[0]]
    device2_data = data['device_windows'][devices[1]]
    
    # Get device names
    device1_name = data['session_metadata']['devices'][devices[0]]['node_name']
    device2_name = data['session_metadata']['devices'][devices[1]]['node_name']
    
    # Extract timestamps from all samples
    timestamps1 = []
    for window in device1_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps1.append(timestamp)
    
    timestamps2 = []
    for window in device2_data:
        for sample in window['samples']:
            # Use 'absolute_timestamp_ms' for global synchronized time
            timestamp = sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0))
            timestamps2.append(timestamp)
    
    if not timestamps1 or not timestamps2:
        print("Insufficient data for drift analysis")
        return
    
    # Sort timestamps to ensure proper ordering
    timestamps1 = sorted(timestamps1)
    timestamps2 = sorted(timestamps2)
    
    # Calculate time range for each device
    duration1 = (timestamps1[-1] - timestamps1[0]) / 1000.0  # seconds
    duration2 = (timestamps2[-1] - timestamps2[0]) / 1000.0  # seconds
    
    print(f"{device1_name} time range: {timestamps1[0]:.0f}ms to {timestamps1[-1]:.0f}ms (duration: {duration1:.2f}s)")
    print(f"{device2_name} time range: {timestamps2[0]:.0f}ms to {timestamps2[-1]:.0f}ms (duration: {duration2:.2f}s)")
    
    # Find common time range for drift calculation
    common_start = max(timestamps1[0], timestamps2[0])
    common_end = min(timestamps1[-1], timestamps2[-1])
    
    # Filter timestamps to common range
    common_ts1 = [t for t in timestamps1 if common_start <= t <= common_end]
    common_ts2 = [t for t in timestamps2 if common_start <= t <= common_end]
    
    if len(common_ts1) < 2 or len(common_ts2) < 2:
        print("Insufficient overlap for drift calculation")
        return
    
    # Calculate average sampling intervals (should be ~10ms at 100Hz)
    intervals1 = np.diff(common_ts1)
    intervals2 = np.diff(common_ts2)
    
    avg_interval1 = np.mean(intervals1)
    avg_interval2 = np.mean(intervals2)
    
    print(f"\n{device1_name} average sample interval: {avg_interval1:.3f}ms (std: {np.std(intervals1):.3f}ms)")
    print(f"{device2_name} average sample interval: {avg_interval2:.3f}ms (std: {np.std(intervals2):.3f}ms)")
    print(f"Expected interval: 10.000ms at 100Hz sampling rate")
    
    # Calculate relative drift between devices
    # Drift = difference in average sampling rate over time
    common_duration = (common_end - common_start) / 1000.0  # seconds
    samples1 = len(common_ts1)
    samples2 = len(common_ts2)
    
    rate1 = samples1 / common_duration if common_duration > 0 else 0
    rate2 = samples2 / common_duration if common_duration > 0 else 0
    
    print(f"\n{device1_name} effective sampling rate: {rate1:.2f} Hz")
    print(f"{device2_name} effective sampling rate: {rate2:.2f} Hz")
    
    # Calculate clock drift in ppm (parts per million)
    # Positive drift means device is running fast
    drift1_ppm = ((avg_interval1 - 10.0) / 10.0) * 1e6 if avg_interval1 > 0 else 0
    drift2_ppm = ((avg_interval2 - 10.0) / 10.0) * 1e6 if avg_interval2 > 0 else 0
    
    print(f"\n{device1_name} clock drift: {drift1_ppm:.0f} ppm ({avg_interval1 - 10.0:+.3f}ms per sample)")
    print(f"{device2_name} clock drift: {drift2_ppm:.0f} ppm ({avg_interval2 - 10.0:+.3f}ms per sample)")
    
    # Calculate cumulative drift over session
    total_drift1 = (avg_interval1 - 10.0) * samples1
    total_drift2 = (avg_interval2 - 10.0) * samples2
    
    print(f"\nCumulative drift over session:")
    print(f"{device1_name}: {total_drift1:+.1f}ms over {samples1} samples")
    print(f"{device2_name}: {total_drift2:+.1f}ms over {samples2} samples")
    
    # Check for large timing jumps (wraparounds or re-syncs)
    max_jump1 = np.max(intervals1) if len(intervals1) > 0 else 0
    max_jump2 = np.max(intervals2) if len(intervals2) > 0 else 0
    
    large_jumps1 = np.sum(intervals1 > 100)  # Jumps > 100ms
    large_jumps2 = np.sum(intervals2 > 100)  # Jumps > 100ms
    
    print(f"\nLarge timing jumps (>100ms):")
    print(f"{device1_name}: {large_jumps1} jumps (max: {max_jump1:.1f}ms)")
    print(f"{device2_name}: {large_jumps2} jumps (max: {max_jump2:.1f}ms)")
    
    # Warnings
    if abs(drift1_ppm) > 1000 or abs(drift2_ppm) > 1000:
        print("\nWARNING: Clock drift exceeds 1000 ppm (0.1%)")
        print("    This may indicate timing issues with the Arduino millis()/micros() functions")
    
    if large_jumps1 > 0 or large_jumps2 > 0:
        print("\nWARNING: Large timing jumps detected")
        print("    This may indicate uint16_t wraparounds or time re-synchronization events")

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
    
    # Analyze clock drift
    analyze_clock_drift(data)

if __name__ == "__main__":
    main()
