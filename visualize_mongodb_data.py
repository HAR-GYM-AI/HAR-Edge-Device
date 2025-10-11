#!/usr/bin/env python3
"""
MongoDB Data Visualization Script for HAR System
Visualizes training data collected from MongoDB to assess data quality
and demonstrate collection success.

This script provides:
1. Raw quaternion movement graphs per exercise and sensor
2. Data distribution analysis
3. Sample count statistics
4. Quality metrics for ML training data
5. Temporal alignment visualization
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
from collections import defaultdict
import seaborn as sns

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'har_system')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'training_data')

# Exercise names mapping
EXERCISE_NAMES = {
    1: "Bicep Curl",
    2: "Bench Press",
    3: "Squat",
    4: "Pull Up"
}

# Node placement names
NODE_NAMES = {
    0: "WRIST",
    1: "BICEP",
    2: "CHEST",
    3: "THIGH"
}

# Color scheme for consistent plotting
COLORS = {
    'qw': '#1f77b4',
    'qx': '#ff7f0e',
    'qy': '#2ca02c',
    'qz': '#d62728'
}


def connect_to_mongodb():
    """Connect to MongoDB and return collection"""
    try:
        print(f"Connecting to MongoDB at {MONGODB_URI}...")
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        print(f"✓ Connected to {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        return client, collection
    except Exception as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        sys.exit(1)


def fetch_all_sessions(collection):
    """Fetch all training sessions from MongoDB"""
    try:
        sessions = list(collection.find({}))
        print(f"✓ Found {len(sessions)} training sessions")
        return sessions
    except Exception as e:
        print(f"✗ Error fetching sessions: {e}")
        return []


def extract_quaternion_data(session):
    """
    Extract quaternion data from a session organized by device and exercise
    
    Returns:
        dict: {device_address: {'node_name': str, 'node_id': int, 'data': [samples]}}
    """
    device_data = {}
    
    if 'device_windows' not in session:
        return device_data
    
    for device_address, windows in session['device_windows'].items():
        # Get device metadata
        device_info = session.get('session_metadata', {}).get('devices', {}).get(device_address, {})
        node_name = device_info.get('node_name', 'UNKNOWN')
        node_id = device_info.get('node_id', -1)
        
        samples = []
        for window in windows:
            for sample in window.get('samples', []):
                samples.append({
                    'timestamp': sample.get('absolute_timestamp_ms', sample.get('timestamp_ms', 0)),
                    'qw': sample.get('qw', 0),
                    'qx': sample.get('qx', 0),
                    'qy': sample.get('qy', 0),
                    'qz': sample.get('qz', 0),
                    'window_type': window.get('window_type', 'unknown')
                })
        
        device_data[device_address] = {
            'node_name': node_name,
            'node_id': node_id,
            'samples': samples
        }
    
    return device_data


def plot_raw_quaternions_per_exercise(sessions, output_dir='visualizations'):
    """
    Plot raw quaternion data for each exercise type
    One figure per exercise, showing all sensors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize sessions by exercise type
    exercise_sessions = defaultdict(list)
    for session in sessions:
        exercise_type = session.get('session_metadata', {}).get('exercise_type', 0)
        exercise_sessions[exercise_type].append(session)
    
    print(f"\n{'='*60}")
    print(f"PLOTTING RAW QUATERNION DATA PER EXERCISE")
    print(f"{'='*60}")
    
    for exercise_type, ex_sessions in exercise_sessions.items():
        exercise_name = EXERCISE_NAMES.get(exercise_type, f"Unknown Exercise {exercise_type}")
        print(f"\n📊 Processing {exercise_name} ({len(ex_sessions)} sessions)...")
        
        # Combine all sessions for this exercise
        all_device_data = defaultdict(lambda: {'samples': [], 'node_name': '', 'node_id': -1})
        
        for session in ex_sessions:
            device_data = extract_quaternion_data(session)
            for device_addr, data in device_data.items():
                all_device_data[device_addr]['samples'].extend(data['samples'])
                all_device_data[device_addr]['node_name'] = data['node_name']
                all_device_data[device_addr]['node_id'] = data['node_id']
        
        if not all_device_data:
            print(f"  ⚠ No data found for {exercise_name}")
            continue
        
        # Create figure with subplots for each sensor
        num_devices = len(all_device_data)
        fig, axes = plt.subplots(num_devices, 1, figsize=(16, 4*num_devices))
        if num_devices == 1:
            axes = [axes]
        
        fig.suptitle(f'{exercise_name} - Raw Quaternion Data', fontsize=16, fontweight='bold')
        
        for idx, (device_addr, data) in enumerate(sorted(all_device_data.items(), 
                                                          key=lambda x: x[1]['node_id'])):
            ax = axes[idx]
            samples = data['samples']
            node_name = data['node_name']
            
            # Sort samples by timestamp
            samples.sort(key=lambda x: x['timestamp'])
            
            # Extract data
            timestamps = np.array([s['timestamp'] for s in samples])
            # Normalize timestamps to start at 0
            if len(timestamps) > 0:
                timestamps = (timestamps - timestamps[0]) / 1000.0  # Convert to seconds
            
            qw = np.array([s['qw'] for s in samples])
            qx = np.array([s['qx'] for s in samples])
            qy = np.array([s['qy'] for s in samples])
            qz = np.array([s['qz'] for s in samples])
            
            # Plot quaternion components
            ax.plot(timestamps, qw, label='qw', color=COLORS['qw'], linewidth=1.0, alpha=0.8)
            ax.plot(timestamps, qx, label='qx', color=COLORS['qx'], linewidth=1.0, alpha=0.8)
            ax.plot(timestamps, qy, label='qy', color=COLORS['qy'], linewidth=1.0, alpha=0.8)
            ax.plot(timestamps, qz, label='qz', color=COLORS['qz'], linewidth=1.0, alpha=0.8)
            
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Quaternion Value', fontsize=10)
            ax.set_title(f'{node_name} Sensor ({len(samples)} samples)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1.1, 1.1])
            
            print(f"  ✓ {node_name}: {len(samples)} samples")
        
        plt.tight_layout()
        filename = f"{output_dir}/{exercise_name.replace(' ', '_').lower()}_raw_quaternions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
        plt.close()


def plot_data_distribution_summary(sessions, output_dir='visualizations'):
    """
    Create a comprehensive summary showing:
    - Sample counts per exercise
    - Session counts per exercise
    - Data distribution across sensors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CREATING DATA DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    
    # Collect statistics
    exercise_stats = defaultdict(lambda: {
        'sessions': 0,
        'total_samples': 0,
        'total_windows': 0,
        'sensor_samples': defaultdict(int),
        'total_reps': 0
    })
    
    for session in sessions:
        metadata = session.get('session_metadata', {})
        exercise_type = metadata.get('exercise_type', 0)
        
        exercise_stats[exercise_type]['sessions'] += 1
        exercise_stats[exercise_type]['total_samples'] += metadata.get('total_samples', 0)
        exercise_stats[exercise_type]['total_windows'] += metadata.get('total_windows', 0)
        exercise_stats[exercise_type]['total_reps'] += metadata.get('rep_count', 0)
        
        # Count samples per sensor
        device_data = extract_quaternion_data(session)
        for device_addr, data in device_data.items():
            node_name = data['node_name']
            exercise_stats[exercise_type]['sensor_samples'][node_name] += len(data['samples'])
    
    # Create summary figure
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Sessions per exercise (bar chart)
    ax1 = plt.subplot(2, 3, 1)
    exercises = [EXERCISE_NAMES.get(e, f"Ex {e}") for e in sorted(exercise_stats.keys())]
    session_counts = [exercise_stats[e]['sessions'] for e in sorted(exercise_stats.keys())]
    bars1 = ax1.bar(exercises, session_counts, color='skyblue', edgecolor='navy', linewidth=1.5)
    ax1.set_ylabel('Number of Sessions', fontsize=11, fontweight='bold')
    ax1.set_title('Sessions Collected per Exercise', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    # 2. Total samples per exercise (bar chart)
    ax2 = plt.subplot(2, 3, 2)
    sample_counts = [exercise_stats[e]['total_samples'] for e in sorted(exercise_stats.keys())]
    bars2 = ax2.bar(exercises, sample_counts, color='lightcoral', edgecolor='darkred', linewidth=1.5)
    ax2.set_ylabel('Total Samples', fontsize=11, fontweight='bold')
    ax2.set_title('Total Samples per Exercise', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    # 3. Total reps per exercise (bar chart)
    ax3 = plt.subplot(2, 3, 3)
    rep_counts = [exercise_stats[e]['total_reps'] for e in sorted(exercise_stats.keys())]
    bars3 = ax3.bar(exercises, rep_counts, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    ax3.set_ylabel('Total Reps', fontsize=11, fontweight='bold')
    ax3.set_title('Total Reps Recorded per Exercise', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    # 4. Sensor distribution heatmap
    ax4 = plt.subplot(2, 3, 4)
    sensor_matrix = []
    sensor_names = sorted(set(s for stats in exercise_stats.values() 
                             for s in stats['sensor_samples'].keys()))
    
    for exercise_type in sorted(exercise_stats.keys()):
        row = [exercise_stats[exercise_type]['sensor_samples'][sensor] for sensor in sensor_names]
        sensor_matrix.append(row)
    
    if sensor_matrix:
        im = ax4.imshow(sensor_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(sensor_names)))
        ax4.set_xticklabels(sensor_names, rotation=45, ha='right')
        ax4.set_yticks(range(len(exercises)))
        ax4.set_yticklabels(exercises)
        ax4.set_title('Sample Distribution: Exercise × Sensor', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(sensor_matrix)):
            for j in range(len(sensor_matrix[i])):
                text = ax4.text(j, i, f'{sensor_matrix[i][j]}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Sample Count')
    
    # 5. Windows per exercise
    ax5 = plt.subplot(2, 3, 5)
    window_counts = [exercise_stats[e]['total_windows'] for e in sorted(exercise_stats.keys())]
    bars5 = ax5.bar(exercises, window_counts, color='plum', edgecolor='purple', linewidth=1.5)
    ax5.set_ylabel('Total Windows', fontsize=11, fontweight='bold')
    ax5.set_title('Total Windows per Exercise', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    # 6. Overall summary statistics (text box)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    total_sessions = sum(s['sessions'] for s in exercise_stats.values())
    total_samples = sum(s['total_samples'] for s in exercise_stats.values())
    total_windows = sum(s['total_windows'] for s in exercise_stats.values())
    total_reps = sum(s['total_reps'] for s in exercise_stats.values())
    
    summary_text = f"""
    📊 OVERALL DATASET SUMMARY
    {'='*35}
    
    Total Sessions:        {total_sessions}
    Total Samples:         {total_samples:,}
    Total Windows:         {total_windows}
    Total Reps:            {total_reps}
    
    Exercises:             {len(exercise_stats)}
    Sensors Used:          {len(sensor_names)}
    
    Avg Samples/Session:   {total_samples/total_sessions if total_sessions > 0 else 0:.0f}
    Avg Windows/Session:   {total_windows/total_sessions if total_sessions > 0 else 0:.0f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('HAR Training Data - Distribution Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f"{output_dir}/data_distribution_summary.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: {filename}")
    plt.close()
    
    # Print summary to console
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total Sessions: {total_sessions}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Total Windows: {total_windows}")
    print(f"Total Reps: {total_reps}")
    print(f"Exercises Covered: {len(exercise_stats)}")
    
    for exercise_type in sorted(exercise_stats.keys()):
        exercise_name = EXERCISE_NAMES.get(exercise_type, f"Exercise {exercise_type}")
        stats = exercise_stats[exercise_type]
        print(f"\n{exercise_name}:")
        print(f"  Sessions: {stats['sessions']}")
        print(f"  Samples: {stats['total_samples']:,}")
        print(f"  Windows: {stats['total_windows']}")
        print(f"  Reps: {stats['total_reps']}")


def plot_quaternion_3d_trajectory(sessions, output_dir='visualizations'):
    """
    Create 3D trajectory plots showing the rotation represented by quaternions
    This helps visualize the movement patterns for each exercise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CREATING 3D QUATERNION TRAJECTORIES")
    print(f"{'='*60}")
    
    # Organize by exercise
    exercise_sessions = defaultdict(list)
    for session in sessions:
        exercise_type = session.get('session_metadata', {}).get('exercise_type', 0)
        exercise_sessions[exercise_type].append(session)
    
    for exercise_type, ex_sessions in exercise_sessions.items():
        exercise_name = EXERCISE_NAMES.get(exercise_type, f"Unknown Exercise {exercise_type}")
        print(f"\n📊 Creating 3D trajectory for {exercise_name}...")
        
        # Combine all sessions
        all_device_data = defaultdict(lambda: {'samples': [], 'node_name': '', 'node_id': -1})
        
        for session in ex_sessions:
            device_data = extract_quaternion_data(session)
            for device_addr, data in device_data.items():
                all_device_data[device_addr]['samples'].extend(data['samples'])
                all_device_data[device_addr]['node_name'] = data['node_name']
                all_device_data[device_addr]['node_id'] = data['node_id']
        
        if not all_device_data:
            continue
        
        # Create 3D plot
        num_devices = len(all_device_data)
        fig = plt.figure(figsize=(18, 5*((num_devices+1)//2)))
        fig.suptitle(f'{exercise_name} - 3D Quaternion Trajectories', 
                    fontsize=16, fontweight='bold')
        
        for idx, (device_addr, data) in enumerate(sorted(all_device_data.items(), 
                                                          key=lambda x: x[1]['node_id'])):
            samples = data['samples']
            node_name = data['node_name']
            
            # Sort by timestamp
            samples.sort(key=lambda x: x['timestamp'])
            
            # Extract quaternion components
            qx = np.array([s['qx'] for s in samples])
            qy = np.array([s['qy'] for s in samples])
            qz = np.array([s['qz'] for s in samples])
            
            # Create 3D subplot
            ax = fig.add_subplot(((num_devices+1)//2), 2, idx+1, projection='3d')
            
            # Plot 3D trajectory using time as color gradient
            scatter = ax.scatter(qx, qy, qz, c=range(len(qx)), 
                               cmap='viridis', s=1, alpha=0.6)
            
            ax.set_xlabel('qx', fontsize=10, fontweight='bold')
            ax.set_ylabel('qy', fontsize=10, fontweight='bold')
            ax.set_zlabel('qz', fontsize=10, fontweight='bold')
            ax.set_title(f'{node_name} ({len(samples)} samples)', 
                        fontsize=11, fontweight='bold')
            
            # Add colorbar for time
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Time Progress', fontsize=9)
            
            print(f"  ✓ {node_name} trajectory plotted")
        
        plt.tight_layout()
        filename = f"{output_dir}/{exercise_name.replace(' ', '_').lower()}_3d_trajectory.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
        plt.close()


def plot_quaternion_variance_analysis(sessions, output_dir='visualizations'):
    """
    Analyze and visualize the variance in quaternion data
    High variance = more dynamic movement
    Low variance = static/stable position
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ANALYZING QUATERNION VARIANCE")
    print(f"{'='*60}")
    
    exercise_variance = defaultdict(lambda: defaultdict(list))
    
    for session in sessions:
        exercise_type = session.get('session_metadata', {}).get('exercise_type', 0)
        device_data = extract_quaternion_data(session)
        
        for device_addr, data in device_data.items():
            node_name = data['node_name']
            samples = data['samples']
            
            if len(samples) < 2:
                continue
            
            # Calculate variance for each quaternion component
            qw = np.array([s['qw'] for s in samples])
            qx = np.array([s['qx'] for s in samples])
            qy = np.array([s['qy'] for s in samples])
            qz = np.array([s['qz'] for s in samples])
            
            exercise_variance[exercise_type][node_name].append({
                'qw_var': np.var(qw),
                'qx_var': np.var(qx),
                'qy_var': np.var(qy),
                'qz_var': np.var(qz),
                'total_var': np.var(qw) + np.var(qx) + np.var(qy) + np.var(qz)
            })
    
    # Create variance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quaternion Variance Analysis - Movement Dynamics', 
                fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    all_exercises = sorted(exercise_variance.keys())
    all_sensors = sorted(set(sensor for ex in exercise_variance.values() for sensor in ex.keys()))
    
    # 1. Total variance heatmap
    ax1 = axes[0, 0]
    variance_matrix = []
    for exercise_type in all_exercises:
        row = []
        for sensor in all_sensors:
            variances = exercise_variance[exercise_type][sensor]
            avg_total_var = np.mean([v['total_var'] for v in variances]) if variances else 0
            row.append(avg_total_var)
        variance_matrix.append(row)
    
    im1 = ax1.imshow(variance_matrix, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(all_sensors)))
    ax1.set_xticklabels(all_sensors, rotation=45, ha='right')
    ax1.set_yticks(range(len(all_exercises)))
    ax1.set_yticklabels([EXERCISE_NAMES.get(e, f"Ex {e}") for e in all_exercises])
    ax1.set_title('Total Quaternion Variance (Higher = More Dynamic)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Total Variance')
    
    # Add value annotations
    for i in range(len(variance_matrix)):
        for j in range(len(variance_matrix[i])):
            text = ax1.text(j, i, f'{variance_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 2. Variance by component (stacked bar)
    ax2 = axes[0, 1]
    width = 0.15
    x = np.arange(len(all_exercises))
    
    for idx, sensor in enumerate(all_sensors):
        qw_vars = [np.mean([v['qw_var'] for v in exercise_variance[ex][sensor]]) 
                  if exercise_variance[ex][sensor] else 0 for ex in all_exercises]
        ax2.bar(x + idx*width, qw_vars, width, label=sensor)
    
    ax2.set_xlabel('Exercise', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average qw Variance', fontsize=11, fontweight='bold')
    ax2.set_title('qw Variance by Sensor', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * (len(all_sensors)-1) / 2)
    ax2.set_xticklabels([EXERCISE_NAMES.get(e, f"Ex {e}") for e in all_exercises], rotation=15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Box plot of total variance distribution
    ax3 = axes[1, 0]
    box_data = []
    box_labels = []
    for exercise_type in all_exercises:
        for sensor in all_sensors:
            variances = exercise_variance[exercise_type][sensor]
            if variances:
                total_vars = [v['total_var'] for v in variances]
                box_data.append(total_vars)
                box_labels.append(f"{EXERCISE_NAMES.get(exercise_type, f'Ex{exercise_type}')}\n{sensor}")
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Total Variance', fontsize=11, fontweight='bold')
    ax3.set_title('Variance Distribution Across All Sessions', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=90, labelsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Sensor comparison across exercises
    ax4 = axes[1, 1]
    for sensor in all_sensors:
        sensor_total_vars = []
        for exercise_type in all_exercises:
            variances = exercise_variance[exercise_type][sensor]
            avg_var = np.mean([v['total_var'] for v in variances]) if variances else 0
            sensor_total_vars.append(avg_var)
        ax4.plot([EXERCISE_NAMES.get(e, f"Ex {e}") for e in all_exercises], 
                sensor_total_vars, marker='o', label=sensor, linewidth=2)
    
    ax4.set_xlabel('Exercise', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Total Variance', fontsize=11, fontweight='bold')
    ax4.set_title('Movement Dynamics Across Exercises', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    filename = f"{output_dir}/quaternion_variance_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: {filename}")
    plt.close()


def plot_temporal_alignment_check(sessions, output_dir='visualizations'):
    """
    Visualize temporal alignment between sensors
    Shows if all sensors are properly synchronized
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CHECKING TEMPORAL ALIGNMENT")
    print(f"{'='*60}")
    
    # Pick a few sessions to visualize
    sample_sessions = sessions[:min(3, len(sessions))]
    
    for session_idx, session in enumerate(sample_sessions):
        metadata = session.get('session_metadata', {})
        exercise_type = metadata.get('exercise_type', 0)
        exercise_name = EXERCISE_NAMES.get(exercise_type, f"Exercise {exercise_type}")
        
        device_data = extract_quaternion_data(session)
        
        if len(device_data) < 2:
            continue
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (device_addr, data) in enumerate(sorted(device_data.items(), 
                                                          key=lambda x: x[1]['node_id'])):
            samples = data['samples']
            node_name = data['node_name']
            
            # Sort by timestamp
            samples.sort(key=lambda x: x['timestamp'])
            
            timestamps = np.array([s['timestamp'] for s in samples])
            # Normalize to start at 0
            if len(timestamps) > 0:
                timestamps = (timestamps - timestamps[0]) / 1000.0  # seconds
            
            # Use qw magnitude as signal indicator
            qw = np.array([s['qw'] for s in samples])
            
            # Offset each sensor vertically for clarity
            offset = idx * 2.5
            ax.plot(timestamps, qw + offset, label=node_name, 
                   color=colors_list[idx % len(colors_list)], linewidth=1.0, alpha=0.8)
            ax.axhline(y=offset, color=colors_list[idx % len(colors_list)], 
                      linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('qw + Offset (per sensor)', fontsize=12, fontweight='bold')
        ax.set_title(f'Temporal Alignment Check - {exercise_name} (Session {session_idx+1})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{output_dir}/temporal_alignment_session_{session_idx+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
        plt.close()


def generate_ml_readiness_report(sessions, output_dir='visualizations'):
    """
    Generate a report assessing the data's readiness for ML training
    """
    print(f"\n{'='*60}")
    print(f"ML READINESS ASSESSMENT")
    print(f"{'='*60}")
    
    report = []
    report.append("="*70)
    report.append("HAR TRAINING DATA - ML READINESS REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall statistics
    total_sessions = len(sessions)
    total_samples = sum(s.get('session_metadata', {}).get('total_samples', 0) for s in sessions)
    
    exercise_counts = defaultdict(int)
    for session in sessions:
        exercise_type = session.get('session_metadata', {}).get('exercise_type', 0)
        exercise_counts[exercise_type] += 1
    
    report.append(f"1. DATASET SIZE")
    report.append(f"   - Total Sessions: {total_sessions}")
    report.append(f"   - Total Samples: {total_samples:,}")
    report.append(f"   - Average Samples per Session: {total_samples/total_sessions if total_sessions > 0 else 0:.0f}")
    report.append("")
    
    # Class balance
    report.append(f"2. CLASS BALANCE")
    for exercise_type, count in sorted(exercise_counts.items()):
        exercise_name = EXERCISE_NAMES.get(exercise_type, f"Exercise {exercise_type}")
        percentage = (count / total_sessions * 100) if total_sessions > 0 else 0
        report.append(f"   - {exercise_name}: {count} sessions ({percentage:.1f}%)")
    
    # Check for class imbalance
    max_count = max(exercise_counts.values()) if exercise_counts else 0
    min_count = min(exercise_counts.values()) if exercise_counts else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    report.append("")
    if imbalance_ratio > 2:
        report.append(f"   ⚠ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
        report.append(f"   Consider collecting more data for underrepresented classes.")
    else:
        report.append(f"   ✓ Classes are reasonably balanced (ratio: {imbalance_ratio:.2f}:1)")
    report.append("")
    
    # Sensor coverage
    report.append(f"3. SENSOR COVERAGE")
    all_sensors = set()
    for session in sessions:
        device_data = extract_quaternion_data(session)
        for data in device_data.values():
            all_sensors.add(data['node_name'])
    
    report.append(f"   - Unique Sensors Used: {len(all_sensors)}")
    report.append(f"   - Sensors: {', '.join(sorted(all_sensors))}")
    report.append("")
    
    # Data quality checks
    report.append(f"4. DATA QUALITY INDICATORS")
    
    # Check for sessions with sufficient samples
    sufficient_samples = sum(1 for s in sessions 
                            if s.get('session_metadata', {}).get('total_samples', 0) >= 100)
    report.append(f"   - Sessions with ≥100 samples: {sufficient_samples}/{total_sessions}")
    
    # Check for multi-device sessions
    multi_device = sum(1 for s in sessions 
                      if len(s.get('device_windows', {})) >= 2)
    report.append(f"   - Multi-device sessions: {multi_device}/{total_sessions}")
    report.append("")
    
    # Recommendations
    report.append(f"5. RECOMMENDATIONS FOR ML TRAINING")
    recommendations = []
    
    if total_sessions < 20:
        recommendations.append("   ⚠ Collect more sessions (target: ≥20 per exercise)")
    else:
        recommendations.append("   ✓ Sufficient number of sessions collected")
    
    if imbalance_ratio > 2:
        recommendations.append("   ⚠ Balance your dataset across all exercise types")
    else:
        recommendations.append("   ✓ Dataset is well-balanced")
    
    if len(all_sensors) < 3:
        recommendations.append("   ⚠ Consider using more sensors for better accuracy")
    else:
        recommendations.append("   ✓ Good sensor diversity")
    
    if total_samples < 1000:
        recommendations.append("   ⚠ Collect more samples (target: ≥1000 total)")
    else:
        recommendations.append("   ✓ Sufficient sample count for initial training")
    
    report.extend(recommendations)
    report.append("")
    
    report.append("="*70)
    
    # Print to console
    for line in report:
        print(line)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    report_file = f"{output_dir}/ml_readiness_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n💾 Report saved to: {report_file}")


def main():
    """Main execution function"""
    print("="*70)
    print("HAR SYSTEM - MONGODB DATA VISUALIZATION")
    print("="*70)
    print()
    
    # Connect to MongoDB
    client, collection = connect_to_mongodb()
    
    # Fetch all sessions
    sessions = fetch_all_sessions(collection)
    
    if not sessions:
        print("No sessions found in database. Exiting.")
        client.close()
        return
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Visualizations will be saved to: {output_dir}/")
    
    # Generate all visualizations
    try:
        # 1. Raw quaternion plots per exercise
        plot_raw_quaternions_per_exercise(sessions, output_dir)
        
        # 2. Data distribution summary
        plot_data_distribution_summary(sessions, output_dir)
        
        # 3. 3D trajectories
        plot_quaternion_3d_trajectory(sessions, output_dir)
        
        # 4. Variance analysis
        plot_quaternion_variance_analysis(sessions, output_dir)
        
        # 5. Temporal alignment check
        plot_temporal_alignment_check(sessions, output_dir)
        
        # 6. ML readiness report
        generate_ml_readiness_report(sessions, output_dir)
        
        print(f"\n{'='*70}")
        print(f"✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nCheck the '{output_dir}' folder for all generated plots and reports.")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        print("\n✓ MongoDB connection closed")


if __name__ == "__main__":
    main()
