import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_flatten_data(file_path):
    """
    Loads nested HAR data from a JSON array file, flattens it into a DataFrame,
    and includes the session ID for each sample.

    Args:
        file_path (str): The path to the JSON data file.

    Returns:
        pd.DataFrame: A DataFrame containing the flattened sensor data,
                      or an empty DataFrame if loading fails.
    """
    print(f"--- Loading and Processing {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            all_sessions = json.load(f)
        print(f"Successfully loaded {len(all_sessions)} session(s).")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading or parsing JSON file: {e}")
        return pd.DataFrame()

    flattened_data = []
    for session in all_sessions:
        session_id = session.get('_id', 'UNKNOWN_ID')
        metadata = session.get('session_metadata', {})
        activity = metadata.get('exercise_type', 'UNKNOWN')
        devices_metadata = metadata.get('devices', {})

        device_windows = session.get('device_windows', {})
        for device_mac, windows in device_windows.items():
            # Use the node_name (e.g., 'CHEST', 'THIGH') for clarity
            device_name = devices_metadata.get(device_mac, {}).get('node_name', device_mac)

            for window in windows:
                for sample in window.get('samples', []):
                    flattened_data.append({
                        'session_id': session_id,
                        'activity': activity,
                        'device': device_name,
                        'qx': sample.get('qx'),
                        'qy': sample.get('qy'),
                        'qz': sample.get('qz')
                    })

    if not flattened_data:
        print("Could not find any samples in the provided JSON file.")
        return pd.DataFrame()

    df = pd.DataFrame(flattened_data)
    # Ensure quaternion data is numeric, coercing errors to NaN
    for col in ['qx', 'qy', 'qz']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['qx', 'qy', 'qz'], inplace=True)
    
    print("--- Data Loading Complete ---")
    print(f"Found {df['session_id'].nunique()} unique sessions.")
    print(f"Found {df['activity'].nunique()} unique activities: {df['activity'].unique()}")
    print(f"Found {df['device'].nunique()} unique devices: {df['device'].unique()}")
    
    return df

def plot_session_time_series(df, session_id):
    """
    Plots the quaternion time-series data for each sensor in a single, specified session.

    Args:
        df (pd.DataFrame): The main DataFrame with all sensor data.
        session_id (str): The ID of the session to plot.
    """
    session_df = df[df['session_id'] == session_id].copy()

    if session_df.empty:
        print(f"\nError: No data found for session_id '{session_id}'. Please choose a valid ID.")
        return

    activity = session_df['activity'].iloc[0]
    devices = sorted(session_df['device'].unique())
    num_devices = len(devices)

    fig, axes = plt.subplots(num_devices, 1, figsize=(18, 5 * num_devices), sharex=True)
    fig.suptitle(f'Time-Series Data for Session: {session_id}\nActivity: {activity}', fontsize=16, y=1.02)

    if num_devices == 1:
        axes = [axes] # Make it iterable

    for i, device in enumerate(devices):
        ax = axes[i]
        device_df = session_df[session_df['device'] == device].reset_index(drop=True)
        
        ax.plot(device_df.index, device_df['qx'], label='QX', alpha=0.9)
        ax.plot(device_df.index, device_df['qy'], label='QY', alpha=0.9)
        ax.plot(device_df.index, device_df['qz'], label='QZ', alpha=0.9)
        
        ax.set_title(f'Sensor: {device}', loc='left', fontsize=12)
        ax.set_ylabel('Quaternion Value')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Time Step (Sample Index)')
    plt.tight_layout()
    
    # Save the figure
    filename = f"session_{session_id}_time_series.png"
    plt.savefig(filename)
    print(f"\nSUCCESS: Time-series plot saved as '{filename}'")
    plt.show()

def plot_quaternion_distributions(df, activity_name):
    """
    Creates box plots to show the statistical distribution of quaternion values
    for each sensor during a specific activity.

    Args:
        df (pd.DataFrame): The main DataFrame.
        activity_name (str): The activity to analyze.
    """
    activity_df = df[df['activity'] == activity_name].copy()

    if activity_df.empty:
        print(f"\nWarning: No data found for activity '{activity_name}'. Skipping distribution plot.")
        return

    # Melt the DataFrame to have a long format suitable for seaborn's boxplot
    df_melted = activity_df.melt(
        id_vars=['device'],
        value_vars=['qx', 'qy', 'qz'],
        var_name='Quaternion Axis',
        value_name='Value'
    )

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_melted, x='device', y='Value', hue='Quaternion Axis', palette="Set2")
    
    plt.title(f'Quaternion Value Distribution for Activity: {activity_name}', fontsize=16)
    plt.xlabel('Sensor Location', fontsize=12)
    plt.ylabel('Quaternion Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Quaternion Axis')
    
    # Save the figure
    filename = f"activity_{activity_name}_distributions.png"
    plt.savefig(filename)
    print(f"SUCCESS: Distribution plot saved as '{filename}'")
    plt.show()

def plot_movement_variance(df):
    """
    Calculates and plots the standard deviation of quaternion values as a proxy for
    movement amount, grouped by activity and sensor.

    Args:
        df (pd.DataFrame): The main DataFrame.
    """
    if df.empty:
        print("\nWarning: DataFrame is empty. Skipping movement variance plot.")
        return
        
    # Calculate the standard deviation for each axis, then average them for a general 'movement' score
    variance_df = df.groupby(['activity', 'device'])[['qx', 'qy', 'qz']].std().mean(axis=1).reset_index(name='movement_variance')
    
    plt.figure(figsize=(16, 9))
    sns.barplot(
        data=variance_df,
        x='activity',
        y='movement_variance',
        hue='device',
        palette='viridis'
    )
    
    plt.title('Comparison of Movement Variance by Sensor Across Activities', fontsize=16)
    plt.xlabel('Activity Type', fontsize=12)
    plt.ylabel('Average Standard Deviation (Movement Variance)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Sensor Location')
    plt.tight_layout()
    
    # Save the figure
    filename = "movement_variance_comparison.png"
    plt.savefig(filename)
    print(f"SUCCESS: Movement variance plot saved as '{filename}'")
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    json_file = 'mongodb_export.json'
    
    # 1. Load and process the data from the JSON file
    main_df = load_and_flatten_data(json_file)

    if not main_df.empty:
        # --- PLOT 1: MOVEMENT VARIANCE ACROSS ALL ACTIVITIES ---
        print("\n--- Generating Plot 1: Overall Movement Variance ---")
        print("This plot shows which sensors are most active during each exercise.")
        plot_movement_variance(main_df)

        # --- PLOT 2: STATISTICAL DISTRIBUTION FOR A SAMPLE ACTIVITY ---
        print("\n--- Generating Plot 2: Statistical Distribution (Box Plot) ---")
        # We'll generate this for the first activity found in the data as an example
        sample_activity = main_df['activity'].unique()[0]
        print(f"This plot shows the range of motion for each sensor during '{sample_activity}'.")
        plot_quaternion_distributions(main_df, sample_activity)

        # --- PLOT 3: DETAILED TIME-SERIES FOR A SPECIFIC SESSION ---
        print("\n--- Generating Plot 3: Detailed Session Time-Series ---")
        # Let the user choose which session to plot
        all_session_ids = main_df['session_id'].unique()
        print("Available Session IDs:")
        for sid in all_session_ids:
            print(f"- {sid}")
        
        # Simple input prompt to choose a session
        chosen_session = input("\nPlease enter one of the session IDs from the list above to plot its details: ").strip()
        
        # Generate the plot for the chosen session
        plot_session_time_series(main_df, chosen_session)