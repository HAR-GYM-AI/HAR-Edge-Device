import json
from collections import defaultdict
from itertools import groupby
import ijson # <-- Added ijson for streaming

def get_session_start_time(session):
    """
    Finds the earliest 'absolute_timestamp_ms' in a session to use as its start time.
    Returns infinity if no timestamps are found, ensuring it sorts last.
    """
    min_timestamp = float('inf')
    
    if 'device_windows' not in session or not isinstance(session.get('device_windows'), dict):
        return min_timestamp

    for device, windows in session.get('device_windows', {}).items():
        for window in windows:
            # The 'fix_timing.py' script creates a single unified window.
            # We can safely assume the samples are sorted and take the first one.
            if 'samples' in window and window['samples']:
                first_sample_ts = window['samples'][0].get('absolute_timestamp_ms')
                if first_sample_ts is not None:
                    min_timestamp = min(min_timestamp, float(first_sample_ts))
                    
    return min_timestamp

def analyze_participant_sequences(file_path):
    """
    Loads session data using a streaming parser, groups it by participant, 
    and checks for non-contiguous exercise blocks.
    """
    print("="*70)
    print("STARTING EXERCISE LABELING PROTOCOL VALIDATION")
    print(f"Loading data from: {file_path}")
    print("="*70)

    sessions_with_metadata = []
    sessions_loaded_count = 0

    try:
        # --- MODIFIED PART: Use ijson for memory-efficient streaming ---
        print("Streaming JSON data to avoid memory errors...")
        with open(file_path, 'r') as f:
            # ijson.items(f, 'item') reads the top-level array item by item
            sessions_generator = ijson.items(f, 'item')
            for session in sessions_generator:
                sessions_loaded_count += 1
                metadata = session.get('session_metadata', {})
                participant_id = metadata.get('participant_id')
                exercise_type = metadata.get('exercise_type')
                session_id = session.get('_id')

                if not all([participant_id, exercise_type, session_id]):
                    print(f"  - Warning: Skipping a session due to missing metadata (ID: {session_id})")
                    continue

                start_time = get_session_start_time(session)
                if start_time == float('inf'):
                    print(f"  - Warning: Skipping session {session_id} because no sample timestamps were found.")
                    continue
                    
                sessions_with_metadata.append({
                    'participant_id': participant_id,
                    'exercise_type': exercise_type,
                    'start_time': start_time,
                    'session_id': session_id
                })
        print(f"Successfully processed {sessions_loaded_count} total sessions from the file.")
        # --- END OF MODIFIED PART ---

    except (ijson.JSONError, FileNotFoundError) as e:
        print(f"FATAL ERROR: Could not load or parse the JSON file: {e}")
        return

    # --- Step 2: Group sessions by participant ---
    participant_sessions = defaultdict(list)
    for session_meta in sessions_with_metadata:
        participant_sessions[session_meta['participant_id']].append(session_meta)

    print(f"\nFound data for {len(participant_sessions)} participants: {sorted(participant_sessions.keys())}")
    print("-" * 70)

    # --- Step 3: Analyze each participant's exercise sequence ---
    total_issues_found = 0
    for participant_id, sessions in sorted(participant_sessions.items()):
        
        sessions.sort(key=lambda x: x['start_time'])
        exercise_sequence = [s['exercise_type'] for s in sessions]
        grouped_sequence = [key for key, group in groupby(exercise_sequence)]
        
        seen_blocks = set()
        duplicate_blocks = set()
        for exercise_block in grouped_sequence:
            if exercise_block in seen_blocks:
                duplicate_blocks.add(exercise_block)
            seen_blocks.add(exercise_block)
        
        print(f"\nAnalyzing Participant ID: {participant_id}")
        print(f"  - Chronological Exercise Sequence: {' -> '.join(exercise_sequence)}")

        if not duplicate_blocks:
            print(f"PROTOCOL OK: All exercises for this participant are in contiguous blocks.")
        else:
            total_issues_found += 1
            print(f"PROTOCOL VIOLATION FOUND!")
            print(f"The following exercises appear in non-contiguous blocks: {', '.join(sorted(list(duplicate_blocks)))}")
            print(f"This suggests a potential labeling error.")
            print("Session IDs in order:")
            for s in sessions:
                 print(f"        - {s['session_id']} ({s['exercise_type']})")

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    if total_issues_found == 0:
        print("Success! No protocol violations were found across any participants.")
    else:
        print(f"Found protocol violations for {total_issues_found} participant(s).")
        print("Review the participants marked with 'PROTOCOL VIOLATION' above.")

# --- Main execution block ---
if __name__ == "__main__":
    json_file = 'mongodb_export_synced.json'
    
    # Before running, ensure ijson is installed:
    # pip install ijson
    
    analyze_participant_sequences(json_file)