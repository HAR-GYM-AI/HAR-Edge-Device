#!/usr/bin/env python3
"""
Random Forest Classifier with Proper Session-Based Splitting
Fixes data leakage by splitting at the SESSION level, not window level
"""

import os
import json
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

def load_json_sessions(data_dir='collected_data', exercise_types=['BICEP_CURL', 'BENCH_PRESS']):
    """Load sessions from JSON files"""
    print(f"Loading sessions from {data_dir}/")
    
    sessions = []
    
    for exercise in exercise_types:
        pattern = f"{data_dir}/session_*_{exercise}_*.json"
        files = glob.glob(pattern)
        print(f"  Found {len(files)} {exercise} session files")
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    session = json.load(f)
                    # Store filepath as session identifier
                    session['_filepath'] = os.path.basename(filepath)
                    sessions.append(session)
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    
    print(f"Total sessions loaded: {len(sessions)}")
    return sessions

def extract_features_by_session(sessions):
    """
    Extract features maintaining session grouping.
    Returns X, y, and groups for proper cross-validation.
    """
    print("\nExtracting features with session grouping...")
    
    X = []  # Features
    y = []  # Labels
    groups = []  # Session identifiers
    session_info = []  # For tracking
    
    for session_idx, session in enumerate(sessions):
        exercise_type = session['session_metadata']['exercise_type']
        session_id = session['_filepath']
        
        # Try sorted_windows first, fallback to device_windows
        sorted_windows = session.get('sorted_windows', [])
        
        if not sorted_windows:
            device_windows = session.get('device_windows', {})
            for device_addr, windows in device_windows.items():
                sorted_windows.extend(windows)
        
        # Filter for short windows only
        short_windows = [w for w in sorted_windows if w.get('window_type') in [0, 'short']]
        
        for window in short_windows:
            features = window.get('features', {})
            if not features:
                features = window
            
            # Extract feature vector (11 features - excluding node_id to avoid leakage)
            feature_vector = [
                features.get('qw_mean', 0),
                features.get('qx_mean', 0),
                features.get('qy_mean', 0),
                features.get('qz_mean', 0),
                features.get('qw_std', 0),
                features.get('qx_std', 0),
                features.get('qy_std', 0),
                features.get('qz_std', 0),
                features.get('sma', 0),
                features.get('dominant_freq', 0),
                features.get('total_samples', 0),
            ]
            
            X.append(feature_vector)
            y.append(exercise_type)
            groups.append(session_idx)  # Group by session
            session_info.append(session_id)
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"\nTotal windows: {len(X)}")
    print(f"Total sessions: {len(np.unique(groups))}")
    print(f"Feature shape: {X.shape}")
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} windows")
    
    return X, y, groups, session_info

def session_based_split(X, y, groups, test_size=0.2):
    """
    Split data by sessions, not by windows.
    Ensures no session appears in both train and test.
    """
    print(f"\nPerforming session-based split (test_size={test_size})...")
    
    # Get unique sessions
    unique_sessions = np.unique(groups)
    n_sessions = len(unique_sessions)
    n_test_sessions = max(1, int(n_sessions * test_size))
    
    # Shuffle and split sessions
    np.random.seed(42)
    shuffled_sessions = np.random.permutation(unique_sessions)
    test_sessions = set(shuffled_sessions[:n_test_sessions])
    train_sessions = set(shuffled_sessions[n_test_sessions:])
    
    # Create train/test masks
    train_mask = np.array([g in train_sessions for g in groups])
    test_mask = np.array([g in test_sessions for g in groups])
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training sessions: {len(train_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")
    print(f"Training windows: {len(X_train)}")
    print(f"Test windows: {len(X_test)}")
    
    # Show train/test distribution
    print(f"\nTraining set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} windows")
    
    print(f"\nTest set distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} windows")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier"""
    print("\nTraining Random Forest...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    print("Training complete!")
    
    return clf

def evaluate_classifier(clf, X_test, y_test):
    """Evaluate the classifier"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS (NO DATA LEAKAGE)")
    print("="*70)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = [
        'qw_mean', 'qx_mean', 'qy_mean', 'qz_mean',
        'qw_std', 'qx_std', 'qy_std', 'qz_std',
        'sma', 'dominant_freq', 'total_samples'
    ]
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(feature_names)):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:20s} : {importances[idx]:.4f}")
    
    print("="*70)
    
    return accuracy, y_pred

def leave_one_session_out_cv(X, y, groups):
    """Perform Leave-One-Session-Out Cross-Validation"""
    print("\n" + "="*70)
    print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
    print("="*70)
    print("This tests generalization to completely unseen sessions...")
    
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    
    print(f"\nPerforming {n_splits}-fold cross-validation (one fold per session)...")
    
    accuracies = []
    fold = 1
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and evaluate
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        # Print progress every 5 folds
        if fold % 5 == 0:
            print(f"  Completed {fold}/{n_splits} folds...")
        fold += 1
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(accuracies):.2%} (+/- {np.std(accuracies):.2%})")
    print(f"  Min Accuracy:  {np.min(accuracies):.2%}")
    print(f"  Max Accuracy:  {np.max(accuracies):.2%}")
    print("="*70)
    
    return accuracies

def main():
    """Main function"""
    print("="*70)
    print("RANDOM FOREST CLASSIFIER (FIXED - NO DATA LEAKAGE)")
    print("HAR System - BICEP_CURL vs BENCH_PRESS")
    print("="*70)
    
    try:
        # Load sessions
        sessions = load_json_sessions('collected_data', ['BICEP_CURL', 'BENCH_PRESS'])
        
        if len(sessions) == 0:
            print("\nNo data found! Please collect some training data first.")
            return
        
        # Extract features with session grouping
        X, y, groups, session_info = extract_features_by_session(sessions)
        
        if len(X) == 0:
            print("\nNo windows found in sessions!")
            return
        
        if len(X) < 10:
            print(f"\nWarning: Only {len(X)} windows found. Need more data.")
            return
        
        # Method 1: Session-based train/test split
        X_train, X_test, y_train, y_test = session_based_split(X, y, groups, test_size=0.2)
        
        # Train and evaluate
        clf = train_random_forest(X_train, y_train)
        accuracy, predictions = evaluate_classifier(clf, X_test, y_test)
        
        # Method 2: Leave-One-Session-Out CV (more rigorous)
        if len(np.unique(groups)) >= 5:  # Only if we have enough sessions
            cv_accuracies = leave_one_session_out_cv(X, y, groups)
        
        # Final assessment
        print("\n" + "="*70)
        print("FINAL ASSESSMENT")
        print("="*70)
        print(f"Session-based test accuracy: {accuracy:.2%}")
        
        if accuracy > 0.85:
            print("EXCELLENT - System generalizes well to new sessions")
        elif accuracy > 0.70:
            print("GOOD - Reasonable generalization, consider more data")
        elif accuracy > 0.60:
            print("MODERATE - May need more training data or better features")
        else:
            print("LOW - Significant room for improvement")
        
        print("\nKey improvements made:")
        print("  Session-based splitting (no session in both train & test)")
        print("  Removed node_id feature (potential leakage source)")
        print("  Proper evaluation methodology")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
