#!/usr/bin/env python3
"""
HAR Data Post-Processing Script
Enhances existing MongoDB data with advanced temporal and quality features

This script adds:
1. Quaternion normalization and outlier detection
2. Temporal features (velocity, acceleration, zero crossing rates)
3. Enhanced frequency domain features
4. Data quality validation
"""

import os
import sys
import math
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'har_system')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'training_data')

@dataclass
class QuaternionSample:
    """Individual quaternion sample"""
    absolute_timestamp_ms: int
    relative_timestamp_ms: int
    qw: float
    qx: float
    qy: float
    qz: float

def connect_to_mongodb():
    """Connect to MongoDB and return collection"""
    try:
        logger.info(f"Connecting to MongoDB at {MONGODB_URI}...")
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]
        logger.info(f"✓ Connected to {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        return client, collection
    except Exception as e:
        logger.error(f"✗ Failed to connect to MongoDB: {e}")
        sys.exit(1)

def normalize_quaternions(samples: List[QuaternionSample]) -> List[QuaternionSample]:
    """Ensure quaternions are properly normalized"""
    normalized = []
    for sample in samples:
        # Calculate magnitude
        norm = math.sqrt(sample.qw**2 + sample.qx**2 + sample.qy**2 + sample.qz**2)
        if norm > 0:
            normalized.append(QuaternionSample(
                absolute_timestamp_ms=sample.absolute_timestamp_ms,
                relative_timestamp_ms=sample.relative_timestamp_ms,
                qw=sample.qw/norm,
                qx=sample.qx/norm,
                qy=sample.qy/norm,
                qz=sample.qz/norm
            ))
        else:
            # Skip invalid quaternions
            continue
    return normalized

def detect_quaternion_outliers(samples: List[QuaternionSample], threshold: float = 3.0) -> List[QuaternionSample]:
    """Remove samples where quaternion magnitude deviates significantly"""
    if len(samples) < 2:
        return samples

    magnitudes = [math.sqrt(s.qw**2 + s.qx**2 + s.qy**2 + s.qz**2) for s in samples]
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)

    valid_samples = []
    for sample, mag in zip(samples, magnitudes):
        if abs(mag - mean_mag) <= threshold * std_mag:
            valid_samples.append(sample)

    logger.info(f"Outlier detection: {len(samples)} -> {len(valid_samples)} samples")
    return valid_samples

def extract_temporal_features(samples: List[QuaternionSample], sampling_rate: int = 100) -> Dict[str, float]:
    """Extract time-domain features from quaternion sequences"""
    if len(samples) < 3:
        return {}

    qw_vals = np.array([s.qw for s in samples])
    qx_vals = np.array([s.qx for s in samples])
    qy_vals = np.array([s.qy for s in samples])
    qz_vals = np.array([s.qz for s in samples])

    features = {}

    # Velocity (first derivative) - multiply by sampling rate for proper units
    qw_vel = np.gradient(qw_vals) * sampling_rate
    qx_vel = np.gradient(qx_vals) * sampling_rate
    qy_vel = np.gradient(qy_vals) * sampling_rate
    qz_vel = np.gradient(qz_vals) * sampling_rate

    # Acceleration (second derivative)
    qw_acc = np.gradient(qw_vel) * sampling_rate
    qx_acc = np.gradient(qx_vel) * sampling_rate
    qy_acc = np.gradient(qy_vel) * sampling_rate
    qz_acc = np.gradient(qz_vel) * sampling_rate

    # Statistical features on derivatives
    features.update({
        'qw_vel_mean': float(np.mean(qw_vel)),
        'qw_vel_std': float(np.std(qw_vel)),
        'qw_vel_max': float(np.max(np.abs(qw_vel))),
        'qw_acc_mean': float(np.mean(qw_acc)),
        'qw_acc_std': float(np.std(qw_acc)),
        'qw_acc_max': float(np.max(np.abs(qw_acc))),

        'qx_vel_mean': float(np.mean(qx_vel)),
        'qx_vel_std': float(np.std(qx_vel)),
        'qx_vel_max': float(np.max(np.abs(qx_vel))),
        'qx_acc_mean': float(np.mean(qx_acc)),
        'qx_acc_std': float(np.std(qx_acc)),
        'qx_acc_max': float(np.max(np.abs(qx_acc))),

        'qy_vel_mean': float(np.mean(qy_vel)),
        'qy_vel_std': float(np.std(qy_vel)),
        'qy_vel_max': float(np.max(np.abs(qy_vel))),
        'qy_acc_mean': float(np.mean(qy_acc)),
        'qy_acc_std': float(np.std(qy_acc)),
        'qy_acc_max': float(np.max(np.abs(qy_acc))),

        'qz_vel_mean': float(np.mean(qz_vel)),
        'qz_vel_std': float(np.std(qz_vel)),
        'qz_vel_max': float(np.max(np.abs(qz_vel))),
        'qz_acc_mean': float(np.mean(qz_acc)),
        'qz_acc_std': float(np.std(qz_acc)),
        'qz_acc_max': float(np.max(np.abs(qz_acc))),
    })

    # Zero crossing rates
    features.update({
        'qw_zcr': float(np.sum(np.diff(np.sign(qw_vals - np.mean(qw_vals)))) / (2 * len(qw_vals))),
        'qx_zcr': float(np.sum(np.diff(np.sign(qx_vals - np.mean(qx_vals)))) / (2 * len(qx_vals))),
        'qy_zcr': float(np.sum(np.diff(np.sign(qy_vals - np.mean(qy_vals)))) / (2 * len(qy_vals))),
        'qz_zcr': float(np.sum(np.diff(np.sign(qz_vals - np.mean(qz_vals)))) / (2 * len(qz_vals))),
    })

    # Movement intensity metrics
    qw_energy = float(np.sum(qw_vals**2))
    qx_energy = float(np.sum(qx_vals**2))
    qy_energy = float(np.sum(qy_vals**2))
    qz_energy = float(np.sum(qz_vals**2))

    features.update({
        'qw_energy': qw_energy,
        'qx_energy': qx_energy,
        'qy_energy': qy_energy,
        'qz_energy': qz_energy,
        'total_energy': qw_energy + qx_energy + qy_energy + qz_energy,
    })

    return features

def extract_enhanced_frequency_features(samples: List[QuaternionSample], sampling_rate: int = 100) -> Dict[str, float]:
    """Extract enhanced frequency-domain features"""
    try:
        from scipy import signal
    except ImportError:
        logger.warning("scipy not available, skipping frequency features")
        return {}

    if len(samples) < 10:
        return {}

    qw_vals = np.array([s.qw for s in samples])

    try:
        # Compute power spectral density
        freqs, psd = signal.welch(qw_vals, fs=sampling_rate, nperseg=min(256, len(qw_vals)))

        # Dominant frequency
        peak_idx = np.argmax(psd)
        dominant_freq = float(freqs[peak_idx])

        # Frequency band energies (exercise-specific bands)
        slow_freq_mask = (freqs >= 0.1) & (freqs < 1.0)    # Very slow movements
        low_freq_mask = (freqs >= 1.0) & (freqs < 3.0)     # Slow movements
        mid_freq_mask = (freqs >= 3.0) & (freqs < 8.0)     # Normal exercise pace
        high_freq_mask = (freqs >= 8.0) & (freqs < 15.0)   # Fast movements
        very_high_mask = (freqs >= 15.0) & (freqs < 25.0)  # Very fast/tremor

        features = {
            'fft_dominant_freq': dominant_freq,
            'fft_slow_band_energy': float(np.sum(psd[slow_freq_mask])),
            'fft_low_band_energy': float(np.sum(psd[low_freq_mask])),
            'fft_mid_band_energy': float(np.sum(psd[mid_freq_mask])),
            'fft_high_band_energy': float(np.sum(psd[high_freq_mask])),
            'fft_very_high_energy': float(np.sum(psd[very_high_mask])),
            'fft_total_energy': float(np.sum(psd)),
            'fft_spectral_centroid': float(np.sum(freqs * psd) / np.sum(psd)),
            'fft_spectral_rolloff': float(freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]),
            'fft_spectral_entropy': float(-np.sum(psd * np.log2(psd + 1e-10)) / np.log2(len(psd))),
        }

        return features
    except Exception as e:
        logger.warning(f"Error computing frequency features: {e}")
        return {}

def process_window_samples(window: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single window to add enhanced features"""
    if 'samples' not in window:
        return window

    # Convert stored samples back to QuaternionSample objects
    samples = []
    for sample_data in window['samples']:
        samples.append(QuaternionSample(
            absolute_timestamp_ms=sample_data.get('absolute_timestamp_ms', 0),
            relative_timestamp_ms=sample_data.get('relative_timestamp_ms', 0),
            qw=sample_data.get('qw', 0),
            qx=sample_data.get('qx', 0),
            qy=sample_data.get('qy', 0),
            qz=sample_data.get('qz', 0),
        ))

    # Apply quality processing
    samples = normalize_quaternions(samples)
    samples = detect_quaternion_outliers(samples)

    if len(samples) < 3:
        logger.warning(f"Window has too few samples after processing: {len(samples)}")
        return window

    # Extract enhanced features
    temporal_features = extract_temporal_features(samples)
    frequency_features = extract_enhanced_frequency_features(samples)

    # Merge with existing features
    enhanced_features = window.get('features', {}).copy()
    enhanced_features.update(temporal_features)
    enhanced_features.update(frequency_features)

    # Add quality metrics
    enhanced_features.update({
        'processed_sample_count': len(samples),
        'quality_score': len(samples) / len(window['samples']) if window['samples'] else 0,
        'processing_timestamp': datetime.now().isoformat(),
    })

    # Update window
    window['features'] = enhanced_features
    window['processed_samples'] = len(samples)

    return window

def process_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """Process all windows in a session"""
    logger.info(f"Processing session with {len(session.get('sorted_windows', []))} windows")

    processed_windows = []
    for i, window in enumerate(session.get('sorted_windows', [])):
        if i % 50 == 0:
            logger.info(f"Processing window {i+1}/{len(session.get('sorted_windows', []))}")

        processed_window = process_window_samples(window)
        processed_windows.append(processed_window)

    session['sorted_windows'] = processed_windows
    session['post_processing'] = {
        'timestamp': datetime.now().isoformat(),
        'features_added': [
            'temporal_velocity_acceleration',
            'zero_crossing_rates',
            'enhanced_frequency_analysis',
            'quality_metrics'
        ],
        'total_windows_processed': len(processed_windows)
    }

    return session

def main():
    """Main post-processing function"""
    print("=" * 70)
    print("HAR DATA POST-PROCESSING - ENHANCED FEATURES")
    print("=" * 70)

    # Connect to MongoDB
    client, collection = connect_to_mongodb()

    try:
        # Get all sessions
        sessions = list(collection.find({}))
        logger.info(f"Found {len(sessions)} sessions to process")

        # Check for already processed sessions
        already_processed = sum(1 for s in sessions if 'post_processing' in s)
        if already_processed > 0:
            logger.warning(f"⚠ Found {already_processed} sessions that have already been post-processed")
            logger.warning(f"These will be re-processed (results should be identical)")

        # Process each session
        processed_count = 0
        skipped_count = 0
        for session in sessions:
            session_id = session.get('_id')

            # Skip if already processed (optional - comment out to force reprocessing)
            if 'post_processing' in session:
                logger.info(f"⏭ Skipping already processed session {session_id}")
                skipped_count += 1
                continue

            logger.info(f"Processing session {session_id}")

            # Process the session
            processed_session = process_session(session)

            # Update in database
            result = collection.replace_one({'_id': session_id}, processed_session)
            if result.modified_count > 0:
                processed_count += 1
                logger.info(f"✓ Updated session {session_id}")
            else:
                logger.warning(f"✗ Failed to update session {session_id}")

        logger.info(f"\n{'='*70}")
        logger.info(f"POST-PROCESSING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Sessions processed: {processed_count}")
        logger.info(f"Sessions skipped (already processed): {skipped_count}")
        logger.info(f"Total sessions in database: {len(sessions)}")
        logger.info(f"Enhanced features added:")
        logger.info(f"  • Temporal: velocity, acceleration, zero-crossing rates")
        logger.info(f"  • Frequency: enhanced spectral analysis")
        logger.info(f"  • Quality: normalization, outlier detection")
        logger.info(f"  • Energy: movement intensity metrics")

    except Exception as e:
        logger.error(f"Error during post-processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        logger.info("✓ MongoDB connection closed")

if __name__ == "__main__":
    main()