
#!/usr/bin/env python3
"""
HAR System - Data Collection Script
Raspberry Pi BLE Interface for Arduino Nano 33 IoT Sensor Nodes

This script provides an interactive menu for:
- Starting/stopping data collection sessions
- Calibrating sensor nodes
- Monitoring node status
- Configuring recording parameters
"""

import asyncio
import sys
import json
import struct
import os
import math
import time
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from bleak import BleakClient, BleakScanner
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import deque
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from post_process_data import extract_temporal_features, extract_enhanced_frequency_features

# ML Model Imports
MODELS_LOADED = False
try:
    import joblib
    MODELS_LOADED = True
except Exception:
    MODELS_LOADED = False

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('har_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Attempt to load ML models (after logger is available)
try:
    if MODELS_LOADED:
        exercise_model = joblib.load('exercise_classifier_model.joblib')
        exercise_label_encoder = joblib.load('label_encoder.joblib')
        exercise_features = joblib.load('exercise_classifier_features.joblib')

        rep_detector_s1 = joblib.load('rep_detector_model_s1.joblib')
        rep_counter_s2 = joblib.load('rep_counter_model_s2.joblib')
        rep_counter_features = joblib.load('rep_counter_features.joblib')

        session_model = joblib.load('session_reps_model.joblib')
        session_features = joblib.load('session_reps_features.joblib')

        logger.info("✓ All ML models loaded successfully")
    else:
        logger.warning("joblib not available; ML models will not be loaded")
        exercise_model = exercise_label_encoder = exercise_features = None
        rep_detector_s1 = rep_counter_s2 = rep_counter_features = None
        session_model = session_features = None
        MODELS_LOADED = False
except Exception as e:
    logger.error(f"Failed to load ML models: {e}")
    MODELS_LOADED = False
    exercise_model = exercise_label_encoder = exercise_features = None
    rep_detector_s1 = rep_counter_s2 = rep_counter_features = None
    session_model = session_features = None

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'har_system')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'training_data')

# MongoDB Connection
mongodb_client = None
mongodb_db = None
mongodb_collection = None

def init_mongodb():
    """Initialize MongoDB connection"""
    global mongodb_client, mongodb_db, mongodb_collection
    try:
        logger.info("Connecting to MongoDB...")
        mongodb_client = MongoClient(MONGODB_URI)
        # Test the connection
        mongodb_client.admin.command('ping')
        mongodb_db = mongodb_client[MONGODB_DB_NAME]
        mongodb_collection = mongodb_db[MONGODB_COLLECTION]
        logger.info(f"Successfully connected to MongoDB: {MONGODB_DB_NAME}.{MONGODB_COLLECTION}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.warning("Continuing without MongoDB - data will only be saved to JSON files")
        return False

def close_mongodb():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        try:
            mongodb_client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

def save_session_to_mongodb(session_data: Dict[str, Any]) -> bool:
    """
    Save a recording session to MongoDB
    
    Args:
        session_data: Complete session data dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    global mongodb_collection
    
    if mongodb_collection is None:
        logger.warning("MongoDB not connected - skipping database save")
        return False
    
    try:
        # Add MongoDB-specific metadata
        session_data['_created_at'] = datetime.now()
        session_data['_data_source'] = 'raspberry_pi_ble'
        
        # Insert the document
        result = mongodb_collection.insert_one(session_data)
        logger.info(f"Session saved to MongoDB with ID: {result.inserted_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving session to MongoDB: {e}")
        return False


# BLE BANDWIDTH OPTIMIZATION NOTES:
# With 4 sensors transmitting simultaneously at 100Hz, each sending:
# - Short windows (1.5s): ~303 packets/window (3 feature + 300 sample packets)
# - Long windows (3s): ~603 packets/window (3 feature + 600 sample packets)
# 
# Total bandwidth: ~1.8-3.6 Mbps shared across all devices
# BLE 5.0 theoretical max: ~2 Mbps, practical: ~1-1.5 Mbps
#
# Mitigation strategies implemented:
# 1. Concurrent processing to handle high packet rates
# 2. Per-device buffering to prevent data loss
# 3. Connection monitoring to detect dropped connections
#
# Firmware-level optimizations (in SensorNode.ino):
# - Stagger packet transmissions between devices
# - Add small delays between packets (1-5ms)
# - Use BLE 5.0 if available for higher bandwidth
# - Consider batching samples or reducing sample rate if needed
class NodePlacement(IntEnum):
    """Node placement identifiers matching Arduino firmware"""
    WRIST = 0
    BICEP = 1
    CHEST = 2
    THIGH = 3

class ExerciseType(IntEnum):
    """Exercise type identifiers"""
    BICEP_CURL = 1
    BENCH_PRESS = 2
    SQUAT = 3
    PULL_UP = 4

# BLE UUIDs (matching Arduino)
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
ORIENTATION_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
CONTROL_CHAR_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

# BLE Command IDs (matching Arduino firmware)
CMD_START = 0x01
CMD_STOP = 0x02
CMD_CALIBRATE = 0x03
CMD_SET_EXERCISE = 0x04
CMD_RESET = 0x05
CMD_TIME_SYNC = 0x06
CMD_HEARTBEAT = 0xFF

# Node placement names
NODE_NAMES = {
    0: "WRIST",
    1: "BICEP",
    2: "CHEST",
    3: "THIGH"
}


# DATA structs
@dataclass
class RecordingConfig:
    """Configuration for a data collection session"""
    exercise_type: ExerciseType
    rep_count: int
    participant_id: int
    session_notes: str = ""

@dataclass
class NodeStatus:
    """Status information for a connected node"""
    node_id: NodePlacement
    is_running: bool
    is_calibrated: bool
    timestamp: int
    connected: bool = False

@dataclass
class WindowFeatures:
    """Features extracted from a window"""
    window_type: int  # 0=short, 1=long
    node_id: int
    window_id: int
    qw_mean: float
    qx_mean: float
    qy_mean: float
    qz_mean: float
    qw_std: float
    qx_std: float
    qy_std: float
    qz_std: float
    sma: float
    dominant_freq: float
    total_samples: int

@dataclass
class QuaternionSample:
    """Individual quaternion sample with synchronized timestamps"""
    absolute_timestamp_ms: int  # UTC timestamp from session start (for global ordering)
    relative_timestamp_ms: int  # Original timestamp from Arduino (for debugging)
    qw: float
    qx: float
    qy: float
    qz: float

@dataclass
class WindowData:
    """Complete window data including features and samples"""
    metadata: Dict[str, Any]
    features: WindowFeatures
    samples: List[QuaternionSample]

@dataclass
class RawQuaternionPacket:
    """Raw quaternion data packet from Arduino"""
    sequence_number: int
    node_id: int
    timestamp: int
    qw: float
    qx: float
    qy: float
    qz: float

# FEATURE EXTRACTION FUNCTIONS (RPI-SIDE)

def calculate_mean(samples: List[QuaternionSample]) -> tuple[float, float, float, float]:
    """Calculate mean of quaternion components"""
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    sum_qw = sum(s.qw for s in samples)
    sum_qx = sum(s.qx for s in samples)
    sum_qy = sum(s.qy for s in samples)
    sum_qz = sum(s.qz for s in samples)
    
    return sum_qw/n, sum_qx/n, sum_qy/n, sum_qz/n

def calculate_std(samples: List[QuaternionSample], mean_qw: float, mean_qx: float, 
                  mean_qy: float, mean_qz: float) -> tuple[float, float, float, float]:
    """Calculate standard deviation of quaternion components"""
    import math
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    sum_sq_qw = sum((s.qw - mean_qw)**2 for s in samples)
    sum_sq_qx = sum((s.qx - mean_qx)**2 for s in samples)
    sum_sq_qy = sum((s.qy - mean_qy)**2 for s in samples)
    sum_sq_qz = sum((s.qz - mean_qz)**2 for s in samples)
    
    return (math.sqrt(sum_sq_qw/n), math.sqrt(sum_sq_qx/n), 
            math.sqrt(sum_sq_qy/n), math.sqrt(sum_sq_qz/n))

def calculate_sma(samples: List[QuaternionSample]) -> float:
    """Calculate Signal Magnitude Area"""
    n = len(samples)
    if n == 0:
        return 0.0
    
    sma = sum(abs(s.qw) + abs(s.qx) + abs(s.qy) + abs(s.qz) for s in samples)
    return sma / n

def calculate_dominant_frequency(samples: List[QuaternionSample], sampling_rate: int = 100) -> float:
    """Calculate dominant frequency using zero-crossing rate"""
    import math
    
    if len(samples) < 2:
        return 0.0
    
    # Calculate quaternion magnitudes
    magnitudes = [math.sqrt(s.qw**2 + s.qx**2 + s.qy**2 + s.qz**2) for s in samples]
    
    # Count zero crossings in the derivative
    zero_crossings = 0
    for i in range(2, len(magnitudes)):
        diff = magnitudes[i] - magnitudes[i-1]
        prev_diff = magnitudes[i-1] - magnitudes[i-2]
        
        if (diff > 0 and prev_diff < 0) or (diff < 0 and prev_diff > 0):
            zero_crossings += 1
    
    # Frequency = (zero crossings / 2) / window duration
    window_duration = len(samples) / sampling_rate
    frequency = (zero_crossings / 2.0) / window_duration
    
    return frequency

def extract_window_features(samples: List[QuaternionSample], window_type: int, 
                           node_id: int, window_id: int) -> WindowFeatures:
    """Extract all features from a window of quaternion samples"""
    # Calculate mean
    mean_qw, mean_qx, mean_qy, mean_qz = calculate_mean(samples)
    
    # Calculate standard deviation
    std_qw, std_qx, std_qy, std_qz = calculate_std(samples, mean_qw, mean_qx, mean_qy, mean_qz)
    
    # Calculate SMA
    sma = calculate_sma(samples)
    
    # Calculate dominant frequency
    dominant_freq = calculate_dominant_frequency(samples)
    
    return WindowFeatures(
        window_type=window_type,
        node_id=node_id,
        window_id=window_id,
        qw_mean=mean_qw,
        qx_mean=mean_qx,
        qy_mean=mean_qy,
        qz_mean=mean_qz,
        qw_std=std_qw,
        qx_std=std_qx,
        qy_std=std_qy,
        qz_std=std_qz,
        sma=sma,
        dominant_freq=dominant_freq,
        total_samples=len(samples)
    )


# ------------------------------
# Live Classification UI + Handler
# ------------------------------
class LiveClassificationUI(tk.Tk):
    """Tkinter UI for live exercise classification"""
    
    def __init__(self, callback_handler):
        super().__init__()
        self.callback_handler = callback_handler
        self.title("HAR Live Classification")
        self.geometry("600x500")
        self.minsize(500, 400)
        
        self.is_recording = False
        self.total_reps = 0
        self.window_count = 0
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI layout"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Live Exercise Classification", 
                         font=("Segoe UI", 20, "bold"))
        title.pack(pady=(0, 20))
        
        # Exercise Label
        self.exercise_label = ttk.Label(main_frame, text="—", 
                                       font=("Segoe UI", 32, "bold"))
        self.exercise_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = ttk.Label(main_frame, text="—%", 
                                         font=("Segoe UI", 18))
        self.confidence_label.pack(pady=5)
        
        # Rep Counter
        self.rep_label = ttk.Label(main_frame, text="Total Reps: 0", 
                                   font=("Segoe UI", 20))
        self.rep_label.pack(pady=15)
        
        # Window Rep Status
        self.window_rep_label = ttk.Label(main_frame, text="Window Status: —", 
                                         font=("Segoe UI", 16))
        self.window_rep_label.pack(pady=5)
        
        # Window Counter
        self.window_label = ttk.Label(main_frame, text="Windows Processed: 0", 
                                      font=("Segoe UI", 14))
        self.window_label.pack(pady=5)
        
        # Exercise Summary (initially hidden)
        self.exercise_summary_frame = ttk.Frame(main_frame)
        self.exercise_summary_frame.pack(pady=15)
        self.exercise_summary_label = ttk.Label(self.exercise_summary_frame, 
                                               text="", 
                                               font=("Segoe UI", 14, "bold"),
                                               foreground="blue")
        
        # Status Log
        log_label = ttk.Label(main_frame, text="Activity Log:", 
                             font=("Segoe UI", 12, "bold"))
        log_label.pack(pady=(10, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=6, wrap="word", 
                               font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", 
                                 command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Set", 
                                    command=self._on_start)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Set", 
                                   command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
    def _on_start(self):
        """Handle start button click"""
        self.is_recording = True
        self.total_reps = 0
        self.window_count = 0
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.exercise_summary_label.config(text="")
        self.window_rep_label.config(text="Window Status: —")
        self.log("🟢 Set started")
        self.callback_handler.on_start()
        
    def _on_stop(self):
        """Handle stop button click"""
        self.is_recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log("🔴 Set stopped - Computing summary...")
        self.callback_handler.on_stop()
        
    def update_classification(self, exercise: str, confidence: float):
        """Update exercise classification display"""
        self.exercise_label.config(text=exercise)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        
    def update_reps(self, reps_delta: int):
        """Update rep counter"""
        self.total_reps += reps_delta
        self.rep_label.config(text=f"Total Reps: {self.total_reps}")
        
    def increment_window(self):
        """Increment window counter"""
        self.window_count += 1
        self.window_label.config(text=f"Windows Processed: {self.window_count}")
        
    def show_session_prediction(self, predicted_reps: int):
        """Show final session rep prediction"""
        self.session_label.config(
            text=f"Final Session Prediction: {predicted_reps} reps"
        )
        self.session_label.pack()
        self.log(f"📊 Session model predicts: {predicted_reps} total reps")
    
    def reset_rep_counter(self):
        """Reset the rep counter display for new exercise"""
        self.total_reps = 0
        self.rep_label.config(text="Total Reps: 0")
    
    def update_window_rep_status(self, has_reps: bool):
        """Update window rep detection status"""
        if has_reps:
            self.window_rep_label.config(text="Window Status: REPS DETECTED", foreground="green")
        else:
            self.window_rep_label.config(text="Window Status: NO REPS", foreground="red")
    
    def show_exercise_summary(self, summary_text: str):
        """Show exercise summary at end of session"""
        self.exercise_summary_label.config(text=summary_text)
        self.exercise_summary_label.pack()
        
    def log(self, message: str):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")


class LiveClassificationHandler:
    """Handles the connection between data collection and UI"""
    
    def __init__(self, menu):
        self.menu = menu
        self.ui = None
        self.session_windows = []
        self.is_active = False
        self.window_buffer = {}  # Buffer to collect features from all sensors for each window
        
        # Live classification data storage
        self.live_predictions = []  # Store prediction results with timestamps
        self.live_sensor_windows = []  # Store raw sensor window data
        self.session_start_time = None
        
        # Exercise and rep tracking
        self.current_exercise = None
        self.exercise_reps = {}  # Track reps per exercise
        self.total_reps = 0
        
        # Exercise change detection
        self.prediction_history = []  # Store recent predictions for change detection
        self.history_size = 3  # Look at last 3 predictions for change detection
        
    def on_start(self):
        """Called when Start Set button is clicked"""
        self.is_active = True
        self.session_windows = []
        self.window_buffer = {}  # Clear window buffer for new session
        
        # Initialize live classification data storage
        self.live_predictions = []
        self.live_sensor_windows = []
        self.session_start_time = datetime.now()
        
        # Reset exercise and rep tracking
        self.current_exercise = None
        self.exercise_reps = {}
        self.total_reps = 0
        self.prediction_history = []  # Reset prediction history
        
        # Schedule coroutine on the menu's asyncio loop from UI thread
        if hasattr(self.menu, 'async_loop') and self.menu.async_loop:
            asyncio.run_coroutine_threadsafe(self.menu.start_live_collection(), self.menu.async_loop)
        else:
            logger.error("No async loop available for scheduling start_live_collection")
        
    def on_stop(self):
        """Called when Stop Set button is clicked"""
        self.is_active = False
        # Schedule coroutine on the menu's asyncio loop from UI thread
        if hasattr(self.menu, 'async_loop') and self.menu.async_loop:
            asyncio.run_coroutine_threadsafe(self.menu.stop_live_collection(), self.menu.async_loop)
        else:
            logger.error("No async loop available for scheduling stop_live_collection")
        
    def process_window_features(self, features: Dict[str, Any], 
                                node_id: int, node_name: str):
        """Process features from a single window - collect from all sensors before predicting"""
        if not self.is_active or not MODELS_LOADED:
            return
            
        try:
            window_start = features.get('window_start_ms')
            if window_start is None:
                logger.error(f"No window_start_ms in features from {node_name}")
                return
            
            logger.info(f"Received window from {node_name} at timestamp {window_start}")
            
            # Find or create a window group within tolerance (1 second)
            window_group_key = self._find_window_group(window_start)
            
            # Add sensor suffix to feature names to match training format
            sensor_suffix = f"_{node_name.upper()}"
            features_with_suffix = {f"{key}{sensor_suffix}": value for key, value in features.items()}
            
            # Store features for this sensor in the window group
            self.window_buffer[window_group_key][node_name] = features_with_suffix
            
            logger.info(f"📊 Window buffer has {len(self.window_buffer)} groups, current group {window_group_key} has {len(self.window_buffer[window_group_key])} sensors: {list(self.window_buffer[window_group_key].keys())}")
            
            # Check if we have data from all 4 sensors for this window group
            expected_sensors = {'CHEST', 'BICEP', 'WRIST', 'THIGH'}
            current_sensors = set(self.window_buffer[window_group_key].keys())
            
            if expected_sensors.issubset(current_sensors):
                # All sensors have data - make combined prediction
                logger.info(f"🎯 All sensors ready for window {window_group_key} - making prediction!")
                self._process_complete_window(window_group_key)
                
                # Clean up old buffers (keep only recent windows)
                self._cleanup_old_buffers(window_group_key)
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
            if self.ui:
                self.ui.log(f"Processing error: {str(e)[:50]}")
    
    def _detect_exercise_change(self, exercise: str, confidence: float) -> bool:
        """Detect if exercise has changed based on prediction history and confidence"""
        # Add current prediction to history
        self.prediction_history.append((exercise, confidence))
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # If we don't have enough history yet, only change if confidence is high
        if len(self.prediction_history) < 2:
            return self.current_exercise != exercise and confidence > 0.7
        
        # Check if majority of recent predictions agree on different exercise
        recent_exercises = [pred[0] for pred in self.prediction_history]
        recent_confidences = [pred[1] for pred in self.prediction_history]
        
        # Count votes for each exercise
        from collections import Counter
        exercise_counts = Counter(recent_exercises)
        
        # Get the most common exercise in recent history
        most_common_exercise, count = exercise_counts.most_common(1)[0]
        
        # Change if:
        # 1. Most common recent exercise is different from current
        # 2. At least 2 out of last 3 predictions agree (for history_size=3)
        # 3. Current prediction has reasonable confidence (>50%)
        should_change = (most_common_exercise != self.current_exercise and 
                        count >= max(2, len(self.prediction_history) // 2 + 1) and
                        confidence > 0.5)
        
        # Also change immediately if confidence drops significantly (possible transition)
        if self.current_exercise and len(recent_confidences) >= 2:
            avg_recent_confidence = sum(recent_confidences[-2:]) / 2
            if confidence < avg_recent_confidence * 0.6 and confidence < 0.8:
                # Confidence dropped significantly, be more willing to change
                should_change = should_change or (exercise != self.current_exercise and confidence > 0.4)
        
        return bool(should_change)  # Ensure Python bool is returned
    
    def _process_complete_window(self, window_start: int):
        """Process a complete window with data from all sensors"""
        try:
            # Start latency tracking
            processing_start_time = time.time()
            
            # Combine features from all sensors
            combined_features = {}
            for sensor_features in self.window_buffer[window_start].values():
                combined_features.update(sensor_features)
            
            # Remove window_start_ms features (not needed for prediction)
            combined_features = {k: v for k, v in combined_features.items() 
                               if not k.endswith('_ms')}
            
            logger.debug(f"Combined features for window {window_start}: {len(combined_features)} features")
            
            # Track feature combination latency
            feature_combination_time = time.time()
            feature_combination_latency = (feature_combination_time - processing_start_time) * 1000
            
            # 1. Exercise Classification with combined features
            exercise, confidence, scores = self._predict_exercise(combined_features)
            
            # Check if exercise changed using smart detection
            exercise_changed = self._detect_exercise_change(exercise, confidence)
            exercise_changed = bool(exercise_changed)  # Convert numpy.bool_ to Python bool
            if exercise_changed:
                old_exercise = self.current_exercise
                self.current_exercise = exercise
                if old_exercise is not None:
                    if self.ui:
                        self.ui.log(f"🔄 Exercise changed from {old_exercise} to {exercise}")
                # Reset UI rep counter for new exercise
                if self.ui:
                    self.ui.reset_rep_counter()
                    self.ui.log(f"🎯 Now tracking: {exercise}")
            
            if self.ui:
                self.ui.update_classification(exercise, confidence)
                self.ui.log(f"🏋️ Detected: {exercise} ({confidence*100:.1f}%)")
            
            # Track exercise classification latency
            exercise_classification_time = time.time()
            exercise_classification_latency = (exercise_classification_time - feature_combination_time) * 1000
            
            # 2. Rep Detection (Window-level) - use combined features
            reps_in_window, rep_confidence = self._predict_window_reps_with_confidence(combined_features)
            
            # Update rep tracking
            if reps_in_window > 0:
                self.total_reps += reps_in_window
                if exercise not in self.exercise_reps:
                    self.exercise_reps[exercise] = 0
                self.exercise_reps[exercise] += reps_in_window
                
                if self.ui:
                    self.ui.update_reps(reps_in_window)
                    self.ui.log(f"✓ +{reps_in_window} rep(s) detected ({exercise})")
            else:
                if self.ui:
                    self.ui.log(f"⭕ No reps detected in this window")
            
            # Update UI with window rep status
            if self.ui:
                self.ui.update_window_rep_status(reps_in_window > 0)
            
            # Track rep detection latency
            rep_detection_time = time.time()
            rep_detection_latency = (rep_detection_time - exercise_classification_time) * 1000
            
            # Calculate end-to-end latency
            processing_end_time = time.time()
            processing_latency_ms = (processing_end_time - processing_start_time) * 1000
            
            # Calculate data collection latency (from first sensor data to processing start)
            first_sensor_time = min(sensor_features.get('window_start_ms', window_start) 
                                  for sensor_features in self.window_buffer[window_start].values())
            data_collection_latency_ms = (processing_start_time * 1000) - first_sensor_time
            
            # Store prediction data with enhanced confidence and latency info
            prediction_data = {
                'timestamp_ms': int(window_start),
                'exercise': exercise,
                'confidence': float(confidence),
                'scores': {k: float(v) for k, v in scores.items()},
                'reps_detected': int(reps_in_window),
                'rep_confidence': rep_confidence,
                'sensor_count': len(self.window_buffer[window_start]),
                'exercise_changed': exercise_changed,
                'latency_ms': {
                    'feature_combination': round(feature_combination_latency, 2),
                    'exercise_classification': round(exercise_classification_latency, 2),
                    'rep_detection': round(rep_detection_latency, 2),
                    'processing_latency': round(processing_latency_ms, 2),
                    'data_collection_latency': round(data_collection_latency_ms, 2),
                    'total_latency': round(processing_latency_ms + data_collection_latency_ms, 2)
                }
            }
            
            # Add individual confidence fields for each exercise
            for exercise_name, confidence_value in scores.items():
                field_name = f"{exercise_name.lower()}_confidence"
                prediction_data[field_name] = round(float(confidence_value), 4)
            self.live_predictions.append(prediction_data)
            
            # Store sensor window data
            window_data = {
                'window_start_ms': int(window_start),
                'sensor_windows': {}
            }
            
            for sensor_name, sensor_features in self.window_buffer[window_start].items():
                # Store the raw sensor features (with suffixes)
                window_data['sensor_windows'][sensor_name] = {
                    'features': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in sensor_features.items()}
                }
            
            self.live_sensor_windows.append(window_data)
            
            # Store original features (without suffixes) for session-level prediction
            # Combine original features from all sensors
            session_features = {}
            for sensor_name, sensor_features in self.window_buffer[window_start].items():
                # Remove suffix from keys for session storage
                original_features = {k.replace(f"_{sensor_name}", ""): float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in sensor_features.items()}
                session_features.update(original_features)
            
            self.session_windows.append(session_features)
            if self.ui:
                self.ui.increment_window()
            
        except Exception as e:
            logger.error(f"Error processing complete window: {e}")
            if self.ui:
                self.ui.log(f"Window processing error: {str(e)[:50]}")
    
    def _find_window_group(self, timestamp: int, tolerance_ms: int = 1000) -> int:
        """Find or create a window group for the given timestamp within tolerance"""
        logger.info(f"🔍 Looking for window group for timestamp {timestamp}")
        # Look for existing window groups within tolerance
        for existing_timestamp in self.window_buffer.keys():
            if abs(timestamp - existing_timestamp) <= tolerance_ms:
                logger.info(f"Found existing group {existing_timestamp} (diff: {abs(timestamp - existing_timestamp)}ms)")
                return existing_timestamp
        
        # No matching group found, create new one
        logger.info(f"🆕 Creating new window group {timestamp}")
        self.window_buffer[timestamp] = {}
        return timestamp
    
    def _cleanup_old_buffers(self, current_window: int, max_buffers: int = 10):
        """Clean up old window buffers to prevent memory buildup"""
        if len(self.window_buffer) <= max_buffers:
            return
            
        # Keep only the most recent buffers
        sorted_timestamps = sorted(self.window_buffer.keys())
        buffers_to_remove = sorted_timestamps[:-max_buffers]  # Keep the last max_buffers
        
        for old_timestamp in buffers_to_remove:
            if old_timestamp != current_window:  # Don't remove the current one
                del self.window_buffer[old_timestamp]
                logger.debug(f"🗑️ Cleaned up old buffer for window {old_timestamp}")
    
    def _show_exercise_summary(self):
        """Show summary of reps performed for each exercise"""
        if not self.exercise_reps:
            if self.ui:
                self.ui.log("No exercises tracked")
            return
            
        if self.ui:
            self.ui.log("Session Summary:")
            for exercise, reps in self.exercise_reps.items():
                self.ui.log(f"  {exercise}: {reps} reps")
            self.ui.log(f"  Total: {self.total_reps} reps")
            
            # Show summary in UI
            summary_text = "Session Complete!\n\n"
            for exercise, reps in self.exercise_reps.items():
                summary_text += f"{exercise}: {reps} reps\n"
            summary_text += f"\nTotal: {self.total_reps} reps"
            
            self.ui.show_exercise_summary(summary_text)

    
    def finalize_session(self):
        """Show exercise summary and save live classification data"""
        if not self.session_windows or not MODELS_LOADED:
            return
            
        try:
            # Show exercise summary instead of session prediction
            self._show_exercise_summary()
            
            # Save live classification data to JSON
            self._save_live_classification_data()
            
        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            if self.ui:
                self.ui.log(f"Session finalization failed")
    
    def _save_live_classification_data(self):
        """Save all live classification data to JSON files"""
        try:
            # Create live_collection_data directory
            data_dir = "live_collection_data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            
            # Prepare session metadata with enhanced latency and confidence tracking
            session_metadata = {
                'session_type': 'live_classification',
                'start_time': self.session_start_time.isoformat(),
                'start_time_utc_ms': int(self.session_start_time.timestamp() * 1000),
                'total_predictions': len(self.live_predictions),
                'total_sensor_windows': len(self.live_sensor_windows),
                'total_reps': self.total_reps,
                'exercises_performed': self.exercise_reps,
                'devices_connected': len(self.menu.connected_clients) if hasattr(self.menu, 'connected_clients') else 0
            }
            
            # Calculate latency statistics
            if self.live_predictions:
                processing_latencies = [p.get('latency_ms', {}).get('processing_latency', 0) for p in self.live_predictions]
                data_collection_latencies = [p.get('latency_ms', {}).get('data_collection_latency', 0) for p in self.live_predictions]
                total_latencies = [p.get('latency_ms', {}).get('total_latency', 0) for p in self.live_predictions]
                
                session_metadata['latency_stats'] = {
                    'processing_latency_ms': {
                        'mean': round(sum(processing_latencies) / len(processing_latencies), 2),
                        'min': round(min(processing_latencies), 2),
                        'max': round(max(processing_latencies), 2),
                        'std': round((sum((x - sum(processing_latencies)/len(processing_latencies))**2 for x in processing_latencies) / len(processing_latencies))**0.5, 2)
                    },
                    'data_collection_latency_ms': {
                        'mean': round(sum(data_collection_latencies) / len(data_collection_latencies), 2),
                        'min': round(min(data_collection_latencies), 2),
                        'max': round(max(data_collection_latencies), 2),
                        'std': round((sum((x - sum(data_collection_latencies)/len(data_collection_latencies))**2 for x in data_collection_latencies) / len(data_collection_latencies))**0.5, 2)
                    },
                    'total_latency_ms': {
                        'mean': round(sum(total_latencies) / len(total_latencies), 2),
                        'min': round(min(total_latencies), 2),
                        'max': round(max(total_latencies), 2),
                        'std': round((sum((x - sum(total_latencies)/len(total_latencies))**2 for x in total_latencies) / len(total_latencies))**0.5, 2)
                    }
                }
            
            # Calculate confidence statistics
            if self.live_predictions:
                exercise_confidences = [p['confidence'] for p in self.live_predictions]
                rep_confidences = [p.get('rep_confidence', {}).get('detection_confidence', 0) for p in self.live_predictions]
                
                session_metadata['confidence_stats'] = {
                    'exercise_confidence': {
                        'mean': round(sum(exercise_confidences) / len(exercise_confidences), 4),
                        'min': round(min(exercise_confidences), 4),
                        'max': round(max(exercise_confidences), 4),
                        'std': round((sum((x - sum(exercise_confidences)/len(exercise_confidences))**2 for x in exercise_confidences) / len(exercise_confidences))**0.5, 4)
                    },
                    'rep_detection_confidence': {
                        'mean': round(sum(rep_confidences) / len(rep_confidences), 4),
                        'min': round(min(rep_confidences), 4),
                        'max': round(max(rep_confidences), 4),
                        'std': round((sum((x - sum(rep_confidences)/len(rep_confidences))**2 for x in rep_confidences) / len(rep_confidences))**0.5, 4)
                    }
                }
            
            # Save predictions data with atomic write (write to temp file first, then rename)
            predictions_data = {
                'session_metadata': session_metadata,
                'predictions': self.live_predictions
            }
            
            predictions_filename = f"{data_dir}/live_predictions_{timestamp}.json"
            temp_predictions_filename = f"{predictions_filename}.tmp"
            
            try:
                # Write to temporary file first
                with open(temp_predictions_filename, 'w') as f:
                    json.dump(predictions_data, f, indent=2)
                
                # Atomic rename to final filename
                os.rename(temp_predictions_filename, predictions_filename)
            except Exception as e:
                # Clean up temp file if something went wrong
                if os.path.exists(temp_predictions_filename):
                    try:
                        os.remove(temp_predictions_filename)
                    except:
                        pass
                raise e
            
            # Save sensor windows data with atomic write
            sensor_data = {
                'session_metadata': session_metadata,
                'sensor_windows': self.live_sensor_windows
            }
            
            sensor_filename = f"{data_dir}/live_sensor_data_{timestamp}.json"
            temp_sensor_filename = f"{sensor_filename}.tmp"
            
            try:
                # Write to temporary file first
                with open(temp_sensor_filename, 'w') as f:
                    json.dump(sensor_data, f, indent=2)
                
                # Atomic rename to final filename
                os.rename(temp_sensor_filename, sensor_filename)
            except Exception as e:
                # Clean up temp file if something went wrong
                if os.path.exists(temp_sensor_filename):
                    try:
                        os.remove(temp_sensor_filename)
                    except:
                        pass
                raise e
            
            logger.info(f"Live classification data saved to {data_dir}/")
            if self.ui:
                self.ui.log(f"💾 Data saved to {data_dir}/")
            
        except Exception as e:
            logger.error(f"Error saving live classification data: {e}")
            if self.ui:
                self.ui.log(f"⚠️ Failed to save data: {str(e)[:50]}")
    
    def _predict_exercise(self, features: Dict) -> tuple:
        """Predict exercise type from features"""
        # For exercise classification, we need SINGLE-SENSOR features
        # The model was trained on individual sensor windows, not aggregated
        
        # Log feature keys for debugging
        logger.debug(f"Input features keys: {sorted(features.keys())}")
        
        if exercise_features is None:
            logger.error("exercise_features not loaded - cannot predict")
            return "UNKNOWN", 0.0, {}
        
        # Create feature vector matching training
        X = self._create_feature_vector(features, exercise_features)
        
        # Debug: log how many features matched
        non_zero = (X != 0).sum()
        logger.debug(f"Exercise prediction: {non_zero}/{len(exercise_features)} features non-zero")
        
        # DEBUG: Log input features and feature vector
        logger.info(f"DEBUG - Input features: {features}")
        logger.info(f"DEBUG - Feature vector shape: {X.shape}, non-zero features: {non_zero}/{len(exercise_features)}")
        logger.info(f"DEBUG - Feature vector sample (first 10): {X[0][:10]}")
        
        pred = exercise_model.predict(X)[0]
        proba = exercise_model.predict_proba(X)[0]
        
        exercise = exercise_label_encoder.inverse_transform([pred])[0]
        confidence = proba[pred]
        
        scores = dict(zip(exercise_label_encoder.classes_, proba))
        
        # DEBUG: Log model output
        logger.info(f"DEBUG - Model prediction: {pred} -> {exercise}")
        logger.info(f"DEBUG - Confidence: {confidence:.4f}")
        logger.info(f"DEBUG - All probabilities: {scores}")
        
        return exercise, confidence, scores
    
    def _predict_window_reps_with_confidence(self, features: Dict) -> tuple:
        """Predict reps in this window using hierarchical model with confidence scores"""
        X = self._create_feature_vector(features, rep_counter_features)
        
        # Stage 1: Activity Detection
        DETECTION_THRESHOLD = 0.40
        s1_proba = rep_detector_s1.predict_proba(X)[0, 1]
        
        if s1_proba <= DETECTION_THRESHOLD:
            # No activity detected
            return 0, {
                'stage1_probability': float(s1_proba),
                'stage1_confidence': float(s1_proba),
                'stage1_threshold': DETECTION_THRESHOLD,
                'stage2_probability': 0.0,  # No activity detected, so stage 2 not run
                'stage2_prediction': -1,   # Invalid prediction (stage 2 not run)
                'stage2_confidence': 0.0,  # No confidence in stage 2
                'final_prediction': 0,
                'detection_confidence': 1.0 - float(s1_proba)  # Confidence in "no activity"
            }
        
        # Stage 2: Rep Counting (1 vs 2+)
        s2_pred = rep_counter_s2.predict(X)[0]
        s2_proba = rep_counter_s2.predict_proba(X)[0]
        
        # For binary classification (1 vs 2+), s2_pred is 0 for 1 rep, 1 for 2+ reps
        reps_predicted = 1 if s2_pred == 0 else 2
        stage2_confidence = float(s2_proba[s2_pred])
        
        return reps_predicted, {
            'stage1_probability': float(s1_proba),
            'stage1_confidence': float(s1_proba),
            'stage1_threshold': DETECTION_THRESHOLD,
            'stage2_probability': float(s2_proba[1] if s2_pred == 1 else s2_proba[0]),  # Probability of predicted class
            'stage2_prediction': int(s2_pred),
            'stage2_confidence': stage2_confidence,
            'final_prediction': reps_predicted,
            'detection_confidence': stage2_confidence  # Overall confidence in the rep detection
        }
    
    def _predict_session_reps(self) -> int:
        """Predict total session reps from all windows"""
        # Aggregate window features
        df_windows = pd.DataFrame(self.session_windows)
        df_numeric = df_windows.select_dtypes(include=np.number)
        
        if df_numeric.empty:
            return 0
        
        # Create session-level aggregations
        aggregations = ['mean', 'std', 'min', 'max', 'median']
        df_agg = df_numeric.agg(aggregations)
        df_flat = df_agg.unstack().to_frame().T
        df_flat.columns = [f'{i}_{j}' for i, j in df_flat.columns]
        
        # Add temporal features if enough windows
        n_windows = len(df_numeric)
        if n_windows > 1:
            mid = n_windows // 2
            
            # First half
            df_first = df_numeric.iloc[:mid].agg(aggregations)
            df_first_flat = df_first.unstack().to_frame().T
            df_first_flat.columns = [f'first_half_{i}_{j}' 
                                     for i, j in df_first_flat.columns]
            
            # Second half
            df_second = df_numeric.iloc[mid:].agg(aggregations)
            df_second_flat = df_second.unstack().to_frame().T
            df_second_flat.columns = [f'second_half_{i}_{j}' 
                                      for i, j in df_second_flat.columns]
            
            # Trends
            time_idx = np.arange(n_windows)
            slopes = {}
            for col in df_numeric.columns:
                m, _ = np.polyfit(time_idx, df_numeric[col], 1)
                slopes[f'{col}_trend_slope'] = m
            df_slopes = pd.DataFrame([slopes])
            
            df_flat = pd.concat([df_flat, df_first_flat, 
                                df_second_flat, df_slopes], axis=1)
        
        # Align with training features
        X = self._align_features(df_flat, session_features)
        
        # Predict
        pred = session_model.predict(X)[0]
        return int(np.round(pred))
    
    def _create_feature_vector(self, features: Dict, 
                               expected_features: List[str]) -> np.ndarray:
        """Create aligned feature vector"""
        df = pd.DataFrame([features])

        # Reindex in one step to add missing columns and order correctly
        df = df.reindex(columns=expected_features, fill_value=0)
        return df.values
    
    def _align_features(self, df: pd.DataFrame, 
                       expected_features: List[str]) -> np.ndarray:
        """Align features with training set"""
        # Reindex to ensure all expected features are present and ordered
        df_aligned = df.reindex(columns=expected_features, fill_value=0)
        return df_aligned.fillna(0).values



# MENU SYSTEM
class DataCollectionMenu:
    """Interactive menu for HAR data collection"""
    
    def __init__(self):
        self.running = True
        self.recording_config: Optional[RecordingConfig] = None
        self.connected_clients: dict[str, BleakClient] = {}  # address -> client
        self.node_statuses: dict[int, NodeStatus] = {}  # node_id -> status
        
        # Initialize MongoDB connection
        self.mongodb_enabled = init_mongodb()
        
        # Window configuration (matching Arduino)
        self.SHORT_WINDOW_SIZE = 150
        self.SHORT_WINDOW_OVERLAP = 0.75
        self.SHORT_WINDOW_STEP = 38  # 150 * (1 - 0.75)
        self.LONG_WINDOW_SIZE = 300
        self.LONG_WINDOW_OVERLAP = 0.5
        self.LONG_WINDOW_STEP = 150  # 300 * (1 - 0.5)
        
        # BLE DATA BUFFERING CONFIGURATION
        # Large buffers to handle high-throughput multi-device BLE data
        # Each device can buffer up to 10000 packets (~100 seconds at 100Hz)
        self.PACKET_QUEUE_MAX_SIZE = 10000  # Per-device queue size
        self.packet_queues: Dict[str, asyncio.Queue] = {}  # address -> queue
        self.processing_tasks: List[asyncio.Task] = []  # Background processing tasks
        self.packets_dropped: Dict[str, int] = {}  # Track dropped packets per device
        self.packets_received: Dict[str, int] = {}  # Track received packets per device

        # Classification
        self.classification_mode: bool = False  # stream to ML when true 
        
        # Live classification
        self.classification_handler = None
        self.classification_ui = None
        self.classification_loop = None 

    def send_features_to_ml(self, *, node_id: int, node_name: str,
                            window_type: int, window_id: int,
                            start_ms: int, end_ms: int,
                            features: Dict[str, Any]) -> None:
        
        """Placeholder function to take data for ML"""
        return
    
    async def handle_live_classification(self):
        """Launch live classification UI"""
        print("\n" + "=" * 70)
        print("LIVE CLASSIFICATION")
        print("=" * 70)

        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        if not MODELS_LOADED:
            print("ERROR: ML models not loaded. Cannot start classification.")
            input("\nPress Enter to continue...")
            return

        print(f"Connected devices: {len(self.connected_clients)}")
        for address in self.connected_clients.keys():
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN")
            print(f"  - {node_name}")

        print("\nLaunching classification UI...")

        # Run UI in separate thread
        # Store the running asyncio loop so UI thread can schedule coroutines
        try:
            self.async_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - get default event loop
            self.async_loop = asyncio.get_event_loop()

        # Create handler and UI (after async_loop is set)
        self.classification_handler = LiveClassificationHandler(self)

        ui_thread = threading.Thread(target=self._run_ui, daemon=True)
        ui_thread.start()

        # Wait for UI to be ready
        await asyncio.sleep(1)

        print("✓ Classification UI ready")
        print("Use the UI to start/stop sets")
        print("Press Enter to close UI and return to menu...")

        await asyncio.get_event_loop().run_in_executor(None, input)

        # Cleanup
        if self.classification_ui:
            self.classification_ui.quit()

    def _run_ui(self):
        """Run UI in separate thread"""
        self.classification_ui = LiveClassificationUI(self.classification_handler)
        self.classification_handler.ui = self.classification_ui
        self.classification_ui.mainloop()

    async def start_live_collection(self):
        """Start live data collection for classification"""
        logger.info("Starting live collection...")
        
        # Initialize buffers
        device_buffers = {}
        for address in self.connected_clients.keys():
            device_buffers[address] = {
                'raw_samples': [],
                'short_window_buffer': [],
                'long_window_buffer': [],
                'samples_since_last_short': 0,
                'samples_since_last_long': 0,
                'short_window_id': 0,
                'long_window_id': 0,
                'windows': [],
                'stats': {'total_packets': 0, 'total_samples': 0},
                'session_absolute_start_ms': int(datetime.now().timestamp() * 1000)
            }
        
        # Store for cleanup
        self.live_buffers = device_buffers

        # Initialize packet queues
        for address in self.connected_clients.keys():
            self.packet_queues[address] = asyncio.Queue(maxsize=self.PACKET_QUEUE_MAX_SIZE)
            self.packets_received[address] = 0
            self.packets_dropped[address] = 0

        # Track first/last packet times per device (used by notification handler)
        self.first_packet_time = {}
        self.last_packet_time = {}
        for address in self.connected_clients.keys():
            self.first_packet_time[address] = None
            self.last_packet_time[address] = None

        # STEP 1: Synchronize timestamps across all devices
        print("\n" + "=" * 70)
        print("SYNCHRONIZING DEVICE TIMESTAMPS")
        print("=" * 70)
        
        # Use relative timestamp (milliseconds from session start = 0)
        sync_timestamp_ms = 0  # Session start is t=0
        
        print(f"Reference timestamp: {sync_timestamp_ms} ms (relative to session start)")
        print(f"Syncing {len(self.connected_clients)} device(s)...\n")
        
        # Send time sync command to all devices
        sync_tasks = []
        for address, client in self.connected_clients.items():
            task = self.send_time_sync_command(client, address, sync_timestamp_ms)
            sync_tasks.append(task)
        
        sync_results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        successful_syncs = sum(1 for result in sync_results if result is True)
        
        if successful_syncs == 0:
            logger.error("Failed to sync time on any device - aborting live collection")
            print("[FAILED] Time synchronization failed on all devices")
            return
        
        if successful_syncs < len(self.connected_clients):
            logger.warning(f"Time sync failed on {len(self.connected_clients) - successful_syncs} device(s)")
            print(f"[WARNING] Time synchronization failed on {len(self.connected_clients) - successful_syncs} device(s)")
            print("Live classification may have timing issues")
        
        print(f"[OK] Time sync completed on {successful_syncs}/{len(self.connected_clients)} devices")

        # Start notification handlers
        self.processing_tasks = []
        for address, client in self.connected_clients.items():
            # Packet processor
            proc_task = asyncio.create_task(
                self.process_packet_queue_live(address, device_buffers)
            )
            self.processing_tasks.append(proc_task)
            
            # BLE notifications
            notif_task = asyncio.create_task(
                self.start_device_notifications(address, client, device_buffers)
            )
            self.processing_tasks.append(notif_task)

        # Send start commands
        for address, client in self.connected_clients.items():
            await self.send_start_command(client, address)

        logger.info("Live collection started")

    async def stop_live_collection(self):
        """Stop live data collection"""
        logger.info("Stopping live collection...")

        # Send stop commands
        for address, client in self.connected_clients.items():
            await self.send_stop_command(client, address)

        # Cancel tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Finalize session prediction
        self.classification_handler.finalize_session()

        logger.info("Live collection stopped")

    async def process_packet_queue_live(self, address: str, 
                                       device_buffers: Dict[str, Dict[str, Any]]):
        """Process packets during live classification"""
        node_id = self.extract_node_id_from_name_by_address(address)
        node_name = NODE_NAMES.get(node_id, "UNKNOWN")
        
        try:
            while True:
                try:
                    addr, data = await asyncio.wait_for(
                        self.packet_queues[address].get(), 
                        timeout=0.1
                    )
                    
                    # Process packet (extract features when window completes)
                    self.handle_orientation_data_live(addr, data, device_buffers)
                    self.packet_queues[address].task_done()
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Packet processor stopped for {node_name}")

    def handle_orientation_data_live(self, address: str, data: bytes, 
                                     device_buffers: Dict[str, Dict[str, Any]]):
        """Handle orientation data and trigger ML predictions"""
        try:
            device_buffers[address]['stats']['total_packets'] += 1
            
            packet = self.parse_orientation_packet(data)
            if not packet:
                return
            
            device_buffers[address]['stats']['total_samples'] += 1
            
            buffer = device_buffers[address]
            
            # Create sample with timestamp
            session_start_utc_ms = buffer['session_absolute_start_ms']
            absolute_timestamp_ms = session_start_utc_ms + packet.timestamp
            
            sample = QuaternionSample(
                absolute_timestamp_ms=absolute_timestamp_ms,
                relative_timestamp_ms=packet.timestamp,
                qw=packet.qw,
                qx=packet.qx,
                qy=packet.qy,
                qz=packet.qz
            )
            
            buffer['raw_samples'].append(sample)
            buffer['long_window_buffer'].append(sample)
            
            if len(buffer['long_window_buffer']) > self.LONG_WINDOW_SIZE:
                buffer['long_window_buffer'].pop(0)
            
            buffer['samples_since_last_long'] += 1
            
            # Check if we should extract long window features
            if (buffer['samples_since_last_long'] >= self.LONG_WINDOW_STEP and
                len(buffer['long_window_buffer']) >= self.LONG_WINDOW_SIZE):
                
                # Sort and extract features
                sorted_buffer = sorted(buffer['long_window_buffer'], 
                                      key=lambda s: s.absolute_timestamp_ms)
                
                # Extract full feature set (matching training data)
                temporal_features = extract_temporal_features(sorted_buffer, sampling_rate=100)
                frequency_features = extract_enhanced_frequency_features(sorted_buffer, sampling_rate=100)
                
                # Basic features
                qw_vals = np.array([s.qw for s in sorted_buffer])
                qx_vals = np.array([s.qx for s in sorted_buffer])
                qy_vals = np.array([s.qy for s in sorted_buffer])
                qz_vals = np.array([s.qz for s in sorted_buffer])
                
                qw_mean, qx_mean, qy_mean, qz_mean = np.mean(qw_vals), np.mean(qx_vals), np.mean(qy_vals), np.mean(qz_vals)
                qw_std, qx_std, qy_std, qz_std = np.std(qw_vals), np.std(qx_vals), np.std(qy_vals), np.std(qz_vals)
                
                # SMA calculation
                sma = np.mean(np.abs(qw_vals) + np.abs(qx_vals) + np.abs(qy_vals) + np.abs(qz_vals))
                
                # Dominant frequency (simplified)
                try:
                    from scipy import signal
                    freqs, psd = signal.welch(qw_vals, fs=100, nperseg=min(256, len(qw_vals)))
                    dominant_freq = float(freqs[np.argmax(psd)])
                except:
                    dominant_freq = 0.0
                
                # Combine all features
                features_dict = {
                    'qw_mean': qw_mean,
                    'qx_mean': qx_mean,
                    'qy_mean': qy_mean,
                    'qz_mean': qz_mean,
                    'qw_std': qw_std,
                    'qx_std': qx_std,
                    'qy_std': qy_std,
                    'qz_std': qz_std,
                    'sma': sma,
                    'dominant_freq': dominant_freq,
                    'total_samples': len(sorted_buffer)
                }
                
                # Add temporal and frequency features
                features_dict.update(temporal_features)
                features_dict.update(frequency_features)
                
                # Add quality metrics (approximated for live classification)
                features_dict['processed_sample_count'] = len(sorted_buffer)
                features_dict['quality_score'] = 1.0  # All samples are considered valid in live mode
                
                # Include synchronized window start for aggregation
                window_start_timestamp_ms = sorted_buffer[0].absolute_timestamp_ms
                features_dict['window_start_ms'] = int(window_start_timestamp_ms)
                
                # Send to ML handler
                node_id = packet.node_id
                node_name = NODE_NAMES.get(node_id, "UNKNOWN")
                self.classification_handler.process_window_features(
                    features_dict, node_id, node_name
                )
                
                buffer['long_window_id'] += 1
                buffer['samples_since_last_long'] = 0
                
        except Exception as e:
            logger.error(f"Error processing live data from {address}: {e}")
        
        
    def clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def print_header(self):
        """Print menu header"""
        print("=" * 70)
        print("HAR DATA COLLECTION SYSTEM".center(70))
        print("Multi-Node Sensor Interface".center(70))
        print("=" * 70)
        print()
    
    def print_node_status(self):
        """Display current node statuses"""
        if self.node_statuses:
            print("+" + "-" * 68 + "+")
            print("| CONNECTED NODES" + " " * 52 + "|")
            print("+" + "-" * 68 + "+")
            for i, (node_id, status) in enumerate(self.node_statuses.items()):
                if i > 0:
                    print("+" + "-" * 68 + "+")
                
                status_color = "\033[92m" if status.is_running else "\033[93m"
                reset_color = "\033[0m"
                
                node_name = NODE_NAMES.get(node_id, "UNKNOWN")
                print(f"| {node_name} (ID: {node_id})" + " " * (45 - len(node_name) - len(str(node_id))) + " |")
                print(f"|   Connected:  {status_color}{'YES' if status.connected else 'NO'}{reset_color:<45}|")
                print(f"|   Running:    {status_color}{'YES' if status.is_running else 'NO'}{reset_color:<45}|")
                print(f"|   Calibrated: {'YES' if status.is_calibrated else 'NO':<45}|")
                print(f"|   Timestamp:  {status.timestamp:<45}|")
            
            print("+" + "-" * 68 + "+")
            print(f"Total connected: {len(self.node_statuses)}/{len(NODE_NAMES)} nodes")
            print()
        else:
            print("+" + "-" * 68 + "+")
            print("| NODE STATUS" + " " * 56 + "|")
            print("| No nodes connected" + " " * 49 + "|")
            print("+" + "-" * 68 + "+")
            print()
        
        # Display MongoDB connection status
        mongodb_status = "\033[92mCONNECTED\033[0m" if self.mongodb_enabled else "\033[93mDISCONNECTED\033[0m"
        print("+" + "-" * 68 + "+")
        print(f"| DATABASE STATUS: MongoDB {mongodb_status:<38}|")
        print("+" + "-" * 68 + "+")
        print()
    
    def print_menu_options(self):
        """Display menu options"""
        print("MAIN MENU:")
        print()
        print("  [1] Start Recording Session")
        print("  [2] Calibrate All Sensor Nodes")
        print("  [3] Request Status from All Nodes")
        print("  [4] View Current Configuration")
        print("  [5] Reset All Sensor Nodes")
        print("  [6] Live classification")
        print("  [0] Exit")
        print()
        print("-" * 70)
    
    def display_menu(self):
        """Display complete menu"""
        self.clear_screen()
        self.print_header()
        self.print_node_status()
        self.print_menu_options()
    
    def get_recording_config(self) -> Optional[RecordingConfig]:
        """Interactive prompt to configure recording session"""
        print()
        print("=" * 70)
        print("RECORDING SESSION CONFIGURATION")
        print("=" * 70)
        print()
        
        # Select exercise type
        print("Select Exercise Type:")
        for ex in ExerciseType:
            print(f"  [{ex.value}] {ex.name.replace('_', ' ').title()}")
        print()
        
        try:
            exercise_choice = int(input("Exercise [1-4]: ").strip())
            if exercise_choice not in [e.value for e in ExerciseType]:
                print("Invalid exercise type")
                return None
            exercise_type = ExerciseType(exercise_choice)
            
            # Get rep count
            rep_count = int(input("Number of reps: ").strip())
            if rep_count <= 0 or rep_count > 100:
                print("Rep count must be between 1 and 100")
                return None
            
            # Get participant ID
            participant_id = int(input("Participant ID: ").strip())
            if participant_id <= 0:
                print("Participant ID must be positive")
                return None
            
            # Optional notes
            session_notes = input("Session notes (optional): ").strip()
            
            config = RecordingConfig(
                exercise_type=exercise_type,
                rep_count=rep_count,
                participant_id=participant_id,
                session_notes=session_notes
            )
            
            # Confirm configuration
            print()
            print("-" * 70)
            print("Configuration Summary:")
            print(f"  Exercise: {exercise_type.name.replace('_', ' ').title()}")
            print(f"  Reps: {rep_count}")
            print(f"  Participant: {participant_id}")
            if session_notes:
                print(f"  Notes: {session_notes}")
            print("-" * 70)
            
            confirm = input("\nProceed with this configuration? [y/N]: ").strip().lower()
            if confirm == 'y':
                return config
            else:
                print("Configuration cancelled")
                return None
                
        except ValueError:
            print("Invalid input - please enter numbers only")
            return None
        except KeyboardInterrupt:
            print("\nConfiguration cancelled")
            return None
    
    def show_current_config(self):
        """Display current recording configuration"""
        print()
        if self.recording_config:
            print("CURRENT CONFIGURATION:")
            print(f"  Exercise: {self.recording_config.exercise_type.name.replace('_', ' ').title()}")
            print(f"  Reps: {self.recording_config.rep_count}")
            print(f"  Participant: {self.recording_config.participant_id}")
            if self.recording_config.session_notes:
                print(f"  Notes: {self.recording_config.session_notes}")
        else:
            print("No recording configuration set")
        print()
        input("Press Enter to continue...")
    
    def parse_orientation_packet(self, data: bytes) -> Optional[RawQuaternionPacket]:
        """
        Parse compressed quaternion orientation packet from Arduino.
        UPDATED for 14-byte packet (padded to 20 bytes): 4-byte timestamp and quantized quaternions.
        """
        # BLE packets are 20 bytes, but only first 14 bytes contain data (rest is padding)
        if len(data) != 20:
            logger.warning(f"Invalid packet length received: {len(data)} bytes. Expected 20.")
            return None
        
        try:
            # Extract first 14 bytes (data), ignore last 6 bytes (padding)
            data = data[:14]
            
            # Packet structure: seq(1) + nodeId(1) + timestamp(4) + qw(2) + qx(2) + qy(2) + qz(2)
            # <B: unsigned char (1), B: unsigned char (1), I: unsigned int (4), h: short (2)
            (
                sequence_number,
                node_id,
                timestamp,
                qw_quant,
                qx_quant,
                qy_quant,
                qz_quant,
            ) = struct.unpack("<BBIhhhh", data)

            # NEW: De-quantize the int16 values back to floats by dividing by the max value.
            CONVERSION_FACTOR = 32767.0
            qw = qw_quant / CONVERSION_FACTOR
            qx = qx_quant / CONVERSION_FACTOR
            qy = qy_quant / CONVERSION_FACTOR
            qz = qz_quant / CONVERSION_FACTOR

            # Validate quaternion components
            if any(math.isnan(v) or math.isinf(v) for v in [qw, qx, qy, qz]):
                logger.warning(f"Invalid quaternion values after de-quantization.")
                return None
            
            return RawQuaternionPacket(
                sequence_number=sequence_number,
                node_id=node_id,
                timestamp=timestamp,  # This is the 4-byte relative timestamp
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
            )
        except (struct.error, ValueError) as e:
            logger.error(f"Error parsing compressed orientation packet: {e}")
            return None
    
    def parse_heartbeat_response(self, data: bytes):
        """Parse heartbeat response from Arduino"""
        # Handle different packet types
        if len(data) == 1:
            # Single byte - might be partial packet or identification
            logger.debug(f"Ignoring single byte packet: 0x{data[0]:02X}")
            return None
            
        elif len(data) == 3 and data[0] == 0xAA:
            # ACK/NACK packet
            logger.debug("Ignoring ACK/NACK packet")
            return None
            
        elif len(data) >= 6 and data[0] == 0xFF:
            # Heartbeat response packet (6 bytes)
            response_id = data[0]
            node_id = data[1]
            is_running = bool(data[2])
            is_calibrated = bool(data[3])
            timestamp = (data[4] << 8) | data[5]  # Combine two bytes (big-endian)
            
            logger.info(f"Parsed heartbeat response: node_id={node_id}, running={is_running}, calibrated={is_calibrated}")
            
            return {
                'response_id': response_id,
                'node_id': node_id,
                'node_name': NODE_NAMES.get(node_id, "UNKNOWN"),
                'is_running': is_running,
                'is_calibrated': is_calibrated,
                'timestamp': timestamp,
                'timestamp_ms': timestamp * 10  # Convert to milliseconds
            }
            
        elif len(data) >= 20 and data[0] == 0xFE:
            # Node identification packet
            logger.debug("Ignoring node identification packet")
            return None
            
        else:
            # Unknown packet format
            logger.warning(f"Unknown packet format: length={len(data)}, first_byte=0x{data[0]:02X if len(data) > 0 else 0:02X}")
            return None
    
    def display_heartbeat_response(self, response: dict):
        """Display formatted heartbeat response"""
        print("\n" + "=" * 50)
        print("HEARTBEAT RESPONSE")
        print("=" * 50)
        
        # Status indicators
        running_status = "[YES]" if response['is_running'] else "[NO]"
        cal_status = "[YES]" if response['is_calibrated'] else "[NO]"
        
        print(f"\nNode Information:")
        print(f"  ID:           {response['node_id']} ({response['node_name']})")
        print(f"  Running:      {running_status}")
        print(f"  Calibrated:   {cal_status}")
        print(f"  Timestamp:    {response['timestamp']} ticks ({response['timestamp_ms']} ms)")
        
        print("\nRaw Data:")
        print(f"  Response ID:  0x{response['response_id']:02X}")
        
        print("=" * 50 + "\n")
    
    async def scan_and_connect_all_devices(self):
        """Scan for and connect to all HAR sensor nodes sequentially"""
        print("Scanning for HAR sensor nodes...")
        
        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]
        
        if not har_devices:
            print("No HAR devices found!")
            return
        
        print(f"Found {len(har_devices)} HAR device(s):")
        for device in har_devices:
            print(f" - {device.name} ({device.address})")
        print()
        
        # Connect to each device sequentially (BLE connections should not be done concurrently)
        successful_connections = 0
        for device in har_devices:
            try:
                result = await self.connect_single_device(device)
                if result:
                    successful_connections += 1
                # Add delay between connections to avoid BLE stack issues
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[FAILED] Failed to connect to {device.name}: {e}")
        
        print(f"\nConnected to {successful_connections}/{len(har_devices)} devices")
        print()
    
    def extract_node_id_from_name(self, device_name: str) -> Optional[int]:
        """Extract node ID from device name (e.g., 'HAR_WRIST' -> 0)"""
        if not device_name or not device_name.startswith("HAR_"):
            return None
        
        # Extract placement name from device name (e.g., "HAR_WRIST" -> "WRIST")
        parts = device_name.split("_")
        if len(parts) < 2:
            return None
        
        placement_name = parts[1].upper()
        
        # Map placement name to node ID
        name_to_id = {name: node_id for node_id, name in NODE_NAMES.items()}
        return name_to_id.get(placement_name)
    
    def extract_node_id_from_name_by_address(self, address: str) -> Optional[int]:
        """Extract node ID from device name by address"""
        # Use the stored mapping
        if hasattr(self, 'address_to_node_id'):
            return self.address_to_node_id.get(address)
        return None
    
    async def connect_single_device(self, device):
        """Connect to a single BLE device"""
        print(f"Connecting to {device.name}...")
        try:
            client = BleakClient(device.address)
            await client.connect()
            
            if client.is_connected:
                print(f"Connected to {device.name}")
                
                self.connected_clients[device.address] = client
                
                # Extract node ID from device name and store mapping
                node_id = self.extract_node_id_from_name(device.name)
                if node_id is not None:
                    self.node_statuses[node_id] = NodeStatus(
                        node_id=NodePlacement(node_id),
                        is_running=False,
                        is_calibrated=False,
                        timestamp=0,
                        connected=True
                    )
                    # Store address -> node_id mapping for later use
                    if not hasattr(self, 'address_to_node_id'):
                        self.address_to_node_id = {}
                    self.address_to_node_id[device.address] = node_id
                
                # Give Arduino time to initialize
                await asyncio.sleep(0.5)
                return True
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            raise e
    
    async def start_device_notifications(self, address: str, client: BleakClient, device_buffers: Dict[str, Dict[str, Any]]):
        """
        Start notification handlers for a single device with buffered async queue processing.
        This prevents data loss by using a large buffer to handle BLE burst traffic.
        """
        try:
            # Create notification handler for this device that puts data into queue
            def notification_handler(sender, data):
                # Fast notification handler - just queue the raw data
                # This minimizes processing time in the BLE callback
                try:
                    # Record packet timing
                    current_time = time.time()
                    if self.first_packet_time.get(address) is None:
                        self.first_packet_time[address] = current_time
                        node_id_inner = self.extract_node_id_from_name_by_address(address)
                        node_name_inner = NODE_NAMES.get(node_id_inner, "UNKNOWN") if node_id_inner is not None else "UNKNOWN"
                        logger.info(f"First packet received from {node_name_inner} at {current_time:.3f}s")
                    self.last_packet_time[address] = current_time
                    
                    # Try to put data in queue without blocking
                    if not self.packet_queues[address].full():
                        self.packet_queues[address].put_nowait((address, bytes(data)))
                        self.packets_received[address] = self.packets_received.get(address, 0) + 1
                    else:
                        # Queue is full - log dropped packet
                        self.packets_dropped[address] = self.packets_dropped.get(address, 0) + 1
                        logger.error(f"BUFFER OVERFLOW: Dropped packet from {address} (queue full)")
                except Exception as e:
                    logger.error(f"Error queuing packet from {address}: {e}")
            
            # Subscribe to orientation characteristic only
            await client.start_notify(ORIENTATION_CHAR_UUID, notification_handler)
            
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            logger.info(f"Started notifications for {node_name} ({address})")
            
            # Keep the handler running (this will run indefinitely until cancelled)
            while True:
                await asyncio.sleep(1.0)
                
                # Log buffer status every 10 seconds
                if int(time.time()) % 10 == 0:
                    queue_size = self.packet_queues[address].qsize()
                    if queue_size > self.PACKET_QUEUE_MAX_SIZE * 0.7:
                        logger.warning(f"{node_name}: Queue at {queue_size}/{self.PACKET_QUEUE_MAX_SIZE} (70%+ full)")
                
        except asyncio.CancelledError:
            # Expected when stopping collection
            pass
        except Exception as e:
            print(f"Error in notification handler for {address}: {e}")
            logger.error(f"Error in notification handler for {address}: {e}")
    
    async def process_packet_queue(self, address: str, device_buffers: Dict[str, Dict[str, Any]]):
        """
        Asynchronously process queued BLE packets for a single device.
        This runs in parallel with BLE notifications to prevent data loss.
        """
        node_id = self.extract_node_id_from_name_by_address(address)
        node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
        
        processed_count = 0
        logger.info(f"Started packet processor for {node_name} ({address})")
        
        try:
            while True:
                try:
                    # Get packet from queue (wait up to 0.1 seconds)
                    addr, data = await asyncio.wait_for(
                        self.packet_queues[address].get(), 
                        timeout=0.1
                    )
                    
                    # Process the packet
                    self.handle_orientation_data(addr, data, device_buffers)
                    processed_count += 1
                    
                    # Mark task as done
                    self.packet_queues[address].task_done()
                    
                    # Log progress every 1000 packets
                    if processed_count % 1000 == 0:
                        queue_size = self.packet_queues[address].qsize()
                        logger.info(f"{node_name}: Processed {processed_count} packets, {queue_size} queued")
                    
                except asyncio.TimeoutError:
                    # No packet available - this is normal, continue waiting
                    continue
                    
                except Exception as e:
                    logger.error(f"Error processing packet from {address}: {e}")
                    continue
                    
        except asyncio.CancelledError:
            # Expected when stopping collection
            queue_size = self.packet_queues[address].qsize()
            logger.info(f"Stopping packet processor for {node_name}: {processed_count} processed, {queue_size} remaining")
            
            # Process remaining packets in queue before exiting
            logger.info(f"Draining remaining {queue_size} packets from {node_name}...")
            drained = 0
            while not self.packet_queues[address].empty():
                try:
                    addr, data = self.packet_queues[address].get_nowait()
                    self.handle_orientation_data(addr, data, device_buffers)
                    self.packet_queues[address].task_done()
                    drained += 1
                except Exception as e:
                    logger.error(f"Error draining packet: {e}")
                    break
            
            logger.info(f"Drained {drained} packets from {node_name}")
            
        except Exception as e:
            logger.error(f"Fatal error in packet processor for {address}: {e}")
    
    async def send_time_sync_command(self, client: BleakClient, address: str, sync_timestamp_ms: int, max_retries: int = 3):
        """Send time synchronization command to a device"""
        for attempt in range(max_retries):
            try:
                # Build time sync command: CMD_TIME_SYNC (1 byte) + timestamp (4 bytes, little-endian)
                command = bytearray([CMD_TIME_SYNC])
                command.extend(sync_timestamp_ms.to_bytes(4, byteorder='little'))
                
                await asyncio.wait_for(
                    client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True),
                    timeout=5.0
                )
                logger.info(f"Time synced on {address} to {sync_timestamp_ms}ms")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout syncing time on {address} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error syncing time on {address} (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)  # Wait before retry
        
        logger.error(f"Failed to sync time on {address} after {max_retries} attempts")
        return False
    
    async def send_start_command(self, client: BleakClient, address: str, max_retries: int = 3):
        """Send start command to a device with retry logic"""
        for attempt in range(max_retries):
            try:
                command = bytes([CMD_START])
                await asyncio.wait_for(
                    client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True),
                    timeout=5.0
                )
                logger.info(f"Started collection on {address}")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout starting {address} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error starting {address} (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0)  # Wait before retry
        
        logger.error(f"Failed to start collection on {address} after {max_retries} attempts")
        return False
    
    async def send_stop_command(self, client: BleakClient, address: str, max_retries: int = 3):
        """Send stop command to a device with retry logic
        
        Note: The Arduino firmware will flush its BLE queue before fully stopping,
        which can take several seconds. We give it time to complete this flush.
        """
        # Check if client is still connected
        if not client.is_connected:
            logger.warning(f"Device {address} already disconnected, skipping stop command")
            return False
            
        for attempt in range(max_retries):
            try:
                command = bytes([CMD_STOP])
                await asyncio.wait_for(
                    client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True),
                    timeout=5.0
                )
                logger.info(f"Stopped collection on {address}")
                
                # Wait for Arduino to flush its BLE queue (up to 5 seconds)
                # The Arduino will transmit all queued packets before fully stopping
                logger.info(f"Waiting for {address} to flush BLE queue...")
                await asyncio.sleep(6.0)  # Give 6 seconds for 5 second flush timeout + buffer
                logger.info(f"Queue flush complete for {address}")
                
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout stopping {address} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error stopping {address} (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0)  # Wait before retry
        
        logger.error(f"Failed to stop collection on {address} after {max_retries} attempts")
        return False
    
    async def finalize_session_data(self, device_buffers: Dict[str, Dict[str, Any]], session_data: Dict[str, Any]):
        """
        Finalize and organize session data.
        UPDATED to create a single, time-ordered list of all windows across all devices.
        ALSO sorts windows chronologically within each device's window list.
        """
        logger.info("Finalizing session data...")
        
        # Create a single list to hold all windows from all devices
        all_windows_chronological = []
        
        total_windows = 0
        total_samples = 0
        total_short_windows = 0
        total_long_windows = 0
        
        # Aggregate windows from all device buffers
        for address, buffer in device_buffers.items():
            device_windows = buffer.get('windows', [])
            
            # CRITICAL: Sort windows for this device chronologically by window start time
            # This ensures device_windows[address] contains windows in time order
            device_windows.sort(key=lambda w: w.get('window_start_ms', float('inf')))
            
            # Add device info to each window
            for window_data in device_windows:
                window_data['device_address'] = address
                
                # Count samples and window types
                total_samples += len(window_data.get('samples', []))
                if window_data['window_type'] == 'short':
                    total_short_windows += 1
                else:
                    total_long_windows += 1
                
                # Add to master list
                all_windows_chronological.append(window_data)
            
            total_windows += len(device_windows)
            
            # Store sorted per-device structure for backward compatibility
            session_data['device_windows'][address] = device_windows
        
        logger.info(f"Aggregated {len(all_windows_chronological)} windows from {len(device_buffers)} devices.")
        
        # CRITICAL STEP: Sort the master list by window start timestamp
        all_windows_chronological.sort(key=lambda w: w.get('window_start_ms', float('inf')))
        
        logger.info("All windows sorted chronologically by absolute timestamp.")
        
        # Add the sorted list to session data (new unified timeline structure)
        session_data['sorted_windows'] = all_windows_chronological
        
        # Update session metadata
        session_data['session_metadata']['total_windows'] = total_windows
        session_data['session_metadata']['total_samples'] = total_samples
        session_data['session_metadata']['short_windows'] = total_short_windows
        session_data['session_metadata']['long_windows'] = total_long_windows
        
        logger.info(f"Session finalized: {total_windows} windows, {total_samples} samples (all chronologically sorted)")
    
    async def monitor_connections(self, device_buffers: Dict[str, Dict[str, Any]], 
                                 required_device_count: int):
        """
        Monitor BLE connection health and perform periodic time re-sync during recording.
        CRITICAL: Raises ConnectionError if any device disconnects, stopping the entire session.
        """
        try:
            sync_counter = 0
            while True:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                sync_counter += 1
                
                disconnected_devices = []
                for address, client in self.connected_clients.items():
                    try:
                        # Simple connection check
                        if not client.is_connected:
                            logger.error(f"Device {address} has disconnected!")
                            disconnected_devices.append(address)
                    except Exception as e:
                        logger.error(f"Connection check failed for {address}: {e}")
                        disconnected_devices.append(address)
                
                # CRITICAL: If ANY device disconnects, abort the entire session
                if disconnected_devices:
                    node_names = []
                    for address in disconnected_devices:
                        node_id = self.extract_node_id_from_name_by_address(address)
                        node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
                        node_names.append(f"{node_name} ({address})")
                        device_buffers[address]['stats']['connection_dropped'] = True
                    
                    error_msg = (f"CONNECTION LOST: {len(disconnected_devices)} device(s) disconnected: "
                               f"{', '.join(node_names)}. Session requires {required_device_count} "
                               f"devices to remain connected. Aborting session.")
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                # Periodic time re-sync every 60 seconds to handle clock drift
                if sync_counter >= 6:  # 6 * 10s = 60 seconds
                    import time
                    # Calculate elapsed time since session start (in ms)
                    current_time = datetime.now()
                    if 'session_absolute_start_ms' in list(device_buffers.values())[0]:
                        session_start_ms = list(device_buffers.values())[0]['session_absolute_start_ms']
                        current_ms = int(current_time.timestamp() * 1000)
                        sync_timestamp_ms = current_ms - session_start_ms  # Relative time
                        logger.info(f"Performing periodic time re-sync: {sync_timestamp_ms}ms (relative)")
                        
                        # Re-sync all connected devices
                        for address, client in self.connected_clients.items():
                            if client.is_connected:
                                try:
                                    await self.send_time_sync_command(client, address, sync_timestamp_ms)
                                except Exception as e:
                                    logger.error(f"Failed to re-sync time on {address}: {e}")
                    
                    sync_counter = 0  # Reset counter
                
        except asyncio.CancelledError:
            # Expected when stopping monitoring
            pass
        except Exception as e:
            logger.error(f"Error in connection monitoring: {e}")
    
    async def request_all_statuses(self):
        """Request status from all connected devices concurrently"""
        if not self.connected_clients:
            print("No devices connected")
            return
        
        print(f"Requesting status from {len(self.connected_clients)} connected device(s)...")
        
        # Request status from all devices concurrently
        status_tasks = []
        for address, client in self.connected_clients.items():
            task = self.request_single_status(address, client)
            status_tasks.append(task)
        
        # Wait for all status requests to complete
        results = await asyncio.gather(*status_tasks, return_exceptions=True)
        
        successful_requests = sum(1 for result in results if not isinstance(result, Exception))
        print(f"[OK] Status received from {successful_requests}/{len(self.connected_clients)} devices")
    
    async def request_single_status(self, address: str, client: BleakClient):
        """Request status from a single device"""
        try:
            success = await self.send_heartbeat_request(client)
            if success:
                print(f"[OK] Status received from {address}")
            else:
                print(f"[FAILED] Failed to get status from {address}")
        except Exception as e:
            print(f"[ERROR] Error requesting status from {address}: {e}")
    
    async def scan_for_device(self):
        """Scan for HAR sensor nodes"""
        print("Scanning for HAR sensor nodes...")
        
        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]
        
        if not har_devices:
            print("No HAR devices found!")
            return None
        
        print(f"Found {len(har_devices)} HAR device(s):\n")
        for i, device in enumerate(har_devices, 1):
            print(f"  [{i}] {device.name}")
            print(f"      Address: {device.address}")
            if hasattr(device, 'rssi') and device.rssi is not None:
                print(f"      RSSI: {device.rssi} dBm")
            print()
        
        if len(har_devices) == 1:
            return har_devices[0]
        
        # Let user choose
        while True:
            try:
                choice = int(input(f"Select device [1-{len(har_devices)}]: "))
                if 1 <= choice <= len(har_devices):
                    return har_devices[choice - 1]
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid selection")
                return None
    
    async def send_heartbeat_request(self, client: BleakClient):
        """Send heartbeat request and wait for response"""
        print("Sending heartbeat request...")
        
        # Prepare command
        command = bytes([CMD_HEARTBEAT])
        
        # Set up notification handler
        response_received = asyncio.Event()
        heartbeat_response = None
        
        def notification_handler(sender, data):
            """Handle incoming notification"""
            nonlocal heartbeat_response
            # Log raw data for debugging
            logger.debug(f"Received notification: {data.hex()}")
            
            parsed = self.parse_heartbeat_response(data)
            if parsed:  # Only count valid heartbeat responses
                heartbeat_response = parsed
                response_received.set()
        
        try:
            # Subscribe to notifications BEFORE sending the command
            await client.start_notify(CONTROL_CHAR_UUID, notification_handler)
            
            # Small delay to ensure notification handler is ready
            await asyncio.sleep(0.1)
            
            # Send heartbeat command
            await client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True)
            print("Heartbeat request sent")
            
            # Wait for response (increased timeout to 5 seconds for BLE queue processing)
            try:
                await asyncio.wait_for(response_received.wait(), timeout=5.0)
                
                if heartbeat_response:
                    self.display_heartbeat_response(heartbeat_response)
                    
                    # Update node status
                    node_id = heartbeat_response['node_id']
                    self.node_statuses[node_id] = NodeStatus(
                        node_id=NodePlacement(node_id),
                        is_running=heartbeat_response['is_running'],
                        is_calibrated=heartbeat_response['is_calibrated'],
                        timestamp=heartbeat_response['timestamp'],
                        connected=True
                    )
                    
                    return True
                else:
                    print("No valid heartbeat response received")
                    return False
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                logger.warning("Heartbeat timeout - no response received within 5 seconds")
                return False
                
        finally:
            # Unsubscribe from notifications
            try:
                await client.stop_notify(CONTROL_CHAR_UUID)
            except Exception as e:
                logger.error(f"Error unsubscribing from notifications: {e}")
    
    async def run(self):
        """Main menu loop"""
        print("Initializing HAR Data Collection System...")
        print("=" * 70)
        
        # Scan and connect to all available devices
        await self.scan_and_connect_all_devices()
        
        # Request status from all connected devices
        if self.connected_clients:
            print("Requesting status from connected devices...")
            await self.request_all_statuses()
            print()
        
        # Main menu loop
        while self.running:
            self.display_menu()
            
            try:
                choice = input("Select option: ").strip()
                
                if choice == '1':
                    # Start recording session
                    await self.handle_start_recording()
                    
                elif choice == '2':
                    # Calibrate sensor
                    await self.handle_calibration()
                    
                elif choice == '3':
                    # Request heartbeat/status
                    await self.handle_heartbeat()
                    
                elif choice == '4':
                    # View configuration
                    self.show_current_config()
                    
                elif choice == '5':
                    # Reset node
                    await self.handle_reset()

                elif choice == '6':
                    await self.handle_live_classification()
                    
                elif choice == '0':
                    # Exit
                    self.running = False
                    print("\nExiting...")
                    
                else:
                    print("Invalid option")
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(2)
        
        # Cleanup: disconnect all clients
        await self.disconnect_all_clients()
    
    async def disconnect_all_clients(self):
        """Disconnect from all connected BLE clients and close MongoDB connection"""
        if self.connected_clients:
            print(f"\nDisconnecting from {len(self.connected_clients)} device(s)...")
            for address, client in self.connected_clients.items():
                try:
                    await client.disconnect()
                    print(f"[OK] Disconnected from {address}")
                except Exception as e:
                    print(f"[ERROR] Error disconnecting from {address}: {e}")
            self.connected_clients.clear()
            self.node_statuses.clear()
            print("All connections closed")
        
        # Clear packet queues and processing tasks
        self.packet_queues.clear()
        self.processing_tasks.clear()
        self.packets_dropped.clear()
        self.packets_received.clear()
        
        # Close MongoDB connection
        close_mongodb()
    
    async def handle_start_recording(self):
        """Handle recording session start"""
        print("\n" + "=" * 70)
        print("START RECORDING SESSION")
        print("=" * 70)
        
        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        # Get configuration
        config = self.get_recording_config()
        if config is None:
            input("\nPress Enter to continue...")
            return
        
        self.recording_config = config
        
        print(f"\nStarting data collection for {len(self.connected_clients)} device(s)...")
        print(f"Exercise: {config.exercise_type.name.replace('_', ' ').title()}")
        print(f"Participant: {config.participant_id}")
        print(f"Target reps: {config.rep_count}")
        
        confirm = input(f"\nBegin recording? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Recording cancelled")
            input("\nPress Enter to continue...")
            return
        
        # Start the recording session
        await self.perform_recording_session(config)
        
        input("\nPress Enter to continue...")
    
    async def perform_recording_session(self, config: RecordingConfig):
        """Perform the actual recording session with concurrent multi-device support"""
        # Create data directory if it doesn't exist
        data_dir = "collected_data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate session filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = f"{data_dir}/session_{config.participant_id}_{config.exercise_type.name}_{timestamp}.json"
        
        print(f"\nRecording to: {session_filename}")
        
        # Track session start time for statistics and global timestamp synchronization
        session_start_time = datetime.now()
        session_start_utc_ms = int(session_start_time.timestamp() * 1000)
        
        # Initialize data collection with per-device buffers
        device_buffers: Dict[str, Dict[str, Any]] = {}  # device_address -> buffer
        session_data = {
            'session_metadata': {
                'participant_id': config.participant_id,
                'exercise_type': config.exercise_type.name,
                'exercise_id': config.exercise_type.value,
                'target_reps': config.rep_count,
                'session_notes': config.session_notes,
                'start_time': timestamp,
                'start_time_utc_ms': session_start_utc_ms,  # Absolute UTC timestamp
                'start_time_iso': session_start_time.isoformat(),
                'devices': {}
            },
            'device_windows': {}  # device_address -> list of windows
        }
        
        # Initialize buffers for each device
        for address in self.connected_clients.keys():
            device_buffers[address] = {
                'raw_samples': [],  # All raw quaternion samples
                'short_window_buffer': [],  # Circular buffer for short window (150 samples)
                'long_window_buffer': [],  # Circular buffer for long window (300 samples)
                'samples_since_last_short': 0,
                'samples_since_last_long': 0,
                'short_window_id': 0,
                'long_window_id': 0,
                'windows': [],  # Extracted windows with features
                'stats': {'total_packets': 0, 'total_samples': 0, 'short_windows': 0, 'long_windows': 0},
                'sync_timestamp_ms': None,  # RPI reference timestamp (ms)
                'first_packet_timestamp': None  # First packet timestamp for offset calculation
            }
            session_data['device_windows'][address] = []
            node_id = self.extract_node_id_from_name_by_address(address)
            session_data['session_metadata']['devices'][address] = {
                'node_id': node_id,
                'node_name': NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            }
        
        # Task list for cleanup
        notification_tasks = []
        
        # Initialize packet queues and processing tasks for buffered data handling
        print("Initializing high-capacity packet buffers...")
        
        # Track first packet time for each device
        self.first_packet_time: Dict[str, float] = {}
        self.last_packet_time: Dict[str, float] = {}
        
        for address in self.connected_clients.keys():
            # Create large async queue for each device (10000 packets = ~100 seconds at 100Hz)
            self.packet_queues[address] = asyncio.Queue(maxsize=self.PACKET_QUEUE_MAX_SIZE)
            self.packets_dropped[address] = 0
            self.packets_received[address] = 0
            self.first_packet_time[address] = None
            self.last_packet_time[address] = None
            
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            logger.info(f"Initialized {self.PACKET_QUEUE_MAX_SIZE}-packet buffer for {node_name}")
        
        try:
            # Start concurrent notification handlers and connection monitoring
            print("Starting concurrent data collection...")
            
            # Start packet processing tasks (one per device)
            for address, client in self.connected_clients.items():
                # Start async packet processor for this device
                processing_task = asyncio.create_task(
                    self.process_packet_queue(address, device_buffers)
                )
                self.processing_tasks.append(processing_task)
                notification_tasks.append(processing_task)
            
            # Start BLE notification handlers
            for address, client in self.connected_clients.items():
                task = asyncio.create_task(
                    self.start_device_notifications(address, client, device_buffers)
                )
                notification_tasks.append(task)
            
            # Add connection monitoring task (must maintain all devices)
            required_device_count = len(self.connected_clients)
            monitor_task = asyncio.create_task(
                self.monitor_connections(device_buffers, required_device_count)
            )
            notification_tasks.append(monitor_task)
            
            # STEP 1: Synchronize timestamps across all devices
            print("\n" + "=" * 70)
            print("SYNCHRONIZING DEVICE TIMESTAMPS")
            print("=" * 70)
            
            # Use relative timestamp (milliseconds from session start = 0)
            # This avoids 32-bit integer overflow issues with Unix epoch timestamps
            sync_timestamp_ms = 0  # Session start is t=0
            session_absolute_start_ms = int(session_start_time.timestamp() * 1000)
            
            print(f"Session start time: {session_start_time.isoformat()}")
            print(f"Reference timestamp: {sync_timestamp_ms} ms (relative to session start)")
            print(f"Syncing {len(self.connected_clients)} device(s)...\n")
            
            # Send time sync command to all devices
            sync_tasks = []
            for address, client in self.connected_clients.items():
                task = self.send_time_sync_command(client, address, sync_timestamp_ms)
                sync_tasks.append(task)
            
            sync_results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            successful_syncs = sum(1 for result in sync_results if result is True)
            
            if successful_syncs == 0:
                logger.error("Failed to sync time on any device - aborting session")
                print("[FAILED] Time synchronization failed on all devices")
                return
            
            if successful_syncs < required_device_count:
                logger.error(f"Time sync failed on {required_device_count - successful_syncs} device(s) - aborting session")
                print(f"[FAILED] Time synchronization failed on {required_device_count - successful_syncs} device(s)")
                print(f"All {required_device_count} devices must be synchronized to start recording")
                return
            
            print(f"[OK] Time synchronized on all {successful_syncs} devices")
            
            # Small delay to ensure sync is processed
            await asyncio.sleep(0.5)
            
            # Store sync timestamp in device buffers for timestamp calculation
            for address in device_buffers.keys():
                device_buffers[address]['sync_timestamp_ms'] = sync_timestamp_ms
                device_buffers[address]['session_absolute_start_ms'] = session_absolute_start_ms
                device_buffers[address]['session_start_utc_ms'] = session_start_utc_ms  # For absolute timestamps
            
            # STEP 2: Start data collection on all devices
            print("\n" + "=" * 70)
            print("STARTING DATA COLLECTION")
            print("=" * 70)
            
            # Send start commands to all devices concurrently
            start_tasks = []
            for address, client in self.connected_clients.items():
                task = self.send_start_command(client, address)
                start_tasks.append(task)
            
            start_results = await asyncio.gather(*start_tasks, return_exceptions=True)
            successful_starts = sum(1 for result in start_results if result is True)
            
            if successful_starts == 0:
                logger.error("Failed to start collection on any device")
                print("[FAILED] Failed to start collection on any device")
                return
            
            if successful_starts < required_device_count:
                logger.error(f"Failed to start collection on {required_device_count - successful_starts} device(s) - aborting session")
                print(f"[FAILED] Failed to start collection on {required_device_count - successful_starts} device(s)")
                print(f"All {required_device_count} devices must start successfully to begin recording")
                return
            
            print(f"[OK] Started collection on all {successful_starts} devices")
            
            # Add stabilization delay to ensure all devices have started and are synchronized
            print("[INFO] Waiting 2 seconds for all devices to stabilize...")
            await asyncio.sleep(2.0)
            
            print(f"[CRITICAL] Recording will immediately STOP if ANY device loses connection")
            
            # Collect data until user presses Enter, connection is lost, or Ctrl+C
            print("\nCollecting data... (Press Enter to stop)")
            print(f"[MONITORING] {required_device_count} device(s) must remain connected")
            
            # Run input monitoring in executor (run_in_executor returns a Future, not a coroutine)
            input_task = asyncio.get_event_loop().run_in_executor(None, input, "")
            
            # Wait for either user input OR connection loss (via monitor_task exception)
            # Create a list of tasks to wait for
            wait_tasks = [input_task, monitor_task]
            
            try:
                # Wait for the first task to complete (either user input or connection monitor)
                done, pending = await asyncio.wait(
                    wait_tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Check if monitor task completed (meaning connection was lost)
                if monitor_task in done:
                    # Get the exception from the monitor task
                    try:
                        await monitor_task
                    except ConnectionError as conn_err:
                        # Connection lost - stop recording immediately
                        print("\n" + "=" * 70)
                        print("!!! CONNECTION LOST - STOPPING RECORDING !!!")
                        print("=" * 70)
                        print(f"\n{conn_err}\n")
                        logger.error(f"Recording stopped due to connection loss: {conn_err}")
                        
                        # Mark session as failed due to connection loss
                        session_data['session_metadata']['status'] = 'FAILED'
                        session_data['session_metadata']['failure_reason'] = 'device_disconnection'
                        session_data['session_metadata']['error_message'] = str(conn_err)
                    except Exception as e:
                        print(f"\n!!! Unexpected error in connection monitoring: {e}")
                        raise
                else:
                    # User pressed Enter - normal stop
                    print("\nRecording stopped by user")
                    session_data['session_metadata']['status'] = 'SUCCESS'
                    session_data['session_metadata']['stopped_by'] = 'user'
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    
            except Exception as e:
                print(f"\nError during data collection: {e}")
                session_data['session_metadata']['status'] = 'FAILED'
                session_data['session_metadata']['error_message'] = str(e)
                raise
            
        except asyncio.CancelledError:
            print("\nRecording cancelled")
            session_data['session_metadata']['status'] = 'CANCELLED'
        except KeyboardInterrupt:
            print("\nRecording interrupted")
            session_data['session_metadata']['status'] = 'INTERRUPTED'
        except ConnectionError as conn_err:
            # Already handled above, but catch here in case it propagates
            print(f"\nRecording failed due to connection loss")
            session_data['session_metadata']['status'] = 'FAILED'
            session_data['session_metadata']['failure_reason'] = 'device_disconnection'
        except Exception as e:
            print(f"Error during recording: {e}")
            session_data['session_metadata']['status'] = 'FAILED'
            session_data['session_metadata']['error_message'] = str(e)
            import traceback
            traceback.print_exc()
        finally:
            # Cancel all notification tasks
            for task in notification_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Send stop command to all devices sequentially to allow proper queue flushing
            # Note: Each device will flush its BLE queue before stopping, which takes ~5-6 seconds
            print("\nStopping data collection...")
            print("Note: Devices will flush their BLE queues to prevent data loss.")
            print("This may take up to 6 seconds per device...\n")
            
            stop_tasks = []
            for address, client in self.connected_clients.items():
                node_id = self.extract_node_id_from_name_by_address(address)
                node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
                print(f"Stopping {node_name}...")
                task = self.send_stop_command(client, address)
                stop_tasks.append(task)
            
            stop_results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            successful_stops = sum(1 for result in stop_results if result is True)
            print(f"\n[OK] Stopped collection on {successful_stops}/{len(self.connected_clients)} devices")
            
            # Calculate session duration
            session_end_time = datetime.now()
            session_duration = (session_end_time - session_start_time).total_seconds()
            
            # Process and finalize collected data
            await self.finalize_session_data(device_buffers, session_data)
            
            # Display collection statistics
            self.display_collection_statistics_multi_device(device_buffers, session_duration)
            
            # Display buffer statistics (pass session_duration for rate calculation)
            self.display_buffer_statistics(session_duration)
            
            # Save collected data
            session_data['session_metadata']['end_time'] = session_end_time.strftime("%Y%m%d_%H%M%S")
            session_data['session_metadata']['duration_seconds'] = session_duration
            session_data['session_metadata']['timing_info'] = {
                'sync_enabled': True,
                'sync_mode': 'relative',
                'sync_timestamp_ms': sync_timestamp_ms,
                'session_absolute_start_ms': session_absolute_start_ms,
                'session_start_datetime': session_start_time.isoformat(),
                'notes': 'All device timestamps synchronized using relative time (t=0 at session start) with periodic re-sync every 60 seconds. To get absolute time: session_absolute_start_ms + timestamp_ms'
            }
            
            # Add buffer statistics to session metadata
            session_data['session_metadata']['buffer_info'] = {
                'queue_max_size': self.PACKET_QUEUE_MAX_SIZE,
                'buffering_enabled': True,
                'per_device_stats': {}
            }
            
            for address in self.connected_clients.keys():
                node_id = self.extract_node_id_from_name_by_address(address)
                node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
                received = self.packets_received.get(address, 0)
                dropped = self.packets_dropped.get(address, 0)
                drop_rate = (dropped / received * 100) if received > 0 else 0.0
                
                session_data['session_metadata']['buffer_info']['per_device_stats'][address] = {
                    'node_name': node_name,
                    'packets_received': received,
                    'packets_dropped': dropped,
                    'drop_rate_percent': round(drop_rate, 2),
                    'data_loss': dropped > 0
                }
            
            # Display session status
            session_status = session_data['session_metadata'].get('status', 'UNKNOWN')
            print("\n" + "=" * 70)
            if session_status == 'SUCCESS':
                print("SESSION COMPLETED SUCCESSFULLY")
                print("=" * 70)
            elif session_status == 'FAILED':
                print("SESSION FAILED")
                print("=" * 70)
                failure_reason = session_data['session_metadata'].get('failure_reason', 'unknown')
                error_msg = session_data['session_metadata'].get('error_message', 'No details available')
                print(f"\nReason: {failure_reason}")
                print(f"Details: {error_msg}")
            else:
                print(f"SESSION STATUS: {session_status}")
                print("=" * 70)
            
            # Check for connection issues
            connection_issues = []
            for address, buffer in device_buffers.items():
                if buffer['stats'].get('connection_dropped', False):
                    node_id = self.extract_node_id_from_name_by_address(address)
                    node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
                    connection_issues.append(f"{node_name} ({address})")
            
            if connection_issues:
                print(f"\n[WARNING] Devices that lost connection: {', '.join(connection_issues)}")
            
            # Save to JSON file
            with open(session_filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            if session_status == 'SUCCESS':
                print(f"\n[SUCCESS] Session complete! Data saved to {session_filename}")
            else:
                print(f"\n[INFO] Partial session data saved to {session_filename}")
            
            # Save to MongoDB (only save successful sessions)
            if self.mongodb_enabled:
                # Only upload successful sessions to MongoDB
                if session_status == 'SUCCESS':
                    print("Saving session to MongoDB...")
                    if save_session_to_mongodb(session_data):
                        print("[SUCCESS] Session data saved to MongoDB")
                    else:
                        print("[WARNING] Failed to save session to MongoDB (data still saved to JSON file)")
                else:
                    print(f"[INFO] Skipping MongoDB upload - session status is '{session_status}'")
                    print("[INFO] Failed/incomplete sessions are only saved locally as JSON files")
                    if session_status == 'FAILED':
                        failure_reason = session_data['session_metadata'].get('failure_reason', 'unknown')
                        if failure_reason == 'device_disconnection':
                            print("[INFO] Session failed due to device disconnection - data incomplete and not uploaded")
            else:
                print("[INFO] MongoDB not connected - data only saved to JSON file")
            
            # Unsubscribe from notifications
            for address, client in self.connected_clients.items():
                try:
                    await client.stop_notify(ORIENTATION_CHAR_UUID)
                    print(f"[OK] Unsubscribed from {address}")
                except Exception as e:
                    print(f"[ERROR] Error unsubscribing from {address}: {e}")
    
    def display_buffer_statistics(self, session_duration: float = None):
        """Display BLE buffer and packet statistics"""
        print(f"\n{'='*70}")
        print("BLE BUFFER STATISTICS")
        print(f"{'='*70}")
        
        total_received = 0
        total_dropped = 0
        device_stats = []
        
        for address in self.connected_clients.keys():
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            
            received = self.packets_received.get(address, 0)
            dropped = self.packets_dropped.get(address, 0)
            
            total_received += received
            total_dropped += dropped
            
            drop_rate = (dropped / received * 100) if received > 0 else 0.0
            packet_rate = (received / session_duration) if session_duration and session_duration > 0 else 0.0
            
            # Calculate device-specific timing
            first_packet = self.first_packet_time.get(address)
            last_packet = self.last_packet_time.get(address)
            device_duration = (last_packet - first_packet) if (first_packet and last_packet) else None
            
            device_stats.append({
                'name': node_name,
                'address': address,
                'received': received,
                'dropped': dropped,
                'rate': packet_rate,
                'first_packet_time': first_packet,
                'last_packet_time': last_packet,
                'duration': device_duration
            })
            
            print(f"Device: {node_name} ({address})")
            print(f"  Packets received: {received}")
            if session_duration:
                print(f"  Packet rate: {packet_rate:.1f} Hz (expected: ~100 Hz)")
            if device_duration:
                print(f"  Active duration: {device_duration:.2f}s")
            print(f"  Packets dropped:  {dropped} ({drop_rate:.2f}%)")
            
            if dropped > 0:
                print(f"  [WARNING] Data loss detected on {node_name}!")
            else:
                print(f"  [OK] No data loss")
            print()
        
        print(f"{'='*70}")
        print("OVERALL BUFFER STATISTICS")
        print(f"{'='*70}")
        print(f"Total packets received: {total_received}")
        print(f"Total packets dropped:  {total_dropped}")
        
        # Analyze packet count variance
        if len(device_stats) > 1:
            received_counts = [d['received'] for d in device_stats]
            min_received = min(received_counts)
            max_received = max(received_counts)
            variance = max_received - min_received
            variance_percent = (variance / min_received * 100) if min_received > 0 else 0.0
            
            print(f"\nPacket Count Variance:")
            print(f"  Min: {min_received} packets")
            print(f"  Max: {max_received} packets")
            print(f"  Variance: {variance} packets ({variance_percent:.1f}%)")
            
            if variance > max_received * 0.05:  # More than 5% variance
                print(f"  [WARNING] Significant variance detected between devices")
                print(f"  This may indicate timing issues or unequal start/stop times")
            else:
                print(f"  [OK] Packet counts are consistent across devices")
            
            # Analyze timing synchronization
            first_times = [d['first_packet_time'] for d in device_stats if d['first_packet_time']]
            last_times = [d['last_packet_time'] for d in device_stats if d['last_packet_time']]
            
            if len(first_times) > 1 and len(last_times) > 1:
                earliest_start = min(first_times)
                latest_start = max(first_times)
                start_spread = latest_start - earliest_start
                
                earliest_end = min(last_times)
                latest_end = max(last_times)
                end_spread = latest_end - earliest_end
                
                print(f"\nTiming Synchronization:")
                print(f"  Start time spread: {start_spread:.3f}s")
                print(f"  End time spread: {end_spread:.3f}s")
                
                if start_spread > 0.1:  # More than 100ms difference
                    print(f"  [WARNING] Devices started at different times (up to {start_spread:.3f}s apart)")
                    print(f"  This explains packet count differences")
                else:
                    print(f"  [OK] All devices started within {start_spread*1000:.0f}ms")
                
                if end_spread > 0.1:
                    print(f"  [WARNING] Devices stopped at different times (up to {end_spread:.3f}s apart)")
                else:
                    print(f"  [OK] All devices stopped within {end_spread*1000:.0f}ms")
        
        if total_dropped > 0:
            overall_drop_rate = (total_dropped / total_received * 100) if total_received > 0 else 0.0
            print(f"\nOverall drop rate: {overall_drop_rate:.2f}%")
            print(f"[WARNING] {total_dropped} packets were lost due to buffer overflow")
            print(f"Consider reducing sample rate or number of devices if this persists")
        else:
            print(f"\n[SUCCESS] No packet loss - all BLE data captured successfully")
        
        print(f"{'='*70}")
    
    def display_collection_statistics_multi_device(self, device_buffers: Dict[str, Dict[str, Any]], 
                                                   session_duration: float):
        """Display statistics about collected window data for multiple devices"""
        print(f"\n{'='*70}")
        print("COLLECTION STATISTICS - MULTI DEVICE")
        print(f"{'='*70}")
        
        total_devices = len(device_buffers)
        total_windows = 0
        total_short_windows = 0
        total_long_windows = 0
        total_samples = 0
        total_packets = 0
        
        for address, buffer in device_buffers.items():
            device_stats = buffer['stats']
            device_windows = buffer['windows']
            
            device_total = len(device_windows)
            device_short = device_stats['short_windows']
            device_long = device_stats['long_windows']
            device_samples = device_stats['total_samples']
            
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            
            print(f"Device: {node_name} ({address})")
            print(f"  Windows: {device_total} total ({device_short} short, {device_long} long)")
            print(f"  Samples: {device_samples}")
            print(f"  Packets: {device_stats['total_packets']} received")
            print()
            
            total_windows += device_total
            total_short_windows += device_short
            total_long_windows += device_long
            total_samples += device_samples
            total_packets += device_stats['total_packets']
        
        # Calculate actual data rate
        data_rate = total_packets / session_duration if session_duration > 0 else 0
        
        print(f"{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Devices: {total_devices}")
        print(f"Session duration: {session_duration:.1f} seconds")
        print(f"Total windows: {total_windows} ({total_short_windows} short, {total_long_windows} long)")
        print(f"Total samples: {total_samples}")
        print(f"Total packets: {total_packets}")
        print(f"Data rate: ~{data_rate:.1f} packets/sec")
        print(f"{'='*70}")
    

    
    def handle_orientation_data(self, address: str, data: bytes, device_buffers: Dict[str, Dict[str, Any]]):
        """
        Handle incoming raw quaternion data from BLE notifications.
        UPDATED to calculate absolute timestamps from session start.
        """
        try:
            # Update packet stats
            device_buffers[address]['stats']['total_packets'] += 1
            
            # Parse orientation packet (14-byte with 4-byte timestamp)
            packet = self.parse_orientation_packet(data)
            if not packet:
                logger.warning(f"Failed to parse orientation packet from {address}")
                return
            
            device_buffers[address]['stats']['total_samples'] += 1
            
            buffer = device_buffers[address]
            
            # === SIMPLIFIED TIMESTAMP HANDLING WITH 4-BYTE TIMESTAMPS ===
            # With 4-byte timestamps (milliseconds), we get ~49 days before wraparound
            # No need for complex wraparound detection in typical recording sessions
            
            # Stage 1: Wait for initial sync to complete
            if buffer.get('session_start_utc_ms') is None:
                logger.debug(f"Skipping packet from {address} - waiting for time sync")
                return
            
            # Stage 2: Calculate absolute timestamp
            # packet.timestamp is relative milliseconds from Arduino's sessionStartTime
            # absolute_timestamp = session_start_utc_ms + relative_timestamp
            session_start_utc_ms = buffer['session_start_utc_ms']
            absolute_timestamp_ms = session_start_utc_ms + packet.timestamp
            
            # Create QuaternionSample with both absolute and relative timestamps
            sample = QuaternionSample(
                absolute_timestamp_ms=absolute_timestamp_ms,  # Global UTC timestamp
                relative_timestamp_ms=packet.timestamp,       # Arduino's relative timestamp (for debugging)
                qw=packet.qw,
                qx=packet.qx,
                qy=packet.qy,
                qz=packet.qz
            )
            
            buffer = device_buffers[address]
            
            # Add to raw samples
            buffer['raw_samples'].append(sample)
            
            # Add to both window buffers
            buffer['short_window_buffer'].append(sample)
            buffer['long_window_buffer'].append(sample)
            
            # Maintain circular buffer sizes
            if len(buffer['short_window_buffer']) > self.SHORT_WINDOW_SIZE:
                buffer['short_window_buffer'].pop(0)
            if len(buffer['long_window_buffer']) > self.LONG_WINDOW_SIZE:
                buffer['long_window_buffer'].pop(0)
            
            buffer['samples_since_last_short'] += 1
            buffer['samples_since_last_long'] += 1
            
            # Check if we should extract short window features
            if (buffer['samples_since_last_short'] >= self.SHORT_WINDOW_STEP and
                len(buffer['short_window_buffer']) >= self.SHORT_WINDOW_SIZE):
                
                # CRITICAL: Sort window samples by absolute timestamp to ensure chronological order
                # BLE packet delivery is not guaranteed to be in-order
                sorted_window_buffer = sorted(buffer['short_window_buffer'], key=lambda s: s.absolute_timestamp_ms)
                
                # Extract features from short window (using sorted samples)
                features = extract_window_features(
                    sorted_window_buffer,
                    window_type=0,  # short
                    node_id=packet.node_id,
                    window_id=buffer['short_window_id']
                )
                
                # Calculate window timing information (using sorted samples with absolute timestamps)
                window_start_timestamp_ms = sorted_window_buffer[0].absolute_timestamp_ms
                window_end_timestamp_ms = sorted_window_buffer[-1].absolute_timestamp_ms
                window_duration_ms = window_end_timestamp_ms - window_start_timestamp_ms
                
                # Store window with features and samples (sorted by absolute timestamp)
                window_data = {
                    'window_type': 'short',
                    'window_id': buffer['short_window_id'],
                    'node_id': packet.node_id,
                    'timestamp': datetime.now().isoformat(),
                    'window_start_ms': window_start_timestamp_ms,  # Absolute UTC start time
                    'window_end_ms': window_end_timestamp_ms,      # Absolute UTC end time
                    'window_duration_ms': window_duration_ms,      # Actual duration
                    'features': {
                        'qw_mean': features.qw_mean,
                        'qx_mean': features.qx_mean,
                        'qy_mean': features.qy_mean,
                        'qz_mean': features.qz_mean,
                        'qw_std': features.qw_std,
                        'qx_std': features.qx_std,
                        'qy_std': features.qy_std,
                        'qz_std': features.qz_std,
                        'sma': features.sma,
                        'dominant_freq': features.dominant_freq,
                        'total_samples': features.total_samples
                    },
                    'samples': [{'absolute_timestamp_ms': s.absolute_timestamp_ms, 
                                'relative_timestamp_ms': s.relative_timestamp_ms,
                                'qw': s.qw, 'qx': s.qx, 'qy': s.qy, 'qz': s.qz} 
                               for s in sorted_window_buffer]
                }
                
                buffer['windows'].append(window_data)
                if self.classification_mode:
                    self.send_features_to_ml(
                    node_id=packet.node_id,
                    node_name=NODE_NAMES.get(packet.node_id, "UNKNOWN"),
                    window_type=window_data["window_type"], 
                    window_id=window_data["window_id"],
                    start_ms=window_data["window_start_ms"],
                    end_ms=window_data["window_end_ms"],
                    features=window_data["features"],
                    )
                buffer['short_window_id'] += 1
                buffer['samples_since_last_short'] = 0
                buffer['stats']['short_windows'] += 1
                
                # Log progress every 10 windows
                if buffer['short_window_id'] % 10 == 0:
                    logger.info(f"Device {address}: Extracted short window {buffer['short_window_id']}")
            
            # Check if we should extract long window features
            if (buffer['samples_since_last_long'] >= self.LONG_WINDOW_STEP and
                len(buffer['long_window_buffer']) >= self.LONG_WINDOW_SIZE):
                
                # CRITICAL: Sort window samples by absolute timestamp to ensure chronological order
                # BLE packet delivery is not guaranteed to be in-order
                sorted_window_buffer = sorted(buffer['long_window_buffer'], key=lambda s: s.absolute_timestamp_ms)
                
                # Extract features from long window (using sorted samples)
                features = extract_window_features(
                    sorted_window_buffer,
                    window_type=1,  # long
                    node_id=packet.node_id,
                    window_id=buffer['long_window_id']
                )
                
                # Calculate window timing information (using sorted samples with absolute timestamps)
                window_start_timestamp_ms = sorted_window_buffer[0].absolute_timestamp_ms
                window_end_timestamp_ms = sorted_window_buffer[-1].absolute_timestamp_ms
                window_duration_ms = window_end_timestamp_ms - window_start_timestamp_ms
                
                # Store window with features and samples (sorted by absolute timestamp)
                window_data = {
                    'window_type': 'long',
                    'window_id': buffer['long_window_id'],
                    'node_id': packet.node_id,
                    'timestamp': datetime.now().isoformat(),
                    'window_start_ms': window_start_timestamp_ms,  # Absolute UTC start time
                    'window_end_ms': window_end_timestamp_ms,      # Absolute UTC end time
                    'window_duration_ms': window_duration_ms,      # Actual duration
                    'features': {
                        'qw_mean': features.qw_mean,
                        'qx_mean': features.qx_mean,
                        'qy_mean': features.qy_mean,
                        'qz_mean': features.qz_mean,
                        'qw_std': features.qw_std,
                        'qx_std': features.qx_std,
                        'qy_std': features.qy_std,
                        'qz_std': features.qz_std,
                        'sma': features.sma,
                        'dominant_freq': features.dominant_freq,
                        'total_samples': features.total_samples
                    },
                    'samples': [{'absolute_timestamp_ms': s.absolute_timestamp_ms,
                                'relative_timestamp_ms': s.relative_timestamp_ms,
                                'qw': s.qw, 'qx': s.qx, 'qy': s.qy, 'qz': s.qz} 
                               for s in sorted_window_buffer]
                }
                
                buffer['windows'].append(window_data)
                if self.classification_mode:
                    self.send_features_to_ml(
                    node_id=packet.node_id,
                    node_name=NODE_NAMES.get(packet.node_id, "UNKNOWN"),
                    window_type=window_data["window_type"],
                    window_id=window_data["window_id"],
                    start_ms=window_data["window_start_ms"],
                    end_ms=window_data["window_end_ms"],
                    features=window_data["features"],
                    )
                buffer['long_window_id'] += 1
                buffer['samples_since_last_long'] = 0
                buffer['stats']['long_windows'] += 1
                
                # Log progress every 5 windows
                if buffer['long_window_id'] % 5 == 0:
                    logger.info(f"Device {address}: Extracted long window {buffer['long_window_id']}")
                    
        except Exception as e:
            logger.error(f"Error processing orientation data from {address}: {e}")
    
    async def handle_calibration(self):
        """Handle sensor calibration on all connected devices with proper positioning guidance"""
        print("\n" + "=" * 70)
        print("SENSOR CALIBRATION - MULTI-NODE SYSTEM")
        print("=" * 70)
        
        if not self.connected_clients:
            print("\nNo devices connected")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nConnected devices: {len(self.connected_clients)}")
        for address in self.connected_clients.keys():
            node_id = self.extract_node_id_from_name_by_address(address)
            node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
            print(f"  - {node_name} ({address})")
        
        print("\n" + "-" * 70)
        print("CALIBRATION PROCEDURE")
        print("-" * 70)
        print("""
This calibration establishes a reference orientation for all sensors.
All sensors must be oriented in the SAME direction during calibration.
SENSOR PLACEMENT & ORIENTATION:
  
  * WRIST sensor:  On dominant wrist, facing FORWARD (screen up)
  * BICEP sensor:  On dominant upper arm, facing FORWARD
  * CHEST sensor:  Center of chest, facing FORWARD
  * THIGH sensor:  On dominant thigh, facing FORWARD

IMPORTANT:
  * All sensors must face the SAME direction (forward)
  * Stand completely still during calibration
  * Maintain T-pose for entire calibration (~10 seconds)
  * Keep body upright and balanced
  * Look straight ahead
        """)
        
        print("-" * 70)
        
        # Get confirmation
        ready = input("\nAre you in the T-pose position and ready? [y/N]: ").strip().lower()
        if ready != 'y':
            print("Calibration cancelled")
            input("\nPress Enter to continue...")
            return
        
        # Countdown
        print("\nStarting calibration in:")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1.0)
        
        print("\nCALIBRATING - HOLD POSITION STILL!\n")
        
        # Send calibration command to all devices concurrently
        calibration_tasks = []
        for address, client in self.connected_clients.items():
            task = self.calibrate_single_device(address, client)
            calibration_tasks.append(task)
        
        # Wait for all calibrations to complete
        results = await asyncio.gather(*calibration_tasks, return_exceptions=True)
        
        # Count successes
        successful_calibrations = sum(1 for result in results if result is True)
        
        print("\n" + "=" * 70)
        if successful_calibrations == len(self.connected_clients):
            print("CALIBRATION SUCCESSFUL")
            print(f"  [OK] All {len(self.connected_clients)} devices calibrated successfully")
        elif successful_calibrations > 0:
            print("CALIBRATION PARTIALLY SUCCESSFUL")
            print(f"  [WARN] {successful_calibrations}/{len(self.connected_clients)} devices calibrated")
        else:
            print("CALIBRATION FAILED")
            print("  [FAILED] No devices were calibrated successfully")
        print("=" * 70)
        
        # Update node statuses
        print("\nUpdating device statuses...")
        await self.request_all_statuses()
        
        input("\nPress Enter to continue...")
    
    async def calibrate_single_device(self, address: str, client: BleakClient, timeout: float = 12.0):
        """Calibrate a single device with timeout"""
        node_id = self.extract_node_id_from_name_by_address(address)
        node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"
        
        try:
            # Send calibration command
            command = bytes([CMD_CALIBRATE])
            await asyncio.wait_for(
                client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True),
                timeout=5.0
            )
            
            print(f" {node_name}: Calibration started...")
            
            # Wait for calibration to complete (Arduino takes ~10 seconds)
            await asyncio.sleep(timeout)
            
            print(f"  [OK] {node_name}: Calibration complete")
            
            # Update node status
            if node_id is not None and node_id in self.node_statuses:
                self.node_statuses[node_id].is_calibrated = True
            
            return True
            
        except asyncio.TimeoutError:
            print(f"{node_name}: Calibration timeout")
            logger.error(f"Calibration timeout for {address}")
            return False
        except Exception as e:
            print(f"{node_name}: Calibration error - {e}")
            logger.error(f"Calibration error for {address}: {e}")
            return False
    
    async def handle_heartbeat(self):
        """Request and display node status from all connected devices"""
        print("\n" + "=" * 70)
        print("NODE STATUS REQUEST")
        print("=" * 70)
        
        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        print(f"Requesting status from {len(self.connected_clients)} connected device(s)...")
        print()
        
        # Request status from all connected devices
        await self.request_all_statuses()
        
        print("\nStatus update complete")
        input("Press Enter to continue...")
    

    
    async def handle_reset(self):
        """Reset all connected sensor nodes"""
        print("\n" + "=" * 70)
        print("RESET SENSOR NODES")
        print("=" * 70)
        print("\nThis will stop any active recording and clear calibration on all devices!")
        
        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        confirm = input(f"\nReset {len(self.connected_clients)} device(s)? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Reset cancelled")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nSending reset command to {len(self.connected_clients)} device(s)...")
        
        # Send reset command to all connected devices
        for address, client in self.connected_clients.items():
            try:
                command = bytes([CMD_RESET])
                await client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True)
                print(f"Reset command sent to {address}")
                
                # Give Arduino time to process
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error sending to {address}: {e}")
        
        print("\nReset commands sent to all devices")
        input("Press Enter to continue...")



# MAIN ENTRY POINT
async def main():
    """Main entry point"""
    menu = DataCollectionMenu()
    await menu.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
