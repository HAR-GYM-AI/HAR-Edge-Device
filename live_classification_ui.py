#!/usr/bin/env python3
"""
Live Classification UI - Integrated Data Collection and Real-time ML Inference

This script combines:
- Tkinter UI for displaying predictions
- BLE data collection from sensor nodes
- Real-time feature extraction and ML inference
- Live exercise classification and rep counting
"""

import asyncio
import json
import joblib
import math
import numpy as np
import pandas as pd
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from bleak import BleakClient, BleakScanner
import logging
import struct
from datetime import datetime
from enum import IntEnum

# Import feature extraction functions
from Data_Collection import (
    QuaternionSample, WindowFeatures, RawQuaternionPacket,
    calculate_mean, calculate_std, calculate_sma, calculate_dominant_frequency,
    extract_window_features, NodePlacement, NODE_NAMES
)
from post_process_data import extract_temporal_features, extract_enhanced_frequency_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load trained models
try:
    exercise_model = joblib.load('exercise_classifier_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    rep_detector_s1 = joblib.load('rep_detector_model_s1.joblib')
    rep_counter_s2 = joblib.load('rep_counter_model_s2.joblib')
    rep_features = joblib.load('rep_counter_features.joblib')
    logger.info("Successfully loaded all trained models")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    exit(1)

EXERCISES = ["Pull-up", "Squat", "Bench Press", "Bicep Curl"]
UPDATE_MS = 333  # ~3 Hz update rate
WINDOW_SIZE = 150  # 1.5 seconds at 100Hz for exercise classification

@dataclass
class LiveSensorBuffer:
    """Buffer for storing recent sensor samples for each device"""
    samples: List[QuaternionSample]
    max_size: int = WINDOW_SIZE

    def add_sample(self, sample: QuaternionSample):
        self.samples.append(sample)
        if len(self.samples) > self.max_size:
            self.samples.pop(0)

    def get_samples(self) -> List[QuaternionSample]:
        return self.samples.copy()

    def has_enough_data(self) -> bool:
        return len(self.samples) >= 50  # Minimum for feature extraction

class LiveClassificationUI:
    """Integrated UI with live BLE data collection and ML inference"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Exercise Classification")
        self.root.geometry("500x250")
        self.root.minsize(450, 220)

        # Data collection components
        self.connected_clients: Dict[str, BleakClient] = {}
        self.sensor_buffers: Dict[str, LiveSensorBuffer] = {}
        self.device_names: Dict[str, str] = {}  # address -> device_name
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # ML state
        self.current_exercise = "—"
        self.current_confidence = 0.0
        self.current_reps = 0
        self.last_prediction_time = 0

        self.setup_ui()
        self.setup_data_collection()

    def setup_ui(self):
        """Set up the tkinter UI"""
        wrap = ttk.Frame(self.root, padding=15)
        wrap.pack(fill="both", expand=True)

        # Exercise label
        self.exercise_label = ttk.Label(
            wrap,
            text=self.current_exercise,
            font=("Segoe UI", 32, "bold")
        )
        self.exercise_label.pack()

        # Confidence
        self.confidence_label = ttk.Label(
            wrap,
            text="—.—%",
            font=("Segoe UI", 22)
        )
        self.confidence_label.pack(pady=(0, 10))

        # Reps
        self.reps_label = ttk.Label(
            wrap,
            text=f"Reps: {self.current_reps}",
            font=("Segoe UI", 20)
        )
        self.reps_label.pack(pady=(0, 15))

        # Status
        self.status_label = ttk.Label(
            wrap,
            text="Initializing...",
            font=("Segoe UI", 12)
        )
        self.status_label.pack(pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(wrap)
        button_frame.pack()

        self.start_btn = ttk.Button(
            button_frame,
            text="Start Live Classification",
            command=self.start_collection
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_collection,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

    def setup_data_collection(self):
        """Initialize data collection components"""
        # BLE UUIDs
        self.SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
        self.ORIENTATION_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
        self.CONTROL_CHAR_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

        # Commands
        self.CMD_START = 0x01
        self.CMD_STOP = 0x02
        self.CMD_HEARTBEAT = 0xFF

    def update_ui(self):
        """Update UI with current predictions"""
        self.exercise_label.config(text=self.current_exercise)
        self.confidence_label.config(text=f"{self.current_confidence:.1f}%")
        self.reps_label.config(text=f"Reps: {self.current_reps}")

        # Update status
        connected_count = len([b for b in self.sensor_buffers.values() if b.has_enough_data()])
        total_devices = len(self.sensor_buffers)
        self.status_label.config(
            text=f"Sensors: {connected_count}/{total_devices} ready"
        )

    def start_collection(self):
        """Start live data collection and classification"""
        if self.running:
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Connecting to sensors...")

        # Start data collection in background thread
        self.collection_thread = threading.Thread(target=self.run_data_collection)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        # Start UI update timer
        self.schedule_ui_update()

    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopped")

    def schedule_ui_update(self):
        """Schedule regular UI updates"""
        if self.running:
            self.update_ui()
            self.root.after(UPDATE_MS, self.schedule_ui_update)

    def run_data_collection(self):
        """Run data collection in background thread"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Run the async data collection
            self.loop.run_until_complete(self.collect_data_async())

        except Exception as e:
            logger.error(f"Data collection error: {e}")
        finally:
            if self.loop:
                self.loop.close()

    async def collect_data_async(self):
        """Async data collection and processing"""
        try:
            # Scan and connect to devices
            await self.scan_and_connect_devices()

            if not self.connected_clients:
                logger.warning("No devices connected")
                return

            # Start data collection from all devices
            await self.start_data_collection()

            # Main processing loop
            while self.running:
                await self.process_sensor_data()
                await asyncio.sleep(0.1)  # 10Hz processing

        except Exception as e:
            logger.error(f"Async collection error: {e}")
        finally:
            await self.disconnect_all_devices()

    async def scan_and_connect_devices(self):
        """Scan for and connect to HAR sensor devices"""
        logger.info("Scanning for HAR sensor devices...")

        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]

        if not har_devices:
            logger.warning("No HAR devices found")
            return

        logger.info(f"Found {len(har_devices)} HAR device(s)")

        # Connect to devices
        for device in har_devices:
            try:
                client = BleakClient(device.address)
                await client.connect()

                # Store connection and initialize buffer
                self.connected_clients[device.address] = client
                self.sensor_buffers[device.address] = LiveSensorBuffer()
                self.device_names[device.address] = device.name

                logger.info(f"Connected to {device.name}")

            except Exception as e:
                logger.error(f"Failed to connect to {device.name}: {e}")

    async def start_data_collection(self):
        """Start data collection on all connected devices"""
        for address, client in self.connected_clients.items():
            try:
                # Start notifications
                await client.start_notify(
                    self.ORIENTATION_CHAR_UUID,
                    lambda data, addr=address: self.handle_sensor_data(addr, data)
                )

                # Send start command
                await client.write_gatt_char(
                    self.CONTROL_CHAR_UUID,
                    bytes([self.CMD_START])
                )

                logger.info(f"Started data collection on {self.device_names[address]}")

            except Exception as e:
                logger.error(f"Failed to start collection on {self.device_names[address]}: {e}")

    def handle_sensor_data(self, address: str, data: bytes):
        """Handle incoming sensor data packets"""
        try:
            packet = self.parse_orientation_packet(data)
            if packet:
                sample = QuaternionSample(
                    absolute_timestamp_ms=int(time.time() * 1000),
                    relative_timestamp_ms=packet.timestamp,
                    qw=packet.qw,
                    qx=packet.qx,
                    qy=packet.qy,
                    qz=packet.qz
                )

                if address in self.sensor_buffers:
                    self.sensor_buffers[address].add_sample(sample)

        except Exception as e:
            logger.error(f"Error handling sensor data from {address}: {e}")

    def parse_orientation_packet(self, data: bytes) -> Optional[RawQuaternionPacket]:
        """Parse quaternion packet from BLE data"""
        if len(data) != 20:
            return None

        try:
            # Unpack binary data (matching Arduino format)
            sequence_number, node_id, timestamp = struct.unpack('<III', data[:12])

            # Unpack quantized quaternions (4 floats)
            qw, qx, qy, qz = struct.unpack('<ffff', data[12:28])

            return RawQuaternionPacket(
                sequence_number=sequence_number,
                node_id=node_id,
                timestamp=timestamp,
                qw=qw, qx=qx, qy=qz, qz=qz  # Note: Arduino sends qz last
            )

        except struct.error:
            return None

    async def process_sensor_data(self):
        """Process collected sensor data and make predictions"""
        current_time = time.time()

        # Only predict if we have recent data and enough time has passed
        if current_time - self.last_prediction_time < 0.5:  # Max 2Hz predictions
            return

        # Check if we have data from all expected sensors
        expected_sensors = {'WRIST', 'BICEP', 'CHEST', 'THIGH'}
        available_sensors = set()

        sensor_data = {}
        for address, buffer in self.sensor_buffers.items():
            if buffer.has_enough_data():
                device_name = self.device_names.get(address, "")
                sensor_type = device_name.replace("HAR_", "") if device_name.startswith("HAR_") else ""
                if sensor_type in expected_sensors:
                    available_sensors.add(sensor_type)
                    sensor_data[sensor_type] = buffer.get_samples()

        # Need all 4 sensors for reliable prediction
        if len(available_sensors) < 4:
            return

        try:
            # Extract enhanced features for each sensor
            features = {}
            for sensor_name, samples in sensor_data.items():
                # Convert QuaternionSample objects to dict format expected by post_process_data
                sample_dicts = [
                    {'qw': s.qw, 'qx': s.qx, 'qy': s.qy, 'qz': s.qz}
                    for s in samples
                ]
                
                # Extract all feature types
                temporal_features = extract_temporal_features(sample_dicts)
                frequency_features = extract_enhanced_frequency_features(sample_dicts)
                basic_features = extract_window_features(samples, window_type=0, node_id=0, window_id=0)
                
                # Combine all features
                features[sensor_name] = {
                    **temporal_features,
                    **frequency_features,
                    'qw_mean': basic_features.qw_mean,
                    'qx_mean': basic_features.qx_mean,
                    'qy_mean': basic_features.qy_mean,
                    'qz_mean': basic_features.qz_mean,
                    'qw_std': basic_features.qw_std,
                    'qx_std': basic_features.qx_std,
                    'qy_std': basic_features.qy_std,
                    'qz_std': basic_features.qz_std,
                    'sma': basic_features.sma,
                    'dominant_freq': basic_features.dominant_freq,
                    'total_samples': basic_features.total_samples
                }

            # Create wide-format feature vector for exercise classification
            feature_vector = []
            feature_names = []
            for sensor in ['CHEST', 'THIGH', 'WRIST', 'BICEP']:
                if sensor in features:
                    for feature_name, value in features[sensor].items():
                        feature_vector.append(value)
                        feature_names.append(f'{feature_name}_{sensor}')
                else:
                    # Fill missing sensors with zeros
                    for _ in range(11):  # 11 features per sensor
                        feature_vector.append(0)
                        feature_names.append(f'feature_{sensor}')

            # Make exercise prediction
            X_exercise = pd.DataFrame([feature_vector], columns=feature_names)
            exercise_pred = exercise_model.predict(X_exercise)[0]
            exercise_proba = exercise_model.predict_proba(X_exercise)[0]

            self.current_exercise = label_encoder.inverse_transform([exercise_pred])[0]
            self.current_confidence = exercise_proba[exercise_pred] * 100

            # Make rep prediction (simplified - just detect if active)
            # For full rep counting, would need longer windows and more complex logic
            rep_features_vector = []
            for feature_name in rep_features:
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    rep_features_vector.append(feature_vector[idx])
                else:
                    rep_features_vector.append(0)

            X_rep = pd.DataFrame([rep_features_vector], columns=rep_features)
            s1_proba = rep_detector_s1.predict_proba(X_rep.values)[0]
            is_active = s1_proba[1] > 0.40  # Detection threshold

            if is_active:
                # Simplified rep counting - in real implementation would track over time
                self.current_reps = (self.current_reps + 1) % 100  # Cycle for demo
            else:
                self.current_reps = 0

            self.last_prediction_time = current_time

        except Exception as e:
            logger.error(f"Prediction error: {e}")

    async def disconnect_all_devices(self):
        """Disconnect from all BLE devices"""
        for address, client in self.connected_clients.items():
            try:
                await client.disconnect()
                logger.info(f"Disconnected from {self.device_names.get(address, address)}")
            except Exception as e:
                logger.error(f"Error disconnecting from {address}: {e}")

        self.connected_clients.clear()
        self.sensor_buffers.clear()
        self.device_names.clear()

    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()

def main():
    """Main entry point"""
    ui = LiveClassificationUI()
    ui.run()

if __name__ == "__main__":
    main()