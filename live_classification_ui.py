#!/usr/bin/env python3
"""
Live Classification UI - Integrated Data Collection and Real-time ML Inference

This script combines:
- Tkinter UI for displaying predictions (from tkinter_ui.py)
- BLE data collection from sensor nodes (from Data_Collection.py)
- Real-time feature extraction and ML inference
- Live exercise classification and rep counting
"""

import asyncio
import json
import joblib
import math
import numpy as np
import pandas as pd
import random
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from bleak import BleakClient, BleakScanner
import logging
from datetime import datetime
from enum import IntEnum
import queue as pyqueue

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Local class definitions to avoid import issues
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
class WindowFeatures:
    """Statistical features extracted from a window of quaternion samples"""
    window_type: int
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
class RawQuaternionPacket:
    """Raw quaternion data packet from Arduino"""
    sequence_number: int
    node_id: int
    timestamp: int
    qw: float
    qx: float
    qy: float
    qz: float

class NodePlacement(IntEnum):
    """Sensor placement locations"""
    CHEST = 0
    THIGH = 1
    WRIST = 2
    BICEP = 3

# Node name mapping
NODE_NAMES = {
    0: "CHEST",
    1: "THIGH",
    2: "WRIST",
    3: "BICEP"
}

# Local feature extraction functions
def calculate_mean(samples):
    """Calculate mean of quaternion components"""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    qw_sum = sum(s.qw for s in samples)
    qx_sum = sum(s.qx for s in samples)
    qy_sum = sum(s.qy for s in samples)
    qz_sum = sum(s.qz for s in samples)
    n = len(samples)
    return qw_sum/n, qx_sum/n, qy_sum/n, qz_sum/n

def calculate_std(samples):
    """Calculate standard deviation of quaternion components"""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    qw_mean, qx_mean, qy_mean, qz_mean = calculate_mean(samples)
    qw_var = sum((s.qw - qw_mean)**2 for s in samples) / len(samples)
    qx_var = sum((s.qx - qx_mean)**2 for s in samples) / len(samples)
    qy_var = sum((s.qy - qy_mean)**2 for s in samples) / len(samples)
    qz_var = sum((s.qz - qz_mean)**2 for s in samples) / len(samples)
    return qw_var**0.5, qx_var**0.5, qy_var**0.5, qz_var**0.5

def calculate_sma(samples):
    """Calculate Signal Magnitude Area"""
    if not samples:
        return 0.0
    qw_mean, qx_mean, qy_mean, qz_mean = calculate_mean(samples)
    return sum(abs(s.qw - qw_mean) + abs(s.qx - qx_mean) + abs(s.qy - qy_mean) + abs(s.qz - qz_mean) for s in samples) / len(samples)

def calculate_dominant_frequency(samples):
    """Calculate dominant frequency using FFT"""
    if len(samples) < 10:
        return 0.0
    # Simple FFT-based frequency analysis
    qw_values = np.array([s.qw for s in samples])
    fft = np.fft.fft(qw_values)
    freqs = np.fft.fftfreq(len(qw_values))
    # Find peak frequency (excluding DC component)
    peak_idx = np.argmax(np.abs(fft[1:])) + 1
    return abs(freqs[peak_idx])

def extract_window_features(samples, window_type=0, node_id=0, window_id=0):
    """Extract features from a window of samples"""
    qw_mean, qx_mean, qy_mean, qz_mean = calculate_mean(samples)
    qw_std, qx_std, qy_std, qz_std = calculate_std(samples)
    sma = calculate_sma(samples)
    dominant_freq = calculate_dominant_frequency(samples)

    return WindowFeatures(
        window_type=window_type,
        node_id=node_id,
        window_id=window_id,
        qw_mean=qw_mean, qx_mean=qx_mean, qy_mean=qy_mean, qz_mean=qz_mean,
        qw_std=qw_std, qx_std=qx_std, qy_std=qy_std, qz_std=qz_std,
        sma=sma,
        dominant_freq=dominant_freq,
        total_samples=len(samples)
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

@dataclass
class RecordingConfig:
    """Configuration for a data collection session"""
    exercise_type: int  # Using int instead of Enum for simplicity
    rep_count: int
    participant_id: int
    session_notes: str = ""

@dataclass
class NodeStatus:
    """Status information for a connected node"""
    node_id: int
    is_running: bool
    is_calibrated: bool
    timestamp: int
    connected: bool = False

@dataclass
class LiveSensorBuffer:
    """Buffer for storing recent sensor samples for each device"""
    samples: List[QuaternionSample] = field(default_factory=list)
    max_size: int = WINDOW_SIZE

    def add_sample(self, sample: QuaternionSample):
        self.samples.append(sample)
        if len(self.samples) > self.max_size:
            self.samples.pop(0)

    def get_samples(self) -> List[QuaternionSample]:
        return self.samples.copy()

    def has_enough_data(self) -> bool:
        return len(self.samples) >= 50  # Minimum for feature extraction

class LiveClassificationController:
    """Controller that manages BLE data collection and launches tkinter UI"""

    def __init__(self):
        # Data collection components
        self.connected_clients: Dict[str, BleakClient] = {}
        self.sensor_buffers: Dict[str, LiveSensorBuffer] = {}
        self.device_names: Dict[str, str] = {}  # address -> device_name
        self.node_statuses: Dict[int, NodeStatus] = {}  # node_id -> status
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ui_process: Optional[subprocess.Popen] = None

        # Window configuration (matching Arduino)
        self.SHORT_WINDOW_SIZE = 150
        self.SHORT_WINDOW_OVERLAP = 0.75
        self.SHORT_WINDOW_STEP = 38  # 150 * (1 - 0.75)
        self.LONG_WINDOW_SIZE = 300
        self.LONG_WINDOW_OVERLAP = 0.5
        self.LONG_WINDOW_STEP = 150  # 300 * (1 - 0.5)

        # BLE DATA BUFFERING CONFIGURATION
        self.PACKET_QUEUE_MAX_SIZE = 10000  # Per-device queue size
        self.packet_queues: Dict[str, asyncio.Queue] = {}  # address -> queue
        self.processing_tasks: List[asyncio.Task] = []  # Background processing tasks
        self.packets_dropped: Dict[str, int] = {}  # Track dropped packets per device
        self.packets_received: Dict[str, int] = {}  # Track received packets per device

        # Prediction / UI communication
        self.prediction_queue: pyqueue.Queue = pyqueue.Queue()
        self.latest_prediction: Dict[str, Any] = {'label': 'N/A', 'prob': 0.0, 'timestamp': None}
        # Packet print timers
        self._last_packet_print_time: Dict[str, float] = {}
        self._processed_since_print: Dict[str, int] = {}
        # Connection monitor
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        # Packet logging options
        self.verbose_packet_logging = True
        self._last_packet_log_time: Dict[str, float] = {}

    def parse_orientation_packet(self, data: bytes) -> Optional[RawQuaternionPacket]:
        """
        Parse compressed quaternion orientation packet from Arduino.
        UPDATED for 14-byte packet (padded to 20 bytes): 4-byte timestamp and quantized quaternions.
        """
        # BLE packets are expected to be 20 bytes (padded); valid payload is first 14 bytes
        if len(data) != 20:
            return None

        try:
            payload = bytes(data[:14])

            # Packet structure: seq(1) + nodeId(1) + timestamp(4) + qw(2) + qx(2) + qy(2) + qz(2)
            (sequence_number, node_id, timestamp, qw_quant, qx_quant, qy_quant, qz_quant) = \
                struct.unpack("<BBIhhhh", payload)

            # De-quantize int16 to float range [-1, 1]
            CONVERSION_FACTOR = 32767.0
            qw = qw_quant / CONVERSION_FACTOR
            qx = qx_quant / CONVERSION_FACTOR
            qy = qy_quant / CONVERSION_FACTOR
            qz = qz_quant / CONVERSION_FACTOR

            # Basic validation
            if any(math.isnan(v) or math.isinf(v) for v in (qw, qx, qy, qz)):
                return None

            return RawQuaternionPacket(
                sequence_number=sequence_number,
                node_id=node_id,
                timestamp=timestamp,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
            )

        except (struct.error, ValueError):
            return None

    def parse_heartbeat_response(self, data: bytes) -> Optional[dict]:
        """Parse heartbeat response from Arduino"""
        # Handle different packet types
        if len(data) == 1:
            response_id = data[0]
            return {
                'response_id': response_id,
                'is_running': (response_id & 0x01) != 0,
                'is_calibrated': (response_id & 0x02) != 0,
                'node_id': 0,  # Default
                'timestamp': 0,
                'timestamp_ms': 0
            }

        elif len(data) == 3 and data[0] == 0xAA:
            response_id, node_id = data[1], data[2]
            return {
                'response_id': response_id,
                'node_id': node_id,
                'is_running': (response_id & 0x01) != 0,
                'is_calibrated': (response_id & 0x02) != 0,
                'timestamp': 0,
                'timestamp_ms': 0
            }

        elif len(data) >= 6 and data[0] == 0xFF:
            response_id, node_id = data[1], data[2]
            timestamp = struct.unpack('<I', data[2:6])[0]
            return {
                'response_id': response_id,
                'node_id': node_id,
                'is_running': (response_id & 0x01) != 0,
                'is_calibrated': (response_id & 0x02) != 0,
                'timestamp': timestamp,
                'timestamp_ms': timestamp
            }

        return None

    async def send_start_command(self, client: BleakClient, address: str, max_retries: int = 3):
        """Send start command to a device"""
        for attempt in range(max_retries):
            try:
                await client.write_gatt_char(
                    CONTROL_CHAR_UUID,
                    bytes([CMD_START])
                )
                logger.info(f"Sent start command to {self.device_names.get(address, address)}")
                return True
            except Exception as e:
                logger.warning(f"Start command attempt {attempt + 1} failed for {address}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)

        logger.error(f"Failed to start collection on {address} after {max_retries} attempts")
        return False

    async def send_calibrate_command(self, client: BleakClient, address: str, timeout: float = 8.0) -> bool:
        """Send calibration command to a device and wait for calibrated flag via heartbeat"""
        try:
            await client.write_gatt_char(CONTROL_CHAR_UUID, bytes([CMD_CALIBRATE]))
            logger.info(f"Sent CALIBRATE to {self.device_names.get(address, address)}")

            # Wait for node status to report calibrated
            start = time.time()
            node_id = self.extract_node_id_from_name(self.device_names.get(address, ""))
            while time.time() - start < timeout:
                status = None
                if node_id is not None:
                    status = self.node_statuses.get(node_id)
                if status and status.is_calibrated:
                    logger.info(f"Device {address} reported calibrated")
                    return True
                await asyncio.sleep(0.2)

            logger.warning(f"Calibration timeout for {address}")
            return False
        except Exception as e:
            logger.error(f"Error sending calibrate to {address}: {e}")
            return False

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

    async def scan_and_connect_devices(self):
        """Scan for and connect to all HAR sensor nodes sequentially"""
        logger.info("Scanning for HAR sensor nodes...")

        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]

        if not har_devices:
            logger.warning("No HAR devices found")
            return

        logger.info(f"Found {len(har_devices)} HAR device(s):")
        for device in har_devices:
            logger.info(f"  - {device.name} ({device.address})")

        # Connect to each device sequentially (BLE connections should not be done concurrently)
        successful_connections = 0
        for device in har_devices:
            try:
                logger.info(f"Connecting to {device.name}...")
                client = BleakClient(device.address)
                await client.connect()

                # Store connection and initialize components
                self.connected_clients[device.address] = client
                self.sensor_buffers[device.address] = LiveSensorBuffer()
                self.device_names[device.address] = device.name
                self.packet_queues[device.address] = asyncio.Queue(maxsize=self.PACKET_QUEUE_MAX_SIZE)
                self.packets_dropped[device.address] = 0
                self.packets_received[device.address] = 0

                # Initialize node status
                node_id = self.extract_node_id_from_name(device.name)
                if node_id is not None:
                    self.node_statuses[node_id] = NodeStatus(
                        node_id=node_id,
                        is_running=False,
                        is_calibrated=False,
                        timestamp=0,
                        connected=True
                    )

                logger.info(f"Connected to {device.name}")
                successful_connections += 1

            except Exception as e:
                logger.error(f"Failed to connect to {device.name}: {e}")

        logger.info(f"\nConnected to {successful_connections}/{len(har_devices)} devices")

    async def start_device_notifications(self, address: str, client: BleakClient):
        """
        Start notification handlers for a single device with buffered async queue processing.
        This prevents data loss by using a large buffer to handle BLE burst traffic.
        """
        try:
            # Start orientation data notifications
            await client.start_notify(
                ORIENTATION_CHAR_UUID,
                # Bleak calls notification handlers with (sender, data). Capture address and forward properly.
                lambda sender, data, addr=address: asyncio.create_task(self.handle_orientation_notification(addr, data))
            )

            # Also subscribe to control/heartbeat notifications if available
            try:
                await client.start_notify(
                    CONTROL_CHAR_UUID,
                    lambda sender, data, addr=address: asyncio.create_task(self.handle_orientation_notification(addr, data))
                )
            except Exception:
                # Not all firmwares may support notify on control char; ignore
                pass

            logger.info(f"Started notifications for {self.device_names[address]}")

        except Exception as e:
            logger.error(f"Failed to start notifications for {address}: {e}")

    async def handle_orientation_notification(self, address: str, data: bytes):
        """Handle orientation data notification"""
        try:
            # Ensure we store immutable bytes in the queue (Bleak may provide a bytearray)
            packet = bytes(data)

            # Try to add to queue without blocking
            await asyncio.wait_for(
                self.packet_queues[address].put(packet),
                timeout=0.1
            )
            self.packets_received[address] += 1

        except asyncio.TimeoutError:
            # Queue is full, drop packet
            self.packets_dropped[address] += 1
            if self.packets_dropped[address] % 100 == 0:  # Log every 100 drops
                logger.warning(f"Dropping packets for {address}, queue full")

        except Exception as e:
            logger.error(f"Error handling notification from {address}: {e}")

    async def process_packet_queue(self, address: str):
        """
        Asynchronously process queued BLE packets for a single device.
        This runs in parallel with BLE notifications to prevent data loss.
        """
        node_id = self.extract_node_id_from_name(self.device_names[address])
        node_name = NODE_NAMES.get(node_id, "UNKNOWN") if node_id is not None else "UNKNOWN"

        processed_count = 0
        logger.info(f"Started packet processor for {node_name} ({address})")

        try:
            while self.running:
                try:
                    # Wait for packet with timeout
                    data = await asyncio.wait_for(
                        self.packet_queues[address].get(),
                        timeout=1.0
                    )

                    # First, check if this is a heartbeat response
                    hb = self.parse_heartbeat_response(data)
                    if hb:
                        node_id = hb.get('node_id', None)
                        if node_id is None:
                            # Try to derive from device mapping
                            node_id = self.extract_node_id_from_name(self.device_names.get(address, ""))
                        if node_id is not None:
                            status = self.node_statuses.get(node_id)
                            if status:
                                status.is_running = hb.get('is_running', status.is_running)
                                status.is_calibrated = hb.get('is_calibrated', status.is_calibrated)
                                status.timestamp = hb.get('timestamp', status.timestamp)
                                status.connected = True
                            else:
                                self.node_statuses[node_id] = NodeStatus(
                                    node_id=node_id,
                                    is_running=hb.get('is_running', False),
                                    is_calibrated=hb.get('is_calibrated', False),
                                    timestamp=hb.get('timestamp', 0),
                                    connected=True
                                )
                        continue

                    # Otherwise treat as orientation packet
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
                            processed_count += 1

                            # Packet logging (print periodic summaries to terminal)
                            now = time.time()
                            last_print = self._last_packet_print_time.get(address, 0)
                            self._processed_since_print[address] = self._processed_since_print.get(address, 0) + 1
                            if now - last_print >= 1.0:
                                count = self._processed_since_print.get(address, 0)
                                node_id = self.extract_node_id_from_name(self.device_names.get(address, ""))
                                node_name = NODE_NAMES.get(node_id, 'UNKNOWN')
                                print(f"[{datetime.now().isoformat()}] {node_name} ({address}) - packets: {count} - last_seq={packet.sequence_number} ts={packet.timestamp}")
                                self._last_packet_print_time[address] = now
                                self._processed_since_print[address] = 0
                            # Verbose: print individual packets throttled to ~20Hz per device
                            if self.verbose_packet_logging:
                                last_log = self._last_packet_log_time.get(address, 0)
                                if now - last_log >= 0.05:
                                    node_id = self.extract_node_id_from_name(self.device_names.get(address, ""))
                                    node_name = NODE_NAMES.get(node_id, 'UNKNOWN')
                                    print(f"PKT {node_name} {address} seq={packet.sequence_number} ts={packet.timestamp} qw={packet.qw:.4f} qx={packet.qx:.4f} qy={packet.qy:.4f} qz={packet.qz:.4f}")
                                    self._last_packet_log_time[address] = now

                            # Perform lightweight inference when buffer has enough data
                            buffer = self.sensor_buffers[address]
                            if buffer.has_enough_data():
                                samples = buffer.get_samples()

                                # Prepare features using post_process_data helpers
                                try:
                                    temporal = extract_temporal_features(samples)
                                    freq = extract_enhanced_frequency_features(samples)
                                except Exception:
                                    temporal = {}
                                    freq = {}

                                # Basic statistical features
                                try:
                                    qw_mean, qx_mean, qy_mean, qz_mean = calculate_mean(samples)
                                    qw_std, qx_std, qy_std, qz_std = calculate_std(samples)
                                    sma = calculate_sma(samples)
                                    dominant = calculate_dominant_frequency(samples)
                                except Exception:
                                    qw_mean = qx_mean = qy_mean = qz_mean = 0.0
                                    qw_std = qx_std = qy_std = qz_std = 0.0
                                    sma = 0.0
                                    dominant = 0.0

                                merged = {}
                                merged.update(temporal)
                                merged.update(freq)
                                merged.update({
                                    'qw_mean': qw_mean, 'qx_mean': qx_mean, 'qy_mean': qy_mean, 'qz_mean': qz_mean,
                                    'qw_std': qw_std, 'qx_std': qx_std, 'qy_std': qy_std, 'qz_std': qz_std,
                                    'sma': sma, 'dominant_freq': dominant,
                                })

                                # Run model prediction (best-effort)
                                try:
                                    X = pd.DataFrame([merged])
                                    if hasattr(exercise_model, 'predict_proba'):
                                        probs = exercise_model.predict_proba(X)
                                        top_idx = int(np.argmax(probs[0]))
                                        label = label_encoder.inverse_transform([top_idx])[0] if 'label_encoder' in globals() else str(top_idx)
                                        prob = float(probs[0][top_idx])
                                    else:
                                        pred = exercise_model.predict(X)
                                        # If label_encoder maps strings, attempt inverse
                                        try:
                                            label = label_encoder.inverse_transform(pred)[0]
                                        except Exception:
                                            label = str(pred[0])
                                        prob = 1.0

                                    self.latest_prediction = {'label': label, 'prob': prob, 'timestamp': time.time(), 'source_address': address}
                                    # Push to UI queue (non-blocking)
                                    try:
                                        self.prediction_queue.put_nowait(self.latest_prediction)
                                    except Exception:
                                        pass
                                except Exception as e:
                                    # Model mismatch or feature error
                                    logger.debug(f"Prediction error: {e}")

                # end while

                except asyncio.TimeoutError:
                    # No packets in queue, continue
                    continue

        except asyncio.CancelledError:
            logger.info(f"Packet processor cancelled for {address}")
        except Exception as e:
            logger.error(f"Error in packet processor for {address}: {e}")
        finally:
            logger.info(f"Packet processor finished for {address}, processed {processed_count} packets")
        

    async def send_stop_command(self, client: BleakClient, address: str, max_retries: int = 3):
        """Send stop command to a device"""
        for attempt in range(max_retries):
            try:
                await client.write_gatt_char(
                    CONTROL_CHAR_UUID,
                    bytes([CMD_STOP])
                )
                logger.info(f"Sent stop command to {self.device_names[address]}")
                return True
            except Exception as e:
                logger.warning(f"Stop command attempt {attempt + 1} failed for {address}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)

        logger.error(f"Failed to stop collection on {address} after {max_retries} attempts")
        return False

    async def start_data_collection(self):
        """Start data collection on all connected devices"""
        logger.info("Starting data collection on all devices...")

        # Start notifications and commands for each device
        tasks = []
        for address, client in self.connected_clients.items():
            # Start notifications
            task = asyncio.create_task(self.start_device_notifications(address, client))
            tasks.append(task)

            # Start packet processing
            task = asyncio.create_task(self.process_packet_queue(address))
            self.processing_tasks.append(task)
            tasks.append(task)

        # Wait for notifications to start
        await asyncio.gather(*tasks[:len(self.connected_clients)], return_exceptions=True)

        # Send start commands
        start_tasks = []
        for address, client in self.connected_clients.items():
            task = asyncio.create_task(self.send_start_command(client, address))
            start_tasks.append(task)

        await asyncio.gather(*start_tasks, return_exceptions=True)

        logger.info("Data collection started on all devices")

        # Start a connection monitor that stops the system if any device disconnects
        if self._monitor_task is None:
            loop = asyncio.get_event_loop()
            self._monitor_task = loop.create_task(self._connection_monitor())

    async def stop_data_collection(self):
        """Stop data collection on all devices"""
        logger.info("Stopping data collection...")

        # Send stop commands
        stop_tasks = []
        for address, client in self.connected_clients.items():
            task = asyncio.create_task(self.send_stop_command(client, address))
            stop_tasks.append(task)

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Cancel processing tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Stop connection monitor (but do not disconnect BLE clients)
        if self._monitor_task:
            try:
                self._monitor_task.cancel()
            except Exception:
                pass
            self._monitor_task = None

        logger.info("Data collection stopped")

    async def disconnect_all_devices(self):
        """Disconnect from all BLE devices"""
        logger.info("Disconnecting from all devices...")

        for address, client in self.connected_clients.items():
            try:
                await client.disconnect()
                logger.info(f"Disconnected from {self.device_names.get(address, address)}")
            except Exception as e:
                logger.error(f"Error disconnecting from {address}: {e}")

        self.connected_clients.clear()
        self.sensor_buffers.clear()
        self.device_names.clear()
        self.node_statuses.clear()

    def launch_tkinter_ui(self):
        """Launch the tkinter UI in a separate process"""
        try:
            logger.info("Launching tkinter UI...")
            # Launch tkinter_ui.py as a subprocess
            # We no longer launch a separate UI process; main process contains UI
            self.ui_process = None
            logger.info("Tkinter UI launched")
        except Exception as e:
            logger.error(f"Failed to launch tkinter UI: {e}")

    def terminate_ui(self):
        """Terminate the tkinter UI process"""
        if self.ui_process:
            try:
                self.ui_process.terminate()
                self.ui_process.wait(timeout=5)
                logger.info("Tkinter UI terminated")
            except Exception as e:
                logger.error(f"Error terminating UI: {e}")
                try:
                    self.ui_process.kill()
                except:
                    pass

    async def run_live_collection(self):
        """Main live collection routine"""
        try:
            logger.info("=" * 70)
            logger.info("STARTING LIVE CLASSIFICATION")
            logger.info("=" * 70)

            # Launch the tkinter UI
            self.launch_tkinter_ui()

            # Give UI time to start
            await asyncio.sleep(2)

            # Scan and connect to devices only if not already connected
            if not self.connected_clients:
                await self.scan_and_connect_devices()

            if not self.connected_clients:
                logger.error("No devices connected - cannot start live classification")
                return

            # Start data collection
            await self.start_data_collection()

            # Keep running until stopped (stop only stops streaming, not BLE connections)
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in live collection: {e}")
        finally:
            # Stop streaming but keep BLE connections; full disconnect will be handled by UI shutdown
            await self.stop_data_collection()
            logger.info("Live collection stopped (BLE connections retained)")

    def start_live_collection(self):
        """Start live collection (called from main thread)"""
        if self.running:
            return

        self.running = True
        logger.info("Starting live classification...")

        # Start the async collection in a separate thread
        self.collection_thread = threading.Thread(target=self.run_async_collection)
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_live_collection(self):
        """Stop live collection"""
        logger.info("Stopping live classification...")
        self.running = False

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=10)

    def run_async_collection(self):
        """Run the async collection in a thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_live_collection())
        except Exception as e:
            logger.error(f"Async collection error: {e}")
        finally:
            if self.loop:
                self.loop.close()

class LiveClassificationUI:
    """Integrated UI with live BLE data collection and ML inference"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Classification Controller")
        self.root.geometry("400x200")

        # Controller instance
        self.controller = LiveClassificationController()

        self.setup_ui()

    def populate_available_devices(self, devices: List[object]):
        """Populate discovered devices into UI widgets so user sees connected-ready state"""
        # devices are bleak device objects with .name and .address
        for dev in devices:
            nid = self.controller.extract_node_id_from_name(dev.name if hasattr(dev, 'name') else '')
            if nid is None:
                continue
            self.controller.device_names[dev.address] = dev.name
            # Mark as connected in controller (but not yet fully connected clients)
            self.controller.node_statuses[nid] = NodeStatus(node_id=nid, is_running=False, is_calibrated=False, timestamp=0, connected=True)
            # Update UI
            if nid in self.sensor_widgets:
                self.sensor_widgets[nid]['status'].config(text='Connected', foreground='blue')

    async def _connection_monitor(self):
        """Monitor BLE client connections and stop the system if any disconnects."""
        logger.info("Starting connection monitor task")
        try:
            while True:
                for addr, client in list(self.connected_clients.items()):
                    try:
                        if not client.is_connected:
                            logger.error(f"Connection lost: {addr}. Stopping live classification.")
                            # Signal stop
                            self.running = False
                            # Cancel processing tasks
                            for t in self.processing_tasks:
                                try:
                                    t.cancel()
                                except Exception:
                                    pass
                            return
                    except Exception as e:
                        logger.error(f"Error checking connection for {addr}: {e}")
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Connection monitor cancelled")

    def setup_ui(self):
        """Set up the control UI"""
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(frame, text="HAR Live Classification", font=("Arial", 16, "bold"))
        title.pack(pady=(0, 20))
        # Top status
        self.status_label = ttk.Label(frame, text="Ready to start", font=("Arial", 12))
        self.status_label.pack(pady=(0, 10))

        # Sensor grid (4 sensors)
        sensors_frame = ttk.Frame(frame)
        sensors_frame.pack(pady=(0, 10), fill='x')

        self.sensor_widgets: Dict[int, Dict[str, Any]] = {}
        for node_id in range(4):
            col = node_id
            sub = ttk.Frame(sensors_frame, relief='groove', padding=8)
            sub.grid(row=0, column=col, padx=5, sticky='nsew')
            label = ttk.Label(sub, text=NODE_NAMES.get(node_id, f"NODE{node_id}"), font=("Arial", 10, "bold"))
            label.pack()
            status = ttk.Label(sub, text="Disconnected", foreground='gray')
            status.pack()
            calib_btn = ttk.Button(sub, text="Calibrate", command=(lambda nid=node_id: self.calibrate_node(nid)))
            calib_btn.pack(pady=(6,0))

            self.sensor_widgets[node_id] = {'frame': sub, 'label': label, 'status': status, 'calib_btn': calib_btn}

        # Control buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=(6,6))

        self.calibrate_all_btn = ttk.Button(button_frame, text="Calibrate All", command=self.calibrate_all)
        self.calibrate_all_btn.pack(side="left", padx=6)

        self.start_set_btn = ttk.Button(button_frame, text="Start Set", command=self.start_collection, state='disabled')
        self.start_set_btn.pack(side="left", padx=6)

        self.stop_btn = ttk.Button(button_frame, text="Stop Set", command=self.stop_collection, state='disabled')
        self.stop_btn.pack(side="left", padx=6)

        # Prediction display
        self.pred_label = ttk.Label(frame, text="Prediction: N/A", font=("Arial", 14, "bold"))
        self.pred_label.pack(pady=(10, 0))

        self.prob_label = ttk.Label(frame, text="Confidence: 0.0", font=("Arial", 12))
        self.prob_label.pack()

        # Periodic UI update
        self.root.after(300, self.ui_update_loop)

    def start_collection(self):
        """Start live collection"""
        self.start_set_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Starting live classification...")

        # Start controller (connects to devices and begins collecting)
        self.controller.start_live_collection()

        # Update status after a short delay
        self.root.after(2000, lambda: self.status_label.config(text="Live classification running"))

    def stop_collection(self):
        """Stop live collection"""
        self.start_set_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Stopping...")

        self.controller.stop_live_collection()

        # Update status after stopping
        self.root.after(2000, lambda: self.status_label.config(text="Stopped"))

    def calibrate_node(self, node_id: int):
        """Trigger calibration for a single node (async)"""
        # Find address for node
        address = None
        for addr, name in self.controller.device_names.items():
            nid = self.controller.extract_node_id_from_name(name)
            if nid == node_id:
                address = addr
                break
        if not address:
            self.sensor_widgets[node_id]['status'].config(text='Not connected', foreground='gray')
            return

        # Run calibration in background
        def do_calib():
            client = self.controller.connected_clients.get(address)
            if not client:
                return
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ok = loop.run_until_complete(self.controller.send_calibrate_command(client, address))
            loop.close()
            # Update UI
            if ok:
                self.sensor_widgets[node_id]['status'].config(text='Calibrated', foreground='green')
            else:
                self.sensor_widgets[node_id]['status'].config(text='Calibration Failed', foreground='red')

        t = threading.Thread(target=do_calib, daemon=True)
        t.start()

    def calibrate_all(self):
        """Calibrate all connected nodes"""
        self.calibrate_all_btn.config(state='disabled')
        # Spawn threads for each connected device
        for addr, client in list(self.controller.connected_clients.items()):
            name = self.controller.device_names.get(addr, '')
            nid = self.controller.extract_node_id_from_name(name)
            if nid is None:
                continue

            def do_all_calib(a=addr, n=nid):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ok = loop.run_until_complete(self.controller.send_calibrate_command(self.controller.connected_clients[a], a))
                loop.close()
                if ok:
                    try:
                        self.sensor_widgets[n]['status'].config(text='Calibrated', foreground='green')
                    except Exception:
                        pass

            threading.Thread(target=do_all_calib, daemon=True).start()

        # Re-enable after a pause
        self.root.after(5000, lambda: self.calibrate_all_btn.config(state='normal'))

    def ui_update_loop(self):
        """Periodic UI updates: prediction and sensor statuses"""
        # Drain prediction queue
        try:
            while True:
                pred = self.controller.prediction_queue.get_nowait()
                label = pred.get('label', 'N/A')
                prob = pred.get('prob', 0.0)
                self.pred_label.config(text=f"Prediction: {label}")
                self.prob_label.config(text=f"Confidence: {prob:.2f}")
        except Exception:
            pass

        # Update sensor statuses from controller.node_statuses
        for nid, widget in self.sensor_widgets.items():
            status = self.controller.node_statuses.get(nid)
            if not status:
                widget['status'].config(text='Disconnected', foreground='gray')
            else:
                if status.is_calibrated:
                    widget['status'].config(text='Calibrated', foreground='green')
                elif status.connected:
                    widget['status'].config(text='Connected', foreground='blue')
                else:
                    widget['status'].config(text='Disconnected', foreground='gray')

        # Enable start if four devices are known (they'll be connected when start runs)
        known_devices = len(self.controller.device_names)
        if known_devices >= 4:
            self.start_set_btn.config(state='normal')

        # Repeat
        self.root.after(300, self.ui_update_loop)

    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()

async def scan_and_display_sensors():
    """Scan for HAR sensor devices and display them in terminal"""
    print("Scanning for HAR sensor devices...")
    print("=" * 50)

    try:
        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]

        if not har_devices:
            print("No HAR sensor devices found!")
            print("   Make sure your sensors are powered on and in range.")
            return []

        print(f"Found {len(har_devices)} HAR device(s):")
        print()

        for i, device in enumerate(har_devices, 1):
            print(f"   {i}. {device.name}")
            print(f"      Address: {device.address}")
            rssi = getattr(device, 'rssi', 'N/A')
            if rssi == 'N/A':
                print(f"      RSSI: N/A")
            else:
                print(f"      RSSI: {rssi} dBm")
            print()

        print("Attempting to connect to devices...")

        # Try to connect briefly to verify they're accessible
        connected_count = 0
        for device in har_devices:
            try:
                client = BleakClient(device.address)
                await client.connect()
                await client.disconnect()
                print(f"   {device.name}: Connection successful")
                connected_count += 1
            except Exception as e:
                print(f"   {device.name}: Connection failed - {str(e)[:50]}...")

        print()
        if connected_count == len(har_devices):
            print(f"All {connected_count} sensors are ready!")
        else:
            print(f"{connected_count}/{len(har_devices)} sensors are accessible.")
            print("   Some sensors may need to be restarted or moved closer.")

        # Return devices list for UI to populate (addresses and names)
        return har_devices

    except Exception as e:
        print(f"Scanning failed: {e}")
        return False

def main():
    """Main entry point - scan sensors first, then launch UI"""
    print("HAR Live Classification System")
    print("Loading environment variables from .env...")

    # Load environment variables
    load_dotenv()

    # Scan for sensors first
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        devices = loop.run_until_complete(scan_and_display_sensors())
        loop.close()

        if not devices:
            print("\nCannot proceed without accessible sensors.")
            print("   Please check your sensor connections and try again.")
            input("\nPress Enter to exit...")
            return

    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        return
    except Exception as e:
        print(f"\nSensor scan failed: {e}")
        return

    print("\nLaunching Live Classification UI...")
    print("   Close the window to stop the system.")
    print()

    # Launch the UI
    try:
        ui = LiveClassificationUI()
        # Populate discovered devices into UI (shows Ready)
        ui.populate_available_devices(devices)
        ui.run()
    except Exception as e:
        print(f"Failed to launch UI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()