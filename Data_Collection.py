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

    def send_features_to_ml(self, *, node_id: int, node_name: str,
                            window_type: int, window_id: int,
                            start_ms: int, end_ms: int,
                            features: Dict[str, Any]) -> None:
        
        """Placeholder function to take data for ML"""
        return
    
    async def handle_live_classification(self):
        print("\n" + "=" * 70)
        print("LIVE CLASSIFICATION")
        print("=" * 70)

        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        self.classifcation_mode = True

        """Function to record IMU data for classification """
        return
        
        
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