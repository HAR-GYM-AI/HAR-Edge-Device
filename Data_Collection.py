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
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from bleak import BleakClient, BleakScanner
import logging

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
    SHOULDER_PRESS = 2
    LATERAL_RAISE = 3
    TRICEP_EXTENSION = 4

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
    """Individual quaternion sample"""
    timestamp: int
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
        
        # Window configuration (matching Arduino)
        self.SHORT_WINDOW_SIZE = 150
        self.SHORT_WINDOW_OVERLAP = 0.75
        self.SHORT_WINDOW_STEP = 38  # 150 * (1 - 0.75)
        self.LONG_WINDOW_SIZE = 300
        self.LONG_WINDOW_OVERLAP = 0.5
        self.LONG_WINDOW_STEP = 150  # 300 * (1 - 0.5)
        
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
            print("┌─ CONNECTED NODES " + "─" * 50 + "┐")
            for i, (node_id, status) in enumerate(self.node_statuses.items()):
                if i > 0:
                    print("├" + "─" * 68 + "┤")
                
                status_color = "\033[92m" if status.is_running else "\033[93m"
                reset_color = "\033[0m"
                
                node_name = NODE_NAMES.get(node_id, "UNKNOWN")
                print(f"│ {node_name} (ID: {node_id})" + " " * (45 - len(node_name) - len(str(node_id))) + "│")
                print(f"│   Connected:  {status_color}{'YES' if status.connected else 'NO'}{reset_color:<45}│")
                print(f"│   Running:    {status_color}{'YES' if status.is_running else 'NO'}{reset_color:<45}│")
                print(f"│   Calibrated: {'YES' if status.is_calibrated else 'NO':<45}│")
                print(f"│   Timestamp:  {status.timestamp:<45}│")
            
            print("└" + "─" * 68 + "┘")
            print(f"Total connected: {len(self.node_statuses)}/{len(NODE_NAMES)} nodes")
            print()
        else:
            print("┌─ NODE STATUS " + "─" * 54 + "┐")
            print("│ No nodes connected                                                 │")
            print("└" + "─" * 68 + "┘")
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
        print("  [0] Exit")
        print()
        print("─" * 70)
    
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
            print("─" * 70)
            print("Configuration Summary:")
            print(f"  Exercise: {exercise_type.name.replace('_', ' ').title()}")
            print(f"  Reps: {rep_count}")
            print(f"  Participant: {participant_id}")
            if session_notes:
                print(f"  Notes: {session_notes}")
            print("─" * 70)
            
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
        """Parse raw quaternion orientation packet from Arduino (20 bytes)"""
        if len(data) != 20:
            return None
        
        try:
            # Packet structure: seq(1) + nodeId(1) + timestamp(2) + qw(4) + qx(4) + qy(4) + qz(4)
            sequence_number = data[0]
            node_id = data[1]
            timestamp = int.from_bytes(data[2:4], byteorder='little')
            qw = struct.unpack('<f', data[4:8])[0]
            qx = struct.unpack('<f', data[8:12])[0]
            qy = struct.unpack('<f', data[12:16])[0]
            qz = struct.unpack('<f', data[16:20])[0]
            
            # Validate quaternion components
            if any(math.isnan(v) or math.isinf(v) for v in [qw, qx, qy, qz]):
                logger.warning(f"Invalid quaternion values: qw={qw}, qx={qx}, qy={qy}, qz={qz}")
                return None
            
            return RawQuaternionPacket(
                sequence_number=sequence_number,
                node_id=node_id,
                timestamp=timestamp,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz
            )
        except (struct.error, ValueError) as e:
            logger.error(f"Error parsing orientation packet: {e}")
            return None
    
    def parse_heartbeat_response(self, data: bytes):
        """Parse heartbeat response from Arduino"""
        # Handle different packet types
        if len(data) == 1:
            # Single byte - might be partial packet or identification
            return None
            
        elif len(data) == 3 and data[0] == 0xAA:
            # ACK/NACK packet
            return None
            
        elif len(data) >= 6 and data[0] == 0xFF:
            # Heartbeat response packet (6 bytes)
            response_id = data[0]
            node_id = data[1]
            is_running = bool(data[2])
            is_calibrated = bool(data[3])
            timestamp = (data[4] << 8) | data[5]  # Combine two bytes (big-endian)
            
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
            return None
            
        else:
            # Unknown packet format
            return None
    
    def display_heartbeat_response(self, response: dict):
        """Display formatted heartbeat response"""
        print("\n" + "═" * 50)
        print("HEARTBEAT RESPONSE")
        print("═" * 50)
        
        # Status indicators
        running_status = "✓ YES" if response['is_running'] else "✗ NO"
        cal_status = "✓ YES" if response['is_calibrated'] else "✗ NO"
        
        print(f"\nNode Information:")
        print(f"  ID:           {response['node_id']} ({response['node_name']})")
        print(f"  Running:      {running_status}")
        print(f"  Calibrated:   {cal_status}")
        print(f"  Timestamp:    {response['timestamp']} ticks ({response['timestamp_ms']} ms)")
        
        print("\nRaw Data:")
        print(f"  Response ID:  0x{response['response_id']:02X}")
        
        print("═" * 50 + "\n")
    
    async def scan_and_connect_all_devices(self):
        """Scan for and connect to all HAR sensor nodes concurrently"""
        print("Scanning for HAR sensor nodes...")
        
        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]
        
        if not har_devices:
            print("No HAR devices found!")
            return
        
        print(f"Found {len(har_devices)} HAR device(s):")
        for device in har_devices:
            print(f"  • {device.name} ({device.address})")
        print()
        
        # Connect to each device concurrently
        connection_tasks = []
        for device in har_devices:
            task = self.connect_single_device(device)
            connection_tasks.append(task)
        
        # Wait for all connections to complete
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = 0
        for device, result in zip(har_devices, results):
            if isinstance(result, Exception):
                print(f"✗ Failed to connect to {device.name}: {result}")
            else:
                successful_connections += 1
        
        print(f"\nConnected to {successful_connections}/{len(har_devices)} devices")
        print()
    
    async def connect_single_device(self, device):
        """Connect to a single BLE device"""
        print(f"Connecting to {device.name}...")
        try:
            client = BleakClient(device.address)
            await client.connect()
            
            if client.is_connected:
                print(f"✓ Connected to {device.name}")
                
                self.connected_clients[device.address] = client
                
                # Initialize node status
                node_id = self.extract_node_id_from_name(device.name)
                if node_id is not None:
                    self.node_statuses[node_id] = NodeStatus(
                        node_id=NodePlacement(node_id),
                        is_running=False,
                        is_calibrated=False,
                        timestamp=0,
                        connected=True
                    )
                
                # Give Arduino time to initialize
                await asyncio.sleep(0.5)
                return True
            else:
                raise Exception("Connection failed")
                
        except Exception as e:
            raise e
    
    def extract_node_id_from_name_by_address(self, address: str) -> Optional[int]:
        """Extract node ID from device name by address"""
        for addr, client in self.connected_clients.items():
            if addr == address:
                # Try to get device name from client if available
                # For now, we'll use the stored node status
                for node_id, status in self.node_statuses.items():
                    if status.connected:  # This is approximate, but works for now
                        return node_id
        return None
    
    async def start_device_notifications(self, address: str, client: BleakClient, device_buffers: Dict[str, Dict[str, Any]]):
        """Start notification handlers for a single device"""
        try:
            # Create notification handler for this device
            def notification_handler(sender, data):
                self.handle_orientation_data(address, data, device_buffers)
            
            # Subscribe to orientation characteristic only
            await client.start_notify(ORIENTATION_CHAR_UUID, notification_handler)
            print(f"✓ Started notifications for {address}")
            
            # Keep the handler running (this will run indefinitely until cancelled)
            while True:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            # Expected when stopping collection
            pass
        except Exception as e:
            print(f"✗ Error in notification handler for {address}: {e}")
    
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
        """Send stop command to a device with retry logic"""
        for attempt in range(max_retries):
            try:
                command = bytes([CMD_STOP])
                await asyncio.wait_for(
                    client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True),
                    timeout=5.0
                )
                logger.info(f"Stopped collection on {address}")
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
        """Finalize and organize session data from device buffers"""
        total_windows = 0
        total_samples = 0
        total_short_windows = 0
        total_long_windows = 0
        
        for address, buffer in device_buffers.items():
            # All windows are already complete (extracted on the fly)
            device_windows = buffer['windows']
            
            # Add device info to each window
            for window_data in device_windows:
                window_data['device_address'] = address
                
                total_samples += len(window_data.get('samples', []))
                if window_data['window_type'] == 'short':
                    total_short_windows += 1
                else:
                    total_long_windows += 1
            
            session_data['device_windows'][address] = device_windows
            total_windows += len(device_windows)
        
        # Update session metadata
        session_data['session_metadata']['total_windows'] = total_windows
        session_data['session_metadata']['total_samples'] = total_samples
        session_data['session_metadata']['short_windows'] = total_short_windows
        session_data['session_metadata']['long_windows'] = total_long_windows
    
    async def monitor_connections(self, device_buffers: Dict[str, Dict[str, Any]]):
        """Monitor BLE connection health during recording"""
        try:
            while True:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                disconnected_devices = []
                for address, client in self.connected_clients.items():
                    try:
                        # Simple connection check
                        if not client.is_connected:
                            logger.warning(f"Device {address} appears disconnected")
                            disconnected_devices.append(address)
                    except Exception as e:
                        logger.warning(f"Connection check failed for {address}: {e}")
                        disconnected_devices.append(address)
                
                # Log connection status
                if disconnected_devices:
                    logger.warning(f"Disconnected devices detected: {disconnected_devices}")
                    for address in disconnected_devices:
                        device_buffers[address]['stats']['connection_dropped'] = True
                
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
        print(f"✓ Status received from {successful_requests}/{len(self.connected_clients)} devices")
    
    async def request_single_status(self, address: str, client: BleakClient):
        """Request status from a single device"""
        try:
            success = await self.send_heartbeat_request(client)
            if success:
                print(f"✓ Status received from {address}")
            else:
                print(f"✗ Failed to get status from {address}")
        except Exception as e:
            print(f"✗ Error requesting status from {address}: {e}")
    
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
            parsed = self.parse_heartbeat_response(data)
            if parsed:  # Only count valid heartbeat responses
                nonlocal heartbeat_response
                heartbeat_response = parsed
                response_received.set()
        
        # Subscribe to notifications
        await client.start_notify(CONTROL_CHAR_UUID, notification_handler)
        
        try:
            # Send heartbeat command
            await client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True)
            print("Heartbeat request sent")
            
            # Wait for response (timeout after 3 seconds)
            try:
                await asyncio.wait_for(response_received.wait(), timeout=3.0)
                
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
                return False
                
        finally:
            # Unsubscribe from notifications
            await client.stop_notify(CONTROL_CHAR_UUID)
    
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
        """Disconnect from all connected BLE clients"""
        if self.connected_clients:
            print(f"\nDisconnecting from {len(self.connected_clients)} device(s)...")
            for address, client in self.connected_clients.items():
                try:
                    await client.disconnect()
                    print(f"✓ Disconnected from {address}")
                except Exception as e:
                    print(f"✗ Error disconnecting from {address}: {e}")
            self.connected_clients.clear()
            self.node_statuses.clear()
            print("All connections closed")
    
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
                'stats': {'total_packets': 0, 'total_samples': 0, 'short_windows': 0, 'long_windows': 0}
            }
            session_data['device_windows'][address] = []
            node_id = self.extract_node_id_from_name_by_address(address)
            session_data['session_metadata']['devices'][address] = {
                'node_id': node_id,
                'node_name': NODE_NAMES.get(node_id, "UNKNOWN")
            }
        
        try:
            # Start concurrent notification handlers and connection monitoring
            print("Starting concurrent data collection...")
            notification_tasks = []
            
            for address, client in self.connected_clients.items():
                task = self.start_device_notifications(address, client, device_buffers)
                notification_tasks.append(task)
            
            # Add connection monitoring task
            monitor_task = self.monitor_connections(device_buffers)
            notification_tasks.append(monitor_task)
            
            # Start all tasks
            notification_coroutines = [task for task in notification_tasks]
            
            # Send start command to all devices concurrently
            print("Starting data collection on all devices...")
            start_tasks = []
            for address, client in self.connected_clients.items():
                task = self.send_start_command(client, address)
                start_tasks.append(task)
            
            start_results = await asyncio.gather(*start_tasks, return_exceptions=True)
            successful_starts = sum(1 for result in start_results if result is True)
            print(f"✓ Started collection on {successful_starts}/{len(self.connected_clients)} devices")
            
            if successful_starts == 0:
                logger.error("Failed to start collection on any device")
                return
            
            # Collect data until user presses Enter or Ctrl+C
            print("\nCollecting data... (Press Enter to stop)")
            
            # Run notification handlers concurrently with input monitoring
            collection_task = asyncio.gather(*notification_coroutines, return_exceptions=True)
            input_task = asyncio.get_event_loop().run_in_executor(None, input, "")
            
            done, pending = await asyncio.wait(
                [collection_task, input_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            print("\nRecording stopped by user")
            
            # Send stop command to all devices concurrently
            print("Stopping data collection...")
            stop_tasks = []
            for address, client in self.connected_clients.items():
                task = self.send_stop_command(client, address)
                stop_tasks.append(task)
            
            stop_results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            successful_stops = sum(1 for result in stop_results if result is True)
            print(f"✓ Stopped collection on {successful_stops}/{len(self.connected_clients)} devices")
            
            # Process and finalize collected data
            await self.finalize_session_data(device_buffers, session_data)
            
            # Display collection statistics
            self.display_collection_statistics_multi_device(device_buffers)
            
            # Save collected data
            session_data['session_metadata']['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(session_filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"\n✓ Session complete! Data saved to {session_filename}")
            
        except Exception as e:
            print(f"Error during recording: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Unsubscribe from notifications
            for address, client in self.connected_clients.items():
                try:
                    await client.stop_notify(ORIENTATION_CHAR_UUID)
                    print(f"✓ Unsubscribed from {address}")
                except Exception as e:
                    print(f"✗ Error unsubscribing from {address}: {e}")
    
    def display_collection_statistics_multi_device(self, device_buffers: Dict[str, Dict[str, Any]]):
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
            node_name = NODE_NAMES.get(node_id, "UNKNOWN")
            
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
        
        print(f"{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Devices: {total_devices}")
        print(f"Total windows: {total_windows} ({total_short_windows} short, {total_long_windows} long)")
        print(f"Total samples: {total_samples}")
        print(f"Total packets: {total_packets}")
        print(f"Data rate: ~{total_packets / 1 if total_packets > 0 else 0:.1f} packets/sec")
        print(f"{'='*70}")
    

    
    def handle_orientation_data(self, address: str, data: bytes, device_buffers: Dict[str, Dict[str, Any]]):
        """Handle incoming raw quaternion data from BLE notifications"""
        try:
            # Update packet stats
            device_buffers[address]['stats']['total_packets'] += 1
            
            # Parse orientation packet
            packet = self.parse_orientation_packet(data)
            if not packet:
                logger.warning(f"Failed to parse orientation packet from {address}")
                return
            
            device_buffers[address]['stats']['total_samples'] += 1
            
            # Convert to QuaternionSample
            sample = QuaternionSample(
                timestamp=packet.timestamp,
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
                
                # Extract features from short window
                features = extract_window_features(
                    buffer['short_window_buffer'],
                    window_type=0,  # short
                    node_id=packet.node_id,
                    window_id=buffer['short_window_id']
                )
                
                # Store window with features and samples
                window_data = {
                    'window_type': 'short',
                    'window_id': buffer['short_window_id'],
                    'node_id': packet.node_id,
                    'timestamp': datetime.now().isoformat(),
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
                    'samples': [{'timestamp': s.timestamp, 'qw': s.qw, 'qx': s.qx, 
                                'qy': s.qy, 'qz': s.qz} 
                               for s in buffer['short_window_buffer']]
                }
                
                buffer['windows'].append(window_data)
                buffer['short_window_id'] += 1
                buffer['samples_since_last_short'] = 0
                buffer['stats']['short_windows'] += 1
                
                # Log progress every 10 windows
                if buffer['short_window_id'] % 10 == 0:
                    logger.info(f"Device {address}: Extracted short window {buffer['short_window_id']}")
            
            # Check if we should extract long window features
            if (buffer['samples_since_last_long'] >= self.LONG_WINDOW_STEP and
                len(buffer['long_window_buffer']) >= self.LONG_WINDOW_SIZE):
                
                # Extract features from long window
                features = extract_window_features(
                    buffer['long_window_buffer'],
                    window_type=1,  # long
                    node_id=packet.node_id,
                    window_id=buffer['long_window_id']
                )
                
                # Store window with features and samples
                window_data = {
                    'window_type': 'long',
                    'window_id': buffer['long_window_id'],
                    'node_id': packet.node_id,
                    'timestamp': datetime.now().isoformat(),
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
                    'samples': [{'timestamp': s.timestamp, 'qw': s.qw, 'qx': s.qx, 
                                'qy': s.qy, 'qz': s.qz} 
                               for s in buffer['long_window_buffer']]
                }
                
                buffer['windows'].append(window_data)
                buffer['long_window_id'] += 1
                buffer['samples_since_last_long'] = 0
                buffer['stats']['long_windows'] += 1
                
                # Log progress every 5 windows
                if buffer['long_window_id'] % 5 == 0:
                    logger.info(f"Device {address}: Extracted long window {buffer['long_window_id']}")
                    
        except Exception as e:
            logger.error(f"Error processing orientation data from {address}: {e}")
    
    async def handle_calibration(self):
        """Handle sensor calibration on all connected devices"""
        print("\n" + "=" * 70)
        print("SENSOR CALIBRATION")
        print("=" * 70)
        print("\nKeep sensors stationary during calibration!")
        
        if not self.connected_clients:
            print("No devices connected")
            input("\nPress Enter to continue...")
            return
        
        confirm = input(f"\nCalibrate {len(self.connected_clients)} device(s)? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Calibration cancelled")
            input("\nPress Enter to continue...")
            return
        
        print(f"\nSending calibration command to {len(self.connected_clients)} device(s)...")
        
        # Send calibration command to all connected devices
        for address, client in self.connected_clients.items():
            try:
                command = bytes([CMD_CALIBRATE])
                await client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True)
                print(f"✓ Calibration started on {address}")
                
                # Give Arduino time to process (calibration takes several seconds)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"✗ Error sending to {address}: {e}")
        
        print("\n✓ Calibration initiated on all devices")
        print("Note: Calibration takes ~10 seconds per device")
        input("Press Enter to continue...")
    
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
        
        print("\n✓ Status update complete")
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
                print(f"✓ Reset command sent to {address}")
                
                # Give Arduino time to process
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"✗ Error sending to {address}: {e}")
        
        print("\n✓ Reset commands sent to all devices")
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