#!/usr/bin/env python3
"""
Simple Heartbeat Test - HAR System
Tests BLE communication with Arduino sensor node
"""

import asyncio
import struct
from bleak import BleakClient, BleakScanner

# BLE UUIDs (matching Arduino)
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
CONTROL_CHAR_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

# Command constants
CMD_HEARTBEAT = 0xFF

DEBUG = True  #see all BLE trafficF

# Node placement names
NODE_NAMES = {
    0: "WRIST",
    1: "BICEP",
    2: "CHEST",
    3: "THIGH"
}

class HeartbeatTester:
    def __init__(self):
        self.control_char = None
        self.all_responses = []  # Track all responses
        
    async def scan_for_device(self):
        """Scan for HAR sensor nodes"""
        print("Scanning for HAR sensor nodes...")
        print("─" * 50)
        
        devices = await BleakScanner.discover(timeout=5.0)
        har_devices = [d for d in devices if d.name and d.name.startswith("HAR_")]
        
        if not har_devices:
            print("No HAR devices found!")
            return None
        
        print(f"Found {len(har_devices)} HAR device(s):\n")
        for i, device in enumerate(har_devices, 1):
            print(f"  [{i}] {device.name}")
            print(f"      Address: {device.address}")
            # RSSI might not always be available
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
    
    def parse_heartbeat_response(self, data: bytes):
        """Parse heartbeat response from Arduino"""
        # Show raw bytes for debugging
        hex_str = ' '.join([f'{b:02x}' for b in data])
        print(f"  [DEBUG] Received {len(data)} bytes: {hex_str}")
        
        # Handle different packet types
        if len(data) == 1:
            # Single byte - might be partial packet or identification
            print(f"  [DEBUG] Ignoring 1-byte response: 0x{data[0]:02X}")
            return None
            
        elif len(data) == 3 and data[0] == 0xAA:
            # ACK/NACK packet
            cmd_id = data[1]
            status = data[2]
            print(f"  [DEBUG] Received ACK packet for command 0x{cmd_id:02X}, status: 0x{status:02X}")
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
            packet_type = data[0]
            node_id = data[1]
            node_placement = data[2]
            sampling_rate = data[3]
            short_window = (data[4] << 8) | data[5]
            long_window = (data[6] << 8) | data[7]
            firmware_version = data[8]
            
            print(f"  [DEBUG] Node Identification Packet:")
            print(f"    Node ID: {node_id} ({NODE_NAMES.get(node_id, 'UNKNOWN')})")
            print(f"    Sampling Rate: {sampling_rate} Hz")
            print(f"    Window Sizes: Short={short_window}, Long={long_window}")
            print(f"    Firmware: v{firmware_version}")
            return None
            
        else:
            # Unknown packet format
            print(f"  [DEBUG] Unexpected response format: {len(data)} bytes")
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
    
    async def send_heartbeat(self, client: BleakClient):
        """Send heartbeat request and wait for response"""
        print("\nSending heartbeat request...")
        
        # Prepare command
        command = bytes([CMD_HEARTBEAT])
        
        # Set up notification handler
        response_received = asyncio.Event()
        self.all_responses = []
        heartbeat_response = None
        
        def notification_handler(sender, data):
            """Handle incoming notification"""
            parsed = self.parse_heartbeat_response(data)
            if parsed:  # Only count valid heartbeat responses
                nonlocal heartbeat_response
                heartbeat_response = parsed
                response_received.set()
            self.all_responses.append(data)
        
        # Subscribe to notifications
        await client.start_notify(CONTROL_CHAR_UUID, notification_handler)
        
        try:
            # Send heartbeat command
            await client.write_gatt_char(CONTROL_CHAR_UUID, command, response=True)
            print("✓ Heartbeat request sent")
            
            # Wait for response (timeout after 3 seconds to catch all packets)
            try:
                await asyncio.wait_for(response_received.wait(), timeout=3.0)
                
                # Display the heartbeat response
                if heartbeat_response:
                    self.display_heartbeat_response(heartbeat_response)
                    print(f"  Received {len(self.all_responses)} response(s)")
                    return True
                else:
                    print("No valid heartbeat response received")
                    return False
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                if self.all_responses:
                    print(f"  Received {len(self.all_responses)} response(s) but no valid heartbeat")
                return False
                
        finally:
            # Unsubscribe from notifications
            await client.stop_notify(CONTROL_CHAR_UUID)
    
    async def run(self):
        """Main test routine"""
        print("\n" + "═" * 50)
        print("HAR SYSTEM - HEARTBEAT TEST")
        print("═" * 50 + "\n")
        
        # Scan for device
        device = await self.scan_for_device()
        if not device:
            return
        
        print(f"\nConnecting to {device.name}...")
        print("─" * 50)
        
        try:
            async with BleakClient(device.address) as client:
                if client.is_connected:
                    print(f"Connected to {device.name}")
                    print(f"  Address: {device.address}\n")
                    # Give Arduino time to send identification packet
                    await asyncio.sleep(0.5)
                    
                    # Send heartbeat
                    success = await self.send_heartbeat(client)
                    
                    if success:
                        # Offer to send more heartbeats
                        while True:
                            print("\nOptions:")
                            print("  [Enter] - Send another heartbeat")
                            print("  [q]     - Quit")
                            
                            choice = input("\nChoice: ").strip().lower()
                            
                            if choice == 'q':
                                break
                            elif choice == '':
                                await self.send_heartbeat(client)
                            else:
                                print("Invalid choice")
                    
                    print("\n✓ Test complete")
                else:
                    print("Failed to connect")
                    
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

async def main():
    tester = HeartbeatTester()
    await tester.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")