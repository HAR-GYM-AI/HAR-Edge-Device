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
        if len(data) < 6:
            print(f"Invalid response length: {len(data)} bytes")
            return None
        
        response_id = data[0]
        node_id = data[1]
        is_running = bool(data[2])
        is_calibrated = bool(data[3])
        timestamp = (data[4] << 8) | data[5]  # Combine two bytes
        
        return {
            'response_id': response_id,
            'node_id': node_id,
            'node_name': NODE_NAMES.get(node_id, "UNKNOWN"),
            'is_running': is_running,
            'is_calibrated': is_calibrated,
            'timestamp': timestamp,
            'timestamp_ms': timestamp * 10  # Convert to milliseconds
        }
    
    def display_heartbeat_response(self, response: dict):
        """Display formatted heartbeat response"""
        print("\n" + "═" * 50)
        print("HEARTBEAT RESPONSE")
        print("═" * 50)
        
        # Status indicators
        running_icon = "yes" if response['is_running'] else "no"
        cal_icon = "yes" if response['is_calibrated'] else "no"
        
        print(f"\nNode Information:")
        print(f"  ID:           {response['node_id']} ({response['node_name']})")
        print(f"  Running:      {running_icon} {response['is_running']}")
        print(f"  Calibrated:   {cal_icon} {response['is_calibrated']}")
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
        response_data = []
        
        def notification_handler(sender, data):
            """Handle incoming notification"""
            response_data.append(data)
            response_received.set()
        
        # Subscribe to notifications
        await client.start_notify(CONTROL_CHAR_UUID, notification_handler)
        
        try:
            # Send heartbeat command
            await client.write_gatt_char(CONTROL_CHAR_UUID, command)
            print("✓ Heartbeat request sent")
            
            # Wait for response (timeout after 2 seconds)
            try:
                await asyncio.wait_for(response_received.wait(), timeout=2.0)
                
                # Parse and display response
                if response_data:
                    response = self.parse_heartbeat_response(response_data[0])
                    if response:
                        self.display_heartbeat_response(response)
                        return True
                    else:
                        print("Failed to parse response")
                        return False
                else:
                    print("No response received")
                    return False
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
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
                    print(f"✓ Connected to {device.name}")
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