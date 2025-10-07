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
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

# CONSTANTS & ENUMS
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

# BLE Command IDs (matching Arduino firmware)
CMD_START = 0x01
CMD_STOP = 0x02
CMD_CALIBRATE = 0x03
CMD_SET_EXERCISE = 0x04
CMD_RESET = 0x05
CMD_TIME_SYNC = 0x06
CMD_TOGGLE_WINDOWING = 0x07
CMD_HEARTBEAT = 0xFF

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


# MENU SYSTEM
class DataCollectionMenu:
    """Interactive menu for HAR data collection"""
    
    def __init__(self):
        self.running = True
        self.recording_config: Optional[RecordingConfig] = None
        self.node_status: Optional[NodeStatus] = None
        
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
        """Display current node status if available"""
        if self.node_status and self.node_status.connected:
            status_color = "\033[92m" if self.node_status.is_running else "\033[93m"
            reset_color = "\033[0m"
            
            print("┌─ NODE STATUS " + "─" * 54 + "┐")
            print(f"│ Placement:     {self.node_status.node_id.name:<52}│")
            print(f"│ Connected:     {status_color}{'YES' if self.node_status.connected else 'NO'}{reset_color:<52}│")
            print(f"│ Running:       {status_color}{'YES' if self.node_status.is_running else 'NO'}{reset_color:<52}│")
            print(f"│ Calibrated:    {'YES' if self.node_status.is_calibrated else 'NO':<52}│")
            print(f"│ Timestamp:     {self.node_status.timestamp:<52}│")
            print("└" + "─" * 68 + "┘")
            print()
        else:
            print("┌─ NODE STATUS " + "─" * 54 + "┐")
            print("│ No node connected                                                  │")
            print("└" + "─" * 68 + "┘")
            print()
    
    def print_menu_options(self):
        """Display menu options"""
        print("MAIN MENU:")
        print()
        print("  [1] Start Recording Session")
        print("  [2] Calibrate Sensor Node")
        print("  [3] Request Node Status (Heartbeat)")
        print("  [4] Toggle Windowing Mode")
        print("  [5] View Current Configuration")
        print("  [6] Reset Sensor Node")
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
    
    async def run(self):
        """Main menu loop"""
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
                    # Toggle windowing mode
                    await self.handle_toggle_windowing()
                    
                elif choice == '5':
                    # View configuration
                    self.show_current_config()
                    
                elif choice == '6':
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
    
    async def handle_start_recording(self):
        """Handle recording session start"""
        print("\n" + "=" * 70)
        print("START RECORDING SESSION")
        print("=" * 70)
        
        # Get configuration
        config = self.get_recording_config()
        if config is None:
            input("\nPress Enter to continue...")
            return
        
        self.recording_config = config
        
        # TODO: Send start command to Arduino
        # TODO: Begin data collection
        # TODO: Monitor for completion
        
        print("\n✓ Configuration saved (recording not yet implemented)")
        input("Press Enter to continue...")
    
    async def handle_calibration(self):
        """Handle sensor calibration"""
        print("\n" + "=" * 70)
        print("SENSOR CALIBRATION")
        print("=" * 70)
        print("\nKeep sensor stationary during calibration!")
        
        confirm = input("\nProceed with calibration? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Calibration cancelled")
            input("\nPress Enter to continue...")
            return
        
        # TODO: Send calibration command to Arduino
        # TODO: Wait for calibration completion
        
        print("\n✓ Calibration initiated (not yet implemented)")
        input("Press Enter to continue...")
    
    async def handle_heartbeat(self):
        """Request and display node status"""
        print("\n" + "=" * 70)
        print("NODE STATUS REQUEST")
        print("=" * 70)
        
        # TODO: Send heartbeat command to Arduino
        # TODO: Parse and display response
        
        print("\nHeartbeat request (not yet implemented)")
        input("\nPress Enter to continue...")
    
    async def handle_toggle_windowing(self):
        """Toggle windowing mode on sensor"""
        print("\n" + "=" * 70)
        print("TOGGLE WINDOWING MODE")
        print("=" * 70)
        
        confirm = input("\nToggle windowing mode? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled")
            input("\nPress Enter to continue...")
            return
        
        # TODO: Send toggle windowing command
        
        print("\n✓ Windowing toggle command sent (not yet implemented)")
        input("Press Enter to continue...")
    
    async def handle_reset(self):
        """Reset sensor node"""
        print("\n" + "=" * 70)
        print("RESET SENSOR NODE")
        print("=" * 70)
        print("\nThis will stop any active recording and clear calibration!")
        
        confirm = input("\nProceed with reset? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Reset cancelled")
            input("\nPress Enter to continue...")
            return
        
        # TODO: Send reset command to Arduino
        
        print("\n✓ Reset command sent (not yet implemented)")
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