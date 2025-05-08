import queue
import threading
import asyncio
import yaml
import struct
from bleak import BleakClient, BleakScanner
import pandas as pd
import os
import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pickle
import sys

# Constants from bleak example
CLASSIFIER_EVENT_TYPES = {
    1: 'ARM_SYNCED', 2: 'ARM_UNSYNCED', 3: 'POSE', 4: 'UNLOCKED',
    5: 'LOCKED', 6: 'SYNC_FAILED', 7: 'UNKNOWN',
}
ARM_VALUES = {0: 'UNKNOWN', 1: 'RIGHT', 2: 'LEFT', 255: 'UNKNOWN'}
POSE_VALUES = {
    0: 'REST', 1: 'FIST', 2: 'WAVE_IN', 3: 'WAVE_OUT',
    4: 'FINGERS_SPREAD', 5: 'DOUBLE_TAP', 255: 'UNKNOWN'
}
XDIRECTION_VALUES = {1: 'TOWARD_WRIST', 2: 'TOWARD_ELBOW', 255: 'UNKNOWN'}
COMMAND = {
    'SET_EMG_IMU_MODE': 1, 'VIBRATE': 3, 'DEEP_SLEEP': 4, 'LED': 6,
    'EXTENDED_VIBRATION': 7, 'SET_SLEEP_MODE': 9, 'UNLOCK': 10, 'USER_ACTION': 11,
}
EMG_MODE = {'OFF': 0, 'FILTERED_50HZ': 1, 'FILTERED': 2, 'RAW': 3}
IMU_MODE = {'OFF': 0, 'SEND_DATA': 1, 'SEND_EVENTS': 2, 'SEND_ALL': 3, 'SEND_RAW': 4}
VIBRATION_DURATION = {'NONE': 0, 'SHORT': 1, 'MEDIUM': 2, 'LONG': 3}
SLEEP_MODE = {'NORMAL': 0, 'NEVER_SLEEP': 1}
UNLOCK_COMMAND = {'UNLOCK_RELOCK': 0, 'UNLOCK_TIMED': 1, 'UNLOCK_HOLD': 2}
CLASSIFIER_MODE = {'DISABLED': 0, 'ENABLED': 1}

# Default Myo services and characteristics (replace with config loading later if needed)
MYO_CONTROL_SERVICE = "d5060001-a904-deb9-4748-2c7f4a124842"
MYO_COMMAND_CHARACTERISTIC = "d5060401-a904-deb9-4748-2c7f4a124842"
MYO_INFO_CHARACTERISTIC = "d5060101-a904-deb9-4748-2c7f4a124842" # Myo info service?
MYO_FIRMWARE_CHARACTERISTIC = "d5060201-a904-deb9-4748-2c7f4a124842" # Firmware revision service?
MYO_IMU_DATA_CHARACTERISTIC = "d5060402-a904-deb9-4748-2c7f4a124842" # IMU service?
MYO_EMG_SERVICE = "d5060005-a904-deb9-4748-2c7f4a124842" # EMG service
MYO_EMG_DATA_0_CHARACTERISTIC = "d5060105-a904-deb9-4748-2c7f4a124842" # EMG data 0 characteristic?
MYO_EMG_DATA_1_CHARACTERISTIC = "d5060205-a904-deb9-4748-2c7f4a124842" # EMG data 1 characteristic?
MYO_EMG_DATA_2_CHARACTERISTIC = "d5060305-a904-deb9-4748-2c7f4a124842" # EMG data 2 characteristic?
MYO_EMG_DATA_3_CHARACTERISTIC = "d5060405-a904-deb9-4748-2c7f4a124842" # EMG data 3 characteristic?
MYO_BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb" # Battery service
MYO_BATTERY_LEVEL_CHARACTERISTIC = "00002a19-0000-1000-8000-00805f9b34fb" # Battery characteristic
MYO_CLASSIFIER_SERVICE = "d5060003-a904-deb9-4748-2c7f4a124842" # Classifier event service?
MYO_CLASSIFIER_EVENT_CHARACTERISTIC = "d5060103-a904-deb9-4748-2c7f4a124842" # Classifier event characteristic?

# Global state variables (consider encapsulating later)
r_addr = None # Will be loaded from config
l_addr = None # Will be loaded from config
min_roll = -30
max_roll = 30
min_pitch = -30
max_pitch = 30
min_yaw = -30
max_yaw = 30
neutral_roll_r = 0
neutral_pitch_r = 0
neutral_yaw_r = 0
neutral_roll_l = 0
neutral_pitch_l = 0
neutral_yaw_l = 0
count = 0
is_recording = False
is_start = False
is_calibrated = False
count_list = [0] * 10
database_file = "database.csv"
record_cache_imu_r = []
record_cache_emg_r = []
record_cache_imu_l = []
record_cache_emg_l = []
last_vals = None
# emg_data = None # Likely need separate L/R EMG stores
preprocess = False
error_str = ""
confirm_prompt = False
latest_quat_r = [1.0, 0.0, 0.0, 0.0] # Initialize to identity quaternion
latest_quat_l = [1.0, 0.0, 0.0, 0.0] # Initialize to identity quaternion
latest_emg_r = None
latest_emg_l = None
latest_adj_ypr_l = [0.0, 0.0, 0.0] # Initialize to neutral orientation

# Queues for communication between BLE threads and main thread
q_data_l = queue.Queue()
q_data_r = queue.Queue()

# Configuration file
CONFIG_FILE = "myo_config.yaml"

def load_config():
    global l_addr, r_addr
    try:
        with open(CONFIG_FILE, "r") as stream:
            config = yaml.safe_load(stream)
            l_addr = config['myo_left_address'] # Expecting MAC address or UUID
            r_addr = config['myo_right_address']
            print(f"Loaded Left Myo Addr: {l_addr}")
            print(f"Loaded Right Myo Addr: {r_addr}")
            return True
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        print("Please create it with 'myo_left_address' and 'myo_right_address'.")
        return False
    except Exception as e:
        print(f"Error reading config file '{CONFIG_FILE}': {e}")
        return False


def cls():
    os.system("cls" if os.name == "nt" else "clear")

# ------------ Myo Setup (Now using Bleak) ---------------

class MyoBleakClient:
    """Handles BLE communication with a Myo armband using Bleak."""
    def __init__(self, address, data_queue, name="Myo"):
        self.address = address
        self.data_queue = data_queue
        self.name = name
        self.client = None
        self.thread = None
        self.loop = None
        self.running = False
        self._current_emg = {} # Store partial EMG data parts {0: [...], 1: [...], ...}
        # Handles to be discovered
        self.imu_handle = None
        self.emg0_handle = None
        self.emg1_handle = None
        self.emg2_handle = None
        self.emg3_handle = None
        self.battery_handle = None
        self.classifier_handle = None
        self._handle_uuid_map = {} # Optional: Map handle back to UUID for easier debugging


    def _notification_callback(self, handle, data):
        """Callback for BLE notifications."""
        timestamp = pygame.time.get_ticks()
        data_type = None
        payload = None

        # Determine data type based on discovered handle
        if handle == self.imu_handle:
            data_type = "IMU"
            values = struct.unpack('<10h', data)
            quat_raw = values[:4]
            acc = values[4:7]
            gyro = values[7:10]
            payload = {'quat': quat_raw, 'acc': acc, 'gyro': gyro}

        elif handle in (self.emg0_handle, self.emg1_handle, self.emg2_handle, self.emg3_handle):
            # Send raw 16-byte EMG data directly
            data_type = "EMG"
            payload = data # Keep payload as raw bytes

        elif handle == self.classifier_handle:
            data_type = "CLASSIFIER"
            event_id, value_id, x_direction_id, _, _, _ = struct.unpack('<6B', data)
            payload = {'event': CLASSIFIER_EVENT_TYPES.get(event_id, 'UNKNOWN'),
                       'value_id': value_id, 'x_dir_id': x_direction_id}

        elif handle == self.battery_handle:
            data_type = "BATTERY"
            payload = int.from_bytes(data, 'little')

        else:
            unknown_uuid = self._handle_uuid_map.get(handle, "Not Discovered")
            print(f"[{self.name}] Unknown/Unhandled Handle: {handle} (UUID: {unknown_uuid}) Data: {data}") # Log unhandled

        # Put data onto the queue (no EMG assembly needed anymore)
        if data_type and payload is not None:
            self.data_queue.put({'source': self.name, 'type': data_type, 'data': payload, 'time': timestamp})

    async def _run_async(self):
        """Asynchronous connection and data handling loop."""
        try:
            print(f"[{self.name}] Attempting to connect to {self.address}...")
            async with BleakClient(self.address) as client:
                self.client = client
                print(f"[{self.name}] Connected: {client.is_connected}")

                # --- Discover Handles ---
                print(f"[{self.name}] Discovering services and characteristics...")
                for service in client.services:
                    for char in service.characteristics:
                        self._handle_uuid_map[char.handle] = char.uuid # Store all handles found
                        if char.uuid == MYO_IMU_DATA_CHARACTERISTIC:
                            self.imu_handle = char.handle
                        elif char.uuid == MYO_EMG_DATA_0_CHARACTERISTIC:
                            self.emg0_handle = char.handle
                        elif char.uuid == MYO_EMG_DATA_1_CHARACTERISTIC:
                            self.emg1_handle = char.handle
                        elif char.uuid == MYO_EMG_DATA_2_CHARACTERISTIC:
                            self.emg2_handle = char.handle
                        elif char.uuid == MYO_EMG_DATA_3_CHARACTERISTIC:
                            self.emg3_handle = char.handle
                        elif char.uuid == MYO_BATTERY_LEVEL_CHARACTERISTIC:
                            self.battery_handle = char.handle
                        elif char.uuid == MYO_CLASSIFIER_EVENT_CHARACTERISTIC:
                            self.classifier_handle = char.handle

                print(f"[{self.name}] Handles mapped: IMU={self.imu_handle}, EMG0-3={self.emg0_handle},{self.emg1_handle},{self.emg2_handle},{self.emg3_handle}, Batt={self.battery_handle}, Classif={self.classifier_handle}")
                # Check if critical handles were found
                required_handles = {
                    'IMU': self.imu_handle, 'EMG0': self.emg0_handle, 'EMG1': self.emg1_handle,
                    'EMG2': self.emg2_handle, 'EMG3': self.emg3_handle, 'Battery': self.battery_handle,
                    'Classifier': self.classifier_handle
                }
                if not all(required_handles.values()):
                    missing = [name for name, handle in required_handles.items() if handle is None]
                    print(f"[{self.name}] Warning: Could not find handles for: {', '.join(missing)}. Some features might not work.")


                # --- Configuration ---
                # Unlock
                await client.write_gatt_char(
                    MYO_COMMAND_CHARACTERISTIC,
                    struct.pack('<3B', COMMAND['UNLOCK'], 1, UNLOCK_COMMAND['UNLOCK_HOLD']),
                    response=True
                )
                # Set sleep mode
                await client.write_gatt_char(
                    MYO_COMMAND_CHARACTERISTIC,
                    struct.pack('<3B', COMMAND['SET_SLEEP_MODE'], 1, SLEEP_MODE['NEVER_SLEEP']),
                    response=True
                )
                # Set EMG/IMU modes (Example: RAW EMG, IMU Data, Classifier Enabled)
                await client.write_gatt_char(
                    MYO_COMMAND_CHARACTERISTIC,
                    struct.pack('<5B', COMMAND['SET_EMG_IMU_MODE'], 3, EMG_MODE['RAW'], IMU_MODE['SEND_DATA'], CLASSIFIER_MODE['DISABLED']),
                    response=True
                )
                print(f"[{self.name}] Configured EMG=RAW, IMU=DATA, Classifier=DISABLED")


                # --- Subscriptions ---
                # Use UUIDs for start_notify, bleak handles the lookup.
                # We store handles mainly for the callback logic.
                try:
                    print(f"[{self.name}] Subscribing to notifications...")
                    if self.battery_handle: await client.start_notify(MYO_BATTERY_LEVEL_CHARACTERISTIC, self._notification_callback)
                    if self.imu_handle: await client.start_notify(MYO_IMU_DATA_CHARACTERISTIC, self._notification_callback)
                    if self.classifier_handle: await client.start_notify(MYO_CLASSIFIER_EVENT_CHARACTERISTIC, self._notification_callback)
                    # EMG (RAW Mode)
                    if self.emg0_handle: await client.start_notify(MYO_EMG_DATA_0_CHARACTERISTIC, self._notification_callback)
                    if self.emg1_handle: await client.start_notify(MYO_EMG_DATA_1_CHARACTERISTIC, self._notification_callback)
                    if self.emg2_handle: await client.start_notify(MYO_EMG_DATA_2_CHARACTERISTIC, self._notification_callback)
                    if self.emg3_handle: await client.start_notify(MYO_EMG_DATA_3_CHARACTERISTIC, self._notification_callback)
                    print(f"[{self.name}] Subscriptions attempted.")
                except Exception as sub_e:
                    print(f"[{self.name}] Error during subscription: {sub_e}")


                # Keep running until stopped
                while self.running:
                    await asyncio.sleep(0.1) # Keep connection alive and process events

        except Exception as e:
            print(f"[{self.name}] Error in BLE connection/loop: {e}")
            self.client = None # Ensure client is None if connection failed/dropped
        finally:
            if self.client and self.client.is_connected:
                 try:
                     # Attempt to stop notifications gracefully using UUIDs
                     print(f"[{self.name}] Stopping notifications...")
                     if self.battery_handle: await client.stop_notify(MYO_BATTERY_LEVEL_CHARACTERISTIC)
                     if self.imu_handle: await client.stop_notify(MYO_IMU_DATA_CHARACTERISTIC)
                     if self.classifier_handle: await client.stop_notify(MYO_CLASSIFIER_EVENT_CHARACTERISTIC)
                     if self.emg0_handle: await client.stop_notify(MYO_EMG_DATA_0_CHARACTERISTIC)
                     if self.emg1_handle: await client.stop_notify(MYO_EMG_DATA_1_CHARACTERISTIC)
                     if self.emg2_handle: await client.stop_notify(MYO_EMG_DATA_2_CHARACTERISTIC)
                     if self.emg3_handle: await client.stop_notify(MYO_EMG_DATA_3_CHARACTERISTIC)
                     print(f"[{self.name}] Notifications stopped.")
                 except Exception as stop_e:
                     print(f"[{self.name}] Error stopping notifications: {stop_e}")
            print(f"[{self.name}] BLE loop finished.")
            self.running = False


    def start(self):
        """Starts the BLE communication thread."""
        if not self.thread or not self.thread.is_alive():
            self.running = True
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._thread_target, daemon=True)
            self.thread.start()
            print(f"[{self.name}] BLE Thread started.")

    def _thread_target(self):
        """Target function for the thread to run the asyncio loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_async())

    def stop(self):
        """Stops the BLE communication thread."""
        if self.running:
            self.running = False
            if self.loop and self.loop.is_running():
                 # Schedule disconnect/cleanup if needed, then stop loop
                 # Note: Direct loop stop might be abrupt. Graceful shutdown is better.
                 # Consider using asyncio.run_coroutine_threadsafe to schedule cleanup
                 self.loop.call_soon_threadsafe(self.loop.stop) # Request loop stop
                 print(f"[{self.name}] Requesting BLE loop stop...")
            if self.thread:
                 self.thread.join(timeout=5.0) # Wait for thread to finish
                 if self.thread.is_alive():
                     print(f"[{self.name}] Warning: BLE thread did not stop gracefully.")
                 else:
                     print(f"[{self.name}] BLE Thread stopped.")
            self.client = None
            self.thread = None
            self.loop = None

    # --- Command Sending Methods (Run in BLE thread's loop) ---
    def _schedule_command(self, coro):
        """Schedules a command coroutine in the BLE thread's loop."""
        if self.running and self.loop:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            print(f"[{self.name}] Cannot send command: Not running or loop missing.")

    async def _vibrate_async(self, duration_type):
        if self.client and self.client.is_connected:
            try:
                await self.client.write_gatt_char(
                    MYO_COMMAND_CHARACTERISTIC,
                    struct.pack('<3B', COMMAND['VIBRATE'], 1, duration_type),
                    response=True
                )
            except Exception as e:
                print(f"[{self.name}] Error sending vibrate command: {e}")

    def vibrate(self, duration_type=VIBRATION_DURATION['SHORT']):
        """Sends a vibrate command."""
        self._schedule_command(self._vibrate_async(duration_type))

    async def _set_leds_async(self, logo_rgb, bar_rgb):
         if self.client and self.client.is_connected:
             try:
                 payload = list(logo_rgb) + list(bar_rgb)
                 await self.client.write_gatt_char(
                     MYO_COMMAND_CHARACTERISTIC,
                     struct.pack('<8B', COMMAND['LED'], 6, *payload),
                     response=True
                 )
             except Exception as e:
                 print(f"[{self.name}] Error sending LED command: {e}")

    def set_leds(self, logo_rgb=(0,0,255), bar_rgb=(0,0,255)):
         """Sets the Myo LEDs."""
         self._schedule_command(self._set_leds_async(logo_rgb, bar_rgb))


# -------- Main Program Loop Adaptation --------

# Placeholder for worker instances
myo_left = None
myo_right = None

def calibrate(quat_r, quat_l):
    global is_calibrated, neutral_roll_r, neutral_pitch_r, neutral_yaw_r, neutral_roll_l, neutral_pitch_l, neutral_yaw_l, is_start
    is_calibrated = True
    is_start = True
    # Use the latest quat values scaled correctly for quat_to_ypr
    neutral_roll_r, neutral_pitch_r, neutral_yaw_r = quat_to_ypr(quat_r)
    neutral_roll_l, neutral_pitch_l, neutral_yaw_l = quat_to_ypr(quat_l)
    print(neutral_roll_r, neutral_pitch_r, neutral_yaw_r)
    print(neutral_roll_l, neutral_pitch_l, neutral_yaw_l)
    print("Calibration complete using current BLE orientation.")
    # Optional: Vibrate Myos on calibration
    if myo_left: myo_left.vibrate()
    if myo_right: myo_right.vibrate()


def start_recording():
    global is_recording, is_start
    if is_calibrated:
        is_recording = True
        is_start = True
        print("Recording started")
    else:
        print("Please calibrate before starting")


def pause_recording():
    global is_start
    is_start = False
    print("Recording paused")


def save_data():
    global record_cache_emg_r, record_cache_imu_r, data, name, gesture, pre_add_rep, database_file, record_cache_emg_l, record_cache_imu_l

    # Save EMG data
    if record_cache_emg_r:
        emg_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_emg_rec_r.p"
        with open(emg_file_path, "wb") as f:
            # Store raw BLE EMG data format (list of 16 bytes per notification)
            pickle.dump(record_cache_emg_r, f)
        print(f"EMG data (Right) saved to {emg_file_path}")

    if record_cache_imu_r:
        imu_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_imu_rec_r.p"
        with open(imu_file_path, "wb") as f:
             # Store raw BLE IMU data format
            pickle.dump(record_cache_imu_r, f)
        print(f"IMU data (Right) saved to {imu_file_path}")

    # Save IMU data (Left)
    if record_cache_emg_l:
        emg_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_emg_rec_l.p"
        with open(emg_file_path, "wb") as f:
            pickle.dump(record_cache_emg_l, f)
        print(f"EMG data (Left) saved to {emg_file_path}")

    if record_cache_imu_l:
        imu_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_imu_rec_l.p"
        with open(imu_file_path, "wb") as f:
            pickle.dump(record_cache_imu_l, f)
        print(f"IMU data (Left) saved to {imu_file_path}")

    # Update repetition count in the dataset and save
    # Ensure data is loaded correctly before modification
    try:
        # Re-read data in case it was modified externally or needs refresh
        data = pd.read_csv(database_file)
        data.loc[(data.name == name) & (data.gesture == gesture), "repetition"] = (
            pre_add_rep
        )
        data.to_csv(database_file, index=False)
        print(f"Database updated and saved to {database_file}")
    except Exception as e:
        print(f"Error updating database: {e}")


    # Reset caches after saving
    record_cache_emg_r = []
    record_cache_imu_r = []
    record_cache_emg_l = []
    record_cache_imu_l = []
    print("Caches cleared after saving")


def save_and_exit():
    global record_cache_emg_r, record_cache_imu_r, pre_add_rep, record_cache_emg_l, record_cache_imu_l
    if (
        record_cache_imu_r
        or record_cache_emg_r
        or record_cache_imu_l
        or record_cache_emg_l
    ):
        pre_add_rep += 1
        save_data()
        print("Data saved.")
    else:
        print("No new data to save.")
    print("Exiting.")
    raise KeyboardInterrupt() # Use standard exit mechanism


def next_repetition():
    global pre_add_rep
    if (
        record_cache_imu_r
        or record_cache_emg_r
        or record_cache_imu_l
        or record_cache_emg_l
    ):
        pre_add_rep += 1
        save_data()
        print("Moving to next repetition")
        # Optional: Vibrate Myos
        if myo_left: myo_left.vibrate(VIBRATION_DURATION['SHORT'])
        if myo_right: myo_right.vibrate(VIBRATION_DURATION['SHORT'])

    else:
        print("No data recorded for this repetition.")


def erase_calibration():
    global neutral_roll_r, neutral_pitch_r, neutral_yaw_r, neutral_pitch_l, neutral_roll_l, neutral_yaw_l
    neutral_roll_r = neutral_pitch_r = neutral_yaw_r = 0
    neutral_pitch_l = neutral_roll_l = neutral_yaw_l = 0
    print("Calibration erased")


def go_back():
    global is_recording, is_calibrated, record_cache_emg_r, record_cache_imu_r, pre_add_rep, record_cache_emg_l, record_cache_imu_l, is_start
    if is_start or is_recording:
        pre_add_rep -= 1
        is_start = False
        is_recording = False
        record_cache_emg_r = []
        record_cache_imu_r = []
        record_cache_emg_l = []
        record_cache_imu_l = []
        print("All actions canceled and parameters reset.")
    else:
        print("No active session to go back from.")


def handle_event(event):
    global confirm_prompt, pre_add_rep # Need latest quats from global scope now
    # Map keys to actions
    key_actions = {
        pygame.K_c: lambda: calibrate(latest_quat_r, latest_quat_l),
        pygame.K_s: start_recording,
        pygame.K_p: pause_recording,
        pygame.K_q: save_and_exit,
        pygame.K_n: next_repetition,
        pygame.K_e: erase_calibration,
        pygame.K_b: go_back,
        pygame.K_y: None # Handled separately for confirm prompt
    }

    if confirm_prompt:
        if event.key == pygame.K_y:
            # Confirm action (e.g., revert)
            go_back() # Assuming go_back handles the revert logic
            print("Revert confirmed.")
        else:
            print("Revert cancelled.")
        confirm_prompt = False # Reset prompt state regardless of key
    elif event.key in key_actions:
        action = key_actions[event.key]
        if action:
            action()
    else:
        print(f"Invalid key pressed: {pygame.key.name(event.key)}")


def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0 * width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def draw(w, nx, ny, nz): # w=1 for right arm, assume 0 for left if needed
    global error_str, confirm_prompt, is_calibrated, is_start, is_recording, pre_add_rep, gesture, name, latest_adj_ypr_l
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0) # Move object back

    # --- Text Display ---
    drawText((-2.6, 1.8, 2), f"{name} {gesture} {pre_add_rep+1}", 18)

    if error_str:
        drawTextwithColor((-2.6, -2.2, 2), error_str, 16, color=(255, 0, 0))
        error_str = "" # Clear error after displaying once

    if confirm_prompt:
         drawTextwithColor(
             (-2.6, -2, 2),
             f"REVERT BACK to {pre_add_rep}? press 'y' to confirm, any other key to abort",
             16, color=(255, 100, 0) # Orange for confirmation
         )
    else:
         # Standard instructions
         if is_calibrated:
             drawText((-2.6, -1.8, 2), "'s': start, 'p': pause, 'n': next rep, 'q': quit", 16)
             drawText((-2.6, -2.0, 2), "'b': back/cancel rep, 'e': erase calib", 16)
             if is_start:
                 state_text = "RECORDING ('p' to pause)" if is_recording else "RESTING ('s' to record)"
                 drawTextwithColor((-2.6, 1.6, 2), state_text, 18, color=(0, 255, 0) if is_recording else (255, 255, 0))
                 # Optionally show rep number being recorded
                 # drawText((-2.6, 1.4, 2), f"Rep: {pre_add_rep + 1}", 16)
             else: # Paused
                  drawTextwithColor((-2.6, 1.6, 2), "PAUSED ('s' to resume)", 18, color=(255, 165, 0)) # Orange for paused
         else: # Not calibrated
             drawTextwithColor((-2.6, 1.6, 2), "NOT CALIBRATED", 18, color=(255, 0, 0))
             drawText((-2.6, -1.8, 2), "Place arms in neutral pose and press 'c'", 16)


    # --- 3D Cube Visualization (Using Right Arm IMU) ---
    # nx, ny, nz should be adjusted yaw, pitch, roll from the right arm
    yaw, pitch, roll = nx, ny, nz
    drawText((-2.6, -1.6, 2), f"R Arm Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}", 16)

    # Display Left Arm YPR
    lyaw, lpitch, lroll = latest_adj_ypr_l
    drawText((-2.6, -1.4, 2), f"L Arm Yaw: {lyaw:.1f}, Pitch: {lpitch:.1f}, Roll: {lroll:.1f}", 16)

    # Apply rotations (adjust order if needed: ZXY is common for Euler angles)
    # Order: Roll around Z, Pitch around X, Yaw around Y
    glRotatef(-roll, 0.0, 0.0, 1.0)  # Roll around Z axis (blue axis in OpenGL standard view)
    glRotatef(pitch, 1.0, 0.0, 0.0) # Pitch around X axis (red axis)
    glRotatef(yaw, 0.0, 1.0, 0.0)   # Yaw around Y axis (green axis)


    # Draw the cube
    glBegin(GL_QUADS)
    # Front Face (positive Z) - Red
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f( 1.0,  0.2,  1.0)
    glVertex3f(-1.0,  0.2,  1.0)
    glVertex3f(-1.0, -0.2,  1.0)
    glVertex3f( 1.0, -0.2,  1.0)
    # Back Face (negative Z) - Yellow
    glColor3f(1.0, 1.0, 0.0)
    glVertex3f( 1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0,  0.2, -1.0)
    glVertex3f( 1.0,  0.2, -1.0)
    # Top Face (positive Y) - Green
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f( 1.0,  0.2, -1.0)
    glVertex3f(-1.0,  0.2, -1.0)
    glVertex3f(-1.0,  0.2,  1.0)
    glVertex3f( 1.0,  0.2,  1.0)
    # Bottom Face (negative Y) - Orange
    glColor3f(1.0, 0.5, 0.0)
    glVertex3f( 1.0, -0.2,  1.0)
    glVertex3f(-1.0, -0.2,  1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f( 1.0, -0.2, -1.0)
    # Right face (positive X) - Magenta
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f( 1.0,  0.2, -1.0)
    glVertex3f( 1.0,  0.2,  1.0)
    glVertex3f( 1.0, -0.2,  1.0)
    glVertex3f( 1.0, -0.2, -1.0)
    # Left face (negative X) - Blue
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0,  0.2,  1.0)
    glVertex3f(-1.0,  0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2,  1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(
        textSurface.get_width(),
        textSurface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        textData,
    )


def drawTextwithColor(position, textString, size, color):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, color, (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(
        textSurface.get_width(),
        textSurface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        textData,
    )


def quat_to_ypr(q_raw):
    # Assuming q_raw is list/tuple of 4 integers from bleak unpack
    # Scale factor used in some Myo implementations
    scale = 16384.0 # (2^14)
    if not all(isinstance(x, (int, float)) for x in q_raw) or len(q_raw) != 4:
        print(f"Warning: Invalid quaternion data received: {q_raw}")
        return [0, 0, 0] # Return neutral orientation on error

    q = [x / scale for x in q_raw]
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3] # Assuming WXYZ order from Myo

    # Calculate Yaw, Pitch, Roll from Quaternion (standard formulas)
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert radians to degrees
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    # Optional: Apply declination correction if needed
    # yaw_deg -= 0 # Declination adjustment (example)

    return [yaw_deg, pitch_deg, roll_deg]


def keep_domain(angle):
    # Normalize angle to -180 to 180 range
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


# check_is_recording_moe uses adjusted roll/pitch
def check_is_recording_moe(adjusted_roll, adjusted_pitch, adjusted_yaw):
    global count_list
    if (
        
        min_pitch < adjusted_pitch
        and adjusted_pitch < max_pitch
    ):
        count_list = count_list[1:]
        count_list.append(1)
    else:
        count_list = count_list[1:]
        count_list.append(-1)
    if sum(count_list) >= 0:
        return False
    else:
        return True


def name_prompt(data):
    while True:
        print("please choose your name")
        for i in range(len(data["name"].unique())):
            print(str(i) + ": " + data["name"].unique()[i])
        name_num = input("type in your name number : ")
        print("\n---------------------------------\n")
        try:
            name_num = int(name_num)
            if 0 <= name_num <= len(data["name"].unique()) - 1:
                return data["name"].unique()[name_num]
            else:
                print("Number is not within the range")

        except ValueError:
            print("Invalid value")


def gesture_prompt(data, name):
    while True:
        print("please choose gesture")
        for i in range(len(data[data.name == name]["gesture"])):
            print(str(i) + ": " + data[data.name == name]["gesture"].iloc[i])
        gesture_num = input("type in gesture number: ")
        print("\n---------------------------------\n")
        try:
            gesture_num = int(gesture_num)
            if 0 <= int(gesture_num) <= len(data[data.name == name]["gesture"]) - 1:
                return data[data.name == name]["gesture"].iloc[gesture_num]
            else:
                print("Number is not within the range")

        except ValueError:
            print("Invalid value")


def mode_prompt():
    while True:
        print("select mode")
        print("0: recording continuous\n1: record only once\n2: add gesture name")
        mode = input("type in mode number: ")
        print("\n---------------------------------\n")
        try:
            mode = int(mode)
            if 0 <= int(mode) <= 2:
                return mode
            else:
                print("Number is not within the range")

        except ValueError:
            print("Invalid value")


def add_new_gesture(data_df, db_file, user_name, max_retries=3):
    if max_retries <= 0:
        print("Maximum retries reached. Exiting gesture addition.")
        return data_df # Return the potentially unmodified DataFrame

    new_gesture_name = input(f"Enter name for the new gesture for '{user_name}': ")
    new_gesture_name = new_gesture_name.strip()

    if not new_gesture_name:
        print("Gesture name cannot be empty.")
        return add_new_gesture(data_df, db_file, user_name, max_retries - 1)

    # Check if gesture already exists for this user
    existing_gestures = data_df[data_df.name == user_name]['gesture'].tolist()
    if new_gesture_name in existing_gestures:
        print(f"Gesture '{new_gesture_name}' already exists for {user_name}.")
        # Optionally ask if they want to retry or exit
        retry = input("Try a different name? (y/n): ").lower()
        if retry == 'y':
            return add_new_gesture(data_df, db_file, user_name, max_retries - 1)
        else:
            return data_df # Return original DataFrame

    # Add the new gesture
    new_row = pd.DataFrame([{'name': user_name, 'gesture': new_gesture_name, 'repetition': 0}])
    # Use concat instead of append (which is deprecated)
    data_df = pd.concat([data_df, new_row], ignore_index=True)

    try:
        data_df.to_csv(db_file, index=False)
        print(f"Added gesture '{new_gesture_name}' for {user_name} and updated {db_file}")
        return data_df # Return the updated DataFrame
    except Exception as e:
        print(f"Error saving updated database to {db_file}: {e}")
        # Consider how to handle save failure - maybe revert the add?
        # For now, just print error and return the df with the added row in memory.
        return data_df


# Data storage formatting functions - adapt to BLE data structure
def mutate_to_data_for_store_emg(emg_payload, rec_status, timestamp):
    # Store 8-byte EMG list, recording status, and timestamp
    # print(emg_payload) # Optional debug print
    return [list(emg_payload), rec_status, timestamp]


def mutate_to_data_for_store_imu(imu_payload, adj_ypr, neut_ypr, rec_status, timestamp):
    # Store raw IMU dict {'quat':..., 'acc':..., 'gyro':...}, adjusted YPR, neutral YPR, status, time
    return [list(imu_payload), list(adj_ypr), list(neut_ypr), rec_status, timestamp]


# Adjusted YPR calculation using raw quat
def quat_to_adjusted_ypr(quat_raw, neutral_yaw, neutral_pitch, neutral_roll):
    try:
        yaw, pitch, roll = quat_to_ypr(quat_raw) # Handles scaling and conversion
        adjusted_yaw = keep_domain(yaw - neutral_yaw)
        adjusted_pitch = keep_domain(pitch - neutral_pitch)
        adjusted_roll = keep_domain(roll - neutral_roll)
        return [adjusted_yaw, adjusted_pitch, adjusted_roll]
    except Exception as e: # Catch potential errors in quat_to_ypr
        # print(f"Error calculating adjusted YPR: {e}")
        return [0.0, 0.0, 0.0]


# Process and store IMU data
def process_imu(record_cache, source_name, imu_payload, adj_ypr, neut_ypr, rec_status, timestamp):
     record_cache.append(
         mutate_to_data_for_store_imu(
             (imu_payload["quat"], imu_payload["acc"], imu_payload["gyro"]), adj_ypr, neut_ypr, rec_status, timestamp
         )
     )

# Process and store EMG data (now expecting 8-byte payload)
def process_emg(record_cache, source_name, emg_payload_8byte, rec_status, timestamp):
     record_cache.append(
         mutate_to_data_for_store_emg(
             emg_payload_8byte, rec_status, timestamp
         )
     )


if __name__ == "__main__":
    cls()
    print("Starting Data Collector (BLE Version)")

    if not load_config():
        sys.exit(1) # Exit if config fails

    if not os.path.exists("examples/data"):
        os.makedirs("examples/data")
        print("Created 'examples/data' directory for saving files.")

    try:
        data = pd.read_csv(database_file)
    except FileNotFoundError:
        print(f"Database file '{database_file}' not found. Creating a new one.")
        # Create a default empty DataFrame structure
        data = pd.DataFrame(columns=['name', 'gesture', 'repetition'])
        # Optionally add a default user/gesture or leave empty
        # data = data.append({'name': 'default_user', 'gesture': 'default_gesture', 'repetition': 0}, ignore_index=True)
        data.to_csv(database_file, index=False) # Save the new empty file
    except Exception as e:
        print(f"Error loading database file '{database_file}': {e}")
        sys.exit(1)


    name = name_prompt(data)
    mode = mode_prompt()
    if mode == 2:
        # Pass the current DataFrame and get the potentially updated one back
        data = add_new_gesture(data, database_file, name, 3)
        # No need to reload from CSV here, data DataFrame is updated in memory
    gesture = gesture_prompt(data, name)
    pre_add_rep = int(
        data.loc[(data.name == name) & (data.gesture == gesture), "repetition"].iloc[0]
    )

    error_str = ""
    confirm_prompt = False

    # Initialize Pygame and OpenGL
    pygame.init()
    video_flags = OPENGL | DOUBLEBUF
    screen = pygame.display.set_mode((640, 480), video_flags)
    pygame.display.set_caption(f"BLE Data Collector - {name} - {gesture}")
    resizewin(640, 480)
    init()

    # Initialize and start BLE clients
    myo_left = MyoBleakClient(l_addr, q_data_l, name="Left")
    myo_right = MyoBleakClient(r_addr, q_data_r, name="Right")
    myo_left.start()
    myo_right.start()

    # Give threads time to start and connect
    print("Waiting for Myo connections...")
    pygame.time.wait(5000) # Wait 5 seconds

    if not (myo_left.client and myo_left.client.is_connected):
         print("Warning: Left Myo failed to connect.")
         # Decide how to handle: exit, continue with one, etc.
    if not (myo_right.client and myo_right.client.is_connected):
         print("Warning: Right Myo failed to connect.")


    frames = 0
    ticks = pygame.time.get_ticks()
    running = True

    try:
        while running:
            # --- Process BLE Data Queues ---
            # Process all available data from both queues without blocking
            processed_imu_r_this_frame = False # Flag to update visualization only once per frame
            while not q_data_l.empty() or not q_data_r.empty():
                 try:
                     # Try getting from right queue first for visualization preference
                     data_item = None
                     if not q_data_r.empty():
                         data_item = q_data_r.get_nowait()
                     elif not q_data_l.empty():
                          data_item = q_data_l.get_nowait()

                     if data_item:
                         source = data_item['source']
                         dtype = data_item['type']
                         payload = data_item['data']
                         timestamp = data_item['time']

                         if dtype == "IMU":
                             quat_raw = payload['quat'] # Raw integers
                             # Store latest raw quat
                             if source == "Right":
                                 latest_quat_r = quat_raw
                                 cache = record_cache_imu_r
                                 adjusted_ypr = quat_to_adjusted_ypr(quat_raw, neutral_yaw_r, neutral_pitch_r, neutral_roll_r)
                             else: # Left
                                 latest_quat_l = quat_raw
                                 cache = record_cache_imu_l
                                 latest_adj_ypr_l = quat_to_adjusted_ypr(quat_raw, neutral_yaw_l, neutral_pitch_l, neutral_roll_l)

                             # Calculate adjusted YPR
                             

                             # Check recording status based *only* on the right arm's movement for now
                             if source == "Right" and is_calibrated:
                                  is_recording = check_is_recording_moe(adjusted_ypr[2], adjusted_ypr[1], adjusted_ypr[0])
                                  # Update visualization only with the latest right arm data
                                  if not processed_imu_r_this_frame:
                                     draw(1, *adjusted_ypr) # Draw using right arm adjusted YPR
                                     pygame.display.flip()
                                     processed_imu_r_this_frame = True


                             # Store data if session started
                             if is_start:
                                 neutral_vector = [neutral_yaw_r, neutral_pitch_r, neutral_roll_r]
                                 process_imu(cache, source, payload, adjusted_ypr, neutral_vector, is_recording, timestamp)

                         elif dtype == "EMG":
                              # Payload is 16 raw bytes
                              emg_unpacked = struct.unpack('<16b', payload) # Unpack into 16 signed bytes
                              emg_part1 = list(emg_unpacked[:8])
                              emg_part2 = list(emg_unpacked[8:])

                              # Store latest raw EMG (store the full unpacked 16 bytes for now)
                              if source == "Right":
                                  latest_emg_r = list(emg_unpacked) # List of 16 bytes
                                  cache = record_cache_emg_r
                              else: # Left
                                  latest_emg_l = list(emg_unpacked) # List of 16 bytes
                                  cache = record_cache_emg_l

                              # Store data if session started - process each 8-byte part
                              if is_start:
                                  # Process first 8 bytes
                                  process_emg(cache, source, emg_part1, is_recording, timestamp)
                                  # Process second 8 bytes (adjust timestamp slightly or use same?)
                                  # Using same timestamp for now as they arrived in the same packet.
                                  process_emg(cache, source, emg_part2, is_recording, timestamp)


                         elif dtype == "CLASSIFIER":
                             # Handle classifier events (e.g., print, trigger actions)
                             event = payload['event']
                             value_id = payload['value_id']
                             print(f"[{source}] Classifier: {event} ({value_id})")
                             # Example: Trigger vibrate on FIST
                             if event == 'POSE' and POSE_VALUES.get(value_id) == 'FIST':
                                 if source == "Right" and myo_right: myo_right.vibrate()
                                 if source == "Left" and myo_left: myo_left.vibrate()


                         elif dtype == "BATTERY":
                             print(f"[{source}] Battery Level: {payload}%")

                 except queue.Empty:
                     pass # Should not happen with check, but good practice
                 except Exception as e:
                      print(f"Error processing data queue item: {e}")
                      print(f"Problematic item: {data_item}")


            # --- Handle Pygame Events ---
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    handle_event(event) # Pass the full event object
                # Handle other events like window resize if needed


            # --- Frame Rate Control ---
            # If no IMU data was processed to trigger a draw/flip, update display here
            if not processed_imu_r_this_frame:
                # Redraw with last known orientation if needed, or just flip
                # For simplicity, let's just ensure display updates periodically
                # If using latest_quat_r, recalculate adjusted YPR before drawing
                adj_ypr_r = quat_to_adjusted_ypr(latest_quat_r, neutral_yaw_r, neutral_pitch_r, neutral_roll_r)
                draw(1, *adj_ypr_r)
                pygame.display.flip()


            frames += 1
            if frames % 100 == 0: # Print FPS occasionally
                now = pygame.time.get_ticks()
                if now - ticks > 1000:
                    fps = frames / ((now - ticks) / 1000.0)
                    # print(f"FPS: {fps:.2f}")
                    ticks = now
                    frames = 0

            pygame.time.wait(10) # Small delay to prevent high CPU usage


    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")
    finally:
        # --- Cleanup ---
        print("Stopping BLE threads...")
        if myo_left:
            myo_left.stop()
        if myo_right:
            myo_right.stop()

        print("Quitting Pygame...")
        pygame.quit()
        print("Exiting.")
        sys.exit()
