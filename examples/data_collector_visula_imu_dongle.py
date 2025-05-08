import multiprocessing
import pandas as pd
import os
import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pickle
from serial.tools.list_ports import comports
import re
import sys
import os
from pyomyo import Myo, emg_mode

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# from src.pyomyo import Myo, emg_mode

r_addr = [233, 28, 231, 168, 151, 205]
l_addr = [48, 12, 163, 25, 55, 236]
min_roll = -30
max_roll = 30
min_pitch = -30
max_pitch = 30
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
emg_data = None
preprocess = False


def cls():
    # Clear the screen in a cross platform way
    # https://stackoverflow.com/questicoons/517970/how-to-clear-the-interpreter-console
    os.system("cls" if os.name == "nt" else "clear")


# ------------ Myo Setup ---------------
q_l = multiprocessing.Queue()
q_r = multiprocessing.Queue()


def detect_tty():
    valid_ttys = []
    for p in comports():
        if re.search(r"PID=2458:0*1", p[2]):
            valid_ttys.append(p[0])
    assert len(valid_ttys) == 2, f"Expected 2 Myo devices, found {len(valid_ttys)}"
    return valid_ttys


def myo_worker(myo_queue, tty, color, mac_address):
    """Worker function for handling Myo data collection."""
    myo_device = Myo(tty=tty, mode=emg_mode.RAW)
    myo_device.connect(addr=mac_address)

    def emg_handler(emg, movement):
        myo_queue.put(emg)

    myo_device.add_emg_handler(emg_handler)

    def imu_handler(quat, acc, gyro):
        imu_data = [quat, acc, gyro]
        myo_queue.put(imu_data)

    myo_device.add_imu_handler(imu_handler)

    myo_device.set_leds(color, color)

    while True:
        myo_device.run()


# -------- Main Program Loop -----------


def calibrate(quat_r, quat_l):
    global is_calibrated, neutral_roll_r, neutral_pitch_r, neutral_yaw_r, neutral_roll_l, neutral_pitch_l, neutral_yaw_l, is_start
    is_calibrated = True
    is_start = True
    neutral_roll_r, neutral_pitch_r, neutral_yaw_r = quat_to_ypr(quat_r)
    neutral_roll_l, neutral_pitch_l, neutral_yaw_l = quat_to_ypr(quat_l)


def start_recording():
    global is_recording, is_start
    if is_calibrated:
        is_recording = True
        is_start = True
        print("Recording started")
    else:
        print("Please calibrate before starting")


def pause_recording():
    is_start = False
    print("Recording paused")


def save_data():
    global record_cache_emg_r, record_cache_imu_r, data, name, gesture, pre_add_rep, database_file, record_cache_emg_l, record_cache_imu_l

    # Save EMG data
    if record_cache_emg_r:
        emg_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_emg_rec_r.p"
        with open(emg_file_path, "wb") as f:
            pickle.dump(record_cache_emg_r, f)
        print(f"EMG data saved to {emg_file_path}")

    if record_cache_imu_r:
        imu_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_imu_rec_r.p"
        with open(imu_file_path, "wb") as f:
            pickle.dump(record_cache_imu_r, f)
        print(f"IMU data saved to {imu_file_path}")

    # Save IMU data
    if record_cache_emg_l:
        emg_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_emg_rec_l.p"
        with open(emg_file_path, "wb") as f:
            pickle.dump(record_cache_emg_l, f)
        print(f"EMG data saved to {emg_file_path}")

    if record_cache_imu_l:
        imu_file_path = f"examples/data/{name}_{gesture}_{pre_add_rep}_imu_rec_l.p"
        with open(imu_file_path, "wb") as f:
            pickle.dump(record_cache_imu_l, f)
        print(f"IMU data saved to {imu_file_path}")

    # Update repetition count in the dataset and save
    data.loc[(data.name == name) & (data.gesture == gesture), "repetition"] = (
        pre_add_rep
    )
    data.to_csv(database_file, index=False)
    print(f"Database updated and saved to {database_file}")

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
        print("Data saved and exiting")
        raise KeyboardInterrupt()
    else:
        print("No data to save")


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
    else:
        print("No data to save")


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
        print("All actions canceled and parameters reset.")
    else:
        print("No active session to go back from.")


key_actions = set(["c", "s", "p", "q", "n", "e", "b"])


def handle_event(event, quat_r, quat_l):
    key_actions = {
        "c": lambda: calibrate(quat_r, quat_l),
        "s": start_recording,
        "p": pause_recording,
        "q": save_and_exit,
        "n": next_repetition,
        "e": erase_calibration,
        "b": go_back,  # Example additional key handling
    }
    action = key_actions.get(event.unicode)
    if action:
        action()
    else:
        print("Invalid key or no action assigned")


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


def draw(w, nx, ny, nz):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    drawText((-2.6, 1.8, 2), f"{name} {gesture} {pre_add_rep}", 18)
    if error_str != "":
        drawTextwithColor(
            (-2.6, -2, 2),
            error_str + ", Press any key to clear error",
            16,
            color=(255, 0, 0),
        )
    else:
        if is_calibrated:
            if confirm_prompt:
                drawTextwithColor(
                    (-2.6, -2, 2),
                    f"REVERT BACK to {pre_add_rep-1} CONFIRM? press 'y' to confirm any to abort",
                    16,
                    color=(255, 0, 0),
                )
            else:
                drawText(
                    (-2.6, -1.8, 2),
                    "'s' to start, 'p' to pause, 'q' to save and exit ",
                    16,
                )
                drawText(
                    (-2.6, -2, 2), "'n' to save and next, 'b' to drop this rep", 16
                )
            if is_start:
                if is_recording:
                    drawText((-2.6, 1.6, 2), "RECORDING 'p' to pause", 18)
                    drawTextwithColor(
                        (-2.6, 1.4, 2),
                        f"RECORDING {gesture} {pre_add_rep+1}",
                        16,
                        color=(0, 255, 0),
                    )
                else:
                    drawText((-2.6, 1.6, 2), "REST 'p' to pause", 18)
                    drawTextwithColor(
                        (-2.6, 1.4, 2),
                        f"RECORDING {gesture} {pre_add_rep+1}",
                        16,
                        color=(0, 255, 0),
                    )
            else:
                drawText((-2.6, 1.6, 2), "PAUSE press 's' to continue", 18)
        else:
            drawTextwithColor(
                (-2.6, 1.6, 2), "press 'c' to calibrate", 18, color=(255, 0, 0)
            )
            drawText(
                (-2.6, -1.8, 2), "Please calibrate first press 'c' to calibrate ", 16
            )
            if confirm_prompt:
                drawTextwithColor(
                    (-2.6, -2, 2),
                    f"REVERT BACK to {pre_add_rep-1} CONFIRM? press 'y' to confirm any to abort",
                    16,
                    color=(255, 0, 0),
                )
            else:
                drawText((-2.6, -2, 2), "'b' to back", 16)
            # drawText((-2.6, 1.6, 2), f"", 18)

    yaw = nx
    pitch = ny
    roll = nz
    drawText((-2.6, -1.6, 2), "Yaw: %f, Pitch: %f, Roll: %f" % (yaw, pitch, roll), 16)
    glRotatef(-roll, 0.00, 0.00, 1.00)
    glRotatef(pitch, 1.00, 0.00, 0.00)
    glRotatef(yaw, 0.00, 1.00, 0.00)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
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


def quat_to_ypr(q):
    q = [x / 16384 for x in q]
    yaw = math.atan2(
        2.0 * (q[1] * q[2] + q[0] * q[3]),
        q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
    )
    pitch = math.asin(-2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = math.atan2(
        2.0 * (q[0] * q[1] + q[2] * q[3]),
        q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
    )
    pitch *= 180.0 / math.pi
    yaw *= 180.0 / math.pi
    yaw -= 0  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min bangkok thailand is idk
    roll *= 180.0 / math.pi
    return [yaw, pitch, roll]


def keep_domain(angle):
    if angle >= 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
    return angle


def check_is_recording_moe(adjusted_roll, adjusted_pitch):
    global count_list
    if (
        min_roll < adjusted_roll
        and adjusted_roll < max_roll
        and min_pitch < adjusted_pitch
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


def add_new_gesture(data, database_file, name):
    new_gesture_name = input("type new gesture name: ")
    try:
        str(new_gesture_name)
        if new_gesture_name not in data[data.name == name]["gesture"].to_list():
            data = data.append(
                {"name": name, "gesture": new_gesture_name, "repetition": 0},
                ignore_index=True,
            )
            data.to_csv(open(database_file, "wb"), index=False)
        else:
            print("This gesture already exists")
            add_new_gesture(data, database_file, name)
    except ValueError:
        print("input invalid")
        add_new_gesture(data, database_file, name)


def mutate_to_data_for_store_emg(emg_data):
    return [emg_data, is_recording, pygame.time.get_ticks()]


def mutate_to_data_for_store_imu(imu_data, adjusted_ypr, neutral_ypr):
    return [imu_data, adjusted_ypr, neutral_ypr, is_recording, pygame.time.get_ticks()]


def quat_to_adjusted_ypr(quat, neutral_yaw, neutral_pitch, neutral_roll):
    try:
        [yaw, pitch, roll] = quat_to_ypr(quat)
        adjusted_yaw = keep_domain(yaw - neutral_yaw)
        adjusted_pitch = keep_domain(pitch - neutral_pitch)
        adjusted_roll = keep_domain(roll - neutral_roll)
    except ValueError:
        adjusted_yaw = 0
        adjusted_pitch = 0
        adjusted_roll = 0
    return [adjusted_yaw, adjusted_pitch, adjusted_roll]


def process_imu(record_cache_imu, imu_data, adjusted_ypr, neutral_ypr):
    record_cache_imu.append(
        [
            imu_data,
            adjusted_ypr,
            neutral_ypr,
            is_recording,
            pygame.time.get_ticks(),
        ]
    )


if __name__ == "__main__":
    data = pd.read_csv(open(database_file, "rb"))
    name = name_prompt(data)
    mode = mode_prompt()
    if mode == 2:
        add_new_gesture(data, database_file, name)
        data = pd.read_csv(open(database_file, "rb"))
    gesture = gesture_prompt(data, name)
    pre_add_rep = int(
        data[(data.name == name) & (data.gesture == gesture)]["repetition"]
    )
    error_str = ""
    confirm_prompt = False

    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    myo_ttys = detect_tty()
    p_left = multiprocessing.Process(
        target=myo_worker, args=(q_l, myo_ttys[0], (255, 0, 0), l_addr)
    )
    p_right = multiprocessing.Process(
        target=myo_worker, args=(q_r, myo_ttys[1], (0, 255, 0), r_addr)
    )

    p_left.start()
    p_right.start()
    screen = pygame.display.set_mode((640, 480), video_flags)
    pygame.display.set_caption("Data collector with IMU")
    resizewin(640, 480)
    init()
    frames = 0
    ticks = pygame.time.get_ticks()

    try:
        while True:
            while not (q_r.empty()) or not (q_l.empty()):
                emg_or_imu_r = q_r.get()
                emg_or_imu_l = q_l.get()
                if emg_or_imu_r is not None:
                    if len(emg_or_imu_r) != 8:
                        imu_r = emg_or_imu_r[0]
                        adjusted_ypr = quat_to_adjusted_ypr(
                            emg_or_imu_r[0],
                            neutral_yaw_r,
                            neutral_pitch_r,
                            neutral_roll_r,
                        )
                        is_recording = check_is_recording_moe(
                            adjusted_ypr[2], adjusted_ypr[1]
                        )
                        draw(1, *adjusted_ypr)
                        pygame.display.flip()
                        if is_start:
                            process_imu(
                                record_cache_imu_r,
                                emg_or_imu_r,
                                adjusted_ypr,
                                [neutral_yaw_r, neutral_pitch_r, neutral_roll_r],
                            )
                    else:
                        if is_start:
                            record_cache_emg_r.append(
                                [emg_or_imu_r, is_recording, pygame.time.get_ticks()]
                            )
                if emg_or_imu_l is not None:
                    if len(emg_or_imu_l) != 8:
                        imu_l = emg_or_imu_l[0]
                        adjusted_ypr = quat_to_adjusted_ypr(
                            emg_or_imu_l[0],
                            neutral_yaw_l,
                            neutral_pitch_l,
                            neutral_roll_l,
                        )
                        if is_start:
                            process_imu(
                                record_cache_imu_l,
                                emg_or_imu_l,
                                adjusted_ypr,
                                [neutral_yaw_l, neutral_pitch_l, neutral_roll_l],
                            )
                    else:
                        if is_start:
                            record_cache_emg_l.append(
                                [emg_or_imu_l, is_recording, pygame.time.get_ticks()]
                            )

                for event in pygame.event.get():
                    if event.type == QUIT:
                        raise KeyboardInterrupt()
                    elif event.type == KEYDOWN:
                        if not confirm_prompt:
                            if event.unicode in key_actions:
                                handle_event(event, imu_r, imu_l)
                        else:
                            if event.unicode == "y":
                                # Confirm prompt handling
                                confirm_prompt = False
                                print("Confirmation accepted")
                            else:
                                # Cancel the prompt
                                confirm_prompt = False
                                print("Confirmation cancelled")
    except KeyboardInterrupt:
        print("Quitting")
        pygame.display.quit()
        p_left.terminate()
        p_right.terminate()
        p_left.join()
        p_right.join()
        pygame.quit()
        sys.exit()
