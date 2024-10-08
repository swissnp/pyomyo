from cmath import e
import multiprocessing
import pandas as pd
from pyparsing import line
from pyomyo import Myo, emg_mode
import os
import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import pickle
import re
from serial.tools.list_ports import comports

# from plot_emgs import plot
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import get_cmap
import queue
import numpy as np

r_addr = [233, 28, 231, 168, 151, 205]
l_addr = [48, 12, 163, 25, 55, 236]
min_roll = -30
max_roll = 30
min_pitch = -30
max_pitch = 30
neutral_roll = 0
neutral_pitch = 0
neutral_yaw = 0
count = 0
is_recording = False
is_start = False
is_calibrated = False
count_list = [0] * 10
database_file = "examples/database.csv"
record_cache_imu = []
record_cache_emg = []
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
    myo_device.add_emg_handler(emg_handler)

    while True:
        myo_device.run()


# -------- Main Program Loop -----------


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
                    (-2.6, -2, 2), "'n' to save and next, 'd' to drop this rep", 16
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
            while not (q.empty()):
                emg_imu = list(q.get())
                if len(emg_imu) != 8:
                    quat, acc, gyro = emg_imu
                    [w, nx, ny, nz] = [x / 16384 for x in quat]
                    try:
                        [yaw, pitch, roll] = quat_to_ypr([w, nx, ny, nz])
                        adjusted_yaw = keep_domain(yaw - neutral_yaw)
                        adjusted_pitch = keep_domain(pitch - neutral_pitch)
                        adjusted_roll = keep_domain(roll - neutral_roll)
                    except ValueError:
                        adjusted_yaw = 0
                        adjusted_pitch = 0
                        adjusted_roll = 0
                    draw(1, adjusted_yaw, adjusted_pitch, adjusted_roll)

                    pygame.display.flip()
                    if is_start:
                        is_recording = check_is_recording_moe(
                            adjusted_roll, adjusted_pitch
                        )
                        imu_data = emg_imu
                        adjusted_ypr = [adjusted_yaw, adjusted_pitch, adjusted_roll]
                        neutral_ypr = [neutral_yaw, neutral_pitch, neutral_roll]
                        record_cache_imu.append(
                            [
                                imu_data,
                                adjusted_ypr,
                                neutral_ypr,
                                is_recording,
                                pygame.time.get_ticks(),
                            ]
                        )
                    # frames += 1
                    # print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))
                else:
                    emg_data = emg_imu
                    ## how to plot emg
                    if is_start:
                        record_cache_emg.append(
                            [emg_data, is_recording, pygame.time.get_ticks()]
                        )
                for ev in pygame.event.get():
                    if ev.type == QUIT:
                        raise KeyboardInterrupt()
                    elif ev.type == KEYDOWN:
                        if not confirm_prompt:
                            if error_str == "":
                                if not is_calibrated:
                                    if ev.unicode == "c":
                                        is_calibrated = True
                                        is_start = True
                                        neutral_roll = roll
                                        neutral_pitch = pitch
                                        neutral_yaw = yaw
                                    elif ev.unicode == "b":
                                        confirm_prompt = True
                                else:
                                    if ev.unicode == "e":
                                        print("Pressed e, erasing calibration")
                                        neutral_roll = 0
                                        neutral_pitch = 0
                                        neutral_yaw = 0
                                    elif ev.unicode == "s":
                                        print("Pressed s, Started")
                                        print("start record")
                                        is_recording = True
                                        is_start = True
                                    elif ev.unicode == "p":
                                        print("Pressed p, Paused")
                                        is_start = False
                                    elif ev.unicode == "q":
                                        if (
                                            record_cache_imu != []
                                            and record_cache_emg != []
                                        ):
                                            print("Pressed q, exit and save")
                                            data = pd.read_csv(
                                                open(database_file, "rb")
                                            )
                                            pre_add_rep = int(
                                                data[
                                                    (data.name == name)
                                                    & (data.gesture == gesture)
                                                ]["repetition"]
                                            )
                                            pickle.dump(
                                                record_cache_emg,
                                                open(
                                                    "examples/data/"
                                                    + name
                                                    + "_"
                                                    + gesture
                                                    + "_"
                                                    + str(pre_add_rep + 1)
                                                    + "_"
                                                    + "emg_rec.p",
                                                    "wb",
                                                ),
                                            )
                                            pickle.dump(
                                                record_cache_imu,
                                                open(
                                                    "examples/data/"
                                                    + name
                                                    + "_"
                                                    + gesture
                                                    + "_"
                                                    + str(pre_add_rep + 1)
                                                    + "_"
                                                    + "imu_rec.p",
                                                    "wb",
                                                ),
                                            )
                                            data.loc[
                                                (data.name == name)
                                                & (data.gesture == gesture),
                                                ["repetition"],
                                            ] = (
                                                int(
                                                    data[
                                                        (data.name == name)
                                                        & (data.gesture == gesture)
                                                    ]["repetition"]
                                                )
                                                + 1
                                            )
                                            data.to_csv(
                                                open(database_file, "wb"), index=False
                                            )
                                            is_start = False
                                            record_cache_emg = []
                                            record_cache_imu = []
                                            print("saved")
                                        else:
                                            error_str = "no data to save"
                                        raise KeyboardInterrupt()
                                    elif ev.unicode == "n":
                                        if (
                                            record_cache_imu != []
                                            and record_cache_emg != []
                                        ):
                                            print("Pressed n, next reptition")
                                            pre_add_rep = int(
                                                data[
                                                    (data.name == name)
                                                    & (data.gesture == gesture)
                                                ]["repetition"]
                                            )
                                            pickle.dump(
                                                record_cache_emg,
                                                open(
                                                    "examples/data/"
                                                    + name
                                                    + "_"
                                                    + gesture
                                                    + "_"
                                                    + str(pre_add_rep + 1)
                                                    + "_"
                                                    + "emg_rec.p",
                                                    "wb",
                                                ),
                                            )
                                            pickle.dump(
                                                record_cache_imu,
                                                open(
                                                    "examples/data/"
                                                    + name
                                                    + "_"
                                                    + gesture
                                                    + "_"
                                                    + str(pre_add_rep + 1)
                                                    + "_"
                                                    + "imu_rec.p",
                                                    "wb",
                                                ),
                                            )
                                            pre_add_rep += 1
                                            data.loc[
                                                (data.name == name)
                                                & (data.gesture == gesture),
                                                ["repetition"],
                                            ] = pre_add_rep
                                            data.to_csv(
                                                open(database_file, "wb"), index=False
                                            )
                                            is_start = False
                                            is_calibrated = False
                                            is_recording = False
                                            record_cache_emg = []
                                            record_cache_imu = []
                                            print("saved")
                                        else:
                                            error_str = "no data to save"
                                    elif ev.unicode == "d":
                                        is_start = False
                                        is_calibrated = False
                                        is_recording = False
                                        record_cache_emg = []
                                        record_cache_imu = []
                                    else:
                                        error_str = ""
                            else:
                                error_str = ""
                        else:
                            if ev.unicode == "y":
                                confirm_prompt = False
                                pre_add_rep = int(
                                    data[
                                        (data.name == name) & (data.gesture == gesture)
                                    ]["repetition"]
                                )
                                pre_add_rep -= 1
                                data.loc[
                                    (data.name == name) & (data.gesture == gesture),
                                    ["repetition"],
                                ] = pre_add_rep
                                data.to_csv(open(database_file, "wb"), index=False)
                                is_start = False
                                is_calibrated = False
                                is_recording = False
                                record_cache_emg = []
                                record_cache_imu = []
                                print("saved")
                            else:
                                confirm_prompt = False
    except KeyboardInterrupt:
        print("Quitting")
        pygame.display.quit()
        plt.close()
        q.close()
        p.terminate()
        p.join()
        pygame.quit()
