#!/usr/bin/env python3      # tells the computer to find Python 3 and use it to run this file
"""
Head + eye controller for Minecraft
Hardware: Raspberry Pi
          + Adafruit BNO085 IMU, worn on the head
          + Raspberry Pi Camera Module 3 NoIR, pointed at one eye and lit by an IR LED

Controls
  Eyes look up / down / left / right    -> W / S / A / D (held while you look)
  Turn head left/right, tilt up/down    -> mouse / camera direction
                                           (joystick style: further from centre = faster)
  Deliberate eye gesture                -> Jump (Space)
      default: a long blink, eyes closed about 0.5 to 1.5 s
      or --jump_gesture double: two quick blinks
  Head nod (quick down and back up)     -> Right-click (interact)
  Eyes up + head tilted to the right    -> Sprint (Ctrl + W)
  Lean head back and hold               -> Crouch (Shift, held while leaning)
  Eyes closed 2 to 5 s, then open       -> Pause / resume (the program starts PAUSED)
  Eyes closed more than 5 s             -> Auto-pause (safety)

Where the actions go (--output)
  print  (default) only prints the actions. Nothing is pressed. Use this for testing.
  udp    sends them to minecraft_receiver.py on the computer running Minecraft
  local  presses keys on this Pi (Minecraft running on the Pi itself)

Recommended first runs
  1. python3 head_eye_controller.py --imu_test     check the head axes and their signs
  2. python3 head_eye_controller.py --tune         tune pupil detection (needs a desktop or VNC)
  3. python3 head_eye_controller.py --preview      calibrate, then dry run (prints actions)
  4. On the PC:  python minecraft_receiver.py
     On the Pi:  python3 head_eye_controller.py --output udp --host <PC IP address> --preview

Keys in the preview window: p = pause/resume, c = re-centre, q = quit.
Every threshold in the Settings class below can be changed from the command line,
e.g. --gaze_enter 0.4 --nod_depth 8. Run with --help to see them all.
"""

# BioGaming_controller

# Overview:
# This is the head & eye controller that runs on the Raspberry Pi.
# About 60 times a second it takes a picture of your eye and finds the pupil, the dark circle in the middle.
# It compares the pupil's position to the initial calibration in order to decide if you are looking up, down, left, or right.
# In order to spot a long blink (jump), it times your blinks.
# A 2-5 second close is a pause, like an on/off switch. The controller stops sending actions to Minecraft.
# It reads the IMU to get your head's position and speed: nod, tilt, and lean.
# Depending on specific action and speed, it pairs these into actions.
# Then, it sends those actions to the the minecraft receiver code.


import argparse     # a tool kit built into Python that allows you to easily add command line arguments to your program (a command line arguement is a way to give your program extra information when you run it)
import collections  # a tool kit built into Python that allows you to use special data structures like deque (a list that can be added to or removed from both ends)
import csv          # a tool kit built into Python that allows you to read and write CSV files (a common file format for storing data, Comma Separated Values)
import dataclasses  # a tool kit built into Python that allows you to easily create classes that are mainly used to store data
import json          # a tool kit built into Python that allows you to read and write JSON files (a common file format for storing data, JavaScript Object Notation)
import math          # a tool kit built into Python that allows you to do math functions like square root, sine, cosine, etc.   
import os            # a tool kit built into Python that allows you to interact with the operating system (like reading files, creating folders, etc.)
import random        # a tool kit built into Python that allows you to generate random numbers and make random choices
import socket        # a tool kit built into Python that allows you to create network connections (like sending and receiving data over the internet)
import threading    # a tool kit built into Python that allows you to run multiple tasks at the same time (like running a game and listening for input at the same time)
import time          # a tool kit built into Python that allows you to work with time (like measuring how long something takes or waiting for a certain amount of time)

import cv2           # a tool kit built into Python that allows you to work with images and videos (OpenCV library)
import numpy as np   # a tool kit built into Python that allows you to work with arrays and do math on them (NumPy library)


# ============================================================ Settings ======

@dataclasses.dataclass
class Settings:
    # --- Eye gaze -> W A S D. Fractions of how far you looked in each direction
    #     during calibration (1.0 = as far as you looked then).
    gaze_enter: float = 0.5        # key goes down when the eye gets this far
    gaze_exit: float = 0.35        # key comes up when the eye drops back below this
    gaze_dwell: float = 0.10       # seconds a direction must be held before its key goes down
    pupil_smoothing: float = 0.5   # 0..0.95, higher = steadier but slower

    # --- Pupil detection (tune with --tune)
    pupil_dark_offset: int = 25    # pupil = pixels within this much of the darkest pixel
    pupil_min_area: int = 60       # pixels
    pupil_max_area: int = 40000
    pupil_min_circularity: float = 0.55

    # --- Blinks. Times are how long the pupil is out of sight, in seconds.
    short_blink_min: float = 0.06
    short_blink_max: float = 0.40
    long_blink_min: float = 0.45
    long_blink_max: float = 1.50
    double_blink_window: float = 0.70
    pause_min: float = 2.0
    pause_max: float = 5.0
    auto_pause_after: float = 5.0

    # --- Head -> mouse. Angles are degrees away from your calibrated neutral position.
    yaw_deadzone: float = 6.0
    pitch_deadzone: float = 8.0
    yaw_gain: float = 30.0         # mouse pixels per second for each degree past the deadzone
    pitch_gain: float = 20.0
    mouse_max_speed: float = 1200.0
    pitch_dwell: float = 0.25      # hold an up/down tilt this long before the camera moves (stops nods moving it)
    invert_y: bool = False

    # --- Head nod -> right-click
    nod_depth: float = 10.0        # degrees the head must dip
    nod_return: float = 0.6        # fraction of the dip that must come back up
    nod_window: float = 0.8        # the whole nod must fit inside this many seconds
    nod_min_speed: float = 40.0    # degrees per second
    nod_cooldown: float = 1.0

    # --- Lean head back -> crouch
    lean_back_angle: float = 25.0
    lean_back_release: float = 18.0
    lean_back_hold: float = 0.3

    # --- Head tilt (ear toward right shoulder), used for sprint
    tilt_right_angle: float = 15.0


DIRECTIONS = ("up", "down", "left", "right")
DIRECTION_KEYS = {"up": "w", "down": "s", "left": "a", "right": "d"}


# ============================================================ Quaternions ===
# BNO085 quaternions are (i, j, k, real).

def q_mul(a, b):
    ai, aj, ak, ar = a
    bi, bj, bk, br = b
    return (ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
            ar * br - ai * bi - aj * bj - ak * bk)


def q_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def q_normalize(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q) if n > 0 else (0.0, 0.0, 0.0, 1.0)


def q_average(quats):
    """Average of several similar quaternions."""
    ref = quats[0]
    total = [0.0, 0.0, 0.0, 0.0]
    for q in quats:
        sign = 1.0 if sum(a * b for a, b in zip(q, ref)) >= 0 else -1.0
        for n in range(4):
            total[n] += sign * q[n]
    return q_normalize(total)


def quat_to_euler(i, j, k, real):
    """Quaternion -> (roll, pitch, yaw) in degrees, in the sensor's own frame."""
    roll = math.atan2(2 * (real * i + j * k), 1 - 2 * (i * i + j * j))
    sinp = 2 * (real * j - k * i)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (real * k + i * j), 1 - 2 * (j * j + k * k))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def parse_axis_map(text):
    """'yaw=-yaw,pitch=-pitch,roll=roll' -> {'yaw': ('yaw', -1), ...}"""
    result = {}
    for part in text.split(","):
        name, src = (s.strip() for s in part.split("="))
        sign = -1 if src.startswith("-") else 1
        src = src.lstrip("+-")
        if name not in ("yaw", "pitch", "roll") or src not in ("yaw", "pitch", "roll"):
            raise ValueError(f"bad --axis_map entry: {part}")
        result[name] = (src, sign)
    if set(result) != {"yaw", "pitch", "roll"}:
        raise ValueError("--axis_map must set yaw, pitch and roll")
    return result


# ============================================================ IMU ===========

class IMUReader(threading.Thread):
    """Reads the BNO085 orientation in the background and keeps the newest value."""

    def __init__(self, rate_hz=100):
        super().__init__(daemon=True)
        import board
        import busio
        from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
        from adafruit_bno08x.i2c import BNO08X_I2C

        self.bno = BNO08X_I2C(busio.I2C(board.SCL, board.SDA))  # address 0x4A
        # Game rotation vector = gyro + accelerometer, no magnetometer.
        # It never jumps near metal or electronics. Yaw drifts slowly, and that
        # drift is cancelled every time you resume or re-centre.
        self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        self.period = 1.0 / rate_hz
        self.lock = threading.Lock()
        self._latest = None
        self.read_errors = 0
        self.running = True

    def run(self):
        while self.running:
            t0 = time.monotonic()
            try:
                q = self.bno.game_quaternion
            except Exception:
                self.read_errors += 1   # occasional bad I2C packets are normal on the Pi
                time.sleep(0.01)
                continue
            if q is not None and sum(v * v for v in q) > 0.25:
                with self.lock:
                    self._latest = (time.monotonic(), q_normalize(q))
            spare = self.period - (time.monotonic() - t0)
            if spare > 0:
                time.sleep(spare)

    def latest(self):
        with self.lock:
            return self._latest

    def wait_ready(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.latest() is None:
            if time.monotonic() > end:
                raise RuntimeError("No data from the BNO085. Check wiring and that "
                                   "'i2cdetect -y 1' shows 0x4a.")
            time.sleep(0.05)

    def stop(self):
        self.running = False


# ============================================================ Camera ========

def setup_camera(width, height, fps, exposure_us, gain, focus_dioptres):
    from libcamera import controls
    from picamera2 import Picamera2

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": fps},
    )
    cam.configure(config)
    cam.start()
    ctrl = {
        "AfMode": controls.AfModeEnum.Manual,   # fixed focus, so the image doesn't "breathe"
        "LensPosition": focus_dioptres,         # 1 / distance in metres. 10 = 10 cm
    }
    if exposure_us > 0:
        ctrl.update({"AeEnable": False, "ExposureTime": exposure_us, "AnalogueGain": gain})
    cam.set_controls(ctrl)
    time.sleep(1.0)
    return cam


def grab_gray(cam):
    request = cam.capture_request()
    try:
        frame = request.make_array("main")
    finally:
        request.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ============================================================ Pupil =========

Pupil = collections.namedtuple("Pupil", "x y area circularity ellipse")


class PupilDetector:
    """Dark-pupil tracking: under IR light the pupil is the darkest round blob."""

    def __init__(self, settings, roi=None):
        self.s = settings
        self.roi = roi  # (x, y, w, h) or None
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, gray):
        """Returns (Pupil or None, mask)."""
        x0 = y0 = 0
        img = gray
        if self.roi:
            x0, y0, w, h = self.roi
            img = gray[y0:y0 + h, x0:x0 + w]

        blur = cv2.GaussianBlur(img, (7, 7), 0)
        small = cv2.resize(blur, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        darkest = float(small.min())
        _, mask = cv2.threshold(blur, darkest + self.s.pupil_dark_offset, 255,
                                cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)  # removes eyelashes

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h_img, w_img = img.shape[:2]
        best, best_score = None, 0.0
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= 0 or by <= 0 or bx + bw >= w_img or by + bh >= h_img:
                continue  # touches the edge: shadow, vignetting or half-visible pupil
            hull = cv2.convexHull(c)  # convex hull ignores the notch from the LED reflection
            area = cv2.contourArea(hull)
            if not (self.s.pupil_min_area <= area <= self.s.pupil_max_area):
                continue
            perim = cv2.arcLength(hull, True)
            circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0
            if circ < self.s.pupil_min_circularity:
                continue
            score = area * circ
            if score > best_score:
                best, best_score = (hull, area, circ), score

        if best is None:
            return None, mask
        hull, area, circ = best
        if len(hull) >= 5:
            (cx, cy), axes, angle = cv2.fitEllipse(hull)
        else:
            m = cv2.moments(hull)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            axes, angle = (2 * math.sqrt(area / math.pi),) * 2, 0.0
        ellipse = ((cx + x0, cy + y0), axes, angle)
        return Pupil(cx + x0, cy + y0, area, circ, ellipse), mask


# ============================================================ Gaze ==========

class GazeClassifier:
    """Turns pupil position into the set of directions being looked toward."""

    def __init__(self, calib, settings):
        self.s = settings
        self.set_calibration(calib)
        self.active = set()
        self.pending = {}

    def set_calibration(self, calib):
        self.center = np.array(calib["center"], dtype=float)
        self.vectors = {d: np.array(calib["offsets"][d], dtype=float) for d in DIRECTIONS}

    def scores(self, p):
        """How far toward each direction (1.0 = as far as during calibration)."""
        offset = np.array(p, dtype=float) - self.center
        return {d: float(offset @ v / (v @ v)) for d, v in self.vectors.items()}

    def reset(self):
        self.active.clear()
        self.pending.clear()

    def update(self, p, now):
        """p = (x, y) or None if the pupil is not visible (then the state is held)."""
        if p is None:
            return set(self.active), None
        sc = self.scores(p)
        for d in DIRECTIONS:
            if d in self.active:
                if sc[d] < self.s.gaze_exit:
                    self.active.discard(d)
            elif sc[d] > self.s.gaze_enter:
                start = self.pending.setdefault(d, now)
                if now - start >= self.s.gaze_dwell:
                    self.active.add(d)
                    self.pending.pop(d, None)
            else:
                self.pending.pop(d, None)
        for a, b in (("up", "down"), ("left", "right")):
            if a in self.active and b in self.active:
                self.active.discard(a if sc[a] < sc[b] else b)
        return set(self.active), sc


class EyeGestures:
    """Blink timing -> events: long_blink, double_blink, pause_toggle, auto_pause."""

    def __init__(self, settings):
        self.s = settings
        self.closed_since = None
        self.suppressed = False
        self.auto_fired = False
        self.last_short = None

    def closed_for(self, now):
        return 0.0 if self.closed_since is None else now - self.closed_since

    def update(self, visible, now, gaze_down):
        s, events = self.s, []
        if not visible:
            if self.closed_since is None:
                self.closed_since = now
                # Looking down makes the eyelid cover the pupil. That is not a blink.
                self.suppressed = gaze_down
                self.auto_fired = False
            elif not self.auto_fired and now - self.closed_since >= s.auto_pause_after:
                events.append("auto_pause")
                self.auto_fired = True
            return events

        if self.closed_since is None:
            return events
        d = now - self.closed_since
        self.closed_since = None
        if self.suppressed or self.auto_fired:
            return events
        if s.pause_min <= d <= s.pause_max:
            events.append("pause_toggle")
        elif s.long_blink_min <= d <= s.long_blink_max:
            events.append("long_blink")
        elif s.short_blink_min <= d <= s.short_blink_max:
            if self.last_short is not None and now - self.last_short <= s.double_blink_window:
                events.append("double_blink")
                self.last_short = None
            else:
                self.last_short = now
        return events


# ============================================================ Head ==========

HeadState = collections.namedtuple(
    "HeadState", "yaw pitch roll dx dy crouch nod tilt_right")


def _speed(angle, deadzone, gain, vmax):
    beyond = abs(angle) - deadzone
    if beyond <= 0:
        return 0.0
    return math.copysign(min(gain * beyond, vmax), angle)


class HeadTracker:
    """IMU orientation -> mouse movement, nod, lean back and tilt.

    After --axis_map: yaw + = head turned right, pitch + = head tilted up/back,
    roll + = right ear toward right shoulder.
    """

    def __init__(self, settings, axis_map):
        self.s = settings
        self.axis_map = axis_map
        self.neutral = (0.0, 0.0, 0.0, 1.0)
        self.history = collections.deque()
        self.cooldown_until = 0.0
        self.freeze_vertical_until = 0.0
        self.pitch_out_since = None
        self.lean_since = None
        self.crouch = False
        self.frac_x = self.frac_y = 0.0

    def set_neutral(self, q):
        self.neutral = q
        self.history.clear()
        self.pitch_out_since = None
        self.lean_since = None
        self.crouch = False

    def angles(self, q):
        rel = q_mul(q_conj(self.neutral), q)   # rotation since the neutral pose
        roll, pitch, yaw = quat_to_euler(*rel)
        raw = {"roll": roll, "pitch": pitch, "yaw": yaw}
        return {name: sign * raw[src] for name, (src, sign) in self.axis_map.items()}

    def update(self, q, now, dt):
        s = self.s
        a = self.angles(q)
        yaw, pitch, roll = a["yaw"], a["pitch"], a["roll"]

        # Nod: a fast dip of nod_depth degrees that comes back up, inside nod_window.
        nod = False
        self.history.append((now, pitch))
        while self.history and now - self.history[0][0] > s.nod_window:
            self.history.popleft()
        if now >= self.cooldown_until and len(self.history) >= 4:
            pitches = [p for _, p in self.history]
            low = min(pitches)
            dip = pitches[0] - low
            if dip >= s.nod_depth and pitch - low >= s.nod_return * dip:
                pts = list(self.history)
                fastest = max(abs(p2 - p1) / (t2 - t1)
                              for (t1, p1), (t2, p2) in zip(pts, pts[1:]) if t2 > t1)
                if fastest >= s.nod_min_speed:
                    nod = True
                    self.cooldown_until = now + s.nod_cooldown
                    self.freeze_vertical_until = now + 0.5
                    self.history.clear()

        # Lean back -> crouch (held), with hysteresis so it doesn't flicker.
        if self.crouch:
            if pitch < s.lean_back_release:
                self.crouch = False
                self.lean_since = None
        elif pitch >= s.lean_back_angle:
            if self.lean_since is None:
                self.lean_since = now
            if now - self.lean_since >= s.lean_back_hold:
                self.crouch = True
        else:
            self.lean_since = None

        tilt_right = roll >= s.tilt_right_angle

        # Mouse: joystick style. Speed grows with angle past the deadzone.
        vx = _speed(yaw, s.yaw_deadzone, s.yaw_gain, s.mouse_max_speed)
        vy = 0.0
        if abs(pitch) > s.pitch_deadzone:
            if self.pitch_out_since is None:
                self.pitch_out_since = now
        else:
            self.pitch_out_since = None
        if (self.pitch_out_since is not None
                and now - self.pitch_out_since >= s.pitch_dwell
                and not self.crouch
                and pitch < s.lean_back_angle
                and now >= self.freeze_vertical_until):
            vy = -_speed(pitch, s.pitch_deadzone, s.pitch_gain, s.mouse_max_speed)
            if s.invert_y:
                vy = -vy

        self.frac_x += vx * dt
        self.frac_y += vy * dt
        dx, dy = int(self.frac_x), int(self.frac_y)
        self.frac_x -= dx
        self.frac_y -= dy
        return HeadState(yaw, pitch, roll, dx, dy, self.crouch, nod, tilt_right)


# ============================================================ Outputs =======

def _fmt_keys(keys):
    return " + ".join(k.upper() for k in sorted(keys)) or "(none)"


class PrintOutput:
    """Dry run: prints what would be pressed."""

    def __init__(self):
        self.held = set()

    def update(self, held, dx, dy, taps):
        if held != self.held:
            print(f"[keys held] {_fmt_keys(held)}")
            self.held = set(held)
        for tap in taps:
            print(f"[tap] {tap.upper()}")

    def release_all(self):
        self.update(set(), 0, 0, [])

    def close(self):
        pass


class UdpOutput:
    """Sends the full state every frame, so a lost packet doesn't matter.
    Taps are repeated for 0.5 s with an id, and the receiver runs each id once."""

    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        self.session = random.randint(1, 2**31)
        self.seq = 0
        self.event_id = 0
        self.recent = collections.deque()

    def update(self, held, dx, dy, taps):
        now = time.monotonic()
        for tap in taps:
            self.event_id += 1
            self.recent.append((now, self.event_id, tap))
        while self.recent and now - self.recent[0][0] > 0.5:
            self.recent.popleft()
        self.seq += 1
        msg = {"v": 1, "session": self.session, "seq": self.seq,
               "held": sorted(held), "dx": int(dx), "dy": int(dy),
               "events": [[i, name] for _, i, name in self.recent]}
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError as exc:
            print(f"[udp] send failed: {exc}")

    def release_all(self):
        for _ in range(3):
            self.update(set(), 0, 0, [])

    def close(self):
        self.release_all()
        self.sock.close()


class LocalOutput:
    """Presses keys on this machine (uses minecraft_receiver.py's input code)."""

    def __init__(self):
        from minecraft_receiver import InputActuator
        self.act = InputActuator()
        print(f"Local input backend: {self.act.backend_name}")

    def update(self, held, dx, dy, taps):
        self.act.set_held(held)
        if dx or dy:
            self.act.move(dx, dy)
        for tap in taps:
            self.act.tap(tap)

    def release_all(self):
        self.act.release_all()

    def close(self):
        self.release_all()


# ============================================================ Display =======

WINDOW = "Head + Eye Controller"


class Display:
    def __init__(self, enabled, roi=None):
        self.enabled = enabled
        self.roi = roi

    def show(self, gray, pupil=None, lines=(), calib=None, mask=None, settings=None):
        """Draws the frame. Returns the key pressed (a str) or None."""
        if not self.enabled:
            return None
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 120, 0), 1)
        if calib is not None:
            c = np.array(calib["center"])
            cv2.drawMarker(vis, tuple(int(v) for v in c), (255, 255, 0), cv2.MARKER_CROSS, 14, 1)
            enter = settings.gaze_enter if settings else 0.5
            for d in DIRECTIONS:
                tgt = c + enter * np.array(calib["offsets"][d])
                cv2.circle(vis, tuple(int(v) for v in tgt), 4, (255, 255, 0), 1)
                cv2.putText(vis, DIRECTION_KEYS[d].upper(), (int(tgt[0]) + 6, int(tgt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if pupil is not None:
            cv2.ellipse(vis, pupil.ellipse, (0, 255, 0), 2)
            cv2.drawMarker(vis, (int(pupil.x), int(pupil.y)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        if mask is not None:
            mh, mw = mask.shape[:2]
            scale = (vis.shape[1] // 4) / mw
            inset = cv2.resize(mask, (int(mw * scale), int(mh * scale)))
            ih, iw = inset.shape[:2]
            vis[0:ih, vis.shape[1] - iw:] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        for n, text in enumerate(lines):
            y = 26 + 24 * n
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow(WINDOW, vis)
        k = cv2.waitKey(1) & 0xFF
        return None if k == 255 else chr(k).lower()

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


# ============================================================ Calibration ===

class Calibrator:
    def __init__(self, cam, imu, detector, display):
        self.cam, self.imu, self.detector, self.display = cam, imu, detector, display

    def collect(self, prompt, ready=2.5, hold=1.5):
        """Gets you ready, then records the pupil position and head orientation for `hold` s."""
        print(f"\n>>> {prompt}")
        points, quats = [], []
        t0 = time.monotonic()
        last_shown = None
        while True:
            el = time.monotonic() - t0
            gray = grab_gray(self.cam)
            pupil, mask = self.detector.detect(gray)
            if el < ready:
                n = math.ceil(ready - el)
                msg = f"{prompt} - starting in {n}"
                if n != last_shown:
                    print(f"    {n}...")
                    last_shown = n
            elif el < ready + hold:
                msg = f"{prompt} - HOLD"
                if last_shown != "hold":
                    print("    hold...")
                    last_shown = "hold"
                if pupil is not None:
                    points.append((pupil.x, pupil.y))
                sample = self.imu.latest() if self.imu else None
                if sample:
                    quats.append(sample[1])
            else:
                break
            if self.display.show(gray, pupil, [msg], mask=mask) == "q":
                raise KeyboardInterrupt
        if len(points) < 10:
            print("    Couldn't see your pupil well enough. Check lighting, focus and "
                  "--tune settings. Trying again.")
            return None
        return np.median(np.array(points), axis=0), quats

    def center_step(self):
        while True:
            res = self.collect("Look at the CENTRE of the screen, keep your head still")
            if res is not None and res[1]:
                return res[0], q_average(res[1])

    def run(self, existing=None):
        center, neutral = self.center_step()
        if existing is not None:
            print("    Using saved eye calibration (run with --recalibrate to redo it).")
            return {"center": center.tolist(), "offsets": existing["offsets"]}, neutral
        offsets = {}
        for d in DIRECTIONS:
            while True:
                res = self.collect(f"Head still. Look {d.upper()} with your EYES only "
                                   f"(as far as is comfortable)")
                if res is None:
                    continue
                vec = res[0] - center
                if np.hypot(*vec) < 5:
                    print("    Your eye barely moved. Try looking further. Trying again.")
                    continue
                offsets[d] = vec.tolist()
                break
        print("    Calibration done.")
        return {"center": center.tolist(), "offsets": offsets}, neutral


# ============================================================ Modes =========

def imu_test(args):
    axis_map = parse_axis_map(args.axis_map)
    imu = IMUReader()
    imu.start()
    imu.wait_ready()
    print("Hold your head in its normal position, looking at the screen...")
    time.sleep(2.0)
    head = HeadTracker(Settings(), axis_map)
    head.set_neutral(imu.latest()[1])
    print("\nNow check each movement:\n"
          "  turn head RIGHT           -> yaw should go POSITIVE\n"
          "  tilt head UP / BACK       -> pitch should go POSITIVE\n"
          "  right ear to right shoulder -> roll should go POSITIVE\n"
          "If one is on the wrong number or has the wrong sign, change --axis_map,\n"
          "e.g. --axis_map yaw=roll,pitch=-pitch,roll=yaw   (Ctrl+C to stop)\n")
    try:
        while True:
            a = head.angles(imu.latest()[1])
            print(f"\r  yaw {a['yaw']:+7.1f}   pitch {a['pitch']:+7.1f}   roll {a['roll']:+7.1f}"
                  f"   (read errors {imu.read_errors})   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        imu.stop()


def tune(args, s, roi):
    cam = setup_camera(args.width, args.height, args.fps, args.exposure, args.gain, args.focus)
    det = PupilDetector(s, roi)
    disp = Display(True, roi)
    cv2.namedWindow(WINDOW)
    nothing = lambda _v: None
    cv2.createTrackbar("dark offset", WINDOW, s.pupil_dark_offset, 120, nothing)
    cv2.createTrackbar("min area", WINDOW, s.pupil_min_area, 5000, nothing)
    cv2.createTrackbar("max area /100", WINDOW, max(1, s.pupil_max_area // 100), 2000, nothing)
    cv2.createTrackbar("circularity %", WINDOW, int(s.pupil_min_circularity * 100), 100, nothing)
    print("Tune until only your pupil is white in the mask (top right) and a green\n"
          "ellipse sits on it while you look around and blink.\n"
          "  r = draw a box around your eye (ROI), x = clear ROI, q = quit and print settings")
    try:
        while True:
            s.pupil_dark_offset = cv2.getTrackbarPos("dark offset", WINDOW)
            s.pupil_min_area = cv2.getTrackbarPos("min area", WINDOW)
            s.pupil_max_area = max(1, cv2.getTrackbarPos("max area /100", WINDOW)) * 100
            s.pupil_min_circularity = cv2.getTrackbarPos("circularity %", WINDOW) / 100
            gray = grab_gray(cam)
            pupil, mask = det.detect(gray)
            info = (f"pupil area {pupil.area:.0f}  circ {pupil.circularity:.2f}"
                    if pupil else "no pupil found")
            key = disp.show(gray, pupil, [info], mask=mask)
            if key == "q":
                break
            if key == "r":
                x, y, w, h = cv2.selectROI(WINDOW, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), False)
                if w > 0 and h > 0:
                    det.roi = disp.roi = (x, y, w, h)
            if key == "x":
                det.roi = disp.roi = None
    finally:
        cam.stop()
        disp.close()
    opts = (f"--pupil_dark_offset {s.pupil_dark_offset} --pupil_min_area {s.pupil_min_area} "
            f"--pupil_max_area {s.pupil_max_area} --pupil_min_circularity {s.pupil_min_circularity:.2f}")
    if det.roi:
        opts += " --roi " + ",".join(str(v) for v in det.roi)
    print(f"\nAdd these to your command:\n  {opts}")


def make_output(args):
    if args.output == "udp":
        if not args.host:
            raise SystemExit("--output udp needs --host <IP of the computer running Minecraft>")
        print(f"Sending actions to {args.host}:{args.port}")
        return UdpOutput(args.host, args.port)
    if args.output == "local":
        return LocalOutput()
    print("Dry run: actions are only printed (use --output udp or local to play).")
    return PrintOutput()


LOG_HEADER = ["t", "paused", "pupil_found", "pupil_x", "pupil_y",
              "score_up", "score_down", "score_left", "score_right",
              "yaw", "pitch", "roll", "keys_held", "taps", "mouse_dx", "mouse_dy"]


def run(args, s, roi):
    axis_map = parse_axis_map(args.axis_map)
    jump_event = "double_blink" if args.jump_gesture == "double" else "long_blink"

    out = make_output(args)
    print("Starting IMU...")
    imu = IMUReader()
    imu.start()
    imu.wait_ready()
    print("Starting camera...")
    cam = setup_camera(args.width, args.height, args.fps, args.exposure, args.gain, args.focus)
    detector = PupilDetector(s, roi)
    display = Display(args.preview, roi)

    existing = None
    if os.path.exists(args.calibration) and not args.recalibrate:
        with open(args.calibration) as f:
            saved = json.load(f)
        if saved.get("frame") == [args.width, args.height] and saved.get("roi") == (list(roi) if roi else None):
            existing = saved
        else:
            print("Saved calibration was made with a different frame size/ROI. Recalibrating.")

    calibrator = Calibrator(cam, imu, detector, display)
    log_file = log = None
    try:
        calib, neutral = calibrator.run(existing)
        calib.update({"frame": [args.width, args.height], "roi": list(roi) if roi else None})
        with open(args.calibration, "w") as f:
            json.dump(calib, f, indent=2)

        gaze = GazeClassifier(calib, s)
        eyes = EyeGestures(s)
        head = HeadTracker(s, axis_map)
        head.set_neutral(neutral)

        if args.log:
            log_file = open(args.log, "w", newline="")
            log = csv.writer(log_file)
            log.writerow(LOG_HEADER)

        paused = True
        print("\nPAUSED. Click into Minecraft, look at the screen, then close your eyes for "
              "2-5 s and open them to start (or press p in the preview).")

        smooth = None
        last = time.monotonic()
        last_status = 0.0
        frames = 0
        fps = 0.0
        fps_t0 = last

        def set_paused(value, reason):
            nonlocal paused
            if value == paused:
                return
            paused = value
            if paused:
                out.release_all()
                gaze.reset()
                print(f"\n=== PAUSED ({reason}) ===")
            else:
                sample = imu.latest()
                if sample:
                    head.set_neutral(sample[1])   # wherever your head is now = neutral
                gaze.reset()
                print(f"\n=== RUNNING ({reason}) ===")

        while True:
            gray = grab_gray(cam)
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            frames += 1
            if now - fps_t0 >= 1.0:
                fps, frames, fps_t0 = frames / (now - fps_t0), 0, now

            # Eye
            pupil, mask = detector.detect(gray)
            if pupil is not None:
                p = np.array([pupil.x, pupil.y])
                a = min(max(s.pupil_smoothing, 0.0), 0.95)
                smooth = p if smooth is None else a * smooth + (1 - a) * p
            gaze_dirs, scores = gaze.update(None if pupil is None else smooth, now)
            events = eyes.update(pupil is not None, now, "down" in gaze_dirs)

            # Head
            q = imu.latest()[1]
            hs = head.update(q, now, dt)

            # Pause control
            if "pause_toggle" in events:
                set_paused(not paused, "eye gesture")
            if "auto_pause" in events:
                set_paused(True, "eyes not seen for a while")

            # Build the actions
            held, taps, dx, dy = set(), [], 0, 0
            if not paused:
                if eyes.closed_for(now) <= s.long_blink_max:
                    held = {DIRECTION_KEYS[d] for d in gaze_dirs}
                if hs.crouch:
                    held.add("shift")
                if "up" in gaze_dirs and hs.tilt_right:
                    held.add("ctrl")       # sprint = Ctrl while W is held
                if jump_event in events:
                    taps.append("space")
                if hs.nod:
                    taps.append("right_click")
                dx, dy = hs.dx, hs.dy
            out.update(held, dx, dy, taps)

            if log:
                sc = scores or {}
                log.writerow([f"{now:.4f}", int(paused), int(pupil is not None),
                              f"{pupil.x:.1f}" if pupil else "", f"{pupil.y:.1f}" if pupil else "",
                              *(f"{sc[d]:.3f}" if d in sc else "" for d in DIRECTIONS),
                              f"{hs.yaw:.2f}", f"{hs.pitch:.2f}", f"{hs.roll:.2f}",
                              "+".join(sorted(held)), "+".join(taps), dx, dy])

            state = "PAUSED" if paused else "RUNNING"
            lines = [f"{state}   keys: {_fmt_keys(held)}   fps {fps:.0f}",
                     f"head yaw {hs.yaw:+.0f}  pitch {hs.pitch:+.0f}  roll {hs.roll:+.0f}",
                     "eye: " + ("  ".join(f"{d[0].upper()}{scores[d]:+.2f}" for d in DIRECTIONS)
                                if scores else f"closed {eyes.closed_for(now):.1f}s")]
            key = display.show(gray, pupil, lines, calib=calib, mask=mask, settings=s)
            if key == "q":
                break
            if key == "p":
                set_paused(not paused, "keyboard")
            if key == "c":
                out.release_all()
                center, neutral = calibrator.center_step()
                calib["center"] = center.tolist()
                gaze.set_calibration(calib)
                gaze.reset()
                head.set_neutral(neutral)
                smooth = None
                print("Re-centred.")

            if now - last_status >= 1.0:
                last_status = now
                print(f"[{state}] fps {fps:4.1f} | {lines[2]} | yaw {hs.yaw:+5.1f} pitch "
                      f"{hs.pitch:+5.1f} roll {hs.roll:+5.1f} | keys {_fmt_keys(held)} | "
                      f"imu errors {imu.read_errors}")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping, releasing all keys...")
        out.release_all()
        out.close()
        imu.stop()
        cam.stop()
        display.close()
        if log_file:
            log_file.close()


# ============================================================ Main ==========

def parse_args():
    p = argparse.ArgumentParser(description="Head + eye controller for Minecraft",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--imu_test", action="store_true", help="show head angles to check --axis_map")
    mode.add_argument("--tune", action="store_true", help="tune pupil detection with sliders")

    p.add_argument("--output", choices=["print", "udp", "local"], default="print")
    p.add_argument("--host", help="IP of the computer running minecraft_receiver.py")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--preview", action="store_true", help="show the eye camera window")
    p.add_argument("--jump_gesture", choices=["long", "double"], default="long",
                   help="long blink or double blink for jump")
    p.add_argument("--axis_map", default="yaw=-pitch,pitch=yaw,roll=roll",
                   help="map sensor angles to head angles. Default assumes the BNO085 stands "
                        "upright on the right arm of the glasses, chip facing outward, X arrow "
                        "pointing forward. Flat on top of the head: yaw=-yaw,pitch=-pitch,roll=roll")
    p.add_argument("--calibration", default="eye_calibration.json")
    p.add_argument("--recalibrate", action="store_true", help="redo the full eye calibration")
    p.add_argument("--log", metavar="FILE.csv", help="log every frame for later analysis")

    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=float, default=60)
    p.add_argument("--exposure", type=int, default=0, help="fixed exposure in µs (0 = auto)")
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--focus", type=float, default=10.0,
                   help="lens position in dioptres = 1/distance in m (10 = 10 cm, the closest it focuses)")
    p.add_argument("--roi", help="x,y,w,h region around the eye (find it with --tune)")

    g = p.add_argument_group("thresholds (see the Settings class)")
    for f in dataclasses.fields(Settings):
        if f.type in (bool, "bool"):
            g.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            g.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return p.parse_args()


def main():
    args = parse_args()
    s = Settings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)})
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    if args.imu_test:
        imu_test(args)
    elif args.tune:
        tune(args, s, roi)
    else:
        run(args, s, roi)


if __name__ == "__main__":
    main()
