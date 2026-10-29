#!/usr/bin/env python3
"""
Minecraft receiver: runs on the computer that runs Minecraft.

It listens for actions sent by head_eye_controller.py on the Raspberry Pi and turns
them into real key presses and mouse movement.

Install
  Windows : nothing extra (uses the built-in SendInput)
  Linux   : pip install evdev      (and access to /dev/uinput, e.g. run with sudo)
  macOS   : pip3 install pyobjc-framework-Quartz
            then System Settings > Privacy & Security > Accessibility: turn on Terminal
            (or whichever app you run this from), and allow incoming connections if the
            firewall asks

Run
  python minecraft_receiver.py                    # listen on UDP port 5555
  python minecraft_receiver.py --pi 192.168.1.42  # only accept that Pi
  python minecraft_receiver.py --dry_run          # print actions, press nothing

Then click into the Minecraft window. If nothing arrives from the Pi for 0.5 s,
every key is released, so a crash or Wi-Fi drop never leaves you walking forever.
Only one controller is accepted: the first one heard, or the one given with --pi.
"""

# BioGaming_receiver

# Overview:
# This function waits for messages from the BioGaming_controller code.
# It checks that the message came from the Raspberry Pi.
# Then it sends the sends the corresponding keypress/mouse movement message to macOS.
# Then it releases every key if the Pi goes quiet for half a second. *subject to change*
# Minecraft just has to be the active window for this to work (in the front).


import argparse
import json
import socket
import sys
import threading
import time

KEYS = ("w", "a", "s", "d", "space", "ctrl", "shift")
BUTTONS = ("right_click", "left_click")
ALL_INPUTS = KEYS + BUTTONS
MAX_MOVE = 500  # clamp per packet, in pixels


# ------------------------------------------------------------ Windows -------

class _WindowsBackend:
    """SendInput with hardware scan codes + relative mouse motion. Works with
    Minecraft's 'Raw Input' setting on or off."""

    name = "Windows SendInput"
    SCAN = {"w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20,
            "space": 0x39, "ctrl": 0x1D, "shift": 0x2A}
    MOUSE = {"right_click": (0x0008, 0x0010), "left_click": (0x0002, 0x0004)}

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                        ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self.ct, self.INPUT, self.KEYBDINPUT, self.MOUSEINPUT = ctypes, INPUT, KEYBDINPUT, MOUSEINPUT
        self.send_input = ctypes.windll.user32.SendInput

    def _send(self, inp):
        self.send_input(1, self.ct.byref(inp), self.ct.sizeof(inp))

    def _key(self, name, up):
        flags = 0x0008 | (0x0002 if up else 0)  # KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        inp = self.INPUT(type=1)
        inp.u.ki = self.KEYBDINPUT(0, self.SCAN[name], flags, 0, 0)
        self._send(inp)

    def _mouse(self, dx, dy, flags):
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(dx, dy, 0, flags, 0, 0)
        self._send(inp)

    def down(self, name):
        if name in self.MOUSE:
            self._mouse(0, 0, self.MOUSE[name][0])
        else:
            self._key(name, up=False)

    def up(self, name):
        if name in self.MOUSE:
            self._mouse(0, 0, self.MOUSE[name][1])
        else:
            self._key(name, up=True)

    def move(self, dx, dy):
        self._mouse(dx, dy, 0x0001)  # MOUSEEVENTF_MOVE (relative)


# ------------------------------------------------------------ Linux ---------

class _EvdevBackend:
    """A virtual keyboard + mouse through /dev/uinput. Works on X11 and Wayland."""

    name = "Linux uinput (evdev)"

    def __init__(self):
        from evdev import UInput, ecodes as e
        self.e = e
        self.map = {"w": e.KEY_W, "a": e.KEY_A, "s": e.KEY_S, "d": e.KEY_D,
                    "space": e.KEY_SPACE, "ctrl": e.KEY_LEFTCTRL, "shift": e.KEY_LEFTSHIFT,
                    "right_click": e.BTN_RIGHT, "left_click": e.BTN_LEFT}
        self.ui = UInput({e.EV_KEY: list(self.map.values()), e.EV_REL: [e.REL_X, e.REL_Y]},
                         name="head-eye-controller")
        time.sleep(0.5)  # give the desktop a moment to notice the new device

    def down(self, name):
        self.ui.write(self.e.EV_KEY, self.map[name], 1)
        self.ui.syn()

    def up(self, name):
        self.ui.write(self.e.EV_KEY, self.map[name], 0)
        self.ui.syn()

    def move(self, dx, dy):
        if dx:
            self.ui.write(self.e.EV_REL, self.e.REL_X, dx)
        if dy:
            self.ui.write(self.e.EV_REL, self.e.REL_Y, dy)
        self.ui.syn()


# ------------------------------------------------------------ macOS ---------

class _MacBackend:
    """Quartz events posted at the HID level, the same place a real keyboard and
    mouse enter. Mouse moves carry delta values, which is what Minecraft reads
    while it has the cursor locked."""

    name = "macOS Quartz events"
    KEYCODES = {"w": 13, "a": 0, "s": 1, "d": 2, "space": 49}
    MODIFIERS = {"ctrl": 59, "shift": 56}   # left Control, left Shift

    def __init__(self):
        import Quartz as Q
        self.Q = Q
        self.flag_bits = {"ctrl": Q.kCGEventFlagMaskControl, "shift": Q.kCGEventFlagMaskShift}
        self.flags = 0
        self.buttons = set()
        b = Q.CGDisplayBounds(Q.CGMainDisplayID())
        self.bounds = (b.origin.x, b.origin.y,
                       b.origin.x + b.size.width - 1, b.origin.y + b.size.height - 1)

    def _post(self, ev):
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def _cursor(self):
        return self.Q.CGEventGetLocation(self.Q.CGEventCreate(None))

    def _key(self, name, down):
        Q = self.Q
        if name in self.MODIFIERS:
            if down:
                self.flags |= self.flag_bits[name]
            else:
                self.flags &= ~self.flag_bits[name]
            ev = Q.CGEventCreateKeyboardEvent(None, self.MODIFIERS[name], down)
            Q.CGEventSetType(ev, Q.kCGEventFlagsChanged)
        else:
            ev = Q.CGEventCreateKeyboardEvent(None, self.KEYCODES[name], down)
        Q.CGEventSetFlags(ev, self.flags)
        self._post(ev)

    def _button(self, name, down):
        Q = self.Q
        if name == "left_click":
            etype = Q.kCGEventLeftMouseDown if down else Q.kCGEventLeftMouseUp
            button = Q.kCGMouseButtonLeft
        else:
            etype = Q.kCGEventRightMouseDown if down else Q.kCGEventRightMouseUp
            button = Q.kCGMouseButtonRight
        (self.buttons.add if down else self.buttons.discard)(name)
        self._post(Q.CGEventCreateMouseEvent(None, etype, self._cursor(), button))

    def down(self, name):
        if name in ("left_click", "right_click"):
            self._button(name, True)
        else:
            self._key(name, True)

    def up(self, name):
        if name in ("left_click", "right_click"):
            self._button(name, False)
        else:
            self._key(name, False)

    def move(self, dx, dy):
        Q = self.Q
        pos = self._cursor()
        x0, y0, x1, y1 = self.bounds
        new = Q.CGPointMake(min(max(pos.x + dx, x0), x1), min(max(pos.y + dy, y0), y1))
        if "left_click" in self.buttons:      # moving while mining = a drag
            etype, button = Q.kCGEventLeftMouseDragged, Q.kCGMouseButtonLeft
        elif "right_click" in self.buttons:
            etype, button = Q.kCGEventRightMouseDragged, Q.kCGMouseButtonRight
        else:
            etype, button = Q.kCGEventMouseMoved, Q.kCGMouseButtonLeft
        ev = Q.CGEventCreateMouseEvent(None, etype, new, button)
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaX, int(dx))
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventDeltaY, int(dy))
        self._post(ev)


# ------------------------------------------------------------ fallback ------

class _PynputBackend:
    name = "pynput"

    def __init__(self):
        from pynput.keyboard import Controller as KB, Key
        from pynput.mouse import Controller as MS, Button
        self.kb, self.ms = KB(), MS()
        self.map = {"w": "w", "a": "a", "s": "s", "d": "d", "space": Key.space,
                    "ctrl": Key.ctrl_l, "shift": Key.shift_l}
        self.buttons = {"right_click": Button.right, "left_click": Button.left}

    def down(self, name):
        if name in self.buttons:
            self.ms.press(self.buttons[name])
        else:
            self.kb.press(self.map[name])

    def up(self, name):
        if name in self.buttons:
            self.ms.release(self.buttons[name])
        else:
            self.kb.release(self.map[name])

    def move(self, dx, dy):
        self.ms.move(dx, dy)


class _DryRunBackend:
    name = "dry run (nothing is pressed)"

    def down(self, name):
        print(f"  down  {name}")

    def up(self, name):
        print(f"  up    {name}")

    def move(self, dx, dy):
        pass


def _pick_backend(dry_run=False):
    if dry_run:
        return _DryRunBackend()
    if sys.platform.startswith("win"):
        return _WindowsBackend()
    if sys.platform == "darwin":
        try:
            return _MacBackend()
        except ImportError:
            raise SystemExit("Missing Quartz. Run:  pip3 install pyobjc-framework-Quartz")
    if sys.platform.startswith("linux"):
        try:
            return _EvdevBackend()
        except Exception as exc:  # not installed or no permission for /dev/uinput
            print(f"evdev unavailable ({exc}); falling back to pynput")
    try:
        return _PynputBackend()
    except Exception as exc:
        raise SystemExit(f"No input backend available: {exc}\n"
                         "Install one: pip install evdev (Linux) or pip install pynput")


# ------------------------------------------------------------ Actuator ------

class InputActuator:
    """Holds keys, taps buttons and moves the mouse. Thread-safe."""

    def __init__(self, dry_run=False):
        self.backend = _pick_backend(dry_run)
        self.backend_name = self.backend.name
        self.lock = threading.Lock()
        self.held = set()

    def set_held(self, keys):
        keys = {k for k in keys if k in ALL_INPUTS}
        with self.lock:
            for k in self.held - keys:
                self.backend.up(k)
            for k in keys - self.held:
                self.backend.down(k)
            self.held = keys

    def move(self, dx, dy):
        dx = max(-MAX_MOVE, min(MAX_MOVE, int(dx)))
        dy = max(-MAX_MOVE, min(MAX_MOVE, int(dy)))
        if dx or dy:
            with self.lock:
                self.backend.move(dx, dy)

    def tap(self, name, duration=0.08):
        if name not in ALL_INPUTS:
            return
        with self.lock:
            self.backend.down(name)
        threading.Timer(duration, self._tap_up, args=(name,)).start()

    def _tap_up(self, name):
        with self.lock:
            if name not in self.held:
                self.backend.up(name)

    def release_all(self):
        self.set_held(set())


# ------------------------------------------------------------ Main ----------

def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "this computer's IP"


def main():
    ap = argparse.ArgumentParser(description="Receive head/eye controller actions for Minecraft")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--pi", help="only accept packets from this IP address")
    ap.add_argument("--timeout", type=float, default=0.5,
                    help="release all keys if nothing arrives for this many seconds")
    ap.add_argument("--dry_run", action="store_true", help="print actions instead of pressing keys")
    args = ap.parse_args()

    act = InputActuator(dry_run=args.dry_run)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.1)

    print(f"Input backend: {act.backend_name}")
    print(f"Listening on UDP port {args.port}. On the Pi use:  --output udp --host {_local_ip()}")
    print("Click into the Minecraft window. Ctrl+C here to stop.\n")

    allowed = args.pi
    last_packet = 0.0
    session = None
    last_event = 0
    shown_held = None
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                if act.held and time.monotonic() - last_packet > args.timeout:
                    act.release_all()
                    shown_held = set()
                    print("No data from the controller: released all keys")
                continue

            if allowed is None:
                allowed = addr[0]
                print(f"Accepting controller at {allowed}")
            if addr[0] != allowed:
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if not isinstance(msg, dict) or msg.get("v") != 1:
                continue
            last_packet = time.monotonic()

            if msg.get("session") != session:   # controller restarted
                session = msg.get("session")
                last_event = 0

            held = {k for k in msg.get("held", []) if k in KEYS}
            act.set_held(held)
            act.move(msg.get("dx", 0), msg.get("dy", 0))
            for event_id, name in msg.get("events", []):
                if isinstance(event_id, int) and event_id > last_event:
                    last_event = event_id
                    act.tap(name)
                    print(f"tap   {name}")
            if held != shown_held:
                print("held  " + (" + ".join(sorted(held)) or "(none)"))
                shown_held = held
    except KeyboardInterrupt:
        pass
    finally:
        act.release_all()
        sock.close()
        print("Stopped, all keys released.")


if __name__ == "__main__":
    main()
