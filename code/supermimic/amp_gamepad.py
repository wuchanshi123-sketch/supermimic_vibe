"""Read Logitech F710 commands through Linux joydev, without GUI dependencies.

The read-only ioctls and ``js_event`` layout follow linux/joystick.h; semantic
axis/button codes follow linux/input-event-codes.h. No calibration is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import glob
import math
import os
from pathlib import Path
import struct
import time


@dataclass(frozen=True)
class GamepadSample:
    connected: bool
    ready: bool
    axes: tuple[float, float, float]
    pressed: frozenset[str]
    path: str | None
    name: str | None
    status: str


# Linux UAPI: _IOR('j', nr, size), ABS_CNT=64, KEY_MAX-BTN_MISC+1=512.
def _ior(number: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (ord("j") << 8) | number


_JSIOCGAXES = _ior(0x11, 1)
_JSIOCGBUTTONS = _ior(0x12, 1)
_JSIOCGNAME = _ior(0x13, 256)
_JSIOCGAXMAP = _ior(0x32, 64)
_JSIOCGBTNMAP = _ior(0x34, 1024)
_EVENT = struct.Struct("=IhBB")
_INIT, _BUTTON, _AXIS = 0x80, 0x01, 0x02
_MAX_READS, _READ_SIZE = 8, 512


def apply_deadzone(value: float, dz: float) -> float:
    """Clamp to [-1, 1], then rescale the portion outside the deadzone."""
    if not math.isfinite(value) or not math.isfinite(dz) or not 0 <= dz < 1:
        raise ValueError("axis must be finite and deadzone must be in [0, 1)")
    magnitude = min(abs(value), 1.0)
    return math.copysign(max(0.0, magnitude - dz) / (1.0 - dz), value)


def map_velocity(axes: tuple[float, float, float], deadzone: float,
                 forward: float = 0.8, backward: float = 0.5,
                 lateral: float = 0.5, yaw: float = 1.0) -> tuple[float, float, float]:
    """Map LEFT_X, LEFT_Y, RIGHT_X to forward, left, CCW velocities."""
    if len(axes) != 3:
        raise ValueError("expected LEFT_X, LEFT_Y, RIGHT_X")
    if any(not math.isfinite(v) or v < 0 for v in (forward, backward, lateral, yaw)):
        raise ValueError("velocity scales must be finite and nonnegative")
    lx, ly, rx = (apply_deadzone(v, deadzone) for v in axes)
    vx = -ly * (forward if ly <= 0 else backward)
    return vx, -lx * lateral, -rx * yaw


def _sysfs_identity(path: str) -> tuple[str | None, int | None, int | None]:
    base = Path("/sys/class/input") / Path(os.path.realpath(path)).name / "device"
    def read(relative: str) -> str | None:
        try:
            return (base / relative).read_text().strip()
        except OSError:
            return None
    def hex_id(relative: str) -> int | None:
        try:
            return int(read(relative) or "", 16)
        except ValueError:
            return None
    return read("name"), hex_id("id/vendor"), hex_id("id/product")


def _mapping(name: str, vendor: int | None, product: int | None,
             axis_codes: list[int], button_codes: list[int]) -> dict:
    lower = name.lower()
    known_name = "logitech" in lower and ("f710" in lower or "rumblepad" in lower)
    recognized = (vendor == 0x046D and product in (0xC21F, 0xC219)) or known_name
    if not recognized:
        raise ValueError("not a recognized Logitech F710/Rumblepad")
    # XInput ABS_Z is a trigger, including its idle value of -1. Only a known
    # DirectInput identity/layout permits using it as the right stick.
    legacy_buttons = button_codes[:10] == list(range(0x120, 0x12A))
    known_xinput = vendor == 0x046D and product == 0xC21F
    known_dinput = ((vendor == 0x046D and product == 0xC219)
                    or (known_name and legacy_buttons))
    if 0x03 in axis_codes:
        right_x = 0x03
    elif known_dinput and not known_xinput:
        right_x = 0x02
    else:
        raise ValueError("missing ABS_RX right-stick mapping; ABS_Z may be a trigger")
    if any(code not in axis_codes for code in (0x00, 0x01, right_x)):
        raise ValueError("missing left-stick or right-stick axis mapping")
    axes = dict(zip(("LEFT_X", "LEFT_Y", "RIGHT_X"),
                    (axis_codes.index(code) for code in (0x00, 0x01, right_x))))
    button_ids = {"a": 0x130, "b": 0x131, "start": 0x13B}
    if all(code in button_codes for code in button_ids.values()):
        buttons = {key: button_codes.index(code) for key, code in button_ids.items()}
    elif right_x == 0x02 and legacy_buttons:
        # Known Logitech DirectInput layout: X, A, B, Y, ..., Back, Start.
        buttons = {"a": 1, "b": 2, "start": 9}
    else:
        raise ValueError("unrecognized A/B/Start button mapping")
    return {"axes": axes, "buttons": buttons,
            "mode": "XInput" if right_x == 0x03 else "DirectInput"}


def _ioctl(fd: int, request: int, size: int) -> bytearray:
    result = bytearray(size)
    fcntl.ioctl(fd, request, result, True)
    return result


def _probe(path: str) -> tuple[int | None, dict]:
    """Return an open usable descriptor, or an error suitable for diagnostics."""
    name, vendor, product = _sysfs_identity(path)
    info = dict(path=path, name=name, vendor=vendor, product=product,
                usable=False, mapping=None, error=None)
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        name = bytes(_ioctl(fd, _JSIOCGNAME, 256)).split(b"\0", 1)[0].decode("utf-8", "replace")
        info["name"] = name
        axis_count = _ioctl(fd, _JSIOCGAXES, 1)[0]
        button_count = _ioctl(fd, _JSIOCGBUTTONS, 1)[0]
        if not 0 < axis_count <= 64:
            raise ValueError("invalid axis count")
        axis_codes = list(_ioctl(fd, _JSIOCGAXMAP, 64)[:axis_count])
        button_codes = list(struct.unpack("=512H", _ioctl(fd, _JSIOCGBTNMAP, 1024)))[:button_count]
        info["mapping"] = _mapping(name, vendor, product, axis_codes, button_codes)
        info.update(usable=True, axis_count=axis_count, button_count=button_count)
        return fd, info
    except (OSError, ValueError) as exc:
        info["error"] = str(exc)
        if fd is not None:
            os.close(fd)
        return None, info


def list_gamepads() -> list[dict]:
    """Inspect every joydev path, including unsupported/inaccessible devices."""
    devices = []
    for path in sorted(glob.glob("/dev/input/js*")):
        fd, info = _probe(path)
        if fd is not None:
            os.close(fd)
        devices.append(info)
    return devices


class F710Reader:
    """Nonblocking F710 reader with hotplug retry and initialization gating."""

    def __init__(self, device: str | None = None, scan_interval: float = 1.0):
        if not math.isfinite(scan_interval) or scan_interval <= 0:
            raise ValueError("scan_interval must be finite and positive")
        self.device = device
        self.scan_interval = scan_interval
        self._fd: int | None = None
        self._info: dict = {}
        self._next_scan = 0.0
        self._status = "waiting for Logitech F710"
        self._closed = False
        self._reset_state()

    def _reset_state(self) -> None:
        self._axes = [0.0, 0.0, 0.0]
        self._buttons: dict[int, bool] = {}
        self._seen: set[int] = set()
        self._buffer = bytearray()

    def _disconnect(self, reason: str) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._reset_state()
        self._status = reason
        self._next_scan = time.monotonic() + self.scan_interval

    def _connect(self, now: float) -> None:
        self._next_scan = now + self.scan_interval
        paths = [self.device] if self.device else sorted(glob.glob("/dev/input/js*"))
        errors = []
        for path in paths:
            fd, info = _probe(path)
            if fd is not None:
                self._fd, self._info = fd, info
                self._reset_state()
                self._status = "initializing"
                return
            errors.append(f"{path}: {info['error']}")
        self._status = "; ".join(errors) or "waiting for Logitech F710 (no joydev devices)"

    def poll(self) -> GamepadSample:
        pressed: set[str] = set()
        now = time.monotonic()
        if not self._closed and self._fd is None and now >= self._next_scan:
            self._connect(now)
        if self._fd is not None:
            mapping = self._info["mapping"]
            axes = list(mapping["axes"].values())
            buttons = {index: name for name, index in mapping["buttons"].items()}
            for _ in range(_MAX_READS):
                try:
                    chunk = os.read(self._fd, _READ_SIZE)
                except OSError as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        break
                    if exc.errno == errno.EINTR:
                        continue
                    self._disconnect(f"disconnected: {exc}")
                    pressed.clear()
                    break
                if not chunk:
                    self._disconnect("disconnected: device reached EOF")
                    pressed.clear()
                    break
                self._buffer.extend(chunk)
                complete = len(self._buffer) // _EVENT.size * _EVENT.size
                for offset in range(0, complete, _EVENT.size):
                    _, value, event_type, number = _EVENT.unpack_from(self._buffer, offset)
                    kind = event_type & ~_INIT
                    initial = bool(event_type & _INIT)
                    if kind == _AXIS and number in axes:
                        self._axes[axes.index(number)] = max(-1.0, min(1.0, value / 32767.0))
                        if initial:
                            self._seen.add(number)
                    elif kind == _BUTTON and number in buttons and value in (0, 1):
                        down = bool(value)
                        if not initial and down and not self._buttons.get(number, False):
                            pressed.add(buttons[number])
                        self._buttons[number] = down
                del self._buffer[:complete]
            if self._fd is not None:
                ready = set(axes).issubset(self._seen)
                self._status = "ready" if ready else "initializing"
        connected = self._fd is not None
        ready = connected and self._status == "ready"
        return GamepadSample(connected, ready,
                             tuple(self._axes) if ready else (0.0, 0.0, 0.0),
                             frozenset(pressed) if ready else frozenset(),
                             self._info.get("path"), self._info.get("name"), self._status)

    def close(self) -> None:
        """Release the descriptor; future polls remain disconnected."""
        self._closed = True
        self._disconnect("closed")
