"""CPU-only joydev checks: PYTHONPATH=code python code/scripts/dev/test_amp_gamepad.py."""
from __future__ import annotations

import errno
import os
import struct
import unittest
from unittest.mock import patch

from supermimic import amp_gamepad as pad


X_AXES = [0, 1, 2, 3, 4, 5, 16, 17]  # ABS_Z is the left trigger.
X_BUTTONS = [0x130, 0x131, 0x134, 0x133, 0x136, 0x137, 0x13A, 0x13B]


def event(value, kind, number):
    return pad._EVENT.pack(0, value, kind, number)


def initial_axes(x=0, y=0, rx=0):
    return b"".join(event(value, pad._INIT | pad._AXIS, number)
                    for value, number in ((x, 0), (y, 1), (rx, 3)))


class FakeJoydev:
    """Exercise real probing/event parsing with mocked kernel system calls."""
    def __init__(self, paths=None):
        self.paths = paths or {"/dev/input/js1": "Logitech Gamepad F710"}
        self.fd_paths = {}
        self.chunks = {}
        self.clock = 0.0
        self.next_fd = 100
        self.read_count = 0
        self.open_flags = []
        self.closed = []

    def start(self, test):
        replacements = {
            "glob.glob": lambda _: list(self.paths),
            "_sysfs_identity": lambda path: (self.paths.get(path), None, None),
            "os.open": self.open,
            "os.read": self.read,
            "os.close": self.closed.append,
            "fcntl.ioctl": self.ioctl,
            "time.monotonic": lambda: self.clock,
        }
        for target, replacement in replacements.items():
            mock = patch.object(pad, target, replacement) if "." not in target else patch(
                f"supermimic.amp_gamepad.{target}", replacement)
            mock.start()
            test.addCleanup(mock.stop)
        return self

    def open(self, path, flags):
        self.open_flags.append(flags)
        if path not in self.paths:
            raise FileNotFoundError(errno.ENOENT, "removed", path)
        fd = self.next_fd
        self.next_fd += 1
        self.fd_paths[fd] = path
        return fd

    def ioctl(self, fd, request, buffer, mutate):
        name = self.paths[self.fd_paths[fd]].encode() + b"\0"
        values = {
            pad._JSIOCGNAME: name,
            pad._JSIOCGAXES: bytes([len(X_AXES)]),
            pad._JSIOCGBUTTONS: bytes([len(X_BUTTONS)]),
            pad._JSIOCGAXMAP: bytes(X_AXES),
            pad._JSIOCGBTNMAP: struct.pack(f"={len(X_BUTTONS)}H", *X_BUTTONS),
        }
        data = values[request]
        buffer[:len(data)] = data
        return 0

    def read(self, fd, count):
        self.read_count += 1
        queue = self.chunks.setdefault(self.fd_paths[fd], [])
        if not queue:
            raise BlockingIOError(errno.EAGAIN, "no pending events")
        chunk = queue.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def send(self, *chunks, path="/dev/input/js1"):
        self.chunks.setdefault(path, []).extend(chunks)


class MappingTests(unittest.TestCase):
    def test_xinput_semantic_map_prefers_rx_over_trigger(self):
        mapping = pad._mapping("Logitech Gamepad F710", None, None, X_AXES, X_BUTTONS)
        self.assertEqual(mapping["axes"], {"LEFT_X": 0, "LEFT_Y": 1, "RIGHT_X": 3})
        self.assertEqual(mapping["buttons"], {"a": 0, "b": 1, "start": 7})
        shuffled = [5, 3, 0, 2, 1]
        mapping = pad._mapping("device", 0x046D, 0xC21F, shuffled, X_BUTTONS[::-1])
        self.assertEqual(mapping["axes"], {"LEFT_X": 2, "LEFT_Y": 4, "RIGHT_X": 1})
        self.assertEqual(mapping["buttons"], {"a": 7, "b": 6, "start": 0})

    def test_known_directinput_mapping(self):
        mapping = pad._mapping("Logitech Cordless RumblePad 2", 0x046D, 0xC219,
                               [0, 1, 2, 5, 16, 17], list(range(0x120, 0x12C)))
        self.assertEqual(mapping["axes"]["RIGHT_X"], 2)
        self.assertEqual(mapping["buttons"], {"a": 1, "b": 2, "start": 9})
        by_name = pad._mapping("Logitech Gamepad F710", None, None,
                               [0, 1, 2, 5], list(range(0x120, 0x12C)))
        self.assertEqual(by_name["axes"]["RIGHT_X"], 2)
        with self.assertRaises(ValueError):
            pad._mapping("Logitech Gamepad F710", None, None, [0, 1, 2], [1] * 12)

    def test_missing_xinput_rx_never_falls_back_to_trigger(self):
        for vendor, product, buttons in ((0x046D, 0xC21F, X_BUTTONS),
                                         (None, None, X_BUTTONS),
                                         (0x046D, 0xC21F, list(range(0x120, 0x12C)))):
            with self.subTest(product=product, buttons=buttons), self.assertRaises(ValueError):
                pad._mapping("Logitech Gamepad F710", vendor, product,
                             [0, 1, 2, 4, 5], buttons)

    def test_unknown_or_incomplete_device_rejected(self):
        for name, axes, buttons in (("KEYBOARD", X_AXES, X_BUTTONS),
                                    ("Logitech Gamepad F710", [0, 1], X_BUTTONS),
                                    ("Logitech Gamepad F710", X_AXES, [0x130])):
            with self.subTest(name=name, axes=axes), self.assertRaises(ValueError):
                pad._mapping(name, None, None, axes, buttons)

    def test_deadzone_rescales_and_velocity_signs(self):
        self.assertEqual(pad.apply_deadzone(0.2, 0.2), 0)
        self.assertAlmostEqual(pad.apply_deadzone(-0.6, 0.2), -0.5)
        self.assertEqual(pad.apply_deadzone(2, 0.2), 1)
        self.assertEqual(pad.map_velocity((-1, -1, -1), 0.2), (0.8, 0.5, 1.0))
        self.assertEqual(pad.map_velocity((1, 1, 1), 0.2), (-0.5, -0.5, -1.0))
        self.assertEqual(pad.map_velocity((0.1, 0.1, 0.1), 0.2), (0, 0, 0))
        self.assertEqual(pad.map_velocity((-1, -1, -1), 0, 2, 3, 4, 5), (2, 4, 5))

    def test_validation(self):
        for value, dz in ((float("nan"), 0.2), (0, -0.1), (0, 1), (0, float("inf"))):
            with self.subTest(value=value, dz=dz), self.assertRaises(ValueError):
                pad.apply_deadzone(value, dz)
        for values in ((0, 0), (0, float("inf"), 0)):
            with self.assertRaises(ValueError):
                pad.map_velocity(values, 0.2)
        for field in ("forward", "backward", "lateral", "yaw"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    pad.map_velocity((0, 0, 0), 0.2, **{field: value})
        for interval in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                pad.F710Reader(scan_interval=interval)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.bus = FakeJoydev().start(self)
        self.reader = pad.F710Reader()
        self.addCleanup(self.reader.close)

    def test_initial_held_buttons_never_emit_edges(self):
        self.bus.send(initial_axes(), event(1, pad._INIT | pad._BUTTON, 0))
        sample = self.reader.poll()
        self.assertTrue(sample.connected and sample.ready)
        self.assertEqual(sample.pressed, frozenset())
        self.bus.send(event(1, pad._BUTTON, 0))
        self.assertFalse(self.reader.poll().pressed)
        self.bus.send(event(0, pad._BUTTON, 0), event(1, pad._BUTTON, 0),
                      event(1, pad._BUTTON, 1), event(1, pad._BUTTON, 7))
        self.assertEqual(self.reader.poll().pressed, frozenset(("a", "b", "start")))
        self.assertFalse(self.reader.poll().pressed)

    def test_partial_initialization_and_trigger_isolation(self):
        self.bus.send(event(-32768, pad._INIT | pad._AXIS, 0))
        sample = self.reader.poll()
        self.assertTrue(sample.connected)
        self.assertFalse(sample.ready)
        self.assertEqual(sample.axes, (0, 0, 0))
        self.bus.send(event(0, pad._INIT | pad._AXIS, 1),
                      event(0, pad._INIT | pad._AXIS, 3), event(32767, pad._AXIS, 2))
        self.assertEqual(self.reader.poll().axes, (-1, 0, 0))

    def test_idle_xinput_trigger_does_not_create_yaw(self):
        self.bus.send(initial_axes(), event(-32768, pad._INIT | pad._AXIS, 2))
        sample = self.reader.poll()
        self.assertTrue(sample.ready)
        self.assertEqual(sample.axes, (0, 0, 0))
        self.assertEqual(pad.map_velocity(sample.axes, 0.15), (0, 0, 0))

    def test_disconnect_reconnect_clears_stale_data_and_edges(self):
        self.bus.send(initial_axes(-32768, -32768, 32767))
        self.assertEqual(self.reader.poll().axes, (-1, -1, 1))
        self.bus.send(event(1, pad._BUTTON, 0), OSError(errno.ENODEV, "unplugged"))
        sample = self.reader.poll()
        self.assertFalse(sample.connected or sample.ready or sample.pressed)
        self.assertEqual(sample.axes, (0, 0, 0))
        self.assertFalse(self.reader.poll().connected)
        self.bus.clock = 1.0
        self.assertFalse(self.reader.poll().ready)
        self.bus.send(initial_axes(), event(1, pad._INIT | pad._BUTTON, 0))
        sample = self.reader.poll()
        self.assertTrue(sample.ready)
        self.assertEqual(sample.axes, (0, 0, 0))
        self.assertFalse(sample.pressed)

    def test_eof_and_io_error_disconnect(self):
        for termination in (b"", OSError(errno.EIO, "gone")):
            with self.subTest(termination=termination):
                self.bus.send(initial_axes(), termination)
                self.assertFalse(self.reader.poll().connected)
                self.bus.clock += 1.0

    def test_wrong_js0_is_rejected_and_real_f710_selected(self):
        self.bus.paths["/dev/input/js0"] = "KEYBOARD Consumer Control"
        sample = self.reader.poll()
        self.assertEqual(sample.path, "/dev/input/js1")
        self.assertEqual(len(self.bus.closed), 1)
        explicit = pad.F710Reader("/dev/input/js0")
        self.addCleanup(explicit.close)
        sample = explicit.poll()
        self.assertFalse(sample.connected)
        self.assertIn("not a recognized", sample.status)
        diagnostics = pad.list_gamepads()
        self.assertEqual([item["usable"] for item in diagnostics], [False, True])
        self.assertTrue(all(flags & os.O_NONBLOCK for flags in self.bus.open_flags))
        self.assertTrue(all(flags & os.O_ACCMODE == os.O_RDONLY for flags in self.bus.open_flags))

    def test_partial_and_malformed_events_and_bounded_poll(self):
        blob = initial_axes() + event(1, 0x7F, 0) + event(1, pad._AXIS, 255)
        self.bus.send(blob[:3])
        self.assertFalse(self.reader.poll().ready)
        self.bus.send(blob[3:11], blob[11:], event(3, pad._BUTTON, 0))
        sample = self.reader.poll()
        self.assertTrue(sample.ready)
        self.assertFalse(sample.pressed)
        self.bus.send(*[event(100, pad._AXIS, 0)] * 100)
        before = self.bus.read_count
        self.reader.poll()
        self.assertEqual(self.bus.read_count - before, pad._MAX_READS)
        self.assertLess(len(self.reader._buffer), pad._EVENT.size)

    def test_close_is_idempotent_and_never_reopens(self):
        self.reader.poll()
        self.reader.close()
        count = len(self.bus.closed)
        self.reader.close()
        self.bus.clock = 100.0
        self.assertFalse(self.reader.poll().connected)
        self.assertEqual(len(self.bus.closed), count)


if __name__ == "__main__":
    unittest.main()
