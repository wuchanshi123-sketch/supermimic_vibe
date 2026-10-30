"""CPU-only F710 playback checks; no joydev, checkpoint, physics, or window.

Run with ``source scripts/local_runtime.sh`` then
``"$SMIMIC_PY" code/scripts/dev/test_amp_gamepad_playback.py``.
"""
from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import play_amp_walk as live
from supermimic.amp_gamepad import GamepadSample


def sample(axes=(0., 0., 0.), pressed=(), *, connected=True, ready=True,
           path="/dev/input/js-test"):
    return GamepadSample(
        connected=connected, ready=ready and connected, axes=axes,
        pressed=frozenset(pressed), path=path if connected else None,
        name="Fake F710" if connected else None,
        status="ready" if connected and ready else "unavailable",
    )


class FakeReader:
    """Deliver already filtered button edges, as the real backend does."""

    def __init__(self, *samples):
        self.samples = deque(samples)
        self.current = sample(connected=False)
        self.closed = False
        self.polls = 0

    def poll(self):
        if self.closed:
            raise AssertionError("poll after close")
        self.polls += 1
        if self.samples:
            self.current = self.samples.popleft()
        else:
            self.current = replace(self.current, pressed=frozenset())
        return self.current

    def close(self):
        self.closed = True


class InputTests(unittest.TestCase):
    def setUp(self):
        self.reader = FakeReader()
        self.gamepad = live.AMPGamepadInput(self.reader)
        self.controls = live.AMPControls()
        self.addCleanup(self.gamepad.close)

    def poll(self, value):
        self.reader.samples.append(value)
        return self.gamepad.poll(self.controls)

    def assert_command(self, expected):
        np.testing.assert_allclose(self.controls.command, expected, atol=1e-7, rtol=0)

    def drive(self):
        self.poll(sample())
        self.poll(sample((0., -1., 0.)))
        self.assert_command((0.8, 0., 0.))

    def test_initial_held_sticks_need_neutral_before_axes_take_control(self):
        self.assertEqual(self.poll(sample((-1., -1., 1.))), ["command"])
        self.assert_command((0., 0., 0.))
        self.assertTrue(self.gamepad.awaiting_neutral)
        self.assertEqual(self.poll(sample((-1., -1., 1.))), [])
        # Values inside the deadzone count as centered.
        self.assertEqual(self.poll(sample((0.1, -0.1, 0.1))), [])
        self.assertFalse(self.gamepad.awaiting_neutral)
        self.assertEqual(self.poll(sample((-1., -1., 1.))), ["command"])
        self.assert_command((0.8, 0.5, -1.))
        self.assertEqual(self.gamepad.label, "F710")
        self.poll(sample((1., 1., -1.)))
        self.assert_command((-0.5, -0.5, 1.))

    def test_b_stops_and_held_stick_cannot_resume_until_centered(self):
        self.drive()
        self.assertEqual(self.poll(sample((0., -1., 0.), ("b",))), ["command"])
        self.assert_command((0., 0., 0.))
        self.assertTrue(self.gamepad.awaiting_neutral)
        for _ in range(3):
            self.assertEqual(self.poll(sample((0., -1., 0.))), [])
            self.assert_command((0., 0., 0.))
        self.poll(sample())
        self.assertEqual(self.poll(sample((0., -1., 0.))), ["command"])
        self.assert_command((0.8, 0., 0.))

    def test_a_stops_and_emits_one_restart_for_one_press_edge(self):
        self.drive()
        self.assertEqual(self.poll(sample((0., -1., 0.), ("a",))),
                         ["command", "restart"])
        self.assert_command((0., 0., 0.))
        for _ in range(3):
            self.assertEqual(self.poll(sample((0., -1., 0.))), [])
        self.assertTrue(self.gamepad.awaiting_neutral)
        self.poll(sample())
        self.poll(sample((0., -1., 0.)))
        self.assert_command((0.8, 0., 0.))

    def test_start_toggles_pause_only_for_backend_press_edges(self):
        self.poll(sample())
        self.assertEqual(self.poll(sample(pressed=("start",))), ["pause"])
        self.assertTrue(self.controls.paused)
        for _ in range(3):
            self.assertEqual(self.poll(sample()), [])
            self.assertTrue(self.controls.paused)
        self.assertEqual(self.poll(sample(pressed=("start",))), ["pause"])
        self.assertFalse(self.controls.paused)

    def test_keyboard_takeover_wins_until_g_toggle_and_neutral(self):
        self.drive()
        self.assertEqual(self.controls.key(live.glfw.KEY_S), "command")
        self.gamepad.keyboard_takeover()
        keyboard_command = self.controls.command.copy()
        self.assertEqual(self.poll(sample((1., 1., 1.), ("a", "b", "start"))), [])
        np.testing.assert_array_equal(self.controls.command, keyboard_command)
        self.assertFalse(self.controls.paused)
        self.assertFalse(self.gamepad.enabled)
        self.assertEqual(self.controls.key(live.glfw.KEY_G), "gamepad")
        self.gamepad.toggle(self.controls)
        self.assertTrue(self.gamepad.enabled)
        self.assert_command((0., 0., 0.))
        self.poll(sample((1., 1., 1.)))
        self.assert_command((0., 0., 0.))
        self.poll(sample())
        self.poll(sample((1., 1., 1.)))
        self.assert_command((-0.5, -0.5, -1.))

    def test_disconnect_stops_active_input_and_reconnect_requires_neutral(self):
        self.drive()
        self.assertEqual(self.poll(sample(connected=False)), ["command"])
        self.assert_command((0., 0., 0.))
        self.assertTrue(self.gamepad.awaiting_neutral)
        self.assertEqual(self.gamepad.disconnects, 1)
        self.assertEqual(self.poll(sample(connected=False)), [])
        self.assertEqual(self.gamepad.disconnects, 1)
        self.poll(sample((0., -1., 0.), path="/dev/input/js-reconnected"))
        self.assert_command((0., 0., 0.))
        self.poll(sample(path="/dev/input/js-reconnected"))
        self.poll(sample((0., -1., 0.), path="/dev/input/js-reconnected"))
        self.assert_command((0.8, 0., 0.))

    def test_disconnect_does_not_zero_keyboard_owned_command(self):
        self.drive()
        self.gamepad.keyboard_takeover()
        self.controls.command[:] = (0.3, 0.1, -0.2)
        self.assertEqual(self.poll(sample(connected=False)), [])
        self.assert_command((0.3, 0.1, -0.2))
        self.poll(sample((0., -1., 0.)))
        self.assert_command((0.3, 0.1, -0.2))
        self.assertFalse(self.gamepad.enabled)

    def test_center_stops_exactly_even_for_change_below_tolerance(self):
        self.poll(sample())
        self.poll(sample((0., -0.153, 0.)))
        self.poll(sample((0., -0.1505, 0.)))
        self.assertGreater(self.controls.command[0], 0.)
        self.assertLess(self.controls.command[0], 0.001)
        self.assertEqual(self.poll(sample()), ["command"])
        np.testing.assert_array_equal(self.controls.command, np.zeros(3))

    def test_unready_sample_stops_and_ignores_buttons_until_initialized(self):
        self.drive()
        self.assertEqual(self.poll(sample((1., 1., 1.), ("a", "start"), ready=False)),
                         ["command"])
        self.assert_command((0., 0., 0.))
        self.assertFalse(self.controls.paused)
        self.assertTrue(self.gamepad.awaiting_neutral)
        self.poll(sample((1., 1., 1.)))
        self.assert_command((0., 0., 0.))
        self.poll(sample())
        self.poll(sample((1., 1., 1.)))
        self.assert_command((-0.5, -0.5, -1.))

    def test_close_releases_reader(self):
        self.gamepad.close()
        self.assertTrue(self.reader.closed)


def fake_env():
    return SimpleNamespace(
        fixed_command=torch.zeros(3), command=torch.zeros(1, 3),
        command_age=torch.tensor([7.]), obs_hist=torch.full((1, 4, 3), 9.),
        state={}, prev_action=torch.zeros(1, 1), xi=torch.zeros(1, 1),
        amp_task=SimpleNamespace(observation=Mock(
            side_effect=lambda state, action, command, xi: command.clone())),
    )


class SetCommandTests(unittest.TestCase):
    def setUp(self):
        # Exercise the actual method without constructing AMPEnv or a policy.
        self.playback = live.AMPPlayback.__new__(live.AMPPlayback)
        self.playback.device = "cpu"
        self.playback.env = fake_env()
        self.playback.command = np.zeros(3, dtype=np.float32)
        self.playback.done = True
        self.playback.failed = True
        self.playback.done_since = 12.5
        self.playback.robot_pose = np.array([1., 2., 0.3], dtype=np.float32)
        self.playback.reset = Mock()

    def test_gamepad_command_preserves_done_pose_and_hold_timer(self):
        self.playback.set_command((0.6, -0.2, 0.4), announce=False, reset_done=False)
        self.playback.reset.assert_not_called()
        self.assertTrue(self.playback.done)
        self.assertTrue(self.playback.failed)
        self.assertEqual(self.playback.done_since, 12.5)
        np.testing.assert_array_equal(self.playback.robot_pose,
                                      np.array([1., 2., 0.3], dtype=np.float32))
        torch.testing.assert_close(self.playback.env.fixed_command,
                                   torch.tensor([0.6, -0.2, 0.4]))
        torch.testing.assert_close(self.playback.env.command, torch.zeros(1, 3))
        torch.testing.assert_close(self.playback.env.command_age, torch.tensor([7.]))
        torch.testing.assert_close(self.playback.env.obs_hist, torch.full((1, 4, 3), 9.))
        self.playback.env.amp_task.observation.assert_not_called()

    def test_keyboard_default_still_resets_ended_episode(self):
        self.playback.set_command((0.3, 0., 0.))
        self.playback.reset.assert_called_once_with()
        torch.testing.assert_close(self.playback.env.fixed_command, torch.tensor([0.3, 0., 0.]))

    def test_live_change_refreshes_only_latest_observation(self):
        self.playback.done = False
        self.playback.set_command((0.3, -0.1, 0.2), announce=False, reset_done=False)
        self.playback.reset.assert_not_called()
        torch.testing.assert_close(self.playback.env.command, torch.tensor([[0.3, -0.1, 0.2]]))
        torch.testing.assert_close(self.playback.env.command_age, torch.zeros(1))
        torch.testing.assert_close(self.playback.env.obs_hist[:, :-1], torch.full((1, 3, 3), 9.))
        torch.testing.assert_close(self.playback.env.obs_hist[:, -1], self.playback.env.command)


class DefaultCheckpointTests(unittest.TestCase):
    def test_available_validated_checkpoint_is_preferred(self):
        with patch.object(Path, "is_file", autospec=True, return_value=True) as exists:
            self.assertEqual(live.default_amp_checkpoint(), Path("runs/amp_walk_best/best.pt"))
        exists.assert_called_once_with(Path("runs/amp_walk_best/best.pt"))

    def test_missing_validated_checkpoint_falls_back_to_training_checkpoint(self):
        with patch.object(Path, "is_file", autospec=True, return_value=False) as exists:
            self.assertEqual(live.default_amp_checkpoint(),
                             Path("runs/amp_walk_24gb_train/last.pt"))
        exists.assert_called_once_with(Path("runs/amp_walk_best/best.pt"))


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += max(duration, 0.001)


class FakePlayback(live.AMPPlayback):
    """Use real command updates with a deterministic stand-in for physics."""

    def __init__(self, clock, *, fall_steps=()):
        self.clock = clock
        self.fall_steps = set(fall_steps)
        self.device = "cpu"
        self.env = fake_env()
        self.command = np.zeros(3, dtype=np.float32)
        self.checkpoint = Path("fake-checkpoint.pt")
        self.checkpoint_iteration = 42
        self.actor_updates = 123
        self.lib = SimpleNamespace(dt=0.02)
        self.total_steps = self.steps = self.restarts = 0
        self.done = self.failed = False
        self.done_since = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.yaw_rate = 0.
        self.changes = []
        self.reset_times = []
        self.reset_commands = []

    def set_command(self, command, *, announce=True, reset_done=True):
        self.changes.append((self.clock.now, self.done, announce, reset_done))
        super().set_command(command, announce=announce, reset_done=reset_done)

    def reset(self):
        self.restarts += 1
        self.reset_times.append(self.clock.now)
        self.reset_commands.append(self.command.copy())
        self.done = self.failed = False
        self.done_since = None
        self.steps = 0

    def step(self):
        if self.done:
            raise AssertionError("stepped a held terminal state")
        self.total_steps += 1
        self.steps += 1
        if self.total_steps in self.fall_steps:
            self.done = self.failed = True
            self.done_since = self.clock.now


class FakeDisplay:
    def update(self, *args, **kwargs):
        raise AssertionError("headless playback must not access the display")


class PlayLoopTests(unittest.TestCase):
    def run_play(self, samples, *, fall_steps=(), key_schedule=(), **options):
        clock = FakeClock()
        pending_keys = deque(key_schedule)
        keys = SimpleNamespace(
            empty=lambda: not pending_keys or pending_keys[0][0] > clock.now,
            get=lambda: pending_keys.popleft()[1],
        )
        reader = FakeReader(*samples)
        gamepad = live.AMPGamepadInput(reader)
        playback = FakePlayback(clock, fall_steps=fall_steps)
        controls = live.AMPControls(command=np.zeros(3, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(headless=True, duration=0.3, loop=True, hold=1.,
                                   max_steps=0, report=Path(directory) / "report.json")
            for key, value in options.items():
                setattr(args, key, value)
            output = io.StringIO()
            with patch.object(live.time, "monotonic", clock.monotonic), \
                    patch.object(live.time, "sleep", clock.sleep), \
                    patch.object(live.queue, "SimpleQueue", return_value=keys), \
                    patch.object(live.mujoco.viewer, "launch_passive",
                                 side_effect=AssertionError("native window requested")), \
                    redirect_stdout(output):
                live.play(args, playback, controls, FakeDisplay(), gamepad=gamepad)
            report = json.loads(args.report.read_text())
        self.assertTrue(reader.closed)
        self.assertFalse(report["gui_opened"])
        self.assertTrue(report["viewer_threads_joined"])
        self.assertEqual(report["gamepad"], gamepad.report())
        line = next(line for line in output.getvalue().splitlines()
                    if line.startswith("AMP_PLAYBACK_REPORT "))
        self.assertEqual(json.loads(line.split(" ", 1)[1]), report)
        return playback, controls, report, reader

    def test_command_changes_after_fall_do_not_bypass_hold(self):
        playback, _, report, reader = self.run_play(
            [sample(), sample((0., -1., 0.)), sample((1., 1., -1.)), sample()],
            fall_steps=(1,), duration=0.2, hold=1.,
        )
        self.assertGreaterEqual(reader.polls, 4)
        self.assertEqual(playback.total_steps, 1)
        self.assertEqual(playback.restarts, 0)
        self.assertTrue(playback.done)
        self.assertEqual(playback.done_since, 0.)
        self.assertGreaterEqual(len(playback.changes), 3)
        self.assertTrue(all(done and not announce and not reset_done
                            for _, done, announce, reset_done in playback.changes))
        self.assertTrue(report["failed"])
        self.assertEqual(report["steps"], 1)

    def test_loop_resumes_only_after_hold_then_max_steps_closes_reader(self):
        playback, _, report, _ = self.run_play(
            [sample(), sample((0., -1., 0.))], fall_steps=(1,), hold=0.1, max_steps=2,
        )
        self.assertEqual(playback.total_steps, 2)
        self.assertEqual(playback.restarts, 1)
        self.assertGreaterEqual(playback.reset_times[0], 0.1)
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["restarts"], 1)
        np.testing.assert_allclose(report["command"], (0.8, 0., 0.))

    def test_no_loop_exits_after_fall_and_writes_report(self):
        playback, _, report, _ = self.run_play(
            [sample(), sample((0., -1., 0.))], fall_steps=(1,), loop=False,
        )
        self.assertEqual(playback.total_steps, 1)
        self.assertEqual(playback.restarts, 0)
        self.assertTrue(report["failed"])
        self.assertEqual(report["checkpoint_iteration"], 42)
        self.assertEqual(report["actor_updates"], 123)

    def test_a_resets_once_at_zero_and_held_stick_does_not_restart_motion(self):
        playback, controls, report, _ = self.run_play(
            [sample(), sample((0., -1., 0.), ("a",)), sample((0., -1., 0.))],
            fall_steps=(1,), hold=10., max_steps=3,
        )
        self.assertEqual(playback.restarts, 1)
        np.testing.assert_array_equal(playback.reset_commands[0], np.zeros(3))
        np.testing.assert_array_equal(controls.command, np.zeros(3))
        self.assertEqual(report["steps"], 3)

    def test_keyboard_and_g_events_route_ownership_in_play_loop(self):
        playback, controls, report, _ = self.run_play(
            [sample((0., -1., 0.)), sample(), sample((0., -1., 0.)),
             sample(), sample((0., -1., 0.))],
            key_schedule=((0., live.glfw.KEY_W), (0.04, live.glfw.KEY_G)),
            max_steps=5,
        )
        self.assertEqual(len(playback.changes), 3)
        self.assertEqual(playback.changes[0], (0., False, True, True))
        self.assertEqual(playback.changes[1], (0.04, False, True, False))
        self.assertEqual(playback.changes[2], (0.08, False, False, False))
        np.testing.assert_allclose(controls.command, (0.8, 0., 0.))
        self.assertTrue(report["gamepad"]["enabled"])

    def test_reader_closes_when_physics_step_raises(self):
        reader = FakeReader(sample())
        gamepad = live.AMPGamepadInput(reader)
        playback = FakePlayback(FakeClock())
        playback.step = Mock(side_effect=RuntimeError("fake step failed"))
        args = SimpleNamespace(headless=True, duration=0., max_steps=1, loop=True,
                               hold=1., report=None)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "fake step failed"):
            live.play(args, playback, live.AMPControls(), FakeDisplay(), gamepad=gamepad)
        self.assertTrue(reader.closed)


if __name__ == "__main__":
    unittest.main()
