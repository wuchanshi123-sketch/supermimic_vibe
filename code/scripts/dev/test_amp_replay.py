"""CPU integration checks for AMP actor bounds and clean episode-aware histories."""

import unittest
from types import SimpleNamespace

import torch

from supermimic.amp_networks import CommandActor
from supermimic.amp_replay import AMPReplay
from supermimic.amp_task import AMPTask


class ReplayTaskBase:
    def __init__(self):
        self.q0 = torch.zeros(29)
        self.qmin = -torch.ones(29)
        self.qmax = torch.ones(29)
        self.kin = SimpleNamespace(body_names=["pelvis", "torso_link"])


def episode(env, t):
    if env == 0:
        return 0 if t < 3 else (1 if t < 7 else 2)
    return 10 if t < 5 else 11


def episode_start(env, t):
    if env == 0:
        return 0 if t < 3 else (3 if t < 7 else 7)
    return 0 if t < 5 else 5


def add_row(replay, t):
    n = replay.n
    s = torch.zeros(n, 71)
    s[:, 2] = 0.8
    s[:, 3] = 1.0
    for e in range(n):
        s[e, 7:36] = t * 100 + e
    successor = s.clone()
    successor[:, 7:36] += 0.25
    executed = torch.full((n, 29), 10000.0 + t)
    xi = torch.zeros(n, 33)
    xi[:, 0] = 0.75
    xi[:, 1:30] = 0.001 * t
    cmd = torch.tensor([[t * 0.1, e * 0.2, -t * 0.05] for e in range(n)])
    previous = torch.stack([torch.full((29,), t * 10.0 + e) for e in range(n)])
    clean = previous + 0.5
    replay.add(
        s, executed, successor, xi, torch.full((n,), t, dtype=torch.long),
        torch.tensor([episode(e, t) for e in range(n)]),
    )
    replay.add_amp(cmd, previous, clean)


class AMPReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.task = AMPTask(ReplayTaskBase())

    def assert_histories(self, capacity, total, n):
        replay = AMPReplay(n, capacity, 33, "cpu")
        for t in range(total):
            add_row(replay, t)
        start, xi, command, history, previous = replay.sample_starts_amp(256, 4, self.task)
        self.assertEqual(history.shape, (256, 4, 96))
        seen = set()
        for i in range(len(start)):
            tag = int(round(float(start[i, 7])))
            t, e = divmod(tag, 100)
            seen.add((t, e))
            first = max(0, total - capacity, episode_start(e, t))
            times = [max(first, j) for j in range(t - 3, t + 1)]
            expected_q = torch.tensor([j * 100 + e + j * 0.001 for j in times])
            torch.testing.assert_close(history[i, :, 9], expected_q, atol=2e-4, rtol=1e-6)
            expected_command = torch.tensor([[j * 0.1, e * 0.2, -j * 0.05] for j in times])
            torch.testing.assert_close(history[i, :, 6:9], expected_command)
            expected_previous = torch.tensor([j * 10.0 + e for j in times])
            torch.testing.assert_close(history[i, :, 67], expected_previous)
            torch.testing.assert_close(previous[i], torch.full((29,), t * 10.0 + e))
            torch.testing.assert_close(command[i], expected_command[-1])
            torch.testing.assert_close(xi[i, 1:30], torch.full((29,), t * 0.001))
        self.assertEqual(len(seen), min(capacity, total) * n)
        # Every sampled physical successor belongs to its action, not the next
        # row's potentially reset state.
        s, s1 = replay.sample_transitions(256)
        torch.testing.assert_close(s1[:, 7:36] - s[:, 7:36], torch.full((256, 29), 0.25))
        return replay

    def test_short_buffer_pads_initial_history(self):
        self.assert_histories(capacity=8, total=3, n=1)

    def test_ring_wrap_and_episode_boundaries(self):
        replay = self.assert_histories(capacity=5, total=9, n=2)
        self.assertEqual(replay.ptr, 4)
        self.assertEqual(replay.size, 5)
        torch.testing.assert_close(replay.clean_action[3, 0], torch.full((29,), 80.5))
        self.assertTrue(bool((replay.a > 1000).all()))

    def test_real_terminal_successor_retained(self):
        replay = AMPReplay(1, 8, 33, "cpu")
        for t in range(4):
            add_row(replay, t)
        s, s1 = replay.sample_transitions(256)
        old_terminal = s[:, 7] == 200
        self.assertTrue(bool(old_terminal.any()))
        torch.testing.assert_close(s1[old_terminal, 7], torch.full_like(s1[old_terminal, 7], 200.25))


class CommandActorTest(unittest.TestCase):
    def test_forward_gradients_and_saturation_bounds(self):
        actor = CommandActor(hidden=(32, 16))
        obs = torch.randn(7, 4 * 96, requires_grad=True)
        actor.obs_norm.update(obs.detach())
        actions = actor(obs)
        self.assertEqual(actions.shape, (7, 29))
        self.assertTrue(bool(torch.isfinite(actions).all()))
        self.assertTrue(bool((actions.abs() <= 3).all()))
        actions.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(obs.grad).all()))
        self.assertGreater(float(obs.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.net.parameters()))
        with torch.no_grad():
            actor.net[-1].weight.zero_()
            actor.net[-1].bias.fill_(1000)
        torch.testing.assert_close(actor(obs.detach()), torch.full((7, 29), 3.0))
        with torch.no_grad():
            actor.net[-1].bias.fill_(-1000)
        torch.testing.assert_close(actor(obs.detach()), torch.full((7, 29), -3.0))


if __name__ == "__main__":
    unittest.main()
