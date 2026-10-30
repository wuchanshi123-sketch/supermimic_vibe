"""Command-conditioned policy for learning AMP locomotion from random weights."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .networks import Normalizer, mlp, run_mlp


class CommandActor(nn.Module):
    """Map proprioception/command history to bounded normalized PD setpoints."""

    def __init__(
        self, history=4, obs_dim=96, hidden=(512, 256, 128), action_limit=3.0,
    ):
        super().__init__()
        if history < 1 or obs_dim < 1:
            raise ValueError("history and obs_dim must be positive")
        if not math.isfinite(action_limit) or action_limit <= 0:
            raise ValueError("action_limit must be finite and positive")
        self.history = int(history)
        self.obs_dim = int(obs_dim)
        self.action_limit = float(action_limit)
        self.obs_norm = Normalizer(self.history * self.obs_dim)
        self.net = mlp(self.history * self.obs_dim, hidden, 29, out_gain=0.01)

    def forward(self, flat_obs):
        raw = run_mlp(self.net, self.obs_norm(flat_obs))
        # Unit slope around zero preserves the small random initialization.
        return self.action_limit * torch.tanh(raw / self.action_limit)
