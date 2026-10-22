"""SONIC's released g1 policy (GEAR-SONIC model_encoder.onnx / model_decoder.onnx) as a
batched PyTorch module, for scoring it on the same benchmark as SuperMimic (App. F).

The ONNX export has batch size 1; this re-implements the parts used in g1 mode:
  encoder: obs_dict (1762) -> g1 branch MLP on [10 frames x (58 joint values + 6 anchor
           orientation values)] -> 64 -> FSQ quantizer (2 x 32 codes) -> token (64)
  decoder: [token (64), proprioceptive history (930)] -> MLP -> action (29, IsaacLab order)
code/scripts/convert_sonic.py extracts the weights and checks this module against the ONNX
graphs.

Encoder input layout (observation_config.yaml order, zeros for observations the g1 mode does
not use): [0:4] encoder mode (g1 = 0) | [4:294] reference joint positions at t, t+5, ..., t+45
(IsaacLab order) | [294:584] reference joint velocities, same frames | ... | [601:661]
reference root orientation relative to the robot pelvis at the same frames, first two columns
of the rotation matrix, row-major | ... (1762 in total).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENC_IN = 1762
DEC_IN = 994
TOKEN = 64


class SonicNet(nn.Module):
  def __init__(self):
    super().__init__()
    enc = (640, 2048, 1024, 512, 512, 64)
    dec = (994, 2048, 2048, 1024, 1024, 512, 512, 29)
    self.enc = nn.ModuleList(nn.Linear(a, b) for a, b in zip(enc[:-1], enc[1:]))
    self.dec = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dec[:-1], dec[1:]))
    # FSQ quantizer: round(tanh(x + shift) * half_l - offset) / half_width.
    self.register_buffer("fsq_shift", torch.zeros(32))
    self.register_buffer("fsq_half_l", torch.ones(32))
    self.register_buffer("fsq_offset", torch.zeros(32))
    self.register_buffer("fsq_half_width", torch.ones(32))

  @staticmethod
  def _mlp(layers, x):
    for k, lin in enumerate(layers):
      x = lin(x)
      if k < len(layers) - 1:
        x = F.silu(x)
    return x

  def encode_g1(self, jpos_jvel, anchor6):
    """jpos_jvel: (B, 580) = 10 frames of joint positions then 10 of joint velocities;
    anchor6: (B, 60). Returns the 64-d token."""
    B = jpos_jvel.shape[0]
    x = torch.cat([jpos_jvel.view(B, 10, 58), anchor6.view(B, 10, 6)], dim=-1).reshape(B, 640)
    z = self._mlp(self.enc, x).view(B, 2, 32)
    z = torch.tanh(z + self.fsq_shift) * self.fsq_half_l - self.fsq_offset
    return (torch.round(z) / self.fsq_half_width).reshape(B, TOKEN)

  def encode_obs(self, obs):
    """Full 1762-d encoder observation in g1 mode (used to check against the ONNX graph)."""
    return self.encode_g1(obs[:, 4:584], obs[:, 601:661])

  def decode(self, token, hist):
    return self._mlp(self.dec, torch.cat([token, hist], dim=-1))
