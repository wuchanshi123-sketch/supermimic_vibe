"""Extract SONIC's released g1 encoder/decoder weights from the ONNX files into a PyTorch
state dict for supermimic.sonic_net.SonicNet, and check the module against the ONNX graphs
(onnx.reference, on random inputs in g1 mode).

usage: convert_sonic.py <dir with model_encoder.onnx, model_decoder.onnx> out.pt
Needs the `onnx` package.
"""

import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from supermimic.sonic_net import DEC_IN, ENC_IN, SonicNet


def tensors(graph):
  t = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
  for n in graph.node:
    if n.op_type == "Constant":
      for a in n.attribute:
        if a.name == "value":
          t[n.output[0]] = numpy_helper.to_array(a.t)
  return t


def main():
  src, out = sys.argv[1], sys.argv[2]
  enc_m = onnx.load(f"{src}/model_encoder.onnx")
  dec_m = onnx.load(f"{src}/model_decoder.onnx")
  net = SonicNet()
  sd = {}
  te = tensors(enc_m.graph)
  gemms = [n for n in enc_m.graph.node if n.op_type == "Gemm" and n.output[0].startswith("/g1/")]
  assert len(gemms) == 5, len(gemms)
  for k, n in enumerate(gemms):
    assert any(a.name == "transB" and a.i == 1 for a in n.attribute)
    sd[f"enc.{k}.weight"] = torch.tensor(te[n.input[1]])
    sd[f"enc.{k}.bias"] = torch.tensor(te[n.input[2]])
  qn = [n for n in enc_m.graph.node if n.output[0].startswith("/quantizer/")]
  def qconst(op, k=0):
    ns = [n for n in qn if n.op_type == op]
    return torch.tensor(te[[i for i in ns[k].input if i in te][0]]).reshape(-1)
  sd["fsq_shift"] = qconst("Add", 0)
  sd["fsq_half_l"] = qconst("Mul", 0)
  sd["fsq_offset"] = qconst("Sub", 0)
  sd["fsq_half_width"] = qconst("Div", 0)
  td = tensors(dec_m.graph)
  mms = [n for n in dec_m.graph.node if n.op_type == "MatMul"]
  adds = [n for n in dec_m.graph.node if n.op_type == "Add"]
  assert len(mms) == 7 and len(adds) == 7
  for k, (mm, ad) in enumerate(zip(mms, adds)):
    sd[f"dec.{k}.weight"] = torch.tensor(td[mm.input[1]]).T.contiguous()
    sd[f"dec.{k}.bias"] = torch.tensor(td[[i for i in ad.input if i in td][0]])
  net.load_state_dict(sd)
  net.eval()

  # Check against the ONNX graphs.
  rng = np.random.default_rng(0)
  ref_e = ReferenceEvaluator(enc_m)
  ref_d = ReferenceEvaluator(dec_m)
  worst_e = worst_d = 0.0
  for trial in range(4):
    x = rng.normal(size=(1, ENC_IN)).astype(np.float32) * 0.5
    x[0, 0:4] = 0.0  # g1 mode
    if trial % 2:
      x[0, 661:] = 0.0  # unused fields zero, as the deploy code fills them
    pre_name = "/g1/module/module.8/Gemm_output_0"
    e_ref, pre_ref = ref_e.run(["encoded_tokens", pre_name], {"obs_dict": x})
    xt = torch.tensor(x)
    B = 1
    feat = torch.cat([xt[:, 4:584].view(B, 10, 58), xt[:, 601:661].view(B, 10, 6)], -1).reshape(B, 640)
    pre_t = net._mlp(net.enc, feat).detach().numpy()
    e_t = net.encode_obs(xt).detach().numpy()
    worst_e = max(worst_e, float(np.abs(pre_ref.reshape(1, -1) - pre_t).max()))
    flips = int((np.abs(e_ref.reshape(1, -1) - e_t) > 1e-6).sum())
    print(f"  trial {trial}: pre-quantizer max diff {np.abs(pre_ref.reshape(1, -1) - pre_t).max():.2e}, token codes differing {flips}/64")
    assert flips <= 1, flips
    h = rng.normal(size=(1, DEC_IN)).astype(np.float32) * 0.5
    h[0, :64] = e_ref
    d_ref = ref_d.run(None, {"obs_dict": h})[0]
    d_t = net.decode(torch.tensor(h[:, :64]), torch.tensor(h[:, 64:])).detach().numpy()
    worst_d = max(worst_d, float(np.abs(d_ref.reshape(1, -1) - d_t).max()))
  print(f"max |torch - onnx|: encoder (pre-quantizer) {worst_e:.2e}, decoder {worst_d:.2e}")
  assert worst_e < 1e-4 and worst_d < 1e-3, (worst_e, worst_d)
  torch.save(net.state_dict(), out)
  print("saved", out)


if __name__ == "__main__":
  main()
