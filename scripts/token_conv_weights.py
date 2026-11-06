"""RMS of the trained token_conv kernels, per layer and per offset (index K-1 multiplies the current
position, index 0 the oldest one), from run checkpoints. The kernels start at zero.

    python scripts/token_conv_weights.py runs/sweeps/stack/stack_s1337 [...] > results/token_conv_weights.json
"""
import json
import sys

import torch

out = {}
for d in sys.argv[1:]:
    sd = torch.load(d + "/model.pt", map_location="cpu", weights_only=False)["model"]
    ks = sorted(k for k in sd if k.endswith("conv.weight"))
    out["/".join(d.rstrip("/").split("/")[-2:])] = {
        k.split(".")[1]: {"rms": sd[k].float().pow(2).mean().sqrt().item(),
                          "rms_by_offset": sd[k].float().pow(2).mean((0, 1)).sqrt().tolist()} for k in ks}
print(json.dumps(out, indent=1))
