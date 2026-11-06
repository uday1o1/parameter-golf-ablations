"""Bits per byte on a held-out slice of the TRAINING shard ("dev" set), for choosing techniques without
looking at the validation tokens the results are reported on (GPU).

    python scripts/dev_eval.py runs/sweeps/ablation/baseline_s1337 [more run dirs ...]

Training reads the first shard sequentially from its start, steps x grad_accum x (batch x seq_len + 1)
tokens (40,965,000 for the L5 budget). The dev set is 2,097,152 tokens starting at token 50,000,000 of
the same shard, which no run reads; it is scored like the validation set (non-overlapping windows of
the training sequence length). Writes <run_dir>/dev.json.
"""
import json
import sys
from pathlib import Path

import torch

from nanolm.config import RunConfig, from_dict
from nanolm.data import byte_lut_tensors, read_shard
from nanolm.device import get_device
from nanolm.evaluate import evaluate
from nanolm.models import build_model

DEV_OFFSET, DEV_TOKENS = 50_000_000, 2_097_152


def main(run_dirs):
    for run in map(Path, run_dirs):
        if (run / "dev.json").exists():
            continue
        ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        cfg = from_dict(RunConfig, ckpt["config"])
        t = cfg.train
        read = t.max_steps * t.grad_accum * (t.batch_size * t.seq_len + 1)
        assert read <= DEV_OFFSET, f"{run}: training read {read:,} tokens, past the dev offset"
        dinfo = get_device("auto", cfg.precision)
        model = build_model(cfg.model).to(dinfo.device)
        model.recur_active = bool(cfg.model.recur_layers)          # recurrence is on at the end of training
        model.load_state_dict(ckpt["model"])
        shard = read_shard(ckpt["data_meta"]["train_files"][0])
        dev = shard[DEV_OFFSET: DEV_OFFSET + DEV_TOKENS + 1]
        T = cfg.eval.eval_seq_len or t.seq_len
        r = evaluate(model, dev, T, cfg.eval.batch_size, byte_lut_tensors(ckpt["data_meta"], dinfo.device), dinfo)
        out = {"dev_bpb": r["bpb"], "dev_loss": r["loss"], "dev_tokens": r["tokens"], "dev_offset": DEV_OFFSET,
               "train_tokens_read": read, "shard": ckpt["data_meta"]["train_files"][0]}
        (run / "dev.json").write_text(json.dumps(out, indent=1))
        print(run.name, json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
