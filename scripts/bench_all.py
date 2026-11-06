"""Training speed of every arm of the ablation and stack sweeps on its own (GPU), with nanolm.bench: a few
real training steps on random tokens, nothing else running. The sweeps share the GPU between runs, so their recorded times
are not comparable; these are.

    python scripts/bench_all.py results/bench.json
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ["ablation", "stack"]


def arms():
    for s in SWEEPS:
        p = ROOT / "configs" / "sweeps" / f"{s}.yaml"
        if not p.exists():
            continue
        spec = yaml.safe_load(p.read_text())
        base = str((p.parent / spec["base"]).resolve())
        for r in spec["runs"]:
            ov = {**spec.get("common", {}), **(r.get("overrides") or {})}
            yield f"{s}/{r['name']}", base, [f"{k}={json.dumps(v)}" for k, v in ov.items()]


def main(out):
    res = {}
    for name, base, ov in arms():
        txt = subprocess.run([sys.executable, "-m", "nanolm.bench", "-c", base, *ov, "--steps", "50"],
                             capture_output=True, text=True, check=True).stdout
        ms, tps = re.search(r"(\d+) ms/step \| ([\d.]+)k tokens/s", txt).groups()
        res[name] = {"ms_per_step": int(ms), "tokens_per_sec": float(tps) * 1e3,
                     "peak_gb": float(re.search(r"peak GPU memory ([\d.]+) GB", txt).group(1)),
                     "params_m": float(re.search(r"([\d.]+)M params", txt).group(1))}
        print(name, res[name], flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
