"""The run statistics of the README, computed from the run records into results/summary.json (no GPU).
The token_conv kernel RMS is in results/token_conv_weights.json and the test results in results/tests.txt.

    python analysis/summarize.py

Sweeps (runs/sweeps/<sweep>/<arm>_s<seed>/): metrics.json (nanolm), dev.json (scripts/dev_eval.py),
compress.json (python -m nanolm.compress). Paired comparisons use the same seeds in both arms.
"""
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t as student_t, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "sweeps"
SEEDS = [1337, 42, 1, 2, 3]
SEEDS3 = [1337, 42, 1]                      # the first three seeds (3-seed numbers of the leaderboard row)
STEPS, TOKENS = 2500, 40_960_000
BUDGET = 6_500_000
SINGLES = ["depth_recur", "leaky_relu2", "value_residual", "mlp3x", "xsa", "qk_gain4", "partial_rope_lnscale",
           "smear_gate", "token_conv", "attn_window"]
METRICS = ["val_bpb", "sliding_bpb", "quant_val_bpb", "dev_bpb"]
# nanolm's tutorial (section 4.8), quoted for comparison
TUTORIAL = {"L5_val_bpb_single": 1.633, "L5_val_bpb_2seeds": (1.6338, 0.0018), "L5_artifact_mb": 6.11,
            "L6_val_bpb": 1.601, "L6_sliding128_bpb": 1.554, "L6_int6_roundtrip": 1.555, "L6_artifact_mb": 5.59,
            "laptop_ablation_delta": {"depth_recur": -0.0053, "leaky_relu2": -0.0053, "value_residual": -0.0033,
                                      "mlp3x": 0.0003, "partial_rope_lnscale": 0.0140},
            "laptop_ablation_p": {"depth_recur": 0.001, "leaky_relu2": 0.04, "value_residual": 0.08, "mlp3x": 0.92,
                                  "partial_rope_lnscale": 0.09},
            "lab_scale_delta": {"depth_recur": -0.0049, "leaky_relu2": -0.0059, "value_residual": -0.0069,
                                "mlp3x": -0.0121, "partial_rope_lnscale": 0.0073},
            "laptop_ablation_seeds": 2, "laptop_l6_2seeds": (1.6011, 0.0017), "laptop_l6_delta": -0.0327,
            "laptop_l6_p": 0.0006, "lab_scale_l6_delta": -0.0352}


def load(sweep):
    out = {}
    for d in sorted((RUNS / sweep).glob("*_s*")):
        if not (d / "metrics.json").exists():
            continue
        arm, seed = d.name.rsplit("_s", 1)
        m = json.loads((d / "metrics.json").read_text())
        assert m["steps"] == STEPS and m["tokens_seen"] == TOKENS, (d, m["steps"], m["tokens_seen"])
        if (d / "dev.json").exists():
            m["dev_bpb"] = json.loads((d / "dev.json").read_text())["dev_bpb"]
        m["train_flops"] = train_flops(d, m)
        out.setdefault(arm, {})[int(seed)] = m
    return out


def describe(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist(), "n": len(v)}


def paired(a, b):
    """b - a over matched seeds: mean, std, paired t-test p, 95% CI, sign agreement."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    sd = d.std(ddof=1)
    p = float(ttest_rel(b, a).pvalue) if sd > 0 else (0.0 if d.mean() else 1.0)
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"mean_diff": float(d.mean()), "std_diff": float(sd), "p": p, "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "n_negative": int((d < 0).sum()), "n": n, "values": d.tolist()}


def compare(A, base, arm, metric, seeds=SEEDS):
    """arm vs base on `metric`, with the noise floor of the two arms (mean of their within-arm stds)."""
    a = [A[base][s][metric] for s in seeds]
    b = [A[arm][s][metric] for s in seeds]
    r = paired(a, b)
    r["floor"] = float(np.mean([np.std(a, ddof=1), np.std(b, ddof=1)]))
    r["resolved"] = bool(abs(r["mean_diff"]) >= 2 * r["floor"])
    return r


def holm(ps):
    """Holm step-down at 0.05: list of booleans (rejected) in the order of ps."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, stop = [False] * len(ps), False
    for k, i in enumerate(order):
        stop = stop or ps[i] > 0.05 / (len(ps) - k)
        rej[i] = not stop
    return rej


def select_stack(A):
    """The selection rule: dev-bpb change below -2x the noise floor of the two arms and negative on
    all 5 seeds."""
    sel = {}
    for t in SINGLES:
        if t not in A or "dev_bpb" not in A[t][SEEDS[0]]:
            continue
        r = compare(A, "baseline", t, "dev_bpb")
        sel[t] = {**r, "selected": bool(r["mean_diff"] < -2 * r["floor"] and r["n_negative"] == len(SEEDS))}
    return sel


def train_flops(run_dir, m):
    """nanolm's estimate (bench.flops_per_token): 6 x non-embedding params + 12 L d T attention, per
    training token; with depth recurrence the re-run layers add their share after recur_start_frac."""
    c = yaml.safe_load((run_dir / "config.yaml").read_text())
    L, d, T = c["model"]["n_layers"], c["model"]["dim"], c["train"]["seq_len"]
    n = m["params_non_embedding"]
    per_tok = 6 * n + 12 * L * d * T
    r = len(c["model"].get("recur_layers") or []) * c["model"].get("recur_times", 1)
    extra = r / L * per_tok * (1 - c["model"].get("recur_start_frac", 0.0))
    return m["tokens_seen"] * (per_tok + extra)


def arm_summary(A, arm):
    r = {k: describe([A[arm][s][k] for s in SEEDS if k in A[arm][s]]) for k in METRICS if k in A[arm][SEEDS[0]]}
    m0 = A[arm][SEEDS[0]]
    r.update({"params": m0["params"], "params_non_embedding": m0["params_non_embedding"],
              "artifact_bytes": describe([A[arm][s]["artifact_bytes"] for s in SEEDS]),
              "fits_budget": all(A[arm][s]["fits_budget"] for s in SEEDS), "quant_bits": m0.get("quant_bits"),
              "train_time_sec": describe([A[arm][s]["train_time_sec"] for s in SEEDS]),
              "train_flops": m0["train_flops"], "code_bytes": m0["code_bytes"]})
    for k in ("val_bpb", "quant_val_bpb"):
        r[k + "_3seeds"] = describe([A[arm][s][k] for s in SEEDS3])
    return r


def ablation():
    A = load("ablation")
    out = {"arms": {a: arm_summary(A, a) for a in A}}
    out["vs_baseline"] = {t: {m: compare(A, "baseline", t, m) for m in METRICS if m in A[t][SEEDS[0]]}
                          for t in A if t != "baseline"}
    fam = [t for t in SINGLES if t in A]
    rej = holm([out["vs_baseline"][t]["val_bpb"]["p"] for t in fam])
    for t, r in zip(fam, rej):
        out["vs_baseline"][t]["val_bpb"]["holm_reject"] = r
    out["selection"] = select_stack(A)
    # nanolm's L6 stack is exactly these five single arms (plus int6 export, which val/dev bpb do not see)
    parts = ["depth_recur", "leaky_relu2", "value_residual", "mlp3x", "xsa"]
    out["l6_additivity"] = {}
    for m in ("val_bpb", "dev_bpb"):
        ss = [sum(A[t][s][m] - A["baseline"][s][m] for t in parts) for s in SEEDS]
        sd = [A["l6_reference"][s][m] - A["baseline"][s][m] for s in SEEDS]
        out["l6_additivity"][m] = {"parts": parts, "sum_of_singles": describe(ss), "stack_delta": describe(sd),
                                   "interaction": paired(ss, sd)}
    # does choosing on dev change anything? the same rule on validation, and how the two sets' changes agree
    ts = [t for t in SINGLES if t in A]
    dv, vv = [out["vs_baseline"][t]["dev_bpb"]["mean_diff"] for t in ts], [out["vs_baseline"][t]["val_bpb"]["mean_diff"] for t in ts]
    val_rule = [t for t in ts if out["vs_baseline"][t]["val_bpb"]["mean_diff"] < -2 * out["vs_baseline"][t]["val_bpb"]["floor"]
                and out["vs_baseline"][t]["val_bpb"]["n_negative"] == len(SEEDS)]
    out["dev_vs_val"] = {"pearson_r": float(np.corrcoef(dv, vv)[0, 1]), "max_abs_diff": float(max(abs(a - b) for a, b in zip(dv, vv))),
                         "selected_by_dev": [t for t in ts if out["selection"][t]["selected"]], "selected_by_val_rule": val_rule}
    out["noise_floor_val"] = float(np.median([np.std([A[a][s]["val_bpb"] for s in SEEDS], ddof=1) for a in A]))
    return out


def lr_sensitivity(A):
    """Muon matrix_lr and Adam lr as multiples of nanolm's 0.04, against the 1x runs of the ablation sweep."""
    L = load("lr")
    out = {}
    for arm in ("baseline", "token_conv", "attn_window"):
        for key in ("mlr", "lr"):
            pts = {}
            for name in L:
                pre, _, f = name.rpartition("_")
                if pre == arm and f.startswith(key) and f[len(key):].replace(".", "").isdigit():
                    pts[f[len(key):]] = {m: describe([L[name][s][m] for s in SEEDS]) for m in ("val_bpb", "dev_bpb")}
                    pts[f[len(key):]]["vs_1x"] = {m: compare({**A, name: L[name]}, arm, name, m) for m in ("val_bpb", "dev_bpb")}
            if pts:
                pts["1"] = {m: describe([A[arm][s][m] for s in SEEDS]) for m in ("val_bpb", "dev_bpb")}
                fs = sorted(pts, key=float)
                best = min(fs, key=lambda k: pts[k]["dev_bpb"]["mean"])
                out[f"{arm}_{key}"] = {"factors": fs, "points": pts, "best_factor_dev": best,
                                       "best_at_edge": best in (fs[0], fs[-1])}
    return out


STACK_FAMILY = ["stack", "stack_matrix_lr0.01", "stack_adam_lr0.02", "stack_plus_mlp3x", "stack_plus_l6"]


def stack_study(A, sel):
    """The final stack against the baseline rerun on the same machine, leave-one-out, additivity, and the
    leaderboard choice (lowest dev bpb among the stack-family arms that fit the budget)."""
    S = load("stack")
    if "stack" not in S or "baseline" not in S:
        return None
    L = load("lr")
    members = [t for t, r in sel.items() if r["selected"]] + ["matrix_lr"]
    single = {t: A[t] for t in members if t in A}
    single["matrix_lr"] = L["baseline_mlr0.5"]            # matrix_lr 0.02 alone, against the ablation baseline
    out = {"members": members, "stack": arm_summary(S, "stack"),
           "vs_baseline": {m: compare(S, "baseline", "stack", m) for m in METRICS},
           "baseline_rerun_vs_ablation_baseline": {m: compare({"a": A["baseline"], "b": S["baseline"]}, "a", "b", m)
                                                    for m in METRICS}}
    out["leave_one_out"] = {}
    for t in members:
        name = f"stack_minus_{t}"
        if name in S:
            out["leave_one_out"][t] = {"arm": arm_summary(S, name),
                                       "stack_minus_t_vs_stack": {m: compare(S, "stack", name, m) for m in METRICS},
                                       "single_vs_baseline": {m: compare({**A, "x": single[t]}, "baseline", "x", m)
                                                              for m in METRICS}}
    loo = list(out["leave_one_out"])
    for t, r in zip(loo, holm([out["leave_one_out"][t]["stack_minus_t_vs_stack"]["val_bpb"]["p"] for t in loo])):
        out["leave_one_out"][t]["stack_minus_t_vs_stack"]["val_bpb"]["holm_reject"] = r
    # additivity, per seed: (stack - baseline, machine b) - sum of the single changes (machine a; the matrix_lr
    # term is lr/baseline_mlr0.5 on machine b against the ablation baseline on machine a)
    for m in ("val_bpb", "dev_bpb", "quant_val_bpb"):
        sd = [S["stack"][s][m] - S["baseline"][s][m] for s in SEEDS]
        ss = [sum(single[t][s][m] - A["baseline"][s][m] for t in members) for s in SEEDS]
        out[f"additivity_{m}"] = {"sum_of_singles": describe(ss), "stack_delta": describe(sd),
                                  "interaction": paired(ss, sd)}
    out["extra_arms"] = {n: {"arm": arm_summary(S, n), "vs_stack": {m: compare(S, "stack", n, m) for m in METRICS}}
                         for n in S if n in STACK_FAMILY[1:]}
    fam = [n for n in STACK_FAMILY if n in S]
    fits = [n for n in fam if all(S[n][s]["fits_budget"] for s in SEEDS)]
    best = min(fits, key=lambda n: np.mean([S[n][s]["dev_bpb"] for s in SEEDS]))
    m0 = S[best][SEEDS[0]]
    out["leaderboard"] = {"arm": best, "candidates_dev": {n: describe([S[n][s]["dev_bpb"] for s in SEEDS]) for n in fam},
                          "quant_val_bpb_3seeds": describe([S[best][s]["quant_val_bpb"] for s in SEEDS3]),
                          "quant_val_bpb_5seeds": describe([S[best][s]["quant_val_bpb"] for s in SEEDS]),
                          "val_bpb_5seeds": describe([S[best][s]["val_bpb"] for s in SEEDS]),
                          "artifact_bytes": describe([S[best][s]["artifact_bytes"] for s in SEEDS]),
                          "artifact_bytes_max": max(S[best][s]["artifact_bytes"] for s in SEEDS),
                          "quant_bits": m0.get("quant_bits"), "config": "configs/sweeps/stack.yaml",
                          "vs_baseline": {m: compare(S, "baseline", best, m) for m in METRICS}}
    # the tutorial's L6 stack (ablation sweep, another machine) against the same stack with ours added
    if "stack_plus_l6" in S:
        out["plus_l6_vs_l6_reference"] = {m: compare({"l6": A["l6_reference"], "x": S["stack_plus_l6"]}, "l6", "x", m)
                                          for m in METRICS}
    return out


def compress_tables():
    out = {}
    pairs = sorted({(f.parts[-3], f.parts[-2].rsplit("_s", 1)[0]) for f in RUNS.glob("*/*_s*/compress.json")})
    for sweep, arm in pairs:
        rows = {}
        for s in SEEDS:
            f = RUNS / sweep / f"{arm}_s{s}" / "compress.json"
            if f.exists():
                rows[s] = json.loads(f.read_text())
        if not rows:
            continue
        q = {}
        for s, r in rows.items():
            for x in r["quant"]:
                q.setdefault(f"int{x['bits']}+{x['compressor']}", []).append(x)
        st = {}
        for s, r in rows.items():
            for x in r["stride"]:
                st.setdefault(x["stride"], []).append(x)
        out[f"{sweep}/{arm}"] = {"seeds": sorted(rows), "quant": {k: {"bytes": describe([x["bytes"] for x in v]),
                                                         "bits_per_param": describe([x["bits_per_param"] for x in v]),
                                                         "bpb": describe([x["bpb"] for x in v]),
                                                         "gap": describe([x["gap"] for x in v])} for k, v in q.items()},
                    "stride": {str(k): {**describe([x["bpb"] for x in v]), "eval_sec": describe([x["sec"] for x in v])}
                               for k, v in st.items()}}
    return out


def machines():
    """The machine of every run (results/machines.json maps the host recorded in meta.json to a machine)."""
    host = {v["host"]: k for k, v in json.loads((ROOT / "results" / "machines.json").read_text()).items()
            if isinstance(v, dict)}
    out = {}
    for f in sorted(RUNS.glob("*/*_s*/meta.json")):
        if (f.parent / "metrics.json").exists():
            arm = f"{f.parts[-3]}/{f.parent.name.rsplit('_s', 1)[0]}"
            out.setdefault(arm, set()).add(host[json.loads(f.read_text())["host"]])
    return {k: "".join(sorted(v)) for k, v in out.items()}


CURVE_STEPS = [500, 1000, 1500, 2000, 2500]


def curve(sweep, arm):
    """In-training validation bpb (first 262,144 val tokens) at CURVE_STEPS, per seed."""
    out = {}
    for s in SEEDS:
        rows = {r["step"]: r["bpb"] for r in map(json.loads, open(RUNS / sweep / f"{arm}_s{s}" / "eval_log.jsonl"))}
        out[s] = [rows[k] for k in CURVE_STEPS]
    return out


def curve_gaps():
    """When does a gain appear? Paired change from the baseline of the same sweep (the lr arm against the
    ablation baseline) at fixed points of training; warmdown starts at 75% (step 1875)."""
    pairs = {"xsa": ("ablation", "xsa", "ablation"), "value_residual": ("ablation", "value_residual", "ablation"),
             "token_conv": ("ablation", "token_conv", "ablation"), "matrix_lr": ("lr", "baseline_mlr0.5", "ablation"),
             "depth_recur": ("ablation", "depth_recur", "ablation"), "l6_reference": ("ablation", "l6_reference", "ablation"),
             "stack": ("stack", "stack", "stack"), "stack_plus_l6": ("stack", "stack_plus_l6", "stack")}
    out = {"steps": CURVE_STEPS}
    for name, (sw, arm, base_sw) in pairs.items():
        a, b = curve(base_sw, "baseline"), curve(sw, arm)
        out[name] = [paired([a[s][i] for s in SEEDS], [b[s][i] for s in SEEDS]) for i in range(len(CURVE_STEPS))]
    return out


def machine_check(A):
    """The same baseline seeds on the second machine: per-seed difference from the first machine's runs."""
    M = load("machine_check")
    return {str(s): {m: M["baseline"][s][m] - A["baseline"][s][m] for m in METRICS} for s in M.get("baseline", {})}


def main():
    A = load("ablation")
    ab = ablation()
    summary = {"tutorial": TUTORIAL, "ablation": ab, "machine_check": machine_check(A), "lr": lr_sensitivity(A),
               "stack": stack_study(A, ab["selection"]), "compress": compress_tables(), "machines": machines(),
               "curve_gaps": curve_gaps()}
    bench = ROOT / "results" / "bench.json"
    if bench.exists():
        summary["bench"] = json.loads(bench.read_text())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
