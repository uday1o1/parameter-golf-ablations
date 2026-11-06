"""All figures, from results/summary.json and the run records (no GPU).

    python analysis/figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
RUNS = ROOT / "runs" / "sweeps"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
SEEDS = [1337, 42, 1, 2, 3]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = plt.get_cmap("tab10").colors
LABEL = {"depth_recur": "depth recurrence", "leaky_relu2": "LeakyReLU²", "value_residual": "value residual",
         "mlp3x": "MLP 3×", "xsa": "XSA (last 3)", "qk_gain4": "QK-gain 4", "partial_rope_lnscale": "partial RoPE + LN-scale",
         "smear_gate": "SmearGate", "token_conv": "token_conv (new)", "attn_window": "attn_window (new)",
         "l6_reference": "tutorial L6 stack", "matrix_lr": "matrix lr 0.02"}


def curve(sweep, arm):
    """Mean and std over seeds of the in-training validation bpb (first 262,144 val tokens)."""
    per = []
    for s in SEEDS:
        f = RUNS / sweep / f"{arm}_s{s}" / "eval_log.jsonl"
        rows = [json.loads(l) for l in open(f)]
        per.append([(r["train_tokens"], r["bpb"]) for r in rows])
    steps = np.array([x for x, _ in per[0]])
    v = np.array([[y for _, y in p] for p in per])
    return steps / 1e6, v.mean(0), v.std(0, ddof=1)


def ablation_fig():
    ab = S["ablation"]["vs_baseline"]
    arms = sorted([a for a in ab if a != "l6_reference"], key=lambda a: ab[a]["val_bpb"]["mean_diff"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(arms))
    for k, (m, off, c) in enumerate((("val_bpb", -0.18, COL[0]), ("dev_bpb", 0.18, COL[1]))):
        d = [ab[t][m]["mean_diff"] for t in arms]
        lo = [ab[t][m]["mean_diff"] - ab[t][m]["ci95"][0] for t in arms]
        hi = [ab[t][m]["ci95"][1] - ab[t][m]["mean_diff"] for t in arms]
        a.errorbar(d, y + off, xerr=[lo, hi], fmt="o", ms=3, color=c, capsize=2, lw=0.8,
                   label="validation" if m == "val_bpb" else "dev (selection set)")
    a.axvline(0, color="k", lw=0.6)
    a.set_yticks(y, [LABEL[t] + (" *" if S["ablation"]["selection"].get(t, {}).get("selected") else "") for t in arms])
    a.invert_yaxis()
    a.set_xlabel("change in bpb vs baseline (mean, 95% CI over 5 seeds)")
    a.set_title("(a) Each technique alone (* = selected for the stack)", loc="left")
    a.legend(frameon=False, loc="upper right")
    for i, arm in enumerate(["baseline"] + [t for t in arms if S["ablation"]["selection"].get(t, {}).get("selected")]):
        x, m, s = curve("ablation", arm)
        b.plot(x, m, color=COL[i % 10], lw=1, label=LABEL.get(arm, arm))
        b.fill_between(x, m - s, m + s, color=COL[i % 10], alpha=0.2)
    st = S.get("stack")
    if st:
        x, m, s = curve("stack", "stack")
        b.plot(x, m, color="k", lw=1.2, label="stack (ours)")
        b.fill_between(x, m - s, m + s, color="k", alpha=0.15)
    b.set_ylim(1.57, 1.9)
    b.set_xlabel("training tokens (M)")
    b.set_ylabel("validation bpb (first 262k tokens)")
    b.set_title("(b) Learning curves, mean ± std", loc="left")
    b.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=200)
    plt.close(fig)


def stack_fig():
    st = S.get("stack")
    if not st:
        return
    loo = st["leave_one_out"]
    ts = list(loo)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    y = np.arange(len(ts))
    d = [loo[t]["stack_minus_t_vs_stack"]["val_bpb"]["mean_diff"] for t in ts]
    ci = [loo[t]["stack_minus_t_vs_stack"]["val_bpb"]["ci95"] for t in ts]
    single = [-loo[t]["single_vs_baseline"]["val_bpb"]["mean_diff"] for t in ts]
    a.errorbar(d, y - 0.15, xerr=[[x - c[0] for x, c in zip(d, ci)], [c[1] - x for x, c in zip(d, ci)]], fmt="o",
               ms=3, color=COL[0], capsize=2, lw=0.8, label="removing it from the stack")
    a.scatter(single, y + 0.15, marker="s", s=12, color=COL[1], label="adding it alone to the baseline")
    a.axvline(0, color="k", lw=0.6)
    a.set_yticks(y, [LABEL[t] for t in ts])
    a.invert_yaxis()
    a.set_xlabel("val bpb gained from the technique")
    a.xaxis.set_major_locator(plt.MaxNLocator(5))
    a.set_title("(a) Leave-one-out", loc="left")
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    q = S["compress"].get("stack/" + st["leaderboard"]["arm"], {}).get("quant", {})
    for i, comp in enumerate(("zlib", "lzma")):
        ks = [k for k in q if k.endswith(comp)]
        ks = sorted(ks, key=lambda k: -int(k[3:].split("+")[0]))
        b.plot([q[k]["bytes"]["mean"] / 1e6 for k in ks], [q[k]["gap"]["mean"] for k in ks], marker="o", ms=3,
               color=COL[i], label=comp)
        for k in ks if comp == "lzma" else []:
            b.annotate(k.split("+")[0], (q[k]["bytes"]["mean"] / 1e6, q[k]["gap"]["mean"]), fontsize=6,
                       xytext=(-4, 4), textcoords="offset points", ha="right")
    b.axvline(6.5, color="k", lw=0.6, ls="--")
    b.set_ylim(None, b.get_ylim()[1] * 1.12)
    b.set_xlim(b.get_xlim()[0] - 0.4, None)
    b.set_xlabel("compressed model (MB)")
    b.set_ylabel("round-trip bpb cost (stride 0)")
    b.set_title("(b) Bits × compressor, leaderboard model", loc="left")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "stack.png", dpi=200)
    plt.close(fig)


def lr_fig():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, key, title in zip(axes, ("mlr", "lr"), ("(a) Muon matrix learning rate", "(b) Adam learning rate")):
        for i, arm in enumerate(("baseline", "token_conv", "attn_window")):
            r = S["lr"].get(f"{arm}_{key}")
            if not r:
                continue
            fs = r["factors"]
            m = [r["points"][f]["dev_bpb"]["mean"] for f in fs]
            e = [r["points"][f]["dev_bpb"]["std"] for f in fs]
            ax.errorbar([0.04 * float(f) for f in fs], m, yerr=e, marker="o", ms=3, capsize=2, lw=1, color=COL[i],
                        label=LABEL.get(arm, arm))
        ax.axvline(0.04, color="k", lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.005, 0.01, 0.02, 0.04, 0.08], ["0.005", "0.01", "0.02", "0.04", "0.08"])
        ax.minorticks_off()
        ax.set_xlabel("learning rate (dotted: nanolm's 0.04)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("dev bpb (mean ± std, 5 seeds)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "lr.png", dpi=200)
    plt.close(fig)


def dynamics_fig():
    """(a) When the gains appear: change from the baseline during training. (b) Additivity per seed."""
    G = S["curve_gaps"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.array(G["steps"]) * 16384 / 1e6
    for i, (k, lab) in enumerate((("stack", "stack (ours)"), ("l6_reference", "tutorial L6 stack"),
                                  ("matrix_lr", "matrix lr 0.02"), ("xsa", LABEL["xsa"]),
                                  ("value_residual", LABEL["value_residual"]), ("token_conv", LABEL["token_conv"]),
                                  ("depth_recur", LABEL["depth_recur"]))):
        m = np.array([g["mean_diff"] for g in G[k]])
        c = "k" if k == "stack" else COL[i]
        a.plot(x, m, marker="o", ms=2.5, lw=1, color=c, label=lab)
    a.axhline(0, color="k", lw=0.6)
    a.axvline(0.75 * 2500 * 16384 / 1e6, color="grey", lw=0.6, ls=":")
    a.set_yscale("symlog", linthresh=0.01)
    a.set_xlabel("training tokens (M; dotted: warmdown starts)")
    a.set_ylabel("bpb change vs baseline")
    a.set_title("(a) When the gains appear (mean over 5 seeds)", loc="left")
    a.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.62, -0.22))
    for key, lab, c in (("additivity_val_bpb", "stack (ours)", "k"), (None, "tutorial L6 stack", COL[1])):
        r = S["stack"][key] if key else S["ablation"]["l6_additivity"]["val_bpb"]
        b.scatter(r["sum_of_singles"]["values"], r["stack_delta"]["values"], s=12, color=c, label=lab)
    lim = [-0.045, -0.012]
    b.plot(lim, lim, color="grey", lw=0.6, ls="--", label="additive")
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("sum of the parts alone (val bpb)")
    b.set_ylabel("stack vs baseline (val bpb)")
    b.set_title("(b) Additivity, one dot per seed", loc="left")
    b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "dynamics.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ablation_fig()
    dynamics_fig()
    stack_fig()
    lr_fig()
