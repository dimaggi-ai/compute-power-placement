#!/usr/bin/env python3
"""Fleet-level scarcity-scheduling figure. Usage: python3 run.py [--figdir ../figures]"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fleet_sim import Fleet, simulate, price_series


def fig(figdir: Path):
    import statistics as st
    cfg = Fleet(seed=0)
    rng = np.random.default_rng(cfg.seed)
    p = price_series(cfg, rng)
    r = simulate(cfg)                                  # the pictured month (seed 0)
    mean_pct = st.mean(simulate(Fleet(seed=s))["saving_pct"] for s in range(24))
    # honest concentration: a policy FREE to curtail EVERY scarcity hour (budget
    # far exceeds the top-1% set), so a high top-1% share is not a budget artifact
    conc = st.mean(simulate(Fleet(seed=s, curtail_budget_frac=0.10))
                   ["saving_from_top1pct_hours_frac"] for s in range(24))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [2, 1]})
    # left: the price series with scarcity spikes and the curtailment threshold
    axL.plot(p, color="#4f81bd", lw=0.9)
    axL.axhline(cfg.curtail_threshold_mwh, color="#c0504d", ls="--", lw=1)
    axL.text(5, cfg.curtail_threshold_mwh + 60, "curtail flexible load above this", color="#c0504d", fontsize=8)
    axL.set_xlabel("hour of the month")
    axL.set_ylabel("price ($/MWh)")
    axL.set_title("A month of nodal prices: a diurnal base with rare scarcity spikes\n"
                  f"(scarcity-aware curtails only its {r['hours_curtailed']} worst hours — "
                  f"{r['curtailed_frac_of_uptime']*100:.2f}% of uptime)")
    # right: where the saving CONCENTRATES, for a policy free to curtail every
    # scarcity hour — this is the finding, and it is not a budget artifact
    axR.bar(["top-1%\nprice hours", "all other\nscarcity hrs"],
            [conc, 1 - conc], color=["#c0504d", "#bbbbbb"])
    axR.set_ylim(0, 1)
    axR.set_ylabel("share of avoided $ (curtail-all policy)")
    axR.set_title(f"Even curtailing every scarcity hour,\n{conc*100:.0f}% of avoided cost is the top-1%")
    axR.text(0, conc + 0.02, f"{conc*100:.0f}%", ha="center", fontsize=9)
    fig.suptitle("Fleet scarcity scheduling — pictured month saves "
                 f"{r['saving_pct']*100:.1f}%, ~{mean_pct*100:.0f}% across seeds "
                 "(ERCOT-scale $5k/MWh cap)", y=1.02)
    fig.tight_layout()
    fig.savefig(figdir / "fleet_scarcity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="../figures")
    a = ap.parse_args()
    figdir = (Path(__file__).resolve().parent / a.figdir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)
    fig(figdir)
    print(f"wrote fleet figure to {figdir}")


if __name__ == "__main__":
    main()
