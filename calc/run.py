#!/usr/bin/env python3
"""Figures for move-vs-stay: the break-even frontier, the energy-share reframe,
and the scarcity-avoidance case. Usage: python3 run.py [--figdir ../figures]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from move_vs_stay import Job, breakeven_spread_mwh, decide, scarcity_avoided_dollars


def fig_breakeven(figdir: Path):
    """Break-even price spread vs job length, for three model sizes / WANs."""
    lengths = np.array([12, 24, 48, 72, 120, 168, 336, 720])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cases = [
        ("70B @ 100 Gbps", Job(params_b=70, wan_gbps=100), "#4f81bd"),
        ("405B @ 100 Gbps", Job(params_b=405, wan_gbps=100), "#4e9a06"),
        ("405B @ 10 Gbps", Job(params_b=405, wan_gbps=10), "#c0504d"),
    ]
    for label, job, color in cases:
        be = [breakeven_spread_mwh(job, L) for L in lengths]
        ax.plot(lengths, be, marker="o", color=color, label=label)
    # a typical real spread band
    ax.axhspan(10, 40, color="#bbbbbb", alpha=0.3)
    ax.text(430, 25, "typical average\nLMP spread ($10–40/MWh)", fontsize=8, color="#555")
    ax.set_xlabel("remaining job length (hours)")
    ax.set_ylabel("break-even price spread ($/MWh)")
    ax.set_yscale("log")
    ax.set_title("Moving on price alone pays only for long jobs\n"
                 "(above the line = staying wins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "breakeven_frontier.png", dpi=150)
    plt.close(fig)


def fig_energy_share(figdir: Path):
    """Energy is a few percent of the GPU-hour cost — the whole reason spread
    arbitrage is weak — vs the scarcity spike that flips it."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    j = Job()
    gpu_cost = j.gpus * j.dollars_per_gpu_h                 # $/h for the fleet
    energy_60 = j.power_mw * 60                             # $/h energy at $60/MWh
    energy_5000 = j.power_mw * 5000                         # $/h at scarcity
    axL.bar(["GPU-hour cost\n($2.5/GPU-h)", "energy @ $60/MWh", "energy @ $5000/MWh\n(scarcity)"],
            [gpu_cost, energy_60, energy_5000],
            color=["#4f81bd", "#4e9a06", "#c0504d"])
    axL.set_ylabel("$ per hour, whole fleet")
    axL.set_title("Energy is ~3% of cost normally,\nbut a scarcity spike is ~2x the fleet's GPU cost")
    for i, v in enumerate([gpu_cost, energy_60, energy_5000]):
        axL.text(i, v, f" ${v/1000:,.0f}k", ha="center", va="bottom", fontsize=8)
    # right: scarcity avoidance vs migration cost
    s = scarcity_avoided_dollars(j, 5000, 12)
    axR.bar(["migration\ncost (once)", "stay through a\n12h $5000/MWh spike", "move + ride\nit out elsewhere"],
            [j.migration_cost_dollars, s["stay_through_spike_dollars"], s["move_and_ride_it_out_dollars"]],
            color=["#888888", "#c0504d", "#4e9a06"])
    axR.set_ylabel("$")
    axR.set_title("Scarcity avoidance is where moving pays\n"
                  f"(net benefit of moving ≈ ${s['net_benefit_of_moving']/1e6:.1f}M)")
    for i, v in enumerate([j.migration_cost_dollars, s["stay_through_spike_dollars"], s["move_and_ride_it_out_dollars"]]):
        axR.text(i, v, f" ${v/1000:,.0f}k", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "energy_share_and_scarcity.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="../figures")
    a = ap.parse_args()
    figdir = (Path(__file__).resolve().parent / a.figdir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)
    fig_breakeven(figdir)
    fig_energy_share(figdir)
    print(f"wrote 2 figures to {figdir}")


if __name__ == "__main__":
    main()
