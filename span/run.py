#!/usr/bin/env python3
"""Figure for stay/stitch/move: the cut decides the price, and the decision
map over the two scarcity types. Usage: python3 run.py [--figdir ../figures]
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stay_stitch_move import (Circuit, Hall, SpanJob, allreduce_cut, ledger,
                              pipeline_cut)


def fig_span(figdir: Path):
    job = SpanJob(deadline_h=200.0)
    circuit = Circuit()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- left: $/useful GPU-hour, the job that fits no single hall
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=10000)
    no_deadline = dataclasses.replace(job, deadline_h=None)
    led_p = ledger(no_deadline, home, away, circuit, pipeline_cut(no_deadline))
    led_a = ledger(no_deadline, home, away, circuit, allreduce_cut(no_deadline))
    get = lambda led, n: next(o for o in led["options"] if o["option"].startswith(n))
    ref = get(led_p, "buy")["dollars_per_useful_gpu_h"]
    bars = [
        ("buy in-hall\n(reference —\nthe hall can't)", ref, "#888888"),
        ("stay: shrink,\nrun 2x as long", get(led_p, "stay")["dollars_per_useful_gpu_h"], "#4f81bd"),
        ("stitch,\npipeline cut", get(led_p, "stitch")["dollars_per_useful_gpu_h"], "#4e9a06"),
        ("stitch,\nall-reduce cut", get(led_a, "stitch")["dollars_per_useful_gpu_h"], "#c0504d"),
    ]
    xs = np.arange(len(bars))
    axL.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars])
    axL.axhline(ref, color="#555", lw=0.8, ls="--")
    for i, (_, v, _) in enumerate(bars):
        axL.text(i, v, f" ${v:,.2f}", ha="center", va="bottom", fontsize=8)
    axL.set_xticks(xs, [b[0] for b in bars], fontsize=8)
    axL.set_ylabel("$ per useful GPU-hour")
    axL.set_title("A job that fits no single hall: the CUT sets the price\n"
                  f"(pipeline: +{get(led_p,'stitch')['dollars_per_useful_gpu_h']/ref-1:.1%};"
                  f" all-reduce: {get(led_a,'stitch')['dollars_per_useful_gpu_h']/ref:.1f}x —"
                  " same circuit, same halls)")

    # ---- right: decision map over the two scarcity types.
    # The away hall holds 60% of the job — the scale-across reality. Give it
    # the whole job's worth of free GPUs and MOVE swallows the stitch region
    # entirely (a one-off migration beats a recurring tax; the tests pin this):
    # the stitch is what you do when no hall is big enough, not a discount move.
    deficits = np.linspace(0.0, 0.6, 25)          # capacity scarcity at home
    spikes = np.linspace(0.0, 24.0, 25)           # power scarcity at home (h/wk @ $5k)
    away_big = Hall("B", free_gpus=int(job.gpus_needed * 0.6))
    codes = {"stay": 0, "stitch": 1, "move": 2, "escalate": 3}
    grid = np.zeros((len(spikes), len(deficits)))
    for i, sp in enumerate(spikes):
        for k, d in enumerate(deficits):
            h = Hall("A", free_gpus=int(job.gpus_needed * (1 - d)),
                     spike_price_mwh=5000.0, spike_hours_per_week=sp)
            led = ledger(job, h, away_big, circuit, pipeline_cut(job))
            grid[i, k] = codes[led["decision"]]
    cmap = matplotlib.colors.ListedColormap(["#4f81bd", "#4e9a06", "#c0504d", "#555555"])
    axR.pcolormesh(deficits * 100, spikes, grid, cmap=cmap, vmin=0, vmax=3, shading="auto")
    axR.set_xlabel("capacity scarcity: % of the job the home hall cannot supply")
    axR.set_ylabel("power scarcity: spike hours/week at home ($5k/MWh)")
    axR.set_title("Two scarcity types, one ledger\n"
                  "(pipeline cut, 200 h deadline, away hall holds 60% of the job)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    axR.legend(handles, ["stay", "stitch", "move", "escalate"],
               loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "stay_stitch_move.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="../figures")
    a = ap.parse_args()
    figdir = (Path(__file__).resolve().parent / a.figdir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)
    fig_span(figdir)
    print(f"wrote 1 figure to {figdir}")


if __name__ == "__main__":
    main()
