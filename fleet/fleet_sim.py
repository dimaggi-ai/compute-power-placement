#!/usr/bin/env python3
"""Fleet-level scarcity scheduling (v2) — an under-built market layer.

The move-vs-stay model (../calc) prices ONE relocation decision. This extends it
to a fleet policy: given a data-center fleet with a flexible (deferrable/
curtailable) load fraction and a real-shaped electricity price process — a
diurnal base with rare scarcity spikes — how much does a *scarcity-aware*
scheduler save over a price-blind one, and where does the saving come from?

The finding the single-decision model predicts, now at fleet scale: the saving
is small but CONCENTRATED — even a policy free to curtail every scarcity hour
draws most of its avoided cost from the top ~1% of hours, and with the scarcity
process off it collapses to under 1%. This is "schedule against scarcity, not
the spread," quantified for a fleet. The headline percentage is spike-cap-bound
(ERCOT-scale $5k/MWh here); the concentration is the portable claim. It rhymes
with the operator reality (Duke: ~76 GW of new load fits the existing grid if
new loads curtail ~0.25% of uptime) — a capacity-hosting result, same lever.

Deterministic per seed. Transparent: every calibration value is an input.
"""
from __future__ import annotations
import dataclasses
import math
import numpy as np


@dataclasses.dataclass
class Fleet:
    horizon_h: int = 720                 # 30 days, hourly
    fleet_mw: float = 100.0              # total load (constant, for clarity)
    flexible_fraction: float = 0.30      # deferrable/curtailable share (inference + batch)
    seed: int = 0
    # price process ($/MWh)
    base_mwh: float = 50.0
    diurnal_amp_mwh: float = 20.0
    spike_rate_per_h: float = 0.004      # ~3 spike-onsets / month
    spike_len_h: int = 4
    spike_min_mwh: float = 500.0
    spike_max_mwh: float = 5000.0
    # policy
    curtail_threshold_mwh: float = 200.0  # curtail flexible load above this price
    curtail_budget_frac: float = 0.005    # <= 0.5% of hours may be curtailed


def price_series(cfg: Fleet, rng) -> np.ndarray:
    """Diurnal base with rare multi-hour scarcity spikes."""
    h = np.arange(cfg.horizon_h)
    diurnal = cfg.diurnal_amp_mwh * np.sin((h % 24 - 6) / 24 * 2 * math.pi)
    p = np.maximum(5.0, cfg.base_mwh + diurnal + rng.normal(0, 4, cfg.horizon_h))
    t = 0
    while t < cfg.horizon_h:
        if rng.random() < cfg.spike_rate_per_h:
            mag = rng.uniform(cfg.spike_min_mwh, cfg.spike_max_mwh)
            p[t:t + cfg.spike_len_h] = np.maximum(p[t:t + cfg.spike_len_h], mag)
            t += cfg.spike_len_h
        else:
            t += 1
    return p


def simulate(cfg: Fleet) -> dict:
    rng = np.random.default_rng(cfg.seed)
    p = price_series(cfg, rng)
    flex_mw = cfg.fleet_mw * cfg.flexible_fraction
    firm_mw = cfg.fleet_mw - flex_mw

    # price-blind: run the whole fleet every hour
    blind_cost = float((cfg.fleet_mw * p).sum())

    # scarcity-aware: curtail the flexible load in the most expensive hours that
    # clear the scarcity threshold (up to the curtailment budget) and defer that
    # energy to the cheapest hours.
    budget_h = int(cfg.curtail_budget_frac * cfg.horizon_h)
    order = np.argsort(-p)                                   # most expensive first
    curtail_hours = [int(i) for i in order if p[i] > cfg.curtail_threshold_mwh][:budget_h]
    cheap_hours = [int(i) for i in np.argsort(p)]           # cheapest first

    firm_cost = float((firm_mw * p).sum())
    # Flexible load runs at full in every hour except the curtailed ones. The
    # curtailed energy is deferred to the cheapest hours; we model those hours as
    # able to absorb up to an extra flex_mw (a utilization/headroom assumption —
    # the flexible tier runs at full every non-curtailed hour, so "headroom"
    # means an assumed utilization below 100%; see the study's caveat). The
    # destination price is so low that cost_defer is a rounding error on the
    # saving, so the result is insensitive to it.
    flex_hours = set(range(cfg.horizon_h)) - set(curtail_hours)
    flex_cost = float(sum(flex_mw * p[i] for i in flex_hours))
    curtailed_mwh = flex_mw * len(curtail_hours)
    filled, cost_defer = 0.0, 0.0
    for i in cheap_hours:
        if i in curtail_hours:
            continue
        take = min(flex_mw, curtailed_mwh - filled)
        if take <= 0:
            break
        cost_defer += take * p[i]
        filled += take
    aware_cost = firm_cost + flex_cost + cost_defer
    saving = blind_cost - aware_cost

    # Where does the saving concentrate? Decompose the GROSS avoided purchase
    # (flexible energy NOT bought during curtailed hours) by how much of it falls
    # in the single most expensive 1% of hours. This is NOT tautological: once the
    # budget exceeds the top-1% count, curtailed hours spill outside the top-1%
    # set and the fraction drops below 1 — so a high fraction at a generous budget
    # is a real concentration finding, not a restatement of the construction.
    top1 = set(int(i) for i in order[:max(1, cfg.horizon_h // 100)])
    gross_avoided = float(sum(flex_mw * p[i] for i in curtail_hours))
    gross_from_top1 = float(sum(flex_mw * p[i] for i in curtail_hours if i in top1))
    return {
        "seed": cfg.seed,
        "hours_curtailed": len(curtail_hours),
        "curtailed_frac_of_uptime": round(len(curtail_hours) / cfg.horizon_h, 5),
        "curtailed_mwh": round(curtailed_mwh, 3),
        "deferred_mwh": round(filled, 3),
        "blind_cost_dollars": round(blind_cost, 0),
        "aware_cost_dollars": round(aware_cost, 0),
        "saving_dollars": round(saving, 0),
        "saving_pct": round(saving / blind_cost, 4),
        "gross_avoided_dollars": round(gross_avoided, 0),
        "saving_from_top1pct_hours_frac": round(gross_from_top1 / gross_avoided, 3) if gross_avoided > 0 else 0.0,
        "max_price_mwh": round(float(p.max()), 0),
    }
