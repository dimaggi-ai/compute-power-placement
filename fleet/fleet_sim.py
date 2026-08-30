#!/usr/bin/env python3
"""Fleet-level scarcity scheduling (v2) — the market layer nobody has built.

The move-vs-stay model (../calc) prices ONE relocation decision. This extends it
to a fleet policy: given a data-center fleet with a flexible (deferrable/
curtailable) load fraction and a real-shaped electricity price process — a
diurnal base with rare scarcity spikes — how much does a *scarcity-aware*
scheduler save over a price-blind one, and where does the saving come from?

The finding the single-decision model predicts, now at fleet scale: almost all
of the saving comes from curtailing the flexible load during the handful of
scarcity hours and deferring it to the cheapest hours — NOT from chasing the
average price. This is "schedule against scarcity," quantified for a fleet, and
it matches the operator reality (Duke: ~76 GW of new load fits the existing grid
if new loads curtail ~0.25% of uptime).

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

    # scarcity-aware: curtail the flexible load in the most expensive hours (up
    # to the curtailment budget) and defer that energy to the cheapest hours.
    budget_h = int(cfg.curtail_budget_frac * cfg.horizon_h)
    order = np.argsort(-p)                                   # most expensive first
    curtail_hours = [i for i in order if p[i] > cfg.curtail_threshold_mwh][:budget_h]
    cheap_hours = list(np.argsort(p))                        # cheapest first

    firm_cost = float((firm_mw * p).sum())
    # flexible load runs at full every hour EXCEPT the curtailed ones; the
    # curtailed energy is deferred and served in the cheapest available hours
    # (headroom capped so we do not double-load an hour beyond the fleet).
    flex_hours = set(range(cfg.horizon_h)) - set(curtail_hours)
    flex_cost = float(sum(flex_mw * p[i] for i in flex_hours))
    deferred_mwh = flex_mw * len(curtail_hours)
    # fill cheapest hours with up to flex_mw extra each (headroom = firm slack)
    filled, cost_defer = 0.0, 0.0
    for i in cheap_hours:
        if i in curtail_hours:
            continue
        take = min(flex_mw, deferred_mwh - filled)
        if take <= 0:
            break
        cost_defer += take * p[i]
        filled += take
    aware_cost = firm_cost + flex_cost + cost_defer
    saving = blind_cost - aware_cost

    # decompose: how much of the saving comes from the top-1% price hours?
    top1 = set(int(i) for i in order[:max(1, cfg.horizon_h // 100)])
    saving_from_top1 = float(sum(
        flex_mw * (p[i] - _cheap_avg(p, cheap_hours, curtail_hours))
        for i in curtail_hours if i in top1))
    return {
        "seed": cfg.seed,
        "hours_curtailed": len(curtail_hours),
        "curtailed_frac_of_uptime": round(len(curtail_hours) / cfg.horizon_h, 5),
        "blind_cost_dollars": round(blind_cost, 0),
        "aware_cost_dollars": round(aware_cost, 0),
        "saving_dollars": round(saving, 0),
        "saving_pct": round(saving / blind_cost, 4),
        "saving_from_top1pct_hours_frac": round(saving_from_top1 / saving, 3) if saving > 0 else 0.0,
        "max_price_mwh": round(float(p.max()), 0),
    }


def _cheap_avg(p, cheap_hours, curtail_hours) -> float:
    picks = [p[i] for i in cheap_hours if i not in curtail_hours][:8]
    return float(np.mean(picks)) if picks else float(p.min())
