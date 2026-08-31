#!/usr/bin/env python3
"""Fleet-level scarcity scheduling (v2) — an under-built market layer.

The move-vs-stay model (../calc) prices ONE relocation decision. This extends it
to a fleet policy: given a data-center fleet with a flexible (deferrable/
curtailable) load fraction and a real-shaped electricity price process — a
diurnal base with rare scarcity spikes — how much does a *scarcity-aware*
scheduler save over a price-blind one, and where does the saving come from?

The finding the single-decision model predicts, now at fleet scale: the saving
is small, it lives entirely in the scarcity hours (with the spike process off no
hour clears the threshold and the saving is exactly zero), and WHICH hours the
budget is spent on decides most of it — ranking the eligible hours by price beats
spending the same budget on a random selection of them in every simulated price
year. This is "schedule against scarcity, not the spread," quantified for a
fleet. The avoided cost is also concentrated in the top ~1% of hours, but that
is a property of the price process rather than of the policy (see the note on
`saving_from_top1pct_hours_frac` below). The headline percentage is bound by the synthetic
spike ceiling ($5k/MWh here, a conservative model input — see validation.py's
`spike-ceiling-is-a-model-input`); what is portable is that RANKING the eligible
hours by price beats not ranking them. It rhymes with the operator reality
(Duke: ~76 GW of new load fits the existing grid if new loads curtail ~0.25% of
annual energy) — a capacity-hosting result, same lever.

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
    # Which above-threshold hours the budget is spent on. The shipped policy
    # ranks by price; the other two exist so the ranking can be switched OFF
    # and the scheduler measured against a scheduler that does not rank.
    # Without them "the policy saves X%" has nothing to be X% better *than*.
    curtail_order: str = "most-expensive-first"


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
    eligible = [int(i) for i in order if p[i] > cfg.curtail_threshold_mwh]
    # The default leaves `eligible` in descending-price order, so the shipped
    # policy is byte-for-byte what it was before this knob existed.
    if cfg.curtail_order == "most-expensive-first":
        pass
    elif cfg.curtail_order == "least-expensive-first":
        eligible.reverse()
    elif cfg.curtail_order == "random":
        np.random.default_rng(cfg.seed + 9_999).shuffle(eligible)
    else:
        raise ValueError(f"unknown curtail_order {cfg.curtail_order!r}")
    curtail_hours = eligible[:budget_h]
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
    # set and the fraction drops below 1. But note what this does NOT show: the
    # ratio is identical under any curtail_order (verified to 6 decimals in
    # ../validation.py), so it measures the PRICE PROCESS, not the policy. It is
    # set by spike_rate_per_h and spike_len_h. For a policy-sensitive measure,
    # compare savings across curtail_order settings.
    top1 = set(int(i) for i in order[:max(1, cfg.horizon_h // 100)])
    gross_avoided = float(sum(flex_mw * p[i] for i in curtail_hours))
    gross_from_top1 = float(sum(flex_mw * p[i] for i in curtail_hours if i in top1))
    return {
        "seed": cfg.seed,
        "hours_curtailed": len(curtail_hours),
        # Hours above the scarcity threshold. When this is <= budget_h the
        # budget does not bind, every eligible hour is curtailed, and the
        # RANKING cannot matter — such runs carry no information about the
        # policy and the validation registry excludes them by name.
        "eligible_hours": len(eligible),
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
