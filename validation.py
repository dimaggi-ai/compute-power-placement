#!/usr/bin/env python3
"""The validation project: both models vs the public record, in one table.

Three kinds of points, plus a column saying which model each one actually
exercises — because "checked against the public record" is a claim about
evidence, and a check that touches no model is not evidence about a model:

  calibrated  compared against a number a cited source publishes. Passing
              proves the repo has not drifted from its citations, not that
              the model predicts. Must carry a ref.
  emergent    a behaviour of a model that nothing was tuned to produce,
              asserted across seeds. Every emergent point here runs on
              SYNTHETIC prices and carries ref '-': no source publishes
              these numbers, and a ref column beside them would dress a
              simulation result as a sourced one.
  sanity      deterministic properties of the models and their inputs,
              pinned so the README's own arithmetic cannot drift. Cites
              no external evidence and claims none.

  model       'calc', 'fleet', or '-' for pure source arithmetic that
              exercises neither. The summary line counts these separately.

The central emergent claim is deliberately one that can FAIL: the shipped
scheduler's PRICE RANKING is measured against the same scheduler with its
ranking removed (random order) and reversed (cheapest-first). If ranking
by price did nothing, these points would fail — and at month scale they
DO fail, which is why they are asserted at year scale and the month-scale
result is stated below rather than hidden.

Anchors this registry deliberately does NOT check are listed in DECLINED
and printed with the table.

References are to REFERENCES.md. Run: python3 validation.py
(exit 1 if any point fails).
"""

from __future__ import annotations

import dataclasses
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "calc"))
sys.path.insert(0, str(HERE / "fleet"))

from fleet_sim import Fleet, simulate                          # noqa: E402
from move_vs_stay import (                                     # noqa: E402
    Job,
    breakeven_spread_mwh,
    decide,
    scarcity_avoided_dollars,
)

MONTH_SEEDS = range(24)     # the study's 24-seed month-scale runs
YEAR_SEEDS = range(24)      # synthetic price years (8,760 h each), same count
YEAR_H = 8760

# Anchors a reader might expect here, and why they are absent. A registry
# that lists only the checks it passes is a highlight reel.
DECLINED: tuple[tuple[str, str], ...] = (
    ("A published price-concentration statistic for a real ISO",
     "the concentration number in this repo is a property of the SYNTHETIC "
     "price process (see concentration-is-a-property-of-the-price-process), "
     "not a measurement of ERCOT. Checking it against a real ISO figure "
     "would require nodal settlement data this repo does not ship, so the "
     "point is labelled sanity and claims nothing about ERCOT."),
    ("The absolute saving percentage as a forecast",
     "it is bound by the synthetic spike ceiling — 4.8% at $5,000/MWh and "
     "2.7% at $2,000/MWh, both pinned below. Neither is a prediction about "
     "any operator's bill; only the RANKING points are portable."),
    ("Duke's 76 GW hosting-capacity result",
     "reproducing it needs a grid model this repo does not have. Only the "
     "curtailment DEPTH is compared, in energy terms, and the model sits at "
     "about half Duke's level (see curtailment-share-of-fleet-energy)."),
    ("Whether REFERENCES.md quotes its sources correctly",
     "every-ref-resolves proves each ref NUMBER resolves to an entry; no "
     "numeric check can prove the entry paraphrases the source faithfully. "
     "That is a human review job and is not claimed here."),
)


@dataclasses.dataclass(frozen=True)
class Point:
    name: str
    kind: str        # 'calibrated' | 'emergent' | 'sanity'
    ref: str         # '[n]' for calibrated; '-' for emergent and sanity
    model: str       # 'calc' | 'fleet' | '-'
    expected: float
    tolerance: float
    actual: float
    note: str

    @property
    def ok(self) -> bool:
        return abs(self.actual - self.expected) <= self.tolerance


def _year(seed: int, order: str = "most-expensive-first") -> dict:
    return simulate(Fleet(seed=seed, horizon_h=YEAR_H, curtail_order=order))


def _binding_years() -> list[int]:
    """Years where the curtailment budget actually binds.

    When there are fewer above-threshold hours than the budget allows, every
    eligible hour is curtailed and the ranking CANNOT matter. Such years say
    nothing about the policy either way, so they are excluded by name rather
    than counted as passes. At year scale none are excluded; the exclusion
    exists so that a re-parameterisation cannot silently pad the count.
    """
    out = []
    for s in YEAR_SEEDS:
        r = _year(s)
        if r["eligible_hours"] > r["hours_curtailed"]:
            out.append(s)
    return out


def points() -> tuple[Point, ...]:
    pts: list[Point] = []
    job = Job()

    # ---------------------------------------------------------- calibrated
    pts.append(Point(
        "state-size-405b", "calibrated", "[5]", "calc",
        expected=405.0 * 16, tolerance=0.0, actual=job.state_gb,
        note="Training state for 405B at the published ZeRO convention of "
             "16 bytes/param = 6,480 GB (~6.5 TB). ZeRO's 16 is fp16 "
             "params (2) + fp16 grads (2) + fp32 params, momentum and "
             "variance (4+4+4) — not 'fp32 weights + Adam moments', which "
             "is 12 and was this repo's gloss until it was corrected.",
    ))
    pts.append(Point(
        "energy-share-of-gpu-cost", "calibrated", "[4]", "calc",
        expected=0.035, tolerance=0.005,
        actual=decide(job, 168.0)["energy_share_of_gpu_cost"],
        note="A burdened 1.4 kW GPU at $60/MWh against a $2.50 GPU-hour = "
             "3.4%. The INPUTS are published figures; the 3-4% band is this "
             "repo's own derivation from them, not a figure anyone "
             "published, and independent TCO analyses put energy at 7-11% "
             "because they use $83-120/MWh. The sensitivity to the power "
             "price is the real content — at $120/MWh this doubles — and "
             "the conclusion (energy is a minority of the GPU-hour) "
             "survives the whole range.",
    ))
    pts.append(Point(
        "curtailed-vs-dc-load-ratio", "calibrated", "[2]", "-",
        expected=0.11, tolerance=0.01, actual=20.0 / 176.0,
        note="Source arithmetic, no model: ~20 TWh 2024 curtailment over "
             "176 TWh 2023 US data-center load = 11.4%. The years do not "
             "match; on IEA's 2024 load figure it is 10.9%, so the "
             "mismatch does not carry the claim.",
    ))
    pts.append(Point(
        "weight-transfer-arithmetic", "calibrated", "[8]", "-",
        expected=0.64, tolerance=0.001, actual=400e12 * 8 / 5e15,
        note="Source arithmetic, no model: 400 TB of weights over "
             "5 Pbit/s inter-campus = 0.64 s, stated verbatim by the "
             "source. Note the 5 Pbit/s is the source's own forward-looking "
             "ASSUMPTION ('a reasonable assumption'), not a measurement, so "
             "this pins an arithmetic claim about a hypothetical fabric.",
    ))

    # ------------------------------------------------------------ emergent
    # The discriminating test: does ranking the eligible hours by price
    # actually buy anything? Same model, same budget, same threshold, same
    # prices — only the CHOICE among above-threshold hours changes.
    binding = _binding_years()
    beats_random, beats_cheapest = 0, 0
    ratio_r, ratio_c = [], []
    for s in binding:
        base = _year(s)["saving_pct"]
        rnd = _year(s, "random")["saving_pct"]
        cheap = _year(s, "least-expensive-first")["saving_pct"]
        beats_random += base > rnd
        beats_cheapest += base > cheap
        ratio_r.append(base / rnd)
        ratio_c.append(base / cheap)
    pts.append(Point(
        "ranking-beats-unranked-in-every-price-year", "emergent", "-", "fleet",
        expected=float(len(binding)), tolerance=0.0, actual=float(beats_random),
        note=f"In all {len(binding)} synthetic price years (8,760 h each), "
             f"spending the SAME curtailment budget on the highest-priced "
             f"eligible hours beats spending it on a random selection of "
             f"the same eligible hours, by {min(ratio_r):.2f}-"
             f"{max(ratio_r):.2f}x (mean {statistics.mean(ratio_r):.2f}x). "
             f"This is the point that can fail: it is the scheduler's "
             f"ranking measured against no ranking. It holds 64/64 over "
             f"range(64). At MONTH scale it fails in 5 of 23 binding seeds "
             f"— a 720-hour draw is too short to decide it, which is why "
             f"the year runs exist.",
    ))
    pts.append(Point(
        "ranking-beats-inverted-in-every-price-year", "emergent", "-", "fleet",
        expected=float(len(binding)), tolerance=0.0,
        actual=float(beats_cheapest),
        note=f"The adversarial control: against a scheduler that curtails "
             f"the CHEAPEST eligible hours, the shipped ranking wins in all "
             f"{len(binding)} years by {min(ratio_c):.2f}-{max(ratio_c):.2f}x "
             f"(mean {statistics.mean(ratio_c):.2f}x). A metric that could "
             f"not separate these two would be measuring the price process, "
             f"not the policy.",
    ))
    saving_5k = statistics.mean(
        simulate(Fleet(seed=s))["saving_pct"] for s in MONTH_SEEDS)
    pts.append(Point(
        "headline-saving-at-5000-ceiling", "emergent", "-", "fleet",
        expected=0.048, tolerance=0.006, actual=saving_5k,
        note="The fleet-bill saving at the shipped $5,000/MWh spike "
             "ceiling (24-seed month mean, 4.83%). Ceiling-bound by "
             "construction — see the point below, which pins the same "
             "figure at ERCOT's post-RTC+B real-time cap. The RANKING "
             "points above, not this percentage, are the portable claim.",
    ))
    saving_2k = statistics.mean(
        simulate(Fleet(seed=s, spike_max_mwh=2000.0))["saving_pct"]
        for s in MONTH_SEEDS)
    pts.append(Point(
        "headline-saving-at-rt-cap-2000", "emergent", "-", "fleet",
        expected=0.027, tolerance=0.005, actual=saving_2k,
        note="The same 24-seed mean with the spike ceiling set to $2,000/MWh "
             "— ERCOT's REAL-TIME system-wide offer cap since the RTC+B "
             "go-live on 2025-12-05, which split the former single $5,000 "
             "cap into $5,000 day-ahead and $2,000 real-time. The saving is "
             "2.71%: the headline roughly halves. Both figures are pinned "
             "here so neither can be quoted without the other.",
    ))

    # -------------------------------------------------------------- sanity
    pts.append(Point(
        "spike-ceiling-is-a-model-input", "sanity", "-", "fleet",
        expected=5000.0, tolerance=0.0, actual=Fleet().spike_max_mwh,
        note="$5,000/MWh is the ceiling of the synthetic spike DISTRIBUTION "
             "— a deliberately conservative model input, not a quoted cap. "
             "It was labelled 'ERCOT's offer cap, used verbatim'; that was "
             "wrong twice over. ERCOT's single $5,000 cap no longer exists "
             "(RTC+B, 2025-12-05: $5,000 day-ahead / $2,000 real-time), and "
             "the series is a NODAL proxy, where LMP = energy + congestion "
             "+ loss and the congestion term is unbounded — REFERENCES.md "
             "[1] records a node at $28,187/MWh. So no cap binds this "
             "series at all; the ceiling is a choice, pinned as one.",
    ))
    pooled_num = pooled_den = 0.0
    inverted_gap = 0.0
    for s in YEAR_SEEDS:
        r = simulate(Fleet(seed=s, horizon_h=YEAR_H, curtail_budget_frac=1.0))
        inv = simulate(Fleet(seed=s, horizon_h=YEAR_H, curtail_budget_frac=1.0,
                             curtail_order="least-expensive-first"))
        g = r["gross_avoided_dollars"]
        if g > 0:
            pooled_num += g * r["saving_from_top1pct_hours_frac"]
            pooled_den += g
        inverted_gap = max(inverted_gap, abs(
            r["saving_from_top1pct_hours_frac"]
            - inv["saving_from_top1pct_hours_frac"]))
    pts.append(Point(
        "concentration-is-a-property-of-the-price-process", "sanity", "-",
        "fleet",
        expected=0.804, tolerance=0.02, actual=pooled_num / pooled_den,
        note=f"With the budget freed to curtail every scarcity hour, 80.4% "
             f"of avoided cost falls in the top 1% of hours "
             f"(dollar-weighted pool over {len(list(YEAR_SEEDS))} price "
             f"years; the mean-of-ratios is higher because spike-poor draws "
             f"contribute a 0/0 sentinel). This is SANITY, not emergent: "
             f"reversing the scheduler's ranking changes it by "
             f"{inverted_gap:.6f} — it is blind to the policy, and is set "
             f"by spike_rate_per_h and spike_len_h. The repo previously "
             f"called it the portable finding; it is a restatement of how "
             f"the price process was built.",
    ))
    curt_frac = statistics.mean(
        simulate(Fleet(seed=s))["curtailed_frac_of_uptime"]
        for s in MONTH_SEEDS)
    pts.append(Point(
        "curtailment-share-of-fleet-energy", "sanity", "-", "fleet",
        expected=0.0012, tolerance=0.0002,
        actual=curt_frac * Fleet().flexible_fraction,
        note="Curtailment expressed in Duke's units. Duke's 0.25% is 0.25% "
             "of annual ENERGY (876,000 GWh -> 2,190 GWh for a 100 GW "
             "load), not 0.25% of hours; this repo previously compared it "
             "against a fraction of hours and reported the model as ABOVE "
             "Duke. It is not: curtailing the 30% flexible tier in 0.40% of "
             "hours is 0.12% of fleet energy, about HALF Duke's level. The "
             "0.5% hour budget is an input, so this pins an input's "
             "consequence, not a finding — hence sanity.",
    ))
    no_spikes = max(
        simulate(Fleet(seed=s, spike_rate_per_h=0.0))["saving_pct"]
        for s in MONTH_SEEDS)
    pts.append(Point(
        "spikes-off-policy-is-inert", "sanity", "-", "fleet",
        expected=0.0, tolerance=0.0, actual=no_spikes,
        note="With the scarcity process off the maximum price across all "
             "seeds is $85/MWh, below the $200 curtail threshold, so ZERO "
             "hours are curtailed and the saving is exactly 0.0 — not "
             "'under 1% from diurnal deferral', which described a mechanism "
             "that never runs. Tolerance is 0.0 because the answer is "
             "exactly zero.",
    ))
    net = scarcity_avoided_dollars(job, 5000.0, 12.0)["net_benefit_of_moving"]
    pts.append(Point(
        "scarcity-headline-arithmetic", "sanity", "-", "calc",
        expected=1.34e6, tolerance=2e4, actual=net,
        note="Deterministic arithmetic pinned: dodging one 12-hour "
             "$5,000/MWh event nets ~$1.34M against the ~$26.5k migration "
             "— the README's headline pair. $5,000 here is illustrative of "
             "a day-ahead-cap-scale event, not a real-time price.",
    ))
    hours_at_30 = (job.migration_cost_dollars / (job.power_mw * 30.0)
                   + job.migration_downtime_h)
    pts.append(Point(
        "breakeven-job-length", "sanity", "-", "calc",
        expected=39.17, tolerance=0.05, actual=hours_at_30,
        note="Job length at which a sustained $30/MWh spread repays one "
             "migration: 39.17 h — days, not hours, so short jobs stay. "
             "Tolerance is 0.05 h, tight enough that any input drift moves "
             "it (the previous 48 +/- 24 h band passed an 8x error in "
             "wan_gbps).",
    ))
    pts.append(Point(
        "breakeven-function-matches-closed-form", "sanity", "-", "calc",
        expected=30.0, tolerance=1e-6,
        actual=breakeven_spread_mwh(job, hours_at_30),
        note="Internal consistency, stated as such: the shipped "
             "breakeven_spread_mwh() evaluated at the length above returns "
             "$30/MWh. This cannot drift (both sides move together) — it "
             "catches a bug in either formula, nothing more. It exists "
             "because the point above re-implements the inverse arithmetic, "
             "leaving the real function unexercised.",
    ))
    refs_defined = set(re.findall(
        r"- \*\*\[(\d+)\]\*\*", (HERE / "REFERENCES.md").read_text()))
    refs_used = {r.strip("[]") for p in pts for r in [p.ref] if p.ref != "-"}
    pts.append(Point(
        "every-ref-resolves", "sanity", "-", "-",
        expected=0.0, tolerance=0.0,
        actual=float(len(refs_used - refs_defined)),
        note=f"Every ref this registry prints resolves to a REFERENCES.md "
             f"entry (used: {sorted(refs_used, key=int)}). Until this "
             f"existed nothing in the registry read REFERENCES.md at all, "
             f"while the README claimed the repo 'cannot drift from its "
             f"citations' — the pins were against literals in this file.",
    ))

    return tuple(pts)


def validate() -> tuple[tuple[Point, ...], bool]:
    pts = points()
    return pts, all(p.ok for p in pts)


def main() -> int:
    pts, ok = validate()
    w = max(len(p.name) for p in pts)
    print(f"{'point':<{w}}  {'kind':<10}  {'ref':<5}  {'model':<6}  "
          f"{'expected':>10}  {'actual':>10}  verdict")
    for p in pts:
        print(f"{p.name:<{w}}  {p.kind:<10}  {p.ref:<5}  {p.model:<6}  "
              f"{p.expected:>10.4g}  {p.actual:>10.4g}  "
              f"{'PASS' if p.ok else 'FAIL'}")
    print()
    print("anchors this registry does NOT check:")
    for what, why in DECLINED:
        print(f"  - {what}\n      {why}")
    print()
    if ok:
        n_cal = sum(1 for p in pts if p.kind == "calibrated")
        n_model = sum(1 for p in pts if p.model != "-")
        print(f"all {len(pts)} points reproduced — {n_cal} calibrated "
              f"against cited figures, {n_model} exercising a model, the "
              f"rest source arithmetic. The emergent points measure the "
              f"scheduler's ranking against the same scheduler unranked "
              f"and reversed, so they can fail; the concentration figure "
              f"is labelled sanity because it cannot.")
    else:
        print("VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
