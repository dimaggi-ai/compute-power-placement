"""Validation-registry tests, including negative controls.

A registry test suite that only asserts the registry passes is a
tautology detector with the detector switched off. Four of the tests
below assert that a deliberately broken model FAILS the registry, and
one asserts no tolerance is wide enough to be meaningless.

Run: python3 test_validation.py
"""

import statistics
import sys
from pathlib import Path

from validation import MONTH_SEEDS, YEAR_H, YEAR_SEEDS, DECLINED, points, validate

sys.path.insert(0, str(Path(__file__).resolve().parent / "fleet"))
from fleet_sim import Fleet, simulate                          # noqa: E402

_POINTS = points()


def test_every_point_reproduces():
    for p in _POINTS:
        assert p.ok, (
            f"{p.name}: expected {p.expected} +/- {p.tolerance} {p.ref}, "
            f"model gives {p.actual:.4f}"
        )


def test_all_three_kinds_present():
    kinds = {p.kind for p in _POINTS}
    assert kinds == {"calibrated", "emergent", "sanity"}, kinds


def test_only_calibrated_points_cite_sources():
    """Emergent points here run on synthetic prices; a ref beside one would
    dress a simulation result as a sourced result. Sanity points claim no
    evidence. So exactly the calibrated points carry a ref."""
    for p in _POINTS:
        if p.kind == "calibrated":
            assert p.ref.startswith("["), f"{p.name} must cite a source"
        else:
            assert p.ref == "-", f"{p.name} ({p.kind}) must not cite a source"


def test_model_column_is_declared():
    for p in _POINTS:
        assert p.model in ("calc", "fleet", "-"), (p.name, p.model)


def test_no_tolerance_is_wide_enough_to_be_meaningless():
    """A band wider than 20% of the value it brackets is a range, not a pin."""
    for p in _POINTS:
        if p.expected:
            rel = p.tolerance / abs(p.expected)
            assert rel <= 0.20, (
                f"{p.name}: +/-{p.tolerance} on {p.expected} is {rel:.0%} — "
                f"too wide to fail"
            )


def test_registry_has_not_silently_shrunk():
    assert len(_POINTS) == 16, len(_POINTS)


def test_declined_anchors_are_disclosed():
    assert len(DECLINED) >= 4
    for what, why in DECLINED:
        assert what and why


# --------------------------------------------------------------- negative
# controls: each asserts the registry FAILS a model it should reject.

def test_negative_control_an_unranked_scheduler_fails_the_ranking_points():
    """The registry's central claim is that ranking by price buys something.
    If a scheduler that picks its curtailment hours at RANDOM scored the
    same, the points would be measuring the price process. It does not."""
    beats = 0
    for s in YEAR_SEEDS:
        base = simulate(Fleet(seed=s, horizon_h=YEAR_H))["saving_pct"]
        rnd = simulate(Fleet(seed=s, horizon_h=YEAR_H,
                             curtail_order="random"))["saving_pct"]
        beats += rnd > base
    assert beats == 0, (
        f"an unranked scheduler beat the shipped one in {beats} years — "
        f"the ranking points cannot discriminate"
    )


def test_negative_control_a_reversed_scheduler_scores_materially_worse():
    """Reversing the ranking must cost real money, not a rounding error."""
    ratios = []
    for s in YEAR_SEEDS:
        base = simulate(Fleet(seed=s, horizon_h=YEAR_H))["saving_pct"]
        cheap = simulate(Fleet(seed=s, horizon_h=YEAR_H,
                               curtail_order="least-expensive-first"))["saving_pct"]
        ratios.append(base / cheap)
    assert min(ratios) > 2.0, min(ratios)


def test_negative_control_a_spikeless_market_fails_the_headline_points():
    """The headline saving must depend on the scarcity process. With spikes
    off it is exactly zero, which is outside both headline bands."""
    saving = statistics.mean(
        simulate(Fleet(seed=s, spike_rate_per_h=0.0))["saving_pct"]
        for s in MONTH_SEEDS)
    for p in _POINTS:
        if p.name.startswith("headline-saving"):
            assert abs(saving - p.expected) > p.tolerance, (
                f"{p.name} accepts a market with no scarcity at all"
            )


def test_negative_control_the_two_ceilings_are_not_interchangeable():
    """The $5,000 and $2,000 headline points must be distinguishable — if one
    band accepted the other's value, pinning both would prove nothing."""
    at_5k = next(p for p in _POINTS if p.name.endswith("5000-ceiling"))
    at_2k = next(p for p in _POINTS if p.name.endswith("rt-cap-2000"))
    assert abs(at_5k.actual - at_2k.expected) > at_2k.tolerance
    assert abs(at_2k.actual - at_5k.expected) > at_5k.tolerance


def test_validate_reports_all_ok():
    pts, ok = validate()
    assert ok
    assert len(pts) == len(_POINTS)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all validation tests passed")
