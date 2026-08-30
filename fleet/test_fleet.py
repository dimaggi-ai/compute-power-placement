"""Fleet scarcity-scheduling invariants. Run: python3 test_fleet.py"""
import statistics as st
from fleet_sim import Fleet, simulate, price_series
import numpy as np


def _mean(key, **kw):
    return st.mean(simulate(Fleet(seed=s, **kw))[key] for s in range(8))


def test_determinism():
    assert simulate(Fleet(seed=3)) == simulate(Fleet(seed=3))
    assert simulate(Fleet(seed=3)) != simulate(Fleet(seed=4))


def test_aware_never_costs_more_than_blind():
    for s in range(8):
        r = simulate(Fleet(seed=s))
        assert r["saving_dollars"] >= -1.0, r          # curtailing can't hurt
        assert 0.0 <= r["aware_cost_dollars"] <= r["blind_cost_dollars"] + 1.0


def test_energy_is_conserved():
    """All curtailed flexible energy is deferred to cheap hours, none dropped —
    the saving must come from a lower PRICE, never from doing less WORK."""
    for s in range(8):
        r = simulate(Fleet(seed=s))
        assert abs(r["deferred_mwh"] - r["curtailed_mwh"]) < 1e-6, r


def test_saving_is_avoided_purchase_minus_cheap_deferral():
    """saving == gross avoided purchase in curtailed hours MINUS the (small) cost
    of serving that energy in the cheapest hours. Independent check that the
    deferral is a rounding error, so the saving is essentially the avoided spike
    purchase — and that a broken pricing path would fail here."""
    for s in range(8):
        r = simulate(Fleet(seed=s))
        if r["gross_avoided_dollars"] == 0:            # spike-free month
            assert r["saving_dollars"] == 0
            continue
        assert 0 < r["saving_dollars"] <= r["gross_avoided_dollars"]
        assert r["saving_dollars"] > 0.9 * r["gross_avoided_dollars"]


def test_small_budget_curtails_only_the_worst_hours():
    """The SHIPPED policy curtails only the few worst hours (<=0.4% of uptime),
    which are by design inside the top-1% set — a deliberate design choice, so
    ~all of its (small) avoided cost is top-1%. This is NOT the finding; the
    finding is the concentration test below."""
    assert _mean("curtailed_frac_of_uptime") <= 0.005 + 1e-9
    assert _mean("hours_curtailed") <= 4
    assert _mean("saving_from_top1pct_hours_frac") > 0.95


def test_concentration_is_not_a_budget_artifact():
    """The real finding: give the scheduler a budget large enough to curtail EVERY
    scarcity hour, in a many-spike regime where curtailed hours far outnumber the
    top-1% set (7 of 720). The top-1% of hours STILL captures a large share of the
    avoided cost — strictly below 1.0 (so it is not the 3-hour budget hiding inside
    the 7-hour top-1% set), yet far above the 1% time-share (a real concentration)."""
    frac = _mean("saving_from_top1pct_hours_frac",
                 spike_rate_per_h=0.05, curtail_budget_frac=0.30)
    assert 0.1 < frac < 0.95, frac


def test_no_spikes_means_almost_no_saving():
    """With the scarcity process off, scarcity-aware scheduling has little to do
    (only the diurnal shape), so the saving collapses. The value is in the spikes,
    not the spread."""
    with_spikes = _mean("saving_pct")
    without = _mean("saving_pct", spike_rate_per_h=0.0)
    assert without < with_spikes
    assert without < 0.01                              # <1% without spikes


def test_headline_percent_is_spike_cap_bound():
    """The saving PERCENTAGE is a function of the spike magnitude, not a universal
    constant: halving the spike cap materially shrinks it. The direction (concentration)
    is robust; the number is calibration-bound, and the docs say so."""
    big = _mean("saving_pct", spike_max_mwh=5000.0)
    small = _mean("saving_pct", spike_max_mwh=1000.0)
    assert small < 0.6 * big                           # cap matters a lot


def test_more_flexible_load_saves_more():
    assert _mean("saving_dollars", flexible_fraction=0.1) < _mean("saving_dollars", flexible_fraction=0.4)


def test_price_process_has_scarcity_tail():
    p = price_series(Fleet(seed=1), np.random.default_rng(1))
    assert p.max() > 400                               # a spike occurred
    assert np.median(p) < 100                          # but most hours are normal


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all fleet tests passed")
