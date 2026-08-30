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


def test_tiny_curtailment_budget_respected():
    """The whole point: a small curtailment budget (<=0.5% of uptime) suffices."""
    assert _mean("curtailed_frac_of_uptime") <= 0.005 + 1e-9
    assert _mean("hours_curtailed") <= 4


def test_saving_dominated_by_scarcity_hours():
    """Almost all the saving comes from the top-1% price hours, not the spread."""
    assert _mean("saving_from_top1pct_hours_frac") > 0.9


def test_no_spikes_means_almost_no_saving():
    """With the scarcity process off, scarcity-aware scheduling has little to do
    (only the diurnal shape), so the saving collapses."""
    with_spikes = _mean("saving_pct")
    without = _mean("saving_pct", spike_rate_per_h=0.0)
    assert without < with_spikes
    assert without < 0.01                              # <1% without spikes


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
