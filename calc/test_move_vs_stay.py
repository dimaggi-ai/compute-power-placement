"""Move-vs-stay invariants. Run: python3 test_move_vs_stay.py"""
import math
from move_vs_stay import (Job, decide, breakeven_spread_mwh,
                          energy_saving_dollars, scarcity_avoided_dollars, BYTES_PER_PARAM)


def test_state_size_and_power():
    j = Job(params_b=405, gpus=16384, kw_per_gpu=1.4)
    assert math.isclose(j.state_gb, 405 * BYTES_PER_PARAM, rel_tol=1e-9)   # 6,480 GB
    assert math.isclose(j.power_mw, 16384 * 1.4 / 1000, rel_tol=1e-9)      # ~22.9 MW


def test_energy_is_a_small_share_of_gpu_cost():
    """The load-bearing reframe: at typical prices electricity is a few percent
    of the GPU-hour cost, so the $/MWh spread cannot move the decision much."""
    j = Job()
    share = (j.power_mw * j.price_here_mwh) / (j.gpus * j.dollars_per_gpu_h)
    assert 0.02 < share < 0.06, share


def test_breakeven_spread_falls_with_job_length():
    """A longer job amortizes the one-off migration cost, so it breaks even at a
    smaller spread."""
    j = Job()
    be = [breakeven_spread_mwh(j, L) for L in (24, 72, 168, 720)]
    assert be == sorted(be, reverse=True)            # strictly decreasing
    assert be[0] > 40 and be[-1] < 5                 # short job needs a big spread; long job tiny


def test_short_jobs_stay_long_jobs_move_on_modest_spread():
    j = Job(price_here_mwh=60, price_there_mwh=30)    # $30/MWh spread
    assert decide(j, 24)["decision"] == "stay"
    assert decide(j, 168)["decision"] == "move"


def test_scarcity_avoidance_dwarfs_migration():
    """Where relocation actually pays: dodging a scarcity spike returns orders
    of magnitude more than the migration costs."""
    j = Job()
    s = scarcity_avoided_dollars(j, spike_price_mwh=5000, spike_hours=12)
    assert s["net_benefit_of_moving"] > 0
    assert s["net_benefit_of_moving"] > 20 * j.migration_cost_dollars


def test_bigger_state_or_slower_wan_raises_cost():
    small = Job(params_b=70)
    big = Job(params_b=405)
    assert big.migration_cost_dollars > small.migration_cost_dollars
    slow = Job(wan_gbps=10)
    fast = Job(wan_gbps=400)
    assert slow.migration_downtime_h > fast.migration_downtime_h


def test_no_spread_never_moves_on_price_alone():
    j = Job(price_here_mwh=50, price_there_mwh=50)    # zero spread
    assert energy_saving_dollars(j, 1000) == 0
    assert decide(j, 720)["decision"] == "stay"



def _refuses(call, field):
    try:
        call()
    except ValueError as exc:
        assert field in str(exc), str(exc)
    else:
        raise AssertionError('expected ValueError for ' + field)


def test_price_refusal_mutation_and_recovery():
    for field in ('price_here_mwh', 'price_there_mwh'):
        for invalid in (float('nan'), float('inf'), -float('inf'), True, '60', None, 10**1000):
            job = Job()
            setattr(job, field, invalid)
            _refuses(lambda: decide(job, 24), field)
            _refuses(lambda: energy_saving_dollars(job, 24), field)
            setattr(job, field, -10)
            assert math.isfinite(decide(job, 24)['net_dollars'])
    for invalid in (float('nan'), float('inf'), True, '60'):
        _refuses(lambda: scarcity_avoided_dollars(Job(), invalid, 2), 'spike_price_mwh')
    assert energy_saving_dollars(Job(price_here_mwh=-10, price_there_mwh=-20), 2) == Job().power_mw * 10 * 2
    assert energy_saving_dollars(Job(price_here_mwh=0, price_there_mwh=0), 2) == 0


def test_horizon_refusal_and_undefined_breakeven():
    job = Job()
    for horizon in (0, job.migration_downtime_h/2, job.migration_downtime_h):
        _refuses(lambda: breakeven_spread_mwh(job, horizon), 'earning time')
        assert decide(job, horizon)['decision'] == 'stay'
    horizon = math.nextafter(job.migration_downtime_h, math.inf)
    assert breakeven_spread_mwh(job, horizon) == job.migration_cost_dollars / (job.power_mw * (horizon-job.migration_downtime_h))
    for horizon in (-1, float('nan'), float('inf'), True, '24'):
        _refuses(lambda: decide(job, horizon), 'job_length_h')
        _refuses(lambda: breakeven_spread_mwh(job, horizon), 'job_length_h')
        _refuses(lambda: energy_saving_dollars(job, horizon), 'remaining_h')
        _refuses(lambda: scarcity_avoided_dollars(job, 40, horizon), 'spike_hours')


def test_nonfinite_derived_cost_is_refused():
    _refuses(lambda: decide(Job(price_here_mwh=1e308), 24), 'finite')
    _refuses(lambda: energy_saving_dollars(Job(price_here_mwh=1e308, price_there_mwh=-1e308), 2), 'finite')
    _refuses(lambda: scarcity_avoided_dollars(Job(), 1e308, 24), 'finite')


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all move-vs-stay tests passed")
