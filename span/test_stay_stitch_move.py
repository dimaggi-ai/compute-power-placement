"""Stay/stitch/move invariants. Every asserted number was measured against the
model, not predicted. Run: python3 test_stay_stitch_move.py"""
import dataclasses
import math

from stay_stitch_move import (Circuit, Hall, SpanJob, allreduce_cut, ledger,
                              pipeline_cut, stitch_tax)

JOB = SpanJob()                       # 405B, 16,384 GPUs, one week at full size
DEADLINED = dataclasses.replace(JOB, deadline_h=200.0)
CIRCUIT = Circuit()                   # 400G, 2 ms metro RTT


def _opt(led, name):
    return next(o for o in led["options"] if o["option"].startswith(name))


def test_the_two_cuts_are_different_regimes():
    """A pipeline cut is a few percent; an all-reduce cut eats most of both
    halls. Same circuit, same halls, same job."""
    tp = stitch_tax(JOB, CIRCUIT, pipeline_cut(JOB))
    ta = stitch_tax(JOB, CIRCUIT, allreduce_cut(JOB))
    assert 0.01 < tp < 0.05, tp                       # measured 0.0165
    assert ta > 0.8, ta                               # measured 0.8663


def test_the_allreduce_tax_is_payload_not_latency():
    """A zero-latency circuit changes nothing: the all-reduce tax is the
    gradient payload against the circuit's bandwidth, so no shorter fibre
    rescues it — only ~10x the bandwidth would even dent it."""
    ta = stitch_tax(JOB, CIRCUIT, allreduce_cut(JOB))
    ta_zero_rtt = stitch_tax(JOB, Circuit(rtt_ms=0.0), allreduce_cut(JOB))
    assert abs(ta - ta_zero_rtt) < 0.001, (ta, ta_zero_rtt)
    ta_10x = stitch_tax(JOB, Circuit(gbps=4000.0), allreduce_cut(JOB))
    assert 0.3 < ta_10x < 0.5, ta_10x                 # measured 0.393 — still ruinous


def test_a_job_that_fits_no_hall_stitches_or_escalates_on_the_cut():
    """The scale-across case: neither hall alone can host the job. With a
    pipeline cut the stitch is the only feasible option; with an all-reduce
    cut nothing is feasible and the ledger says escalate — the named cut
    decides whether scale-across exists at all."""
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=10000)
    led = ledger(DEADLINED, home, away, CIRCUIT, pipeline_cut(DEADLINED))
    assert led["decision"] == "stitch"
    assert not _opt(led, "stay")["feasible"] and not _opt(led, "move")["feasible"]
    led2 = ledger(DEADLINED, home, away, CIRCUIT, allreduce_cut(DEADLINED))
    assert led2["decision"] == "escalate"


def test_a_pipeline_stitch_costs_percent_an_allreduce_stitch_costs_multiples():
    """Per useful GPU-hour against the buy reference: the pipeline stitch is a
    ~2% premium; the all-reduce stitch is ~7.5x. This is W2's dollar ledger —
    'rent a circuit' and 'buy in-hall' as comparable numbers."""
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=10000)
    led = ledger(JOB, home, away, CIRCUIT, pipeline_cut(JOB))
    ref = _opt(led, "buy")["dollars_per_useful_gpu_h"]
    pipe = _opt(led, "stitch")["dollars_per_useful_gpu_h"]
    assert 1.0 < pipe / ref < 1.05, pipe / ref        # measured 1.021
    led2 = ledger(JOB, home, away, CIRCUIT, allreduce_cut(JOB))
    ar = _opt(led2, "stitch")["dollars_per_useful_gpu_h"]
    assert ar / ref > 5, ar / ref                     # measured 7.49


def test_stitching_for_cheaper_power_is_dominated():
    """The ../calc echo, one level up: with no capacity shortage, a stitch to
    reach cheaper power loses to BOTH staying and moving — it pays the tax and
    the circuit without escaping the expensive hall."""
    home = Hall("A", free_gpus=20000, price_mwh=60)
    away = Hall("B", free_gpus=20000, price_mwh=30)
    led = ledger(JOB, home, away, CIRCUIT, pipeline_cut(JOB))
    stitch = _opt(led, "stitch")["dollars"]
    assert stitch > _opt(led, "stay")["dollars"]
    assert stitch > _opt(led, "move")["dollars"]
    assert led["decision"] != "stitch"


def test_a_home_spike_favors_move_over_stitch_over_stay():
    """Power scarcity at home: the stitch's home half keeps paying spike
    prices for the whole run, so a full move wins whenever the away hall can
    host the job; staying at full size is worst of all."""
    home = Hall("A", free_gpus=20000, spike_price_mwh=5000, spike_hours_per_week=12)
    away = Hall("B", free_gpus=20000)
    led = ledger(DEADLINED, home, away, CIRCUIT, pipeline_cut(DEADLINED))
    assert led["decision"] == "move"
    m, st, sy = (_opt(led, n)["dollars"] for n in ("move", "stitch", "stay"))
    assert m < st < sy, (m, st, sy)


def test_without_a_deadline_shrinking_wins_on_dollars():
    """Under perfect scaling (which flatters it, and the note says so), the
    shrunk job spends the same GPU-hours and merely takes twice the calendar:
    it is the cheapest option in dollars. The case for the stitch is the
    calendar, not the meter."""
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=20000)
    led = ledger(JOB, home, away, CIRCUIT, pipeline_cut(JOB))
    assert led["decision"] == "stay"
    stay = _opt(led, "stay")
    assert stay["duration_h"] == 2 * JOB.hours_at_full
    # with the deadline the same scenario flips to move (away CAN host here)
    led2 = ledger(DEADLINED, home, away, CIRCUIT, pipeline_cut(DEADLINED))
    assert led2["decision"] == "move"


def test_every_option_delivers_the_same_useful_work():
    """The ledger's unit is the useful GPU-hour: dollars_per_useful_gpu_h x
    useful work reproduces each option's total, so no option quietly delivers
    less than the job."""
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=10000)
    led = ledger(JOB, home, away, CIRCUIT, pipeline_cut(JOB))
    for o in led["options"]:
        if o["feasible"]:
            assert math.isclose(o["dollars_per_useful_gpu_h"] * JOB.useful_gpu_hours,
                                o["dollars"], rel_tol=1e-3), o


def test_the_buy_reference_is_never_the_decision():
    """buy-in-hall is the floor the options are judged against, not a verdict:
    the scarce hall by definition cannot sell the hour."""
    home, away = Hall("A", free_gpus=8192), Hall("B", free_gpus=10000)
    for cut in (pipeline_cut(JOB), allreduce_cut(JOB)):
        led = ledger(JOB, home, away, CIRCUIT, cut)
        assert not led["decision"].startswith("buy")


def test_migration_mechanics_match_the_calc_model():
    """One repo, one migration: the span job's downtime and cost reproduce
    ../calc/move_vs_stay.py for the same inputs (0.644 h, ~$26.5k)."""
    assert math.isclose(JOB.migration_downtime_h, 0.644, abs_tol=1e-3)
    assert math.isclose(JOB.migration_cost_dollars, 26508, rel_tol=1e-3)


def test_a_destination_that_fits_the_job_makes_move_swallow_the_stitch():
    """Measured over the whole decision grid (25x25: deficit 0-60%, spikes
    0-24 h/wk): when the away hall can host the ENTIRE job, the stitch never
    wins a single cell — a one-off migration beats a recurring tax every time.
    The stitch is what you do when no hall is big enough, not a discount move."""
    away = Hall("B", free_gpus=20000)
    for spike in (0.0, 12.0, 24.0):
        for d in (0.0, 0.25, 0.5):
            h = Hall("A", free_gpus=int(DEADLINED.gpus_needed * (1 - d)),
                     spike_price_mwh=5000.0, spike_hours_per_week=spike)
            led = ledger(DEADLINED, h, away, CIRCUIT, pipeline_cut(DEADLINED))
            assert led["decision"] != "stitch", (spike, d, led["decision"])


def test_when_no_hall_can_sum_to_the_job_the_ledger_escalates():
    """The knife edge the decision map shows at its far right: two halls that
    together are ONE GPU short of the job leave nothing feasible under the
    deadline, and the ledger says escalate rather than rounding the job down."""
    home = Hall("A", free_gpus=6553)
    away = Hall("B", free_gpus=int(DEADLINED.gpus_needed * 0.6))   # 9830
    assert home.free_gpus + away.free_gpus == DEADLINED.gpus_needed - 1
    led = ledger(DEADLINED, home, away, CIRCUIT, pipeline_cut(DEADLINED))
    assert led["decision"] == "escalate"


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all stay/stitch/move tests passed")
