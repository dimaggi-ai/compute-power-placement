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
    assert stay["duration_h"] == 2 * JOB.hours_at_full + JOB.reload_h
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
    """Checked over a 3x3 sample (deficit 0, 25, 50%; spikes 0, 12, 24 h/wk):
    when the away hall can host the ENTIRE job, the stitch never
    wins a single cell — a one-off migration beats a recurring tax every time.
    The stitch is what you do when no hall is big enough, not a discount move."""
    away = Hall("B", free_gpus=20000)
    for spike in (0.0, 12.0, 24.0):
        for d in (0.0, 0.25, 0.5):
            h = Hall("A", free_gpus=int(DEADLINED.gpus_needed * (1 - d)),
                     spike_price_mwh=5000.0, spike_hours_per_week=spike)
            led = ledger(DEADLINED, h, away, CIRCUIT, pipeline_cut(DEADLINED))
            assert led["decision"] != "stitch", (spike, d, led["decision"])


def test_decision_grid_has_no_stitch_when_destination_fits_the_whole_job():
    """Exercise all 625 cells of run.py's decision grid, not just nine samples.

    numpy.linspace(0, stop, 25) uses these steps and fixes the final endpoint;
    spelling them out keeps this standalone model test free of plotting imports.
    The 60%-capacity control is the actual plotted scenario and must contain a
    stitch, so the comparison does not pass merely because stitch disappeared.
    """
    job = SpanJob(deadline_h=200.0)
    deficits = [i * (0.6 / 24) for i in range(24)] + [0.6]
    spikes = [i * (24.0 / 24) for i in range(24)] + [24.0]
    whole = Hall("B", free_gpus=job.gpus_needed)
    plotted = Hall("B", free_gpus=int(job.gpus_needed * 0.6))
    count = plotted_stitches = 0
    for spike in spikes:
        for deficit in deficits:
            home = Hall("A", free_gpus=int(job.gpus_needed * (1 - deficit)),
                        spike_price_mwh=5000.0, spike_hours_per_week=spike)
            actual = ledger(job, home, whole, CIRCUIT, pipeline_cut(job))
            assert actual["decision"] != "stitch", (spike, deficit, actual)
            control = ledger(job, home, plotted, CIRCUIT, pipeline_cut(job))
            plotted_stitches += control["decision"] == "stitch"
            count += 1
    assert count == 625
    assert plotted_stitches > 0


def test_when_no_hall_can_sum_to_the_job_the_ledger_escalates():
    """The knife edge the decision map shows at its far right: two halls that
    together are ONE GPU short of the job leave nothing feasible under the
    deadline, and the ledger says escalate rather than rounding the job down."""
    home = Hall("A", free_gpus=6553)
    away = Hall("B", free_gpus=int(DEADLINED.gpus_needed * 0.6))   # 9830
    assert home.free_gpus + away.free_gpus == DEADLINED.gpus_needed - 1
    led = ledger(DEADLINED, home, away, CIRCUIT, pipeline_cut(DEADLINED))
    assert led["decision"] == "escalate"


def test_stitch_transition_can_miss_a_deadline_that_compute_alone_meets():
    job = dataclasses.replace(JOB, deadline_h=171.0)
    led = ledger(job, Hall("home", 8192), Hall("away", 10000), CIRCUIT, pipeline_cut(job))
    assert not _opt(led, "stitch")["feasible"], led
    assert led["decision"] == "escalate", led
    recovered = ledger(dataclasses.replace(job, deadline_h=171.47), Hall("home", 8192),
                       Hall("away", 10000), CIRCUIT, pipeline_cut(job))
    assert recovered["decision"] == "stitch", recovered
    assert _opt(recovered, "stitch")["duration_h"] == 171.5


def test_shrunk_stay_reload_can_miss_a_deadline_and_recover():
    job = dataclasses.replace(JOB, deadline_h=336.2)
    led = ledger(job, Hall("home", 8192), Hall("away", 0), CIRCUIT, pipeline_cut(job))
    assert not _opt(led, "stay")["feasible"], led
    assert led["decision"] == "escalate", led
    recovered = ledger(dataclasses.replace(job, deadline_h=336.5), Hall("home", 8192),
                       Hall("away", 0), CIRCUIT, pipeline_cut(job))
    assert recovered["decision"] == "stay", recovered
    assert _opt(recovered, "stay")["duration_h"] == 336.5


def test_move_cannot_round_elapsed_down_to_meet_a_deadline():
    job = dataclasses.replace(JOB, deadline_h=168.62)
    led = ledger(job, Hall("home", 0), Hall("away", JOB.gpus_needed), CIRCUIT, pipeline_cut(job))
    assert not _opt(led, "move")["feasible"], led
    assert "elapsed duration 168.644h" in _opt(led, "move")["note"], led
    assert led["decision"] == "escalate", led
    recovered = ledger(dataclasses.replace(job, deadline_h=168.65), Hall("home", 0),
                       Hall("away", JOB.gpus_needed), CIRCUIT, pipeline_cut(job))
    assert recovered["decision"] == "move", recovered
    assert _opt(recovered, "move")["duration_h"] == 168.6


def test_full_size_stay_cannot_round_elapsed_up_to_refuse_a_deadline():
    job = dataclasses.replace(JOB, hours_at_full=168.06, deadline_h=168.08, reload_h=50.0)
    led = ledger(job, Hall("home", JOB.gpus_needed), Hall("away", 0), CIRCUIT, pipeline_cut(job))
    assert _opt(led, "stay")["feasible"], led
    assert led["decision"] == "stay", led
    assert _opt(led, "stay")["duration_h"] == 168.1  # display only, no reload at full size


def test_exact_elapsed_threshold_and_adjacent_floats_for_each_option():
    # Binary-exact fixture: four hours of work, a 0.5 h reload, no WAN state,
    # and a pipeline cut with 0.25 s overhead per 1 s step (five compute hours).
    job = SpanJob(params_b=0.0, gpus_needed=4, hours_at_full=4.0, step_time_s=1.0)
    circuit = Circuit(rtt_ms=0.0, gbps=8.0)
    cut = pipeline_cut(job, activation_gb_per_step=0.25)
    for option, home_count, elapsed in (("stay", 2, 8.5), ("stay", 4, 4.0),
                                        ("stitch", 2, 5.5), ("move", 0, 4.5)):
        for deadline, feasible in ((math.nextafter(elapsed, -math.inf), False),
                                   (elapsed, True), (math.nextafter(elapsed, math.inf), True)):
            led = ledger(dataclasses.replace(job, deadline_h=deadline), Hall("home", home_count),
                         Hall("away", 4), circuit, cut)
            row = _opt(led, option)
            assert row["feasible"] is feasible, (option, elapsed, deadline, led)
            if not feasible:
                assert all(row[key] is None for key in ("dollars", "duration_h", "dollars_per_useful_gpu_h")), row
                assert f"elapsed duration {elapsed}h" in row["note"], row
            assert _opt(led, "buy")["feasible"]  # the reference is not scheduled
        # Preserve the historical zero sentinel alongside the default None.
        zero = ledger(dataclasses.replace(job, deadline_h=0.0), Hall("home", home_count),
                      Hall("away", 4), circuit, cut)
        unbounded = ledger(job, Hall("home", home_count), Hall("away", 4), circuit, cut)
        assert zero == unbounded, (zero, unbounded)


def test_zero_transition_and_insufficient_capacity_do_not_invent_time_or_capacity():
    job = SpanJob(params_b=0.0, gpus_needed=4, hours_at_full=4.0, reload_h=0.0)
    circuit = Circuit(rtt_ms=0.0)
    cut = pipeline_cut(job, activation_gb_per_step=0.0)
    led = ledger(job, Hall("home", 2), Hall("away", 4), circuit, cut)
    assert [_opt(led, option)["duration_h"] for option in ("stay", "stitch", "move")] == [8.0, 4.0, 4.0]
    for home, away in ((0, 0), (0, 3), (2, 1)):
        refused = ledger(dataclasses.replace(job, deadline_h=4.0), Hall("home", home),
                         Hall("away", away), circuit, cut)
        assert refused["decision"] == "escalate", refused
        assert not any(o["feasible"] for o in refused["options"][:3]), refused


def test_elapsed_correction_preserves_existing_compute_and_transition_dollar_charges():
    # Frozen synthetic baseline: adding calendar transition time must not charge
    # another compute, power or circuit interval on top of existing transition costs.
    led = ledger(JOB, Hall("home", 8192), Hall("away", 20000), CIRCUIT, pipeline_cut(JOB))
    assert list(led) == ["decision", "stitch_tax", "cut", "capacity_deficit_gpus", "options"]
    assert [row["option"] for row in led["options"]] == ["stay", "stitch", "move", "buy-in-hall (reference)"]
    for row in led["options"]:
        assert list(row) == ["option", "feasible", "dollars", "duration_h", "dollars_per_useful_gpu_h", "note"]
    expected = {"stay": (7122731.0, 2.5877), "stitch": (7261051.0, 2.6380),
                "move": (7138999.0, 2.5936), "buy": (7112491.0, 2.5840)}
    for option, (dollars, per_useful) in expected.items():
        row = _opt(led, option)
        assert row["dollars"] == dollars, row
        assert row["dollars_per_useful_gpu_h"] == per_useful, row
    assert [_opt(led, option)["duration_h"] for option in ("stay", "stitch", "move")] == [336.5, 171.5, 168.6]


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all stay/stitch/move tests passed")
