#!/usr/bin/env python3
"""Stay / stitch / move: one ledger, two scarcity types.

The move-vs-stay model (../calc) prices ONE relocation against ONE scarcity
type — power. This model adds the second scarcity type — capacity: the hall the
job wants does not have the accelerators — and the third option that only
exists once there are two halls: STITCH a circuit between them and run the job
across both.

The question, in the operator's units: a job needs G accelerators for H hours;
the home hall is short, or about to hit a power spike, or both. Four numbers,
one currency (dollars per USEFUL GPU-hour):

  stay    — shrink to what the home hall has and run longer
  stitch  — rent an inter-hall circuit and run across both halls, paying the
            circuit hold cost and the RTT/bandwidth tax on every step
  move    — migrate the whole job to the other hall (../calc mechanics: drain,
            WAN state transfer, reload — the fleet idles throughout)
  buy     — the reference price: what the same useful GPU-hour would cost if
            the scarce hall could simply sell you one. It cannot — that is
            what capacity scarcity means — but every option above must be
            compared to it, or the ledger has no floor.

The finding this model lands (and the tests pin): the STITCH TAX IS THE NAMED
CUT, not the circuit price. Cut the job at a pipeline-stage boundary and the
circuit carries activations — a few GB a step — and the tax is a few percent.
Cut it through a data-parallel All-Reduce and the circuit carries the gradient
exchange every step, and the tax eats most of both halls: every GPU-hour on
EITHER side of the cut buys a fraction of the useful work. The same circuit,
the same halls, the same job — one cut is a stitch, the other is a mistake.
And the power-scarcity echo of ../calc holds one level up: stitching to reach
cheaper power (with no capacity shortage) never pays, for the same reason
spread-arbitrage never paid — the fix costs more than the fuel.

Perfect scaling is assumed for the shrunk job, which flatters STAY; the model
says so rather than hiding it. All inputs are explicit. Calibration sources:
../REFERENCES.md; migration mechanics identical to ../calc/move_vs_stay.py.
"""
from __future__ import annotations
import dataclasses

BYTES_PER_PARAM = 16          # fp32 weights + Adam moments (ZeRO convention, as ../calc)
GRAD_BYTES_PER_PARAM = 2      # bf16 gradients — what a data-parallel cut exchanges


@dataclasses.dataclass(frozen=True)
class Hall:
    hall_id: str
    free_gpus: int
    price_mwh: float = 60.0
    # expected scarcity at this hall while the job runs: how many hours a week
    # the price sits at spike level. An expectation, not a forecast.
    spike_price_mwh: float = 0.0
    spike_hours_per_week: float = 0.0

    def expected_price_mwh(self) -> float:
        """Duration-weighted expected $/MWh, spikes folded in."""
        if self.spike_hours_per_week <= 0:
            return self.price_mwh
        frac = min(1.0, self.spike_hours_per_week / 168.0)
        return (1 - frac) * self.price_mwh + frac * self.spike_price_mwh


@dataclasses.dataclass(frozen=True)
class Circuit:
    rtt_ms: float = 2.0            # metro round trip; 2 ms ~ 200 km of fibre
    gbps: float = 400.0            # one rented 400G wave
    hold_dollars_per_h: float = 15.0   # circuit hold cost while the job runs


@dataclasses.dataclass(frozen=True)
class Cut:
    """The named cut: what actually crosses the circuit each optimizer step."""
    name: str
    gb_over_cut_per_step: float    # payload the circuit carries per step
    crossings_per_step: int        # serialized RTT crossings per step


@dataclasses.dataclass(frozen=True)
class SpanJob:
    params_b: float = 405.0
    gpus_needed: int = 16_384
    hours_at_full: float = 168.0       # one week at full size
    step_time_s: float = 5.0           # seconds per optimizer step at full size (input, not a claim)
    kw_per_gpu: float = 1.4
    dollars_per_gpu_h: float = 2.5     # what a GPU-hour costs in either hall
    deadline_h: float | None = None    # calendar limit on wall-clock duration
    # migration mechanics — same formula and defaults as ../calc/move_vs_stay.py
    wan_gbps: float = 100.0
    reload_h: float = 0.5
    egress_per_gb: float = 0.02

    @property
    def useful_gpu_hours(self) -> float:
        return self.gpus_needed * self.hours_at_full

    @property
    def state_gb(self) -> float:
        return self.params_b * BYTES_PER_PARAM        # params_b * 1e9 * 16 / 1e9

    @property
    def power_mw_full(self) -> float:
        return self.gpus_needed * self.kw_per_gpu / 1000.0

    @property
    def migration_downtime_h(self) -> float:
        return self.reload_h + (self.state_gb * 8) / self.wan_gbps / 3600.0

    @property
    def migration_cost_dollars(self) -> float:
        idle = self.gpus_needed * self.migration_downtime_h * self.dollars_per_gpu_h
        return idle + self.state_gb * self.egress_per_gb


def pipeline_cut(job: SpanJob, activation_gb_per_step: float = 4.0) -> Cut:
    """Cut at a pipeline-stage boundary: the circuit carries one stage's
    activations forward and gradients back. The payload is an input because it
    is batch- and model-shape-dependent; the default is a few GB, which is the
    right order for a frontier step. Two serialized crossings per step (the
    pipeline flush at the step boundary); microbatch overlap hides the rest."""
    return Cut("pipeline-stage", activation_gb_per_step, 2)


def allreduce_cut(job: SpanJob) -> Cut:
    """Cut through data parallelism: the circuit carries the gradient exchange
    every step — the full bf16 gradient at least once in each direction
    (reduce one way, the reduced result back). This is the cut the span
    contract names as usually illegal, and the ledger shows why."""
    gb = 2 * job.params_b * GRAD_BYTES_PER_PARAM      # 2 * params_b*1e9*2 / 1e9
    return Cut("data-parallel all-reduce", gb, 2)


def stitch_tax(job: SpanJob, circuit: Circuit, cut: Cut) -> float:
    """Fraction of wall-clock lost to the cut: serialized RTT crossings plus
    the payload's serialization time on the circuit, per step, against the
    step time. 0 = free; 1 = the job never advances."""
    extra_s = (cut.crossings_per_step * circuit.rtt_ms / 1000.0
               + cut.gb_over_cut_per_step * 8 / circuit.gbps)
    return extra_s / (job.step_time_s + extra_s)


def _energy_dollars(mw: float, hall: Hall, hours: float) -> float:
    return mw * hall.expected_price_mwh() * hours


def _option(name, feasible, dollars, duration_h, note, job):
    per_useful = dollars / job.useful_gpu_hours if feasible else float("inf")
    return {
        "option": name,
        "feasible": feasible,
        "dollars": round(dollars, 0) if feasible else None,
        "duration_h": round(duration_h, 1) if feasible else None,
        "dollars_per_useful_gpu_h": round(per_useful, 4) if feasible else None,
        "note": note,
    }


def ledger(job: SpanJob, home: Hall, away: Hall,
           circuit: Circuit, cut: Cut) -> dict:
    """The three-way decision — stay in the scarce hall, stitch a circuit, or
    move — plus the buy reference, priced per useful GPU-hour delivered.

    The job is RUNNING at the home hall when scarcity lands, so every option
    pays its transition: stay-shrunk pays a restart at the smaller size;
    stitch pays the same state transfer and restart as move (the away half
    needs the checkpoint there before it can hold a step — an upper bound,
    charged so the stitch is never flattered against the move); move pays the
    full migration of ../calc."""
    R = job.useful_gpu_hours
    deficit = max(0, job.gpus_needed - home.free_gpus)

    # ---- STAY: shrink to the home hall and run longer (perfect scaling).
    g_home = min(job.gpus_needed, home.free_gpus)
    if g_home > 0:
        dur = R / g_home
        mw = g_home * job.kw_per_gpu / 1000.0
        dollars = (R * job.dollars_per_gpu_h            # same GPU-hours, longer
                   + _energy_dollars(mw, home, dur)
                   + (g_home * job.reload_h * job.dollars_per_gpu_h
                      if deficit else 0.0))   # restart only if it must shrink
        stay = _option(
            "stay", True, dollars, dur,
            ("shrunk to %d of %d GPUs; perfect scaling assumed (flatters this "
             "option); the longer run is exposed to the home hall's spikes for "
             "its whole duration" % (g_home, job.gpus_needed)) if deficit
            else "full size at home",
            job)
    else:
        stay = _option("stay", False, 0, 0, "the home hall has nothing free", job)
    if stay["feasible"] and job.deadline_h and stay["duration_h"] > job.deadline_h:
        stay = _option("stay", False, 0, 0,
                       f"shrunk duration {stay['duration_h']}h misses the "
                       f"deadline ({job.deadline_h}h)", job)

    # ---- STITCH: full size across both halls, tax on every step.
    tax = stitch_tax(job, circuit, cut)
    g_away_needed = deficit
    can_stitch = home.free_gpus > 0 and away.free_gpus >= g_away_needed and deficit > 0
    if deficit == 0:
        # no capacity scarcity: a stitch could only be chasing power prices —
        # priced anyway, so the echo of ../calc is a number, not an assertion
        can_stitch = away.free_gpus >= job.gpus_needed // 2
        g_away_needed = job.gpus_needed // 2 if can_stitch else 0
    if can_stitch and tax < 1.0:
        g_home_used = job.gpus_needed - g_away_needed
        dur = R / (job.gpus_needed * (1 - tax))
        mw_home = g_home_used * job.kw_per_gpu / 1000.0
        mw_away = g_away_needed * job.kw_per_gpu / 1000.0
        dollars = (job.gpus_needed * dur * job.dollars_per_gpu_h   # taxed GPU-hours
                   + circuit.hold_dollars_per_h * dur
                   + _energy_dollars(mw_home, home, dur)
                   + _energy_dollars(mw_away, away, dur)
                   + job.migration_cost_dollars)   # state to the away half + restart
        stitch = _option(
            "stitch", True, dollars, dur,
            f"cut: {cut.name}; tax {tax:.1%} of every step on both sides; "
            f"{g_home_used} GPUs home + {g_away_needed} away; the home share "
            "stays exposed to home spikes", job)
    else:
        stitch = _option("stitch", False, 0, 0,
                         "away hall cannot supply the deficit" if deficit
                         else "no capacity to reach", job)
    if stitch["feasible"] and job.deadline_h and stitch["duration_h"] > job.deadline_h:
        stitch = _option("stitch", False, 0, 0,
                         f"taxed duration {stitch['duration_h']}h misses the "
                         f"deadline ({job.deadline_h}h)", job)

    # ---- MOVE: the whole job to the away hall (destination must exist).
    if away.free_gpus >= job.gpus_needed:
        dur = job.migration_downtime_h + job.hours_at_full
        dollars = (job.migration_cost_dollars
                   + R * job.dollars_per_gpu_h
                   + _energy_dollars(job.power_mw_full, away, job.hours_at_full))
        move = _option(
            "move", True, dollars, dur,
            f"migration idles the fleet {job.migration_downtime_h:.2f}h; "
            "requires the full fleet free at the destination — priced only "
            "because it is", job)
    else:
        move = _option("move", False, 0, 0,
                       "the away hall cannot host the whole job (the "
                       "destination-fleet precondition of ../calc)", job)
    if move["feasible"] and job.deadline_h and move["duration_h"] > job.deadline_h:
        move = _option("move", False, 0, 0, "migration misses the deadline", job)

    # ---- BUY: the reference floor, not an option — the scarce hall by
    # definition cannot sell it.
    buy_dollars = (R * job.dollars_per_gpu_h
                   + _energy_dollars(job.power_mw_full, home, job.hours_at_full))
    buy = _option("buy-in-hall (reference)", True, buy_dollars,
                  job.hours_at_full,
                  "what the useful GPU-hour would cost if the scarce hall "
                  "could sell you one — the floor every option is judged "
                  "against, not a verdict", job)

    options = [stay, stitch, move]
    feasible = [o for o in options if o["feasible"]]
    decision = (min(feasible, key=lambda o: o["dollars"])["option"]
                if feasible else "escalate")
    return {
        "decision": decision,
        "stitch_tax": round(tax, 4),
        "cut": cut.name,
        "capacity_deficit_gpus": deficit,
        "options": options + [buy],
    }
