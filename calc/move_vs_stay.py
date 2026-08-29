#!/usr/bin/env python3
"""Move-vs-stay: when does relocating a training job to cheaper/available power
beat staying put? A transparent decision model priced in GPU-hours and dollars.

The question the grid heat-map poses: idle GPUs sit in one place, cheap or
curtailed power in another. Should you move the job to the power? This model
answers it in the operator's units and lands the honest finding: on a *running*
fleet, electricity is only a few percent of the GPU-hour cost, so chasing the
$/MWh spread almost never pays — a migration idles the whole fleet and requires
an equal idle fleet to already exist at the destination. Relocation wins at the
*scarcity* boundary (avoiding a price spike), not on the average spread. So the
lever is "schedule against scarcity," not "arbitrage the spread" — the direct
continuation of the scheduling repo's power-as-a-schedulable-resource pattern.

All inputs are explicit; every number a reader can push on. Sources for the
calibration values are in ../REFERENCES.md.
"""
from __future__ import annotations
import dataclasses
import math

BYTES_PER_PARAM = 16          # fp32 weights + Adam moments (ZeRO convention)


@dataclasses.dataclass
class Job:
    params_b: float = 405.0            # billions of parameters
    gpus: int = 16_384
    kw_per_gpu: float = 1.4            # burdened power draw per GPU (incl. cooling)
    dollars_per_gpu_h: float = 2.5
    # migration mechanics
    wan_gbps: float = 100.0           # inter-site bandwidth for the state transfer
    reload_h: float = 0.5             # drain + rendezvous + reload at destination
    egress_per_gb: float = 0.02       # $/GB WAN egress
    # electricity
    price_here_mwh: float = 60.0
    price_there_mwh: float = 30.0

    @property
    def state_gb(self) -> float:
        return self.params_b * 1e9 * BYTES_PER_PARAM / 1e9      # = params_b * 16

    @property
    def power_mw(self) -> float:
        return self.gpus * self.kw_per_gpu / 1000.0

    @property
    def migration_downtime_h(self) -> float:
        """Drain+reload plus the WAN transfer of the full training state."""
        transfer_h = (self.state_gb * 8) / self.wan_gbps / 3600.0   # GB*8=Gbit / Gbps / s→h
        return self.reload_h + transfer_h

    @property
    def migration_cost_dollars(self) -> float:
        """The cost of moving once: the whole fleet idles for the downtime, plus
        WAN egress. (The precondition — an idle equal fleet must exist at the
        destination — is a capacity cost, not modelled here; see study.md.)"""
        idle = self.gpus * self.migration_downtime_h * self.dollars_per_gpu_h
        egress = self.state_gb * self.egress_per_gb
        return idle + egress


def energy_saving_dollars(job: Job, remaining_h: float) -> float:
    """Electricity saved by running `remaining_h` at the cheaper site."""
    spread = job.price_here_mwh - job.price_there_mwh
    return job.power_mw * spread * remaining_h


def decide(job: Job, job_length_h: float) -> dict:
    """Move vs stay over a job of `job_length_h`. Moving pays the downtime up
    front, then earns the price spread over the remaining time."""
    d = job.migration_downtime_h
    remaining = max(0.0, job_length_h - d)
    saving = energy_saving_dollars(job, remaining)
    cost = job.migration_cost_dollars
    net = saving - cost
    energy_share = (job.power_mw * job.price_here_mwh) / (job.gpus * job.dollars_per_gpu_h)
    return {
        "decision": "move" if net > 0 else "stay",
        "net_dollars": round(net, 0),
        "energy_saving_dollars": round(saving, 0),
        "migration_cost_dollars": round(cost, 0),
        "migration_downtime_h": round(d, 2),
        "state_gb": round(job.state_gb, 0),
        "energy_share_of_gpu_cost": round(energy_share, 4),
    }


def breakeven_spread_mwh(job: Job, job_length_h: float) -> float:
    """The price spread ($/MWh) at which moving exactly breaks even over a job
    of `job_length_h`. Below this, staying wins."""
    remaining = max(1e-9, job_length_h - job.migration_downtime_h)
    return job.migration_cost_dollars / (job.power_mw * remaining)


def scarcity_avoided_dollars(job: Job, spike_price_mwh: float, spike_hours: float) -> dict:
    """The value of moving OFF a site about to hit a scarcity price for
    `spike_hours`, vs staying and paying the spike. This is where relocation
    actually pays — a single avoided spike dwarfs the migration cost."""
    stay_cost = job.power_mw * spike_price_mwh * spike_hours
    move_cost = job.migration_cost_dollars + job.power_mw * job.price_there_mwh * spike_hours
    return {
        "stay_through_spike_dollars": round(stay_cost, 0),
        "move_and_ride_it_out_dollars": round(move_cost, 0),
        "net_benefit_of_moving": round(stay_cost - move_cost, 0),
    }
