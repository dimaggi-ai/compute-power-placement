# Compute↔Power Placement: schedule against scarcity, don't arbitrage the spread

**Idle GPUs sit in one place; cheap or curtailed power sits in another — so should you move the job to the power?** The grid heat map makes it tempting: US ISOs publish 5-minute nodal price maps that span from *negative* prices to $28,000/MWh within months [1], while ~20 TWh of renewables were curtailed in 2024 and 226 GW of data-center load waits years in the interconnection queue [2, 3]. This repository answers the move-vs-stay question in the operator's units — and the answer reframes the problem.

**TL;DR:** on a *running* fleet, electricity is only **~3–4% of the GPU-hour cost** [4], so chasing the $/MWh spread barely moves the total — and a migration idles the whole fleet and needs an idle fleet already waiting at the destination. Moving on price alone pays only for **long jobs against a sustained spread**. But relocation pays **decisively at the scarcity boundary**: dodging one 12-hour $5,000/MWh spike is worth **~$1.3M** against a ~$27k migration. The lever is **"schedule against scarcity," not "arbitrage the spread"** — the continuation of the [scheduling repo's](https://github.com/dimaggi-ai/scheduler-vs-more-gpus) power-as-a-schedulable-resource pattern, from modulating draw in place to moving work across the map.

*Fourth quantitative pillar of the DIMAGGI series on turning GPU capital into usable compute. Full analysis in [docs/study.md](docs/study.md); all claims trace to [REFERENCES.md](REFERENCES.md).*

---

## The reframe, in one figure

![Energy share and scarcity](figures/energy_share_and_scarcity.png)

Energy is a rounding error on the GPU-hour cost normally (~3%), so the average price spread cannot move the decision — but a scarcity spike is ~2× the fleet's entire GPU cost, and *that* is what moving should avoid.

## The break-even frontier

![Break-even frontier](figures/breakeven_frontier.png)

For a 405B job (~6.5 TB state) on 16,384 GPUs, one migration costs ~$27k (mostly ~0.6 h of fleet idle). Earning that back on a $30/MWh spread needs the job to keep running for days; typical *average* spreads ($10–40/MWh) sit right in the marginal band, so most jobs should stay. The [`move-vs-stay` model](calc/move_vs_stay.py) computes the decision, the break-even spread, and the scarcity-avoidance benefit for any job.

A companion [`fleet`](fleet/) model extends this from one decision to a fleet policy, and the result is a *concentration*, not a spread: under an ERCOT-scale $5k/MWh spike cap, a scarcity-aware scheduler that curtails the flexible load in only the ~0.4% worst hours saves ~5% of the fleet's energy bill (a 24-seed mean; **3.4% in the pictured month**, and the percentage roughly thirds if the spike cap halves). What's robust across calibrations is *where* the saving lives: even a policy free to curtail **every** scarcity hour still draws **~72% of its avoided cost from the single top-1% of hours**, and the saving vanishes with the spikes turned off. The portable claim is "schedule against scarcity, not the spread," now at fleet scale.

## The hard boundary: what can actually move

Inference geo-routes freely (890+ GW of wind within 50 ms RTT of Azure [7]); **synchronous frontier training does not** (43 ms coast-to-coast RTT kills it [8]) — the escape route is async low-communication training (DiLoCo-class), proven at 10B but not frontier scale [9]. So the highest-value workload is the least movable, which is exactly why the honest lever is scarcity-aware scheduling and siting — the production instances are power-seeking operators (Crusoe, Lancium, Soluna) and grid-interactive designs (Emerald AI, the 96 MW Aurora facility) [12, 13].

## Reproduce

```
pip install -r calc/requirements.txt
make test        # move-vs-stay + fleet scarcity-scheduling invariants
make figures     # break-even, energy-share/scarcity, and the fleet figure
```

Python 3.11+, `numpy`, `matplotlib`. Every calibration value is an input; see [docs/study.md](docs/study.md#what-a-skeptic-should-attack) for what to push on (the destination-fleet precondition, data gravity, the $/GPU-hour assumption).

## Series — turning GPU capital into usable compute

- **GPU Cluster Networking** ([network-vs-more-gpus](https://github.com/dimaggi-ai/network-vs-more-gpus))
- **GPU Cluster Scheduling** ([scheduler-vs-more-gpus](https://github.com/dimaggi-ai/scheduler-vs-more-gpus)) — power as a schedulable resource
- **Compute↔Power Placement** (this work) — move work across the grid map, priced
- **Chaos Fidelity Standard** ([ai-cluster-chaos-fidelity](https://github.com/dimaggi-ai/ai-cluster-chaos-fidelity)) · **Reliability Economics** ([reliability-economics](https://github.com/dimaggi-ai/reliability-economics))

---

*Margaret (Maggie) Nanyonga — Founder & Principal Architect, [DIMAGGI AI](https://dimaggi.ai). Governed AI infrastructure: the control, reliability, and audit layer for autonomous systems operating production networks and compute.*
