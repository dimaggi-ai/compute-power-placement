# Compute↔power placement: schedule against scarcity, don't arbitrage the spread

Reference numbers point to [REFERENCES.md](../REFERENCES.md). Numbers reproduce with `python3 calc/run.py`.

## The heat map is real

Overlay the compute fleet on a grid heat map and the mismatch is literal: US ISOs publish 5-minute nodal price maps (ERCOT's live LMP contour; every ISO via the EIA portal) [1], and the dispersion is extreme — CAISO spent ~13% of 2024 hours at *negative* prices while one ERCOT node averaged **$28,187/MWh** for an hour in Feb 2025 [1]. Both sides are stranded: ~**20 TWh** of US wind and solar were curtailed in 2024 (≈11% of all US data-center electricity) [2], while **226 GW** of mostly-data-center load waits ~5 years in ERCOT's interconnection queue [3]. So the founder's picture — idle GPUs here, cheap or curtailed power there — is a measured, daily reality, not a metaphor.

The tempting conclusion is "move the compute to the power." This study asks, in the operator's units, when that actually pays — and the answer reframes the whole problem.

## The reframe: energy is a rounding error on a running fleet

![Energy share and scarcity](../figures/energy_share_and_scarcity.png)

At $2–3/GPU-hour, electricity is only **~3–4% of the GPU-hour cost** (a burdened H100 draws ~1.4 kW; at $60/MWh that is ~$0.08/h against a ~$2.5/h GPU-hour) [4]. So chasing the $/MWh *spread* on a *running* fleet moves total cost by low single digits — and moving is not free: a migration idles the **whole** fleet during drain + WAN transfer + reload, and it requires an **equal idle fleet to already exist at the destination** (a capacity cost that dwarfs any energy saving). The [`move-vs-stay` model](../calc/move_vs_stay.py) makes this precise.

## When does moving on price alone pay?

![Break-even frontier](../figures/breakeven_frontier.png)

For a 405B-parameter job (≈6.5 TB of training state [5]) at $2.5/GPU-hour on 16,384 GPUs, one migration costs ~$27,000 (mostly the ~0.6 h of fleet idle; WAN egress is ~$130). Earning that back on a $30/MWh spread requires the job to *keep running* long enough:

| Remaining job length | Break-even spread needed | Decision at a $30/MWh spread |
|---|---|---|
| 24 h | ~$49/MWh | **stay** |
| 72 h | ~$16/MWh | move |
| 168 h (1 week) | ~$7/MWh | move |
| 720 h (1 month) | ~$2/MWh | move |

So pure spread-arbitrage pays only for **long-running** jobs against a **sustained** advantage — and typical *average* nodal spreads are $10–40/MWh, right in the marginal band. For any short or medium job, staying wins. Relocating a job to chase the mean price is, for most jobs, a loss.

## Where relocation decisively pays: scarcity

The picture flips at the **scarcity boundary**. Staying through a single 12-hour, $5,000/MWh price spike costs the fleet **~$1.38M** in electricity; moving off it (migration + running elsewhere) costs ~$35k — a **~$1.3M net benefit from one avoided spike**, versus the ~$27k it costs to move. One avoided scarcity-day is worth more than a year of average-price optimization. The same logic covers the larger prizes the model does not price directly: **queue access** (226 GW stuck; a Duke study finds 76 GW of new load fits the existing grid if loads curtail just 0.25% of uptime [6]) and PJM capacity at a record $329/MW-day [3].

**The thesis, therefore:** the lever is not "arbitrage the $/MWh spread" — energy is too small a share for that to matter on a running fleet. It is **"schedule against scarcity"**: modulate and, where the workload and fabric allow, relocate to *avoid the spikes and reach the stranded power/queue capacity*. The geographical-load-balancing literature established both the price-routing idea and its limit — that naive price-chasing can *raise* total energy use — a generation ago [14]; the AI-era contribution is pricing it in GPU-hours against real nodal dispersion. This is the direct continuation of the [scheduling repo's](https://github.com/dimaggi-ai/scheduler-vs-more-gpus) Pattern 6 (power as a schedulable resource) [15] — from modulating draw in place to moving work across the map.

## The hard boundary: what can actually move

The finding above assumes the job *can* relocate. The physics constrains which can:

- **Inference geo-routes freely** — it is stateless-per-request and latency-tolerant across a region; "890+ GW of wind capacity lies within 50 ms RTT of Azure data centers" [7]. This is the most movable AI workload.
- **Synchronous frontier training does not** — it needs one low-latency fabric; 43 ms coast-to-coast RTT kills naive synchronous training across sites (Gemini trains across metro-scale datacenters only because "Google's network latencies and bandwidths are sufficient") [8]. The escape route is asynchronous, low-communication training (DiLoCo-class, 400–500× less communication), proven at 10B parameters across continents but **not yet at frontier scale** [9].
- **Batch/checkpoint-portable work moves in between** — Google already shifts moveable compute (media processing) between datacenters on carbon signals [10], and EPRI's DCFlex initiative is demonstrating "geospatial load shifting" (Ashburn↔Chicago) with hyperscaler members [11].

So the highest-value workload (synchronous training) is the least movable, which is exactly why the honest lever is *scarcity-aware scheduling and siting* — power-seeking operators (Crusoe, Lancium as an ERCOT Controllable Load Resource, Soluna on curtailed wind) and grid-interactive designs (Emerald AI cutting 25% for 3 h without SLA violation; the 96 MW Aurora certified facility) are the production instances [12, 13].

## What a skeptic should attack

- **The migration cost omits the destination-fleet precondition.** Moving is cheap in bandwidth ($130 egress) and time (~0.6 h), but an *equal idle fleet must exist at the destination* — the dominant real cost. This model prices the move given that fleet exists; it does not price standing up the second fleet.
- **Deterministic single-move economics.** Real placement is a sequence of decisions against a stochastic price process; this model prices one move against a known spread and one known spike, which is enough to establish the *ordering* (scarcity ≫ spread) but not an operating policy.
- **The 3–4% energy share assumes ~$2.5/GPU-hour.** At much lower effective GPU costs (amortized owned hardware) energy share rises and spread-arbitrage strengthens; the model exposes `dollars_per_gpu_h` for exactly this.
- **Data gravity is unmodelled.** A frontier training corpus is tens of TB and must be pre-replicated; the model assumes the dataset is already at the destination.

## What it is

A transparent decision model, in GPU-hours and dollars, that settles a question the grid heat map makes tempting — and reframes it: on a running fleet, move to escape *scarcity and reach stranded capacity*, not to arbitrage the average spread. The contribution nobody has published is the *ordering* and the break-even frontier in native operator units; the calibration values are all exposed as inputs. Invariants (the energy-share reframe, the break-even monotonicity, the scarcity dominance) are enforced by [tests](../calc/test_move_vs_stay.py).
