"""Causal scarcity policy with explicit backlog and catch-up headroom.

The existing full-horizon price sorter is an offline bound. This controller
sees the current price only. Deferral preserves total MWh; catch-up needs a
declared extra flexible-load envelope, as in the original experiment.
"""
import math
import numpy as np
from .fleet_sim import Fleet,price_series,simulate


def replay(prices, *, fleet_mw=100., flexible_fraction=.3, threshold=200., budget_fraction=.005, cheap_threshold=50.):
    prices=list(prices)
    if not prices or any(not math.isfinite(p) or p<0 for p in prices): raise ValueError('finite nonnegative prices required')
    if not math.isfinite(fleet_mw) or fleet_mw<=0 or not 0<flexible_fraction<=1 or not 0<=budget_fraction<1: raise ValueError('invalid capacity/budget')
    if not all(math.isfinite(v) and v>=0 for v in (threshold,cheap_threshold)): raise ValueError('invalid price thresholds')
    flexible=fleet_mw*flexible_fraction; backlog=0.;used=0;cost=0.;served=0.;actions=[]
    budget=int(len(prices)*budget_fraction)
    for i,price in enumerate(prices):
        remaining=len(prices)-i-1
        # Keep enough remaining slots to clear the debt even if every future
        # price is expensive. The horizon is known, future prices are not.
        can_defer=remaining>math.ceil(backlog/flexible)
        curtail=price>threshold and used<budget and can_defer
        extra=0.
        if curtail:
            draw=fleet_mw-flexible;backlog+=flexible;used+=1
        else:
            if price<=cheap_threshold or remaining<math.ceil(backlog/flexible): extra=min(backlog,flexible)
            draw=fleet_mw+extra;backlog-=extra
        served+=draw;cost+=draw*price
        actions.append(dict(hour=i,price=price,draw_mw=draw,backlog_mwh=backlog,curtailed=curtail))
    target=fleet_mw*len(prices)
    assert math.isclose(served+backlog,target,abs_tol=1e-6)
    blind=fleet_mw*sum(prices)
    return dict(evidence_class='simulated',input_class='scenario',cost_dollars=cost,
                blind_cost_dollars=blind,saving_dollars=blind-cost if backlog<1e-6 else None,
                served_mwh=served,unserved_mwh=backlog,extra_headroom_mw=flexible,actions=actions,
                limitations=['Synthetic prices unless caller supplies external observations.',
                             'Extra catch-up power/compute headroom is an explicit assumption.',
                             'Constant conversion between deferred energy and useful work; no migration or SLA model.'])


def experiment():
    rows=[]
    for cap in (2000,5000):
        for seed in range(24):
            cfg=Fleet(horizon_h=8760,seed=seed,spike_max_mwh=cap)
            prices=price_series(cfg,np.random.default_rng(seed))
            result=replay(prices)
            result.pop('actions')
            rows.append(dict(seed=seed,spike_ceiling=cap,causal=result,offline_oracle=simulate(cfg)))
    return dict(runs=rows,evidence_class='simulated',baseline='Identical price sequence, flexible fraction and curtailment budget.',
                qualification='Offline comparator can rank future prices and shift work backwards in time; it is not a deployable policy.')


if __name__=='__main__':
    import json
    from pathlib import Path
    report=experiment();out=Path('results/causal-scarcity.json');out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(report["runs"])} causal/oracle comparisons; outstanding work disclosed')
