"""Causal fleet invariants. Run from the repository root:

    python3 -m fleet.test_causal

An assertion failure exits nonzero, as required by make test and CI.
"""
import math
from fleet.causal import replay


def test_energy_and_headroom_conservation():
    for prices in ([10,500,500,10],[500]*8,[10]*8,[10,10,10,500]):
        r=replay(prices,budget_fraction=.5)
        assert r['unserved_mwh']==0
        assert math.isclose(r['served_mwh'],100*len(prices))
        assert all(70<=a['draw_mw']<=130 for a in r['actions'])


def test_future_prices_cannot_change_past_actions():
    a=replay([10,500,100,20,30,40],budget_fraction=.5)
    b=replay([10,500,100,5000,5000,5000],budget_fraction=.5)
    assert a['actions'][:3]==b['actions'][:3]


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_"))
    for name in tests:
        globals()[name]()
        print(f"ok {name}")
    print(f"all {len(tests)} causal tests passed")
