.PHONY: all test figures
all: test
test:                ## move-vs-stay + fleet scarcity-scheduling invariants
	cd calc && python3 test_move_vs_stay.py
	cd fleet && python3 test_fleet.py
figures:             ## break-even, energy-share/scarcity, fleet scarcity
	cd calc && python3 run.py
	cd fleet && python3 run.py
