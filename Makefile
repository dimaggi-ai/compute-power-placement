.PHONY: all test figures
all: test
test:                ## move-vs-stay invariants
	cd calc && python3 test_move_vs_stay.py
figures:             ## break-even frontier + energy-share/scarcity
	cd calc && python3 run.py
