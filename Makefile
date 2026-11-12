# Compute-Power Placement — one-command checks.
.PHONY: all test validation figures
all: test
test: validation     ## move-vs-stay + fleet + causal + stay/stitch/move + the registry
	cd calc && python3 test_move_vs_stay.py
	cd fleet && python3 test_fleet.py
	python3 -m fleet.test_causal
	cd span && python3 test_stay_stitch_move.py
validation:          ## the registry: both models vs the public record
	python3 test_validation.py
	python3 validation.py
figures:             ## break-even, energy-share/scarcity, fleet, stay/stitch/move
	cd calc && python3 run.py
	cd fleet && python3 run.py
	cd span && python3 run.py
