.PHONY: all test figures validation
all: test
test: validation     ## move-vs-stay + fleet invariants + the validation registry
	cd calc && python3 test_move_vs_stay.py
	cd fleet && python3 test_fleet.py
validation:          ## the registry: both models vs the public record
	python3 test_validation.py
	python3 validation.py
figures:             ## break-even, energy-share/scarcity, fleet scarcity
	cd calc && python3 run.py
	cd fleet && python3 run.py
