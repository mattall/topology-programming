install:
	pip install .

save:
ifndef name
	$(eval name := $(shell date "+%d-%m-%y-%H-%M"))
endif
	mkdir -p archive/$(name)/results
	mkdir -p archive/$(name)/reports
	cp -r data/results/* archive/$(name)/results
	cp -r data/reports/* archive/$(name)/reports

clean:
	rm -rf data/results/* data/reports/* logs/*.txt src/onset/__pycache__ src/onset/utilities/__pycache__
	rm gurobi.log

.PHONY: clean save install
