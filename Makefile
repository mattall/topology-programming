install:
	pip install .

save:
ifndef name
	$(eval name := $(shell date "+%d-%m-%y-%H-%M"))
endif
	mkdir -p data/archive/$(name)/results
	mkdir -p data/archive/$(name)/reports
	cp -r data/results/* data/archive/$(name)/results
	cp -r data/reports/* data/archive/$(name)/reports

clean:
	rm -rf data/results/* data/reports/* logs/*.txt src/onset/__pycache__ src/onset/utilities/__pycache__ .temp/*
	rm gurobi.log

.PHONY: clean save install
