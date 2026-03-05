all: prep fit

prep:
	python3 main.py --scenario=prep_data \
			--n_prep=10 \
			--n_symbol=2048

fit:
	python main.py --scenario=fit

sample:
	python main.py --scenario=sample
