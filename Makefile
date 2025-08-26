
.PHONY: setup data features baseline lstm test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python src/data/download_data.py --tickers AAPL MSFT --period 5y --interval 1d

features:
	python src/features/make_features.py --input data/raw --output data/processed

baseline:
	python src/models/baseline.py --data data/processed --ticker AAPL

lstm:
	python src/models/train_lstm.py --data data/processed --ticker AAPL --epochs 3

test:
	pytest -q
