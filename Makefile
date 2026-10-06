PYTHON ?= python

.PHONY: install dev test lint format backtest train forecast figures app notebook docker clean

install:  ## Install the package and runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt && $(PYTHON) -m pip install -e .

dev:  ## Install dev tooling
	$(PYTHON) -m pip install -r requirements-dev.txt && $(PYTHON) -m pip install -e .

test:  ## Run the test-suite
	$(PYTHON) -m pytest

lint:  ## Static checks
	$(PYTHON) -m ruff check src app tests

format:  ## Auto-format
	$(PYTHON) -m ruff format src app tests && $(PYTHON) -m ruff check --fix src app tests

backtest:  ## Walk-forward backtest of every model
	$(PYTHON) -m tesla_forecast.cli backtest

train:  ## Backtest + fit final models + save artifacts/pipeline.joblib
	$(PYTHON) -m tesla_forecast.cli train

forecast:  ## Print the forecast using the saved pipeline
	$(PYTHON) -m tesla_forecast.cli forecast

figures:  ## Save report PNGs to reports/figures
	$(PYTHON) -m tesla_forecast.cli figures

app:  ## Launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

notebook:  ## Execute the analysis notebook end-to-end
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/01_tesla_advanced_forecasting.ipynb

docker:  ## Build the container image
	docker build -t tesla-forecast .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info artifacts/*.joblib
