# Downtime-forecast MLOps pipeline — local runner.
PY ?= python

.PHONY: help install data pipeline train drift test serve clean docker-build

help:
	@echo "make install   - install core requirements"
	@echo "make data      - prepare data (validate, features, daily series)"
	@echo "make pipeline  - full train -> evaluate -> register -> report"
	@echo "make drift     - run drift monitoring and write reports/drift.md"
	@echo "make test      - run the fast pytest suite (no training)"
	@echo "make serve     - launch the FastAPI service on :8000"
	@echo "make docker-build - build the API image"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -c "from src import data; data.prepare()"

pipeline:
	$(PY) -m src.pipeline

drift:
	$(PY) -m src.drift

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t downtime-forecast-api:latest .

clean:
	rm -rf runs models/champion reports/figures/*.png __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
