.PHONY: install lint format test test-fast docker-up docker-down generate-data train evaluate dashboard clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	mypy src

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

generate-data:
	python scripts/generate_data.py --samples 50000

train:
	python scripts/train.py --config configs/model_config.yaml

evaluate:
	python scripts/evaluate.py

dashboard:
	streamlit run src/visualization/dashboard.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
