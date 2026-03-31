.PHONY: install test lint typecheck bench clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

typecheck:
	mypy src/quench/

bench:
	pytest tests/ -v -k "benchmark" --benchmark-only

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache dist *.egg-info
