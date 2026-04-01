.PHONY: install native-build native-develop test lint format typecheck bench clean

install:
	pip install -e ".[dev]"

native-build:
	cargo build --manifest-path native/Cargo.toml --release

native-develop:
	maturin develop --manifest-path native/Cargo.toml

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/quench/

bench:
	pytest tests/ -v -k "benchmark" --benchmark-only

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache dist *.egg-info
