# Contributing to Quench

## Development Setup

```bash
git clone https://github.com/itsbryanman/quench.git
cd quench
pip install -e ".[dev]"
```

## Before Submitting a PR

Run the full check suite:

```bash
make test       # all tests pass
make lint       # no ruff violations
make typecheck  # mypy strict clean
```

## Code Style

- Python 3.10+ with modern syntax (type unions via `|`, match statements where appropriate)
- Complete type annotations on all public functions
- Line length limit: 100 characters
- Use `ruff` for formatting and linting

## Testing

- Every new feature or bugfix needs tests
- Entropy coder changes require round-trip correctness tests
- Compression quality tests must demonstrate improvement over baselines
