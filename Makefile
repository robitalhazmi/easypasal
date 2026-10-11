.PHONY: help setup lint format test check clean ingest index serve eval bench

help:
	@echo "EasyPasal Development Commands:"
	@echo "  make setup   - Install dependencies and pre-commit hooks via uv"
	@echo "  make lint    - Run ruff linter, formatting checks, and mypy type checks"
	@echo "  make format  - Auto-format code and fix lint issues where possible"
	@echo "  make test    - Run unit and integration tests with pytest"
	@echo "  make check   - Run all linting and test suites"
	@echo "  make ingest  - Download and parse regulatory documents from corpus manifest (Phase 1)"
	@echo "  make index   - Build Qdrant vector store and BM25 sparse index (Phase 1)"
	@echo "  make serve   - Start the FastAPI service and Gradio UI locally (Phase 1)"
	@echo "  make eval    - Run the evaluation harness on the gold evaluation set (Phase 3)"
	@echo "  make bench   - Run CPU inference latency benchmark (Phase 5)"
	@echo "  make clean   - Remove cache directories and temporary build artifacts"

setup:
	@which uv >/dev/null 2>&1 || (echo "uv not found. Please install uv (e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh' or 'pip install uv')" && exit 1)
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest tests/ -v --cov=src/easypasal --cov-report=term-missing

check: lint test

clean:
	rm -rf .venv/ .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage* dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

ingest:
	uv run python -m easypasal.ingest.download --update-hashes

parse:
	uv run python -m easypasal.ingest.parser

chunk:
	uv run python -m easypasal.ingest.chunker --strategy structure_aware
	uv run python -m easypasal.ingest.chunker --strategy fixed_size

index:
	@echo "Indexing pipeline not yet implemented (scheduled for Phase 1 / T1.4)."
	@exit 1

serve:
	@echo "Serving pipeline not yet implemented (scheduled for Phase 1 / T1.8)."
	@exit 1

eval:
	@echo "Evaluation runner not yet implemented (scheduled for Phase 3 / T3.2)."
	@exit 1

bench:
	@echo "Benchmark runner not yet implemented (scheduled for Phase 5 / T5.2)."
	@exit 1
