.PHONY: help setup lint format test check clean ingest index serve eval compare bench

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
	@echo "  make compare - Compare baseline and candidate runs with bootstrap CIs (Phase 3)"
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
	uv run python -m easypasal.index.build --strategy structure_aware

serve:
	uv run python -m easypasal.api.app --host 0.0.0.0 --port 8000

label-helper:
	uv run python -m eval.labeling_helper --sample

split-eval:
	uv run python -m eval.split_eval --input data/gold/gold_dataset.jsonl --output-dir data/gold/ --seed 42

validate-gold:
	uv run python -m eval.validate_gold data/gold/gold_dataset.jsonl data/gold/dev.jsonl data/gold/test.jsonl

agreement:
	uv run python -m eval.compute_self_agreement --evaluate

eval:
	uv run python -m eval.run_eval --dataset data/gold/dev.jsonl

eval-judged:
	uv run python -m eval.run_judged_eval --dataset data/gold/dev.jsonl

calibrate:
	uv run python -m eval.compute_judge_calibration

compare:
	uv run python -m eval.compare_variants --baseline data/eval_outputs/eval_baseline.json --candidate data/eval_outputs/eval_candidate.json

bench:
	@echo "Benchmark runner not yet implemented (scheduled for Phase 5 / T5.2)."
	@exit 1
