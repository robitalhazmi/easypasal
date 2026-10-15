.PHONY: help setup lint format test check clean ingest index serve eval compare smoke-eval exp-e1 exp-e2 exp-e3 exp-e4 exp-e5 exp-e6 bench

help:
	@echo "EasyPasal Development Commands:"
	@echo "  make setup      - Install dependencies and pre-commit hooks via uv"
	@echo "  make lint       - Run ruff linter, formatting checks, and mypy type checks"
	@echo "  make format     - Auto-format code and fix lint issues where possible"
	@echo "  make test       - Run unit and integration tests with pytest"
	@echo "  make check      - Run all linting and test suites"
	@echo "  make ingest     - Download and parse regulatory documents from corpus manifest (Phase 1)"
	@echo "  make index      - Build Qdrant vector store and BM25 sparse index (Phase 1)"
	@echo "  make serve      - Start the FastAPI service and Gradio UI locally (Phase 1)"
	@echo "  make eval       - Run the evaluation harness on the gold evaluation set (Phase 3)"
	@echo "  make compare    - Compare baseline and candidate runs with bootstrap CIs (Phase 3)"
	@echo "  make smoke-eval - Run fast CI smoke evaluation gate against stored baseline (Phase 3)"
	@echo "  make exp-e1     - Run Experiment E1: Chunking Strategy comparison on dev & test (Phase 4)"
	@echo "  make exp-e2     - Run Experiment E2: Embedding Models comparison on dev & test (Phase 4)"
	@echo "  make exp-e3     - Run Experiment E3: Retrieval Pipeline comparison on dev & test (Phase 4)"
	@echo "  make exp-e4     - Run Experiment E4: Generator Models comparison on CPU (Phase 4)"
	@echo "  make exp-e5     - Run Experiment E5: DSPy + GEPA prompt optimization (Phase 5)"
	@echo "  make exp-e6     - Run Experiment E6: vLLM serving benchmark vs CPU baseline (Phase 5)"
	@echo "  make bench      - Run serving throughput and latency benchmark (Phase 5)"
	@echo "  make clean      - Remove cache directories and temporary build artifacts"

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

BASELINE ?= data/eval_outputs/eval_baseline.json
CANDIDATE ?= data/eval_outputs/e1_variant_c_structure_enriched_dev.json

compare:
	uv run python -m eval.compare_variants --baseline $(BASELINE) --candidate $(CANDIDATE)

smoke-eval:
	uv run python -m eval.smoke_eval

exp-e1:
	uv run python -m experiments.e1_chunking.run_experiment --split both

exp-e2:
	uv run python -m experiments.e2_embeddings.run_experiment --split both --mode dense

exp-e3:
	uv run python -m experiments.e3_retrieval.run_experiment --split both

exp-e4:
	uv run python -m experiments.e4_generator.run_experiment --split both

exp-e5:
	uv run python -m experiments.e5_gepa.run_experiment --provider mock

exp-e6:
	uv run python -m experiments.e6_vllm.run_benchmark --mock

bench:
	uv run python -m experiments.e6_vllm.run_benchmark --mock
