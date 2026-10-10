# Contributing to EasyPasal

Thank you for your interest in contributing to **EasyPasal**!

EasyPasal is an evaluation-first, citation-grounded compliance assistant for Indonesian financial regulation. We welcome contributions that improve code quality, documentation, test coverage, and evaluation methodology.

---

## Code of Conduct

All contributors and participants agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Git

### Getting Started

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/robitalhazmi/easypasal.git
   cd easypasal
   ```

2. **Set up the virtual environment and pre-commit hooks:**
   ```bash
   make setup
   ```

3. **Verify the installation:**
   ```bash
   make check
   ```

---

## Core Guidelines & Principles

1. **Grounded or Silent:**
   - Any feature touching generation must strictly preserve the cite-or-abstain principle (cite exact regulation, *Pasal*, and *Ayat*, or abstain).
2. **Honest Claims & Reproducibility:**
   - Never commit fabricated metrics, synthetic results presented as real, or uncalibrated scores. Use `TBD` where data has not yet been computed.
   - All numbers presented in the documentation must be reproducible with a single documented script.
3. **No Secrets or Large Binary Files:**
   - Never commit API tokens, passwords, `.env` files, or raw regulation PDF binaries. Use `data/corpus_manifest.csv` to record document metadata and download URLs.
4. **Strict Typing & Testing:**
   - All new code must be fully typed (passing `mypy`) and formatted/linted (passing `ruff`).
   - Every new function or module must be accompanied by unit tests in `tests/`.

---

## Development Workflow

### Useful Make Targets

```bash
make setup    # Install dependencies & configure pre-commit hooks
make format   # Automatically format code with ruff
make lint     # Run ruff linter, formatting checks, and mypy type checks
make test     # Run pytest with test coverage
make check    # Run lint + test suites
```

### Git & Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(scope): ...` — New features or pipeline components
- `fix(scope): ...` — Bug fixes
- `docs(scope): ...` — Documentation updates
- `test(scope): ...` — Adding or updating tests
- `chore(scope): ...` — Tooling, dependency, or configuration updates
- `ci: ...` — GitHub Actions or workflow updates

### Submitting a Pull Request

1. Create a descriptive feature branch (`feat/ingest-enhancement`, `fix/bm25-tokenizer`).
2. Implement your changes with corresponding unit tests.
3. Run `make check` locally to ensure all tests and linters pass cleanly.
4. Open a Pull Request against `main`. Ensure CI passes.
