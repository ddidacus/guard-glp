# Makefile for guard-glp
#
# Convenience targets that mirror the CI workflows:
#   - `make check` reproduces .github/workflows/code-checks.yaml
#   - `make test`  reproduces .github/workflows/pytest.yaml (minus the Codecov upload, which is CI-only)
#
# All targets run through `uv`, matching CI exactly.

.PHONY: help install lint format format-check typecheck check test precommit all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment (uv sync)
	uv sync

lint: ## Run Ruff lint checks
	uv run ruff check .

format: ## Auto-format the codebase with Ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

typecheck: ## Run pyright type checks
	uv run pyright

check: lint format-check typecheck ## Run all code-quality checks (== code-checks.yaml)

test: ## Run pytest with coverage (== pytest.yaml)
	uv run pytest --cov=glp --cov-report=term-missing --cov-report=xml

precommit: ## Run all pre-commit hooks against all files
	uv run pre-commit run --all-files

all: check test ## Run all checks and tests
