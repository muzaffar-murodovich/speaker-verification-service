.DEFAULT_GOAL := help
.PHONY: help setup test lint format eval clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev tools (needs uv)
	uv sync

test: ## Run the test suite
	uv run pytest

lint: ## Check style and common bugs
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix style
	uv run ruff check --fix .
	uv run ruff format .

eval: ## Reproduce the headline numbers (not built yet; first EER lands on D5)
	@echo "eval: no evaluation pipeline yet; see LOG.md for progress" >&2
	@exit 1

clean: ## Remove caches (keeps data/ and artifacts/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
