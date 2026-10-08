.DEFAULT_GOAL := help
.PHONY: help setup model test lint format eval clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev tools (needs uv)
	uv sync

MODEL_REV := a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b
MODEL_SHA := d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0
MODEL := artifacts/models/voxceleb_ECAPA512_LM.onnx

model: ## Download the pinned WeSpeaker ECAPA-TDNN and verify its checksum
	mkdir -p $(dir $(MODEL))
	curl -fL -o $(MODEL) https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/$(MODEL_REV)/$(notdir $(MODEL))
	echo "$(MODEL_SHA)  $(MODEL)" | shasum -a 256 -c

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
