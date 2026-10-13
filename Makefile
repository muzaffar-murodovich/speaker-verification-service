.DEFAULT_GOAL := help
.PHONY: help setup model data embeddings embeddings-videos test lint format eval clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev tools (needs uv)
	uv sync

MODEL_REV := a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b
MODEL_SHA := d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0
MODEL := artifacts/models/voxceleb_ECAPA512_LM.onnx

model: ## Download the pinned WeSpeaker ECAPA-TDNN and verify its checksum
	mkdir -p $(dir $(MODEL))
	curl -fL -o $(MODEL) https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/$(MODEL_REV)/$(notdir $(MODEL))
	echo "$(MODEL_SHA)  $(MODEL)" | shasum -a 256 -c

# VoxCeleb1 from a Hugging Face mirror at a pinned revision. voxceleb1.sha256 holds the
# official Oxford checksums (the same values torchaudio's VoxCeleb1 loader uses), so a
# passing check proves the mirror files are the original archives.
VOX_REV := b8a8825c87582831cc217b6ff84a36558ced84d6
VOX_URL := https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/$(VOX_REV)/vox1
VOX_DIR := data/voxceleb1
VOX_FILES := $(shell awk '{print $$2}' voxceleb1.sha256)

data: ## Download VoxCeleb1 (~34 GB), verify checksums, unpack to data/voxceleb1/wav
	mkdir -p $(VOX_DIR)/archives
	for f in $(VOX_FILES); do \
		curl -fL -C - --retry 20 --retry-all-errors --retry-delay 10 -o $(VOX_DIR)/archives/$$f $(VOX_URL)/$$f || exit 1; \
	done
	cd $(VOX_DIR)/archives && shasum -a 256 -c $(CURDIR)/voxceleb1.sha256
	test -f $(VOX_DIR)/.unpacked || ( \
		cat $(VOX_DIR)/archives/vox1_dev_wav_parta? > $(VOX_DIR)/vox1_dev_wav.zip && \
		unzip -q -o $(VOX_DIR)/vox1_dev_wav.zip -d $(VOX_DIR) && \
		unzip -q -o $(VOX_DIR)/archives/vox1_test_wav.zip -d $(VOX_DIR) && \
		rm $(VOX_DIR)/vox1_dev_wav.zip && touch $(VOX_DIR)/.unpacked )

EMB_DIR := artifacts/embeddings/voxceleb1

embeddings: ## Embed every VoxCeleb1 clip (~1.6 h on Mac CPU); a rerun resumes
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(VOX_DIR)/archives/vox1_meta.csv $(EMB_DIR)

embeddings-videos: ## One embedding per VoxCeleb1 video, clips joined (~3.5 h); a rerun resumes
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(VOX_DIR)/archives/vox1_meta.csv $(EMB_DIR)-videos --join-videos

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
