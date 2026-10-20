.DEFAULT_GOAL := help
.PHONY: help setup model data embeddings embeddings-videos trials duplicates scores eer det test lint format eval clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev tools (needs uv)
	uv sync

MODEL_REV := a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b
MODEL_SHA := d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0
MODEL := artifacts/models/voxceleb_ECAPA512_LM.onnx

model: $(MODEL) ## Download the pinned WeSpeaker ECAPA-TDNN and verify its checksum

$(MODEL):
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

data: $(VOX_DIR)/.unpacked ## Download VoxCeleb1 (~34 GB), verify checksums, unpack to data/voxceleb1/wav

$(VOX_DIR)/.unpacked:
	mkdir -p $(VOX_DIR)/archives
	for f in $(VOX_FILES); do \
		curl -fL -C - --retry 20 --retry-all-errors --retry-delay 10 -o $(VOX_DIR)/archives/$$f $(VOX_URL)/$$f || exit 1; \
	done
	cd $(VOX_DIR)/archives && shasum -a 256 -c $(CURDIR)/voxceleb1.sha256
	cat $(VOX_DIR)/archives/vox1_dev_wav_parta? > $(VOX_DIR)/vox1_dev_wav.zip
	unzip -q -o $(VOX_DIR)/vox1_dev_wav.zip -d $(VOX_DIR)
	unzip -q -o $(VOX_DIR)/archives/vox1_test_wav.zip -d $(VOX_DIR)
	rm $(VOX_DIR)/vox1_dev_wav.zip
	touch $@

# Each long step writes a .done file when it finishes, so make skips it next time.
# An interrupted run writes none, and the rerun resumes where it stopped.
EMB_DIR := artifacts/embeddings/voxceleb1
META := $(VOX_DIR)/archives/vox1_meta.csv

embeddings: $(EMB_DIR)/.done ## Embed every VoxCeleb1 clip (~3.5 h on the home PC); a rerun resumes

$(EMB_DIR)/.done: | $(MODEL) $(VOX_DIR)/.unpacked
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(META) $(EMB_DIR)
	touch $@

embeddings-videos: $(EMB_DIR)-videos/.done ## One embedding per video, clips joined (~3.2 h on the home PC)

$(EMB_DIR)-videos/.done: | $(MODEL) $(VOX_DIR)/.unpacked
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(META) $(EMB_DIR)-videos --join-videos
	touch $@

TRIALS := artifacts/trials/voxceleb1.tsv

DUPLICATES := artifacts/trials/voxceleb1-shared-videos.tsv

duplicates: $(DUPLICATES) ## Find video pairs of one speaker that hold the same audio (~30 min)

$(DUPLICATES): | $(EMB_DIR)/.done
	uv run python -m svs.duplicates $(VOX_DIR)/wav $(EMB_DIR) $@

trials: | $(EMB_DIR)/.done $(EMB_DIR)-videos/.done $(DUPLICATES) ## Build the trial list; shared-audio videos are one session
	uv run python -m svs.trial_list $(EMB_DIR) $(EMB_DIR)-videos $(META) $(DUPLICATES) $(TRIALS)

SCORES := artifacts/scores/voxceleb1-cosine.tsv

scores: ## Cosine score of every trial in the VoxCeleb1 trial list
	uv run python -m svs.score $(TRIALS) $(EMB_DIR) $(EMB_DIR)-videos $(SCORES)

eer: ## EER per duration bucket from the score file
	uv run python -m svs.metrics $(SCORES)

DET := artifacts/plots/voxceleb1-det.png

det: ## DET plot, one line per duration bucket
	uv run python -m svs.det $(SCORES) $(DET)

test: ## Run the test suite
	uv run pytest

lint: ## Check style and common bugs
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix style
	uv run ruff check --fix .
	uv run ruff format .

# eval.sha256 holds the files of the run in LOG.md. Another machine may round the
# embeddings differently, so a mismatch is reported, not treated as a failure.
eval: ## Reproduce the EER and DET; a clean clone first runs model, data, embeddings (hours)
	$(MAKE) trials
	$(MAKE) scores
	$(MAKE) eer
	$(MAKE) det
	@shasum -a 256 -c eval.sha256 \
		|| echo "eval: files differ from the run recorded in LOG.md; the numbers above are this machine's" >&2

clean: ## Remove caches (keeps data/ and artifacts/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
