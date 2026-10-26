.DEFAULT_GOAL := help
.PHONY: help setup g729 model data musan cohort degraded-embeddings degraded-scores embeddings embeddings-videos trials duplicates scores asnorm-scores eer eer-asnorm eer-conditions degraded-cohorts degraded-asnorm-scores eer-conditions-asnorm operating-points operating-points-conditions operating-points-table det test lint format eval clean

help: ## Show available targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev tools (needs uv)
	uv sync

G729 := artifacts/bin/g729

g729: $(G729) ## Build the G.729 round-trip tool (needs libbcg729: brew install libbcg729 / apt install libbcg729-dev)

$(G729): tools/g729.c
	mkdir -p $(dir $(G729))
	cc -O2 -Wall -Wextra -o $(G729) tools/g729.c $$(pkg-config --cflags --libs libbcg729)

MODEL_REV := a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b
MODEL_SHA := d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0
MODEL := artifacts/models/voxceleb_ECAPA512_LM.onnx

model: $(MODEL) ## Download the pinned WeSpeaker ECAPA-TDNN and verify its checksum

$(MODEL):
	mkdir -p $(dir $(MODEL))
	curl -fL -o $(MODEL) https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/$(MODEL_REV)/$(notdir $(MODEL))
	echo "$(MODEL_SHA)  $(MODEL)" | shasum -a 256 -c

# The mirror is gated (since about 2026-10-21): accept its terms on Hugging Face, then
# `hf auth login`, or set HF_TOKEN.
HF_AUTH = Authorization: Bearer $${HF_TOKEN:-$$(cat $$HOME/.cache/huggingface/token)}

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
		curl -fL -C - --retry 20 --retry-all-errors --retry-delay 10 -H "$(HF_AUTH)" -o $(VOX_DIR)/archives/$$f $(VOX_URL)/$$f || exit 1; \
	done
	cd $(VOX_DIR)/archives && shasum -a 256 -c $(CURDIR)/voxceleb1.sha256
	cat $(VOX_DIR)/archives/vox1_dev_wav_parta? > $(VOX_DIR)/vox1_dev_wav.zip
	unzip -q -o $(VOX_DIR)/vox1_dev_wav.zip -d $(VOX_DIR)
	unzip -q -o $(VOX_DIR)/archives/vox1_test_wav.zip -d $(VOX_DIR)
	rm $(VOX_DIR)/vox1_dev_wav.zip
	touch $@

# MUSAN noise for the degradation harness (handbook §6.1 lists MUSAN as public data).
# CC BY 4.0: Snyder, Chen, Povey, "MUSAN: A Music, Speech, and Noise Corpus", 2015.
# The MD5 is OpenSLR's own checksum.md5; only the noise part is unpacked.
MUSAN_URL := https://www.openslr.org/resources/17/musan.tar.gz
MUSAN_MD5 := 0c472d4fc0c5141eca47ad1ffeb2a7df
MUSAN_DIR := data/musan

musan: $(MUSAN_DIR)/.unpacked ## Download MUSAN (~11 GB), verify its MD5, unpack noise/ to data/musan

$(MUSAN_DIR)/.unpacked:
	mkdir -p $(MUSAN_DIR)
	curl -fL -C - --retry 20 --retry-all-errors --retry-delay 10 -o $(MUSAN_DIR)/musan.tar.gz $(MUSAN_URL)
	test "$$(openssl md5 < $(MUSAN_DIR)/musan.tar.gz | awk '{print $$NF}')" = "$(MUSAN_MD5)"
	tar -xzf $(MUSAN_DIR)/musan.tar.gz -C data musan/noise
	touch $@

# D8 impostor cohort: 5,000 VoxCeleb2 dev speakers, one clip each, read from the split
# archive by HTTP range requests (~20 min). The build refuses a speaker whose name is in
# VoxCeleb1. cohorts/ holds the committed list; a build that differs from it is a new
# cohort version, so make fails instead of overwriting it.
VOX2_DIR := data/voxceleb2
VOX2_META := $(VOX2_DIR)/vox2_meta.csv
VOX2_META_SHA := 9a4f437fcf78b8030e597941703ab82a4d1af314f6eba3952afafe8a3664bd1b
VGG_META := $(VOX2_DIR)/identity_meta.csv
VGG_META_URL := https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/5a803a046e9532232c381f3419b13aab5b565189/meta/identity_meta.csv
VGG_META_SHA := 3e0caaeb1cc30bf784edbfdf77e988cfedb56eaf3d2ba87d7724e4a8f7ff737f
COHORT := artifacts/cohort/voxceleb2-dev-v1

$(VOX2_META):
	mkdir -p $(VOX2_DIR)
	curl -fL -H "$(HF_AUTH)" -o $@ https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/$(VOX_REV)/vox2/vox2_meta.csv
	echo "$(VOX2_META_SHA)  $@" | shasum -a 256 -c

$(VGG_META):
	mkdir -p $(VOX2_DIR)
	curl -fL -H "$(HF_AUTH)" -o $@ $(VGG_META_URL)
	echo "$(VGG_META_SHA)  $@" | shasum -a 256 -c

cohort: $(COHORT).npz ## Build the AS-Norm cohort from VoxCeleb2 dev and check it equals cohorts/

$(COHORT).npz: | $(MODEL) $(VOX_DIR)/.unpacked $(VOX2_META) $(VGG_META)
	uv run python -m svs.cohort $(META) $(VOX2_META) $(VGG_META) $(VOX2_DIR)/wav $(COHORT).tsv $@
	cmp $(COHORT).tsv cohorts/$(notdir $(COHORT)).tsv || { rm $@; exit 1; }

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
	uv run python -m svs.score $(TRIALS) $(EMB_DIR) $(EMB_DIR)-videos $(SCORES) --condition pcm16k

# D6: the same trials under each degraded condition (handbook §3.3, SVS guide §8.1).
# Bucket and pairs come from the clean trial list; only the audio changes. A trial
# whose degraded audio falls below 3.0 s of speech is refused and counted, not scored.
DEGRADED := pcm8k g711a g711u g729 opus noise15 noise5
COND_DIR := artifacts/embeddings/conditions
COND_SCORES := $(DEGRADED:%=artifacts/scores/voxceleb1-%-cosine.tsv)

$(COND_DIR)/%/clips/.done: | $(MODEL) $(VOX_DIR)/.unpacked $(G729) $(MUSAN_DIR)/.unpacked
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(META) $(dir $@) --condition $*
	touch $@

$(COND_DIR)/%/videos/.done: | $(MODEL) $(VOX_DIR)/.unpacked $(G729) $(MUSAN_DIR)/.unpacked
	uv run python -m svs.embed_cache $(VOX_DIR)/wav $(META) $(dir $@) --join-videos --condition $*
	touch $@

degraded-embeddings: $(DEGRADED:%=$(COND_DIR)/%/clips/.done) $(DEGRADED:%=$(COND_DIR)/%/videos/.done) ## Embed clips and videos under every degraded condition (long; see LOG.md)

artifacts/scores/voxceleb1-%-cosine.tsv: $(COND_DIR)/%/clips/.done $(COND_DIR)/%/videos/.done
	uv run python -m svs.score $(TRIALS) $(COND_DIR)/$*/clips $(COND_DIR)/$*/videos $@ --condition $*

degraded-scores: $(COND_SCORES) ## Score the clean trial list under every degraded condition

eer: ## EER per duration bucket from the score file
	uv run python -m svs.metrics $(SCORES)

# D7: the same trials under each condition; the clean thresholds are the reference.
# COMMON=1 scores every condition on the trials that no condition refused.
eer-conditions: ## EER per condition and bucket, and the clean threshold under each condition
	uv run python -m svs.damage pcm16k=$(SCORES) $(join $(DEGRADED:%=%=),$(COND_SCORES)) $(if $(COMMON),--common)

ASNORM_SCORES := artifacts/scores/voxceleb1-asnorm.tsv

asnorm-scores: $(COHORT).npz ## AS-Norm score of every clean trial against the cohort
	uv run python -m svs.score $(TRIALS) $(EMB_DIR) $(EMB_DIR)-videos $(ASNORM_SCORES) --condition pcm16k --cohort $(COHORT).npz

eer-asnorm: ## EER per duration bucket from the AS-Norm score file
	uv run python -m svs.metrics $(ASNORM_SCORES)

# M2: AS-Norm per condition. The committed cohort members, degraded like the trials
# (SVS guide §4.3: the cohort matches the audio it normalises).
COND_COHORTS := $(DEGRADED:%=$(COHORT)-%.npz)
COND_ASNORM := $(DEGRADED:%=artifacts/scores/voxceleb1-%-asnorm.tsv)

$(COHORT)-%.npz: | $(MODEL) $(G729) $(MUSAN_DIR)/.unpacked
	uv run python -m svs.cohort_condition cohorts/$(notdir $(COHORT)).tsv $(VOX2_DIR)/wav $* $@

degraded-cohorts: $(COND_COHORTS) ## Embed the cohort under every degraded condition

artifacts/scores/voxceleb1-%-asnorm.tsv: $(COND_DIR)/%/clips/.done $(COND_DIR)/%/videos/.done $(COHORT)-%.npz
	uv run python -m svs.score $(TRIALS) $(COND_DIR)/$*/clips $(COND_DIR)/$*/videos $@ --condition $* --cohort $(COHORT)-$*.npz

degraded-asnorm-scores: $(COND_ASNORM) ## AS-Norm score of the trials under every degraded condition

eer-conditions-asnorm: ## As eer-conditions, on AS-Norm scores with each condition's cohort
	uv run python -m svs.damage pcm16k=$(ASNORM_SCORES) $(join $(DEGRADED:%=%=),$(COND_ASNORM)) $(if $(COMMON),--common)

OPERATING_POINTS := artifacts/calibration/voxceleb1-pcm16k.tsv

operating-points: ## STRONG and PROBABLE threshold per bucket from FRR at a fixed FAR
	uv run python -m svs.calibrate $(EMB_DIR) $(EMB_DIR)-videos $(META) $(COHORT).npz $(ASNORM_SCORES) $(OPERATING_POINTS)

# D9 per condition: fresh non-targets from that condition's embeddings, normalised
# with that condition's cohort; the targets come from its AS-Norm score file.
COND_POINTS := $(DEGRADED:%=artifacts/calibration/voxceleb1-%.tsv)

artifacts/calibration/voxceleb1-%.tsv: artifacts/scores/voxceleb1-%-asnorm.tsv
	uv run python -m svs.calibrate $(COND_DIR)/$*/clips $(COND_DIR)/$*/videos $(META) $(COHORT)-$*.npz $< $@ --condition $*

operating-points-conditions: $(COND_POINTS) ## As operating-points, for every degraded condition

# M2: one table, condition x bucket x band, with EER, threshold, precision, recall and
# trial counts. Committed: the band function reads it.
OPTABLE := docs/operating-points.tsv
comma := ,

operating-points-table: ## Join EER and operating points of every condition into one table
	uv run python -m svs.optable pcm16k=$(ASNORM_SCORES),$(OPERATING_POINTS) \
		$(foreach c,$(DEGRADED),$(c)=artifacts/scores/voxceleb1-$(c)-asnorm.tsv$(comma)artifacts/calibration/voxceleb1-$(c).tsv) \
		--out $(OPTABLE)

DET := artifacts/plots/voxceleb1-det.png

det: ## DET plot, one line per duration bucket
	uv run python -m svs.det $(SCORES) $(DET)

test: $(G729) ## Run the test suite
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
