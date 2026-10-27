"""Quality deciles and the usability gate on a trial list."""

import numpy as np

from svs.quality_eval import decile_rows, gate_row, read_trials

QUALITY_HEADER = (
    "# appendix-c-v1 condition=pcm16k\n"
    "unit\tsource\tnet_seconds\tsnr_db\tspeech_ratio\tclipping_ratio\tbandwidth_hz"
    "\tturn_consistency\tquality\n"
)


def test_a_trial_takes_its_worse_side_and_its_refusal(tmp_path):
    quality = tmp_path / "quality.tsv"
    quality.write_text(
        QUALITY_HEADER
        + "a\tclip\t5.0\t20\t0.9\t0\t8000\t1\t0.90\n"
        + "b\tclip\t5.0\t20\t0.9\t0\t8000\t1\t0.40\n"
        + "c\tclip\t5.0\t20\t0.9\t0\t8000\t1\t0.80\n"
    )
    scores = tmp_path / "scores.tsv"
    scores.write_text(
        "enrol\ttest\ttarget\tbucket\tsource\tscore\n"
        "a\tb\t1\t3-10s\tclip\t0.7\n"
        "a\tc\t0\t3-10s\tclip\t0.1\n"
    )
    trials = read_trials(scores, quality)["3-10s"]
    assert trials["quality"].tolist() == [0.40, 0.80]
    assert trials["reason"].tolist() == ["E_SV_LOW_QUALITY", ""]
    assert trials["target"].tolist() == [True, False]


def test_deciles_split_by_quality_within_a_bucket():
    # 20 trials, quality rising with the index; targets are the even indices. The worst
    # tenth is trials 0 and 1, the best 18 and 19.
    trials = {
        "quality": np.arange(20) / 20,
        "score": np.where(np.arange(20) % 2 == 0, 0.9, 0.1),
        "target": np.arange(20) % 2 == 0,
    }
    worst, best, every = decile_rows(trials)
    assert (worst["quality_min"], worst["quality_max"]) == (0.0, 0.05)
    assert (best["quality_min"], best["quality_max"]) == (0.9, 0.95)
    assert (worst["n_target"], worst["n_nontarget"]) == (1, 1)
    assert every["n_target"] + every["n_nontarget"] == 20


def test_equal_qualities_do_not_split_by_file_order():
    # 100 trials at quality 1.0, the 50 targets first as in a list built by type. The best
    # tenth must sample both types, not take the last 10 rows of the file.
    trials = {
        "quality": np.ones(100),
        "score": np.where(np.arange(100) < 50, 0.9, 0.1),
        "target": np.arange(100) < 50,
    }
    worst, best, _ = decile_rows(trials)
    assert 0 < best["n_target"] < 10
    assert best["share_at_top"] == 1.0


def test_gate_counts_refusals_and_scores_only_what_passes():
    # The refused target (score 0.05) is the only error; the gate removes it.
    trials = {
        "score": np.array([0.05, 0.9, 0.8, 0.1, 0.2]),
        "target": np.array([True, True, True, False, False]),
        "reason": np.array(["E_SV_LOW_QUALITY", "", "", "", ""]),
    }
    g = gate_row(trials)
    assert g["refused"] == {"E_SV_LOW_QUALITY": 1}
    assert g["refused_target"] == 1
    assert g["eer_before"] > 0
    assert g["eer_after"] == 0.0
