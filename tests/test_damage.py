"""EER per condition, refusals, and the clean threshold applied to degraded audio."""

import pytest

from svs.damage import damage_rows
from svs.trials import BUCKETS

HEADER = "enrol\ttest\ttarget\tbucket\tsource\tgender\tcondition\tscore\n"
# The hand-counted example of test_metrics: EER 25 % at threshold 0.6.
TRIALS = [(1, 0.1), (1, 0.6), (1, 0.7), (1, 0.9), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.8)]


def write(path, condition, shift=0.0, drop=(), genders=("f",)):
    """The same eight trials per bucket and gender, scores moved by shift, drop left out."""
    lines = []
    for bucket in BUCKETS:
        for gender in genders:
            for i, (is_target, score) in enumerate(TRIALS):
                enrol, test = f"e{bucket}{gender}{i}", f"t{bucket}{gender}{i}"
                if (enrol, test) not in drop:
                    fields = [enrol, test, is_target, bucket, "clip", gender, condition]
                    lines.append("\t".join(map(str, fields)) + f"\t{score + shift}\n")
    path.write_text(HEADER + "".join(lines))
    return str(path)


def row(rows, condition, group="3-10s"):
    return next(r for r in rows if r["condition"] == condition and r["group"] == group)


FIRST_TARGET = ("e3-10sf0", "t3-10sf0")  # the target scored 0.1 in 3-10s


def test_same_eer_but_the_clean_threshold_misses_most_targets(tmp_path):
    # Every degraded score is 0.15 lower. The order of the trials is unchanged, so the
    # EER stays 25 %, now at threshold 0.45. The clean threshold 0.6 rejects the targets
    # 0.45 and 0.55 too: FRR 3 of 4. It accepts only the non-target 0.65: FAR 1 of 4.
    files = {
        "pcm16k": write(tmp_path / "clean.tsv", "pcm16k"),
        "g729": write(tmp_path / "g729.tsv", "g729", shift=-0.15),
    }
    rows = damage_rows(files)
    clean, g729 = row(rows, "pcm16k"), row(rows, "g729")
    assert (clean["eer"], clean["threshold"]) == (0.25, 0.6)
    assert (clean["frr_at_clean"], clean["far_at_clean"]) == (0.25, 0.25)
    assert g729["eer"] == 0.25
    assert g729["threshold"] == pytest.approx(0.45)
    assert (g729["frr_at_clean"], g729["far_at_clean"]) == (0.75, 0.25)


def test_a_trial_missing_from_a_condition_is_counted_as_refused(tmp_path):
    files = {
        "pcm16k": write(tmp_path / "clean.tsv", "pcm16k"),
        "noise5": write(tmp_path / "noise5.tsv", "noise5", drop={FIRST_TARGET}),
    }
    rows = damage_rows(files)
    assert (row(rows, "noise5")["refused"], row(rows, "noise5")["n_target"]) == (1, 3)
    assert row(rows, "noise5", "10-30s")["refused"] == 0
    assert row(rows, "pcm16k")["refused"] == 0


def test_common_only_drops_a_refused_trial_from_every_condition(tmp_path):
    files = {
        "pcm16k": write(tmp_path / "clean.tsv", "pcm16k"),
        "noise5": write(tmp_path / "noise5.tsv", "noise5", drop={FIRST_TARGET}),
    }
    clean = row(damage_rows(files, common_only=True), "pcm16k")
    assert (clean["n_target"], clean["n_nontarget"], clean["refused"]) == (3, 4, 0)
    # Targets 0.6, 0.7, 0.9. At threshold 0.7 the target 0.6 is missed (1 of 3) and the
    # non-target 0.8 accepted (1 of 4); no threshold brings the two closer.
    assert (clean["eer"], clean["threshold"]) == (pytest.approx((1 / 3 + 1 / 4) / 2), 0.7)


def test_groups_by_bucket_and_gender(tmp_path):
    files = {"pcm16k": write(tmp_path / "clean.tsv", "pcm16k", genders=("f", "m"))}
    rows = damage_rows(files, by_gender=True)
    assert [r["group"] for r in rows][:3] == ["3-10s f", "3-10s m", "10-30s f"]
    assert row(rows, "pcm16k", "3-10s m")["n_target"] == 4


def test_clean_scores_are_required(tmp_path):
    with pytest.raises(ValueError, match="pcm16k score file is needed"):
        damage_rows({"g729": write(tmp_path / "g729.tsv", "g729")})


def test_a_condition_cannot_score_a_trial_the_clean_file_lacks(tmp_path):
    files = {
        "pcm16k": write(tmp_path / "clean.tsv", "pcm16k", drop={FIRST_TARGET}),
        "g729": write(tmp_path / "g729.tsv", "g729"),
    }
    with pytest.raises(ValueError, match="clean file does not have"):
        damage_rows(files)
