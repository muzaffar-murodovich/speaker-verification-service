"""Equal error rate and minDCF."""

import numpy as np
import pytest

from svs.metrics import eer, min_dcf, read_bucket_scores


def test_eer_of_a_hand_counted_example():
    # At threshold 0.6: one target (0.1) is missed, one non-target (0.8) is accepted.
    # 1 of 4 on each side, so the EER is 25 %.
    target = np.array([0.1, 0.6, 0.7, 0.9])
    nontarget = np.array([0.2, 0.3, 0.4, 0.8])
    assert eer(target, nontarget) == (0.25, 0.6)


def test_separated_scores_give_zero_eer():
    assert eer(np.array([0.7, 0.9]), np.array([0.1, 0.3]))[0] == 0.0


def test_a_tie_between_target_and_nontarget_is_a_coin_flip():
    # One score for both: accepting takes the non-target too, rejecting loses the target.
    assert eer(np.array([0.5]), np.array([0.5]))[0] == 0.5


def test_eer_needs_both_kinds_of_trial():
    with pytest.raises(ValueError, match="both target and non-target"):
        eer(np.array([0.5]), np.array([]))


def test_min_dcf_of_a_hand_counted_example():
    # p_target 0.01, equal costs. At threshold 0.7 one of three targets (0.4) is missed and
    # no non-target is accepted: DCF = 0.01 * 1/3, divided by 0.01 gives 1/3. Threshold 0.4
    # instead accepts one of four non-targets: 0.99 * 1/4 = 0.2475, far more costly.
    target = np.array([0.9, 0.7, 0.4])
    nontarget = np.array([0.6, 0.3, 0.2, 0.1])
    cost, threshold = min_dcf(target, nontarget)
    assert cost == pytest.approx(1 / 3)
    assert threshold == 0.7


def test_min_dcf_never_exceeds_rejecting_everything():
    # Any threshold at or below 0.5 accepts the non-target (cost 0.99 / 0.01 = 99);
    # rejecting every trial costs exactly the normaliser, so the minimum is 1.0.
    cost, threshold = min_dcf(np.array([0.5]), np.array([0.5]))
    assert (cost, threshold) == (pytest.approx(1.0), np.inf)


def test_min_dcf_weighs_false_alarms_by_the_prior():
    # One missed target versus one accepted non-target, 2 trials each side. With equal
    # costs and p_target 0.5 the two errors cost the same; at 0.01 the miss is cheaper.
    target, nontarget = np.array([0.3, 0.9]), np.array([0.1, 0.5])
    assert min_dcf(target, nontarget, p_target=0.5)[0] == pytest.approx(0.5)
    assert min_dcf(target, nontarget)[1] == 0.9


def test_scores_are_grouped_by_bucket_and_gender(tmp_path):
    path = tmp_path / "scores.tsv"
    path.write_text(
        "target\tbucket\tgender\tscore\n"
        "1\t3-10s\tm\t0.9\n0\t3-10s\tm\t0.1\n1\t3-10s\tf\t0.8\n0\t10-30s\tf\t0.2\n"
    )
    by_bucket = read_bucket_scores(path)
    assert by_bucket["3-10s"][0].tolist() == [0.9, 0.8]
    by_gender = read_bucket_scores(path, by_gender=True)
    assert list(by_gender)[:4] == ["3-10s f", "3-10s m", "10-30s f", "10-30s m"]
    assert by_gender["3-10s f"][0].tolist() == [0.8]
    assert by_gender["3-10s m"][1].tolist() == [0.1]
    assert by_gender["10-30s f"][1].tolist() == [0.2]
