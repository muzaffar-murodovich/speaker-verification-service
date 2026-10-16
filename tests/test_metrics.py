"""Equal error rate."""

import numpy as np
import pytest

from svs.metrics import eer


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
