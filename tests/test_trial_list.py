"""Trial list: each bucket comes from its own source, and a trial never mixes sources."""

from svs.trial_list import build_trial_list
from svs.trials import Utterance


def pool(prefix, seconds):
    """Two male speakers, three sessions each, every utterance the same length."""
    return [
        Utterance(f"{prefix}-{s}{v}", speaker=s, session=f"{s}{v}", gender="m", duration=seconds)
        for s in "PQ"
        for v in range(3)
    ]


def test_short_buckets_use_clips_and_long_buckets_use_videos():
    # Clips of 5 s and 15 s cover the short buckets; videos of 60 s and 200 s the long ones.
    clips = pool("c5", 5.0) + pool("c15", 15.0)
    videos = pool("v60", 60.0) + pool("v200", 200.0)
    rows = build_trial_list({"clip": clips, "video": videos}, n_target=2, n_nontarget=2)

    clip_ids = {u.utt_id for u in clips}
    sources = {}
    for source, trial in rows:
        sources.setdefault(trial.bucket, set()).add(source)
        both_clips = {trial.enrol.utt_id, trial.test.utt_id} <= clip_ids
        assert both_clips == (source == "clip")
    expected = {"3-10s": "clip", "10-30s": "clip", "30-120s": "video", ">120s": "video"}
    assert sources == {bucket: {source} for bucket, source in expected.items()}
    assert len(rows) == 4 * (2 + 2)
