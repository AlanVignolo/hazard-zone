"""Tests for delivered-rate verification.

A camera can negotiate a rate it does not deliver. Measured on this
hardware: a camera advertising 1280x720 at both 30 and 15 fps accepted 30/1,
reported 30 through V4L2, and produced a steady 15 -- at every resolution it
offers, including one advertising 60. Nothing in the stack objected.

These test the judgement, not the measurement: SourceState is a plain
NamedTuple, so rate_ok can be checked without a camera, which is the point
of keeping the decision out of the hardware layer.
"""

from hazard_zone.gst import SourceState
from hazard_zone.pipeline import build_pipeline


def state(requested, measured):
    """A state carrying only the fields these tests judge."""
    return SourceState(
        running=True,
        attempts=1,
        reconnects=0,
        last_error="",
        fatal=False,
        requested_fps=requested,
        measured_fps=measured,
    )


# -- the shortfall this was written for ------------------------------------


def test_half_the_requested_rate_is_not_ok():
    """The measured case: 30 requested, 15 delivered."""
    assert not state(30.0, 15.0).rate_ok


def test_the_real_rate_requested_is_ok():
    """Asking for what the camera actually delivers passes."""
    assert state(15.0, 14.9).rate_ok


# -- edge cases ------------------------------------------------------------


def test_jitter_within_tolerance_is_ok():
    """A live source never delivers an exact rate; 95% is not a fault."""
    assert state(30.0, 28.5).rate_ok


def test_exactly_at_the_threshold_is_ok():
    """80% is the boundary and is inclusive, so the limit is deliberate."""
    assert state(30.0, 24.0).rate_ok


def test_just_below_the_threshold_is_not_ok():
    assert not state(30.0, 23.9).rate_ok


def test_delivering_more_than_requested_is_ok():
    """Faster than asked is not a problem the system should report."""
    assert state(15.0, 30.0).rate_ok


# -- an unknown is not a fault ---------------------------------------------


def test_nothing_measured_yet_is_ok():
    """Before the grace period there is no measurement, which is not a fault.

    Reporting a shortfall during startup would make every start look broken.
    """
    assert state(30.0, 0.0).rate_ok


def test_nothing_requested_is_ok():
    """A pipeline built without a rate cannot be judged against one."""
    assert state(0.0, 15.0).rate_ok


def test_healthy_is_independent_of_the_rate():
    """A slow camera is still a running camera.

    This is the decision in one assertion: a rate shortfall degrades the
    system, it does not stop it. A hazard zone watched at 15 fps is worse
    than at 30 and far better than one not watched at all.
    """
    slow = state(30.0, 15.0)
    assert slow.healthy
    assert not slow.rate_ok


# -- the rate travels with the pipeline ------------------------------------


def test_pipeline_carries_the_requested_rate():
    """Without this the comparison has nothing to compare against."""
    assert build_pipeline("/dev/video0", 1280, 720, 15, fourcc="MJPG").fps == 15.0


def test_pipeline_rate_defaults_to_zero():
    """An unset rate disables the check rather than failing it."""
    from hazard_zone.pipeline import Pipeline

    assert Pipeline("launch", False, "reason").fps == 0.0
