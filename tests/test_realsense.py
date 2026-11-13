"""Tests for the depth camera backend that need no camera.

Everything this backend does with a device is untestable here, which is why
it does as little as possible. What can be checked is the contract around
the device: construction touches nothing, and a machine without the SDK
gets a reason instead of an exception.
"""

import importlib.util

import pytest

from hazard_zone.realsense import RealSenseSource

NO_SDK = importlib.util.find_spec("pyrealsense2") is None


def test_construction_touches_no_hardware():
    source = RealSenseSource()
    assert source.has_depth
    assert source.calibration is None
    assert not source.state.running


def test_the_requested_rate_is_reported():
    assert RealSenseSource(fps=15).state.requested_fps == 15.0


def test_alignment_can_no_longer_be_requested():
    """Removed on measurement: 15 ms of a 33 ms frame. See projection."""
    with pytest.raises(TypeError):
        RealSenseSource(align=True)  # type: ignore[call-arg]


@pytest.mark.skipif(not NO_SDK, reason="the SDK is installed here")
def test_without_the_sdk_start_fails_with_a_reason():
    """A missing SDK fails identically on every retry, so it is fatal."""
    source = RealSenseSource()
    assert not source.start(timeout=0.1)
    state = source.state
    assert state.fatal
    assert "SDK unavailable" in state.last_error
