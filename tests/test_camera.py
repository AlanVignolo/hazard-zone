"""Tests for choosing the camera backend from the environment.

The environment variable is configuration, so a bad one must stop the
system before any device is touched. These tests pass a probe that fails
if called, which is how "before any device is touched" is checked rather
than assumed.
"""

import pytest

from hazard_zone.camera import (
    CAMERA_ENV,
    CameraSettings,
    CameraUnavailable,
    camera_from_env,
    choose_v4l2,
    make_camera,
)
from hazard_zone.gst import GstSource
from hazard_zone.realsense import RealSenseSource


def probe_must_not_run():
    raise AssertionError("a device was probed before the configuration was accepted")


def env(value):
    return {CAMERA_ENV: value}


# -- reading the variable --------------------------------------------------


@pytest.mark.parametrize("value", ["realsense", "v4l2"])
def test_each_accepted_value_names_its_backend(value):
    assert camera_from_env(env(value)) == value


def test_case_and_spaces_are_forgiven():
    assert camera_from_env(env("  RealSense ")) == "realsense"


def test_unset_is_an_error_naming_every_choice():
    with pytest.raises(ValueError) as error:
        camera_from_env({})
    message = str(error.value)
    assert CAMERA_ENV in message
    assert "realsense" in message and "v4l2" in message


@pytest.mark.parametrize("value", ["", "realsens", "gst", "depth", "0"])
def test_anything_else_is_an_error_naming_the_value(value):
    """An allowlist: nothing unrecognised falls through to either backend."""
    with pytest.raises(ValueError) as error:
        camera_from_env(env(value))
    message = str(error.value)
    assert repr(value) in message
    assert "realsense" in message and "v4l2" in message


# -- configuration fails before hardware -----------------------------------


@pytest.mark.parametrize("environ", [{}, env("realsens")])
def test_a_bad_configuration_touches_no_device(environ):
    with pytest.raises(ValueError):
        make_camera(environ, probe=probe_must_not_run)


def test_realsense_is_built_without_probing():
    camera = make_camera(env("realsense"), probe=probe_must_not_run)
    assert isinstance(camera.source, RealSenseSource)
    assert camera.kind == "realsense"
    assert not camera.source.state.running


def test_realsense_gets_the_settings():
    settings = CameraSettings(width=640, height=480, fps=15)
    source = make_camera(env("realsense"), settings, probe_must_not_run).source
    assert isinstance(source, RealSenseSource)
    assert (source.width, source.height) == (640, 480)
    assert source.state.requested_fps == 15.0


# -- the V4L2 choice -------------------------------------------------------


def test_v4l2_builds_a_gstreamer_source(two_cameras):
    camera = make_camera(env("v4l2"), probe=lambda: two_cameras)
    assert isinstance(camera.source, GstSource)
    assert camera.kind == "v4l2"
    assert camera.v4l2 is not None and camera.v4l2.ok


def test_v4l2_choice_can_be_steered_to_the_depth_camera(two_cameras):
    choice = choose_v4l2(two_cameras, CameraSettings(card="depth"))
    assert choice.ok
    assert choice.node.node.path == "/dev/video6"
    assert choice.pipeline.launch.startswith("v4l2src device=/dev/video6 ")
    assert "width=1280,height=720,framerate=30/1" in choice.pipeline.launch


def test_v4l2_asks_the_pipeline_for_an_advertised_rate(two_cameras):
    """Selection treats fps as a floor; the caps filter must be exact."""
    choice = choose_v4l2(two_cameras, CameraSettings(card="depth", fps=29.97))
    assert choice.ok
    assert choice.pipeline.fps == pytest.approx(29.97)


def test_no_nodes_is_unavailable_with_the_reason():
    with pytest.raises(CameraUnavailable) as error:
        make_camera(env("v4l2"), probe=lambda: [])
    assert not error.value.choice.ok
    assert error.value.choice.node.reason == "no video nodes found"


def test_an_unmatched_card_keeps_what_was_ruled_out(two_cameras):
    """The log must be able to say which nodes lost, and why."""
    with pytest.raises(CameraUnavailable) as error:
        make_camera(env("v4l2"), CameraSettings(card="thermal"), lambda: two_cameras)
    rejected = dict(error.value.choice.node.rejected)
    assert "/dev/video0" in rejected and "/dev/video6" in rejected


def test_a_rate_no_mode_reaches_is_unavailable(two_cameras):
    with pytest.raises(CameraUnavailable) as error:
        make_camera(env("v4l2"), CameraSettings(fps=120), lambda: two_cameras)
    assert error.value.choice.fmt is None
