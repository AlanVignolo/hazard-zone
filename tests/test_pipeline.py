"""Tests for GStreamer pipeline descriptions.

These assert exact strings, unlike the diagnostic-message tests elsewhere in
the project. A launch string is a contract with another program: "vide/x-raw"
is valid Python and meaningless to GStreamer, and the failure appears only on
the device.
"""

import pytest

from hazard_zone.pipeline import (
    COMPRESSED_INPUTS,
    DECODERS_HW,
    DECODERS_SW,
    MEDIA_TYPES,
    build_pipeline,
)

# --------------------------------------------------------------------------
# build_pipeline
#
# These assert exact strings, unlike the diagnostic-message tests elsewhere.
# A launch string is a contract with another program: "vide/x-raw" is valid
# Python and meaningless to GStreamer, and the failure surfaces only on the
# device, as a negotiation error that names no cause.
# --------------------------------------------------------------------------


def test_software_pipeline_is_exactly_what_was_measured():
    """The 44.6 fps path, verbatim."""
    assert build_pipeline("/dev/video0", 1280, 720, 60).launch == (
        "v4l2src device=/dev/video0 io-mode=2"
        " ! image/jpeg,width=1280,height=720,framerate=60/1"
        " ! jpegdec"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink name=sink drop=true max-buffers=1 sync=false"
    )


def test_accelerated_pipeline_uses_two_converters():
    """nvvidconv cannot change colour format and memory space in one step."""
    launch = build_pipeline("/dev/video0", 1280, 720, 60, accelerated=True).launch
    assert launch == (
        "v4l2src device=/dev/video0 io-mode=2"
        " ! image/jpeg,width=1280,height=720,framerate=60/1"
        " ! nvjpegdec"
        " ! nvvidconv"
        " ! video/x-raw(memory:NVMM),format=NV12"
        " ! nvvidconv"
        " ! video/x-raw,format=BGRx"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink name=sink drop=true max-buffers=1 sync=false"
    )


def test_raw_format_gets_no_decoder():
    launch = build_pipeline("/dev/video0", 640, 480, 30, fourcc="YUYV").launch
    assert "video/x-raw,format=YUYV,width=640,height=480,framerate=30/1" in launch
    assert "jpegdec" not in launch


def test_media_type_matches_the_fourcc():
    assert "image/jpeg" in build_pipeline("/dev/video0", 640, 480, 30, "MJPG").launch
    assert "video/x-h264" in build_pipeline("/dev/video0", 640, 480, 30, "H264").launch


def test_fourcc_is_normalised():
    """Real fourccs arrive space-padded and sometimes lower case."""
    for code in ("MJPG", "mjpg", "MJPG ", " MJPG"):
        assert "jpegdec" in build_pipeline("/dev/video0", 640, 480, 30, code).launch


def test_output_is_always_bgr():
    """Whatever the path, the consumer gets what OpenCV and the detector use."""
    for accel in (False, True):
        for code in ("MJPG", "YUYV", "H264"):
            launch = build_pipeline(
                "/dev/video0", 640, 480, 30, code, accelerated=accel
            ).launch
            assert launch.endswith(
                "video/x-raw,format=BGR"
                " ! appsink name=sink drop=true max-buffers=1 sync=false"
            ), (code, accel)


# --- the dropping policy is not optional ----------------------------------


def test_appsink_always_drops_and_holds_one():
    for accel in (False, True):
        for code in ("MJPG", "YUYV"):
            launch = build_pipeline(
                "/dev/video0", 640, 480, 30, code, accelerated=accel
            ).launch
            assert "drop=true" in launch
            assert "max-buffers=1" in launch


def test_appsink_does_not_sync_to_the_clock():
    """A live frame already exists; pacing it to the clock only adds latency."""
    assert "sync=false" in build_pipeline("/dev/video0", 640, 480, 30).launch


def test_sink_can_be_named_for_lookup():
    launch = build_pipeline("/dev/video0", 640, 480, 30, sink_name="cam0").launch
    assert "appsink name=cam0" in launch


def test_fakesink_for_measurement():
    launch = build_pipeline("/dev/video0", 640, 480, 30, sink="fakesink").launch
    assert launch.endswith("fakesink name=sink sync=false")
    assert "appsink" not in launch


# --- frame rate as an exact fraction --------------------------------------


def test_integer_rate_becomes_a_simple_fraction():
    assert "framerate=30/1" in build_pipeline("/dev/video0", 640, 480, 30).launch
    assert "framerate=60/1" in build_pipeline("/dev/video0", 640, 480, 60).launch


def test_float_integer_rate_does_not_become_a_decimal():
    assert "framerate=30/1" in build_pipeline("/dev/video0", 640, 480, 30.0).launch


def test_ntsc_rates_become_their_exact_ratio():
    """29.97 is really 30000/1001; the decimal would never match the camera."""
    assert (
        "framerate=30000/1001" in build_pipeline("/dev/video0", 640, 480, 29.97).launch
    )
    assert (
        "framerate=60000/1001" in build_pipeline("/dev/video0", 640, 480, 59.94).launch
    )


def test_rate_never_renders_as_a_decimal():
    for fps in (6, 15, 24, 25, 29.97, 30, 50, 59.94, 60, 90, 120, 120.101, 300):
        launch = build_pipeline("/dev/video0", 640, 480, fps).launch
        rate = launch.split("framerate=")[1].split(",")[0].split(" ")[0]
        assert "/" in rate and "." not in rate, (fps, rate)


# --- sources --------------------------------------------------------------


def test_file_source_replays_a_video():
    """Replay makes the whole system testable without a camera, repeatably."""
    launch = build_pipeline("clip.mp4", 1280, 720, 30, source="file").launch
    assert launch.startswith("filesrc location=clip.mp4 ! decodebin")
    assert "v4l2src" not in launch


def test_file_source_adds_no_decoder_of_its_own():
    """decodebin has already decoded, whatever the container held."""
    launch = build_pipeline("clip.mp4", 1280, 720, 30, "MJPG", source="file").launch
    assert "jpegdec" not in launch


def test_file_source_negotiates_no_capture_rate():
    """A file plays at its own rate; there is no device to negotiate with."""
    assert (
        "framerate="
        not in build_pipeline("clip.mp4", 1280, 720, 30, source="file").launch
    )


def test_rtsp_source_builds_a_depayloader_chain():
    launch = build_pipeline("rtsp://cam/stream", 1280, 720, 25, source="rtsp").launch
    assert launch.startswith("rtspsrc location=rtsp://cam/stream latency=0")
    assert "rtph264depay ! h264parse ! avdec_h264" in launch


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        build_pipeline("/dev/video0", 640, 480, 30, source="webcam")


# --- decoder selection ----------------------------------------------------


def test_software_and_hardware_pick_different_decoders():
    sw = build_pipeline("/dev/video0", 640, 480, 30, "MJPG").launch
    hw = build_pipeline("/dev/video0", 640, 480, 30, "MJPG", accelerated=True).launch
    assert "jpegdec" in sw and "nvjpegdec" not in sw
    assert "nvjpegdec" in hw


def test_every_media_type_has_decoders_on_both_paths():
    """A format the caps filter can declare must be decodable either way."""
    for code in MEDIA_TYPES:
        assert code in DECODERS_SW, code
        assert code in DECODERS_HW, code


def test_compressed_format_without_support_is_rejected():
    """An unsupported compressed format must fail, not pass as raw.

    Treating it as raw would emit video/x-raw,format=VP9, which GStreamer
    rejects on the device with a negotiation error naming no cause.
    """
    with pytest.raises(ValueError, match="VP9"):
        build_pipeline("/dev/video0", 640, 480, 30, "VP9 ")


def test_every_compressed_input_either_works_or_raises():
    """No compressed format may silently take the raw path."""
    for code in COMPRESSED_INPUTS:
        try:
            launch = build_pipeline("/dev/video0", 640, 480, 30, code).launch
        except ValueError:
            continue  # explicit rejection is the acceptable outcome
        assert f"video/x-raw,format={code}" not in launch, code


# --- the reasoning that travels with the pipeline -------------------------


def test_reason_names_the_decoder_used():
    assert "jpegdec" in build_pipeline("/dev/video0", 640, 480, 30).reason
    assert (
        "nvjpegdec"
        in build_pipeline("/dev/video0", 640, 480, 30, accelerated=True).reason
    )


def test_reason_says_when_no_decoder_is_needed():
    assert "no decoder" in build_pipeline("/dev/video0", 640, 480, 30, "YUYV").reason


def test_accelerated_flag_is_reported():
    assert not build_pipeline("/dev/video0", 640, 480, 30).accelerated
    assert build_pipeline("/dev/video0", 640, 480, 30, accelerated=True).accelerated


def test_pipeline_stringifies_to_its_launch_string():
    p = build_pipeline("/dev/video0", 640, 480, 30)
    assert str(p) == p.launch
