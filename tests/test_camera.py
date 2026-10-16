"""Tests for V4L2 capability parsing and format selection.

None of these touch hardware: the parser is fed captured text and the
selector is fed plain objects. See the ``hardware`` marker in pyproject.toml
for the tests that do need a real camera.
"""

from pathlib import Path

import pytest

from hazard_zone.camera import COMPRESSED_FOURCCS, CameraFormat, parse_v4l2_formats

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def generic_usb_text() -> str:
    """Captured ``v4l2-ctl --list-formats-ext`` output from a typical USB cam."""
    return (FIXTURES / "v4l2_generic_usb.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generic_usb(generic_usb_text: str):
    """The same output, already parsed."""
    return parse_v4l2_formats(generic_usb_text)


# --------------------------------------------------------------------------
# Happy path: the structure we expect from a known input
# --------------------------------------------------------------------------


def test_parses_expected_number_of_modes(generic_usb):
    # 3 MJPG resolutions + 2 YUYV resolutions
    assert len(generic_usb) == 5


def test_parses_both_pixel_formats(generic_usb):
    assert {f.fourcc for f in generic_usb} == {"MJPG", "YUYV"}


def test_parses_resolution_and_rates(generic_usb):
    hd = next(f for f in generic_usb if f.fourcc == "MJPG" and f.height == 720)
    assert (hd.width, hd.height) == (1280, 720)
    assert hd.rates == (60.0, 30.0)


def test_keeps_camera_reported_order(generic_usb):
    # MJPG was listed as [0], YUYV as [1]; order carries the camera's own
    # preference and callers may rely on it as a tie-breaker.
    assert [f.fourcc for f in generic_usb] == ["MJPG"] * 3 + ["YUYV"] * 2


def test_parses_non_integer_frame_rate(generic_usb):
    vga = next(f for f in generic_usb if f.fourcc == "MJPG" and f.height == 480)
    assert 120.101 in vga.rates


# --------------------------------------------------------------------------
# The compressed flag drives which decoder the pipeline needs
# --------------------------------------------------------------------------


def test_mjpg_is_flagged_compressed(generic_usb):
    assert all(f.compressed for f in generic_usb if f.fourcc == "MJPG")


def test_yuyv_is_not_flagged_compressed(generic_usb):
    assert not any(f.compressed for f in generic_usb if f.fourcc == "YUYV")


def test_compressed_flag_is_case_insensitive():
    # Some drivers report lowercase fourccs.
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'mjpg' (Motion-JPEG)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.compressed


# --------------------------------------------------------------------------
# Invariants: must hold for any input, not just this fixture
# --------------------------------------------------------------------------


def test_rates_are_sorted_descending(generic_usb):
    for f in generic_usb:
        assert list(f.rates) == sorted(f.rates, reverse=True)


def test_rates_are_unique(generic_usb):
    for f in generic_usb:
        assert len(set(f.rates)) == len(f.rates)


def test_max_fps_matches_first_rate(generic_usb):
    for f in generic_usb:
        assert f.max_fps == f.rates[0]


def test_pixels_matches_width_times_height(generic_usb):
    for f in generic_usb:
        assert f.pixels == f.width * f.height


def test_no_duplicate_modes(generic_usb):
    keys = [(f.fourcc, f.width, f.height) for f in generic_usb]
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------
# Edge cases: where the bugs actually live
# --------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert parse_v4l2_formats("") == []


def test_input_with_no_formats_returns_empty_list():
    assert parse_v4l2_formats("ioctl: VIDIOC_ENUM_FMT\n\tType: Video Capture\n") == []


def test_last_resolution_is_not_dropped():
    """Regression guard for the classic missing final flush.

    A line-by-line accumulating parser only emits a record when it sees the
    *next* boundary. Without an explicit flush after the loop, the final
    resolution of the input silently disappears.
    """
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 1280x720\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.017s (60.000 fps)\n"
    )
    assert [(f.width, f.height) for f in formats] == [(1280, 720), (640, 480)]
    assert formats[-1].rates == (60.0,)


def test_resolution_without_frame_rates_is_kept_with_empty_fps():
    """A resolution the camera lists but gives no interval for still exists.

    Dropping it would hide a real capability; reporting it with empty fps
    lets the selector reject it explicitly instead of never seeing it.
    """
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 1280x720\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert len(formats) == 2
    assert formats[0].rates == ()
    assert formats[0].max_fps == 0.0
    assert formats[1].rates == (30.0,)


def test_frame_rate_before_any_resolution_is_ignored():
    """Stray interval lines must not be attributed to a nonexistent size."""
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.017s (60.000 fps)\n"
    )
    assert len(formats) == 1
    assert formats[0].rates == (60.0,)


def test_format_with_no_resolutions_is_skipped():
    formats = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t[1]: 'YUYV' (YUYV 4:2:2)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert [f.fourcc for f in formats] == ["YUYV"]


def test_unknown_fourcc_is_treated_as_raw():
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'XY12' (Some vendor format)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.fourcc == "XY12"
    assert not fmt.compressed


def test_duplicate_intervals_are_collapsed():
    (fmt,) = parse_v4l2_formats(
        "\t[0]: 'MJPG' (Motion-JPEG)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    assert fmt.rates == (30.0,)


# --------------------------------------------------------------------------
# CameraFormat.supports: the NTSC tolerance
# --------------------------------------------------------------------------


def test_supports_exact_rate():
    fmt = CameraFormat("MJPG", 1280, 720, (60.0, 30.0), True)
    assert fmt.supports(30)
    assert fmt.supports(60)


def test_supports_rejects_unavailable_rate():
    fmt = CameraFormat("MJPG", 1280, 720, (60.0, 30.0), True)
    assert not fmt.supports(15)
    assert not fmt.supports(120)


def test_supports_tolerates_ntsc_rates():
    """29.97 and 59.94 are what cameras actually report for "30" and "60"."""
    fmt = CameraFormat("MJPG", 1280, 720, (59.94, 29.97), True)
    assert fmt.supports(30)
    assert fmt.supports(60)


def test_supports_respects_tolerance_boundary():
    fmt = CameraFormat("MJPG", 640, 480, (30.0,), True)
    assert fmt.supports(30.5, tol=0.5)  # exactly at the edge: inclusive
    assert not fmt.supports(30.6, tol=0.5)
    assert not fmt.supports(31.0)


def test_supports_is_false_when_no_rates_reported():
    fmt = CameraFormat("MJPG", 1280, 720, (), True)
    assert not fmt.supports(30)


# --------------------------------------------------------------------------
# Sanity check on the module-level constant
# --------------------------------------------------------------------------


def test_compressed_fourccs_are_uppercase():
    # parse_v4l2_formats upper-cases before looking up, so the set must be too.
    assert all(c == c.upper() for c in COMPRESSED_FOURCCS)
