"""Tests for the running system's parts that need no camera.

The wiring in ``main`` needs the device and is judged there, against the
live view's criteria. What is tested here is everything around it: the
measurement, the encoder's pacing and choice of frame, the per-interval
figures, and that bad configuration stops the process before it starts.
"""

import threading
import time

import numpy as np
import pytest

from hazard_zone.app import (
    EXIT_CONFIG,
    Encoder,
    IntervalStats,
    Result,
    main,
    measure_centre,
    percentile,
    status_line,
)
from hazard_zone.camera import CAMERA_ENV
from hazard_zone.frames import Broadcast, Frame
from hazard_zone.projection import DepthMap
from hazard_zone.web import PASSWORD_ENV
from tests.test_projection import CALIB, wall

JOIN = 2.0


def frame(seq=1, depth=None, captured_at=None):
    image = np.zeros((720, 1280, 3), np.uint8)
    stamp = time.monotonic() if captured_at is None else captured_at
    return Frame(image, seq, stamp, depth)


# -- the measurement ---------------------------------------------------------


def test_the_centre_reads_the_wall_in_front():
    mark = measure_centre(frame(depth=DepthMap(wall(2.0), CALIB)))
    assert (mark.u, mark.v) == (640.0, 360.0)
    assert mark.ok
    assert mark.text == "2.00 m"


def test_a_frame_without_depth_says_so():
    mark = measure_centre(frame())
    assert not mark.ok
    assert mark.text == "no depth"


def test_a_map_with_no_readings_is_refused_not_zero():
    mark = measure_centre(frame(depth=DepthMap(wall(0.0), CALIB)))
    assert not mark.ok
    assert "m" not in mark.text.split()  # no distance is shown


# -- figures per interval ----------------------------------------------------


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def test_stats_cover_the_interval_and_reset():
    clock = Clock()
    stats = IntervalStats(clock)
    stats.count("processed", 30)
    stats.sample("encode_ms", 4.0)
    clock.now += 1.0
    assert stats.take() == (1.0, {"processed": 30}, {"encode_ms": [4.0]})
    clock.now += 2.0
    assert stats.take() == (2.0, {}, {})


@pytest.mark.parametrize(
    "values, fraction, expected",
    [
        ([], 0.95, 0.0),
        ([5.0], 0.95, 5.0),
        (list(range(101)), 0.95, 95),
        ([3, 1, 2], 0.5, 2),
    ],
)
def test_percentile(values, fraction, expected):
    assert percentile(values, fraction) == expected


def test_the_status_line_carries_each_criterion():
    counts = {"processed": 150, "encoded": 75, "sent": 150}
    # Two high values in twenty: one alone would sit above the p95.
    samples = {"encode_ms": [4.0] * 18 + [7.5] * 2, "age_ms": [40.0] * 18 + [90.0] * 2}
    line = status_line(5.0, counts, samples, capture_fps=30.2, tabs=2)
    for figure in ("30.2", "30.0", "15.0", "7.5", "90.0"):
        assert figure in line, figure


def test_no_tabs_reports_nothing_sent():
    line = status_line(5.0, {"sent": 0}, {}, capture_fps=30.0, tabs=0)
    assert "0.0" in line


# -- the encoder -------------------------------------------------------------


def fake_draw(image, marks, status):
    return status.encode()


@pytest.fixture
def pipeline():
    results: Broadcast[Result] = Broadcast()
    images: Broadcast[bytes] = Broadcast()
    stats = IntervalStats()
    return results, images, stats


def test_the_encoder_draws_the_newest_result(pipeline):
    """Results published between two encodes are skipped, not queued."""
    results, images, stats = pipeline
    for seq in range(1, 6):
        results.put(Result(frame(seq), ()))
    encoder = Encoder(results, images, stats, fps=100, draw=fake_draw)
    encoder.start()
    try:
        got = images.wait_newer(0, timeout=JOIN)
        assert got is not None
        assert got[1] == b"frame 5"
    finally:
        encoder.stop()


def test_the_encoder_never_exceeds_its_rate(pipeline):
    """Invariant: published faster than the view rate, encoded at most at it."""
    results, images, stats = pipeline
    fps, seconds = 20.0, 0.6
    encoder = Encoder(results, images, stats, fps=fps, draw=fake_draw)
    encoder.start()
    deadline = time.monotonic() + seconds
    seq = 0
    while time.monotonic() < deadline:
        seq += 1
        results.put(Result(frame(seq), ()))
        time.sleep(0.002)  # a producer at ~500 fps
    encoder.stop()
    _, counts, _ = stats.take()
    assert 0 < counts["encoded"] <= fps * seconds + 2


def test_the_encoder_records_cost_and_age(pipeline):
    results, images, stats = pipeline
    encoder = Encoder(results, images, stats, fps=100, draw=fake_draw)
    encoder.start()
    try:
        results.put(Result(frame(1, captured_at=time.monotonic() - 0.05), ()))
        assert images.wait_newer(0, timeout=JOIN) is not None
    finally:
        encoder.stop()
    _, _, samples = stats.take()
    assert len(samples["encode_ms"]) == 1
    # Windows' monotonic clock ticks every ~15.6 ms, so two close readings
    # can be equal; the age is at least the 50 ms given, less a tick.
    assert samples["age_ms"][0] >= 50.0 - 16.0


def test_a_frame_that_cannot_be_drawn_does_not_end_the_view(pipeline, capsys):
    results, images, stats = pipeline
    calls = []

    def flaky_draw(image, marks, status):
        calls.append(status)
        if len(calls) == 1:
            raise ValueError("bad frame")
        return status.encode()

    encoder = Encoder(results, images, stats, fps=100, draw=flaky_draw)
    encoder.start()
    try:
        results.put(Result(frame(1), ()))
        deadline = time.monotonic() + JOIN
        while len(calls) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        results.put(Result(frame(2), ()))
        got = images.wait_newer(0, timeout=JOIN)
        assert got is not None and got[1] == b"frame 2"
    finally:
        encoder.stop()
    assert "frame 1" in capsys.readouterr().out


def test_stopping_the_encoder_closes_the_images(pipeline):
    """Closing wakes every browser tab waiting for an image."""
    results, images, stats = pipeline
    encoder = Encoder(results, images, stats, draw=fake_draw)
    encoder.start()
    stopper = threading.Thread(target=encoder.stop)
    stopper.start()
    stopper.join(JOIN)
    assert not stopper.is_alive()
    assert images.closed


# -- configuration stops the process before it starts ------------------------


def test_no_password_exits_with_the_configuration_status(capsys):
    assert main({CAMERA_ENV: "realsense"}) == EXIT_CONFIG
    assert PASSWORD_ENV in capsys.readouterr().err


def test_no_camera_choice_exits_with_the_configuration_status(capsys):
    assert main({PASSWORD_ENV: "long enough password"}) == EXIT_CONFIG
    assert CAMERA_ENV in capsys.readouterr().err
