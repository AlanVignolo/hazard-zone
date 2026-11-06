"""Tests for the lifecycle every camera backend shares.

The reconnection policy, the shutdown path and the health report live in
FrameSource, so they are tested here once, against a fake backend that
touches no hardware. The real backends only open and release a device, and
those parts need the device.
"""

import threading

import pytest

from hazard_zone.frames import LatestFrame
from hazard_zone.source import FrameSource

#: Seconds any test waits on a thread before declaring it hung. Generous,
#: because a slow CI machine must not turn into a flaky test; a deadlock
#: still fails instead of hanging the suite.
JOIN = 2.0


class FakeSource(FrameSource):
    """A backend whose device behaves as scripted.

    Each call to ``_open`` takes the next outcome: "ok" opens, "lost" fails
    in a way worth retrying, "fatal" fails in a way that is not. Running out
    of script is fatal, so a test that miscounts ends instead of looping.
    """

    def __init__(self, outcomes, **kwargs):
        kwargs.setdefault("retry_base", 0.001)
        kwargs.setdefault("retry_max", 0.004)
        super().__init__(requested_fps=30.0, **kwargs)
        self.outcomes = list(outcomes)
        self.tried = threading.Event()
        self.opened = threading.Event()
        self.releases = 0

    def _open(self, timeout):
        self.tried.set()
        outcome = self.outcomes.pop(0) if self.outcomes else "fatal"
        if outcome == "ok":
            self._mark_running()
            self.opened.set()
            return True
        self._fail(outcome, fatal=outcome == "fatal")
        return False

    def _release(self):
        with self._lock:
            self._running = False
        self.releases += 1


def run_in_thread(source):
    thread = threading.Thread(target=source.run_forever, kwargs={"poll": 0.001})
    thread.start()
    return thread


# -- the contract ----------------------------------------------------------


def test_a_backend_without_release_cannot_be_built():
    """The fault the base class exists for: a leak found on a laptop."""

    class Leaky(FrameSource):
        def _open(self, timeout):
            return True

    with pytest.raises(TypeError):
        Leaky(requested_fps=30.0)


def test_construction_touches_nothing():
    source = FakeSource(["ok"])
    state = source.state
    assert (state.running, state.attempts, state.measured_fps) == (False, 0, 0.0)
    assert source.releases == 0


def test_depth_is_denied_unless_a_backend_claims_it():
    assert not FakeSource([]).has_depth


def test_a_given_slot_is_the_one_published_into():
    slot = LatestFrame()
    assert FakeSource([], slot=slot).frames is slot


# -- start and stop --------------------------------------------------------


def test_start_counts_every_attempt():
    source = FakeSource(["lost", "ok"])
    assert not source.start()
    assert source.start()
    assert source.state.attempts == 2


def test_failure_leaves_the_reason():
    source = FakeSource(["lost"])
    source.start()
    assert source.state.last_error == "lost"
    assert not source.state.fatal


def test_opening_clears_the_previous_error():
    source = FakeSource(["lost", "ok"])
    source.start()
    source.start()
    assert source.state.last_error == ""
    assert source.state.healthy


def test_stop_on_a_source_never_started_is_safe():
    source = FakeSource([])
    source.stop()
    assert source.frames.closed


def test_stop_twice_is_safe():
    source = FakeSource(["ok"])
    source.start()
    source.stop()
    source.stop()
    assert source.frames.closed
    assert not source.state.running


def test_stop_joins_the_worker_thread():
    source = FakeSource(["ok"])
    release = threading.Event()
    source._worker = threading.Thread(target=release.wait, args=(JOIN,))
    source._worker.start()
    worker = source._worker
    release.set()
    source.stop()
    assert not worker.is_alive()
    assert source._worker is None


# -- reconnection ----------------------------------------------------------


def test_recoverable_failures_are_retried_until_the_camera_opens():
    source = FakeSource(["lost", "lost", "ok"])
    thread = run_in_thread(source)
    try:
        assert source.opened.wait(JOIN)
        state = source.state
        assert (state.attempts, state.reconnects, state.running) == (3, 2, True)
    finally:
        source.stop()
        thread.join(JOIN)
    assert not thread.is_alive()


def test_a_fatal_failure_ends_the_loop_without_retrying():
    """A configuration error fails identically forever; retrying hides it."""
    source = FakeSource(["lost", "fatal", "ok"])
    thread = run_in_thread(source)
    thread.join(JOIN)
    assert not thread.is_alive()
    state = source.state
    assert (state.attempts, state.fatal) == (2, True)
    assert source.outcomes == ["ok"]


def test_the_loop_closes_the_slot_when_it_ends():
    """Otherwise a consumer waits on a camera nobody is supervising."""
    source = FakeSource(["fatal"])
    thread = run_in_thread(source)
    thread.join(JOIN)
    assert source.frames.closed


def test_a_lost_camera_keeps_the_same_slot():
    """The consumer must not have to restart because of a loose cable."""
    source = FakeSource(["ok", "ok"])
    slot = source.frames
    thread = run_in_thread(source)
    try:
        assert source.opened.wait(JOIN)
        source.opened.clear()
        source._fail("unplugged", fatal=False)
        assert source.opened.wait(JOIN)
        assert source.frames is slot
        assert not slot.closed
        assert source.state.reconnects == 1
    finally:
        source.stop()
        thread.join(JOIN)


def test_stop_ends_a_loop_waiting_to_retry():
    source = FakeSource(["lost"] * 1000, retry_base=60.0, retry_max=60.0)
    thread = run_in_thread(source)
    # start() clears the stop flag, so stopping before the first attempt
    # would be undone by it. Waiting for the attempt tests the retry wait,
    # which is what this is about.
    assert source.tried.wait(JOIN)
    source.stop()
    thread.join(JOIN)
    assert not thread.is_alive()


# -- the rate --------------------------------------------------------------


class Clock:
    """A clock that moves only when told to."""

    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def test_measured_rate_is_frames_over_time_since_start():
    clock = Clock()
    source = FakeSource(["ok"], clock=clock)
    source.start()
    for _ in range(30):
        source.frames.put("x")
    clock.now += 2.0
    assert source.state.measured_fps == pytest.approx(15.0)


def test_requested_rate_is_reported_alongside():
    assert FakeSource([]).state.requested_fps == 30.0


def test_a_shortfall_warns_once_and_carries_the_number(capsys):
    clock = Clock()
    source = FakeSource(["ok"], clock=clock)
    source.start()
    for _ in range(45):
        source.frames.put("x")
    clock.now += 3.0
    source._check_rate()
    source._check_rate()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert "15.0" in out


def test_no_judgement_inside_the_grace_period(capsys):
    """The startup burst makes an early average meaningless."""
    clock = Clock()
    source = FakeSource(["ok"], clock=clock)
    source.start()
    for _ in range(10):
        source.frames.put("x")
    clock.now += 3.0
    source._check_rate()
    assert capsys.readouterr().out == ""


# -- regression: rate counted frames from before a reconnection -------------


def reconnected(clock, frames_before, frames_after, seconds_after):
    """A source that ran, lost the camera, and opened it again."""
    source = FakeSource(["ok", "ok"], clock=clock)
    source.start()
    for _ in range(frames_before):
        source.frames.put("x")
    clock.now += 60.0
    source.start()
    for _ in range(frames_after):
        source.frames.put("x")
    clock.now += seconds_after
    return source


def test_rate_after_reconnect_counts_only_this_connection():
    """Measured before the fix: 1800 old frames over 3 s read as 615 fps."""
    source = reconnected(Clock(), frames_before=1800, frames_after=45, seconds_after=3)
    assert source.state.measured_fps == pytest.approx(15.0)


def test_shortfall_after_reconnect_still_warns(capsys):
    """The old count read high, so a camera back at half rate went unreported."""
    source = reconnected(Clock(), frames_before=1800, frames_after=45, seconds_after=3)
    source._check_rate()
    assert "15.0" in capsys.readouterr().out


def test_grace_period_restarts_on_reconnect(capsys):
    """Old frames must not count toward the sample needed to judge.

    Before the fix, 40 old frames plus 10 new ones passed the 45-frame
    grace and were judged as 16.7 fps from three seconds of a new
    connection.
    """
    source = reconnected(Clock(), frames_before=40, frames_after=10, seconds_after=3)
    source._check_rate()
    assert capsys.readouterr().out == ""


def test_frames_since_start_ignores_the_previous_connection():
    """RealSense waits for this to be positive before reporting it opened."""
    source = reconnected(Clock(), frames_before=100, frames_after=0, seconds_after=0)
    assert source._published_since_start() == 0
