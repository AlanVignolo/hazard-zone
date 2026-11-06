"""What every camera source promises, independent of how it is opened.

A hazard monitor must not care whether frames come from V4L2 or from a
vendor SDK: the zone logic, the detector and the CAN output all work on a
frame and a health report. This module writes that contract down, and holds
the lifecycle that every backend shares -- reconnection, health, rate
verification -- so it exists once rather than once per camera.

It is an abstract base class rather than a documented convention because
the cost of getting it wrong is asymmetric. A duck-typed backend missing
``_release`` imports fine, starts fine, and leaks the device on shutdown --
a fault that appears once, on the machine, after a week. The base class
turns that into an error at construction, on a development laptop.

Nothing here touches hardware, so the contract and its tests run anywhere.
"""

import abc
import threading
import time
from typing import Callable, NamedTuple, Optional

from hazard_zone.frames import LatestFrame

#: Frames and seconds to let pass before judging the delivered rate. The
#: first frames arrive from the camera's buffer faster than the camera
#: produces them, so an average taken too early reads high: measured here,
#: a camera delivering a steady 15 fps reported 25 after one second and
#: converged on 15 only after ten.
GRACE_FRAMES = 45
GRACE_SECONDS = 3.0


class SourceState(NamedTuple):
    """What the source is doing and why, read as one unit.

    Exposed so the main loop can report system health over CAN without
    reaching into the source's internals.

    The two rates are separate on purpose. A camera that accepts 30 fps and
    delivers 15 leaves everything downstream computing reaction times from a
    number twice the truth, which in a hazard monitor is the difference
    between warning before someone reaches the machine and warning after.
    Carrying the measured rate makes the real figure available instead of
    assumed.
    """

    running: bool
    attempts: int
    reconnects: int
    last_error: str
    fatal: bool
    requested_fps: float = 0.0
    measured_fps: float = 0.0

    @property
    def healthy(self) -> bool:
        return self.running and not self.fatal

    @property
    def rate_ok(self) -> bool:
        """Whether the camera is delivering close to the requested rate.

        True when nothing was requested or nothing measured yet: an unknown
        is not a fault. The 80% threshold tolerates the ordinary jitter of a
        live source while catching a camera delivering half of what it
        promised.
        """
        if not self.requested_fps or not self.measured_fps:
            return True
        return self.measured_fps >= 0.8 * self.requested_fps


class FrameSource(abc.ABC):
    """A camera, however it is opened.

    Subclasses differ only in how they reach the device, which is the two
    methods they must write: ``_open`` and ``_release``. Everything above
    them -- the frame slot, the health report, the reconnection policy --
    lives here, because the failures are the same: a camera gets unplugged
    whether it was opened through V4L2 or through a vendor SDK.

    Construction must touch no hardware. That is what lets a source be
    built and inspected on a machine with no camera, and it is why
    ``start`` exists separately rather than the constructor opening the
    device.
    """

    def __init__(
        self,
        requested_fps: float,
        slot: Optional[LatestFrame] = None,
        retry_base: float = 1.0,
        retry_max: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Prepare the shared state without opening anything.

        Args:
            requested_fps: Rate asked of the camera, compared against the
                delivered one. 0 disables the comparison.
            slot: Where to publish frames; a new one is created if omitted.
                Exposed before ``start`` so a consumer can be wired up
                before the device is opened.
            retry_base: Seconds before the first retry. Each failure doubles
                the wait, so a camera unplugged for an hour is not probed
                thousands of times.
            retry_max: Ceiling for that delay, so recovery stays prompt
                after a long outage.
            clock: Monotonic time source. Injectable so the rate
                arithmetic can be tested without waiting for real seconds.
        """
        self.frames = slot if slot is not None else LatestFrame()
        self._requested_fps = float(requested_fps)
        self._retry_base = retry_base
        self._retry_max = retry_max
        self._clock = clock

        #: The backend's own thread: GStreamer's bus watcher, or the SDK's
        #: capture loop. Joined on stop, whichever it is.
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._running = False
        self._attempts = 0
        self._reconnects = 0
        self._last_error = ""
        self._fatal = False
        self._started_at = 0.0
        # The slot outlives a reconnection, so its counter includes every
        # frame since the first start. Rates and "has a frame arrived" are
        # questions about this connection, so they subtract this.
        self._published_at_start = 0
        self._rate_warned = False

    # -- what a backend must write -----------------------------------------

    @abc.abstractmethod
    def _open(self, timeout: float) -> bool:
        """Open the device and begin publishing frames.

        Calls ``_mark_running`` once frames can flow, and ``_fail`` with the
        reason otherwise.
        """

    @abc.abstractmethod
    def _release(self) -> None:
        """Tear the device down without closing the frame slot.

        Must set ``_running`` to False, and be safe to call twice and on a
        device that never opened: it runs after failures.
        """

    # -- the contract ------------------------------------------------------

    @property
    def has_depth(self) -> bool:
        """Whether this source can report distance in metres.

        Defaults to False, so a backend gains depth by saying so rather
        than by forgetting to deny it. The hazard zone is defined in
        metres, and a source that cannot measure them has to be detectable
        before the zone logic asks.
        """
        return False

    def start(self, timeout: float = 10.0) -> bool:
        """Open the device and begin publishing frames.

        Returns:
            True if frames are flowing. False leaves the reason in
            ``state.last_error`` rather than raising, because a camera that
            is not there yet is an ordinary condition for a machine-mounted
            system, not an exception.
        """
        with self._lock:
            self._stopping.clear()
            self._attempts += 1
        return self._open(timeout)

    def stop(self) -> None:
        """Release the device and close the frame slot.

        Safe to call twice, and on a source that was never started:
        shutdown paths run after failures. Closing the slot wakes a
        consumer blocked waiting for a frame; without it that thread would
        wait for something never coming, and the process would not exit.
        """
        self._stopping.set()
        self._release()
        self.frames.close()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(2.0)

    def run_forever(self, poll: float = 0.5) -> None:
        """Keep the source alive, reconnecting after recoverable faults.

        Intended as the body of a supervising thread. A camera being
        unplugged is an ordinary event for a machine-mounted system, so the
        delay grows after each failure rather than probing a missing device
        continuously. Configuration errors end the loop instead, because
        they fail identically every time.
        """
        delay = self._retry_base
        while not self._stopping.is_set():
            if self.start():
                delay = self._retry_base
                while not self._stopping.is_set() and self.state.running:
                    self._stopping.wait(poll)
            if self._stopping.is_set() or self.state.fatal:
                break
            # Release but leave the slot open: a reconnection publishes into
            # the same slot, and the consumer must not have to restart
            # because the camera was briefly unplugged.
            self._release()
            with self._lock:
                self._reconnects += 1
            self._stopping.wait(delay)
            delay = min(delay * 2.0, self._retry_max)
        self.stop()

    @property
    def state(self) -> SourceState:
        """Current state, read atomically."""
        with self._lock:
            elapsed = self._clock() - self._started_at if self._started_at else 0.0
            published = self._published_since_start()
            return SourceState(
                running=self._running,
                attempts=self._attempts,
                reconnects=self._reconnects,
                last_error=self._last_error,
                fatal=self._fatal,
                requested_fps=self._requested_fps,
                measured_fps=published / elapsed if elapsed > 0 else 0.0,
            )

    # -- for backends ------------------------------------------------------

    def _mark_running(self) -> None:
        """Record that the device opened, and restart the rate measurement."""
        with self._lock:
            self._running = True
            self._last_error = ""
            self._started_at = self._clock()
            self._published_at_start = self.frames.published
            self._rate_warned = False

    def _published_since_start(self) -> int:
        """Frames published since the device last opened."""
        return self.frames.published - self._published_at_start

    def _fail(self, message: str, fatal: bool) -> None:
        with self._lock:
            self._running = False
            self._last_error = message
            if fatal:
                self._fatal = True

    def _check_rate(self) -> None:
        """Compare the delivered rate against the requested one, once.

        Called for every frame, so it must be cheap: it does nothing but
        read two counters until the sample is large enough to judge.

        The check cannot happen at startup. Negotiation succeeds -- the
        camera accepts the rate and starts -- and the shortfall only exists
        over time, so a rate error is invisible until frames have been
        counted. GRACE_FRAMES exists because the first frames arrive from
        the camera's buffer in a burst, which makes an early average
        meaningless.

        This warns and does not fail. A hazard zone watched at 15 fps is
        worse than at 30 and far better than one not watched at all, so a
        rate shortfall degrades the system rather than stopping it. What
        must not happen is the shortfall going unrecorded: the measured
        rate is published in SourceState for anything downstream that
        computes with it.
        """
        if self._rate_warned or not self._requested_fps:
            return

        elapsed = self._clock() - self._started_at
        published = self._published_since_start()
        if published < GRACE_FRAMES or elapsed < GRACE_SECONDS:
            return

        measured = published / elapsed
        self._rate_warned = True
        if measured < 0.8 * self._requested_fps:
            print(
                f"WARNING: camera delivers {measured:.1f} fps, "
                f"{self._requested_fps:g} was requested "
                f"({100.0 * measured / self._requested_fps:.0f}% of it). "
                "The system keeps running at the real rate; "
                "anything computing from the requested rate is wrong."
            )
