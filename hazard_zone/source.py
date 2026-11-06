"""What every camera source promises, independent of how it is opened.

A hazard monitor must not care whether frames come from V4L2 or from a
vendor SDK: the zone logic, the detector and the CAN output all work on a
frame and a health report. This module writes that contract down.

It is an abstract base class rather than a documented convention because
the cost of getting it wrong is asymmetric. A duck-typed backend missing
``stop`` imports fine, starts fine, and leaks the device on shutdown -- a
fault that appears once, on the machine, after a week. The base class turns
that into an error at construction, on a development laptop.

Nothing here touches hardware, so the contract and its tests run anywhere.
"""

import abc
from typing import NamedTuple

from hazard_zone.frames import LatestFrame


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

    Subclasses differ only in how they reach the device. Everything above
    them -- the frame slot, the health report, the reconnection policy --
    is the same, because the failures are the same: a camera gets
    unplugged whether it was opened through V4L2 or through a vendor SDK.

    Construction must touch no hardware. That is what lets a source be
    built and inspected on a machine with no camera, and it is why
    ``start`` exists separately rather than the constructor opening the
    device.
    """

    #: Where frames are published. Subclasses fill this in __init__ so a
    #: consumer can be wired up before the device is opened.
    frames: LatestFrame

    @abc.abstractmethod
    def start(self, timeout: float = 10.0) -> bool:
        """Open the device and begin publishing frames.

        Returns:
            True if frames are flowing. False leaves the reason in
            ``state.last_error`` rather than raising, because a camera that
            is not there yet is an ordinary condition for a machine-mounted
            system, not an exception.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the device and close the frame slot.

        Must be safe to call twice, and safe to call on a source that was
        never started: shutdown paths run after failures.
        """

    @abc.abstractmethod
    def run_forever(self, poll: float = 0.5) -> None:
        """Keep the source alive, reconnecting after recoverable faults.

        Intended as the body of a supervising thread. Returns only on a
        fatal fault or after ``stop``.
        """

    @property
    @abc.abstractmethod
    def state(self) -> SourceState:
        """Current state, read automatically."""

    @property
    def has_depth(self) -> bool:
        """Whether this source can report distance in metres.

        Defaults to False, so a backend gains depth by saying so rather
        than by forgetting to deny it. The hazard zone is defined in
        metres, and a source that cannot measure them has to be detectable
        before the zone logic asks.
        """
        return False
