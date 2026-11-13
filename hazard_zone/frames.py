"""Passing the newest value between threads.

Holds what happens when a producer runs faster than its consumers can keep
up: keep the newest, drop the rest. LatestFrame does it for one consumer and
counts the drops; Broadcast does it for many readers, for results that
several viewers watch. Free of GStreamer and of any device access, so the
concurrency behaviour -- the hard part -- is testable on a development
machine.
"""

import threading
import time
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Optional, Tuple, TypeVar

if TYPE_CHECKING:  # for annotations only; frames needs no numpy at import
    from hazard_zone.projection import DepthMap


class Frame(NamedTuple):
    """One captured image together with when it arrived.

    The timestamps travel with the image because latency can only be
    measured against the moment of capture, and a reading that mixed one
    frame's image with another's timestamp would report a false latency.

    Depth travels with the colour image for the same reason. A depth map
    taken a frame later than the colour it is sampled against puts a moving
    person at the wrong distance, and the error grows with their speed --
    at 1.5 m/s one frame of skew is 5 cm, which is tolerable, but nothing
    downstream could detect a larger skew if the two were held separately.
    The SDK already delivers the pair time-aligned, so keeping them together
    preserves a guarantee rather than inventing one.

    Attributes:
        image: The pixel data, typically an (H, W, 3) BGR array.
        seq: Monotonic counter of frames published, starting at 1. Gaps in
            the sequence a consumer observes are the frames it missed.
        captured_at: Monotonic clock reading taken when the frame was
            published.
        depth: The depth map with its calibration, or None when the
            source has no depth sensor. None means "this camera cannot
            measure distance", which is information: a consumer that needs
            metres must say so rather than read zeros as "touching the
            lens". The calibration travels with every frame rather than
            being read once from the source, so a camera swapped during a
            reconnection cannot have the previous one's numbers applied.
    """

    image: Any
    seq: int
    captured_at: float
    depth: Optional["DepthMap"] = None

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this frame was captured."""
        return (time.monotonic() if now is None else now) - self.captured_at

    @property
    def has_depth(self) -> bool:
        """Whether this frame can answer a question in metres."""
        return self.depth is not None


class LatestFrame:
    """A one-slot buffer that keeps the newest frame and drops the rest.

    A queue would preserve every frame at the cost of unbounded latency: with
    the camera 5 fps faster than the consumer, the frame being processed is
    two seconds old after ten seconds and a minute old after five. For a
    hazard monitor that is the worst possible failure -- the alarm would fire
    long after someone entered the zone, with nothing in the logs to show it.

    So this keeps exactly one frame. Latency stays bounded by one processing
    cycle no matter how long the system runs, and the cost is frames, which
    are nearly identical to their neighbours: at 30 fps a person walking at
    1.5 m/s moves 5 cm between them. Time is what cannot be recovered.

    Dropped frames are counted rather than silently discarded. That counter
    is how you know whether the detector is keeping up, and a counter that
    climbs is an early warning that something is degrading.

    Safe to use from two threads: one publishing, one consuming.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._dropped = 0
        self._closed = False

    def put(
        self,
        image: Any,
        captured_at: Optional[float] = None,
        depth: Optional["DepthMap"] = None,
    ) -> int:
        """Publish a frame, replacing any the consumer has not taken.

        Called from the capture thread. Never blocks and never fails: the
        camera must not be held up by a slow consumer, because stalling it
        would reintroduce exactly the latency this class exists to avoid.

        Args:
            image: The frame data.
            captured_at: Monotonic time of capture. Defaults to now; pass it
                explicitly when the source reports its own timestamp, which
                is more accurate than the moment this call happens to run.
            depth: Depth map with its calibration, or None from a
                colour-only source.

        Returns:
            The sequence number assigned to this frame.
        """
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
            self._seq += 1
            self._frame = Frame(
                image=image,
                seq=self._seq,
                captured_at=time.monotonic() if captured_at is None else captured_at,
                depth=depth,
            )
            self._ready.notify()
            return self._seq

    def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Take the current frame, waiting for one if the slot is empty.

        Taking empties the slot, so the next frame to arrive is not counted
        as dropped. Returns None on timeout and on close, which a caller
        distinguishes with ``closed``.

        Args:
            timeout: Seconds to wait. None waits indefinitely, which is
                unsuitable for a service that has to shut down cleanly; a
                finite timeout lets the consumer loop check for shutdown.

        Returns:
            The newest frame, or None if none arrived in time or the slot
            was closed.
        """
        with self._ready:
            if self._frame is None and not self._closed:
                self._ready.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    def peek(self) -> Optional[Frame]:
        """Look at the current frame without taking it.

        Leaving it in place means it will be counted as dropped when the next
        one arrives, which is correct: nobody consumed it.
        """
        with self._lock:
            return self._frame

    def close(self) -> None:
        """Mark the stream finished and wake anyone waiting.

        Without this a consumer blocked in ``get`` would wait for a frame
        that is never coming, and the process would not exit.
        """
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    @property
    def closed(self) -> bool:
        """Whether the stream has been closed."""
        with self._lock:
            return self._closed

    @property
    def dropped(self) -> int:
        """Frames overwritten before anyone consumed them.

        Roughly the camera's rate minus the consumer's rate. Steady is
        normal; climbing means something is getting worse.
        """
        with self._lock:
            return self._dropped

    @property
    def published(self) -> int:
        """Frames published, whether consumed or dropped."""
        with self._lock:
            return self._seq

    def stats(self) -> dict:
        """Published, dropped and consumed counts, read as one unit."""
        with self._lock:
            return {
                "published": self._seq,
                "dropped": self._dropped,
                "consumed": self._seq - self._dropped - (self._frame is not None),
                "closed": self._closed,
            }


T = TypeVar("T")


class Broadcast(Generic[T]):
    """The newest value, for any number of readers.

    LatestFrame has one consumer by design: taking a frame empties the slot,
    so a second reader would steal frames from the first. Results headed for
    the live view have several readers -- one per open browser tab -- and
    each must see the same values without taking them from the others.

    So nothing is taken. Every value gets a sequence number, and a reader
    asks for anything newer than the last number it saw. A reader that falls
    behind skips straight to the newest value, for the same reason
    LatestFrame drops frames: a picture of the past is worth less than the
    present, and a backlog would grow without bound.

    Safe to use from any number of threads.
    """

    def __init__(self) -> None:
        self._ready = threading.Condition()
        self._latest: Optional[Tuple[int, T]] = None
        self._closed = False

    def put(self, value: T) -> int:
        """Publish a value, waking every waiting reader. Never blocks.

        Returns:
            The sequence number given to the value, starting at 1.
        """
        with self._ready:
            seq = self._latest[0] + 1 if self._latest is not None else 1
            self._latest = (seq, value)
            self._ready.notify_all()
            return seq

    def wait_newer(
        self, seen: int, timeout: Optional[float] = None
    ) -> Optional[Tuple[int, T]]:
        """Wait for a value newer than ``seen``; return it with its number.

        Args:
            seen: The last sequence number this reader received; 0 for none.
            timeout: Seconds to wait. None waits until a value or a close.

        Returns:
            ``(seq, value)``, or None on timeout or close with nothing newer.
            A value published before the close is still delivered, so a
            reader can drain what was there when the producer stopped.
        """
        with self._ready:
            self._ready.wait_for(lambda: self._is_newer(seen) or self._closed, timeout)
            return self._latest if self._is_newer(seen) else None

    def latest(self) -> Optional[Tuple[int, T]]:
        """The newest value and its number, without waiting; None if none."""
        with self._ready:
            return self._latest

    def close(self) -> None:
        """Wake every waiting reader for good: the producer has stopped."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    @property
    def closed(self) -> bool:
        with self._ready:
            return self._closed

    def _is_newer(self, seen: int) -> bool:
        return self._latest is not None and self._latest[0] > seen
