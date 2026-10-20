"""Moving frames from the camera to the application.

Holds the latest-frame slot: the piece that decides what happens when the
camera produces faster than the consumer can keep up. Deliberately free of
GStreamer and of any device access, so the concurrency behaviour -- which is
the hard part -- can be tested on a development machine.
"""

import threading
import time
from typing import Any, NamedTuple, Optional


class Frame(NamedTuple):
    """One captured image together with when it arrived.

    The timestamps travel with the image because latency can only be
    measured against the moment of capture, and a reading that mixed one
    frame's image with another's timestamp would report a false latency.

    Attributes:
        image: The pixel data, typically an (H, W, 3) BGR array.
        seq: Monotonic counter of frames published, starting at 1. Gaps in
            the sequence a consumer observes are the frames it missed.
        captured_at: Monotonic clock reading taken when the frame was
            published.
    """

    image: Any
    seq: int
    captured_at: float

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this frame was captures"""
        return (time.monotonic() if now is None else now) - self.captured_at


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

    def put(self, image: Any, captured_at: Optional[float] = None) -> int:
        """Publish a frame, replacing any the consumer has not taken.

        Called from the capture thread. Never blocks and never fails: the
        camera must not be held up by a slow consumer, because stalling it
        would reintroduce exactly the latency this class exists to avoid.

        Args:
            image: The frame data.
            captured_at: Monotonic time of capture. Defaults to now; pass it
                explicitly when the source reports its own timestamp, which
                is more accurate than the moment this call happens to run.

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
