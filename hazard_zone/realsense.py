"""Capturing from a RealSense camera through the vendor SDK.

The hardware boundary for the depth path, the counterpart of ``gst`` for
the generic one. Both publish into a LatestFrame and report the same
SourceState, so nothing above this module knows which one is running.

Why this path exists at all, when GStreamer already works: measured on
this camera, V4L2 delivers 15 fps at every mode it advertises, including
the ones it advertises as 30 and 60, and accepts 30 without complaint. The
SDK delivers 30. Part of the camera's behaviour is not reachable over UVC.
The SDK also provides the depth scale and the intrinsics, which V4L2
cannot report and without which a depth reading is a number with no units.

Nothing here is testable without the camera, so there is as little here as
possible. The decisions -- the frame slot's dropping policy, the health
thresholds -- live in modules that need no device.
"""

import contextlib
import threading
from typing import Any, Optional

from hazard_zone.frames import LatestFrame
from hazard_zone.projection import Calibration, DepthMap, Extrinsics, Intrinsics
from hazard_zone.source import FrameSource

#: SDK errors that retrying cannot fix. Everything else is treated as a
#: camera that will come back, because on a machine-mounted system that is
#: the common case. This is an allowlist of fatal conditions rather than of
#: recoverable ones: an unrecognised failure gets retried, which wastes
#: time, whereas the reverse would present a wiring mistake as permanent.
FATAL_ERRORS = (
    "No device connected",
    "Couldn't resolve requests",
    "failed to set power state",
    "Invalid Value",
)


class RealSenseSource(FrameSource):
    """Runs a RealSense pipeline and publishes colour frames with depth.

    Unlike GStreamer, the SDK has no callback: ``wait_for_frames`` blocks
    until a frame is ready. So this owns a capture thread, which GstSource
    does not need -- GStreamer brings its own streaming thread.

        Depth is published as the sensor delivered it, in its own lens's grid,
    together with the calibration needed to carry a colour pixel into it
    (``projection.colours_to_depth``). The whole map is not aligned to
    colour. ``rs.align`` did that here and measured 8 ms of CPU but 15 ms
    of this loop per frame, because while it aligned nothing was waiting
    for the camera -- all for 921,600 pixels, when the detector needs about
    five. Carrying the detected points instead costs 3.2 ms for twenty,
    measured, and only on the frames the consumer actually takes.

    The calibration is copied out of the SDK as plain numbers, so nothing
    the SDK owns outlives this thread. Its pool of framesets is fixed, and
    holding them elsewhere would stall capture.

    Construction touches no hardware, so a source can be built and
    inspected on a machine with no camera; ``start`` is what needs a device.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        slot: Optional[LatestFrame] = None,
        retry_base: float = 1.0,
        retry_max: float = 30.0,
    ) -> None:
        """Prepare a source without opening the device.

        Args:
            width: Colour width. 1280x720 at 30 fps is the measured working
                configuration on this camera.
            height: Colour height.
            fps: Rate to request for both streams. The SDK requires them to
                match.
            depth_width: Depth width. 640 rather than 1280 because colour
                and depth both at 720p do not start on this camera -- the
                streams do not fit the bus together, and the SDK reports it
                as a frame that never arrives. Not a compromise: depth is
                sampled only where the detector found someone, so
                resolution there buys nothing.
            depth_height: Depth height.
            slot: Where to publish frames; a new one is created if omitted.
            retry_base: Seconds before the first retry; each failure
                doubles the wait.
            retry_max: Ceiling for that delay, so recovery stays prompt
                after a long outage.
        """
        super().__init__(fps, slot, retry_base, retry_max)
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_width = depth_width
        self.depth_height = depth_height

        self._rs: Any = None  # the pyrealsense2 module, imported on start
        self._pipe: Any = None
        self._calibration: Optional[Calibration] = None

    # -- the contract ------------------------------------------------------

    @property
    def has_depth(self) -> bool:
        return True

    def _open(self, timeout: float) -> bool:
        """Open the camera and start the capture thread.

        The SDK's own start is synchronous -- it raises if the requested
        streams cannot be satisfied -- so unlike the GStreamer path there is
        no asynchronous transition to wait out. ``timeout`` bounds the wait
        for the first frame instead, which is what distinguishes a camera
        that opened from one that opened and delivers nothing.

        Returns:
            True once a first frame has arrived.
        """
        rs = self._import_rs()
        if rs is None:
            return False

        config = rs.config()
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.fps,
        )

        pipe = rs.pipeline()
        try:
            profile = pipe.start(config)
        except RuntimeError as exc:
            self._fail(f"cannot start camera: {exc}", fatal=self._is_fatal(str(exc)))
            return False

        # Read the calibration from the running profile rather than
        # hardcoding it. These numbers are specific to this unit and this
        # resolution; a swapped camera applying the previous one's values
        # would place every detection at the wrong distance, silently.
        try:
            depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            colour = profile.get_stream(rs.stream.color).as_video_stream_profile()
            calibration = Calibration(
                depth_scale=float(
                    profile.get_device().first_depth_sensor().get_depth_scale()
                ),
                depth=Intrinsics.from_sdk(depth.get_intrinsics()),
                colour=Intrinsics.from_sdk(colour.get_intrinsics()),
                colour_to_depth=Extrinsics.from_sdk(colour.get_extrinsics_to(depth)),
                depth_to_colour=Extrinsics.from_sdk(depth.get_extrinsics_to(colour)),
            )
        except (RuntimeError, ValueError) as exc:
            # ValueError is a lens model this code cannot represent, which
            # fails identically on every retry: fatal, like the SDK's own.
            with contextlib.suppress(RuntimeError):
                pipe.stop()
            self._fail(f"cannot read calibration: {exc}", fatal=True)
            return False

        with self._lock:
            self._pipe = pipe
            self._calibration = calibration
        self._mark_running()

        self._worker = threading.Thread(
            target=self._capture, name="realsense", daemon=True
        )
        self._worker.start()

        # A camera that opens and delivers nothing is a real failure mode --
        # measured here, requesting colour and depth both at 720p starts
        # without error and never produces a frame. Waiting for the first
        # one is what separates the two cases.
        waited = 0.0
        while waited < timeout:
            if self._published_since_start() > 0:
                return True
            if not self._running:
                return False
            self._stopping.wait(0.1)
            waited += 0.1
        self._fail(f"no frame within {timeout:g} s of starting", fatal=False)
        self._release()
        return False

    # -- what only this source has -----------------------------------------

    @property
    def calibration(self) -> Optional[Calibration]:
        """The calibration of the camera last opened; None before start.

        For logging at startup. Code that reads distances takes it from the
        frame instead, which cannot disagree with the depth map it came
        with.
        """
        return self._calibration

    # -- internals ---------------------------------------------------------

    def _import_rs(self) -> Any:
        """Import the SDK on first use.

        Imported here rather than at module level so this module still
        imports on a machine without the SDK, keeping the pure modules it
        references -- and their tests -- reachable there.
        """
        if self._rs is not None:
            return self._rs
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            self._fail(f"RealSense SDK unavailable: {exc}", fatal=True)
            return None
        self._rs = rs
        return rs

    def _is_fatal(self, message: str) -> bool:
        return any(f in message for f in FATAL_ERRORS)

    def _release(self) -> None:
        """Tear the camera down without closing the frame slot."""
        with self._lock:
            pipe, self._pipe, self._running = self._pipe, None, False
        if pipe is not None:
            # Stopping a pipeline that already failed raises. There is
            # nothing to recover and nothing to report: the error that got
            # us here is already in _last_error, and overwriting it with
            # this one would hide the cause.
            with contextlib.suppress(RuntimeError):
                pipe.stop()

    def _capture(self) -> None:
        """Pull frames until stopped. Runs in its own thread.

        The SDK blocks rather than calling back, so this loop is the
        equivalent of GStreamer's streaming thread. A timeout on
        ``wait_for_frames`` rather than an indefinite wait, so an
        unresponsive camera surfaces as an error instead of a hang.
        """
        import numpy as np

        while not self._stopping.is_set():
            pipe, calibration = self._pipe, self._calibration
            if pipe is None:
                return
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                colour_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not colour_frame:
                    continue

                # .copy() is required: the SDK owns this memory and reuses
                # it for the next frame, while the frame outlives this
                # iteration by design -- LatestFrame holds it for another
                # thread. An array pointing at reused memory does not
                # raise; it quietly contains a different frame's pixels.
                image = np.asanyarray(colour_frame.get_data()).copy()
                depth = (
                    DepthMap(np.asanyarray(depth_frame.get_data()).copy(), calibration)
                    if depth_frame and calibration is not None
                    else None
                )
                self.frames.put(image, depth=depth)
                self._check_rate()
            except RuntimeError as exc:
                # Includes the timeout. Recoverable unless the message says
                # otherwise: run_forever reopens the camera.
                self._fail(f"capture failed: {exc}", fatal=self._is_fatal(str(exc)))
                return
            except Exception as exc:
                self._fail(f"capture thread failed: {exc}", fatal=False)
                return
