"""Running a GStreamer pipeline and publishing its frames.

The hardware boundary for capture. All the decisions live elsewhere --
``negotiate`` chooses the device and mode, ``pipeline`` describes the element
chain, ``frames`` holds the dropping policy -- so this module only connects
them. That is deliberate: nothing here can be tested without a camera, so
there should be as little here as possible.

GStreamer reports failures on a message bus rather than by raising, so a
pipeline nobody listens to stops producing frames with nothing logged. This
watches that bus, separates a disconnected camera from a misconfigured
pipeline, and retries only the former.
"""

import threading
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from hazard_zone.frames import LatestFrame

if TYPE_CHECKING:  # imported for type checking only, never at runtime
    from hazard_zone.pipeline import Pipeline

#: Error text that means retrying cannot help. GStreamer reports these the
#: same way as transient faults, so the message has to be inspected: a camera
#: that was unplugged will come back, while a format that cannot be
#: negotiated will fail identically forever, and retrying it presents a
#: configuration mistake as a hardware fault.
FATAL_ERRORS = (
    "not-negotiated",
    "no such element",
    "could not link",
    "Could not open device",
    "Device is not a capture device",
)


class SourceState(NamedTuple):
    """What the source is doing and why, read as one unit.

    Exposed so the main loop can report system health over CAN without
    reaching into the source's internals.
    """

    running: bool
    attempts: int
    reconnects: int
    last_error: str
    fatal: bool

    @property
    def healthy(self) -> bool:
        return self.running and not self.fatal


class GstSource:
    """Runs a pipeline and publishes frames into a LatestFrame.

    Frames arrive by callback in GStreamer's own streaming thread, which is
    the native mechanism: no polling, and no latency beyond the frame itself.
    The callback does as little as possible, because anything slow there
    throttles the pipeline.

    The buffer is copied. Wrapping it would be free, but the mapping is only
    valid until the callback returns and the frame outlives it by design --
    LatestFrame holds it for another thread. An array pointing at unmapped
    memory does not raise: the process dies with a segmentation fault, no
    traceback, which a service meant to run unattended for a week cannot
    afford. The copy costs roughly 120 MB/s at 720p44, and disappears in the
    planned GPU-resident version.

    Construction touches no hardware, so a GstSource can be built and
    inspected anywhere; ``start`` is what needs a device.
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        slot: Optional[LatestFrame] = None,
        sink_name: str = "sink",
        retry_base: float = 1.0,
        retry_max: float = 30.0,
    ) -> None:
        """Prepare a source without opening the device.

        Args:
            pipeline: Description from ``build_pipeline``. Its sink must be
                an appsink whose name matches ``sink_name``.
            slot: Where to publish frames; a new one is created if omitted.
            sink_name: Name of the appsink element in the pipeline.
            retry_base: Seconds before the first retry. Each failure doubles
                the wait, so a camera unplugged for an hour is not probed
                thousands of times.
            retry_max: Ceiling for that delay, so recovery stays prompt
                after a long outage.
        """
        self.pipeline = pipeline
        self.frames = slot if slot is not None else LatestFrame()
        self._sink_name = sink_name
        self._retry_base = retry_base
        self._retry_max = retry_max

        self._gst: Any = None  # the Gst module, imported on first start
        self._pipe: Any = None
        self._bus_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._running = False
        self._attempts = 0
        self._reconnects = 0
        self._last_error = ""
        self._fatal = False

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 10.0) -> bool:
        """Build the pipeline and wait for it to reach PLAYING.

        Reaching PLAYING is asynchronous for a live source: the state change
        returns ASYNC and completes later. Returning before the transition
        finishes would let a caller ask for a frame from a pipeline still
        opening the device, and conclude the camera is dead.

        Returns:
            True if the pipeline is playing.
        """
        gst = self._import_gst()
        if gst is None:
            return False

        with self._lock:
            self._stopping.clear()
            self._attempts += 1

        try:
            pipe = gst.parse_launch(self.pipeline.launch)
        except Exception as exc:  # GLib.Error, not an OSError subclass
            self._fail(f"cannot build pipeline: {exc}", fatal=True)
            return False

        sink = pipe.get_by_name(self._sink_name)
        if sink is None:
            self._fail(
                f"pipeline has no element named {self._sink_name!r}",
                fatal=True,
            )
            return False

        sink.set_property("emit-signals", True)
        sink.connect("new-sample", self._on_sample)

        with self._lock:
            self._pipe = pipe

        self._bus_thread = threading.Thread(
            target=self._watch_bus, args=(pipe,), name="gst-bus", daemon=True
        )
        self._bus_thread.start()

        if pipe.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            self._fail("pipeline refused to start", fatal=False)
            return False

        # ASYNC is the normal answer for a live source; wait it out.
        change, state, _ = pipe.get_state(int(timeout * gst.SECOND))
        if change == gst.StateChangeReturn.FAILURE or state != gst.State.PLAYING:
            self._fail(
                f"pipeline did not reach PLAYING within {timeout:g}",
                fatal=False,
            )
            return False

        with self._lock:
            self._running = True
            self._last_error = ""
        return True

    def stop(self) -> None:
        """Stop the pipeline, release the device and close the frame slot.

        Closing the slot wakes a consumer blocked waiting for a frame. Without
        it that thread would wait for something never coming, and the process
        would not exit.
        """
        self._stopping.set()
        self._release()
        self.frames.close()
        thread, self._bus_thread = self._bus_thread, None
        if thread is not None:
            thread.join(2.0)

    def run_forever(self, poll: float = 0.5) -> None:
        """Keep the pipeline alive, reconnecting after recoverable faults.

        Intended as the body of a supervising thread. A camera being
        unplugged is an ordinary event for a machine-mounted system, so the
        delay grows after each failure rather than probing a missing device
        continuously. Configuration errors end the loop instead, because they
        fail identically every time.
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

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> SourceState:
        """Current state, read atomically."""
        with self._lock:
            return SourceState(
                running=self._running,
                attempts=self._attempts,
                reconnects=self._reconnects,
                last_error=self._last_error,
                fatal=self._fatal,
            )

    # -- internals ---------------------------------------------------------

    def _import_gst(self) -> Any:
        """Import GStreamer on first use.

        Imported here rather than at module level so this module still
        imports on a machine without GStreamer, keeping the pure modules it
        references -- and their tests -- reachable there.
        """
        if self._gst is not None:
            return self._gst
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError) as exc:
            self._fail(f"GStreamer bindings unavailable: {exc}", fatal=True)
            return None
        if not Gst.is_initialized():
            Gst.init(None)
        self._gst = Gst
        return Gst

    def _fail(self, message: str, fatal: bool) -> None:
        with self._lock:
            self._running = False
            self._last_error = message
            if fatal:
                self._fatal = True

    def _release(self) -> None:
        """Tear the pipeline down without closing the frame slot."""
        with self._lock:
            pipe, self._pipe, self._running = self._pipe, None, False
        if pipe is not None and self._gst is not None:
            pipe.set_state(self._gst.State.NULL)

    def _on_sample(self, sink: Any) -> int:
        """Receive one frame. Runs in GStreamer's streaming thread.

        Exceptions are caught and reported rather than raised: this is called
        from C, and an exception reaching C code aborts the process instead of
        propagating.
        """
        gst = self._gst
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.OK

            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")

            ok, info = buf.map(gst.MapFlags.READ)
            if not ok:
                return gst.FlowReturn.OK
            try:
                import numpy as np

                expected = width * height * 3
                if info.size < expected:
                    self._fail(
                        f"frame is {info.size} bytes, expected {expected} for {width}x{height}",
                        fatal=True,
                    )
                    return gst.FlowReturn.ERROR
                # .copy() is required: the mapping ends when this returns and
                # the frame is consumed later, from another thread.
                image = (
                    np.frombuffer(info.data, dtype=np.uint8, count=expected)
                    .reshape(height, width, 3)
                    .copy()
                )
            finally:
                buf.unmap(info)

            self.frames.put(image)
            return gst.FlowReturn.OK
        except Exception as exc:
            self._fail(f"frame callback failed: {exc}", fatal=False)
            return gst.FlowReturn.ERROR

    def _watch_bus(self, pipe: Any) -> None:
        """Watch the message bus until the pipeline stops or fails."""
        gst = self._gst
        bus = pipe.get_bus()
        wanted = gst.MessageType.ERROR | gst.MessageType.EOS | gst.MessageType.WARNING

        while not self._stopping.is_set():
            # A timeout rather than an indefinite wait, so this thread
            # notices it has been asked to stop.
            msg = bus.timed_pop_filtered(100 * gst.MSECOND, wanted)
            if msg is None:
                continue

            if msg.type == gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                where = msg.src.get_name() if msg.src is not None else "pipeline"
                deb = debug or ""
                haystack = f"{err.message} {deb}"
                fatal = any(f in haystack for f in FATAL_ERRORS)
                self._fail(f"{where}: {err.message}", fatal=fatal)
                return
            if msg.type == gst.MessageType.EOS:
                self._fail("end of stream", fatal=False)
                return
            if msg.type == gst.MessageType.WARNING:
                warn, _ = msg.parse_warning()
                with self._lock:
                    self._last_error = "warning: {warn.message}"
