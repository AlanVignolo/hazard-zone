"""The running system: camera, measurement and live view, wired together.

Each stage runs in its own thread and hands the newest result to the next,
dropping what it cannot keep up with, so no stage can slow the one before:

    camera -> LatestFrame -> main loop -> Broadcast[Result]
           -> encoder (15 fps) -> Broadcast[bytes] -> browser tabs

The main loop is where the detector will run. Today it measures the
distance at the centre of the frame, which exercises the whole path --
depth, calibration, projection, drawing, encoding, serving -- before there
is anything to detect.

Configuration is read before anything is opened: a missing camera choice
or password stops the process with a message and exit status 2, without
touching the camera or the network.

Every few seconds a status line reports what the live view's acceptance
criteria are judged on, counted over that interval rather than averaged
since start: a running average hides both the startup burst and a later
slowdown.
"""

import os
import signal
import sys
import threading
import time
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Tuple

from hazard_zone.camera import CameraUnavailable, make_camera
from hazard_zone.frames import Broadcast, Frame
from hazard_zone.projection import colours_to_depth
from hazard_zone.view import Mark, distance_mark, render
from hazard_zone.web import ViewServer, password_from_env

#: Frames per second sent to the browser. Enough to follow movement; each
#: one costs an encode, so the camera's 30 would double that work for no
#: visible gain.
VIEW_FPS = 15.0

#: Port the live view listens on.
VIEW_PORT = 8080

#: Seconds between status lines.
STATUS_SECONDS = 5.0

#: Exit statuses, so a service manager can tell a configuration mistake
#: (fix the configuration; restarting will not help) from a runtime fault.
EXIT_OK = 0
EXIT_FAULT = 1
EXIT_CONFIG = 2


class Result(NamedTuple):
    """One frame and what the system concluded about it."""

    frame: Frame
    marks: Tuple[Mark, ...]


def measure_centre(frame: Frame) -> Mark:
    """The distance at the centre of the frame, or why there is none."""
    height, width = frame.image.shape[:2]
    u, v = width / 2.0, height / 2.0
    if frame.depth is None:
        return distance_mark(u, v, None)
    match = colours_to_depth(frame.depth.raw, frame.depth.calibration, [(u, v)])[0]
    return distance_mark(u, v, match)


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


class IntervalStats:
    """Counts and timings since the last report, then reset.

    Thread-safe: the encoder adds samples while the main loop reports.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._since = clock()
        self._counts: dict = {}
        self._samples: dict = {}

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + n

    def sample(self, name: str, value: float) -> None:
        with self._lock:
            self._samples.setdefault(name, []).append(value)

    def take(self) -> Tuple[float, dict, dict]:
        """(seconds covered, counts, samples) since the last take; resets."""
        with self._lock:
            now = self._clock()
            covered, self._since = now - self._since, now
            counts, self._counts = self._counts, {}
            samples, self._samples = self._samples, {}
        return covered, counts, samples


class Encoder:
    """Turns results into JPEGs for the view, at most ``fps`` per second.

    Runs in its own thread so that encoding never costs the main loop time.
    It always encodes the newest result: between two encodes the main loop
    may publish several, and the ones in between are never drawn.
    """

    def __init__(
        self,
        results: Broadcast[Result],
        images: Broadcast[bytes],
        stats: IntervalStats,
        fps: float = VIEW_FPS,
        draw: Callable[..., bytes] = render,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.results = results
        self.images = images
        self.stats = stats
        self.period = 1.0 / fps
        self._draw = draw
        self._clock = clock
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="encoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop encoding and close the images. Safe to call twice."""
        self._stopping.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(2.0)
        self.images.close()

    def _run(self) -> None:
        seen = 0
        due = self._clock()
        while not self._stopping.is_set():
            wait = due - self._clock()
            if wait > 0 and self._stopping.wait(wait):
                return
            got = self.results.wait_newer(seen, timeout=0.5)
            if got is None:
                if self.results.closed:
                    return
                continue
            seen, result = got
            started = self._clock()
            due = started + self.period
            try:
                jpeg = self._draw(result.frame.image, result.marks, _status(result))
            except Exception as exc:  # a bad frame must not end the view
                print(f"view: could not draw frame {result.frame.seq}: {exc}")
                continue
            finished = self._clock()
            self.images.put(jpeg)
            self.stats.count("encoded")
            self.stats.sample("encode_ms", (finished - started) * 1000.0)
            self.stats.sample("age_ms", (finished - result.frame.captured_at) * 1000.0)


def _status(result: Result) -> str:
    return f"frame {result.frame.seq}"


def status_line(
    covered: float, counts: dict, samples: dict, capture_fps: float, tabs: int
) -> str:
    """One line with the figures the live view is judged on."""
    if covered <= 0:
        return "status: no time elapsed"

    def rate(name: str) -> float:
        return float(counts.get(name, 0)) / covered

    per_tab = rate("sent") / tabs if tabs else 0.0
    return (
        f"capture {capture_fps:4.1f} fps | "
        f"processed {rate('processed'):4.1f} fps | "
        f"encoded {rate('encoded'):4.1f} fps | "
        f"sent {per_tab:4.1f} fps x {tabs} tab(s) | "
        f"encode p95 {percentile(samples.get('encode_ms', []), 0.95):5.1f} ms | "
        f"age p95 {percentile(samples.get('age_ms', []), 0.95):5.1f} ms"
    )


def main(environ: Mapping[str, str] = os.environ) -> int:
    """Run the system until interrupted. Returns the process exit status."""
    try:
        password = password_from_env(environ)
        camera = make_camera(environ)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except CameraUnavailable as exc:
        print(f"no camera: {exc}", file=sys.stderr)
        for path, why in exc.choice.node.rejected:
            print(f"  {path:<16} {why}", file=sys.stderr)
        return EXIT_FAULT

    stopping = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        stopping.set()

    # Ctrl+C and a service manager's stop both end the loop cleanly.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    source = camera.source
    results: Broadcast[Result] = Broadcast()
    images: Broadcast[bytes] = Broadcast()
    stats = IntervalStats()
    encoder = Encoder(results, images, stats)
    server = ViewServer(images, password, port=VIEW_PORT)

    supervisor = threading.Thread(target=source.run_forever, name="camera", daemon=True)
    print(f"camera: {camera.kind}; live view on port {VIEW_PORT}")
    supervisor.start()
    encoder.start()
    server.start()

    status_due = time.monotonic() + STATUS_SECONDS
    sent_before = 0
    try:
        while not stopping.is_set():
            frame = source.frames.get(timeout=0.5)
            if frame is None:
                if source.frames.closed:
                    break  # the camera's supervisor gave up: a fatal fault
            else:
                results.put(Result(frame, (measure_centre(frame),)))
                stats.count("processed")

            if time.monotonic() >= status_due:
                status_due += STATUS_SECONDS
                sent_now = server.sent
                stats.count("sent", sent_now - sent_before)
                sent_before = sent_now
                state = source.state
                print(status_line(*stats.take(), state.measured_fps, server.streams))
    finally:
        source.stop()
        results.close()
        encoder.stop()
        server.stop()
        supervisor.join(2.0)

    state = source.state
    if state.fatal:
        print(f"camera failed: {state.last_error}", file=sys.stderr)
        return EXIT_FAULT
    return EXIT_OK
