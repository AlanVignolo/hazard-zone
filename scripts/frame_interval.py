"""Measure the interval between frames, which is what the rate actually is.

The previous measurement timed ``wait_for_frames`` and read a median of
41 ms while counting 30 frames a second, which is a contradiction: at 30 fps
a frame arrives every 33 ms. The two cannot both be right, so this measures
the thing that is not ambiguous.

The difference: ``wait_for_frames`` returns immediately when a frame was
already waiting, so its duration measures how late *we* were to ask, not
how long the camera took. The interval between consecutive arrivals is the
camera's rate, and it is the figure every reaction-time calculation
downstream depends on.

Two clocks are compared, because they answer different questions:
  * the monotonic clock here -- when this process saw the frame, which is
    what the detector will actually work with
  * the SDK's own frame timestamp -- when the camera exposed it, which
    isolates the camera's behaviour from this process's scheduling

If the arrival intervals are ragged but the camera's are even, the jitter
is ours, not the camera's. That is a different problem with a different fix.

Run on the device, with the camera attached:

    python3 scripts/frame_interval.py --seconds 20
"""

import argparse
import sys
import time
from typing import List, Optional


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile, without numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(name: str, values: List[float]) -> None:
    """Print min, median, p95 and max. Never the mean: it hides the tail."""
    if not values:
        print(f"{name:18s} no samples")
        return
    print(
        f"{name:18s} n={len(values):4d}  "
        f"min={min(values):7.2f}  "
        f"median={percentile(values, 0.50):7.2f}  "
        f"p95={percentile(values, 0.95):7.2f}  "
        f"max={max(values):7.2f}   ms"
    )


def histogram(values: List[float], edges: List[float]) -> None:
    """Show the distribution, because a median hides two populations.

    A median of 41 ms is produced equally by every frame taking 41 ms and
    by half taking 0 and half taking 82. Those are different systems, and
    only the shape distinguishes them.
    """
    if not values:
        return
    print("\n  interval distribution:")
    low = 0.0
    for edge in edges:
        count = sum(1 for v in values if low <= v < edge)
        bar = "#" * int(60.0 * count / len(values))
        print(f"    {low:5.0f}-{edge:5.0f} ms  {count:5d}  {bar}")
        low = edge
    count = sum(1 for v in values if v >= low)
    bar = "#" * int(60.0 * count / len(values))
    print(f"    {low:5.0f}+      ms  {count:5d}  {bar}")


def run(seconds: float, fps: int, align: bool) -> Optional[dict]:
    """Capture for a while, recording when each frame arrived."""
    import pyrealsense2 as rs

    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    pipe = rs.pipeline()
    try:
        pipe.start(config)
    except RuntimeError as exc:
        print(f"could not start: {exc}", file=sys.stderr)
        return None

    aligner = rs.align(rs.stream.color) if align else None

    arrivals: List[float] = []  # monotonic clock, this process
    stamps: List[float] = []  # the SDK's, the camera
    numbers: List[int] = []  # the camera's own frame counter
    per_second: List[int] = []

    deadline = time.monotonic() + seconds
    window_end = time.monotonic() + 1.0
    in_window = 0

    try:
        while time.monotonic() < deadline:
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                print(f"frame did not arrive: {exc}", file=sys.stderr)
                break

            # Timestamp and frame number are read before aligning: align
            # produces a new frame object, and the question here is when
            # the camera delivered, not when we finished processing.
            colour = frames.get_color_frame()
            if not colour:
                continue
            arrivals.append(time.monotonic())
            stamps.append(colour.get_timestamp())  # milliseconds
            numbers.append(colour.get_frame_number())

            if aligner is not None:
                aligner.process(frames)

            in_window += 1
            now = time.monotonic()
            if now >= window_end:
                per_second.append(in_window)
                in_window = 0
                window_end = now + 1.0
    finally:
        pipe.stop()

    return {
        "arrivals": arrivals,
        "stamps": stamps,
        "numbers": numbers,
        "per_second": per_second,
    }


def report(result: dict, fps: int) -> None:
    arrivals = result["arrivals"]
    stamps = result["stamps"]
    numbers = result["numbers"]

    if len(arrivals) < 3:
        print("\nVERDICT: could not measure -- fewer than three frames.")
        return

    # The first interval is discarded: it spans the pipeline opening the
    # device, so it describes startup, not the steady state.
    gaps_here = [
        (arrivals[i] - arrivals[i - 1]) * 1000.0 for i in range(2, len(arrivals))
    ]
    gaps_camera = [stamps[i] - stamps[i - 1] for i in range(2, len(stamps))]

    expected = 1000.0 / fps
    print(f"\nexpected interval at {fps} fps: {expected:.2f} ms\n")
    summarise("arrival interval", gaps_here)
    summarise("camera interval", gaps_camera)

    histogram(gaps_here, [10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0])

    # Gaps in the camera's own counter are frames it produced and we never
    # saw -- a different loss from a slow interval, and invisible in timing.
    skipped = sum(
        numbers[i] - numbers[i - 1] - 1
        for i in range(1, len(numbers))
        if numbers[i] > numbers[i - 1]
    )
    print(f"\nframes per second: {result['per_second']}")
    print(f"frames the camera numbered but we never received: {skipped}")

    print()
    here = percentile(gaps_here, 0.50)
    cam = percentile(gaps_camera, 0.50)
    print(
        f"VERDICT: a frame arrives every {here:.1f} ms here "
        f"({1000.0 / here:.1f} fps), and the camera stamps them "
        f"{cam:.1f} ms apart ({1000.0 / cam:.1f} fps)."
    )
    if abs(here - cam) > 0.2 * expected:
        print("         The two disagree: the jitter is this process, not the camera.")
    elif abs(here - expected) > 0.2 * expected:
        print(
            "         Both agree and both differ from the requested rate: "
            "the camera is not delivering it."
        )
    else:
        print(
            "         Both agree with the requested rate. The 41 ms "
            "figure measured waiting, not arriving."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="skip alignment, to see whether it is what disturbs the rate",
    )
    args = parser.parse_args()

    result = run(args.seconds, args.fps, align=not args.no_align)
    if result is None:
        return 1
    report(result, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
