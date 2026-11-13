"""Measure what rs.align costs per frame.

The question this answers: colour and depth come from lenses a few
centimetres apart, so a pixel means a different point in each. ``rs.align``
reprojects the depth map into the colour lens so both agree, and it does
that for every pixel of every frame. At 1280x720 that is 921,600 points,
30 times a second.

The frame budget at 30 fps is 33 ms, and detection, zone logic and CAN
output all have to fit in it too. So the cost is measured before anything
is built on top, and the threshold was decided before the measurement:
under 10 ms the choice stands, 10-20 ms is a debt to pay when the detector
exists, over 20 ms means aligning whole frames is the wrong approach and
only the few detected points should be projected instead.

Run on the device, with the camera attached:

    python3 -m scripts.align_cost --seconds 20
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

from scripts.stats import percentile, summarise


def run(
    seconds: float,
    colour: Tuple[int, int],
    depth: Tuple[int, int],
    fps: int,
) -> Optional[dict]:
    """Capture for a while, timing the align step separately from the wait.

    The two are timed apart because they fail differently. A slow
    ``wait_for_frames`` means the camera is not delivering; a slow
    ``align.process`` means our own processing does not fit the budget.
    Mixed into one figure, neither is diagnosable.
    """
    import pyrealsense2 as rs

    config = rs.config()
    config.enable_stream(rs.stream.color, colour[0], colour[1], rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth[0], depth[1], rs.format.z16, fps)

    pipe = rs.pipeline()
    try:
        profile = pipe.start(config)
    except RuntimeError as exc:
        print(f"could not start: {exc}", file=sys.stderr)
        return None

    # Aligning *to* colour, because the detector works on colour: the
    # depth map must end up in the detector's coordinate system, not the
    # other way round.
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"depth scale: {scale} m per unit")

    waits: List[float] = []
    aligns: List[float] = []
    per_second: List[int] = []

    deadline = time.monotonic() + seconds
    window_end = time.monotonic() + 1.0
    in_window = 0

    try:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                print(f"frame did not arrive: {exc}", file=sys.stderr)
                break
            t1 = time.monotonic()

            aligned = align.process(frames)
            # Fetching the frames is part of the cost: the SDK is lazy and
            # some of the work happens when the result is actually taken.
            depth_frame = aligned.get_depth_frame()
            colour_frame = aligned.get_color_frame()
            t2 = time.monotonic()

            if not depth_frame or not colour_frame:
                continue

            waits.append((t1 - t0) * 1000.0)
            aligns.append((t2 - t1) * 1000.0)

            # Counted per interval, not as a running average since start.
            # The camera's buffer empties in a burst at startup, which
            # inflates a since-start average for minutes and makes a steady
            # system look like a degrading one.
            in_window += 1
            now = time.monotonic()
            if now >= window_end:
                per_second.append(in_window)
                in_window = 0
                window_end = now + 1.0
    finally:
        pipe.stop()

    return {
        "waits": waits,
        "aligns": aligns,
        "per_second": per_second,
        "depth_shape": (depth_frame.get_height(), depth_frame.get_width()),
        "colour_shape": (colour_frame.get_height(), colour_frame.get_width()),
    }


def report(result: dict, fps: int) -> None:
    """Print the measurement and the verdict decided in advance."""
    aligns = result["aligns"]
    counts = result["per_second"]

    print()
    summarise("wait_for_frames", result["waits"])
    summarise("align.process", aligns)
    print()
    print(f"frames per second: {counts}")
    print(
        f"depth after align: {result['depth_shape']}   colour: {result['colour_shape']}"
    )

    if not aligns:
        print("\nVERDICT: could not measure -- no frames arrived.")
        return

    # p95, not the median: the budget has to hold for the slow frames too.
    cost = percentile(aligns, 0.95)
    print()
    if cost < 10.0:
        print(f"VERDICT: align costs {cost:.1f} ms at p95, under 10. Option A stands.")
    elif cost < 20.0:
        print(
            f"VERDICT: align costs {cost:.1f} ms at p95, between 10 and 20. "
            "Option A for now; project only the detected points once the "
            "detector exists."
        )
    else:
        print(
            f"VERDICT: align costs {cost:.1f} ms at p95, over 20. "
            "Whole-frame alignment does not fit the budget."
        )

    if counts:
        # Ignore the first interval: it contains the startup burst.
        steady = counts[1:] or counts
        slow = [c for c in steady if c < 0.9 * fps]
        if slow:
            print(
                f"         rate fell below {0.9 * fps:.0f} fps in "
                f"{len(slow)} of {len(steady)} intervals -- align is the "
                "only new work, so it is the cause."
            )
        else:
            print(f"         rate held at {fps} fps throughout.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--colour", default="1280x720")
    parser.add_argument("--depth", default="640x480")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    def size(text: str) -> Tuple[int, int]:
        w, h = text.lower().split("x")
        return int(w), int(h)

    result = run(args.seconds, size(args.colour), size(args.depth), args.fps)
    if result is None:
        return 1
    report(result, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
