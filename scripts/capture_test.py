"""Capture from a real camera and measure what the latest-frame slot does.

A diagnostic tool, not part of the system. It answers one question: does
latency stay constant when the consumer is slower than the camera?

That is the whole argument for keeping one frame instead of a queue. With a
queue, a consumer running 19 fps behind falls 19 frames further behind every
second, so after five minutes the detector judges a scene from a minute ago --
the person is already at the machine. With the slot, the consumer always gets
the newest frame, so latency is one processing cycle however long the run
lasts.

    python3 -m scripts.capture_test --seconds 15
    python3 -m scripts.capture_test --seconds 15 --consume-ms 40
    python3 -m scripts.capture_test --seconds 15 --raw-only
    python3 -m scripts.capture_test --seconds 15 --realsense

The --consume-ms option is the point of the exercise: it simulates a detector
taking that long per frame. 40 ms is 25 fps, well under what the camera
delivers, which is exactly the condition the design has to survive.
--realsense runs the depth camera through its SDK backend instead of the
GStreamer one, and also counts how many frames carried depth with its
calibration.
"""

import argparse
import time

from hazard_zone.gst import GstSource
from hazard_zone.negotiate import pick_format, select_node
from hazard_zone.pipeline import build_pipeline
from hazard_zone.probe import probe_video_nodes
from hazard_zone.realsense import RealSenseSource
from scripts.stats import percentile, summarise

RULE = "=" * 72


def choose_mode(args):
    """Run the real discovery path and return a pipeline, or None.

    Reuses the production selection logic rather than hardcoding a device, so
    a failure here is a failure of something that ships.
    """
    nodes = probe_video_nodes()
    if not nodes:
        print("No /dev/video* nodes found. Is a camera attached?")
        return None

    node = select_node(nodes, fps=args.fps, path=args.device, card=args.card)
    if not node.ok:
        print(f"NO USABLE NODE: {node.reason}")
        for path, why in node.rejected:
            print(f"  {path:<16} {why}")
        return None

    fmt = pick_format(
        node.node.colour_formats,
        args.width,
        args.height,
        args.fps,
        allow_compressed=not args.raw_only,
    )
    if not fmt.ok:
        print(f"NO USABLE FORMAT: {fmt.reason}")
        return None

    mode = fmt.fmt
    print("device  : {}  ({})".format(node.node.path, node.node.card or "-"))
    print(f"because : {node.reason}")
    print(f"format  : {mode.fourcc.strip()} {mode.width}x{mode.height}")
    print(f"because : {fmt.reason}")

    # The pipeline asks for a rate the camera advertises, not the rate the
    # selection asked for: selection treats fps as a floor, but a caps filter
    # is an exact contract and an unadvertised value fails to negotiate.
    rate = fmt.fps
    pipeline = build_pipeline(
        device=node.node.path,
        width=mode.width,
        height=mode.height,
        fps=rate,
        fourcc=mode.fourcc,
        accelerated=args.accelerated,
    )
    print(f"rate    : {rate:g} fps requested of the pipeline")
    print("decode  : {}".format("hardware" if pipeline.accelerated else "software"))
    print(f"\n{pipeline.launch}\n")
    return pipeline


def run(source, seconds, consume_ms):
    """Consume frames for a while, recording latency and gaps.

    Returns (latencies, gaps, frames, with_depth); the first two in
    milliseconds. Latency is measured from the capture timestamp to the
    moment the consumer holds the frame; the gap is between consecutive
    frames the consumer actually saw.
    """
    latencies = []
    gaps = []
    frames = 0
    with_depth = 0
    budget = consume_ms / 1000.0
    deadline = time.monotonic() + seconds
    previous = None

    while time.monotonic() < deadline:
        frame = source.frames.get(timeout=1.0)
        if frame is None:
            if source.frames.closed:
                break
            print("  (no frame for 1 s)")
            continue

        taken = time.monotonic()
        latencies.append(frame.age(taken) * 1000.0)
        if previous is not None:
            gaps.append((taken - previous) * 1000.0)
        previous = taken
        frames += 1
        with_depth += frame.has_depth

        if budget:
            # A stand-in for inference. Sleeping, not spinning: the point is
            # to be slow at taking frames, and burning a core would change
            # what the capture thread gets scheduled.
            time.sleep(budget)

    return latencies, gaps, frames, with_depth


def report(args, source, latencies, gaps, frames, with_depth, elapsed):
    """Print the measurement and say what it means.

    The verdict is spelled out because the numbers alone invite the wrong
    reading: a high drop rate looks like a bug and is the design working.
    """
    stats = source.frames.stats()
    state = source.state
    published = stats["published"]

    print("\n" + RULE)
    print("RESULT")
    print(RULE)
    print(f"  elapsed              {elapsed:.1f} s")
    print(
        f"  published            {published}  ({published / elapsed if elapsed else 0.0:.1f} fps from the camera)"
    )
    print(
        f"  consumed             {frames}  ({frames / elapsed if elapsed else 0.0:.1f} fps by this loop)"
    )
    print(
        "  dropped              {}  ({:.0f}% of published)".format(
            stats["dropped"],
            100.0 * stats["dropped"] / published if published else 0.0,
        )
    )
    print(f"  with depth           {with_depth} of {frames} consumed")
    print(f"  reconnects           {state.reconnects}")
    if state.last_error:
        print(f"  last error           {state.last_error}")

    print("\n  distributions:")
    summarise("latency", latencies)
    summarise("gap between frames", gaps)

    if len(latencies) < 20:
        return

    tenth = max(1, len(latencies) // 10)
    first = percentile(latencies[:tenth], 0.50)
    last = percentile(latencies[-tenth:], 0.50)
    growth = last - first

    print("\n" + RULE)
    print("VERDICT")
    print(RULE)
    print(f"  median latency, first 10% of the run : {first:.1f} ms")
    print(f"  median latency, last 10% of the run  : {last:.1f} ms")

    # One frame interval of tolerance: the slot holds at most one frame, so
    # drift within a frame time is scheduling noise, not a queue.
    tolerance = max(10.0, 1000.0 / args.fps)
    if growth > tolerance:
        print(
            f"\n  LATENCY IS GROWING by {growth:.1f} ms. Frames are queueing"
            " somewhere they should not be."
        )
    else:
        print(
            f"\n  Latency is flat ({growth:+.1f} ms). The slot is doing its job:"
            " the consumer always gets the newest frame."
        )

    if args.consume_ms and stats["dropped"] == 0:
        print(
            "\n  No drops with a slow consumer is suspicious -- check that"
            " --consume-ms is really slower than the camera."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=15.0, help="how long to capture")
    p.add_argument(
        "--consume-ms",
        type=float,
        default=0.0,
        help="simulated work per frame; 40 makes the consumer a 25 fps detector",
    )
    p.add_argument("--fps", type=float, default=30.0, help="minimum frame rate")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", help="force a specific /dev/videoN")
    p.add_argument("--card", help="match a substring of the camera name")
    p.add_argument("--raw-only", action="store_true", help="reject compressed modes")
    p.add_argument(
        "--accelerated", action="store_true", help="use the NVIDIA decode elements"
    )
    p.add_argument(
        "--realsense",
        action="store_true",
        help="use the depth camera's SDK backend instead of GStreamer",
    )
    args = p.parse_args()

    print(RULE)
    print("CAPTURE TEST")
    print(RULE)

    if args.realsense:
        source = RealSenseSource(args.width, args.height, int(args.fps))
    else:
        pipeline = choose_mode(args)
        if pipeline is None:
            return
        source = GstSource(pipeline)

    started = time.monotonic()
    if not source.start():
        state = source.state
        print(f"FAILED TO START: {state.last_error}")
        print("fatal (not worth retrying)" if state.fatal else "recoverable")
        return
    print(f"playing after {time.monotonic() - started:.2f} s")

    if args.realsense:
        print(f"calibration: {source.calibration}")

    if args.consume_ms:
        print(
            f"consumer simulating {args.consume_ms:.0f} ms per frame ({1000.0 / args.consume_ms:.0f} fps capacity)"
        )
    print(f"capturing for {args.seconds:g} s ...")

    begin = time.monotonic()
    try:
        latencies, gaps, frames, with_depth = run(source, args.seconds, args.consume_ms)
    except KeyboardInterrupt:
        print("\ninterrupted")
        latencies, gaps, frames, with_depth = [], [], 0, 0
    elapsed = time.monotonic() - begin

    # Report before stopping: stop() closes the slot, and the counters have
    # to be read from the source that produced them.
    try:
        report(args, source, latencies, gaps, frames, with_depth, elapsed)
    finally:
        source.stop()


if __name__ == "__main__":
    main()
