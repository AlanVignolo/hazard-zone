"""Measure what the main loop will do every frame, on the device.

Times each step separately, so a slow total says which one is slow.
Thresholds fixed before measuring, judged at p95:

  1. preparing a 1280x720 frame for the network: under 4 ms
  2. inference -- copy in, network, copy out -- from Python: under 12 ms
  3. reading the output into detections, in a crowded scene: under 4 ms
  4. the distance of eight boxes: under 7 ms

The frame is noise: preparing and inferring cost the same whatever is in
it. Reading the output does not -- it grows with the number of confident
boxes -- and noise yields almost none, so that step is also timed on a
made-up output of a crowded scene, and that is the one judged.

The distances are read from one real depth map, taken from the camera at
start with its calibration; the camera is closed before anything is timed,
so its capture thread does not share the processor with the measurement.
The eight boxes are laid across that map wherever it is: the cost hardly
depends on what they cover (docs/depth.md 4.6).

Run on the device, with an engine built by trtexec, the depth camera
plugged in, and the GPU held at its top clock (docs/problems.md, 11-17):

    HAZARD_ZONE_ENGINE=models/yolov8n_384x640_fp16.engine \\
        python3 -m scripts.detector_bench --seconds 20
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

from hazard_zone.detector import decode, prepare, warm_up
from hazard_zone.distance import box_distances
from hazard_zone.inference import Engine, engine_path_from_env
from hazard_zone.projection import DepthMap
from hazard_zone.realsense import RealSenseSource
from scripts.stats import percentile, summarise

PREPARE_LIMIT_MS = 4.0
INFER_LIMIT_MS = 12.0
DECODE_LIMIT_MS = 4.0
DISTANCE_LIMIT_MS = 7.0

#: Eight people side by side across a 1280x720 frame, as (x1, y1, x2, y2).
PEOPLE = [(40.0 + 150 * k, 160.0, 160.0 + 150 * k, 560.0) for k in range(8)]


def crowded(candidates: int) -> np.ndarray:
    """A made-up engine output: eight people, each proposed by several cells.

    Sixty confident candidates is more than a real crowded frame produces
    after the network; every other candidate scores near zero, as real
    output does.
    """
    rng = np.random.RandomState(0)
    out = np.zeros((1, 84, candidates), np.float32)
    out[0, 2:4] = rng.uniform(10, 120, (2, candidates))
    out[0, 4:] = rng.uniform(0, 0.05, (80, candidates))
    for k in range(60):
        out[0, 0, k] = 80 * (k % 8) + 40 + rng.uniform(-3, 3)
        out[0, 1, k] = 192 + rng.uniform(-3, 3)
        out[0, 2, k] = 60 + rng.uniform(-3, 3)  # one person-sized box,
        out[0, 3, k] = 150 + rng.uniform(-3, 3)  # proposed a little differently
        out[0, 4, k] = 0.6 + 0.3 * rng.rand()  # class 0: person
    return out


def one_depth_map(seconds: float = 5.0) -> Optional[DepthMap]:
    """One depth map with its calibration, from the camera, which is then
    closed. None when no frame with depth arrives in time."""
    source = RealSenseSource()
    if not source.start():
        print(f"camera failed to start: {source.state.last_error}")
        return None
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = source.frames.get(timeout=1.0)
            if frame is not None and frame.depth is not None:
                return frame.depth
        return None
    finally:
        source.stop()


def verdict(name: str, values: List[float], limit: float) -> bool:
    value = percentile(values, 0.95)
    mark = "ok  " if value <= limit else "FAIL"
    print(f"  [{mark}] {name}: p95 {value:.2f} ms (limit {limit:g})")
    return value <= limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    try:
        path = engine_path_from_env(os.environ)
    except ValueError as exc:
        print(f"configuration error: {exc}")
        return 2

    depth = one_depth_map()
    if depth is None:
        print("no depth map from the camera: distances cannot be timed")
        return 2

    engine = Engine(path)
    shapes = engine.open()
    print(f"engine: {path}")
    print(f"  input {shapes.width}x{shapes.height}, {shapes.candidates} candidates")

    # The first prepare takes ~285 ms (docs/detector.md 3.11): paid here.
    warm_up(engine.input)
    frame = np.random.RandomState(0).randint(0, 256, (720, 1280, 3)).astype(np.uint8)
    prepare_ms: List[float] = []
    infer_ms: List[float] = []
    decode_ms: List[float] = []
    crowded_ms: List[float] = []
    one_ms: List[float] = []
    eight_ms: List[float] = []
    scene = crowded(shapes.candidates)
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            start = time.perf_counter()
            box = prepare(frame, engine.input)
            prepared = time.perf_counter()
            raw = engine.infer()
            inferred = time.perf_counter()
            decode(raw, box)
            decoded = time.perf_counter()
            decode(scene, box)
            decoded_crowded = time.perf_counter()
            box_distances(PEOPLE[:1], depth)
            measured_one = time.perf_counter()
            box_distances(PEOPLE, depth)
            done = time.perf_counter()
            prepare_ms.append((prepared - start) * 1000.0)
            infer_ms.append((inferred - prepared) * 1000.0)
            decode_ms.append((decoded - inferred) * 1000.0)
            crowded_ms.append((decoded_crowded - decoded) * 1000.0)
            one_ms.append((measured_one - decoded_crowded) * 1000.0)
            eight_ms.append((done - measured_one) * 1000.0)

    finally:
        engine.close()

    print()
    summarise("prepare", prepare_ms)
    summarise("infer", infer_ms)
    summarise("decode", decode_ms)
    summarise("decode, crowded", crowded_ms)
    summarise("distance, 1 box", one_ms)
    summarise("distance, 8 boxes", eight_ms)
    steps = zip(prepare_ms, infer_ms, crowded_ms, eight_ms)
    total = [p + i + c + d for p, i, c, d in steps]
    summarise("all, crowded", total)
    print(f"\n{len(total)} frames, {len(total) / args.seconds:.1f} per second")

    print("\nVERDICT")
    checks = [
        verdict("prepare", prepare_ms, PREPARE_LIMIT_MS),
        verdict("infer", infer_ms, INFER_LIMIT_MS),
        verdict("decode, crowded", crowded_ms, DECODE_LIMIT_MS),
        verdict("distance, 8 boxes", eight_ms, DISTANCE_LIMIT_MS),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
