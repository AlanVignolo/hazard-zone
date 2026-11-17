"""Measure preparing a frame and running the network, on the device.

Times the two steps the main loop will do every frame, separately, so a
slow total says which one is slow. Thresholds fixed before measuring, judged at p95:

  1. preparing a 1280x720 frame for the network: under 4 ms
  2. inference -- copy in, network, copy out -- from Python: under 12 ms
  3. reading the output into detections, in a crowded scene: under 4 ms


The frame is noise: preparing and inferring cost the same whatever is in
it. Reading the output does not -- it grows with the number of confident
boxes -- and noise yields almost none, so that step is also timed on a
made-up output of a crowded scene, and that is the one judged.

Run on the device, with an engine built by trtexec:

    HAZARD_ZONE_ENGINE=models/yolov8n_384x640_fp16.engine \\
        python3 -m scripts.detector_bench --seconds 20
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np

from hazard_zone.detector import decode, prepare, warm_up
from hazard_zone.inference import Engine, engine_path_from_env
from scripts.stats import percentile, summarise

PREPARE_LIMIT_MS = 4.0
INFER_LIMIT_MS = 12.0
DECODE_LIMIT_MS = 4.0


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
            done = time.perf_counter()
            prepare_ms.append((prepared - start) * 1000.0)
            infer_ms.append((inferred - prepared) * 1000.0)
            decode_ms.append((decoded - inferred) * 1000.0)
            crowded_ms.append((done - decoded) * 1000.0)

    finally:
        engine.close()

    print()
    summarise("prepare", prepare_ms)
    summarise("infer", infer_ms)
    summarise("decode", decode_ms)
    summarise("decode, crowded", crowded_ms)
    total = [p + i + c for p, i, c in zip(prepare_ms, infer_ms, crowded_ms)]
    summarise("all, crowded", total)
    print(f"\n{len(total)} frames, {len(total) / args.seconds:.1f} per second")

    print("\nVERDICT")
    checks = [
        verdict("prepare", prepare_ms, PREPARE_LIMIT_MS),
        verdict("infer", infer_ms, INFER_LIMIT_MS),
        verdict("decode, crowded", crowded_ms, DECODE_LIMIT_MS),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
