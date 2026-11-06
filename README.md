# hazard-zone

Edge AI hazard-zone monitor for machinery. A camera watches the danger zone
around a machine; when a person, vehicle or animal enters it, the system
detects them and signals the machine controller over CAN.

Everything runs on the device, next to the sensor: no internet, no server.
Target: a Jetson running Python 3.8.

## Status

Work in progress.

| Piece | State |
| --- | --- |
| Camera discovery and mode negotiation (V4L2) | done |
| Capture from any USB camera (GStreamer) | done |
| Capture with depth through the camera SDK | done |
| Letterbox and mapping detections back | done |
| Distance to a detection, in metres | in progress |
| Detector (TensorRT) | planned |
| Tracker | planned |
| Zone logic | planned |
| CAN output | planned |
| Metrics and logging | planned |

## Layout

Two ways to get frames, one contract between them. Modules marked *device*
touch hardware; everything else is pure and tested on any machine.

| Module | What it does |
| --- | --- |
| `frames.py` | `LatestFrame`: keeps only the newest frame and counts the ones dropped |
| `source.py` | `FrameSource`: the contract every camera backend meets, and the shared reconnection and health logic |
| `geometry.py` | Fits a frame into the detector's input and maps boxes back |
| **Generic path** | any USB camera, over V4L2 |
| `v4l2.py` | Parses `v4l2-ctl` output into `VideoNode` and `CameraFormat` |
| `negotiate.py` | Chooses the video node and the capture mode |
| `probe.py` | *device* — runs `v4l2-ctl` and hands the text to `v4l2.py` |
| `pipeline.py` | Writes the GStreamer launch string |
| `gst.py` | *device* — runs that pipeline and publishes frames |
| **Depth path** | depth camera, through its SDK |
| `realsense.py` | *device* — colour plus depth, with the depth scale and calibration |

Why two paths: measured on the device, the depth camera delivers 15 fps over
V4L2 at every mode, including those it advertises at 30 and 60, and 30 fps
through its SDK. The SDK also reports the depth scale and the lens
calibration, which V4L2 cannot.

`scripts/` holds diagnostic tools that are not part of the system; run them
from the repository root as `python3 -m scripts.<name>`.

| Script | What it measures |
| --- | --- |
| `probe_cameras` | Which node and mode the negotiator would choose |
| `capture_test` | That latency stays flat when the consumer is slower than the camera |
| `frame_interval` | The real interval between frames |
| `align_cost` | The per-frame cost of aligning the whole depth map |

## Design

The system is built so that everything except a thin hardware layer runs and
is tested on a development machine. Capability parsing, geometry, zone state
and CAN frame construction are pure functions over plain data; only capture,
inference and bus access touch the device.

```
pytest              # logic, runs anywhere
pytest -m hardware  # requires camera, GPU or CAN bus
```

## Development

```
uv sync
pre-commit install --hook-type commit-msg --hook-type pre-commit
pytest
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## License

MIT
