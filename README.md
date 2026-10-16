# hazard-zone

Edge AI hazard-zone monitor for machinery. A camera watches the danger zone
around a machine; when a person, vehicle or animal enters it, the system
detects them and signals the machine controller over CAN.

Everything runs on the device, next to the sensor: no internet, no server.

## Status

Work in progress. See `docs/` for the design.

| Piece | State |
| --- | --- |
| Camera capability negotiation | in progress |
| GStreamer pipeline | planned |
| Detector (TensorRT) | planned |
| Tracker | planned |
| Zone logic | planned |
| CAN output | planned |
| Metrics and logging | planned |

## Design

The system is built so that everything except a thin hardware layer runs and
is tested on a development machine. Capability parsing, geometry, zone state
and CAN frame construction are pure functions over plain data; only capture,
inference and bus access touch the device.

pytest              # logic, runs anywhere
pytest -m hardware  # requires camera, GPU or CAN bus

## Development

uv sync
pre-commit install --hook-type commit-msg --hook-type pre-commit
pytest

Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## License

MIT
