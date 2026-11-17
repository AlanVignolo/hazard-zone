"""Tests for running the network.

What needs no GPU is tested everywhere: where the engine comes from, and
which engines are accepted. Running one needs the Jetson; those tests carry
the ``hardware`` mark and run there with ``pytest -m hardware``, with
HAZARD_ZONE_ENGINE pointing at a built engine.
"""

import os
import threading

import numpy as np
import pytest

from hazard_zone.inference import (
    ENGINE_ENV,
    OUTPUT_ROWS,
    Engine,
    EngineShapes,
    Tensor,
    check_shapes,
    engine_path_from_env,
)

# The engine built on 2026-11-15: YOLOv8n, 640x384 input (docs/detector.md 1.13).
INPUT = Tensor("images", True, (1, 3, 384, 640), np.float32)
OUTPUT = Tensor("output0", False, (1, 84, 5040), np.float32)


# -- where the engine comes from -----------------------------------------------


def test_unset_is_an_error_naming_the_variable():
    with pytest.raises(ValueError, match=ENGINE_ENV):
        engine_path_from_env({})


def test_a_path_that_is_not_a_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        engine_path_from_env({ENGINE_ENV: str(tmp_path / "missing.engine")})
    with pytest.raises(ValueError, match="not a file"):
        engine_path_from_env({ENGINE_ENV: str(tmp_path)})  # a directory


def test_an_existing_file_is_accepted(tmp_path):
    engine = tmp_path / "yolov8n.engine"
    engine.write_bytes(b"not really an engine")
    assert engine_path_from_env({ENGINE_ENV: str(engine)}) == str(engine)


# -- which engines are accepted (3.6) ---------------------------------------------


def test_the_engine_in_use_is_read_not_assumed():
    assert check_shapes([INPUT, OUTPUT]) == EngineShapes(
        "images", "output0", 640, 384, 5040
    )


def test_a_square_engine_is_read_just_as_well():
    """The first engine built: 640x640, 8,400 candidates (1.3)."""
    square = [
        INPUT._replace(shape=(1, 3, 640, 640)),
        OUTPUT._replace(shape=(1, 84, 8400)),
    ]
    assert check_shapes(square) == EngineShapes("images", "output0", 640, 640, 8400)


def test_the_order_tensors_are_listed_in_does_not_matter():
    assert check_shapes([OUTPUT, INPUT]) == check_shapes([INPUT, OUTPUT])


@pytest.mark.parametrize(
    "tensors, found",
    [
        ([INPUT], "0"),
        ([INPUT, OUTPUT, OUTPUT._replace(name="extra")], "2"),
        ([INPUT, INPUT._replace(name="second"), OUTPUT], "2"),
    ],
    ids=["no output", "two outputs", "two inputs"],
)
def test_anything_but_one_input_and_one_output_is_refused(tensors, found):
    with pytest.raises(ValueError, match=found):
        check_shapes(tensors)


@pytest.mark.parametrize(
    "tensors, message",
    [
        ([INPUT._replace(shape=(2, 3, 384, 640)), OUTPUT], r"\(1, 3, H, W\)"),
        ([INPUT._replace(shape=(1, 1, 384, 640)), OUTPUT], r"\(1, 3, H, W\)"),
        ([INPUT, OUTPUT._replace(shape=(1, 85, 5040))], str(OUTPUT_ROWS)),
        ([INPUT, OUTPUT._replace(shape=(1, 5040, 84))], str(OUTPUT_ROWS)),
        ([INPUT._replace(shape=(-1, 3, 384, 640)), OUTPUT], "dynamic"),
        ([INPUT._replace(dtype=np.float16), OUTPUT], "float32"),
        ([INPUT, OUTPUT._replace(dtype=np.int8)], "float32"),
    ],
    ids=[
        "batch of two",
        "one channel",
        "85 rows: another model",
        "transposed output",
        "dynamic batch",
        "half-precision input",
        "integer output",
    ],
)
def test_an_engine_this_code_cannot_read_is_refused(tensors, message):
    """Refused at start, rather than producing boxes that mean nothing."""
    with pytest.raises(ValueError, match=message):
        check_shapes(tensors)


def test_an_engine_does_nothing_until_opened():
    engine = Engine("/nowhere/yolov8n.engine")
    assert engine.shapes is None
    with pytest.raises(RuntimeError, match="not open"):
        engine.infer()
    with pytest.raises(RuntimeError, match="not open"):
        _ = engine.input
    engine.close()  # safe on an engine never opened


# -- on the device ------------------------------------------------------------------


@pytest.fixture
def opened():
    path = os.environ.get(ENGINE_ENV)
    if not path:
        pytest.skip(f"{ENGINE_ENV} is not set")
    engine = Engine(path)
    engine.open()
    yield engine
    engine.close()


@pytest.mark.hardware
def test_the_engine_runs_and_returns_its_declared_shape(opened):
    shapes = opened.shapes
    assert opened.input.shape == (1, 3, shapes.height, shapes.width)
    output = opened.infer()
    assert output.shape == (1, OUTPUT_ROWS, shapes.candidates)
    assert np.isfinite(output).all()


@pytest.mark.hardware
def test_the_same_input_gives_the_same_output(opened):
    opened.input[:] = 0.5
    first = opened.infer().copy()
    second = opened.infer()
    assert np.array_equal(first, second)


@pytest.mark.hardware
def test_inferring_from_another_thread_is_refused(opened):
    """A CUDA context is current in one thread only (3.3)."""
    errors = []

    def other():
        try:
            opened.infer()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join(5.0)
    assert errors and "thread" in errors[0]
