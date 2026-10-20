"""Tests for the latest-frame slot.

Concurrency tests cannot assert an exact sequence of events, because the
order in which threads run is the operating system's decision. They assert
invariants instead -- relations that must hold whatever that order turns out
to be -- and use events rather than sleeps to synchronise, so a loaded
machine cannot turn a passing test into a failing one.
"""

import threading
import time

import pytest

from hazard_zone.capture import (  # noqa: E402
    COMPRESSED_INPUTS,
    DECODERS_HW,
    DECODERS_SW,
    MEDIA_TYPES,
    Frame,
    LatestFrame,
    build_pipeline,
)

# --------------------------------------------------------------------------
# Single-threaded behaviour: the dropping policy
# --------------------------------------------------------------------------


def test_get_returns_what_was_put():
    slot = LatestFrame()
    slot.put("a")
    assert slot.get().image == "a"


def test_newer_frame_replaces_older():
    slot = LatestFrame()
    slot.put("old")
    slot.put("new")
    assert slot.get().image == "new"


def test_replacing_counts_a_drop():
    slot = LatestFrame()
    slot.put("a")
    assert slot.dropped == 0
    slot.put("b")
    assert slot.dropped == 1


def test_consuming_prevents_the_drop():
    """A frame someone took is not a frame that was lost."""
    slot = LatestFrame()
    slot.put("a")
    slot.get()
    slot.put("b")
    assert slot.dropped == 0


def test_many_unconsumed_frames_count_as_many_drops():
    slot = LatestFrame()
    for i in range(10):
        slot.put(i)
    assert slot.dropped == 9
    assert slot.get().image == 9


def test_slot_holds_only_one_frame():
    """The reason this class exists: latency cannot accumulate."""
    slot = LatestFrame()
    for i in range(100):
        slot.put(i)
    assert slot.get().image == 99
    assert slot.get(timeout=0) is None


def test_sequence_numbers_increment_from_one():
    slot = LatestFrame()
    assert slot.put("a") == 1
    assert slot.put("b") == 2
    assert slot.get().seq == 2


def test_sequence_counts_dropped_frames_too():
    """Gaps in the sequence a consumer sees are the frames it missed."""
    slot = LatestFrame()
    slot.put("a")
    slot.put("b")
    slot.put("c")
    assert slot.get().seq == 3
    assert slot.published == 3


def test_peek_leaves_the_frame_in_place():
    slot = LatestFrame()
    slot.put("a")
    assert slot.peek().image == "a"
    assert slot.get().image == "a"


def test_peeked_frame_is_still_dropped_if_replaced():
    """Looking is not consuming."""
    slot = LatestFrame()
    slot.put("a")
    slot.peek()
    slot.put("b")
    assert slot.dropped == 1


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def test_frame_carries_a_capture_time():
    before = time.monotonic()
    slot = LatestFrame()
    slot.put("a")
    after = time.monotonic()
    assert before <= slot.get().captured_at <= after


def test_explicit_capture_time_is_kept():
    """A source that reports its own timestamp is more accurate than now()."""
    slot = LatestFrame()
    slot.put("a", captured_at=1234.5)
    assert slot.get().captured_at == 1234.5


def test_age_measures_from_capture():
    frame = Frame("img", seq=1, captured_at=100.0)
    assert frame.age(now=100.25) == pytest.approx(0.25)


def test_age_defaults_to_now():
    slot = LatestFrame()
    slot.put("a")
    age = slot.get().age()
    assert 0 <= age < 1.0


def test_age_is_never_negative_under_a_clock_change():
    """monotonic() cannot go backwards, which is why it is used here.

    time.time() can, after an NTP correction or a timezone change, and a
    latency measured against it could come out negative.
    """
    slot = LatestFrame()
    for _ in range(50):
        slot.put("a")
        assert slot.get().age() >= 0


# --------------------------------------------------------------------------
# Waiting
# --------------------------------------------------------------------------


def test_get_with_zero_timeout_does_not_block():
    slot = LatestFrame()
    started = time.monotonic()
    assert slot.get(timeout=0) is None
    assert time.monotonic() - started < 0.5


def test_get_times_out_when_no_frame_arrives():
    slot = LatestFrame()
    started = time.monotonic()
    assert slot.get(timeout=0.05) is None
    assert time.monotonic() - started >= 0.04


def test_get_wakes_when_a_frame_arrives():
    """The consumer must not poll: it sleeps until notified."""
    slot = LatestFrame()
    started = threading.Event()

    def produce():
        started.wait(1.0)
        slot.put("late")

    thread = threading.Thread(target=produce)
    thread.start()
    started.set()
    frame = slot.get(timeout=1.0)
    thread.join(1.0)

    assert frame is not None
    assert frame.image == "late"


def test_waiting_consumer_returns_promptly():
    """Waiting must end when the frame arrives, not when the timeout expires."""
    slot = LatestFrame()
    threading.Timer(0.02, slot.put, args=("x",)).start()
    started = time.monotonic()
    frame = slot.get(timeout=5.0)
    elapsed = time.monotonic() - started
    assert frame is not None
    assert elapsed < 1.0, f"woke after {elapsed:.2f}s; was it polling?"


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


def test_close_marks_the_slot_closed():
    slot = LatestFrame()
    assert not slot.closed
    slot.close()
    assert slot.closed


def test_get_on_a_closed_empty_slot_returns_none():
    slot = LatestFrame()
    slot.close()
    assert slot.get(timeout=5.0) is None


def test_close_does_not_discard_a_pending_frame():
    """Shutting down must not throw away work already captured."""
    slot = LatestFrame()
    slot.put("a")
    slot.close()
    assert slot.get().image == "a"


def test_close_wakes_a_waiting_consumer():
    """Without this a consumer would wait forever for a frame never coming."""
    slot = LatestFrame()
    threading.Timer(0.02, slot.close).start()
    started = time.monotonic()
    assert slot.get(timeout=5.0) is None
    assert time.monotonic() - started < 1.0


def test_close_is_idempotent():
    slot = LatestFrame()
    slot.close()
    slot.close()
    assert slot.closed


# --------------------------------------------------------------------------
# Concurrency: invariants that must hold in any interleaving
# --------------------------------------------------------------------------


def test_counts_balance_under_concurrent_access():
    """published == dropped + consumed + still held, always.

    The exact split depends on thread scheduling and is not asserted; the
    equation is what must hold no matter how the threads interleave.
    """
    slot = LatestFrame()
    total = 2000
    consumed = []
    stop = threading.Event()

    def produce():
        for i in range(total):
            slot.put(i)
        stop.set()

    def consume():
        while not stop.is_set() or slot.peek() is not None:
            frame = slot.get(timeout=0.01)
            if frame is not None:
                consumed.append(frame.seq)

    producer = threading.Thread(target=produce)
    consumer = threading.Thread(target=consume)
    producer.start()
    consumer.start()
    producer.join(10.0)
    consumer.join(10.0)

    held = 1 if slot.peek() is not None else 0
    assert slot.published == total
    assert slot.dropped + len(consumed) + held == total


def test_no_frame_is_consumed_twice():
    """Sequence numbers a consumer sees must be strictly increasing."""
    slot = LatestFrame()
    seen = []
    stop = threading.Event()

    def produce():
        for i in range(1000):
            slot.put(i)
        stop.set()

    def consume():
        while not stop.is_set() or slot.peek() is not None:
            frame = slot.get(timeout=0.01)
            if frame is not None:
                seen.append(frame.seq)

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert seen == sorted(seen)
    assert len(seen) == len(set(seen))


def test_drop_counter_is_not_lost_under_contention():
    """A counter incremented without a lock loses updates; this checks it.

    Two producers racing on the same slot would, with an unprotected
    counter, report fewer drops than actually happened.
    """
    slot = LatestFrame()
    per_thread = 1000

    def produce():
        for i in range(per_thread):
            slot.put(i)

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    published = 4 * per_thread
    assert slot.published == published
    # Nobody consumed, so every frame but the last was overwritten.
    assert slot.dropped == published - 1


def test_stats_are_internally_consistent():
    """The four numbers must describe one instant, not four.

    Read without a single lock, frames arriving between the reads would make
    the counts contradict each other.
    """
    slot = LatestFrame()
    stop = threading.Event()

    def produce():
        while not stop.is_set():
            slot.put("x")

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        for _ in range(500):
            s = slot.stats()
            held = 1 if s["published"] > s["dropped"] + s["consumed"] else 0
            assert s["dropped"] + s["consumed"] + held == s["published"], s
    finally:
        stop.set()
        thread.join(5.0)


def test_consumer_always_gets_the_newest_available_frame():
    """The point of the design: never an old frame when a newer one exists."""
    slot = LatestFrame()
    for i in range(50):
        slot.put(i)
        slot.put(i + 1000)
        frame = slot.get(timeout=0.1)
        assert frame.image == i + 1000


# --------------------------------------------------------------------------
# build_pipeline
#
# These assert exact strings, unlike the diagnostic-message tests elsewhere.
# A launch string is a contract with another program: "vide/x-raw" is valid
# Python and meaningless to GStreamer, and the failure surfaces only on the
# device, as a negotiation error that names no cause.
# --------------------------------------------------------------------------


def test_software_pipeline_is_exactly_what_was_measured():
    """The 44.6 fps path, verbatim."""
    assert build_pipeline("/dev/video0", 1280, 720, 60).launch == (
        "v4l2src device=/dev/video0 io-mode=2"
        " ! image/jpeg,width=1280,height=720,framerate=60/1"
        " ! jpegdec"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink name=sink drop=true max-buffers=1 sync=false"
    )


def test_accelerated_pipeline_uses_two_converters():
    """nvvidconv cannot change colour format and memory space in one step."""
    launch = build_pipeline("/dev/video0", 1280, 720, 60, accelerated=True).launch
    assert launch == (
        "v4l2src device=/dev/video0 io-mode=2"
        " ! image/jpeg,width=1280,height=720,framerate=60/1"
        " ! nvjpegdec"
        " ! nvvidconv"
        " ! video/x-raw(memory:NVMM),format=NV12"
        " ! nvvidconv"
        " ! video/x-raw,format=BGRx"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink name=sink drop=true max-buffers=1 sync=false"
    )


def test_raw_format_gets_no_decoder():
    launch = build_pipeline("/dev/video0", 640, 480, 30, fourcc="YUYV").launch
    assert "video/x-raw,format=YUYV,width=640,height=480,framerate=30/1" in launch
    assert "jpegdec" not in launch


def test_media_type_matches_the_fourcc():
    assert "image/jpeg" in build_pipeline("/dev/video0", 640, 480, 30, "MJPG").launch
    assert "video/x-h264" in build_pipeline("/dev/video0", 640, 480, 30, "H264").launch


def test_fourcc_is_normalised():
    """Real fourccs arrive space-padded and sometimes lower case."""
    for code in ("MJPG", "mjpg", "MJPG ", " MJPG"):
        assert "jpegdec" in build_pipeline("/dev/video0", 640, 480, 30, code).launch


def test_output_is_always_bgr():
    """Whatever the path, the consumer gets what OpenCV and the detector use."""
    for accel in (False, True):
        for code in ("MJPG", "YUYV", "H264"):
            launch = build_pipeline(
                "/dev/video0", 640, 480, 30, code, accelerated=accel
            ).launch
            assert launch.endswith(
                "video/x-raw,format=BGR"
                " ! appsink name=sink drop=true max-buffers=1 sync=false"
            ), (code, accel)


# --- the dropping policy is not optional ----------------------------------


def test_appsink_always_drops_and_holds_one():
    for accel in (False, True):
        for code in ("MJPG", "YUYV"):
            launch = build_pipeline(
                "/dev/video0", 640, 480, 30, code, accelerated=accel
            ).launch
            assert "drop=true" in launch
            assert "max-buffers=1" in launch


def test_appsink_does_not_sync_to_the_clock():
    """A live frame already exists; pacing it to the clock only adds latency."""
    assert "sync=false" in build_pipeline("/dev/video0", 640, 480, 30).launch


def test_sink_can_be_named_for_lookup():
    launch = build_pipeline("/dev/video0", 640, 480, 30, sink_name="cam0").launch
    assert "appsink name=cam0" in launch


def test_fakesink_for_measurement():
    launch = build_pipeline("/dev/video0", 640, 480, 30, sink="fakesink").launch
    assert launch.endswith("fakesink name=sink sync=false")
    assert "appsink" not in launch


# --- frame rate as an exact fraction --------------------------------------


def test_integer_rate_becomes_a_simple_fraction():
    assert "framerate=30/1" in build_pipeline("/dev/video0", 640, 480, 30).launch
    assert "framerate=60/1" in build_pipeline("/dev/video0", 640, 480, 60).launch


def test_float_integer_rate_does_not_become_a_decimal():
    assert "framerate=30/1" in build_pipeline("/dev/video0", 640, 480, 30.0).launch


def test_ntsc_rates_become_their_exact_ratio():
    """29.97 is really 30000/1001; the decimal would never match the camera."""
    assert (
        "framerate=30000/1001" in build_pipeline("/dev/video0", 640, 480, 29.97).launch
    )
    assert (
        "framerate=60000/1001" in build_pipeline("/dev/video0", 640, 480, 59.94).launch
    )


def test_rate_never_renders_as_a_decimal():
    for fps in (6, 15, 24, 25, 29.97, 30, 50, 59.94, 60, 90, 120, 120.101, 300):
        launch = build_pipeline("/dev/video0", 640, 480, fps).launch
        rate = launch.split("framerate=")[1].split(",")[0].split(" ")[0]
        assert "/" in rate and "." not in rate, (fps, rate)


# --- sources --------------------------------------------------------------


def test_file_source_replays_a_video():
    """Replay makes the whole system testable without a camera, repeatably."""
    launch = build_pipeline("clip.mp4", 1280, 720, 30, source="file").launch
    assert launch.startswith("filesrc location=clip.mp4 ! decodebin")
    assert "v4l2src" not in launch


def test_file_source_adds_no_decoder_of_its_own():
    """decodebin has already decoded, whatever the container held."""
    launch = build_pipeline("clip.mp4", 1280, 720, 30, "MJPG", source="file").launch
    assert "jpegdec" not in launch


def test_file_source_negotiates_no_capture_rate():
    """A file plays at its own rate; there is no device to negotiate with."""
    assert (
        "framerate="
        not in build_pipeline("clip.mp4", 1280, 720, 30, source="file").launch
    )


def test_rtsp_source_builds_a_depayloader_chain():
    launch = build_pipeline("rtsp://cam/stream", 1280, 720, 25, source="rtsp").launch
    assert launch.startswith("rtspsrc location=rtsp://cam/stream latency=0")
    assert "rtph264depay ! h264parse ! avdec_h264" in launch


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        build_pipeline("/dev/video0", 640, 480, 30, source="webcam")


# --- decoder selection ----------------------------------------------------


def test_software_and_hardware_pick_different_decoders():
    sw = build_pipeline("/dev/video0", 640, 480, 30, "MJPG").launch
    hw = build_pipeline("/dev/video0", 640, 480, 30, "MJPG", accelerated=True).launch
    assert "jpegdec" in sw and "nvjpegdec" not in sw
    assert "nvjpegdec" in hw


def test_every_media_type_has_decoders_on_both_paths():
    """A format the caps filter can declare must be decodable either way."""
    for code in MEDIA_TYPES:
        assert code in DECODERS_SW, code
        assert code in DECODERS_HW, code


def test_compressed_format_without_support_is_rejected():
    """An unsupported compressed format must fail, not pass as raw.

    Treating it as raw would emit video/x-raw,format=VP9, which GStreamer
    rejects on the device with a negotiation error naming no cause.
    """
    with pytest.raises(ValueError, match="VP9"):
        build_pipeline("/dev/video0", 640, 480, 30, "VP9 ")


def test_every_compressed_input_either_works_or_raises():
    """No compressed format may silently take the raw path."""
    for code in COMPRESSED_INPUTS:
        try:
            launch = build_pipeline("/dev/video0", 640, 480, 30, code).launch
        except ValueError:
            continue  # explicit rejection is the acceptable outcome
        assert f"video/x-raw,format={code}" not in launch, code


# --- the reasoning that travels with the pipeline -------------------------


def test_reason_names_the_decoder_used():
    assert "jpegdec" in build_pipeline("/dev/video0", 640, 480, 30).reason
    assert (
        "nvjpegdec"
        in build_pipeline("/dev/video0", 640, 480, 30, accelerated=True).reason
    )


def test_reason_says_when_no_decoder_is_needed():
    assert "no decoder" in build_pipeline("/dev/video0", 640, 480, 30, "YUYV").reason


def test_accelerated_flag_is_reported():
    assert not build_pipeline("/dev/video0", 640, 480, 30).accelerated
    assert build_pipeline("/dev/video0", 640, 480, 30, accelerated=True).accelerated


def test_pipeline_stringifies_to_its_launch_string():
    p = build_pipeline("/dev/video0", 640, 480, 30)
    assert str(p) == p.launch
