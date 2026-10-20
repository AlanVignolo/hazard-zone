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

from hazard_zone.capture import Frame, LatestFrame

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
