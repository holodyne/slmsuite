"""Tests for :mod:`slmsuite._progress`: the ``tqdm`` stand-in and :func:`progress_bar`."""

import threading

import pytest
import tqdm.std

import slmsuite
from slmsuite import progress_bar
from slmsuite.hardware.cameraslms import _settle


def test_modules_share_the_package_tqdm():
    """Every loop imports ``tqdm`` from the package, so one hook covers them all."""
    assert _settle.tqdm is slmsuite.tqdm


def test_tqdm(subtests):
    """Test the ``tqdm`` dispatch: a real bar without a handler, a reporter with one."""
    with subtests.test("no handler on this thread gives a real tqdm bar"):
        bar = slmsuite.tqdm(range(2), desc="plain", disable=True)
        assert isinstance(bar, tqdm.std.tqdm)
        assert list(bar) == [0, 1]

    with subtests.test("wrapping an iterable reports a start, each step, and a close"):
        events = []
        with progress_bar(lambda event: events.append(event) or False):
            assert list(slmsuite.tqdm(range(3), desc="settling")) == [0, 1, 2]
        assert [event["kind"] for event in events] == [
            "start",
            "update",
            "update",
            "update",
            "close",
        ]
        assert events[0]["desc"] == "settling"
        assert events[1]["total"] == 3
        assert events[-2]["n"] == 3

    with subtests.test("the update() form reports too"):
        events = []
        with progress_bar(lambda event: events.append(event) or False):
            bar = slmsuite.tqdm(total=2, desc="iterating")
            bar.update()
            bar.update()
            bar.close()
        assert [event["n"] for event in events] == [0, 1, 2, 2]
        assert events[-1]["kind"] == "close"

    with subtests.test("an iterable without a length reports an unknown total"):
        events = []
        with progress_bar(lambda event: events.append(event) or False):
            assert list(slmsuite.tqdm(iter(range(2)))) == [0, 1]
        assert {event["total"] for event in events} == {None}
        assert events[-1]["n"] == 2

    with subtests.test("set_description reports the new description"):
        events = []
        with progress_bar(lambda event: events.append(event) or False):
            bar = slmsuite.tqdm(total=1, desc="old")
            bar.set_description("new")
            bar.close()
        assert [(event["kind"], event["desc"]) for event in events] == [
            ("start", "old"),
            ("update", "new"),
            ("close", "new"),
        ]

    with subtests.test("a truthy return cancels the loop where it stands"):
        seen = []
        with progress_bar(lambda event: event["kind"] == "update" and event["n"] >= 2):
            with pytest.raises(KeyboardInterrupt):
                seen.extend(slmsuite.tqdm(range(100), desc="long"))
        assert seen == [0, 1]

    with subtests.test("a loop left by break or an exception still reports a close"):
        events = []
        with progress_bar(lambda event: events.append(event) or False):
            for _ in slmsuite.tqdm(range(5)):
                break
            with pytest.raises(RuntimeError):
                for _ in slmsuite.tqdm(range(5)):
                    raise RuntimeError
        assert [event["kind"] for event in events] == ["start", "close"] * 2

    with subtests.test("a truthy return on close does not raise after the loop finished"):
        with progress_bar(lambda event: event["kind"] == "close"):
            assert list(slmsuite.tqdm(range(2))) == [0, 1]

    with subtests.test("a disabled bar stays a real tqdm bar"):
        with progress_bar(lambda event: True):
            assert isinstance(slmsuite.tqdm(range(1), disable=True), tqdm.std.tqdm)


def test_progress_bar(subtests):
    """Test :func:`progress_bar` scoping: per thread, and restored on exit."""
    with subtests.test("handlers are per thread"):
        mine, theirs = [], []

        def other():
            with progress_bar(lambda event: theirs.append(event) or False):
                list(slmsuite.tqdm(range(2), desc="theirs"))

        with progress_bar(lambda event: mine.append(event) or False):
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
            list(slmsuite.tqdm(range(2), desc="mine"))
        assert {event["desc"] for event in mine} == {"mine"}
        assert {event["desc"] for event in theirs} == {"theirs"}

    with subtests.test("nested blocks restore the outer handler"):
        outer, inner = [], []
        with progress_bar(lambda event: outer.append(event) or False):
            with progress_bar(lambda event: inner.append(event) or False):
                list(slmsuite.tqdm(range(1), desc="inner"))
            list(slmsuite.tqdm(range(1), desc="outer"))
        assert {event["desc"] for event in inner} == {"inner"}
        assert {event["desc"] for event in outer} == {"outer"}

    with subtests.test("the thread is left without a handler afterwards"):
        assert isinstance(slmsuite.tqdm(range(1), disable=True), tqdm.std.tqdm)
