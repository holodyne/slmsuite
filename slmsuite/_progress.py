"""
Progress reporting for slmsuite's loops.

Every loop draws its bar with ``from slmsuite import tqdm``, which resolves here to
``tqdm.auto.tqdm``. A program driving slmsuite (a GUI, a scheduler, a test) can instead
take each step as data: :func:`progress_bar` routes the steps on its thread to a handler,
and a truthy return raises :class:`KeyboardInterrupt` inside the loop. Other threads keep
their bars.
"""

import contextlib
import os
import threading

if os.environ.get("SLMSUITE_TQDM_ASCII", "").lower() not in ("", "0"):
    from tqdm import tqdm as _tqdm  # Plain-text bars, e.g. for docs-ready notebooks.
else:
    from tqdm.auto import tqdm as _tqdm

_local = threading.local()


class _Reporter:
    """A ``tqdm`` stand-in that reports each step to a handler instead of drawing a bar."""

    def __init__(self, handler, iterable=None, desc=None, total=None, *args, **kwargs):
        self.iterable = iterable
        self.desc = desc or ""
        self.n = 0
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total = total
        self._handler = handler
        self._closed = False
        self._report("start")

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()

    def update(self, n=1):
        self.n += int(n)
        self._report("update")

    def close(self):
        if not self._closed:
            self._closed = True
            self._report("close")

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ""
        self._report("update")

    def _report(self, kind):
        event = {"kind": kind, "desc": self.desc, "n": self.n, "total": self.total}
        if self._handler(event) and kind != "close":
            raise KeyboardInterrupt(f"'{self.desc}' was canceled." if self.desc else "Canceled.")


def tqdm(iterable=None, *args, **kwargs):
    """The progress bar every slmsuite loop uses.

    ``tqdm.auto.tqdm``, unless :func:`progress_bar` is active on the calling thread and the
    bar is not disabled, in which case each step is reported to that handler instead.
    Setting the environment variable ``SLMSUITE_TQDM_ASCII`` to a nonzero value forces
    plain-text bars.
    """
    handler = getattr(_local, "handler", None)
    if handler is None or kwargs.get("disable"):
        return _tqdm(iterable, *args, **kwargs)
    return _Reporter(handler, iterable, *args, **kwargs)


@contextlib.contextmanager
def progress_bar(handler):
    """Route the progress of slmsuite loops on this thread to ``handler`` for a ``with`` block.

    Lets a program (a GUI, a scheduler) follow and stop a calibration or an optimization
    without parsing terminal output. Other threads are unaffected, so a progress bar in
    the notebook stays a progress bar.

    Parameters
    ----------
    handler : callable
        ``handler(event) -> bool``. ``event`` is a dict with ``"kind"`` (``"start"``,
        ``"update"`` or ``"close"``), ``"desc"``, ``"n"`` and ``"total"`` (``None`` when
        unknown). A truthy return on a ``"start"`` or ``"update"`` raises
        :class:`KeyboardInterrupt` inside the loop, for the caller to catch; a canceled
        calibration is not rolled back.

    Examples
    --------
    .. code-block:: python

        with slmsuite.progress_bar(lambda event: print(event["n"], event["total"])):
            fs.wavefront_calibrate_superpixel()
    """
    previous = getattr(_local, "handler", None)
    _local.handler = handler
    try:
        yield
    finally:
        _local.handler = previous
