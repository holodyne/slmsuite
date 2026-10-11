"""Progress reporting for slmsuite's long loops.

Every loop draws its bar with ``from slmsuite import tqdm``, which resolves here. By default
it is plain :class:`tqdm.tqdm`. A program driving slmsuite (a GUI, a scheduler, a test)
wants each step as data and a way to stop the loop instead: :class:`progress_to` routes the
steps on its thread to a handler, and a truthy return raises :class:`ProgressCancelled`
inside the loop. Other threads keep their bars, and a new loop upstream is covered without
registering anything.
"""
import os
import threading

if os.environ.get("SLMSUITE_TQDM_ASCII", "").lower() not in ("", "0"):
    from tqdm import tqdm as _tqdm         # Plain-text bars, e.g. for docs-ready notebooks.
else:
    from tqdm.auto import tqdm as _tqdm

_local = threading.local()


class ProgressCancelled(Exception):
    """Raised inside a loop whose :func:`progress_to` handler returned a truthy value."""


class _Reporter:
    """A ``tqdm`` stand-in that reports each step to a handler instead of drawing a bar."""

    def __init__(self, iterable=None, desc=None, total=None, handler=None, **kwargs):
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
        if self.iterable is None:
            return
        for item in self.iterable:
            yield item
            self.n += 1
            self._report("update")
        self.close()

    def __len__(self):
        return self.total or 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

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

    set_description_str = set_description

    def write(self, message, **kwargs):
        self.desc = str(message)
        self._report("update")

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def _report(self, kind):
        event = {"kind": kind, "desc": self.desc, "n": self.n, "total": self.total}
        if self._handler(event):
            raise ProgressCancelled(f"'{self.desc}' was cancelled." if self.desc else "Cancelled.")


def tqdm(iterable=None, *args, **kwargs):
    """The progress bar every slmsuite loop uses.

    Plain :class:`tqdm.tqdm`, unless :class:`progress_to` is active on the calling thread,
    in which case each step is reported to that handler instead. Setting the environment
    variable ``SLMSUITE_TQDM_ASCII`` to a nonzero value forces plain-text bars.
    """
    handler = getattr(_local, "handler", None)
    if handler is None:
        return _tqdm(iterable, *args, **kwargs)
    return _Reporter(iterable, *args, handler=handler, **kwargs)


class progress_to:
    """Route the progress of slmsuite loops on this thread to ``handler`` for a ``with`` block.

    Lets a program (a GUI, a scheduler) follow and stop a calibration or an optimization
    without parsing terminal output. Other threads are unaffected, so a progress bar in
    the notebook stays a progress bar.

    Parameters
    ----------
    handler : callable
        ``handler(event) -> bool``. ``event`` is a dict with ``"kind"`` (``"start"``,
        ``"update"`` or ``"close"``), ``"desc"``, ``"n"`` and ``"total"`` (``None`` when
        unknown). A truthy return raises :class:`ProgressCancelled` inside the loop, for
        the caller to catch; a cancelled calibration is not rolled back.

    Examples
    --------
    .. code-block:: python

        with slmsuite.progress_to(lambda event: print(event["n"], event["total"])):
            fs.wavefront_calibrate_superpixel()
    """

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        self._previous = getattr(_local, "handler", None)
        _local.handler = self.handler
        return self

    def __exit__(self, *exc_info):
        _local.handler = self._previous
        return False
