"""Matplotlib plot interception for headless and programmatic use."""
import contextlib
import pathlib

_fig_counts = {}        # name/prefix → count; tracks per-context numbering in save mode
_save_dir = None
_prefix = ""
_current_handler = None # set by configure_plotting(); None means call real plt.show()


def _slmsuite_plt_show(name=None, *args, **kwargs):
    """Route an internal ``plt.show()`` call through :func:`configure_plotting`.

    Does not touch global ``plt.show``; other packages are unaffected.

    Parameters
    ----------
    name : str, optional
        Label for this call site.  Used by ``"save"`` mode as the filename stem.
    *args, **kwargs
        Forwarded to ``plt.show()`` when no handler is active.
    """
    import matplotlib.pyplot as plt
    if _current_handler is not None:
        _current_handler(name=name, *args, **kwargs)
    else:
        plt.show(*args, **kwargs)


def configure_plotting(mode="show", save_dir=None, prefix="", headless=False):
    """Configure how slmsuite plots are displayed or saved.

    Affects :func:`_slmsuite_plt_show` calls only; global ``plt.show`` is
    never patched.

    Parameters
    ----------
    mode : str or callable
        One of:

        ``"show"``
            Pass through to ``plt.show()`` (default).
        ``"suppress"``
            Close all figures silently.
        ``"save"``
            Save figures to ``save_dir`` as PNGs, then close.
        callable
            Called in place of ``plt.show()``.

    save_dir : str or path-like, optional
        Output directory for ``"save"`` mode.  Created if absent.
    prefix : str, optional
        Fallback filename stem for ``"save"`` mode when no ``name`` is passed
        to :func:`_slmsuite_plt_show`.  The same prefix shares a counter
        across calls; a new prefix resets to 1.
    headless : bool, optional
        Switch to the ``"Agg"`` backend and call ``plt.ioff()``.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    global _fig_counts, _save_dir, _prefix, _current_handler

    if headless:
        matplotlib.use("Agg")
        plt.ioff()

    if mode == "show":
        _current_handler = None

    elif mode == "suppress":
        def _suppress(name=None, **kwargs):
            plt.close("all")
        _current_handler = _suppress

    elif mode == "save":
        if save_dir is None:
            raise ValueError("save_dir is required for mode='save'")
        _save_dir = pathlib.Path(save_dir)
        _save_dir.mkdir(parents=True, exist_ok=True)
        _prefix = prefix

        def _save(name=None, **kwargs):
            from slmsuite._logging import logger
            ctx = name or _prefix
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figs:
                _fig_counts[ctx] = _fig_counts.get(ctx, 0) + 1
                fname = (f"{ctx}_fig{_fig_counts[ctx]}.png" if ctx
                         else f"fig{_fig_counts[ctx]}.png")
                path = _save_dir / fname
                fig.savefig(path, dpi=150, bbox_inches="tight")
                logger.debug("Saved plot: %s", path)
            plt.close("all")

        _current_handler = _save

    elif callable(mode):
        _current_handler = mode

    else:
        raise ValueError(f"Unknown mode: {mode!r}")


@contextlib.contextmanager
def capture_plots():
    """Intercept :func:`_slmsuite_plt_show` and collect the resulting figures.

    The previously-active handler is restored on exit.  Captured figures are
    not closed automatically; call ``plt.close('all')`` when done.

    Yields
    ------
    list of matplotlib.figure.Figure
        Figures that would have been shown, in order of capture.

    Examples
    --------
    >>> with slmsuite.capture_plots() as figs:
    ...     hologram.plot_farfield()
    >>> plt.close('all')
    """
    global _current_handler

    prev_handler = _current_handler
    captured = []

    def _capture(name=None, **kwargs):
        import matplotlib.pyplot as plt
        captured.extend(plt.figure(n) for n in plt.get_fignums())

    _current_handler = _capture
    try:
        yield captured
    finally:
        _current_handler = prev_handler
