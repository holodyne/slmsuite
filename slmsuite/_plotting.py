"""Matplotlib plot interception for headless and programmatic use."""
import pathlib

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
        _current_handler(*args, name=name, **kwargs)
    else:
        plt.show(*args, **kwargs)


def configure_plotting(
    mode="show",
    save_dir=None,
    headless=False,
    extension="png",
    savefig_kwargs=None
):
    """Configure how slmsuite plots are displayed or saved.

    Parameters
    ----------
    mode : str or callable
        One of:

        ``"show"``
            Pass through to ``plt.show()`` (default).
        ``"suppress"``
            Close all figures silently.
        ``"save"``
            Save figures to ``save_dir`` (using ``extension`` and
            ``savefig_kwargs``), then close.
        callable
            Called in place of ``plt.show()``, though it must also accept ``name=``
            which is used to identify the plot.

    save_dir : str or path-like, optional
        Output directory for ``"save"`` mode.  Created if absent.  The filename
        stem comes from the ``name`` passed to :func:`_slmsuite_plt_show`
        (falling back to ``"fig"``).
    headless : bool, optional
        Switch to the ``"Agg"`` backend and call ``plt.ioff()``.  Selecting any
        other matplotlib backend is left to the user (e.g. the ``MPLBACKEND``
        environment variable or a Jupyter ``%matplotlib`` magic).
    extension : str, optional
        File extension (and thus output format) for ``"save"`` mode, e.g.
        ``"png"`` (default), ``"pdf"``, or ``"svg"``.  Ignored for other modes.
    savefig_kwargs : dict, optional
        Extra keyword arguments forwarded to ``Figure.savefig`` in ``"save"``
        mode, merged over the defaults ``{"dpi": 150, "bbox_inches": "tight"}``
        (user keys take precedence).  Ignored for other modes.
    """
    import matplotlib
    if headless:
        try:
            import matplotlib.pyplot as plt
            plt.switch_backend("Agg")
        except Exception:
            matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    global _current_handler

    if headless:
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
        save_dir = pathlib.Path(save_dir)
        merged = {"dpi": 150, "bbox_inches": "tight", **(savefig_kwargs or {})}

        def _save(name=None, **kwargs):
            from slmsuite._logging import make_logger
            from slmsuite.holography.analysis.files import generate_path
            logger = make_logger("plotting")
            ctx = name or "fig"
            figs = [plt.figure(n) for n in plt.get_fignums()]
            for n, fig in enumerate(figs):
                ctx_ = ctx if len(figs) == 1 else f"{ctx}_{n}"
                path = generate_path(save_dir, ctx_, extension=extension)
                fig.savefig(path, **merged)
                logger.debug("Saved plot: %s", path)
            plt.close("all")

        _current_handler = _save

    elif callable(mode):
        _current_handler = mode

    else:
        raise ValueError(f"Unknown mode: {mode!r}")
