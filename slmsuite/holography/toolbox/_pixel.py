"""
Element pattern of one SLM pixel: the farfield envelope that multiplies the
discrete Fourier transform of the pixel phases.
"""

import copy

import numpy as np

from slmsuite.misc.xp import as_numpy, get_array_module

# Fractions of the on-axis power at which the element pattern is contoured in plots.
_PLOT_LEVELS = (0.5, 0.75, 0.9)


class Pixel:
    r"""
    The response of one pixel of an SLM, as its farfield amplitude :meth:`amplitude`.

    The farfield of a pixelated SLM is the discrete Fourier transform of its pixel
    phases multiplied by the Fourier transform of a single pixel, the element pattern
    :math:`A(\vec{k})`. This class models a flat rectangular reflector that covers a
    fraction :math:`f_x \times f_y` of the pixel pitch :math:`p_x \times p_y` with
    power reflectance :math:`R`, surrounded by an absorbing gap:

    .. math:: A(\vec{k}) = \sqrt{R}\, f_x f_y\,
        \text{sinc}(f_x p_x k_x)\, \text{sinc}(f_y p_y k_y),

    where :math:`\text{sinc}(x) = \sin(\pi x) / (\pi x)`. Summed over every Nyquist zone,
    :math:`|A|^2` carries :math:`R f_x f_y` of the incident power.

    Tip
    ~~~
    A pixel with another element pattern is a subclass that overrides
    :meth:`amplitude`; nothing else reads :attr:`fill` or :attr:`reflectance`.

    Attributes
    ----------
    pitch : numpy.ndarray
        Pixel pitch ``(x, y)`` in wavelengths, as
        :attr:`~slmsuite.hardware.slms.slm.SLM.pitch`.
    fill : numpy.ndarray
        Linear fill ``(x, y)``: the active width over the pitch along each axis.
    reflectance : float
        Power reflectance of the active area.
    """

    def __init__(self, pitch, fill=1, gap=None, reflectance=1):
        r"""
        Parameters
        ----------
        pitch : float OR (float, float)
            See :attr:`pitch`. A scalar is a square pixel.
        fill : float OR (float, float)
            A pair is the linear fill ``(x, y)``. A scalar is the area fill factor that
            vendors quote, converted to a linear fill per axis by assuming a gap of the
            same width along both axes (the square root for a square pixel).
        gap : float OR (float, float) OR None
            Width of the gap between active areas ``(x, y)`` in wavelengths. Takes
            precedence over ``fill``.
        reflectance : float
            See :attr:`reflectance`.

        Raises
        ------
        ValueError
            If the linear fill is outside :math:`(0, 1]` or the reflectance is outside
            :math:`[0, 1]`.
        """
        self._pitch = np.broadcast_to(np.asarray(pitch, dtype=float), (2,)).copy()

        if gap is None and np.ndim(fill) == 0:
            # Uniform gap g with (px - g) (py - g) = fill px py.
            total = np.sum(self._pitch)
            gap = (total - np.sqrt(total**2 - 4 * np.prod(self._pitch) * (1 - fill))) / 2
        if gap is not None:
            fill = 1 - np.asarray(gap, dtype=float) / self._pitch
        self._fill = np.broadcast_to(np.asarray(fill, dtype=float), (2,)).copy()
        self._reflectance = float(reflectance)

        if not np.all((self._fill > 0) & (self._fill <= 1)):
            raise ValueError(f"Linear fill {self._fill} must be in (0, 1].")
        if not 0 <= self._reflectance <= 1:
            raise ValueError(f"Reflectance {self._reflectance} must be in [0, 1].")

    @property
    def pitch(self):
        return self._pitch

    @property
    def fill(self):
        return self._fill

    @property
    def reflectance(self):
        return self._reflectance

    def amplitude(self, kxy):
        r"""
        The element pattern :math:`A(\vec{k})`.

        Parameters
        ----------
        kxy : array_like
            Spatial frequencies of the displayed pattern in the ``"kxy"`` basis, of shape
            ``(2, ...)``. This is the outgoing :math:`\vec{k}` less that of the incident
            beam, so it is unbounded: frequencies past the Nyquist limit address the
            higher zones.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            Amplitude of shape ``kxy.shape[1:]`` on the backend of ``kxy``. A subclass
            may return a complex array.
        """
        xp = get_array_module(kxy)
        kxy = xp.asarray(kxy)
        if not xp.issubdtype(kxy.dtype, xp.floating):
            kxy = kxy.astype(float)
        (ax, ay) = self._fill * self._pitch
        return (
            float(np.sqrt(self._reflectance) * np.prod(self._fill))
            * xp.sinc(float(ax) * kxy[0])
            * xp.sinc(float(ay) * kxy[1])
        )

    def power(self, kxy):
        r"""
        The power of the element pattern, :math:`|A(\vec{k})|^2`. See :meth:`amplitude`.
        """
        return abs(self.amplitude(kxy)) ** 2

    def _plot(self, ax, power, extent):
        """
        Contours an image of :meth:`power` onto a farfield plot, labeled by the fraction
        of the on-axis power, leaving the limits of ``ax`` as they are.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        power : numpy.ndarray OR cupy.ndarray
            :meth:`power` on the grid of the plotted image.
        extent : (float, float, float, float)
            ``(left, right, bottom, top)`` of that image, as
            :meth:`matplotlib.axes.Axes.imshow` takes it.
        """
        (h, w) = power.shape
        (sy, sx) = (max(1, h // 256), max(1, w // 256))
        power = as_numpy(power[::sy, ::sx])

        peak = float(self.power(np.zeros(2))) or float(power.max())
        levels = [f * peak for f in _PLOT_LEVELS if power.min() < f * peak < power.max()]
        if len(levels) == 0:
            return

        (left, right, bottom, top) = extent
        x = left + (np.arange(w)[::sx] + 0.5) * (right - left) / w
        y = top + (np.arange(h)[::sy] + 0.5) * (bottom - top) / h

        limits = (ax.get_xlim(), ax.get_ylim())
        contours = ax.contour(x, y, power, levels=levels, colors="w", linewidths=0.8)
        ax.clabel(
            contours, fmt=lambda level: f"{level / peak:.0%}", fontsize="x-small", inline=False
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    @classmethod
    def resolve(cls, slm, pixel=None):
        """
        Resolve a ``pixel`` argument into a :class:`Pixel` on the pitch of ``slm``.

        Parameters
        ----------
        slm : :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`~slmsuite.hardware.cameraslms.CameraSLM`
        pixel : Pixel OR float OR (float, float) OR dict OR None
            ``None`` stays ``None``: the pixel is not modeled. A :class:`Pixel` of any
            subclass is copied onto the pitch of ``slm``. A number or pair is ``fill``.
            A dict holds the keyword arguments of the constructor, where ``gap_um`` in
            microns may stand in for ``gap``.

        Returns
        -------
        Pixel OR None
        """
        if pixel is None:
            return None

        slm = getattr(slm, "slm", slm)
        pitch = np.array(slm.pitch, dtype=float)

        if isinstance(pixel, Pixel):
            pixel = copy.copy(pixel)
            pixel._pitch = pitch
            return pixel

        if isinstance(pixel, dict):
            kwargs = dict(pixel)
            if "gap_um" in kwargs:
                kwargs["gap"] = np.asarray(kwargs.pop("gap_um"), dtype=float) / slm.wav_um
            return cls(pitch, **kwargs)

        return cls(pitch, fill=pixel)

    def pickle(self, attributes=True, metadata=False):
        """
        Return an h5-serializable dict that :meth:`resolve` turns back into a
        :class:`Pixel` of this fill and reflectance; a subclass with parameters of its own
        overrides this. Compatible with the pickling recursion used by
        :meth:`SLM.save() <slmsuite.hardware.slms.slm.SLM.save>`. ``attributes`` and
        ``metadata`` are accepted for that interface and ignored.
        """
        return {"fill": self._fill, "reflectance": self._reflectance}

    def __repr__(self):
        return f"{type(self).__name__}(fill={self._fill!r}, reflectance={self._reflectance!r})"
