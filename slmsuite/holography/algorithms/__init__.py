"""
GPU-accelerated holography algorithms.

This module is currently focused on
`Gerchberg-Saxton (GS) <http://www.u.arizona.edu/~ppoon/GerchbergandSaxton1972.pdf>`_
iterative Fourier transform phase retrieval algorithms
via the :class:`~slmsuite.holography.algorithms.Hologram` class;
however, support for complex holography and other algorithms
(e.g. `gradient descent algorithms <https://doi.org/10.1364/AO.21.002758>`_)
is also planned. Additionally, so-called Weighted Gerchberg-Saxton (WGS) algorithms for hologram
generation with or without closed-loop camera feedback are supported, especially for
the `generation of optical focus arrays <https://doi.org/10.1364/OL.44.003178>`_,
a subset of general image formation. We also support `Mixed Region Amplitude Freedom (MRAF)
<https://doi.org/10.1007/s10043-018-0456-x>`_ feedback.

Tip
~~~
This module makes use of the GPU-accelerated computing library :mod:`cupy`
(`GitHub <https://docs.cupy.dev/en/stable/reference/index.html>`_).
If :mod:`cupy` is not supported, then :mod:`numpy` is used as a fallback, though
CPU alone is significantly slower. Using :mod:`cupy` is highly encouraged.
The only algorithm that has no CPU fallback is :class:`CompressedSpotHologram`
in a Zernike basis beyond 2D or 3D spots. Other cases can use the CPU (however slowly).

Note
~~~~
Internally, algorithms is split into several hidden files
to enhance clarity and reduce file length.

- ``_header.py`` : The common imports for all the files.
- ``_stats.py`` : Statistics and plotting common to all hologram classes.
- ``_hologram.py`` : The core file. Contains the actual algorithms (:class:`Hologram`).
- ``_feedback.py`` : Infrastructure for image feedback (:class:`FeedbackHologram`).
- ``_spots.py`` : Infrastructure for spot-specific holography
  (:class:`SpotHologram`, :class:`CompressedSpotHologram`).
"""

from slmsuite.holography.algorithms._feedback import FeedbackHologram as _FeedbackHologram
from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._hologram import Hologram as _Hologram
from slmsuite.holography.algorithms._multiplane import MultiplaneHologram as _MultiplaneHologram
from slmsuite.holography.algorithms._spots import (
    CompressedSpotHologram as _CompressedSpotHologram,
    SpotHologram as _SpotHologram,
)


# Hack to get automodule to put the classes in the correct location.
class Hologram(_Hologram):
    pass


class FeedbackHologram(_FeedbackHologram):
    pass


class SpotHologram(_SpotHologram):
    pass


class CompressedSpotHologram(_CompressedSpotHologram):
    pass


class MultiplaneHologram(_MultiplaneHologram):
    pass


# Hack to get the class and attribute docs to work.
Hologram.__doc__ = _Hologram.__doc__
FeedbackHologram.__doc__ = _FeedbackHologram.__doc__
SpotHologram.__doc__ = _SpotHologram.__doc__
CompressedSpotHologram.__doc__ = _CompressedSpotHologram.__doc__
MultiplaneHologram.__doc__ = _MultiplaneHologram.__doc__
