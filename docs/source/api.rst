*************
API Reference
*************

This page provides an auto-generated summary of |slmsuite|_'s API. You can
find the source on `GitHub <https://github.com/holodyne/slmsuite>`_.

|slmsuite|_ is divided into two modules:
algorithms and analysis in :mod:`~slmsuite.holography` and
connectivity to physical devices via :mod:`~slmsuite.hardware`.

Holography
==========

.. py:module:: slmsuite.holography

As a core functionality, |slmsuite|_ **optimizes** nearfield phase profiles
(applied to :mod:`~slmsuite.hardware.slms`) to produce desired farfield results
(measured by :mod:`~slmsuite.hardware.cameras`). These methods are provided in:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   algorithms

Additionally, quality-of-life support methods are provided,
divided into SLM- and camera- centric categories:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   toolbox
   analysis

Hardware
========

.. py:module:: slmsuite.hardware

A central concept of |slmsuite|_ is **experimental** holography.
Thus, we require interfaces to control the hardware used in experiment.
While some common hardware implementions are included, we welcome
`contributions <https://github.com/holodyne/slmsuite/blob/main/CONTRIBUTING.md>`_
to expand the scope and utility of the package!
Hardware is divided into two main categories:

.. currentmodule:: slmsuite.hardware
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   slms
   cameras

The unification of these two categories is stored in:

.. currentmodule:: slmsuite.hardware
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   cameraslms

We also support hosting cameras and SLMs on remote servers:

.. currentmodule:: slmsuite.hardware
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   remote

Package
=======

.. py:module:: slmsuite

|slmsuite|_ reports its progress through the standard :mod:`logging` module and shows
plots through :mod:`matplotlib`. Package-wide settings for both are provided at the
top level:

.. currentmodule:: slmsuite
.. autosummary::
   :toctree: _autosummary

   configure_logging
   get_log
   make_logger
   configure_plotting

With :mod:`cupy` installed, SLMs, cameras, and holograms keep their arrays on the GPU.
Helpers for moving data between :mod:`numpy` and :mod:`cupy` are provided in:

.. currentmodule:: slmsuite.misc
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   xp

API Formalism
=============

Data-Order Conventions
~~~~~~~~~~~~~~~~~~~~~~
:mod:`slmsuite` follows the ``shape = (h, w)`` and ``vector = (x, y)`` formalism adopted by
the :mod:`numpy` ecosystem. :mod:`numpy`, :mod:`scipy`, :mod:`matplotlib`, etc generally follow this
formalism. The ``shape`` and indexing of an array or image always uses the inverted ``(h, w)`` form,
but other functions such as ``numpy.meshgrid(x, y)`` (default), ``scipy.odr.Data(x, y)``, or
``matplotlib.pyplot.scatter(x, y)`` use the standard cartesian ``(x, y)`` form that is more familiar
to users. This is not ideal and causes confusion, but this is the formalism generally
adopted by the community.

We additionally adopt the convention that
a list of :math:`N` vectors with dimension :math:`D` is represented with shape
``(D, N)``. This is so that linear transformations can be done with a direct and easy
multiplication with a ``(D, D)`` matrix.
Stacks of :math:`W \times H` images, however,
are stored with the inverted shape ``(N, H, W)``.
These conventions are arguably consistent with the chosen ordering.

:math:`k`-Space Bases
~~~~~~~~~~~~~~~~~~~~~
This package uses a number of spatial bases or coordinate spaces to describe the
:math:`k`-space of the SLM. Some coordinate spaces are
directly used by the user (most often the camera basis ``"ij"`` used for feedback).
Other bases are less often used directly, but are important to how holograms are
optimized under the hood (esp. ``"knm"``, the coordinate space of optimization when
using discrete Fourier transforms). The following describes the bases that general
functions (e.g. :class:`~slmsuite.holography.algorithms.Hologram`) accept. Other units
for coordinate bases can be converted to these standard bases using
:meth:`~slmsuite.holography.toolbox.convert_vector()`.

.. list-table:: Bases used in :mod:`slmsuite`.
   :widths: 20 80
   :header-rows: 1

   * - Basis
     - Meaning
   * - ``"kxy"``
     - Normalized basis of the SLM's :math:`k`-space in normalized units.
       For small angles, this is equivalent to radians.
       Centered at ``(kx, ky) = (0, 0)``. This basis is what the SLM projects in angular
       space (which maps to the camera's image space via the Fourier transform
       implemented by the optical train separating the two).

       The edge of Fourier space **that is accessible to the SLM** corresponds to
       :math:`\pm\frac{\lambda}{2\Delta x}` radians, dependant on the pixel size
       :math:`\Delta x` of the SLM. For most SLMs and wavelengths, this angle
       corresponds to a few degrees at most.
       Thus, this edge is generally within the small angle approximation.

       The true edge of the hemisphere of Fourier space corresponds to when
       :math:`\sqrt{k_x^2 + k_y^2} = 1`.
       At this boundary,
       `numerical aperture <https://en.wikipedia.org/wiki/Numerical_aperture>`_ is 1.
       Note that the small angle approximation breaks down in this limit.
   * - ``"knm"``
     - Nyquist basis of the SLM's computational :math:`k`-space for a given
       discrete computational grid of shape ``shape``.
       Centered at ``(kn, km) = (shape[1]/2, shape[0]/2)``.
       ``"knm"`` is a discrete version of the continuous ``"kxy"``. This is
       important because holograms need to be stored in computer memory, a discrete
       medium with pixels, rather than being purely continuous. For instance, in
       :class:`~slmsuite.holography.algorithms.SpotHologram`,
       spots targeting specific continuous angles are rounded to
       the nearest discrete pixels of ``"knm"`` space in practice
       (though :class:`~slmsuite.holography.algorithms.CompressedSpotHologram`
       transcends this limitation).
       Then, this ``"knm"`` space image is handled as a
       standard image/array, and operations such as the discrete Fourier transform
       (instrumental for numerical hologram optimization) can be applied.

       The edge of ``"knm"`` space is bounded by zero and the extent of ``shape``.
       In ``"kxy"`` space, this edge corresponds to the Nyquist limit of the SLM
       and is strictly smaller than the full extent of Fourier space. Increasing the
       ``shape`` of ``"knm"`` space increases the resolution of the grid in Fourier
       space, as the edge of ``"knm"`` space is fixed by the SLM.
   * - ``"ij"``
     - Pixel basis of the image returned by
       :meth:`~slmsuite.hardware.cameras.camera.Camera.get_image()`.
       Centered at :attr:`~slmsuite.hardware.cameras.camera.Camera.center`,
       ``(i, j) = (cam.shape[1]/2, cam.shape[0]/2)``.
       Is in the image space of the camera.

       This is the frame the user sees, so it follows the camera's current
       window of interest (WOI), binning, and orientation (flips and 90 degree rotations).
       The same physical point has different ``"ij"`` coordinates after any of these
       change.

       The bounds of pixel space may be larger or smaller than Fourier or Nyquist space,
       depending upon the imaging optics that separate the camera and SLM.
   * - ``"ijraw"``
     - Pixel basis of the physical camera sensor: full frame, unbinned, and without the
       orientation transform. Its origin is the corner of the sensor.

       Unlike ``"ij"``, this basis does not change with the camera's WOI, binning, or
       orientation. Calibrations are stored in it (e.g. the
       :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``"fourier"`` entry),
       so a calibration stays valid when the camera is cropped, binned, or rotated.
       :meth:`~slmsuite.holography.toolbox.convert_vector()` converts between ``"ijraw"``
       and ``"ij"`` given the camera's current settings.

See the first tip in :class:`~slmsuite.holography.algorithms.Hologram`
to learn more about ``"kxy"`` and ``"knm"`` space.

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/holodyne/slmsuite