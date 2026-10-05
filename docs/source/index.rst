slmsuite
========

|slmsuite|_ is a Python package for high-precision
spatial light modulator (SLM) control and holography.

.. sidebar:: Hone Your Spots

   |slmsuite|_ enables **spot-specific**
   `Zernike
   <https://slmsuite.holodyne.com/en/latest/_examples/zernike_holography.html>`_
   aberration correction; capable of optimizing spot imaging
   `across a field of view
   <https://slmsuite.holodyne.com/en/latest/_examples/multipoint_calibration.html>`_.

   .. image:: https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-zernike-spots-dark.gif
      :class: only-dark
      :alt:
         Alongside steering spots in three dimensions, this feature enables removal of
         aberration across a wide field of view. Every spot in the field of view
         is sourced along a different optical path, and slmsuite can correct for each
         different aberration experienced.

   .. image:: https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-zernike-spots.gif
      :class: only-light
      :alt:
         Alongside steering spots in three dimensions, this feature enables removal of
         aberration across a wide field of view. Every spot in the field of view
         is sourced along a different optical path, and slmsuite can correct for each
         different aberration experienced.

   |

.. sidebar:: Enhance Your Holography

   Camera
   `feedback
   <https://slmsuite.holodyne.com/en/latest/_autosummary/slmsuite.holography.algorithms.FeedbackHologram.html#slmsuite.holography.algorithms.FeedbackHologram>`_
   and phase
   `flattening
   <https://slmsuite.holodyne.com/en/latest/_autosummary/slmsuite.holography.algorithms.Hologram.html#slmsuite.holography.algorithms.Hologram.reset_phase>`_
   in |slmsuite|_ lead to crisp and accurate experimental
   `holograms
   <https://slmsuite.holodyne.com/en/latest/_examples/experimental_holography.html>`_,
   even across
   `multiple planes
   <https://slmsuite.holodyne.com/en/latest/_examples/multiplane_holography.html>`_  of focus, color, or basis.

   .. image:: https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-slmsuite-3d-dark.gif
      :class: only-dark
      :alt:
         Advanced meta-hologram programming optimizes spot and image objectives simultaneously.

   .. image:: https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-slmsuite-3d.gif
      :class: only-light
      :alt:
         Advanced meta-hologram programming optimizes spot and image objectives simultaneously.

   |

.. raw:: html

   <p align="center">
   <a href="https://pypi.org/project/slmsuite/"><img alt="PyPi Package" src="https://img.shields.io/pypi/v/slmsuite"></a>
   <a href="https://github.com/holodyne/slmsuite/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/holodyne/slmsuite?color=purple"></a>
   <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
   <a href="https://github.com/holodyne/slmsuite"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/holodyne/slmsuite"></a>
   </p>

Contact us at slmsuite@holodyne.com or on
`GitHub <https://github.com/holodyne/slmsuite>`_
`Issues <https://github.com/holodyne/slmsuite/issues>`_
or
`Discussions <https://github.com/holodyne/slmsuite/discussions>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   examples
   api
   tips

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/holodyne/slmsuite