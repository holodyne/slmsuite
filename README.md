<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/holodyne/slmsuite/main/docs/source/static/slmsuite-dark.svg">
<img alt="slmsuite" src="https://raw.githubusercontent.com/holodyne/slmsuite/main/docs/source/static/slmsuite.svg" width="256">
</picture>
</p>

<h2 align="center">High-Performance Spatial Light Modulator Control and Holography</h2>

<p align="center">
<a href="https://slmsuite.readthedocs.io/en/latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/slmsuite/badge/?version=latest"></a>
<a href="https://pypi.org/project/slmsuite/"><img alt="PyPi Package" src="https://img.shields.io/pypi/v/slmsuite"></a>
<a href="https://github.com/holodyne/slmsuite/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/holodyne/slmsuite?color=purple"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

`slmsuite` combines GPU-accelerated beamforming algorithms with optimized hardware control, automated calibration, and user-friendly scripting to enable high-performance programmable optics with modern spatial light modulators.

## Key Features
- [GPU-accelerated iterative phase retrieval  algorithms](https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Computational-Holography) (e.g. Gerchberg-Saxton, weighted GS, or phase-stationary WGS)
- [A simple hardware-control interface](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#Loading-Hardware) for working with various SLMs and cameras
- [Automated Fourier- to image-space coordinate transformations](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#Fourier-Calibration): choose how much light goes to which camera pixels; `slmsuite` takes care of the rest!
- [Automated wavefront calibration](https://slmsuite.readthedocs.io/en/latest/_examples/wavefront_calibration.html) to improve manufacturer-supplied flatness maps or compensate for additional aberrations along the SLM imaging train
- Optimized [optical focus/spot arrays](https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html#Spot-Arrays) using [camera feedback](https://slmsuite.readthedocs.io/en/latest/_examples/experimental_holography.html#A-Uniform-Square-Array), automated statistics, and numerous analysis routines
- [Mixed region amplitude freedom](https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.Hologram.html#slmsuite.holography.algorithms.Hologram.optimize), which ignores unused far-field regions in favor of optimized hologram performance in high-interest areas.
- [Toolboxes for structured light](https://slmsuite.readthedocs.io/en/latest/_examples/structured_light.html#), imprinting sectioned phase masks, SLM unit conversion, padding and unpadding data, and more
- A fully-featured [example library](https://slmsuite.readthedocs.io/en/latest/examples.html) that demonstrates these and other features

## Installation

Install the stable version of `slmsuite` from [PyPI](https://pypi.org/project/slmsuite/) using:

```console
pip install slmsuite
```

Install the latest version of `slmsuite` from GitHub using:

```console
pip install git+https://github.com/holodyne/slmsuite
```

## Documentation and Examples

Extensive
[documentation](https://slmsuite.readthedocs.io/en/latest/)
and
[API reference](https://slmsuite.readthedocs.io/en/latest/api.html)
are available through readthedocs.

Examples can be found embedded in
[documentation](https://slmsuite.readthedocs.io/en/latest/examples.html)
or directly in
[source](https://github.com/holodyne/slmsuite-examples).

## Support

Users are encouraged to submit [issues](https://github.com/holodyne/slmsuite/issues) and [discussions](https://github.com/holodyne/slmsuite/discussions) via GitHub.

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-zernike-spots-dark.gif">
<img alt="slmsuite" src="https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/ex-zernike-spots.gif" width="512">
</picture>
</p>

`slmsuite` is maintained by [Holodyne Labs, Inc](https://www.holodyne.com). Contact slmsuite@holodyne.com for additional information.