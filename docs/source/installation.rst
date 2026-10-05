.. _installation:

Installation
============

All commands below use ``pip``. For editable (development) installs from a local
clone, pass the ``-e`` flag (e.g. ``pip install -e .``).

.. tip::

   We recommend `uv <https://docs.astral.sh/uv/>`_ as a fast, modern package
   manager. To use it, substitute ``uv pip`` for ``pip`` in all commands below.

PyPI
----

Install the stable version of |slmsuite|_ from `PyPI <https://pypi.org/project/slmsuite/>`_ using:

.. code-block:: console

    pip install slmsuite

GitHub
------

Install the latest version of |slmsuite|_ from `GitHub <https://github.com/holodyne/slmsuite>`_ using:

.. code-block:: console

    pip install git+https://github.com/holodyne/slmsuite

One can also clone |slmsuite|_ directly and add its directory to the Python path.
*Remember to install the dependencies (next sections)*.

.. code-block:: console

    git clone https://github.com/holodyne/slmsuite

Required Dependencies
---------------------

The following python packages are necessary to run |slmsuite|_. These are listed as PyPI
dependencies and thus are installed automatically if PyPI is used to install.

- `python <https://www.python.org/>`_ >= 3.10
- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_
- `opencv-python <https://github.com/opencv/opencv-python>`_
- `matplotlib <https://matplotlib.org/>`_
- `h5py <https://www.h5py.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

One can also install these dependencies directly by calling the following inside
the package directory.

.. code-block:: console

    pip install -e .

Hardware Dependencies
---------------------

The following python packages are *optional* acceleration or hardware requirements, which
the user can install selectively.

- GPU ``pip install -e ".[gpu]"``
    - `cupy <https://cupy.dev/>`_, highly recommended for GPU-accelerated holography.
      Once installed, holograms and SLMs (unless ``gpu=False``) run on :mod:`cupy`;
      camera frames are returned in host memory unless a simulated camera is called
      with ``get_image(get=False)``. Without :mod:`cupy`, :mod:`numpy` is used as a backup.
      The ``gpu`` extra installs ``cupy-cuda13x`` (CUDA 13). For another CUDA version,
      skip the extra and ``pip install cupy-cudaYYx``, finding ``YY`` with
      ``nvcc --version``.
- Gradients ``pip install -e ".[torch]"``
    - `pytorch <https://pytorch.org/>`_, required for gradient-based (``"CG"``) hologram
      optimization with a :mod:`torch.optim` optimizer, either in GPU or CPU mode. Uses :mod:`cupy` - :mod:`torch`
      `interoperability <https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch>`_
      to pass data between modules without copying overhead, even on the GPU.
- Cameras ``pip install -e ".[cameras]"``
    - `instrumental-lib <https://github.com/mabuchilab/Instrumental>`_
    - `pylablib <https://github.com/AlexShkarin/pyLabLib>`_
    - `pymmcore <https://github.com/micro-manager/pymmcore>`_
    - `pypylon <https://github.com/basler/pypylon>`_
    - `mvsdk <https://www.mindvision.com.cn/category/software/demo-development-routine/>`_ (non-PyPI)
    - `PySpin <https://www.flir.com/products/spinnaker-sdk/>`_ (non-PyPI)
    - `tisgrabber <https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber>`_ (non-PyPI)
    - `thorlabs_tsi_sdk <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (non-PyPI)
    - `VmbPy <https://github.com/alliedvision/VmbPy>`_ (non-PyPI)
    - Other cameras are loaded directly via .dll.
- SLMs ``pip install -e ".[slms]"``
    - `pyglet <https://pyglet.org/>`_
    - `hidapi <https://pypi.org/project/hidapi/>`_ and
      `pyyaml <https://pypi.org/project/PyYAML/>`_, for Texas Instruments PLMs
    - Other SLMs are loaded via their vendor's SDK or .dll.
- Image saving ``pip install -e ".[images]"``
    - For most images and videos, `imageio <https://imageio.readthedocs.io/en/stable/>`_
    - Many video formats additionally require `pyav <https://pypi.org/project/av/>`_
    - For .gif optimization, `pygifsicle <https://pypi.org/project/pygifsicle/>`_

Jupyter
-------

We highly recommended using `Jupyter <https://jupyter.org>`_
notebooks for interactive computing. Consider also using
`IPython <https://ipython.org/>`_
`magic <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#magics-explained>`_,
features like |autoreload|_ or |matplotlibs|_.

- `jupyter <https://jupyter.org>`_
- `ipywidgets <https://ipywidgets.readthedocs.io/>`_ and
  `ipyevents <https://github.com/mwcraig/ipyevents>`_, for the ``live()`` viewer of
  cameras and SLMs
- The ``images`` extra

Outside Jupyter, ``plt.show()`` blocks; call :func:`slmsuite.configure_plotting` with
``mode="suppress"`` or ``mode="save"`` (and ``headless=True`` without a display).

Use the following to install recommended jupyter-related packages.

.. code-block:: console

    pip install -e ".[jupyter]"

All Dependencies
----------------

To install the optional dependencies for GPU, gradients, SLMs, cameras, images, Jupyter,
docs, and testing at once, use the ``dev`` extra.

.. code-block:: console

    pip install -e ".[dev]"


.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/holodyne/slmsuite

.. |autoreload| replace:: ``%autoreload 2``
.. _autoreload: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html

.. |matplotlibs| replace:: ``%matplotlib inline``
.. _matplotlibs: https://ipython.readthedocs.io/en/stable/interactive/plotting.html