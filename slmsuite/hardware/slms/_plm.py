"""
Texas Instruments PLM chips, shared by the drivers of the controllers that address them.

:class:`_PLMModel` reads a chip's shape, pitch and displacement ratios from
``texasinstruments.yaml`` and derives the phase lookup table from them.
"""

import os

import numpy as np

from slmsuite.hardware.slms.slm import LUT_SIZE

try:
    import yaml
except ImportError:
    yaml = None

MODEL_DB_PATH = os.path.join(os.path.dirname(__file__), "texasinstruments.yaml")


def _load_model_db():
    """The model database read from ``texasinstruments.yaml``."""
    if yaml is None:
        raise ImportError(
            "pyyaml is required to read the model database. "
            "Install it with `pip install slmsuite[slms]`."
        )
    with open(MODEL_DB_PATH) as f:
        return yaml.safe_load(f)


class _DLPModel:
    """
    Mixin for :class:`~slmsuite.hardware.slms.slm.SLM` drivers of Texas Instruments DLP
    chips, whose model database is ``texasinstruments.yaml``. A subclass reads the
    models which hold its :attr:`_model_key`.

    Attributes
    ----------
    model_config : dict
        Model configuration from ``texasinstruments.yaml``.
    """

    _model_key = None

    @classmethod
    def _load_models(cls):
        """The models of this kind of chip in the database."""
        return {
            name: config for (name, config) in _load_model_db().items() if cls._model_key in config
        }

    @classmethod
    def load_model_config(cls, model_name):
        """
        Load model configuration from ``texasinstruments.yaml``.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g. ``"p47"``, ``"p67"``, ``"p99"``).

        Returns
        -------
        dict
            Model configuration.

        Raises
        ------
        ValueError
            If the model is not in the database.
        """
        model_db = cls._load_models()

        if model_name not in model_db:
            available = list(model_db.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available}")

        return model_db[model_name]

    @classmethod
    def get_model_list(cls):
        """
        Get the list of models of this kind of chip in the database.

        Returns
        -------
        list of str
            Model identifiers available in ``texasinstruments.yaml``.
        """
        return list(cls._load_models().keys())


class _PLMModel(_DLPModel):
    """
    Mixin for :class:`~slmsuite.hardware.slms.slm.SLM` drivers of Texas Instruments PLM
    chips: the models of ``texasinstruments.yaml`` with displacement ratios.
    """

    _model_key = "displacement_ratios"
    _gamma_sign = +1  # Increasing displacement increases phase delay.

    @property
    def _model_bitdepth(self):
        """Bits per pixel, from the number of displacement ratios in :attr:`model_config`."""
        return int(np.log2(len(self.model_config["displacement_ratios"])))

    @property
    def _model_pixel(self):
        """The ``pixel`` keys of :attr:`model_config`, or ``None`` for a model without them."""
        return self.model_config.get("pixel", None)

    def set_gamma(self, gamma=None, levels=None, lut_size=LUT_SIZE):
        r"""
        See :meth:`~slmsuite.hardware.slms.slm.SLM.set_gamma`. As a PLM's levels are unevenly
        spaced in phase, ``None`` restores the model's table (its displacement ratios, with
        the top level one step short of :math:`2\pi`) rather than a linear response.
        """
        if gamma is None:
            gamma = (
                np.array(self.model_config["displacement_ratios"])
                * (self.bitresolution - 1)
                / self.bitresolution
            )
            levels = None
        return super().set_gamma(gamma, levels=levels, lut_size=lut_size)
