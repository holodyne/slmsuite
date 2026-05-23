"""Interface to experimental devices."""

import datetime
import warnings

from slmsuite import __version__
from slmsuite.misc.files import generate_path, latest_path, load_h5, save_h5


class _Picklable:
    """
    Class for hardware objects to handle state saving.
    """

    _pickle = []  # Baseline parameters to pickle.
    _pickle_data = []  #

    def pickle(self, attributes=True, metadata=True):
        """
        Returns a dictionary containing selected attributes of this class.

        Parameters
        ----------
        attributes : bool OR list of str
            If ``False``, pickles only baseline attributes, usually single floats.
            If ``True``, also pickles 'heavy' attributes such as large images and calibrations.
            If ``list of str``, pickles the keys in the given list.
            By default, the chosen attributes should be things that can be written to
            .h5 files: scalars and lists of scalars.
        metadata : bool
            If ``True``, package the dictionary as the
            ``"__meta__"`` value of a superdictionary which also contains:
            ``"__version__"``, the current slmsuite version,
            ``"__time__"``, the time formatted as a date string, and
            ``"__timestamp__"``, the time formatting as a floating point timestamp.
            This information is used as standard metadata for calibrations and saving.
        """
        # Parse attributes.
        recursive_attributes = attributes is True  # Heavy pickling only if True.
        if isinstance(attributes, bool):
            attributes = self._pickle + (self._pickle_data if attributes else [])

        # Assemble the dictionary.
        pickled = {}
        pickled["__class__"] = str(self)

        for k in attributes:
            if not hasattr(self, k):
                warnings.warn(f"Expected attribute '{k}' not present in {self}.")
            else:
                attr = getattr(self, k)

                if hasattr(attr, "pickle"):
                    pickled[k] = attr.pickle(attributes=recursive_attributes, metadata=False)
                else:
                    pickled[k] = attr

        # Return the result.
        if metadata:
            t = datetime.datetime.now()
            return {
                "__version__": __version__,
                "__time__": str(t),
                "__timestamp__": t.timestamp(),
                "__meta__": pickled,
            }
        else:
            return pickled

    def save(self, path=".", name=None, **kwargs):
        """
        Saves the dictionary returned from :meth:`pickle()` to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        path : str
            Path to directory to save in. Default is current directory.
        name : str OR None
            Name of the save file. If ``None``, will use :attr:`name` + ``'-pickle'``.
        **kwargs
            Passed to :meth:`pickle()` to customize how and what data is saved.

        Returns
        -------
        str
            The file path that the pickled data was saved to.
        """
        if name is None:
            name = self.name + "-pickle"
        file_path = generate_path(path, name, extension="h5")

        save_h5(file_path, self.pickle(**kwargs))

        return file_path
