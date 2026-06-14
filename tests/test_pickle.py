"""
Unit tests for the _Picklable base class, which handles object serialization and saving.
"""
import pytest
import tempfile
import os

import h5py

from slmsuite._pickling import _Picklable
from slmsuite import __version__


class _TestPicklableClass(_Picklable):
    """Concrete _Picklable used by most tests."""

    _pickle = ["basic_attr", "name"]
    _pickle_data = ["heavy_attr"]

    def __init__(self):
        self.basic_attr = 42
        self.heavy_attr = [1, 2, 3, 4, 5]
        self.name = "test_object"
        self.unpickled_attr = "not_pickled"

    def __str__(self):
        return "TestPicklableClass"


class TestPicklable:
    """Tests for the _Picklable base class."""

    @pytest.fixture(autouse=True)
    def _obj(self):
        self.obj = _TestPicklableClass()

    def test_pickle_attributes(self, subtests):
        """Test pickle output for different `attributes` values."""
        with subtests.test("attributes=False keeps only _pickle"):
            result = self.obj.pickle(attributes=False, metadata=False)

            assert result["__class__"] == "TestPicklableClass"
            assert result["basic_attr"] == 42
            assert result["name"] == "test_object"
            assert "heavy_attr" not in result
            assert "unpickled_attr" not in result

        with subtests.test("attributes=True includes _pickle_data"):
            result = self.obj.pickle(attributes=True, metadata=False)

            assert result["__class__"] == "TestPicklableClass"
            assert result["basic_attr"] == 42
            assert result["name"] == "test_object"
            assert result["heavy_attr"] == [1, 2, 3, 4, 5]
            assert "unpickled_attr" not in result

        with subtests.test("custom attribute list"):
            result = self.obj.pickle(attributes=["basic_attr"], metadata=False)

            assert result["__class__"] == "TestPicklableClass"
            assert result["basic_attr"] == 42
            assert "name" not in result
            assert "heavy_attr" not in result

        with subtests.test("__str__ used as __class__"):
            result = self.obj.pickle(attributes=False, metadata=False)
            assert result["__class__"] == "TestPicklableClass"

    def test_pickle_metadata(self, subtests):
        """Test pickle metadata and version info."""
        with subtests.test("metadata fields present"):
            result = self.obj.pickle(attributes=False, metadata=True)

            assert result["__version__"] == __version__
            assert isinstance(result["__time__"], str)
            assert isinstance(result["__timestamp__"], float)

            meta = result["__meta__"]
            assert "__class__" in meta
            assert "basic_attr" in meta
            assert "name" in meta

    def test_pickle_edge_cases(self, subtests):
        """Test warnings, nested objects, and empty _pickle lists."""
        with subtests.test("missing attribute warns"):
            with pytest.warns(
                UserWarning, match="Expected attribute 'nonexistent' not present"
            ):
                result = self.obj.pickle(attributes=["nonexistent"], metadata=False)
            assert "__class__" in result

        with subtests.test("nested Picklable is recursively pickled"):

            class _Nested(_Picklable):
                _pickle = ["nested_value"]

                def __init__(self):
                    self.nested_value = "nested"

                def __str__(self):
                    return "NestedPicklable"

            self.obj.nested_obj = _Nested()
            result = self.obj.pickle(attributes=["nested_obj"], metadata=False)

            nested = result["nested_obj"]
            assert nested["__class__"] == "NestedPicklable"
            assert nested["nested_value"] == "nested"

        with subtests.test("empty _pickle lists yield only __class__"):

            class _Empty(_Picklable):
                _pickle = []
                _pickle_data = []

                def __init__(self):
                    self.some_attr = "value"

                def __str__(self):
                    return "EmptyPicklable"

            result = _Empty().pickle(attributes=True, metadata=False)
            assert "__class__" in result
            assert "some_attr" not in result

    def test_save(self, subtests):
        """Test save method: default name, custom name, kwargs, and missing name."""
        with subtests.test("default name from .name attribute"):
            with tempfile.TemporaryDirectory() as d:
                path = self.obj.save(path=d)
                assert os.path.exists(path)
                assert path.endswith(".h5")
                assert "test_object-pickle" in path

        with subtests.test("custom name"):
            with tempfile.TemporaryDirectory() as d:
                path = self.obj.save(path=d, name="custom_name")
                assert os.path.exists(path)
                assert "custom_name" in path

        with subtests.test("kwargs forwarded to pickle"):
            with tempfile.TemporaryDirectory() as d:
                path = self.obj.save(path=d, attributes=True)
                with h5py.File(path, "r") as f:
                    assert "__meta__" in f
                    assert "heavy_attr" in f["__meta__"]

        with subtests.test("no name attribute raises AttributeError"):

            class _NoName(_Picklable):
                _pickle = ["value"]

                def __init__(self):
                    self.value = 123

                def __str__(self):
                    return "NoNamePicklable"

            with tempfile.TemporaryDirectory() as d:
                with pytest.raises(AttributeError):
                    _NoName().save(path=d)