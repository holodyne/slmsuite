"""
Unit tests for the _Picklable base class, which handles object serialization and saving.
"""
import os

import pytest

from slmsuite import __version__
from slmsuite._pickling import _Picklable
from slmsuite.misc.files import load_h5


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

    def test_pickle(self, subtests):
        """Test pickle() attribute selection, metadata, and recursion."""
        with subtests.test("attributes=False keeps only the baseline _pickle attrs"):
            result = self.obj.pickle(attributes=False, metadata=False)
            assert result == {
                "__class__": "_TestPicklableClass",
                "basic_attr": 42,
                "name": "test_object",
            }

        with subtests.test("attributes=True adds _pickle_data"):
            result = self.obj.pickle(attributes=True, metadata=False)
            assert result == {
                "__class__": "_TestPicklableClass",
                "basic_attr": 42,
                "name": "test_object",
                "heavy_attr": [1, 2, 3, 4, 5],
            }

        with subtests.test("an explicit attribute list overrides _pickle/_pickle_data"):
            result = self.obj.pickle(attributes=["basic_attr"], metadata=False)
            assert result == {"__class__": "_TestPicklableClass", "basic_attr": 42}

        with subtests.test("metadata=True wraps the pickle with version and timestamp"):
            result = self.obj.pickle(attributes=False, metadata=True)
            assert result["__version__"] == __version__
            assert isinstance(result["__time__"], str)
            assert isinstance(result["__timestamp__"], float)
            assert result["__meta__"] == {
                "__class__": "_TestPicklableClass",
                "basic_attr": 42,
                "name": "test_object",
            }

        with subtests.test("a missing attribute warns and is skipped"):
            with pytest.warns(
                UserWarning, match="Expected attribute 'nonexistent' not present"
            ):
                result = self.obj.pickle(attributes=["nonexistent"], metadata=False)
            assert result == {"__class__": "_TestPicklableClass"}

        with subtests.test("a nested Picklable is recursively pickled"):
            class _Nested(_Picklable):
                _pickle = ["nested_value"]

                def __init__(self):
                    self.nested_value = "nested"

                def __str__(self):
                    return "NestedPicklable"

            self.obj.nested_obj = _Nested()
            result = self.obj.pickle(attributes=["nested_obj"], metadata=False)
            assert result == {
                "__class__": "_TestPicklableClass",
                "nested_obj": {"__class__": "_Nested", "nested_value": "nested"},
            }

        with subtests.test("empty _pickle lists yield only __class__"):
            class _Empty(_Picklable):
                _pickle = []
                _pickle_data = []

                def __init__(self):
                    self.some_attr = "value"

                def __str__(self):
                    return "EmptyPicklable"

            assert _Empty().pickle(attributes=True, metadata=False) == {
                "__class__": "_Empty"
            }

    def test_save(self, subtests, temp_dir):
        """Test save() file naming, kwargs forwarding, and the .name requirement."""
        with subtests.test("the default name comes from .name"):
            path = self.obj.save(path=temp_dir)
            assert os.path.exists(path)
            assert os.path.basename(path).startswith("test_object-pickle")
            assert path.endswith(".h5")

        with subtests.test("a name kwarg overrides the default"):
            path = self.obj.save(path=temp_dir, name="custom_name")
            assert os.path.basename(path).startswith("custom_name")

        with subtests.test("kwargs are forwarded to pickle()"):
            path = self.obj.save(path=temp_dir, attributes=False)
            saved = load_h5(path)
            assert "heavy_attr" not in saved["__meta__"]

        with subtests.test("no .name attribute raises AttributeError"):
            class _NoName(_Picklable):
                _pickle = ["value"]

                def __init__(self):
                    self.value = 123

                def __str__(self):
                    return "NoNamePicklable"

            with pytest.raises(AttributeError):
                _NoName().save(path=temp_dir)
