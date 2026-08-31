"""
Unit tests for slmsuite.holography.analysis.files module.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import pytest

from slmsuite.holography.analysis.files import (
    _max_numeric_id, generate_path, latest_path,
    load_h5, save_h5, read_h5, write_h5, _load_image,
    _gray2rgb, save_image,
)


def _touch(path, content="test"):
    """Create a file with minimal content."""
    with open(path, "w") as f:
        f.write(content)


def test_max_numeric_id(temp_dir, subtests):
    """Test _max_numeric_id() finds the highest numeric suffix among matching files or dirs."""
    with subtests.test("no matches returns -1"):
        assert _max_numeric_id(temp_dir, "empty", "txt", "file", 5) == -1

    with subtests.test("finds the highest id among files"):
        for name in ["alpha_00001.txt", "alpha_00003.txt", "alpha_00002.txt"]:
            _touch(os.path.join(temp_dir, name))
        assert _max_numeric_id(temp_dir, "alpha", "txt", "file", 5) == 3

    with subtests.test("finds the highest id among directories"):
        for name in ["beta_00001", "beta_00005", "beta_00003"]:
            os.makedirs(os.path.join(temp_dir, name))
        assert _max_numeric_id(temp_dir, "beta", None, "dir", 5) == 5

    with subtests.test("ignores names that do not match the prefix or digit count"):
        for name in ["gamma_00001.txt", "other_00099.txt", "gamma_00002.txt", "gamma.txt"]:
            _touch(os.path.join(temp_dir, name))
        assert _max_numeric_id(temp_dir, "gamma", "txt", "file", 5) == 2


def test_generate_path(temp_dir, subtests):
    """Test generate_path(): positional and keyword signatures, increments, dirs, and options."""
    with subtests.test("positional signature"):
        assert generate_path(temp_dir, "test", "txt", "file", 5, 1) == \
            os.path.join(temp_dir, "test_00000.txt")

    with subtests.test("keyword signature, defaults start numbering at zero"):
        assert generate_path(temp_dir, "data", extension="h5") == \
            os.path.join(temp_dir, "data_00000.h5")

    with subtests.test("extension=None omits the dot"):
        assert generate_path(temp_dir, "run", None, "file", 5, 1) == \
            os.path.join(temp_dir, "run_00000")

    with subtests.test("digit_count sets the zero-padding width"):
        assert generate_path(temp_dir, "dig", "txt", "file", 3, 1) == \
            os.path.join(temp_dir, "dig_000.txt")

    with subtests.test("path_count returns consecutive paths"):
        assert generate_path(temp_dir, "multi", "txt", "file", 5, 3) == \
            [os.path.join(temp_dir, f"multi_0000{i}.txt") for i in range(3)]

    with subtests.test("increments past the highest existing id"):
        for name in ["inc_00000.txt", "inc_00001.txt"]:
            _touch(os.path.join(temp_dir, name))
        assert generate_path(temp_dir, "inc", "txt", "file", 5, 1) == \
            os.path.join(temp_dir, "inc_00002.txt")

    with subtests.test("kind='dir' creates the directory"):
        result = generate_path(temp_dir, "asdir", None, "dir", 5, 1)
        assert result == os.path.join(temp_dir, "asdir_00000")
        assert os.path.isdir(result)

    with subtests.test("creates missing parent directories"):
        nested = os.path.join(temp_dir, "nested", "deep")
        result = generate_path(nested, "test", "txt", "file", 5, 1)
        assert os.path.isdir(nested)
        assert result == os.path.join(nested, "test_00000.txt")


def test_latest_path(temp_dir, subtests):
    """Test latest_path(): empty case, highest id, name filtering, and directories."""
    with subtests.test("no matches returns None"):
        assert latest_path(temp_dir, "test", "txt", "file", 5) is None

    with subtests.test("returns the path with the highest id, ignoring other names"):
        for name in ["test_00001.txt", "test_00003.txt", "test_00002.txt", "other_00099.txt"]:
            _touch(os.path.join(temp_dir, name))
        assert latest_path(temp_dir, "test", "txt", "file", 5) == \
            os.path.join(temp_dir, "test_00003.txt")

    with subtests.test("works with directories"):
        for name in ["dir_00001", "dir_00005", "dir_00003"]:
            os.makedirs(os.path.join(temp_dir, name))
        assert latest_path(temp_dir, "dir", None, "dir", 5) == os.path.join(temp_dir, "dir_00005")


def test_save_h5(temp_dir, subtests):
    """Test save_h5() writes data that load_h5() reproduces exactly."""
    path = os.path.join(temp_dir, "data.h5")

    with subtests.test("round-trip reproduces scalars, strings, arrays, None, and nested dicts"):
        data = {
            "integer": 42,
            "float": 3.14,
            "string": "hello",
            "unicode": "café",
            "int_array": np.array([1, 2, 3], dtype=np.int32),
            "float_array": np.arange(12.0).reshape(3, 4),
            "string_array": np.array(["a", "bb", "ccc"]),
            "none": None,
            "none_array": np.array([None, None], dtype=object),
            "nested": {
                "inner": {"value": 7, "array": np.array([[1, 2], [3, 4]])},
                "sibling": "leaf",
            },
        }
        save_h5(path, data)
        loaded = load_h5(path)
        np.testing.assert_equal(loaded, data)
        assert loaded["int_array"].dtype == np.int32

    with subtests.test("staggered arrays raise ValueError"):
        with pytest.raises(ValueError, match="staggered arrays"):
            save_h5(path, {"staggered": [[1, 2], [3, 4, 5]]})

    with subtests.test("exceptions other than ValueError propagate unchanged"):
        class BadObj:
            def __array__(self, *args, **kwargs):
                raise TypeError("cannot convert")

        with pytest.raises(TypeError, match="cannot convert"):
            save_h5(path, {"bad": BadObj()})

    with subtests.test("mode='w' overwrites the previous contents"):
        save_h5(path, {"first": 123}, mode="w")
        save_h5(path, {"second": 456}, mode="w")
        assert set(load_h5(path).keys()) == {"second"}

    with subtests.test("write_h5 is an alias of save_h5"):
        write_h5(path, {"aliased": 7})
        assert load_h5(path)["aliased"] == 7


def test_load_h5(temp_dir, subtests):
    """Test load_h5() reading of a raw HDF5 structure, independent of save_h5()."""
    path = os.path.join(temp_dir, "raw.h5")

    with subtests.test("decode_bytes=True decodes strings and byte arrays; False leaves them raw"):
        with h5py.File(path, "w") as f:
            f["scalar"] = b"hello"
            f["array"] = np.array([b"hello", b"world"])

        decoded = load_h5(path, decode_bytes=True)
        assert decoded["scalar"] == "hello"
        np.testing.assert_array_equal(decoded["array"], ["hello", "world"])

        raw = load_h5(path, decode_bytes=False)
        assert raw["scalar"] == b"hello"
        assert isinstance(raw["array"][0], bytes)

    with subtests.test("the __none__ placeholder decodes to None, or an all-None list with its shape"):
        with h5py.File(path, "w") as f:
            f["scalar"] = False
            f["scalar"].attrs["__none__"] = True
            f["stack"] = np.zeros((2, 2), dtype=bool)
            f["stack"].attrs["__none__"] = True

        loaded = load_h5(path)
        assert loaded["scalar"] is None
        assert loaded["stack"] == [[None, None], [None, None]]

    with subtests.test("groups become nested dicts"):
        with h5py.File(path, "w") as f:
            f.create_group("outer")["value"] = 5
        assert load_h5(path)["outer"]["value"] == 5

    with subtests.test("read_h5 is an alias of load_h5"):
        assert read_h5(path)["outer"]["value"] == 5


def test_load_image(temp_dir, subtests):
    """Test _load_image() for missing files, shape padding, and brightness inversion."""
    with subtests.test("raises ValueError when the file does not exist"):
        with pytest.raises(ValueError, match="Image not found"):
            _load_image("nonexistent_file.png", (100, 100))

    with subtests.test("pads to the requested shape, with or without an intermediate zoom"):
        img_path = os.path.join(temp_dir, "test.png")
        img = np.zeros((100, 100), dtype=np.uint8)
        img[10:90, 10:90] = 120
        cv2.imwrite(img_path, img)
        for target_shape in (None, (80, 80)):
            result = _load_image(img_path, (100, 100), target_shape=target_shape)
            assert result.shape == (100, 100)

    with subtests.test("a predominantly bright image is inverted before the amplitude is taken"):
        img_path = os.path.join(temp_dir, "bright.png")
        cv2.imwrite(img_path, np.full((80, 80), 220, dtype=np.uint8))
        result = _load_image(img_path, (100, 100))
        # sqrt() runs on the raw uint8 image, so the amplitude is float16-precision.
        assert result.max() == pytest.approx(np.sqrt(255 - 220), rel=1e-3)

    with subtests.test("a predominantly dark image is left unchanged"):
        img_path = os.path.join(temp_dir, "dark.png")
        cv2.imwrite(img_path, np.full((80, 80), 30, dtype=np.uint8))
        result = _load_image(img_path, (100, 100))
        assert result.max() == pytest.approx(np.sqrt(30), rel=1e-3)


def test_gray2rgb(subtests):
    """Test _gray2rgb() conversion of grayscale images to RGBA."""
    with subtests.test("2D input gains a leading stack axis"):
        result = _gray2rgb(np.ones((10, 10), dtype=np.uint8) * 100)
        assert result.shape == (1, 10, 10)

    with subtests.test("images already carrying an RGB or RGBA channel pass through unchanged"):
        for channels in (3, 4):
            img = np.ones((2, 10, 10, channels), dtype=np.uint8) * 100
            np.testing.assert_array_equal(_gray2rgb(img), img)

    with subtests.test("shapes beyond a stack of 2D images raise RuntimeError"):
        with pytest.raises(RuntimeError, match="could not be parsed"):
            _gray2rgb(np.ones((2, 3, 10, 10, 1), dtype=np.uint8))

    with subtests.test("cmap='default'/'grayscale' are aliases for cmap=True/False"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        np.testing.assert_array_equal(_gray2rgb(img, cmap="default"), _gray2rgb(img, cmap=True))
        np.testing.assert_array_equal(_gray2rgb(img, cmap="grayscale"), _gray2rgb(img, cmap=False))

    cases = {
        "grayscale": dict(cmap=False),
        "default colormap": dict(cmap=True),
        "named colormap, integer lut defaults to the image max": dict(cmap="viridis"),
        "explicit lut": dict(cmap="viridis", lut=100),
        "float image, normalized": dict(cmap="viridis", normalize=True, dtype=float),
        "float image, unnormalized": dict(cmap="viridis", normalize=False, dtype=float),
        "grayscale with an out-of-range lut": dict(cmap=False, lut=300),
    }
    for label, kwargs in cases.items():
        with subtests.test(f"produces well-formed output: {label}"):
            kwargs = dict(kwargs)
            is_float = kwargs.pop("dtype", None) is float
            img = np.random.rand(1, 10, 10) if is_float else \
                np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
            result = _gray2rgb(img, **kwargs)
            assert result.dtype == np.uint8
            if kwargs.get("cmap") is not False:
                assert result.shape[-1] == 4

    with subtests.test("colormap objects are accepted with or without a .colors attribute"):
        import matplotlib.pyplot as plt

        class NoColorsCmap:
            """Colormap-like object exposing N and __call__ but no .colors."""
            N = 10

            def __call__(self, x):
                x = np.asarray(x, dtype=float)
                rgba = np.zeros((*x.shape, 4))
                rgba[..., 0] = x / self.N
                rgba[..., 3] = 1.0
                return rgba

        cases = [
            (plt.get_cmap("viridis", 64), 64, np.array([[[0, 10], [20, 63]]], dtype=np.uint8)),
            (NoColorsCmap(), 10, np.array([[[0, 2], [4, 9]]], dtype=np.int32)),
        ]
        for cmap, lut, img in cases:
            assert _gray2rgb(img, cmap=cmap, lut=lut).shape[-1] == 4

    with subtests.test("NaN pixels become fully transparent"):
        img = np.full((1, 10, 10), 0.5)
        img[0, 3, 3] = np.nan
        result = _gray2rgb(img, cmap="viridis")
        assert result[0, 3, 3, 3] == 0

    with subtests.test("a scalar border paints all four edges the same value"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 100
        result = _gray2rgb(img, cmap="viridis", border=255)
        assert result[0, 0, 0, 0] == 255
        assert result[0, -1, 0, 0] == 255
        assert result[0, 0, -1, 0] == 255

    with subtests.test("a list border sets only the given channels"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 100
        result = _gray2rgb(img, cmap="viridis", border=[255, 128])
        assert result[0, 0, 0, 0] == 255
        assert result[0, 0, 0, 1] == 128


def test_save_image(temp_dir, subtests):
    """Test save_image() writes files via imageio, with colormap and border options."""
    gray = lambda: np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    cases = {
        "single grayscale png": (gray(), "test.png", {}),
        "single image with a colormap": (gray(), "test_cmap.png", {"cmap": "viridis"}),
        "stack saved as an animated gif": (
            np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8), "test.gif", {},
        ),
        "float image": (np.random.rand(10, 10), "test_float.png", {"cmap": "viridis"}),
        "float image, unnormalized": (
            np.random.rand(10, 10) * 0.5, "test_nonorm.png", {"cmap": "viridis", "normalize": False},
        ),
        "border option": (gray(), "test_border.png", {"cmap": "viridis", "border": 255}),
    }
    for label, (img, name, kwargs) in cases.items():
        with subtests.test(label):
            path = os.path.join(temp_dir, name)
            save_image(path, img, **kwargs)
            assert os.path.exists(path)

    with subtests.test("raises ValueError when imageio is not installed"):
        path = os.path.join(temp_dir, "test_missing_imageio.png")
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "imageio", None)
            with pytest.raises(ValueError, match="imageio is required"):
                save_image(path, gray())
