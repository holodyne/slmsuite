"""
Unit tests for slmsuite.holography.analysis.files module.
"""
import pytest
import sys
import tempfile
import time
import os
import warnings
import h5py
import numpy as np


def _safe_unlink(path, retries=5, delay=0.1):
    """Remove a file, retrying on PermissionError (Windows h5py file lock)."""
    for i in range(retries):
        try:
            os.unlink(path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(delay)

from slmsuite.holography.analysis.files import (
    _max_numeric_id, generate_path, latest_path,
    load_h5, save_h5, read_h5, write_h5, _load_image,
    _gray2rgb, save_image,
)


def _touch(path, content="test"):
    """Helper to create a file with minimal content."""
    with open(path, "w") as f:
        f.write(content)


def test_max_numeric_id(subtests):
    """Test _max_numeric_id for files, directories, and edge cases."""
    with subtests.test("no files returns -1"):
        with tempfile.TemporaryDirectory() as d:
            assert _max_numeric_id(d, "test", "txt", "file", 5) == -1

    with subtests.test("finds highest among files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "test_00003.txt", "test_00002.txt"]:
                _touch(os.path.join(d, name))
            assert _max_numeric_id(d, "test", "txt", "file", 5) == 3

    with subtests.test("finds highest among directories"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001", "test_00005", "test_00003"]:
                os.makedirs(os.path.join(d, name))
            assert _max_numeric_id(d, "test", None, "dir", 5) == 5

    with subtests.test("ignores non-matching files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "other_00005.txt", "test_00002.txt", "test.txt"]:
                _touch(os.path.join(d, name))
            assert _max_numeric_id(d, "test", "txt", "file", 5) == 2


def test_generate_path(subtests):
    """Test generate_path for single/multiple paths, directories, and options."""
    with subtests.test("single file path"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 5, 1)
            assert result == os.path.join(d, "test_00000.txt")

    with subtests.test("multiple file paths"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 5, 3)
            assert isinstance(result, list)
            assert len(result) == 3
            assert result[0] == os.path.join(d, "test_00000.txt")
            assert result[1] == os.path.join(d, "test_00001.txt")
            assert result[2] == os.path.join(d, "test_00002.txt")

    with subtests.test("directory path is created"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", None, "dir", 5, 1)
            assert result == os.path.join(d, "test_00000")
            assert os.path.isdir(result)

    with subtests.test("increments past existing files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00000.txt", "test_00001.txt"]:
                _touch(os.path.join(d, name))
            result = generate_path(d, "test", "txt", "file", 5, 1)
            assert result == os.path.join(d, "test_00002.txt")

    with subtests.test("creates missing parent directories"):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "nested", "deep")
            result = generate_path(nested, "test", "txt", "file", 5, 1)
            assert os.path.exists(nested)
            assert result == os.path.join(nested, "test_00000.txt")

    with subtests.test("no extension"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", None, "file", 5, 1)
            assert result == os.path.join(d, "test_00000")

    with subtests.test("custom digit count"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 3, 1)
            assert result == os.path.join(d, "test_000.txt")


def test_latest_path(subtests):
    """Test latest_path for files and directories."""
    with subtests.test("no files returns None"):
        with tempfile.TemporaryDirectory() as d:
            assert latest_path(d, "test", "txt", "file", 5) is None

    with subtests.test("returns path with highest id"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "test_00003.txt", "test_00002.txt"]:
                _touch(os.path.join(d, name))
            result = latest_path(d, "test", "txt", "file", 5)
            assert result == os.path.join(d, "test_00003.txt")

    with subtests.test("works with directories"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001", "test_00005", "test_00003"]:
                os.makedirs(os.path.join(d, name))
            result = latest_path(d, "test", None, "dir", 5)
            assert result == os.path.join(d, "test_00005")


def _make_tmp_h5():
    """Create a closed temporary .h5 file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    return tmp.name


def test_save_and_load_h5(subtests):
    """Test HDF5 save/load roundtrip for various data types and options."""
    with subtests.test("simple data types"):
        path = _make_tmp_h5()
        try:
            data = {
                "integer": 42,
                "float": 3.14,
                "string": "hello",
                "array": np.array([1, 2, 3, 4, 5]),
                "none_value": None,
            }
            save_h5(path, data)
            loaded = load_h5(path)

            assert loaded["integer"] == 42
            assert loaded["float"] == pytest.approx(3.14)
            assert loaded["string"] == "hello"
            np.testing.assert_array_equal(loaded["array"], [1, 2, 3, 4, 5])
            assert loaded["none_value"] == False  # None becomes False
        finally:
            _safe_unlink(path)

    with subtests.test("nested dictionaries"):
        path = _make_tmp_h5()
        try:
            data = {
                "level1": {
                    "level2": {
                        "value": 123,
                        "array": np.array([[1, 2], [3, 4]]),
                    },
                    "simple": "test",
                },
                "top_level": 456,
            }
            save_h5(path, data)
            loaded = load_h5(path)

            assert loaded["level1"]["level2"]["value"] == 123
            np.testing.assert_array_equal(
                loaded["level1"]["level2"]["array"], [[1, 2], [3, 4]]
            )
            assert loaded["level1"]["simple"] == "test"
            assert loaded["top_level"] == 456
        finally:
            _safe_unlink(path)

    with subtests.test("string arrays"):
        path = _make_tmp_h5()
        try:
            data = {
                "string_array": np.array(["hello", "world", "test"]),
                "single_string": "test_string",
            }
            save_h5(path, data)
            loaded = load_h5(path)

            np.testing.assert_array_equal(
                loaded["string_array"], ["hello", "world", "test"]
            )
            assert loaded["single_string"] == "test_string"
        finally:
            _safe_unlink(path)

    with subtests.test("staggered arrays raise ValueError"):
        path = _make_tmp_h5()
        try:
            data = {"staggered": [[1, 2], [3, 4, 5]]}
            with pytest.raises(ValueError, match="staggered arrays"):
                save_h5(path, data)
        finally:
            if os.path.exists(path):
                _safe_unlink(path)

    with subtests.test("decode_bytes=True vs False"):
        path = _make_tmp_h5()
        try:
            with h5py.File(path, "w") as f:
                f["string_data"] = b"hello"
                f["byte_array"] = np.array([b"hello", b"world"])

            loaded = load_h5(path, decode_bytes=True)
            assert loaded["string_data"] == "hello"
            np.testing.assert_array_equal(
                loaded["byte_array"], ["hello", "world"]
            )

            loaded = load_h5(path, decode_bytes=False)
            assert loaded["string_data"] == b"hello"
            assert isinstance(loaded["byte_array"][0], bytes)
        finally:
            _safe_unlink(path)

    with subtests.test("backwards-compat aliases read_h5/write_h5"):
        path = _make_tmp_h5()
        try:
            write_h5(path, {"test": 123})
            loaded = read_h5(path)
            assert loaded["test"] == 123
        finally:
            _safe_unlink(path)

    with subtests.test("mode parameter"):
        path = _make_tmp_h5()
        try:
            save_h5(path, {"first": 123}, mode="w")
            save_h5(path, {"second": 456}, mode="w")
            loaded = load_h5(path)
            assert "second" in loaded
        finally:
            _safe_unlink(path)

    with subtests.test("non-ValueError re-raised"):
        path = _make_tmp_h5()
        try:

            class BadObj:
                def __array__(self, *args, **kwargs):
                    raise TypeError("cannot convert")

            with pytest.raises(TypeError, match="cannot convert"):
                save_h5(path, {"bad": BadObj()})
        finally:
            if os.path.exists(path):
                _safe_unlink(path)



# Test files functions TODO

def test_generate_path(subtests):
    with tempfile.TemporaryDirectory() as tmpdir:
        with subtests.test("first path starts at zero"):
            p = generate_path(tmpdir, "data", extension="h5")
            assert os.path.basename(p) == "data_00000.h5"

        with subtests.test("second call increments id"):
            open(p, "w").close()
            p2 = generate_path(tmpdir, "data", extension="h5")
            assert os.path.basename(p2) == "data_00001.h5"

        with subtests.test("no extension"):
            p3 = generate_path(tmpdir, "run")
            assert os.path.basename(p3) == "run_00000"

        with subtests.test("creates parent directories"):
            subdir = os.path.join(tmpdir, "a", "b", "c")
            generate_path(subdir, "test", extension="txt")
            assert os.path.isdir(subdir)

        with subtests.test("path_count returns list of sequential paths"):
            paths = generate_path(tmpdir, "batch", extension="npy", path_count=3)
            assert len(paths) == 3
            bases = [os.path.basename(q) for q in paths]
            assert bases[1] == bases[0].replace("_00000", "_00001")
            assert bases[2] == bases[0].replace("_00000", "_00002")


def test_latest_path(subtests):
    with tempfile.TemporaryDirectory() as tmpdir:
        with subtests.test("returns None when no files exist"):
            result = latest_path(tmpdir, "missing", extension="h5")
            assert result is None

        with subtests.test("returns highest numeric id"):
            open(os.path.join(tmpdir, "file_00000.h5"), "w").close()
            open(os.path.join(tmpdir, "file_00002.h5"), "w").close()
            result = latest_path(tmpdir, "file", extension="h5")
            assert os.path.basename(result) == "file_00002.h5"

        with subtests.test("ignores files with different names"):
            open(os.path.join(tmpdir, "other_00099.h5"), "w").close()
            result = latest_path(tmpdir, "file", extension="h5")
            assert os.path.basename(result) == "file_00002.h5"


def test_save_load_h5(subtests):
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.h5")

        with subtests.test("roundtrip numeric array"):
            arr = np.array([1.0, 2.0, 3.0])
            save_h5(fpath, {"arr": arr})
            data = load_h5(fpath)
            np.testing.assert_array_equal(data["arr"], arr)

        with subtests.test("roundtrip string decoded to str"):
            save_h5(fpath, {"label": "hello"})
            data = load_h5(fpath)
            assert data["label"] == "hello"
            assert isinstance(data["label"], str)

        with subtests.test("roundtrip nested dict"):
            nested = {"outer": {"inner": np.array([1, 2, 3])}}
            save_h5(fpath, nested)
            data = load_h5(fpath)
            np.testing.assert_array_equal(data["outer"]["inner"], nested["outer"]["inner"])

        with subtests.test("None is written as False"):
            save_h5(fpath, {"nothing": None})
            data = load_h5(fpath)
            assert data["nothing"] == False

        with subtests.test("integer array dtype preserved"):
            arr = np.array([1, 2, 3], dtype=np.int32)
            save_h5(fpath, {"iarr": arr})
            data = load_h5(fpath)
            assert data["iarr"].dtype == np.int32

        with subtests.test("2D array roundtrip"):
            arr2d = np.arange(12, dtype=float).reshape(3, 4)
            save_h5(fpath, {"mat": arr2d})
            data = load_h5(fpath)
            np.testing.assert_array_equal(data["mat"], arr2d)


def test_load_image(subtests):
    """Test _load_image for error handling, basic loading, inversion, and rotation."""
    with subtests.test("file not found raises ValueError"):
        with pytest.raises(ValueError, match="Image not found"):
            _load_image("nonexistent_file.png", (100, 100))

    import cv2

    with subtests.test("loads image to requested shape"):
        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "test.png")
            img = np.zeros((80, 80), dtype=np.uint8)
            img[20:60, 20:60] = 100
            cv2.imwrite(img_path, img)
            result = _load_image(img_path, (100, 100))
            assert result.shape == (100, 100)

    with subtests.test("target_shape zoom reduces image before padding"):
        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "test2.png")
            img = np.zeros((100, 100), dtype=np.uint8)
            img[10:90, 10:90] = 120
            cv2.imwrite(img_path, img)
            # zoom 100→80 via target_shape, then pad to 100
            result = _load_image(img_path, (100, 100), target_shape=(80, 80))
            assert result.shape == (100, 100)

    with subtests.test("bright image is inverted before sqrt"):
        with tempfile.TemporaryDirectory() as d:
            img_path = os.path.join(d, "bright.png")
            # dark image: mean ~30, not inverted; sqrt(30)≈5.5
            img_dark = np.ones((80, 80), dtype=np.uint8) * 30
            cv2.imwrite(img_path, img_dark)
            result_dark = _load_image(img_path, (100, 100))
            # bright image: mean ~220, inverted to ~35; sqrt(35)≈5.9
            img_bright = np.ones((80, 80), dtype=np.uint8) * 220
            cv2.imwrite(img_path, img_bright)
            result_bright = _load_image(img_path, (100, 100))
            # Both should produce similar-magnitude results since inversion normalises
            assert result_dark.shape == (100, 100)
            assert result_bright.shape == (100, 100)
            # Bright image inverts before sqrt; peak values should be comparable
            assert result_bright.max() == pytest.approx(result_dark.max(), rel=0.5)


def test_gray2rgb(subtests):
    """Test _gray2rgb for various inputs, cmaps, lut, normalization, NaN, borders."""
    with subtests.test("2D input reshaped to 3D"):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        result = _gray2rgb(img)
        assert result.ndim == 3
        assert result.shape[0] == 1

    with subtests.test("already RGBA passthrough"):
        img = np.ones((2, 10, 10, 4), dtype=np.uint8) * 100
        result = _gray2rgb(img)
        np.testing.assert_array_equal(result, img)

    with subtests.test("already RGB passthrough"):
        img = np.ones((2, 10, 10, 3), dtype=np.uint8) * 100
        result = _gray2rgb(img)
        np.testing.assert_array_equal(result, img)

    with subtests.test(">3D invalid shape raises RuntimeError"):
        img = np.ones((2, 3, 10, 10, 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="could not be parsed"):
            _gray2rgb(img)

    with subtests.test("cmap=False grayscale"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 128
        result = _gray2rgb(img, cmap=False)
        assert result.dtype == np.uint8

    with subtests.test("cmap=True uses default colormap"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 50
        img[0, 5, 5] = 200
        result = _gray2rgb(img, cmap=True)
        assert result.shape[-1] == 4  # RGBA

    with subtests.test("cmap='default' treated as True"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap="default")
        assert result.shape[-1] == 4

    with subtests.test("cmap='grayscale' treated as False"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap="grayscale")
        assert result.dtype == np.uint8

    with subtests.test("cmap string name"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap="viridis")
        assert result.shape[-1] == 4

    with subtests.test("float image with normalize"):
        img = np.random.rand(1, 10, 10).astype(np.float64)
        result = _gray2rgb(img, cmap="viridis", normalize=True)
        assert result.dtype == np.uint8

    with subtests.test("float image without normalize"):
        img = np.random.rand(1, 10, 10).astype(np.float64) * 0.5
        result = _gray2rgb(img, cmap="viridis", normalize=False)
        assert result.dtype == np.uint8

    with subtests.test("integer image with lut"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap="viridis", lut=100)
        assert result.shape[-1] == 4

    with subtests.test("float image with lut=None uses default"):
        img = np.random.rand(1, 10, 10).astype(np.float64)
        result = _gray2rgb(img, cmap=False)
        assert result.dtype == np.uint8

    with subtests.test("integer image lut=None uses nanmax"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap="viridis")
        assert result.dtype == np.uint8

    with subtests.test("NaN handling sets transparent"):
        img = np.ones((1, 10, 10), dtype=np.float64) * 0.5
        img[0, 3, 3] = np.nan
        result = _gray2rgb(img, cmap="viridis")
        assert result[0, 3, 3, 3] == 0  # alpha channel zero for NaN

    with subtests.test("border scalar"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 100
        result = _gray2rgb(img, cmap="viridis", border=255)
        assert result[0, 0, 0, 0] == 255
        assert result[0, -1, 0, 0] == 255
        assert result[0, 0, -1, 0] == 255

    with subtests.test("border list"):
        img = np.ones((1, 10, 10), dtype=np.uint8) * 100
        result = _gray2rgb(img, cmap="viridis", border=[255, 128])
        assert result[0, 0, 0, 0] == 255
        assert result[0, 0, 0, 1] == 128

    with subtests.test("grayscale cmap=False lut > 256 clamped"):
        img = np.array([[[0, 50], [100, 200]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap=False, lut=300)
        assert result.dtype == np.uint8

    with subtests.test("colormap object with N attribute (ListedColormap)"):
        import matplotlib.pyplot as plt
        cm = plt.get_cmap("viridis", 64)
        img = np.array([[[0, 10], [20, 63]]], dtype=np.uint8)
        result = _gray2rgb(img, cmap=cm, lut=64)
        assert result.shape[-1] == 4

    with subtests.test("colormap without .colors (mock)"):

        class NoColorsCmap:
            """Colormap-like object with N but no colors attr."""
            N = 10

            def __call__(self, x):
                x = np.asarray(x, dtype=float)
                rgba = np.zeros((*x.shape, 4))
                rgba[..., 0] = x / max(self.N, 1)
                rgba[..., 3] = 1.0
                return rgba

        cm = NoColorsCmap()
        img = np.array([[[0, 2], [4, 9]]], dtype=np.int32)
        result = _gray2rgb(img, cmap=cm, lut=10)
        assert result.shape[-1] == 4


def test_save_image(subtests):
    """Test save_image for single images, stacks, and various options."""
    with subtests.test("single grayscale png"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            save_image(path, img)
            assert os.path.exists(path)

    with subtests.test("single image with colormap"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_cmap.png")
            img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            save_image(path, img, cmap="viridis")
            assert os.path.exists(path)

    with subtests.test("stack saved as gif"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.gif")
            imgs = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)
            save_image(path, imgs)
            assert os.path.exists(path)

    with subtests.test("float image"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_float.png")
            img = np.random.rand(10, 10).astype(np.float64)
            save_image(path, img, cmap="viridis")
            assert os.path.exists(path)

    with subtests.test("normalize=False with float"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_nonorm.png")
            img = np.random.rand(10, 10).astype(np.float64) * 0.5
            save_image(path, img, cmap="viridis", normalize=False)
            assert os.path.exists(path)

    with subtests.test("border parameter"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_border.png")
            img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
            save_image(path, img, cmap="viridis", border=255)
            assert os.path.exists(path)

    with subtests.test("imageio not available raises ValueError"):
        import sys
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            with pytest.MonkeyPatch.context() as mp:
                mp.setitem(sys.modules, "imageio", None)
                with pytest.raises(ValueError, match="imageio is required"):
                    save_image(path, img)