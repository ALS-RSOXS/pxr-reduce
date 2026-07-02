import numpy as np
import pytest

from pxr_reduce.io.fits_io import ImageStore, read_fits, read_fits_header, read_fits_image


def test_read_fits_header_and_image(synthetic_fits_factory):
    image = np.arange(9, dtype=float).reshape(3, 3)
    path = synthetic_fits_factory(0, image, header={"EXPOSURE": 1.5, "Sample Theta": 2.0})

    header = read_fits_header(path)
    assert header["EXPOSURE"] == 1.5
    assert header["Sample Theta"] == 2.0

    loaded = read_fits_image(path)
    np.testing.assert_array_equal(loaded, image)


def test_read_fits_returns_both(synthetic_fits_factory):
    image = np.ones((4, 4), dtype=float)
    path = synthetic_fits_factory(1, image, header={"EXPOSURE": 0.5})
    header, img = read_fits(path)
    assert header["EXPOSURE"] == 0.5
    np.testing.assert_array_equal(img, image)


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fits_image(tmp_path / "nope.fits")


@pytest.fixture
def store(synthetic_fits_factory):
    paths = {}
    for i in range(5):
        img = np.full((3, 3), i, dtype=float)
        paths[i] = synthetic_fits_factory(i, img)
    return ImageStore(paths, cache_size=2)


def test_store_len_and_contains(store):
    assert len(store) == 5
    assert 3 in store
    assert 99 not in store


def test_store_indices_sorted(store):
    assert store.indices() == [0, 1, 2, 3, 4]


def test_store_get_returns_correct_image(store):
    np.testing.assert_array_equal(store.get(2), np.full((3, 3), 2.0))
    np.testing.assert_array_equal(store[4], np.full((3, 3), 4.0))


def test_store_missing_index_raises(store):
    with pytest.raises(KeyError):
        store.get(42)


def test_store_cache_is_bounded(store):
    # cache_size=2; touch three distinct images
    store.get(0)
    store.get(1)
    store.get(2)
    assert len(store._cache) == 2
    # most recently used retained
    assert 2 in store._cache
    assert 0 not in store._cache


def test_store_stack(store):
    stacked = store.stack([0, 1, 2])
    assert stacked.shape == (3, 3, 3)
    np.testing.assert_array_equal(stacked[1], np.full((3, 3), 1.0))


def test_store_iter_images(store):
    seen = dict(store.iter_images())
    assert set(seen) == {0, 1, 2, 3, 4}
    np.testing.assert_array_equal(seen[3], np.full((3, 3), 3.0))


def test_store_clear_cache(store):
    store.get(0)
    assert len(store._cache) >= 1
    store.clear_cache()
    assert len(store._cache) == 0


def test_store_rejects_bad_cache_size(synthetic_fits_factory):
    path = synthetic_fits_factory(0, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ImageStore({0: path}, cache_size=0)
