"""Unit tests for DataPreprocessor."""
import os

import numpy as np
import pytest

from config import Config
from data_preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor(Config())


def test_split_has_no_overlap(preprocessor):
    train, val, test = preprocessor.split_data_indices(40)
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    assert sorted(train + val + test) == list(range(40))


def test_split_raises_on_overlap(preprocessor):
    preprocessor.config.TEST_BATTERY_START = 0  # overlap with val start 0
    with pytest.raises(ValueError, match="overlap"):
        preprocessor.split_data_indices(40)


def test_normalize_then_apply_roundtrip(preprocessor):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 30, 30, 3)).astype(np.float32) * 5 + 2
    normalized, params = preprocessor.normalize_data(data, method='minmax')
    assert normalized.min() >= 0.0 - 1e-6
    assert normalized.max() <= 1.0 + 1e-6
    restored = preprocessor.denormalize_data(normalized, params)
    np.testing.assert_allclose(restored, data, rtol=1e-5, atol=1e-5)


def test_apply_normalization_matches_fit(preprocessor):
    rng = np.random.default_rng(1)
    train = rng.normal(size=(8, 30, 30, 3)).astype(np.float32)
    fitted, params = preprocessor.normalize_data(train, method='minmax')
    applied = preprocessor.apply_normalization(train, params)
    np.testing.assert_allclose(fitted, applied, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cycle_shape", [(5, 1), (1, 5)])
def test_extract_discharge_data_tolerates_scipy_layouts(preprocessor, tmp_path, cycle_shape):
    """extract_discharge_data must work on scipy.io.savemat output, which
    promotes 1-D struct arrays to 2-D with either (N,1) or (1,N) layout."""
    from scipy.io import loadmat, savemat

    cyc_dtype = [('V', 'O'), ('T', 'O'), ('Qd', 'O')]
    n = 5
    V = np.linspace(3.7, 1.9, 1000).reshape(-1, 1)
    T = (25 + np.zeros_like(V))
    Qd = np.linspace(0, 1.1, 1000).reshape(-1, 1)
    cycle_arr = np.empty(cycle_shape, dtype=cyc_dtype)
    for idx in np.ndindex(cycle_shape):
        cycle_arr[idx] = (V, T, Qd)
    wrapped = np.empty((1, 1), dtype=object)
    wrapped[0, 0] = cycle_arr
    top = np.empty((1, 1), dtype=[('cycles', 'O')])
    top[0, 0] = (wrapped,)
    path = tmp_path / "sample.mat"
    savemat(str(path), {'batteryDischargeData': top})

    bd = loadmat(str(path))['batteryDischargeData']
    out = preprocessor.extract_discharge_data(bd)
    assert len(out) == 1
    assert len(out[0]['Vd']) == n


def test_save_and_load_norm_params(preprocessor, tmp_path):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(4, 30, 30, 3)).astype(np.float32)
    _, params = preprocessor.normalize_data(data, method='minmax')
    path = tmp_path / "norm.npz"
    preprocessor.save_norm_params(params, str(path))
    loaded = preprocessor.load_norm_params(str(path))
    assert loaded['method'] == params['method']
    np.testing.assert_allclose(loaded['min'], params['min'])
    np.testing.assert_allclose(loaded['max'], params['max'])
