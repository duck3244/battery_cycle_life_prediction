"""Unit tests for utility helpers."""
import json
import math

import numpy as np

from utils import sanitize_for_json, set_global_seed


def test_sanitize_replaces_inf_and_nan():
    data = {'a': float('nan'), 'b': float('inf'), 'c': -float('inf'), 'd': 1.5}
    cleaned = sanitize_for_json(data)
    assert cleaned['a'] is None
    assert cleaned['b'] is None
    assert cleaned['c'] is None
    assert cleaned['d'] == 1.5
    # round-trips through json
    json.dumps(cleaned)


def test_sanitize_handles_numpy_types():
    data = {
        'arr': np.array([1.0, float('nan'), 3.0]),
        'int': np.int64(5),
        'float': np.float32(2.5),
    }
    cleaned = sanitize_for_json(data)
    assert cleaned['arr'] == [1.0, None, 3.0]
    assert cleaned['int'] == 5
    assert cleaned['float'] == 2.5


def test_set_global_seed_is_deterministic():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)
