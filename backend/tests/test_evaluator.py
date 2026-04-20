"""Unit tests for ModelEvaluator metrics behavior."""
import json
import os

import numpy as np
import pytest

from config import Config
from evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator(Config())


def test_metrics_on_perfect_predictions(evaluator):
    y = np.array([100.0, 200.0, 300.0])
    m = evaluator.calculate_metrics(y, y.copy())
    assert m['MSE'] == 0
    assert m['RMSE'] == 0
    assert m['MAE'] == 0
    assert m['MAPE'] == 0.0
    assert m['R2_Score'] == 1.0


def test_mape_handles_all_zero_truth(evaluator):
    y_true = np.zeros(5)
    y_pred = np.array([0.1, 0.2, 0.0, 0.3, 0.0])
    m = evaluator.calculate_metrics(y_true, y_pred)
    assert np.isnan(m['MAPE'])


def test_mape_ignores_zero_entries(evaluator):
    y_true = np.array([0.0, 100.0, 100.0])
    y_pred = np.array([5.0, 110.0, 90.0])
    m = evaluator.calculate_metrics(y_true, y_pred)
    # Expected: average of |10/100| and |10/100| = 10.0%
    assert m['MAPE'] == pytest.approx(10.0)


def test_report_is_json_serializable_with_zero_truth(evaluator, tmp_path):
    y_true = np.zeros(3)
    y_pred = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "report.json"
    evaluator.create_evaluation_report(y_true, y_pred, save_path=str(path))
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    # MAPE should be null (sanitized from NaN), not a Python float('nan')
    assert data['metrics']['MAPE'] is None
