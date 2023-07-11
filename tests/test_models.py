"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[4, 2], [7, 4], [10, 6]], [10, 6]),
        ([[-4, 2], [-1, 4], [3, -1]], [3, 4]),
    ])
def test_daily_max(test,expected):
    """Test that max function works for an array of zeros, positive, and negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[4, 2], [7, 4], [10, 6]], [4, 2]),
        ([[-4, 2], [-1, 4], [3, -1]], [-4, -1]),
    ])
def test_daily_min(test,expected):
    """Test that min function works for an array of zeros, positive, and negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hey', 'What'], ['is', 'up']])


@pytest.mark.parametrize(
    "test, expected, expected_raises",
    [   # None for expected raises without negative values, otherwise ValueError
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None,),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None,),
        ([[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
         [[0, 1, 1], [1, 1, 1], [1, 1, 1]], None,),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], ValueError,),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None,)
    ])
def test_patient_normalize(test, expected, expected_raises):
    """Test normalization works for arrays of one and positive integers.
    Assumption that test accuracy of two decimals is sufficient."""
    from inflammation.models import patient_normalize
    if expected_raises is not None:
        with pytest.raises(expected_raises):
            npt.assert_almost_equal(patient_normalize(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalize(np.array(test)), np.array(expected), decimal=2)