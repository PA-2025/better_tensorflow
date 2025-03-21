import pytest
import better_tensorflow


def test_sum_as_string():
    assert better_tensorflow.sum_as_string(1, 1) == "2"
