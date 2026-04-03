"""Tests for contracts.round_metric()."""

import math

from p2p.contracts import round_metric


def test_large_return():
    assert round_metric(1003.5914303455962) == 1003.6


def test_small_score():
    assert round_metric(0.41) == 0.41


def test_tiny_value():
    assert round_metric(0.001234567) == 0.0012346


def test_zero():
    assert round_metric(0.0) == 0.0


def test_negative():
    assert round_metric(-123.456789) == -123.46


def test_integer_passthrough():
    assert round_metric(42) == 42


def test_none_passthrough():
    assert round_metric(None) is None


def test_nan_passthrough():
    result = round_metric(float("nan"))
    assert math.isnan(result)


def test_inf_passthrough():
    assert round_metric(float("inf")) == float("inf")


def test_float_with_many_digits():
    assert round_metric(12345.6789) == 12346.0


def test_idempotent():
    v = 1003.5914303455962
    assert round_metric(round_metric(v)) == round_metric(v)


def test_custom_sig():
    assert round_metric(1003.5914, sig=3) == 1000.0
    assert round_metric(1003.5914, sig=6) == 1003.59
