"""Environment-driven configuration is validated with clear errors."""

from __future__ import annotations

import pytest

from app.config import MAX_TOP_K, _int_env


def test_unset_variable_uses_default(monkeypatch):
    monkeypatch.delenv("FV_TEST_INT", raising=False)
    assert _int_env("FV_TEST_INT", 7) == 7


def test_blank_variable_uses_default(monkeypatch):
    monkeypatch.setenv("FV_TEST_INT", "  ")
    assert _int_env("FV_TEST_INT", 7) == 7


def test_valid_value_is_parsed(monkeypatch):
    monkeypatch.setenv("FV_TEST_INT", "3")
    assert _int_env("FV_TEST_INT", 7) == 3


@pytest.mark.parametrize("value", ["1", str(MAX_TOP_K)])
def test_bounds_are_inclusive(monkeypatch, value):
    monkeypatch.setenv("FV_TEST_INT", value)
    assert _int_env("FV_TEST_INT", 5, maximum=MAX_TOP_K) == int(value)


def test_non_integer_is_rejected_with_the_variable_name(monkeypatch):
    monkeypatch.setenv("FV_TEST_INT", "five")
    with pytest.raises(ValueError, match="FV_TEST_INT must be an integer"):
        _int_env("FV_TEST_INT", 5)


@pytest.mark.parametrize("value", ["0", "-2", str(MAX_TOP_K + 1)])
def test_out_of_range_is_rejected(monkeypatch, value):
    monkeypatch.setenv("FV_TEST_INT", value)
    with pytest.raises(ValueError, match="FV_TEST_INT must be >= 1"):
        _int_env("FV_TEST_INT", 5, maximum=MAX_TOP_K)
