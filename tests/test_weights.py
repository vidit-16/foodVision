"""Weight resolution: download, caching and failure paths, with the network mocked."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from app import model as model_module
from app.model import WeightsUnavailableError, load_classes, resolve_weights

PAYLOAD = b"pretend checkpoint bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(model_module, "WEIGHTS_PATH_OVERRIDE", None)
    monkeypatch.setattr(model_module, "WEIGHTS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(model_module, "WEIGHTS_SHA256", PAYLOAD_SHA)
    return tmp_path


def _fake_urlopen(payload: bytes, calls: list):
    def urlopen(url, timeout=None):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    return urlopen


def test_download_writes_verified_file_and_passes_a_timeout(monkeypatch, cache):
    calls: list = []
    monkeypatch.setattr(model_module.urllib.request, "urlopen", _fake_urlopen(PAYLOAD, calls))
    path = resolve_weights()
    assert path.read_bytes() == PAYLOAD
    assert calls and calls[0][1] == model_module.DOWNLOAD_TIMEOUT_SECONDS
    assert not list(cache.glob("*.part"))


def test_cached_file_is_reused_without_network(monkeypatch, cache):
    (cache / model_module.WEIGHTS_FILENAME).write_bytes(PAYLOAD)

    def boom(*args, **kwargs):
        raise AssertionError("network should not be touched")

    monkeypatch.setattr(model_module.urllib.request, "urlopen", boom)
    assert resolve_weights().read_bytes() == PAYLOAD


def test_corrupt_cache_is_replaced_by_a_fresh_download(monkeypatch, cache):
    (cache / model_module.WEIGHTS_FILENAME).write_bytes(b"stale")
    calls: list = []
    monkeypatch.setattr(model_module.urllib.request, "urlopen", _fake_urlopen(PAYLOAD, calls))
    assert resolve_weights().read_bytes() == PAYLOAD
    assert len(calls) == 1


def test_network_failure_raises_and_leaves_no_partial_file(monkeypatch, cache):
    def offline(url, timeout=None):
        raise OSError("network unreachable")

    monkeypatch.setattr(model_module.urllib.request, "urlopen", offline)
    with pytest.raises(WeightsUnavailableError, match="could not download"):
        resolve_weights()
    assert list(cache.iterdir()) == []


def test_download_with_wrong_checksum_is_discarded(monkeypatch, cache):
    monkeypatch.setattr(
        model_module.urllib.request, "urlopen", _fake_urlopen(b"tampered", [])
    )
    with pytest.raises(WeightsUnavailableError, match="checksum mismatch"):
        resolve_weights()
    assert list(cache.iterdir()) == []


def test_empty_sha_skips_verification(monkeypatch, tmp_path):
    local = tmp_path / "w.pt"
    local.write_bytes(b"anything")
    monkeypatch.setattr(model_module, "WEIGHTS_PATH_OVERRIDE", str(local))
    monkeypatch.setattr(model_module, "WEIGHTS_SHA256", "")
    assert resolve_weights() == local


@pytest.mark.parametrize(
    "content, error",
    [(None, FileNotFoundError), ("{not json", ValueError), ("[]", ValueError),
     ('{"a": 1}', ValueError), ("[1, 2]", ValueError)],
)
def test_bad_class_files_fail_with_clear_errors(monkeypatch, tmp_path, content, error):
    path = tmp_path / "classes.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(model_module, "CLASSES_PATH", path)
    load_classes.cache_clear()
    try:
        with pytest.raises(error):
            load_classes()
    finally:
        load_classes.cache_clear()


def test_valid_class_file_loads(monkeypatch, tmp_path):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    monkeypatch.setattr(model_module, "CLASSES_PATH", path)
    load_classes.cache_clear()
    try:
        assert load_classes() == ["a", "b"]
    finally:
        load_classes.cache_clear()
