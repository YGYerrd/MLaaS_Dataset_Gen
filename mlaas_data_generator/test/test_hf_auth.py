from __future__ import annotations

import os

from mlaas_data_generator.hf_auth import load_hf_token_from_file, resolve_hf_token_file


def test_resolve_hf_token_file_uses_env_override(monkeypatch, tmp_path):
    token_file = tmp_path / "custom.token"
    monkeypatch.setenv("MLAAS_HF_TOKEN_FILE", str(token_file))

    assert resolve_hf_token_file() == token_file.resolve()


def test_load_hf_token_from_file_sets_env_vars(monkeypatch, tmp_path):
    token_file = tmp_path / ".hf_token"
    token_file.write_text("# comment\nHF_TOKEN=hf_test_token\n", encoding="utf-8")
    monkeypatch.setenv("MLAAS_HF_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = load_hf_token_from_file(quiet=True)

    assert token == "hf_test_token"
    assert os.getenv("HF_TOKEN") == "hf_test_token"
    assert os.getenv("HUGGING_FACE_HUB_TOKEN") == "hf_test_token"


def test_load_hf_token_from_file_preserves_existing_env(monkeypatch, tmp_path):
    token_file = tmp_path / ".hf_token"
    token_file.write_text("hf_from_file\n", encoding="utf-8")
    monkeypatch.setenv("MLAAS_HF_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = load_hf_token_from_file(quiet=True)

    assert token == "hf_from_env"
    assert os.getenv("HF_TOKEN") == "hf_from_env"
    assert os.getenv("HUGGING_FACE_HUB_TOKEN") is None


def test_load_hf_token_from_file_ignores_unknown_key(monkeypatch, tmp_path):
    token_file = tmp_path / ".hf_token"
    token_file.write_text("OTHER_KEY=value\nHF_TOKEN=hf_after_unknown\n", encoding="utf-8")
    monkeypatch.setenv("MLAAS_HF_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = load_hf_token_from_file(quiet=True)

    assert token == "hf_after_unknown"
