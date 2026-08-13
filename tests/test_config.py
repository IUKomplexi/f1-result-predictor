"""Tests for f1core.config: writer/reader roundtrip, validation, atomic write."""

from __future__ import annotations

from f1core.config import (
    DEFAULTS,
    config_to_toml,
    load_config,
    save_config,
    validate_config,
)


def test_roundtrip_preserves_all_sections(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config()
    save_config(cfg, p)
    back = load_config(p)
    # load_config merges defaults, so compare only the keys the writer emitted.
    assert back["model"]["params"] == cfg["model"]["params"]
    assert back["api"]["base_url"] == cfg["api"]["base_url"]
    assert back["data"]["start_season"] == cfg["data"]["start_season"]
    assert back["report"]["backtest"] == cfg["report"]["backtest"]


def test_none_features_written_as_comment(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config()
    cfg["features"]["enabled"] = None
    save_config(cfg, p)
    text = p.read_text(encoding="utf-8")
    assert "\nenabled = " not in text  # no uncommented enabled assignment
    assert "registry defaults" in text
    # Reading it back yields None (registry-default fallback).
    back = load_config(p)
    assert back["features"]["enabled"] is None


def test_explicit_features_roundtrip(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config()
    cfg["features"]["enabled"] = ["grid", "driver_id"]
    save_config(cfg, p)
    back = load_config(p)
    assert back["features"]["enabled"] == ["grid", "driver_id"]


def test_atomic_write_creates_no_temp_leftover(tmp_path):
    p = tmp_path / "config.toml"
    save_config(load_config(), p)
    leftovers = [f for f in tmp_path.iterdir() if f.name != "config.toml"]
    assert leftovers == []


def test_invalid_config_raises_and_does_not_write(tmp_path):
    p = tmp_path / "config.toml"
    bad = {**load_config(), "features": {"enabled": ["nope"]}}
    try:
        save_config(bad, p)
    except ValueError as exc:
        assert "unknown feature" in str(exc)
    else:  # pragma: no cover - failure would silently write a bad file
        raise AssertionError("save_config should have raised")
    assert not p.exists()


def test_validate_config_detects_bad_types_and_ranges():
    assert validate_config({"data": {"start_season": 2030, "end_season": 2020}})
    assert validate_config({"api": {"timeout": "soon"}})
    assert validate_config({"model": {"params": {"bogus": 1}}})
    assert validate_config({"model": {"params": {"max_iter": "many"}}})
    assert validate_config({}) == []


def test_validate_config_rejects_non_integral_and_bool_ints():
    # int-typed fields must be integral and not booleans (bool is an int subclass).
    assert validate_config({"data": {"start_season": 2026.5}})
    assert validate_config({"data": {"start_season": True}})
    assert validate_config({"api": {"max_retries": 2.5}})
    # Integral floats are fine for int fields.
    assert validate_config({"data": {"start_season": 2026.0}}) == []
    assert validate_config({"data": {"start_season": 2026}}) == []


def test_config_to_toml_writes_model_params_subtable():
    cfg = DEFAULTS
    text = config_to_toml(cfg)
    assert "[model.params]" in text
    assert "learning_rate = 0.03" in text
    assert "[api]" in text and "[weather]" in text  # section order preserved


# --------------------------------------------------------------------------
# run_* wrappers resolve their path/season defaults from config (no explicit
# args) — config.toml is the single source of truth for the pipeline.
# --------------------------------------------------------------------------


def test_run_fetch_resolves_defaults_from_config(monkeypatch):
    import f1data.fetch as fetch_module

    cfg = {
        "api": {"user_agent": "test", "sleep_seconds": 0.5},
        "data": {"cache_dir": "custom/raw", "start_season": 2011, "end_season": 2012},
    }
    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(fetch_module, "F1Client", FakeClient)
    monkeypatch.setattr(
        fetch_module, "fetch_season",
        lambda client, season: {"calendar": [], "results": {}, "sprints": []},
    )
    result = fetch_module.run(cfg=cfg)  # every run arg left at its None default

    assert captured["cache_dir"] == "custom/raw"
    assert captured["sleep_seconds"] == 0.5
    assert result["start"] == 2011 and result["end"] == 2012


def test_model_params_reads_config_only():
    from f1core.config import load_config
    from model.train import model_params

    params = model_params(load_config())  # repo defaults
    assert params["max_iter"] == DEFAULTS["model"]["params"]["max_iter"]
    overridden = {
        "model": {"params": {"max_iter": 123}},
    }
    assert model_params(overridden)["max_iter"] == 123
