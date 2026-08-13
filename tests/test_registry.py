"""Tests for the feature registry: completeness, selection, fingerprint keying."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features
from features.registry import (
    CATEGORIES,
    REGISTRY,
    all_feature_ids,
    category_meta,
    default_enabled,
    enabled_features,
    feature_fingerprint,
)
from model.train import (
    FEATURES,
    load_checkpoint,
    prepare,
    save_checkpoint,
    train_final_model,
)


def _mini_df() -> pd.DataFrame:
    """Two drivers, same team, three races (leakage-safe synthetic history)."""
    return add_features(
        pd.DataFrame(
            [
                # season round date circuit driver team grid pos points status sprint_points
                [2020, 1, "2020-03-01", "c1", "a", "t1", 1, 1, 25.0, "Finished", 0.0],
                [2020, 1, "2020-03-01", "c1", "b", "t1", 2, 2, 18.0, "Finished", 0.0],
                [2020, 2, "2020-03-08", "c2", "a", "t1", 2, 3, 15.0, "Finished", 0.0],
                [2020, 2, "2020-03-08", "c2", "b", "t1", 1, 1, 25.0, "Finished", 0.0],
                [2020, 3, "2020-03-15", "c1", "a", "t1", 1, 1, 25.0, "Finished", 0.0],
                [2020, 3, "2020-03-15", "c1", "b", "t1", 2, 2, 18.0, "Finished", 0.0],
            ],
            columns=[
                "season", "round", "date", "circuit_id", "driver_id",
                "constructor_id", "grid", "position", "points", "status",
                "sprint_points",
            ],
        )
    )


def test_registry_complete_and_consistent():
    """Registry == NUMERIC + CATEGORICAL, in the same order and kinds."""
    expected_ids = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert all_feature_ids() == expected_ids
    assert FEATURES == expected_ids  # model.train derives from the registry
    for spec in REGISTRY:
        assert spec.id in expected_ids
        assert spec.category in CATEGORIES
        assert spec.builder, f"{spec.id} has no builder"
        assert spec.rationale, f"{spec.id} has no rationale"
        # kind must match where the column lives in build.py
        assert (spec.kind == "numeric") == (spec.id in NUMERIC_FEATURES)
        # default follows the category: only core is on by default
        assert spec.default == (spec.category == "core")
    # every category is actually used
    used = {spec.category for spec in REGISTRY}
    assert used <= set(CATEGORIES)


def test_category_meta_covers_all_categories_in_order():
    """category_meta() drives the dashboard feature groups: every category
    has a label, in CATEGORIES display order, with no extras."""
    meta = category_meta()
    assert [m["id"] for m in meta] == list(CATEGORIES)
    assert all(m["label"] for m in meta)
    assert all(m["id"] in CATEGORIES for m in meta)


def test_defaults_roundtrip():
    """No config ⇒ registry defaults; explicit config; CLI overrides."""
    assert enabled_features({}) == default_enabled()
    assert enabled_features(None) == default_enabled()
    # An empty explicit list also falls back to the registry defaults (an
    # empty model matrix would be a footgun, not a configuration).
    assert enabled_features({"features": {"enabled": []}}) == default_enabled()
    # Explicit config list wins (validated, registry order preserved).
    explicit = enabled_features({"features": {"enabled": ["grid", "season"]}})
    assert explicit == ["grid", "season"]
    # Overrides apply on top of defaults and are deduplicated.
    over = enabled_features({}, enable=["season"], disable=["grid"])
    assert "season" in over and "grid" not in over
    assert over == [f for f in all_feature_ids() if f in set(over)]
    # Disabling is idempotent and safe whether or not the feature is on.
    without_season = enabled_features({}, disable=["season"])
    assert without_season == [f for f in default_enabled() if f != "season"]
    # Unknown ids are rejected in every position.
    with pytest.raises(ValueError, match="unknown"):
        enabled_features({"features": {"enabled": ["not_a_feature"]}})
    with pytest.raises(ValueError, match="unknown"):
        enabled_features({}, enable=["not_a_feature"])
    with pytest.raises(ValueError, match="unknown"):
        enabled_features({}, disable=["not_a_feature"])


def test_fingerprint_stable_and_sensitive_to_selection():
    fp_full = feature_fingerprint(all_feature_ids())
    assert fp_full == feature_fingerprint(all_feature_ids())  # deterministic
    assert len(fp_full) == 12
    subset = all_feature_ids()[:-1]  # drop the last feature
    assert feature_fingerprint(subset) != fp_full
    # Order matters: the same set in a different order is a different model.
    reordered = all_feature_ids()[::-1]
    assert feature_fingerprint(reordered) != fp_full


def test_toggling_changes_matrix_columns():
    df = _mini_df()
    subset = ["grid", "driver_id"]
    X_full, _ = prepare(df)
    X_sub, _ = prepare(df, features=subset)
    assert set(X_sub.columns) == set(subset)
    assert set(X_full.columns) == set(all_feature_ids())
    assert "grid" in X_full.columns and "season" not in X_sub.columns
    # Categorical conversion only applies to selected categoricals.
    assert str(X_sub["driver_id"].dtype) == "category"
    assert str(X_sub["grid"].dtype) != "category"


def test_prepare_rejects_unknown_feature():
    df = _mini_df()
    with pytest.raises(ValueError, match="not present in the dataset"):
        prepare(df, features=["grid", "bogus_feature"])


def test_checkpoint_fingerprint_roundtrip_and_mismatch(tmp_path):
    df = _mini_df()
    full = all_feature_ids()
    subset = full[:-2]
    model = train_final_model(df, features=full)

    path = tmp_path / "hurdle.joblib"
    save_checkpoint(model, path, features=full)
    # Same selection loads fine and predicts identically.
    loaded = load_checkpoint(path, expected=full)
    X, _ = prepare(df, features=full)
    np.testing.assert_allclose(
        model.predict_expected_points(X), loaded.predict_expected_points(X)
    )
    # A different selection must be rejected, not silently reused.
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, expected=subset)
    # Default expected = full set also matches.
    load_checkpoint(path)


def test_legacy_checkpoint_without_fingerprint_still_validates(tmp_path):
    """Pre-registry checkpoints (features list only) keep working."""
    df = _mini_df()
    model = train_final_model(df)
    path = tmp_path / "legacy.joblib"
    joblib.dump({"models": model, "features": list(all_feature_ids())}, path)
    loaded = load_checkpoint(path)  # fingerprint derived from the stored list
    X, _ = prepare(df)
    np.testing.assert_allclose(
        model.predict_expected_points(X), loaded.predict_expected_points(X)
    )
    joblib.dump({"models": model, "features": ["old_feature"]}, path)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path)


def test_dataset_cache_fingerprint_roundtrip(tmp_path):
    """build_dataset embeds the feature fingerprint; stale caches rebuild."""
    from features import build as fb

    mini = {
        "calendar": [{"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                      "race_name": "R1", "is_sprint_round": False}],
        "results": {1: [{"season": 2020, "round": 1, "position": 1, "grid": 1,
                         "points": 25.0, "status": "Finished", "driver_id": "a",
                         "constructor_id": "t1"}]},
        "qualifying": {}, "sprints": {},
    }
    monkeypatch_seen = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fb, "fetch_season",
                        lambda client, s: (monkeypatch_seen.append(s), mini)[1])
    try:
        cache = tmp_path / "feat.parquet"
        df1 = fb.build_dataset(object(), [2020], cache_path=cache)
        assert fb._read_cache_fingerprint(cache) == fb._dataset_fingerprint()
        # Second load is served from cache (no refetch).
        fb.build_dataset(object(), [2020], cache_path=cache)
        assert monkeypatch_seen == [2020]
        # A cache stamped with the wrong fingerprint must be rebuilt.
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df1)
        meta = dict(table.schema.metadata or {})
        meta[b"feature_fingerprint"] = b"deadbeef"
        pq.write_table(table.replace_schema_metadata(meta), cache)
        fb.build_dataset(object(), [2020], cache_path=cache)
        assert monkeypatch_seen == [2020, 2020]
    finally:
        monkeypatch.undo()


def test_dataset_cache_rebuilds_when_raw_cache_is_newer(tmp_path):
    """Newer raw responses than the parquet force a rebuild (fresh race data).

    ``f1 fetch`` after a race weekend adds raw JSON without changing the
    feature fingerprint or season coverage; the dataset must still rebuild.
    """
    import os

    from features import build as fb

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    class FakeClient:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

    mini = {
        "calendar": [{"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                      "race_name": "R1", "is_sprint_round": False}],
        "results": {1: [{"season": 2020, "round": 1, "position": 1, "grid": 1,
                         "points": 25.0, "status": "Finished", "driver_id": "a",
                         "constructor_id": "t1"}]},
        "qualifying": {}, "sprints": {},
    }
    monkeypatch_seen = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fb, "fetch_season",
                        lambda client, s: (monkeypatch_seen.append(s), mini)[1])
    try:
        cache = tmp_path / "feat.parquet"
        client = FakeClient(raw_dir)
        fb.build_dataset(client, [2020], cache_path=cache)
        assert monkeypatch_seen == [2020]
        # No new raw data: the cache is reused.
        fb.build_dataset(client, [2020], cache_path=cache)
        assert monkeypatch_seen == [2020]
        # A new raw response, deterministically newer than the parquet.
        fresh = raw_dir / "fresh.json"
        fresh.write_text("{}")
        os.utime(fresh, (cache.stat().st_mtime + 10, cache.stat().st_mtime + 10))
        fb.build_dataset(client, [2020], cache_path=cache)
        assert monkeypatch_seen == [2020, 2020]
    finally:
        monkeypatch.undo()
