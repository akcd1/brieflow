"""Unit tests for joint class alignment helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_DIR))

from lib.aggregate.align import stratified_subsample


def test_stratified_subsample_respects_total_budget():
    classes = np.array(["A"] * 50_000 + ["B"] * 50_000 + ["C"] * 50_000)
    rng = np.random.default_rng(0)
    idx = stratified_subsample(classes, n_total=30_000, min_per_class=5_000, rng=rng)
    assert len(idx) == 30_000
    assert len(set(idx)) == len(idx)  # no duplicates


def test_stratified_subsample_meets_floor_for_rare_classes():
    # Class "R" has only 2000 rows — below floor of 5000
    # Class "A","B" each have 100_000 rows
    classes = np.array(["R"] * 2_000 + ["A"] * 100_000 + ["B"] * 100_000)
    rng = np.random.default_rng(0)
    idx = stratified_subsample(classes, n_total=30_000, min_per_class=5_000, rng=rng)

    picked_classes = classes[idx]
    # R contributes all 2000
    assert (picked_classes == "R").sum() == 2_000
    # A and B share the remaining 28_000 proportionally (50/50)
    assert abs((picked_classes == "A").sum() - 14_000) < 100
    assert abs((picked_classes == "B").sum() - 14_000) < 100


def test_stratified_subsample_floor_applies_when_possible():
    # All classes large enough to meet floor
    classes = np.array(["A"] * 20_000 + ["B"] * 20_000 + ["C"] * 20_000)
    rng = np.random.default_rng(0)
    idx = stratified_subsample(classes, n_total=15_000, min_per_class=5_000, rng=rng)
    picked = classes[idx]
    assert (picked == "A").sum() == 5_000
    assert (picked == "B").sum() == 5_000
    assert (picked == "C").sum() == 5_000


def test_stratified_subsample_deterministic_with_same_seed():
    classes = np.array(["A"] * 10_000 + ["B"] * 10_000)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    idx1 = stratified_subsample(classes, n_total=1_000, min_per_class=100, rng=rng1)
    idx2 = stratified_subsample(classes, n_total=1_000, min_per_class=100, rng=rng2)
    np.testing.assert_array_equal(idx1, idx2)


def test_stratified_subsample_total_greater_than_rows_returns_all():
    classes = np.array(["A"] * 100 + ["B"] * 100)
    rng = np.random.default_rng(0)
    idx = stratified_subsample(classes, n_total=1_000, min_per_class=10, rng=rng)
    assert len(idx) == 200


from lib.aggregate.aggregate import aggregate as aggregate_fn


def test_aggregate_default_groups_by_pert_col_only():
    metadata = pd.DataFrame(
        {
            "pert": ["g1", "g1", "g2", "g2"],
            "class": ["A", "B", "A", "B"],
        }
    )
    embeddings = np.array([[1.0], [3.0], [5.0], [7.0]])
    emb, meta = aggregate_fn(embeddings, metadata, "pert", method="mean")
    assert len(meta) == 2  # g1, g2 (class ignored)
    assert set(meta["pert"]) == {"g1", "g2"}


def test_aggregate_groups_by_class_and_pert():
    metadata = pd.DataFrame(
        {
            "pert": ["g1", "g1", "g2", "g2"],
            "class": ["A", "B", "A", "B"],
        }
    )
    embeddings = np.array([[1.0], [3.0], [5.0], [7.0]])
    emb, meta = aggregate_fn(
        embeddings,
        metadata,
        "pert",
        method="mean",
        group_cols=["class", "pert"],
    )
    assert len(meta) == 4  # each (class, pert) is its own row
    assert set(zip(meta["class"], meta["pert"])) == {
        ("A", "g1"),
        ("B", "g1"),
        ("A", "g2"),
        ("B", "g2"),
    }
    # Values: (A,g1)=1, (B,g1)=3, (A,g2)=5, (B,g2)=7
    row_map = {(r["class"], r["pert"]): emb[i, 0] for i, r in meta.iterrows()}
    assert row_map[("A", "g1")] == 1.0
    assert row_map[("B", "g2")] == 7.0


def test_aggregate_group_cols_preserves_cell_count():
    metadata = pd.DataFrame(
        {
            "pert": ["g1"] * 6,
            "class": ["A", "A", "A", "B", "B", "B"],
        }
    )
    embeddings = np.ones((6, 2))
    emb, meta = aggregate_fn(
        embeddings,
        metadata,
        "pert",
        method="mean",
        group_cols=["class", "pert"],
    )
    assert len(meta) == 2
    assert set(meta["cell_count"]) == {3}
