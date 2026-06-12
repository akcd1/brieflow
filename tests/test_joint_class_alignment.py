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


from lib.aggregate.align import validate_joint_align_config


def test_validate_joint_align_config_rejects_multi_chunk_joint():
    with pytest.raises(ValueError, match="num_align_batches=2 is invalid in joint"):
        validate_joint_align_config(is_joint=True, num_align_batches=2)


def test_validate_joint_align_config_allows_single_chunk_joint():
    # No raise expected.
    validate_joint_align_config(is_joint=True, num_align_batches=1)


def test_validate_joint_align_config_allows_multi_chunk_non_joint():
    # Non-joint may chunk freely; the per-chunk TVN concern is joint-only.
    validate_joint_align_config(is_joint=False, num_align_batches=8)


from lib.aggregate.align import tvn_on_controls_joint


def _make_joint_synthetic(n_per_class=400, n_features=8, seed=0):
    """Two classes with different baselines and different control covariances.

    - Class A controls: mean=0, isotropic spread.
    - Class B controls: mean shifted along feature 0, anisotropic spread (3x along feature 1).
    - Each class has 50% NT controls labelled "nontargeting" and 50% perturbations.
    - Two batches per class so per-batch CORAL has data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    feats = []
    for cls, (mean_shift, scale_y) in [("A", (0.0, 1.0)), ("B", (5.0, 3.0))]:
        for batch in ["b1", "b2"]:
            n_ctrl = n_per_class // 4
            n_pert = n_per_class // 4
            ctrl = rng.normal(size=(n_ctrl, n_features))
            ctrl[:, 0] += mean_shift
            ctrl[:, 1] *= scale_y
            pert = rng.normal(size=(n_pert, n_features))
            pert[:, 0] += mean_shift + 2.0  # perturbation effect: shift along feature 0
            pert[:, 1] *= scale_y
            feats.append(ctrl)
            feats.append(pert)
            rows.extend(
                [(cls, batch, "nontargeting")] * n_ctrl + [(cls, batch, "g1")] * n_pert
            )
    embeddings = np.vstack(feats).astype(np.float64)
    metadata = pd.DataFrame(rows, columns=["class", "batch_values", "pert"])
    return embeddings, metadata


def test_tvn_on_controls_joint_centers_controls_per_class():
    embeddings, metadata = _make_joint_synthetic()
    out = tvn_on_controls_joint(
        embeddings.copy(),
        metadata,
        pert_col="pert",
        control_key="nontargeting",
        batch_col="batch_values",
    )
    for cls in metadata["class"].unique():
        cls_mask = (metadata["class"] == cls).to_numpy()
        ctrl_mask = cls_mask & (metadata["pert"] == "nontargeting").to_numpy()
        ctrl = out[ctrl_mask]
        assert np.abs(ctrl.mean(axis=0)).max() < 0.5, (
            f"class {cls} control mean not centered: {ctrl.mean(axis=0)}"
        )


def test_tvn_on_controls_joint_pooled_controls_centered():
    embeddings, metadata = _make_joint_synthetic()
    out = tvn_on_controls_joint(
        embeddings.copy(),
        metadata,
        pert_col="pert",
        control_key="nontargeting",
        batch_col="batch_values",
    )
    ctrl_mask = (metadata["pert"] == "nontargeting").to_numpy()
    pooled = out[ctrl_mask]
    assert np.abs(pooled.mean(axis=0)).max() < 0.5


def test_tvn_on_controls_joint_shared_basis_aligns_classes():
    """After TVN, class-A and class-B controls should overlap (same mean, similar spread)."""
    embeddings, metadata = _make_joint_synthetic()
    out = tvn_on_controls_joint(
        embeddings.copy(),
        metadata,
        pert_col="pert",
        control_key="nontargeting",
        batch_col="batch_values",
    )
    ctrl_a = out[(metadata["class"] == "A") & (metadata["pert"] == "nontargeting")]
    ctrl_b = out[(metadata["class"] == "B") & (metadata["pert"] == "nontargeting")]
    mean_diff = np.linalg.norm(ctrl_a.mean(axis=0) - ctrl_b.mean(axis=0))
    assert mean_diff < 1.0, f"class control means still separated: {mean_diff}"
    std_a = ctrl_a.std(axis=0)
    std_b = ctrl_b.std(axis=0)
    assert np.abs(std_a - std_b).max() < 1.0


def test_tvn_on_controls_joint_preserves_row_order():
    embeddings, metadata = _make_joint_synthetic()
    out = tvn_on_controls_joint(
        embeddings.copy(),
        metadata,
        pert_col="pert",
        control_key="nontargeting",
        batch_col="batch_values",
    )
    assert out.shape == embeddings.shape


def test_tvn_on_controls_joint_uses_control_col_when_provided():
    """When control_col is set, it identifies controls; pert_col is used only for labels."""
    embeddings, metadata = _make_joint_synthetic()
    metadata = metadata.rename(columns={"pert": "barcode"})
    metadata["gene"] = metadata["barcode"].replace({"nontargeting": "nontargeting"})
    metadata.loc[metadata["barcode"] == "g1", "gene"] = "GENE1"
    out = tvn_on_controls_joint(
        embeddings.copy(),
        metadata,
        pert_col="barcode",
        control_key="nontargeting",
        batch_col="batch_values",
        control_col="gene",
    )
    ctrl_mask = (metadata["gene"] == "nontargeting").to_numpy()
    pooled = out[ctrl_mask]
    assert np.abs(pooled.mean(axis=0)).max() < 0.5
