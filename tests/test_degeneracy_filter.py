"""Unit tests for the degeneracy feature filter in the aggregate module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the workflow's `lib` package importable without installing the package
WORKFLOW_LIB_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_LIB_DIR))

from lib.aggregate.filter import degeneracy_keep_list


def test_constant_near_dead_low_unique_dropped_single_class():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "constant": np.ones(n),
            "near_dead": np.linspace(0.0, 1e-5, n),  # var ~8e-12 < 1e-8
            "low_unique": rng.integers(0, 3, n).astype(float),  # 3 unique < 100
            "normal": rng.normal(0.0, 1.0, n),  # many unique, var ~1
        }
    )
    classes = pd.Series(["A"] * n)
    kept, report = degeneracy_keep_list(df, classes, min_cells=20)
    assert set(report["dropped"]) == {"constant", "near_dead", "low_unique"}
    assert kept == ["normal"]


def test_keep_if_informative_in_any_class():
    n = 100
    rng = np.random.default_rng(1)
    classes = pd.Series(["A"] * n + ["B"] * n)
    df = pd.DataFrame(
        {
            # variable in A, constant in B -> kept (informative in A)
            "class_specific": np.concatenate([rng.normal(0, 1, n), np.full(n, 5.0)]),
            # constant in both -> dropped
            "dead_both": np.concatenate([np.ones(n), np.ones(n)]),
        }
    )
    kept, report = degeneracy_keep_list(df, classes, min_cells=20)
    assert "class_specific" in kept
    assert "dead_both" not in kept


def test_min_cells_excludes_small_class_from_rescue():
    n_big, n_small = 100, 5
    rng = np.random.default_rng(2)
    classes = pd.Series(["A"] * n_big + ["B"] * n_big + ["C"] * n_small)
    # constant in the big classes; only the tiny (sub-threshold) class C varies
    vals = np.concatenate([np.ones(n_big), np.ones(n_big), rng.normal(0, 1, n_small)])
    df = pd.DataFrame({"feat": vals})
    kept, report = degeneracy_keep_list(df, classes, min_cells=20)
    assert "feat" not in kept
    assert "C" not in report["qualifying_classes"]


def test_permissive_thresholds_keep_everything():
    n = 100
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"constant": np.ones(n), "normal": rng.normal(0, 1, n)})
    classes = pd.Series(["A"] * n)
    kept, report = degeneracy_keep_list(
        df, classes, freq_cut=0.0, unique_count_floor=0, var_floor=0.0, min_cells=20
    )
    assert kept == ["constant", "normal"]
    assert report["n_dropped"] == 0
