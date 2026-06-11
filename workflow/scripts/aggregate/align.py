import math
import gc
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
import numpy as np

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    centerscale_by_batch,
    tvn_on_controls,
    tvn_on_controls_joint,
    stratified_subsample,
)
from lib.aggregate.filter import harmonize_pool_schema, degeneracy_keep_list
from lib.aggregate.perturbation_score import perturbation_score

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_selection"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.feature_selection"
)
np.random.seed(0)

# PCA subsample constants (hardcoded per design)
PCA_SUBSET = 100_000
JOINT_PCA_SUBSAMPLE = 100_000
JOINT_PCA_MIN_PER_CLASS = 5_000

is_joint = snakemake.wildcards.cell_class == "joint"

# Filter out empty parquet files to avoid schema conflicts
non_empty_paths = [
    p for p in snakemake.input.filtered_paths if pq.read_metadata(p).num_rows > 0
]

if len(non_empty_paths) == 0:
    print("WARNING: No cells in input, writing empty output")
    pq.write_table(pa.table({}), snakemake.output[0])
    exit(0)

cell_dataset = ds.dataset(non_empty_paths, format="parquet")
total_rows = cell_dataset.count_rows()
print(
    f"Number of rows across {len(non_empty_paths)} non-empty parquet files: {total_rows}"
)

use_classifier = snakemake.params.get("use_classifier", False)
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, use_classifier)

# Pool-schema handling:
# - Joint: full harmonize (schema intersection + pool-level drop_cols_threshold),
#   because joint pools across classes and is more sensitive to pool-level NaN.
# - Non-joint: intersection only (drop_cols_threshold=None). This is the minimum
#   needed to prevent cross-well NaN leaking into the PCA sample when per-well
#   missing_values_filter drops different columns in different wells. Pre-joint
#   align would crash on such data; this adds no extra cleanup beyond what's
#   strictly required to keep align from crashing.
if is_joint:
    kept_metadata_cols, kept_feature_cols, _pool_report = harmonize_pool_schema(
        non_empty_paths,
        metadata_cols,
        drop_cols_threshold=snakemake.params.get("drop_cols_threshold"),
    )
else:
    kept_metadata_cols, kept_feature_cols, _pool_report = harmonize_pool_schema(
        non_empty_paths,
        metadata_cols,
        drop_cols_threshold=None,
    )
scan_cols = kept_metadata_cols + kept_feature_cols

# ---- Step 1: Build PCA ----
if is_joint:
    # Load class labels only (cheap), then stratified subsample, then read those rows.
    class_arr = cell_dataset.to_table(columns=["class"]).to_pandas()["class"].to_numpy()
    rng = np.random.default_rng(0)
    sample_indices = stratified_subsample(
        class_arr, JOINT_PCA_SUBSAMPLE, JOINT_PCA_MIN_PER_CLASS, rng
    )
    print(
        f"JOINT PCA: stratified subsample of {len(sample_indices)} rows across "
        f"classes {dict(pd.Series(class_arr[sample_indices]).value_counts())}"
    )
    sample_df = (
        cell_dataset.scanner(columns=scan_cols)
        .take(pa.array(sample_indices))
        .to_pandas(use_threads=True, memory_pool=None)
    )
else:
    n_sample = min(PCA_SUBSET, total_rows)
    random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
    random_indices.sort()
    sample_df = (
        cell_dataset.scanner(columns=scan_cols)
        .take(random_indices)
        .to_pandas(use_threads=True, memory_pool=None)
    )

# Residual NaN handling (joint only). Non-joint uses per-batch `.dropna(axis=1)`
# below, matching pre-joint behavior.
if is_joint:
    _sample_pre = len(sample_df)
    sample_df = sample_df.dropna(subset=kept_feature_cols).reset_index(drop=True)
    if len(sample_df) < _sample_pre:
        print(
            f"[pool] dropped {_sample_pre - len(sample_df)} PCA-sample rows with residual NaN"
        )

metadata, features = split_cell_data(sample_df, metadata_cols)

# ---- Degeneracy feature filter (global keep-if-any, applied before PCA) ----
# Decided once on the class-stratified PCA sample, then propagated to the batch
# loop via kept_feature_cols/scan_cols so every cell keeps the identical set.
degeneracy_cfg = snakemake.params.get("degeneracy_filter") or {}
if degeneracy_cfg.get("enabled", False):
    classes_for_deg = (
        metadata["class"]
        if "class" in metadata.columns
        else pd.Series("__all__", index=metadata.index)
    )
    keep_cols, deg_report = degeneracy_keep_list(
        features,
        classes_for_deg,
        freq_cut=degeneracy_cfg.get("freq_cut", 0.05),
        unique_count_floor=degeneracy_cfg.get("unique_count_floor", 100),
        var_floor=degeneracy_cfg.get("var_floor", 1e-8),
        min_cells=degeneracy_cfg.get("min_cells", 20),
    )
    print(
        f"[degeneracy] dropping {deg_report['n_dropped']} of "
        f"{deg_report['n_features']} features "
        f"(per-class degenerate: {deg_report['per_class_degenerate']})"
    )
    keep_set = set(keep_cols)
    kept_feature_cols = [c for c in kept_feature_cols if c in keep_set]
    if not kept_feature_cols:
        raise ValueError(
            "[degeneracy] all feature columns were dropped; check "
            "freq_cut/unique_count_floor/var_floor thresholds or the input data "
            f"(n_features={deg_report['n_features']})"
        )
    features = features[kept_feature_cols]
    scan_cols = kept_metadata_cols + kept_feature_cols

metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)
pca = PCA(n_components=snakemake.params.variance_or_ncomp).fit(
    centerscale_by_batch(features, metadata, "batch_values")
)

# ---- Step 2: Batched alignment ----
num_align_batches = snakemake.params.num_align_batches
all_indices = np.random.permutation(total_rows)
chunk_size = math.ceil(total_rows / num_align_batches)
subset_indices = [
    all_indices[i * chunk_size : (i + 1) * chunk_size] for i in range(num_align_batches)
]


def _compute_perturbation_score_joint(
    subset_df: pd.DataFrame, metadata_cols_local: list
) -> pd.DataFrame:
    """Compute perturbation_score and perturbation_auc per class and stitch back."""
    parts = []
    subset_df = subset_df.copy()
    for cls, part in subset_df.groupby("class", sort=False):
        part = part.copy()
        part["perturbation_score"] = np.nan
        part["perturbation_auc"] = np.nan
        perturbation_score(
            part,
            metadata_cols_local,
            snakemake.params.perturbation_name_col,
            snakemake.params.control_key,
            perturbation_id_col=snakemake.params.perturbation_id_col,
            control_name_col=snakemake.params.get("control_name_col"),
            batch_cols=snakemake.params.batch_cols,
        )
        parts.append(part)
    out = pd.concat(parts, axis=0)
    # Preserve original row order
    return out.loc[subset_df.index]


writer = None
for i, indices in enumerate(subset_indices):
    print(f"Processing subset {i + 1}/{num_align_batches} with {len(indices)} cells")

    if is_joint:
        subset_df = (
            cell_dataset.scanner(columns=scan_cols)
            .take(pa.array(indices))
            .to_pandas(use_threads=True, memory_pool=None)
        )
        # Drop residual per-row NaN (pool-level col drops above have already
        # eliminated systematically-missing columns).
        _pre = len(subset_df)
        subset_df = subset_df.dropna(subset=kept_feature_cols).reset_index(drop=True)
        if len(subset_df) < _pre:
            print(
                f"[pool] dropped {_pre - len(subset_df)} rows with residual NaN "
                f"from batch {i + 1}"
            )
    else:
        # Non-joint: scan the intersected schema. `.dropna(axis=1)` matches the
        # pre-joint batch loop — it's a no-op when intersection already removed
        # cross-well divergent cols, but preserves the pre-joint safety net
        # against any residual NaN.
        subset_df = (
            cell_dataset.scanner(columns=scan_cols)
            .take(pa.array(indices))
            .to_pandas(use_threads=True, memory_pool=None)
            .dropna(axis=1)
        )

    subset_df["perturbation_score"] = np.nan
    subset_df["perturbation_auc"] = np.nan
    chunk_metadata_cols = metadata_cols + ["perturbation_score", "perturbation_auc"]

    if not snakemake.params.skip_perturbation_score:
        if is_joint:
            subset_df = _compute_perturbation_score_joint(
                subset_df, chunk_metadata_cols
            )
        else:
            perturbation_score(
                subset_df,
                chunk_metadata_cols,
                snakemake.params.perturbation_name_col,
                snakemake.params.control_key,
                perturbation_id_col=snakemake.params.perturbation_id_col,
                control_name_col=snakemake.params.get("control_name_col"),
                batch_cols=snakemake.params.batch_cols,
            )

    for col in subset_df.columns:
        if is_numeric_dtype(subset_df[col]):
            subset_df[col] = subset_df[col].astype("float32")

    metadata, features = split_cell_data(subset_df, chunk_metadata_cols)
    del subset_df
    gc.collect()

    metadata, features = prepare_alignment_data(
        metadata,
        features,
        snakemake.params.batch_cols,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
        snakemake.params.perturbation_id_col,
    )

    features = centerscale_by_batch(features, metadata, "batch_values")
    features = pca.transform(features)

    if is_joint:
        features = tvn_on_controls_joint(
            features,
            metadata,
            snakemake.params.perturbation_name_col,
            snakemake.params.control_key,
            "batch_values",
            control_col=snakemake.params.get("control_name_col"),
        )
    else:
        features = tvn_on_controls(
            features,
            metadata,
            snakemake.params.perturbation_name_col,
            snakemake.params.control_key,
            "batch_values",
            control_col=snakemake.params.get("control_name_col"),
        )

    feature_columns = [f"PC_{j}" for j in range(features.shape[1])]
    features = pd.DataFrame(features, index=metadata.index, columns=feature_columns)
    aligned_cell_data = pd.concat([metadata, features], axis=1)
    del features
    gc.collect()

    aligned_cell_data = pa.Table.from_pandas(aligned_cell_data, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(snakemake.output[0], aligned_cell_data.schema)
    writer.write_table(aligned_cell_data)

if writer is not None:
    writer.close()
