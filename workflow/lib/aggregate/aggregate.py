"""This module provides functionality for aggregating embeddings based on metadata.

It includes a function to apply mean or median aggregation to replicate embeddings
for each perturbation, along with returning metadata containing perturbation labels
and cell counts.
"""

import numpy as np
import pandas as pd


def aggregate(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    method="mean",
    ps_probability_threshold=None,
    ps_percentile_threshold=None,
    group_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply mean or median aggregation to replicate embeddings and perturbation scores.

    Rows with perturbation_score below the threshold are dropped (NaNs kept). The function
    returns aggregated embeddings and metadata with grouping labels, cell counts, and
    aggregated perturbation scores.

    Args:
        embeddings: The embeddings to be aggregated.
        metadata: The metadata containing information about the embeddings.
        pert_col: The column in the metadata containing perturbation information.
            When `group_cols` is None, grouping is by this column alone.
        method: Aggregation method, "mean" or "median".
        ps_probability_threshold: Threshold for filtering based on perturbation score.
        ps_percentile_threshold: Percentile threshold for filtering based on perturbation score.
        group_cols: Optional list of columns to group by (e.g. ["class", pert_col] for
            joint class alignment). Defaults to [pert_col].

    Returns:
        tuple:
            - np.ndarray: Aggregated embeddings.
            - pd.DataFrame: Metadata with grouping labels, cell counts, and aggregated
              perturbation scores.
    """
    aggregated_embeddings = []
    aggregated_metadata = []

    metadata = metadata.reset_index(drop=True)
    aggr_func = (
        np.mean if method == "mean" else np.median if method == "median" else None
    )
    if aggr_func is None:
        raise ValueError(f"Invalid aggregation method: {method}")

    if group_cols is None:
        group_cols = [pert_col]

    if ps_probability_threshold is not None:
        mask = metadata["perturbation_score"].isna() | (
            metadata["perturbation_score"] >= ps_probability_threshold
        )
        metadata = metadata.loc[mask].reset_index(drop=True)
        embeddings = embeddings[mask.to_numpy(), :]

    if ps_percentile_threshold is not None:
        threshold_value = np.nanpercentile(
            metadata["perturbation_score"], ps_percentile_threshold * 100
        )
        mask = metadata["perturbation_score"].isna() | (
            metadata["perturbation_score"] >= threshold_value
        )
        metadata = metadata.loc[mask].reset_index(drop=True)
        embeddings = embeddings[mask.to_numpy(), :]

    grouping = metadata.groupby(group_cols)
    for group_key, group in grouping:
        final_emb = aggr_func(embeddings[group.index.values, :], axis=0)
        aggregated_embeddings.append(final_emb)

        # group_key is a scalar when groupby receives a string, a tuple when it
        # receives a list (even a list of length 1).
        if isinstance(group_key, tuple):
            key_values = list(group_key)
        else:
            key_values = [group_key]

        agg_meta = dict(zip(group_cols, key_values))
        agg_meta["cell_count"] = len(group)

        # Always include perturbation_auc if present (needed for gene-level filtering in clustering)
        if "perturbation_auc" in metadata.columns:
            agg_meta["perturbation_auc"] = group["perturbation_auc"].iloc[0]

        if ps_probability_threshold is not None or ps_percentile_threshold is not None:
            # aggregate perturbation score with same function
            pert_score = (
                aggr_func(group["perturbation_score"].dropna())
                if not group["perturbation_score"].isna().all()
                else np.nan
            )
            agg_meta["aggregated_perturbation_score"] = pert_score

        aggregated_metadata.append(agg_meta)

    return np.vstack(aggregated_embeddings), pd.DataFrame(aggregated_metadata)
