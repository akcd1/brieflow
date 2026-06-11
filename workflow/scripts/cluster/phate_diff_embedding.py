"""Save a higher-dimensional PHATE embedding of the cleaned aggregated PCs, in the
same diffusion space the clustering uses, for point-to-point distance measurement.

Uses the identical PHATE settings as phate_leiden_clustering (same knn, distance
metric, random seed, and default landmark approximation) via the shared `run_phate`
helper; only the number of output coordinates differs (default 50 instead of 2). So
Euclidean distances in this embedding approximate distances in the space where the
clustering actually groups points.

Outputs:
  - output[0]  <...>__phate_int50.npy          (n, n_components) float64; row order
                                                 matches the row-ids file below.
  - output[1]  <...>__phate_int50_row_ids.txt   one "<class>:<cell_barcode_0>" per line.
"""

import re

import numpy as np
import pandas as pd

from lib.cluster.phate_leiden_clustering import run_phate

_PC_RE = re.compile(r"^PC_\d+$")

df = pd.read_csv(snakemake.input[0], sep="\t")

pc_cols = sorted((c for c in df.columns if _PC_RE.match(c)), key=lambda c: int(c[3:]))
if not pc_cols:
    raise ValueError(f"no PC_ columns found in {snakemake.input[0]}")

# Row identifier matching the clustering's points: "<class>:<cell_barcode_0>".
row_ids = (df["class"].astype(str) + ":" + df["cell_barcode_0"].astype(str)).tolist()

X = df[pc_cols].copy()
X.index = row_ids

n_components = int(snakemake.params.n_components)
metric = snakemake.params.phate_distance_metric

# run_phate matches the clustering's PHATE (random_state=42, knn=10, knn_dist=metric,
# default landmarking); n_components is forwarded to the PHATE constructor via kwargs.
emb_df, _ = run_phate(X, metric=metric, n_components=n_components)

np.save(snakemake.output[0], emb_df.values.astype("float64"))
with open(snakemake.output[1], "w") as fh:
    fh.write("\n".join(row_ids) + "\n")

print(
    f"[phate_diff_embedding] saved {emb_df.shape[0]} x {emb_df.shape[1]} embedding "
    f"(metric={metric}, n_components={n_components}) -> {snakemake.output[0]}"
)
