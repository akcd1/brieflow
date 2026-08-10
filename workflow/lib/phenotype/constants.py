"""Constants relevant to phenotype data processing."""

DEFAULT_METADATA_COLS = [
    "plate",
    "well",
    "tile",
    "cell_0",
    "i_0",
    "j_0",
    "site",
    "cell_1",
    "i_1",
    "j_1",
    "distance",
    "fov_distance_0",
    "fov_distance_1",
    "cell_barcode_0",
    "gene_symbol_0",
    "gene_id_0",
    "cell_barcode_1",
    "gene_symbol_1",
    "gene_id_1",
    "no_recomb_0",
    "no_recomb_1",
    "Q_min_0",
    "Q_min_1",
    "Q_recomb_0",
    "Q_recomb_1",
    "cell_barcode_peak_0",
    "cell_barcode_peak_1",
    "cell_barcode_count_0",
    "cell_barcode_count_1",
    "mapped_single_gene",
    "channels_min",
    "nucleus_i",
    "nucleus_j",
    "nucleus_bounds_0",
    "nucleus_bounds_1",
    "nucleus_bounds_2",
    "nucleus_bounds_3",
    "cell_i",
    "cell_j",
    "cell_bounds_0",
    "cell_bounds_1",
    "cell_bounds_2",
    "cell_bounds_3",
    "cytoplasm_i",
    "cytoplasm_j",
    "cytoplasm_bounds_0",
    "cytoplasm_bounds_1",
    "cytoplasm_bounds_2",
    "cytoplasm_bounds_3",
    "row",
    "col",
]


def metadata_cols(object_name: str = "nucleus") -> list[str]:
    """Return DEFAULT_METADATA_COLS with the object columns renamed.

    The pipeline's segmented object is called a nucleus by default, but a
    screen may name it something else (a vacuole, for instance). Only the
    object's own columns are rewritten -- cell and cytoplasm columns describe
    different objects and are left alone.
    """
    if object_name == "nucleus":
        return list(DEFAULT_METADATA_COLS)

    return [
        f"{object_name}_{col[len('nucleus_'):]}" if col.startswith("nucleus_") else col
        for col in DEFAULT_METADATA_COLS
    ]
