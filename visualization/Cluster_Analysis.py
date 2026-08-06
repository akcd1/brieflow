import streamlit as st
import uuid

st.set_page_config(
    page_title="Cluster Analysis - Brieflow Analysis",
    layout="wide",
)

import pandas as pd
import glob
import os
import json

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.config import load_config
from src.filesystem import FileSystem
from src.filtering import create_filter_radio, apply_filter
from src.config import BRIEFLOW_OUTPUT_PATH, STATIC_ASSET_URL_ROOT, STATIC_ASSET_PATH

# =====================
# CONSTANTS
CLUSTER_ROOT = os.path.join(BRIEFLOW_OUTPUT_PATH, "cluster")

# Common hover data columns
HOVER_COLUMNS = [
    "gene_symbol_0",
    "class",
    "cell_barcode_0",
    "cluster",
    "cell_count",
    "source",
]

# Indices for accessing customdata array
GENE_SYMBOL_INDEX = 0
CLASS_INDEX = 1
CELL_BARCODE_INDEX = 2
CLUSTER_INDEX = 3
CELL_COUNT_INDEX = 4
SOURCE_INDEX = 5

# =====================
# FUNCTIONS


def has_mozzarellm_analysis(channel_combo: str) -> bool:
    """Check if a channel combo has mozzarellm analysis for any cell_class/leiden_resolution."""
    channel_dir = os.path.join(CLUSTER_ROOT, channel_combo)
    if not os.path.exists(channel_dir):
        return False
    # Check all cell_class/leiden_resolution subdirectories for mozzarellm/clusters
    for cell_class in os.listdir(channel_dir):
        cell_class_dir = os.path.join(channel_dir, cell_class)
        if not os.path.isdir(cell_class_dir):
            continue
        for leiden_res in os.listdir(cell_class_dir):
            mozzarellm_clusters = os.path.join(
                cell_class_dir, leiden_res, "cluster_analysis", "clusters"
            )
            if os.path.exists(mozzarellm_clusters):
                return True
    return False


def has_mozzarellm_for_cell_class(channel_combo: str, cell_class: str) -> bool:
    """Check if mozzarellm exists for channel_combo + cell_class + any leiden_resolution."""
    cell_class_dir = os.path.join(CLUSTER_ROOT, channel_combo, cell_class)
    if not os.path.exists(cell_class_dir):
        return False
    for leiden_res in os.listdir(cell_class_dir):
        mozzarellm_clusters = os.path.join(
            cell_class_dir, leiden_res, "cluster_analysis", "clusters"
        )
        if os.path.exists(mozzarellm_clusters):
            return True
    return False


def has_mozzarellm_for_leiden(channel_combo: str, cell_class: str, leiden_res) -> bool:
    """Check if mozzarellm exists for the exact channel_combo + cell_class + leiden_resolution."""
    # Convert to int then string to handle float values like 15.0 -> "15"
    leiden_str = str(int(float(leiden_res)))
    mozzarellm_clusters = os.path.join(
        CLUSTER_ROOT,
        channel_combo,
        cell_class,
        leiden_str,
        "cluster_analysis",
        "clusters",
    )
    return os.path.exists(mozzarellm_clusters)


# -- Data Load Methods --
# Load and merge cluster TSV files
@st.cache_data
def load_cluster_data():
    # Find all relevant TSV files
    tsv_files = glob.glob(
        f"{CLUSTER_ROOT}/**/phate_leiden_clustering.tsv", recursive=True
    )

    # Read each file and add source attribute
    dfs = []
    for file_path in tsv_files:
        rel_path = os.path.relpath(file_path, CLUSTER_ROOT)
        dirname = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path, sep="\t")
        df["source_full_path"] = file_path
        df["source"] = base_name
        parts = dirname.split(os.sep)
        for i, part in enumerate(parts):
            df[f"dir_level_{i}"] = part

        # Two supported layouts:
        #   3-level: <channel_combo>/<cell_class>/<leiden_resolution>/...
        #   4-level: <channel_combo>/<segmentation>/<cell_class>/<leiden_resolution>/...
        # The 4-level layout is used by projects that segment into a primary
        # ("cell") and secondary ("second_obj") set before applying the cell
        # classifier. Detect by depth and rename accordingly.
        if len(parts) >= 4:
            df.rename(
                columns={
                    "dir_level_0": "channel_combo",
                    "dir_level_1": "segmentation",
                    "dir_level_2": "cell_class",
                    "dir_level_3": "leiden_resolution",
                },
                inplace=True,
            )
        else:
            df["segmentation"] = "(none)"
            df.rename(
                columns={
                    "dir_level_0": "channel_combo",
                    "dir_level_1": "cell_class",
                    "dir_level_2": "leiden_resolution",
                },
                inplace=True,
            )

        dfs.append(df)

    # Concatenate all dataframes
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data
def load_montage_data(root_dir, gene_name):
    # Find all montage files
    files = FileSystem.find_files(
        root_dir + "/" + gene_name, include_all=["montages"], extensions=["png"]
    )

    # Extract features from the file paths
    filtered_df = FileSystem.extract_features(root_dir, files)

    # Add additional columns based on the file path structure
    filtered_df["gene"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-3])
    filtered_df["guide"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-2])
    filtered_df["channel"] = filtered_df["file_path"].apply(
        lambda x: x.split("/")[-1].split("__")[0]
    )

    return filtered_df


# -- Cluster scatter methods --
# Extract item value from selected point
def get_item_value_from_point(selected_point, groupby_column):
    # Get value from customdata which contains the hover_data values
    if "customdata" in selected_point and len(selected_point["customdata"]) > 0:
        if groupby_column in HOVER_COLUMNS:
            col_index = HOVER_COLUMNS.index(groupby_column)
            if col_index < len(selected_point["customdata"]):
                return str(selected_point["customdata"][col_index])

    # Fallback to legendgroup as a last resort (for compatibility)
    if "legendgroup" in selected_point:
        return selected_point["legendgroup"]

    return None


# Helper function to create a scatter trace
def make_scatter_trace(x, y, marker, text, customdata, name, showlegend, color=None):
    hovertemplate = (
        "PHATE_0=%{x}<br>"
        "PHATE_1=%{y}<br>"
        f"gene_symbol_0=%{{customdata[{GENE_SYMBOL_INDEX}]}}<br>"
        f"class=%{{customdata[{CLASS_INDEX}]}}<br>"
        f"cell_barcode_0=%{{customdata[{CELL_BARCODE_INDEX}]}}<br>"
        f"cluster=%{{customdata[{CLUSTER_INDEX}]}}<br>"
        f"cell_count=%{{customdata[{CELL_COUNT_INDEX}]}}<br>"
        f"source=%{{customdata[{SOURCE_INDEX}]}}<br>"
        "<extra></extra>"
    )
    # Optionally override color in marker
    if color is not None:
        marker = dict(marker, color=color)
    return go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=marker,
        text=text,
        customdata=customdata,
        name=name,
        hovertemplate=hovertemplate,
        showlegend=False,
    )


# -- Display helpers --
def display_gene_montages(gene_montages_root, gene):
    gene_dir = os.path.join(gene_montages_root, gene)
    if not os.path.exists(gene_dir):
        st.warning(f"No montage directory found for gene {gene}")
    else:
        montage_data = load_montage_data(gene_montages_root, gene)
        if montage_data.empty:
            st.write(f"No montage data found for gene {gene}")
        else:
            # Add filters for guide and channel
            available_guides = sorted(montage_data["guide"].unique())

            # Initialize session state for selected guide if it doesn't exist
            if f"selected_guide_{gene}" not in st.session_state:
                st.session_state[f"selected_guide_{gene}"] = None

            # Define a callback for when the guide dropdown changes
            def on_guide_select():
                st.session_state[f"selected_guide_{gene}"] = st.session_state[
                    f"guide_dropdown_{gene}"
                ]

            # Determine the index of the selected guide in the dropdown
            selected_index = 0
            selected_guide = st.session_state.get(f"selected_guide_{gene}", None)

            if selected_guide in available_guides:
                selected_index = available_guides.index(selected_guide)
            elif available_guides:
                # If no guide is selected yet or the previously selected guide is not available, select the first one
                selected_guide = available_guides[0]
                st.session_state[f"selected_guide_{gene}"] = selected_guide

            # Create a dropdown to select a guide
            selected_guide = st.selectbox(
                "Select Guide",
                available_guides,
                index=selected_index,
                key=f"guide_dropdown_{gene}",  # Use a stable key based on the selected gene
                on_change=on_guide_select,
            )

            # Filter the data based on selections
            filtered_montage_data = montage_data[
                (montage_data["guide"] == selected_guide)
            ]

            if len(filtered_montage_data) > 0:
                # Display each image in the filtered data
                for _, row in filtered_montage_data.iterrows():
                    # Construct the full path including the montages directory
                    image_path = os.path.join(gene_montages_root, row["file_path"])
                    channel_name = row["channel"]
                    channel_name = channel_name.replace("CH-", "")

                    try:
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Channel: {channel_name}")
                        else:
                            st.error(f"Image file not found: {image_path}")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")

                # Add download button for overlay TIFF
                overlay_tiff_path = os.path.join(
                    gene_montages_root, gene, selected_guide, f"overlay_montage.tiff"
                )

                if os.path.exists(overlay_tiff_path):
                    if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
                        # Use nginx-served static files when configured
                        relative_path = overlay_tiff_path.replace(STATIC_ASSET_PATH, "")
                        static_url = f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                        st.markdown(f"[Download Overlay TIFF]({static_url})")
                    else:
                        # Fall back to direct download when running locally
                        with open(overlay_tiff_path, "rb") as f:
                            st.download_button(
                                label="Download Overlay TIFF",
                                data=f,
                                file_name=f"{gene}_{selected_guide}_{row['channel']}_overlay.tiff",
                                key=f"download_{gene}_{selected_guide}_{row['channel']}_{uuid.uuid4()}",
                            )
                else:
                    st.warning(f"No overlay tiff found: {overlay_tiff_path}")
            else:
                st.warning(f"No image found for {gene} - {selected_guide}")


def display_cluster(cluster_data, cell_class=None, channel_combo=None):
    r"""
    :param cluster_data: a dataframe from load_cluster_data
    :param cell_class: the selected cell class filter value
    :param channel_combo: the selected channel combo filter value
    :param container: an st.container or equivalent that UI elements will be added to
    """
    global st
    # Display the data
    if not cluster_data.empty:
        # Always treat grouping column as categorical for discrete color maps
        if st.session_state.groupby_column in cluster_data.columns:
            cluster_data[st.session_state.groupby_column] = cluster_data[
                st.session_state.groupby_column
            ].astype(str)

        # Build a color map using the group names and the color palette
        group_names = cluster_data[st.session_state.groupby_column].unique()

        # Create a color palette optimized for visibility on a black background
        def get_optimized_color_palette(num_colors):
            # Use a perceptually uniform colormap that works well on dark backgrounds
            # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
            colormap_name = "turbo"  # Good visibility on dark backgrounds

            # Get evenly spaced colors from the colormap
            cmap = plt.get_cmap(colormap_name)
            colors = [
                mcolors.rgb2hex(cmap(i / (num_colors - 1 if num_colors > 1 else 1)))
                for i in range(num_colors)
            ]

            return colors

        # Get enough colors for all groups
        optimized_palette = get_optimized_color_palette(len(group_names))
        color_map = {group: optimized_palette[i] for i, group in enumerate(group_names)}

        # Always compute selected_data and other_data
        selected_item = st.session_state.get("selected_item", None)
        groupby_column = st.session_state.groupby_column
        selected_data = cluster_data[
            cluster_data[groupby_column].astype(str) == str(selected_item)
        ]
        other_data = cluster_data[
            cluster_data[groupby_column].astype(str) != str(selected_item)
        ]

        # Use plotly.graph_objects for full control
        fig = go.Figure()

        selected_gene = st.session_state.get("selected_gene", None)

        # Render every group as TWO traces: non-matching base points first,
        # then a highlight trace for any rows whose gene_symbol_0 matches
        # selected_gene. The highlight trace runs for ALL groups, so the same
        # gene in different clusters / classes is always visible.
        for group in group_names:
            group_df = cluster_data[cluster_data[groupby_column] == group]

            # Determine the group's base style based on cluster selection.
            if selected_item is None:
                base_color = color_map[group]
                base_opacity = 1.0
                base_size = 8
                base_line = None
            elif group == selected_item:
                base_color = color_map[group]
                base_opacity = 1.0
                base_size = 10
                base_line = dict(width=2, color="black")
            else:
                base_color = "gray"
                base_opacity = 0.3
                base_size = 8
                base_line = None

            # Split into matching / non-matching by selected_gene.
            if selected_gene:
                match_mask = group_df["gene_symbol_0"] == selected_gene
                base_df = group_df[~match_mask]
                match_df = group_df[match_mask]
            else:
                base_df = group_df
                match_df = group_df.iloc[0:0]

            # Base trace (non-matching).
            if not base_df.empty:
                marker_kwargs = dict(
                    color=base_color, size=base_size, opacity=base_opacity
                )
                if base_line is not None:
                    marker_kwargs["line"] = base_line
                fig.add_trace(
                    make_scatter_trace(
                        x=base_df["PHATE_0"],
                        y=base_df["PHATE_1"],
                        marker=dict(**marker_kwargs),
                        text=base_df["gene_symbol_0"],
                        customdata=base_df[HOVER_COLUMNS],
                        name=str(group),
                        showlegend=False,
                    )
                )

            # Highlight trace (selected gene, this group).
            if not match_df.empty:
                # Always use the cluster's true color (not gray) so the user
                # can see which cluster each instance belongs to, even when
                # the cluster isn't the currently-selected one.
                fig.add_trace(
                    make_scatter_trace(
                        x=match_df["PHATE_0"],
                        y=match_df["PHATE_1"],
                        marker=dict(
                            color=color_map[group],
                            size=15,
                            opacity=1.0,
                            symbol="circle",
                            line=dict(width=3, color="white"),
                        ),
                        text=match_df["gene_symbol_0"],
                        customdata=match_df[HOVER_COLUMNS],
                        name=f"{selected_gene} (Selected)",
                        showlegend=False,
                    )
                )

        # Update layout
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            title="",
            width=1000,
            height=800,
        )

        # Apply saved zoom coordinates if they exist
        if (
            st.session_state.zoom_xrange is not None
            and st.session_state.zoom_yrange is not None
        ):
            fig.update_layout(
                xaxis=dict(range=st.session_state.zoom_xrange),
                yaxis=dict(range=st.session_state.zoom_yrange),
            )

        # Display the plot with click event handling
        event = st.plotly_chart(
            fig, use_container_width=True, key="cluster_plot", on_select="rerun"
        )

        # Handle click events
        if event.selection and event.selection.points:
            selected_point = event.selection.points[0]

            # Get the item value from the selected point
            item_value = get_item_value_from_point(
                selected_point, st.session_state.groupby_column
            )

            # Get the gene value from the selected point
            gene_value = None
            if "customdata" in selected_point and len(selected_point["customdata"]) > 0:
                if GENE_SYMBOL_INDEX < len(selected_point["customdata"]):
                    gene_value = str(selected_point["customdata"][GENE_SYMBOL_INDEX])

            # Update session state if the item has changed
            if item_value and (
                st.session_state.selected_item != item_value
                or st.session_state.selected_gene != gene_value
            ):
                st.session_state.selected_item = item_value
                st.session_state.selected_gene = gene_value

                # Store current zoom state before rerunning
                if hasattr(event, "relayoutData") and event.relayoutData:
                    if (
                        "xaxis.range[0]" in event.relayoutData
                        and "xaxis.range[1]" in event.relayoutData
                    ):
                        st.session_state.zoom_xrange = [
                            event.relayoutData["xaxis.range[0]"],
                            event.relayoutData["xaxis.range[1]"],
                        ]
                    if (
                        "yaxis.range[0]" in event.relayoutData
                        and "yaxis.range[1]" in event.relayoutData
                    ):
                        st.session_state.zoom_yrange = [
                            event.relayoutData["yaxis.range[0]"],
                            event.relayoutData["yaxis.range[1]"],
                        ]

                st.rerun()

        # Save zoom coordinates from the event if available
        if hasattr(event, "relayoutData") and event.relayoutData:
            if (
                "xaxis.range[0]" in event.relayoutData
                and "xaxis.range[1]" in event.relayoutData
            ):
                st.session_state.zoom_xrange = [
                    event.relayoutData["xaxis.range[0]"],
                    event.relayoutData["xaxis.range[1]"],
                ]
            if (
                "yaxis.range[0]" in event.relayoutData
                and "yaxis.range[1]" in event.relayoutData
            ):
                st.session_state.zoom_yrange = [
                    event.relayoutData["yaxis.range[0]"],
                    event.relayoutData["yaxis.range[1]"],
                ]

    else:
        st.write("No cluster data files found.")


def cluster_table(cluster_data):
    # Display data overview
    st.markdown("## Cluster Data Overview")
    # If an item is selected, filter the dataframe
    source_tsv = cluster_data["source_full_path"].unique()[0]
    if os.path.exists(source_tsv):
        table_data = pd.read_csv(source_tsv, sep="\t")
        if st.session_state.selected_item:
            # Convert selected_item to integer since cluster column is int64
            try:
                selected_item_int = int(st.session_state.selected_item)
                table_data = table_data[table_data["cluster"] == selected_item_int]
            except ValueError:
                st.error(f"Invalid cluster value: {st.session_state.selected_item}")

            if len(table_data.index) == 0:
                st.warning(f"⚠️ WARNING: No data found in the TSV file: {source_tsv}")
            else:
                table_data.set_index("gene_symbol_0", inplace=True)
                st.dataframe(table_data)
        else:
            if len(table_data.index) == 0:
                st.warning(f"⚠️ WARNING: No data found in the TSV file: {source_tsv}")
            else:
                table_data.set_index("gene_symbol_0", inplace=True)
                st.dataframe(table_data)
    else:
        st.warning(f"⚠️ WARNING: Source TSV file not found at: {source_tsv}")


def feature_table(cell_class, channel_combo):
    # Feature Data Overview
    st.markdown("## Feature Data Overview")
    st.markdown(
        "Median feature values per gene after center scaling all single cell data on control cells by well."
    )
    # Construct the feature table path
    feature_table_path = os.path.join(
        BRIEFLOW_OUTPUT_PATH,
        "aggregate",
        "tsvs",
        f"CeCl-{cell_class}_ChCo-{channel_combo}__features_genes.tsv",
    )
    # Load and display the feature table if it exists
    if os.path.exists(feature_table_path):
        feature_df = pd.read_csv(feature_table_path, sep="\t")
        feature_df.set_index("gene_symbol_0", inplace=True)

        # Create a container with a fixed height and scrolling
        with st.container():
            # Display the dataframe with all columns and sorting enabled
            st.dataframe(
                feature_df,
                use_container_width=True,
                height=400,  # Fixed height for scrolling
                column_config={
                    # Configure all columns to be sortable
                    col: st.column_config.NumberColumn(width="medium")
                    for col in feature_df.columns
                },
            )
    else:
        st.warning(f"⚠️ WARNING: Feature table not found at: {feature_table_path}")


def cluster_size_charts(channel_combo, cell_class, leiden_resolution):
    # Create two equal-sized columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Cluster Sizes")
        # Construct the path to the cluster sizes plot
        cluster_sizes_path = os.path.join(
            BRIEFLOW_OUTPUT_PATH,
            "cluster",
            channel_combo,
            cell_class,
            leiden_resolution,
            "cluster_sizes.png",
        )

        # Display the plot if it exists
        if os.path.exists(cluster_sizes_path):
            st.image(cluster_sizes_path, use_container_width=True)
        else:
            st.warning(f"Cluster sizes plot not found at: {cluster_sizes_path}")

    with col2:
        st.markdown("### Cluster Enrichment")
        # Construct the path to the enrichment pie chart
        enrichment_pie_path = os.path.join(
            BRIEFLOW_OUTPUT_PATH,
            "cluster",
            channel_combo,
            cell_class,
            leiden_resolution,
            "CB-Real__pie_chart.png",
        )

        # Display the plot if it exists
        if os.path.exists(enrichment_pie_path):
            st.image(enrichment_pie_path, use_container_width=True)
        else:
            st.warning(
                f"Cluster enrichment pie chart not found at: {enrichment_pie_path}"
            )


def get_available_llm_combinations(channel_combo: str) -> list:
    """Find all cell_class/resolution combinations that have LLM data."""
    available = []
    channel_dir = os.path.join(CLUSTER_ROOT, channel_combo)
    if not os.path.exists(channel_dir):
        return available
    for cell_class in os.listdir(channel_dir):
        cell_class_dir = os.path.join(channel_dir, cell_class)
        if not os.path.isdir(cell_class_dir):
            continue
        for leiden_res in os.listdir(cell_class_dir):
            mozzarellm_clusters = os.path.join(
                cell_class_dir, leiden_res, "cluster_analysis", "clusters"
            )
            if os.path.exists(mozzarellm_clusters) and os.listdir(mozzarellm_clusters):
                available.append((cell_class, leiden_res))
    return available


_THEME_FIT_CHIP = {
    "core": ("#064e3b", "#6ee7b7"),  # green
    "peripheral": ("#78350f", "#fcd34d"),  # amber
    "off_theme": ("#374151", "#9ca3af"),  # gray
}
_THEME_FIT_DEFAULT = ("#1f2937", "#d1d5db")

_ALIGNMENT_BADGE = {
    "core_aspect": ("#064e3b", "#6ee7b7", "core aspect"),
    "sub_aspect": ("#1e3a8a", "#93c5fd", "sub aspect"),
    "alternative_theme": ("#7f1d1d", "#fca5a5", "alternative theme"),
}


def _v2_dominant_process(c):
    return c.get("task1", {}).get("dominant_process", "")


def _v2_pathway_confidence_str(c):
    pc = c.get("task1", {}).get("pathway_confidence", {}) or {}
    if not pc:
        return ""
    return f"{pc.get('tier', '')} ({pc.get('pct_fitting_theme', '')}%)"


def _v2_summary(c):
    return c.get("task1", {}).get("summary", "")


def _v2_genes_by_category(c, category):
    return [
        g for g in c.get("task1", {}).get("genes", []) if g.get("category") == category
    ]


def _is_nontargeting(gene_symbol):
    return isinstance(gene_symbol, str) and gene_symbol.startswith("nontargeting_")


_CLASS_DOT_COLOR = {
    "Bystander": "#06b6d4",  # cyan
    "Infected": "#f43f5e",  # rose
    "Uninfected": "#eab308",  # gold
}
_CLASS_DOT_DIM = "#374151"  # muted gray for absent

# Order in which the three dots are stacked (left -> right).
_CLASS_DOT_ORDER = ["Bystander", "Infected", "Uninfected"]


def _build_class_membership(c):
    """Return dict[gene -> set[class]] from class_analysis.per_class_genes."""
    per_class_genes = c.get("class_analysis", {}).get("per_class_genes", {}) or {}
    membership: dict[str, set] = {}
    for cls, genes in per_class_genes.items():
        for g in genes or []:
            membership.setdefault(g, set()).add(cls)
    return membership


def _class_dots_html(classes_present):
    """Three small colored dots: B / I / U. Filled when present, dim when absent."""
    parts = []
    for cls in _CLASS_DOT_ORDER:
        color = _CLASS_DOT_COLOR[cls] if cls in classes_present else _CLASS_DOT_DIM
        parts.append(
            f"<span style='display:inline-block; width:6px; height:6px; "
            f"border-radius:50%; background:{color}; margin-right:2px; "
            f"vertical-align:middle;'></span>"
        )
    return "".join(parts)


_CATEGORY_CHIP_COLORS = {
    "balanced": ("#374151", "#d1d5db"),  # gray
    "class_dominant_": ("#5b21b6", "#d8b4fe"),  # purple
    "class_depleted_": ("#78350f", "#fcd34d"),  # amber
    "size_below_min": ("#7f1d1d", "#fca5a5"),  # red
}


def _category_chip_colors(category):
    if category == "balanced" or category == "size_below_min":
        return _CATEGORY_CHIP_COLORS[category]
    if category.startswith("class_dominant_"):
        return _CATEGORY_CHIP_COLORS["class_dominant_"]
    if category.startswith("class_depleted_"):
        return _CATEGORY_CHIP_COLORS["class_depleted_"]
    return ("#374151", "#d1d5db")


def _gene_chip(gene_record, *, class_membership):
    """Render one gene chip. Background encodes theme_fit; class-presence dots
    sit before the gene name; the LLM's per-gene reasoning lives in a tooltip."""
    gene = gene_record.get("gene", "")
    fit = gene_record.get("theme_fit", "")
    bg, fg = _THEME_FIT_CHIP.get(fit, _THEME_FIT_DEFAULT)
    reasoning = (gene_record.get("reasoning", "") or "").replace('"', "'")
    classes_present = class_membership.get(gene, set())
    classes_text = ",".join(sorted(classes_present)) if classes_present else "none"
    if reasoning:
        title_attr = (
            f' title="{reasoning} | theme_fit: {fit} | classes: {classes_text}"'
        )
    else:
        title_attr = f' title="theme_fit: {fit} | classes: {classes_text}"'
    dots = _class_dots_html(classes_present)
    return (
        f"<span style='background:{bg}; color:{fg}; border-radius:4px;"
        f" padding:2px 8px; margin-right:4px; display:inline-block;"
        f" margin-bottom:4px;'{title_attr}>{dots}"
        f"<span style='margin-left:4px;'>{gene}</span></span>"
    )


def _render_overview_tab(c):
    # Control-dominated banner
    if c.get("is_control_dominated"):
        pct = int(round((c.get("control_gene_fraction", 0.0) or 0.0) * 100))
        st.markdown(
            f"""
            <div style='background-color:#451a03; border-left:4px solid #f59e0b;
                        border-radius:6px; padding:12px 16px; margin-bottom:12px;'>
                <span style='color:#fcd34d; font-weight:600;'>⚠ Control-dominated cluster</span>
                <span style='color:#fde68a;'>— {pct}% nontargeting controls.
                Theme is fixed by the deterministic rule.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main card: theme + summary + chips
    established = _v2_genes_by_category(c, "established")
    uncharacterized = _v2_genes_by_category(c, "uncharacterized")
    class_membership = _build_class_membership(c)

    # Optionally hide nontargeting chips
    show_controls = False
    if c.get("is_control_dominated"):
        show_controls = st.checkbox(
            "Show nontargeting control chips",
            value=False,
            key=f"show_controls_{st.session_state.get('selected_item', '')}",
        )
    if not show_controls:
        established = [
            g for g in established if not _is_nontargeting(g.get("gene", ""))
        ]
        uncharacterized = [
            g for g in uncharacterized if not _is_nontargeting(g.get("gene", ""))
        ]

    est_chips = (
        "".join(_gene_chip(g, class_membership=class_membership) for g in established)
        or "<span style='color:#6b7280;'>(none)</span>"
    )
    unc_chips = (
        "".join(
            _gene_chip(g, class_membership=class_membership) for g in uncharacterized
        )
        or "<span style='color:#6b7280;'>(none)</span>"
    )

    # Header chips: confidence + cluster_category + (optional) qc caveat
    cat = c.get("cluster_category", "") or ""
    cat_bg, cat_fg = _category_chip_colors(cat)
    header_chips = (
        f"<span style='background:#1e3a8a; color:#93c5fd; border-radius:6px;"
        f" padding:4px 12px; font-weight:600; margin-right:6px;'>"
        f"Confidence: {_v2_pathway_confidence_str(c)}</span>"
        f"<span style='background:{cat_bg}; color:{cat_fg}; border-radius:6px;"
        f" padding:4px 12px; font-weight:600; margin-right:6px;'"
        f" title='Bin: {cat}'>{cat or '(no bin)'}</span>"
    )
    if c.get("qc_caveat_flag"):
        qc_reason = (c.get("qc_caveat_reason", "") or "").replace('"', "'")
        header_chips += (
            f"<span style='background:#7c2d12; color:#fdba74; border-radius:6px;"
            f" padding:4px 12px; font-weight:600;'"
            f' title="{qc_reason}">⚠ QC caveat</span>'
        )

    st.markdown(
        f"""
        <div style='background-color:#1e1e1e; border-radius:10px; padding:20px;
                    margin-bottom:20px; box-shadow:0 2px 8px #00000040;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
                <div>
                    <span style='font-size:1.3em; font-weight:bold; color:#e0e0e0;'>Dominant Process:</span>
                    <span style='font-size:1.3em; color:#60a5fa; font-weight:bold;'> {_v2_dominant_process(c)}</span>
                </div>
                <div>{header_chips}</div>
            </div>
            <div style='margin-top:10px; margin-bottom:10px; font-size:1.05em; color:#d1d5db;'>
                {_v2_summary(c)}
            </div>
            <div style='margin-top:18px;'>
                <span style='font-weight:600; color:#60a5fa;'>Established ({len(established)}):</span>
                <span style='margin-left:8px;'>{est_chips}</span>
            </div>
            <div style='margin-top:10px;'>
                <span style='font-weight:600; color:#c084fc;'>Uncharacterized ({len(uncharacterized)}):</span>
                <span style='margin-left:8px;'>{unc_chips}</span>
            </div>
            <div style='margin-top:14px; font-size:0.85em; color:#6b7280; line-height:1.6;'>
                Hover a gene chip for the LLM's per-gene reasoning.<br>
                <b>Background</b> = theme_fit:
                <span style='display:inline-block; width:14px; height:10px; border-radius:2px; background:#064e3b; vertical-align:middle;'></span> core,
                <span style='display:inline-block; width:14px; height:10px; border-radius:2px; background:#78350f; vertical-align:middle;'></span> peripheral,
                <span style='display:inline-block; width:14px; height:10px; border-radius:2px; background:#374151; vertical-align:middle;'></span> off-theme.&nbsp;&nbsp;
                <b>Dots (left to right)</b> = class presence:
                <span style='display:inline-block; width:8px; height:8px; border-radius:50%; background:#06b6d4; vertical-align:middle;'></span> Bystander,
                <span style='display:inline-block; width:8px; height:8px; border-radius:50%; background:#f43f5e; vertical-align:middle;'></span> Infected,
                <span style='display:inline-block; width:8px; height:8px; border-radius:50%; background:#eab308; vertical-align:middle;'></span> Uninfected
                (dim = absent).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_subthemes_tab(c):
    sub_themes = c.get("sub_themes", []) or []
    triggers = c.get("sub_theme_triggers", {}) or {}

    # Footnote summarising trigger status per class.
    trigger_lines = []
    for cls, info in triggers.items():
        triggered = info.get("triggered", False)
        reason = info.get("reason", "")
        marker = "✓" if triggered else "·"
        color = "#6ee7b7" if triggered else "#6b7280"
        trigger_lines.append(
            f"<span style='color:{color}; margin-right:14px;'>{marker} <b>{cls}</b>: {reason}</span>"
        )
    if trigger_lines:
        st.markdown(
            f"<div style='font-size:0.9em; margin-bottom:14px; color:#9ca3af;'>"
            f"Trigger status: {' '.join(trigger_lines)}</div>",
            unsafe_allow_html=True,
        )

    if not sub_themes:
        st.info(
            "No sub-themes triggered. See the 'Trigger status' line above for the per-class reason."
        )
        return

    for s in sub_themes:
        cls = s.get("class", s.get("class_label", "?"))
        alignment = s.get("alignment", "")
        bg, fg, label_text = _ALIGNMENT_BADGE.get(
            alignment, ("#374151", "#d1d5db", alignment or "?")
        )
        confidence = s.get("confidence", "")
        size = s.get("subset_size", 0)
        genes = s.get("subset_genes", []) or []
        if len(genes) > 20:
            gene_text = ", ".join(genes[:20]) + f", …+{len(genes) - 20} more"
        else:
            gene_text = ", ".join(genes)
        description = s.get("alignment_description", "")

        st.markdown(
            f"""
            <div style='background-color:#1e1e1e; border-radius:10px; padding:16px 20px;
                        margin-bottom:14px; box-shadow:0 2px 8px #00000040;'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
                    <div>
                        <span style='font-size:1.15em; font-weight:bold; color:#e0e0e0;'>{cls}</span>
                        <span style='color:#9ca3af; margin-left:8px;'>(n={size})</span>
                    </div>
                    <div>
                        <span style='background:{bg}; color:{fg}; border-radius:6px;
                                     padding:3px 10px; font-weight:600; margin-right:6px;'>{label_text}</span>
                        <span style='background:#1e3a8a; color:#93c5fd; border-radius:6px;
                                     padding:3px 10px; font-weight:600;'>conf: {confidence}</span>
                    </div>
                </div>
                <div style='margin-top:10px; color:#d1d5db; font-size:1em;'>{description}</div>
                <div style='margin-top:8px; color:#6b7280; font-size:0.85em;'>Genes: {gene_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_details_tab(c):
    ca = c.get("class_analysis", {}) or {}

    # 1. Bin & composition
    st.markdown("**Bin & composition**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(
            {
                "cluster_category": c.get("cluster_category"),
                "additional_depleted_classes": c.get("additional_depleted_classes", []),
                "size": c.get("size"),
                "n_unique_genes": c.get("n_unique_genes"),
                "n_real_genes": c.get("n_real_genes"),
                "n_control_genes": c.get("n_control_genes"),
                "control_gene_fraction": c.get("control_gene_fraction"),
                "is_control_dominated": c.get("is_control_dominated"),
            }
        )
    with col2:
        st.write(
            {
                "class_fractions": c.get("class_fractions", {}),
                "class_vs_expected": c.get("class_vs_expected", {}),
            }
        )

    # 2. Set ops
    st.markdown("**Set operations**")
    st.write(
        {
            "pairwise_jaccard": ca.get("pairwise_jaccard", {}),
            "three_way_jaccard": ca.get("three_way_jaccard"),
            "per_class_size": ca.get("per_class_size", {}),
            "per_class_pct_unique": ca.get("per_class_pct_unique", {}),
        }
    )

    # 3. Triggers
    st.markdown("**Sub-theme triggers**")
    triggers = c.get("sub_theme_triggers", {}) or {}
    st.write(
        {
            cls: f"triggered={info.get('triggered')} — {info.get('reason')}"
            for cls, info in triggers.items()
        }
    )

    # 4. QC dropout
    st.markdown("**QC dropout**")
    st.write(
        {
            "qc_dropout_per_class": c.get("qc_dropout_per_class", {}),
            "qc_caveat_flag": c.get("qc_caveat_flag"),
            "qc_caveat_reason": c.get("qc_caveat_reason"),
        }
    )

    # 5. Per-gene reasoning table
    st.markdown("**Per-gene reasoning**")
    genes = c.get("task1", {}).get("genes", []) or []
    if genes:
        df_genes = pd.DataFrame(
            [
                {
                    "gene": g.get("gene", ""),
                    "category": g.get("category", ""),
                    "theme_fit": g.get("theme_fit", ""),
                    "reasoning": g.get("reasoning", ""),
                }
                for g in genes
            ]
        )
        st.dataframe(df_genes, use_container_width=True, hide_index=True)
    else:
        st.write("(no per-gene records)")

    # 6. Anonymization mapping (debug)
    with st.expander("Anonymization mapping (debug)", expanded=False):
        st.write(c.get("anonymization_mapping", {}))


def display_cluster_json(cluster_data, container=st.container()):
    if (
        "selected_item" in st.session_state
        and st.session_state.selected_item is not None
    ):
        # Because the interphase folder has mixed case
        cluster_dir = os.path.dirname(cluster_data["source_full_path"].unique()[0])
        cluster_id = str(st.session_state.selected_item)

        # v2: cluster_analysis/clusters/{id}.json (no "cluster_" prefix on filename)
        mozzarellm_clusters_dir = os.path.join(
            cluster_dir, "cluster_analysis", "clusters"
        )
        cluster_json_path = os.path.join(mozzarellm_clusters_dir, f"{cluster_id}.json")

        # Always show the section header
        st.markdown("### LLM Cluster Analysis")

        if os.path.exists(cluster_json_path):
            with open(cluster_json_path, "r") as f:
                c = json.load(f)

            n_subthemes = len(c.get("sub_themes", []) or [])
            tab_labels = [
                "Overview",
                f"Sub-themes ({n_subthemes})" if n_subthemes else "Sub-themes",
                "Details",
            ]
            tab_overview, tab_subthemes, tab_details = st.tabs(tab_labels)
            with tab_overview:
                _render_overview_tab(c)
            with tab_subthemes:
                _render_subthemes_tab(c)
            with tab_details:
                _render_details_tab(c)
        else:
            # Show placeholder card when LLM data is not available
            current_cell_class = st.session_state.get("cell_class", "unknown")
            current_resolution = st.session_state.get("leiden_resolution", "unknown")
            channel_combo = st.session_state.get("channel_combo", "")

            # Find which combinations have LLM data
            available = get_available_llm_combinations(channel_combo)

            # Smart context: tailor message based on what's wrong
            if not available:
                available_text = "No LLM analysis available for this dataset."
            else:
                # Check if current cell class has any LLM data
                cell_classes_with_llm = set(cc for cc, res in available)
                resolutions_for_current_class = [
                    res for cc, res in available if cc == current_cell_class
                ]

                if current_cell_class in cell_classes_with_llm:
                    # Right cell class, wrong resolution
                    res_list = ", ".join(sorted(resolutions_for_current_class, key=int))
                    available_text = (
                        f"Available for {current_cell_class} at resolution: {res_list}"
                    )
                else:
                    # Wrong cell class
                    available_text = (
                        f"Available for: {', '.join(sorted(cell_classes_with_llm))}"
                    )

            st.markdown(
                f"""
                <div style='background-color:#1e1e1e; border-radius:10px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px #00000040; border: 1px solid #374151;'>
                    <div style='color:#9ca3af; font-size:1.1em;'>
                        <span style='font-size:1.2em;'>ℹ️</span>
                        LLM analysis is not available for <strong>{current_cell_class}</strong> cells
                        at resolution <strong>{current_resolution}</strong>.
                    </div>
                    <div style='margin-top:12px; color:#6b7280; font-size:0.95em;'>
                        {available_text}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def display_uniprot_info():
    if st.session_state.selected_gene:
        source_tsv = cluster_data["source_full_path"].unique()[0]
        if os.path.exists(source_tsv):
            table_data = pd.read_csv(source_tsv, sep="\t")
            table_data = table_data[
                table_data["gene_symbol_0"] == st.session_state.selected_gene
            ]
            if len(table_data.index) != 0:
                st.write(
                    f"Uniprot Entry: [{table_data['uniprot_entry'].values[0]}]({table_data['uniprot_link'].values[0]})"
                )
                function_text = table_data["uniprot_function"].values[0]
                if isinstance(function_text, str) and function_text.strip():
                    st.markdown(f"Uniprot Function:\n>{function_text}")
                else:
                    st.write("Uniprot Function: Not available")


# -- Search/Filter state management --


def initialize_session_state() -> None:
    """Initialize all session state variables used in the cluster analysis.

    This function sets up all the necessary session state variables with their default values.
    It should be called at the start of the script to ensure all required state variables
    are properly initialized.
    """
    # Initialize basic selection states
    if "selected_item" not in st.session_state:
        st.session_state.selected_item = None
    if "groupby_column" not in st.session_state:
        st.session_state.groupby_column = "cluster"
    if "selected_gene" not in st.session_state:
        st.session_state.selected_gene = None
    if "selected_guide" not in st.session_state:
        st.session_state.selected_guide = None

    # Initialize zoom coordinates
    if "zoom_xrange" not in st.session_state:
        st.session_state.zoom_xrange = None
    if "zoom_yrange" not in st.session_state:
        st.session_state.zoom_yrange = None

    # Initialize search state
    if "last_gene_search" not in st.session_state:
        st.session_state.last_gene_search = ""
    if "last_cluster_search" not in st.session_state:
        st.session_state.last_cluster_search = ""

    # Initialize gene selection dropdowns
    if "selected_gene_global" not in st.session_state:
        st.session_state.selected_gene_global = None
    if "selected_gene_cluster" not in st.session_state:
        st.session_state.selected_gene_cluster = None

    # Initialize cell class
    if "cell_class" not in st.session_state:
        st.session_state.cell_class = "all"

    # Initialize cluster dropdown
    if "cluster_dropdown" not in st.session_state:
        st.session_state.cluster_dropdown = None

    # Initialize filter counter for unique keys
    if "filter_counter" not in st.session_state:
        st.session_state.filter_counter = 0


def on_global_gene_select() -> None:
    """Callback function for global gene selection.

    Updates the selected gene and its associated cluster in the session state.
    When a gene is selected globally, it also updates the cluster selection to match
    the cluster containing the selected gene.
    """
    gene = st.session_state.selected_gene_global
    st.session_state.selected_gene = gene
    # Set cluster to the gene's cluster
    gene_row = cluster_data[cluster_data["gene_symbol_0"] == gene]
    if not gene_row.empty:
        cluster_num = str(gene_row["cluster"].iloc[0])
        st.session_state.selected_item = cluster_num
        st.session_state.cluster_dropdown = cluster_num
        st.session_state.selected_gene_cluster = gene
    else:
        st.session_state.selected_item = None
        st.session_state.cluster_dropdown = None
        st.session_state.selected_gene_cluster = None


def on_cluster_select() -> None:
    """Callback function for cluster selection.

    Updates the selected cluster and its associated gene in the session state.
    When a cluster is selected, it automatically selects the first gene in that cluster.
    If 'Select a cluster to view' is chosen, it clears all selections.
    """
    cluster = st.session_state.cluster_dropdown
    if cluster == "Select a cluster...":
        st.session_state.selected_item = None
        st.session_state.selected_gene = None
        st.session_state.selected_gene_global = None
        st.session_state.selected_gene_cluster = None
    else:
        st.session_state.selected_item = cluster
        # Find the first gene in this cluster
        cluster_genes = get_cluster_genes(cluster_data, cluster)
        if cluster_genes:
            first_gene = cluster_genes[0]
            st.session_state.selected_gene = first_gene
            st.session_state.selected_gene_global = first_gene
            st.session_state.selected_gene_cluster = first_gene
        else:
            st.session_state.selected_gene = None
            st.session_state.selected_gene_global = None
            st.session_state.selected_gene_cluster = None


def on_cluster_gene_select() -> None:
    """Callback function for gene selection within a cluster.

    Updates the selected gene in both global and cluster contexts.
    This ensures that gene selection is synchronized between the global
    and cluster-specific views.
    """
    gene = st.session_state.selected_gene_cluster
    st.session_state.selected_gene = gene
    st.session_state.selected_gene_global = gene


def on_channel_combo_change():
    """Callback function for channel combo selection."""
    st.session_state.channel_combo = st.session_state.channel_combo_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


def on_segmentation_change():
    """Callback function for segmentation selection."""
    st.session_state.segmentation = st.session_state.segmentation_radio_main
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


def on_cell_class_change():
    """Callback function for cell class selection."""
    st.session_state.cell_class = st.session_state.cell_class_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


def on_leiden_resolution_change():
    """Callback function for leiden resolution selection."""
    st.session_state.leiden_resolution = st.session_state.leiden_resolution_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


# Apply filters
def apply_all_filters(data):
    """Apply all filters to the cluster data in the correct order."""
    # Channel Combo filter - handle directly
    channel_combo_options = sorted(data["channel_combo"].unique().tolist())
    # Initialize channel combo in session state if needed
    if "channel_combo" not in st.session_state:
        st.session_state.channel_combo = (
            channel_combo_options[0] if channel_combo_options else None
        )

    # Format channel combo display label
    def format_channel_combo(combo: str) -> str:
        return combo

    # Create the radio button with a stable key
    selected_channel_combo = st.sidebar.radio(
        "**Channel Combo** - *Used to subset features during aggregation*",
        channel_combo_options,
        index=channel_combo_options.index(st.session_state.channel_combo)
        if st.session_state.channel_combo in channel_combo_options
        else 0,
        key="channel_combo_radio_main",
        on_change=on_channel_combo_change,
        format_func=format_channel_combo,
    )
    data = apply_filter(data, "channel_combo", selected_channel_combo)

    # Segmentation filter (4-level layouts only). For 3-level layouts the
    # synthetic value "(none)" is the sole option and the filter is skipped.
    if "segmentation" in data.columns:
        seg_options = sorted(
            str(x) for x in data["segmentation"].dropna().unique().tolist()
        )
        # Skip the UI when there's nothing to choose (3-level layout).
        if seg_options and seg_options != ["(none)"]:
            if "segmentation" not in st.session_state:
                st.session_state.segmentation = seg_options[0]
            if st.session_state.segmentation not in seg_options:
                st.session_state.segmentation = seg_options[0]
            selected_segmentation = st.sidebar.radio(
                "**Segmentation** - *Primary or secondary object segmentation*",
                seg_options,
                index=seg_options.index(st.session_state.segmentation)
                if st.session_state.segmentation in seg_options
                else 0,
                key="segmentation_radio_main",
                on_change=on_segmentation_change,
            )
            data = apply_filter(data, "segmentation", selected_segmentation)

    # Cell Class filter - derived from the loaded data so projects with
    # non-default cell_class taxonomies (e.g. "joint", "Bystander", "Infected",
    # "Uninfected") render correctly instead of crashing on .index().
    discovered_cell_classes = sorted(
        str(x) for x in data["cell_class"].dropna().unique().tolist()
    )
    cell_class_options = ["all"] + [c for c in discovered_cell_classes if c != "all"]
    # Initialize cell class in session state if needed
    if "cell_class" not in st.session_state:
        st.session_state.cell_class = "all"
    # If session state holds a value not present in the discovered options
    # (e.g. config default for a different channel_combo), fall back to "all".
    if st.session_state.cell_class not in cell_class_options:
        st.session_state.cell_class = "all"

    # Format cell class display label
    def format_cell_class(cc: str) -> str:
        return cc

    # Create the radio button with a stable key
    selected_cell_class = st.sidebar.radio(
        "**Cell Class** - *Used to subset single cell data with classifier provided during aggregation*",
        cell_class_options,
        index=cell_class_options.index(st.session_state.cell_class)
        if st.session_state.cell_class in cell_class_options
        else 0,
        key="cell_class_radio_main",
        on_change=on_cell_class_change,
        format_func=format_cell_class,
    )
    data = apply_filter(data, "cell_class", selected_cell_class)

    # Leiden Resolution filter - handle directly
    leiden_options = sorted(
        data["leiden_resolution"].unique().tolist(), key=lambda x: float(x)
    )
    # Initialize leiden resolution in session state if needed
    if "leiden_resolution" not in st.session_state:
        st.session_state.leiden_resolution = (
            leiden_options[0] if leiden_options else None
        )

    # Format leiden resolution display label
    def format_leiden(lr: str) -> str:
        return str(lr)

    # Create the radio button with a stable key
    selected_lr = st.sidebar.radio(
        """**Leiden Resolution** - *Used in the Leiden clustering algorithm to determine gene clusters*""",
        leiden_options,
        index=leiden_options.index(st.session_state.leiden_resolution)
        if st.session_state.leiden_resolution in leiden_options
        else 0,
        key="leiden_resolution_radio_main",
        on_change=on_leiden_resolution_change,
        format_func=format_leiden,
    )
    data = apply_filter(data, "leiden_resolution", selected_lr)

    return data


# Calculate cluster_genes after all filters are applied
def get_cluster_genes(data, cluster_id):
    """Get sorted list of genes for a given cluster."""
    if not cluster_id:
        return []
    try:
        cluster_val = int(cluster_id)
    except Exception:
        cluster_val = cluster_id
    return sorted(data[data["cluster"] == cluster_val]["gene_symbol_0"].unique())


# ===

# Call initialize_session_state at the start of the script
initialize_session_state()

# Apply config defaults on first load
if not st.session_state.get("config_defaults_applied", False):
    try:
        _config = load_config()
        # v2 cluster_analysis section preferred; fall back to legacy mozzarellm.
        _mozzarellm = _config.get("cluster_analysis") or _config.get("mozzarellm", {})
        if _mozzarellm:
            if "cell_class" in _mozzarellm:
                st.session_state.cell_class = _mozzarellm["cell_class"]
            if "channel_combo" in _mozzarellm:
                st.session_state.channel_combo = _mozzarellm["channel_combo"]
            if "leiden_resolution" in _mozzarellm:
                st.session_state.leiden_resolution = str(
                    int(_mozzarellm["leiden_resolution"])
                )
    except Exception:
        pass  # If config loading fails, fall back to existing defaults
    st.session_state.config_defaults_applied = True

# Load and filter cluster data
cluster_data = load_cluster_data()

# Sort clusters numerically instead of alphabetically
all_genes = sorted(cluster_data["gene_symbol_0"].unique())
all_clusters = sorted(
    [str(c) for c in cluster_data["cluster"].unique()], key=lambda x: int(x)
)

st.sidebar.title("Filters")
cluster_data = apply_all_filters(cluster_data)
cluster_genes = get_cluster_genes(cluster_data, st.session_state.selected_item)

# --- UI Layout ---
st.title("Cluster Analysis")
st.markdown(
    "*Click a cluster to see details, panning and zooming is easily done through the top right of the cluster panel*"
)

# Add filters section in sidebar FIRST
# Remove duplicate call to apply_all_filters since it's already called above
# cluster_data = apply_all_filters(cluster_data)  # This line is removed

# --- Widget Rendering ---
col1, col2 = st.columns(2)
with col1:
    # Global gene dropdown with placeholder
    gene_placeholder = "Select a gene..."
    gene_options = [gene_placeholder] + all_genes
    gene_val = (
        st.session_state.selected_gene
        if st.session_state.selected_gene in all_genes
        else gene_placeholder
    )
    st.session_state.selected_gene_global = gene_val
    selected_gene = st.selectbox(
        "Gene Search",
        options=gene_options,
        index=gene_options.index(gene_val) if gene_val in gene_options else 0,
        key="selected_gene_global",
        on_change=on_global_gene_select,
    )
    # Only update if a real gene is selected
    if selected_gene != gene_placeholder:
        st.session_state.selected_gene = selected_gene

with col2:
    # Cluster dropdown.
    # Streamlit warns if the same widget is given both an `index=` default
    # AND a key whose session_state slot is also written manually. We drive
    # the widget purely from session_state.cluster_dropdown — the assignment
    # below keeps it in sync with selected_item (e.g. when the user clicks a
    # point on the PHATE scatter), and the widget itself omits `index=`.
    cluster_options = ["Select a cluster..."] + all_clusters
    cluster_val = (
        st.session_state.selected_item
        if st.session_state.selected_item in all_clusters
        else "Select a cluster..."
    )
    st.session_state.cluster_dropdown = cluster_val
    st.selectbox(
        "Cluster Search",
        options=cluster_options,
        key="cluster_dropdown",
        on_change=on_cluster_select,
    )

cell_class = st.session_state.cell_class
channel_combo = st.session_state.channel_combo
leiden_resolution = st.session_state.leiden_resolution

if not st.session_state.selected_item:
    # No cluster selected: Just show the full width cluster plot
    display_cluster(
        cluster_data,
        cell_class=st.session_state.cell_class,
        channel_combo=st.session_state.channel_combo,
    )
    cluster_table(cluster_data)
    feature_table(cell_class, channel_combo)
    cluster_size_charts(channel_combo, cell_class, leiden_resolution)

else:
    # Cluster selected: Two columns: plot | detail.
    col1, col2 = st.columns([1, 1])
    with col1:
        display_cluster(
            cluster_data, cell_class=cell_class, channel_combo=channel_combo
        )
        cluster_table(cluster_data)
        feature_table(cell_class, channel_combo)
        cluster_size_charts(channel_combo, cell_class, leiden_resolution)

    with col2:
        # Selected Gene info
        cell_class = st.session_state.get("cell_class", "all")

        selected_gene_info_df = cluster_data[
            cluster_data["cluster"] == st.session_state.selected_item
        ]
        genes = sorted(selected_gene_info_df["gene_symbol_0"].tolist())
        gene_montages_root = os.path.join(
            BRIEFLOW_OUTPUT_PATH, "aggregate", "montages", f"{cell_class}__montages"
        )

        ## Cluster Info
        # Create two columns for the title and clear button
        title_col, button_col = st.columns([2, 1])
        with title_col:
            st.write(f"## Cluster {st.session_state.selected_item}: {len(genes)} genes")
        with button_col:
            # Add float styling and red color specifically for the Close Cluster button
            st.markdown(
                """
                <style>
                div[data-testid="stButton"] button {
                    float: right;
                    background-color: #dc2626 !important;
                    border-color: #dc2626 !important;
                    color: white !important;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )
            # Show selected item and clear button if an item is selected
            if st.button("Close Cluster"):
                st.session_state.selected_item = None
                st.session_state.selected_gene = None
                st.rerun()

        display_cluster_json(cluster_data)

        ## Montages
        # Check if gene_montages_root directory exists
        if os.path.exists(gene_montages_root):
            st.markdown("#### Gene Montages")

            # Cluster gene dropdown
            if cluster_genes:
                gene_val = (
                    st.session_state.selected_gene
                    if st.session_state.selected_gene in cluster_genes
                    else cluster_genes[0]
                )
                st.session_state.selected_gene_cluster = gene_val
                st.selectbox(
                    "Select a gene to view (within this cluster)",
                    options=cluster_genes,
                    index=cluster_genes.index(gene_val),
                    key="selected_gene_cluster",
                    on_change=on_cluster_gene_select,
                )
            else:
                st.write("No genes found in this cluster.")

            display_uniprot_info()

            # Display montages only for the selected gene
            if st.session_state.selected_gene:
                display_gene_montages(
                    gene_montages_root, st.session_state.selected_gene
                )
            else:
                # If no gene is selected yet, select the first one
                if genes:
                    st.session_state.selected_gene = genes[0]
                    st.rerun()
                else:
                    st.write("No genes found in this cluster.")
        else:
            st.warning(
                f"⚠️ WARNING: Gene montages root directory does not exist: {gene_montages_root}"
            )
