"""Log run parameters to a per-plate CSV file after phenotyping completes."""

from datetime import datetime
from pathlib import Path

import pandas as pd


def flatten_config(config, parent_key="", sep="."):
    """Flatten a nested config dict to dot-notation keys."""
    items = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items


# Flatten the config
run_config = dict(snakemake.params.run_config)
flat_config = flatten_config(run_config)

# Add run metadata
flat_config["run_date"] = datetime.now().isoformat()
flat_config["run_name"] = Path(run_config["all"]["root_fp"]).stem
flat_config["plate"] = str(snakemake.wildcards.plate)

# Write single-row CSV
df = pd.DataFrame([flat_config])
output_path = Path(snakemake.output[0])
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Logged run parameters to {output_path}")
