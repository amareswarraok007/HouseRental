from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


DATA_FILES = {
    "metrics": "metrics.csv",
    "logs": "logs.csv",
    "orders": "orders.csv",
    "deployments": "deployments.csv",
}


def load_csvs(base_path: Path) -> Dict[str, pd.DataFrame]:
    data = {}
    for key, filename in DATA_FILES.items():
        file_path = base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {filename}")
        data[key] = pd.read_csv(file_path)
    return data
