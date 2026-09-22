"""Recording parsing and physical-FBG column mapping for both dataset orders.

Column 0 is time; sensor columns are 1 through 12. No detection or UI logic.
The UI owns caching while this module reads and selects recording data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def load_recording_frame(path_text: str) -> pd.DataFrame:
    """Read one recording once and keep only the two most recent files in RAM."""
    path = Path(path_text)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        usecols=range(13),
        decimal=",",
        engine="c",
    ).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[0]).sort_values(0).drop_duplicates(0)
    if len(frame) < 20:
        raise ValueError("The recording contains fewer than 20 valid samples")
    return frame


def load_physical_fbg_from_frame(
    frame: pd.DataFrame, physical_fbg: int
) -> dict[str, object]:
    """Select one physical FBG after detecting this recording's stored order."""
    if physical_fbg not in range(1, 13):
        raise ValueError("Physical FBG must be between 1 and 12")
    medians = frame.loc[:, 1:12].median(axis=0, skipna=True)
    if not np.isfinite(medians.loc[1]) or not np.isfinite(medians.loc[12]):
        raise ValueError("Stored wavelength order cannot be identified")
    ascending = bool(medians.loc[1] < medians.loc[12])
    stored_by_physical_fbg = {
        sensor: sensor if ascending else 13 - sensor for sensor in range(1, 13)
    }
    stored_column = stored_by_physical_fbg[physical_fbg]
    selected = frame.loc[:, [0, stored_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 20:
        raise ValueError(f"Physical FBG {physical_fbg} has too few valid samples")
    return {
        "time_s": selected[0].to_numpy(dtype=float),
        "raw_nm": selected[stored_column].to_numpy(dtype=float),
        "physical_fbg": physical_fbg,
        "stored_column": stored_column,
        "stored_order": "ascending" if ascending else "descending",
    }
