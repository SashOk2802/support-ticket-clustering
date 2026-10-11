"""Dataset loading shared by preprocessing, benchmarking and inference."""

import csv
from pathlib import Path

import pandas as pd


def read_dataset(path):
    """Read UTF-8 CSV (comma/semicolon/tab) or XLSX without logging row contents."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            sample = source.read(65536)
        try:
            separator = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            separator = ","
        return pd.read_csv(path, sep=separator, encoding="utf-8-sig")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    raise ValueError("Supported dataset formats: .csv and .xlsx")


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")


def load_text_dataset(path, text_column="text"):
    frame = read_dataset(path)
    require_columns(frame, [text_column])
    valid_mask = frame[text_column].notna() & frame[text_column].astype(str).str.strip().ne("")
    texts = frame.loc[valid_mask, text_column].astype(str).str.strip().tolist()
    if len(texts) < 3:
        raise ValueError("At least three non-empty texts are required for clustering.")
    return frame, texts, valid_mask
