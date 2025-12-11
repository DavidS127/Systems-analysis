import os
import json
import hashlib
from datetime import datetime
from typing import Tuple, Dict

import pandas as pd
import numpy as np

# Creation of constants for defining which columns should exist for the training and the ID
EXPECTED_TRAIN_COLS = {"Id", "Province/State", "Country/Region", "Date", "ConfirmedCases", "Fatalities"}
ALTERNATE_TRAIN_COLS = {"Province/State", "Country/Region", "Date", "ConfirmedCases", "Fatalities"}

EXPECTED_TEST_COLS = {"ForecastId", "Province/State", "Country/Region", "Date"}
ALTERNATE_TEST_COLS = {"Province/State", "Country/Region", "Date", "Id"}  

# ---- Helpers ----
#path is the route of the file, with the block size we define the read data of each iteration
def file_hash(path: str, block_size: int = 65536) -> str: #we indicate the return type (string)
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256() # Creation of an object of the function hash SHA256
    #The open file is assigned to f variable --
    #f.read(block_size) brings out the first block an keeps calling this function until the file returns an empty bytes chain
    with open(path, "rb") as f: #The file is opened like a binary the "with" ensures the closing of the file
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block) #each block processes internally the bytes from the block
    return h.hexdigest() #returns a hexadecimal chain (64 characters)

def safe_read_csv(path: str, parse_dates: list = None) -> pd.DataFrame:
    """Read CSV with safe defaults and parse dates."""
    if parse_dates is None:
        parse_dates = ["Date"] if "Date" in pd.read_csv(path, nrows=0).columns else []
    return pd.read_csv(path, parse_dates=parse_dates, keep_default_na=True, na_values=["", "NA", "N/A", "null"])

def create_region_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add/normalize a 'region' column combining Country/Region and Province/State."""
    df = df.copy()
    # Ensure columns exist
    if "Province/State" not in df.columns:
        df["Province/State"] = ""
    if "Country/Region" not in df.columns:
        raise ValueError("Missing required column 'Country/Region'")
    df["Province/State"] = df["Province/State"].fillna("").astype(str).str.strip()
    df["Country/Region"] = df["Country/Region"].fillna("").astype(str).str.strip()
    # create region key
    df["region"] = df["Country/Region"]
    # add province only if non-empty
    mask = df["Province/State"] != ""
    df.loc[mask, "region"] = df.loc[mask, "Country/Region"] + "__" + df.loc[mask, "Province/State"]
    return df

def validate_columns(df: pd.DataFrame, expected: set, alt_expected: set = None) -> Tuple[bool, str]:
    """Return (ok, message)."""
    cols = set(df.columns)
    if expected.issubset(cols):
        return True, "All expected columns present."
    if alt_expected is not None and alt_expected.issubset(cols):
        return True, "Alternate expected columns present."
    missing = expected - cols
    return False, f"Missing columns: {missing}. Present columns: {cols}"

def date_range_info(df: pd.DataFrame, date_col: str = "Date") -> Tuple[str, str]:
    if date_col not in df.columns:
        return None, None
    smin = df[date_col].min()
    smax = df[date_col].max()
    return str(smin), str(smax)

def compute_basic_stats(df: pd.DataFrame, numeric_cols: list) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in numeric_cols:
        if c in df.columns:
            col = df[c].dropna()
            stats[c] = {
                "count": int(col.count()),
                "min": float(col.min()) if not col.empty else None,
                "max": float(col.max()) if not col.empty else None,
                "mean": float(col.mean()) if not col.empty else None,
                "std": float(col.std()) if not col.empty else None
            }
    return stats

# ---- Main ingestion function ----
def load_and_validate(train_path: str, test_path: str, save_meta: bool = True, meta_path: str = "data_manifest.json") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load train/test CSVs and perform basic validation and normalization.
    Returns: (train_df, test_df, metadata)
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # compute file hashes (for traceability)
    train_hash = file_hash(train_path)
    test_hash = file_hash(test_path) 

    # read csvs safely
    train_df = safe_read_csv(train_path, parse_dates=["Date"])
    test_df = safe_read_csv(test_path, parse_dates=["Date"])

    # basic column validation
    ok_train, msg_train = validate_columns(train_df, EXPECTED_TRAIN_COLS, ALTERNATE_TRAIN_COLS)
    ok_test, msg_test = validate_columns(test_df, EXPECTED_TEST_COLS, ALTERNATE_TEST_COLS)
    if not ok_train:
        raise ValueError(f"Train columns validation failed: {msg_train}")
    if not ok_test:
        raise ValueError(f"Test columns validation failed: {msg_test}")

    # Normalize whitespace and types for Country/Province
    for df in (train_df, test_df):
        if "Province/State" not in df.columns:
            df["Province/State"] = ""
        df["Province/State"] = df["Province/State"].astype(str).str.strip()
        if "Country/Region" in df.columns:
            df["Country/Region"] = df["Country/Region"].astype(str).str.strip()

    # ensure date column parsed, otherwise try to coerce
    for df, path in [(train_df, train_path), (test_df, test_path)]:
        if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            except Exception:
                raise ValueError(f"Unable to parse 'Date' column in {path}")
        # check for NaT
        if df["Date"].isna().any():
            nbad = int(df["Date"].isna().sum())
            raise ValueError(f"Found {nbad} unparsable dates in {path}")

    # create region key
    train_df = create_region_key(train_df)
    test_df = create_region_key(test_df)

    # fix cumulative non-decreasing check presence by doing a basic check for negative deltas
    # (we don't modify values here, just note anomalies)
    anomalies = {}
    for col in ("ConfirmedCases", "Fatalities"):
        if col in train_df.columns:
            neg_deltas = []
            grouped = train_df.sort_values(["region", "Date"]).groupby("region")
            for region, g in grouped:
                vals = g[col].fillna(0).values
                diffs = np.diff(vals)
                if (diffs < 0).any():
                    neg_deltas.append(region)
            anomalies[col + "_negative_deltas_regions"] = neg_deltas

    # basic statistics
    stats_train = compute_basic_stats(train_df, ["ConfirmedCases", "Fatalities"])
    stats_test = compute_basic_stats(test_df, [])

    # date ranges
    train_min_date, train_max_date = date_range_info(train_df)
    test_min_date, test_max_date = date_range_info(test_df)

    # duplicates check
    dup_train = int(train_df.duplicated(subset=["region", "Date"]).sum())
    dup_test = int(test_df.duplicated(subset=["region", "Date"]).sum())

    # prepare metadata
    metadata = {
        "train_file": os.path.abspath(train_path),
        "test_file": os.path.abspath(test_path),
        "train_hash": train_hash,
        "test_hash": test_hash,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": {"min": train_min_date, "max": train_max_date},
        "test_date_range": {"min": test_min_date, "max": test_max_date},
        "train_stats": stats_train,
        "test_stats": stats_test,
        "anomalies": anomalies,
        "duplicates": {"train": dup_train, "test": dup_test},
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # save metadata
    if save_meta:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return train_df, test_df, metadata


# --- CLI support ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and validate COVID-19 Kaggle train/test CSVs.")
    parser.add_argument("--train", type=str, default="Project/data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="Project/data/test.csv", help="Path to test.csv")
    parser.add_argument("--meta", type=str, default="Project/models/data_manifest.json", help="Path to save metadata json")
    args = parser.parse_args()

    try:
        tr, te, meta = load_and_validate(args.train, args.test, save_meta=True, meta_path=args.meta)
        print("Loaded train rows:", len(tr), "test rows:", len(te))
        print("Metadata saved to:", args.meta)
    except Exception as e:
        print("ERROR during ingestion:", e)
        raise