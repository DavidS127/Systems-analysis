import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. DATA CLEANING
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ConfirmedCases" not in df.columns:
        df["ConfirmedCases"] = 0
    if "Fatalities" not in df.columns:
        df["Fatalities"] = 0

    df = df.sort_values(["region", "Date"]).reset_index(drop=True)

    numeric_cols = ["ConfirmedCases", "Fatalities"]

    df[numeric_cols] = (
        df.groupby("region")[numeric_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    df[numeric_cols] = df[numeric_cols].fillna(0)

    df["ConfirmedCases_norm"] = (
        (df["ConfirmedCases"] - df["ConfirmedCases"].min()) /
        (df["ConfirmedCases"].max() - df["ConfirmedCases"].min() + 1e-9)
    )

    df["Fatalities_norm"] = (
        (df["Fatalities"] - df["Fatalities"].min()) /
        (df["Fatalities"].max() - df["Fatalities"].min() + 1e-9)
    )

    df["ConfirmedCases_std"] = (
        (df["ConfirmedCases"] - df["ConfirmedCases"].mean()) /
        (df["ConfirmedCases"].std() + 1e-9)
    )
    df["Fatalities_std"] = (
        (df["Fatalities"] - df["Fatalities"].mean()) /
        (df["Fatalities"].std() + 1e-9)
    )

    df["case_jump"] = df.groupby("region")["ConfirmedCases"].diff().fillna(0)
    df["fatal_jump"] = df.groupby("region")["Fatalities"].diff().fillna(0)
    df["jump_flag"] = (df["case_jump"].abs() > 50) | (df["fatal_jump"].abs() > 20)

    return df


# 2. FEATURE ENGINEERING
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dayofweek"] = df["Date"].dt.dayofweek
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    df["case_growth_rate"] = (
        df.groupby("region")["ConfirmedCases"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["fatal_growth_rate"] = (
        df.groupby("region")["Fatalities"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["cases_ma7"] = (
        df.groupby("region")["ConfirmedCases"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["fatal_ma7"] = (
        df.groupby("region")["Fatalities"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    return df