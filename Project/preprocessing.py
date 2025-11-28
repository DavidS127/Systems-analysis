"""
preprocessing.py

Módulo de limpieza, ingeniería de características
y visualización para datasets COVID con columnas:
['region', 'Date', 'ConfirmedCases', 'Fatalities']

Funciones principales:
- clean_data(df)
- feature_engineering(df)
- plot_preprocessing_effects(original_df, cleaned_df, region)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# 1. LIMPIEZA DE LOS DATOS
# ===============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Asegurar columnas para TEST (que no las tiene)
    if "ConfirmedCases" not in df.columns:
        df["ConfirmedCases"] = 0
    if "Fatalities" not in df.columns:
        df["Fatalities"] = 0

    # Asegurar orden temporal
    df = df.sort_values(["region", "Date"]).reset_index(drop=True)

    numeric_cols = ["ConfirmedCases", "Fatalities"]

    # Imputación segura por región
    df[numeric_cols] = (
        df.groupby("region")[numeric_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    # Imputación final
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Normalización
    df["ConfirmedCases_norm"] = (
        (df["ConfirmedCases"] - df["ConfirmedCases"].min()) /
        (df["ConfirmedCases"].max() - df["ConfirmedCases"].min() + 1e-9)
    )

    df["Fatalities_norm"] = (
        (df["Fatalities"] - df["Fatalities"].min()) /
        (df["Fatalities"].max() - df["Fatalities"].min() + 1e-9)
    )

    # Estandarización
    df["ConfirmedCases_std"] = (
        (df["ConfirmedCases"] - df["ConfirmedCases"].mean()) /
        (df["ConfirmedCases"].std() + 1e-9)
    )
    df["Fatalities_std"] = (
        (df["Fatalities"] - df["Fatalities"].mean()) /
        (df["Fatalities"].std() + 1e-9)
    )

    # Saltos bruscos
    df["case_jump"] = df.groupby("region")["ConfirmedCases"].diff().fillna(0)
    df["fatal_jump"] = df.groupby("region")["Fatalities"].diff().fillna(0)
    df["jump_flag"] = (df["case_jump"].abs() > 50) | (df["fatal_jump"].abs() > 20)

    return df



# ===============================
# 2. FEATURE ENGINEERING
# ===============================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas características derivadas:
    - Día de la semana
    - Semana del año
    - Tasa de crecimiento diaria
    - Promedios móviles
    """

    df = df.copy()

    # Variables temporales
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Tasa de crecimiento diaria
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

    # Promedios móviles de 7 días
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


# ===============================
# 3. VISUALIZACIONES
# ===============================
def plot_preprocessing_effects(original_df, cleaned_df, region):
    """
    Muestra la diferencia entre el dataset original y el limpiado:
    - ConfirmedCases original vs limpio
    - Detección de saltos bruscos
    """

    orig = original_df[original_df["region"] == region]
    clean = cleaned_df[cleaned_df["region"] == region]

    # -------- Comparación antes/después --------
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=orig, x="Date", y="ConfirmedCases", label="Original")
    sns.lineplot(data=clean, x="Date", y="ConfirmedCases", label="Limpio")
    plt.title(f"Comparación de casos antes y después del preprocesamiento – {region}")
    plt.xlabel("Fecha")
    plt.ylabel("Casos")
    plt.grid(True)
    plt.show()

    # -------- Saltos bruscos --------
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=clean,
        x="Date",
        y="case_jump",
        hue="jump_flag",
        dodge=False,
        palette="viridis",
        legend=False
    )
    plt.title(f"Saltos bruscos detectados – {region}")
    plt.xlabel("Fecha")
    plt.ylabel("Cambio diario en casos")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
