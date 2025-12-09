"""
modeling.py

Opción A:
- Entrena modelos base: RandomForest, GradientBoosting, XGBoost
- Validación cruzada
- Ajuste simple de hiperparámetros
- Análisis de sensibilidad básico
- Guardado en .pkl con joblib
- Reporte final en models/model_report.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import xgboost as xgb


# ======================================================
# UTILIDADES
# ======================================================
def ensure_models_folder():
    os.makedirs("Project/models", exist_ok=True)


def get_features(df):
    """
    Devuelve las columnas que se usarán como entrada del modelo.
    """
    return [
        "ConfirmedCases", "Fatalities",
        "cases_ma7", "fatal_ma7",
        "case_growth_rate", "fatal_growth_rate",
        "dayofweek", "weekofyear"
    ]


# ======================================================
# ENTRENAR MODELOS BASE
# ======================================================
def train_model(train_df, model_type="xgb"):
    """
    Entrena un modelo base con validación cruzada y lo guarda en /models/.
    
    model_type: "xgb", "rf", "gbr"
    """

    ensure_models_folder()

    X = train_df[get_features(train_df)]
    y = train_df["ConfirmedCases"]  # objetivo base

    # Selección del modelo
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=350,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model_name = "random_forest"

    elif model_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4
        )
        model_name = "gradient_boosting"

    else:  # xgb
        model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist"
        )
        model_name = "xgboost"

    # ========================
    # VALIDACIÓN CRUZADA
    # ========================
    print(f"\nValidación cruzada para {model_name}...")

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mean_cv = -cv_scores.mean()
    std_cv = cv_scores.std()

    print(f"MAE CV: {mean_cv:.4f} ± {std_cv:.4f}")

    # ========================
    # ENTRENAMIENTO FINAL
    # ========================
    print(f"Entrenando modelo final {model_name}...")
    model.fit(X, y)

    # Guardar modelo
    model_path = f"Project/models/{model_name}.pkl"
    dump(model, model_path)

    print(f"Modelo guardado en: {model_path}")

    # ========================
    # GUARDADO DE RESULTADOS CV
    # ========================
    report_path = "Project/models/model_report.csv"
    row = {
        "modelo": model_name,
        "cv_mae_media": mean_cv,
        "cv_mae_std": std_cv
    }

    if os.path.exists(report_path):
        report = pd.read_csv(report_path)
        report = pd.concat([report, pd.DataFrame([row])], ignore_index=True)
    else:
        report = pd.DataFrame([row])

    report.to_csv(report_path, index=False)

    print(f"Reporte actualizado: {report_path}")

    return model


# ======================================================
# ANÁLISIS DE SENSIBILIDAD
# ======================================================
def sensitivity_analysis(model, df, perturbation=0.05, samples=200):
    """
    Mide cómo cambia la predicción cuando perturbamos las entradas ligeramente.
    
    perturbation = 0.05 → (+/- 5%)
    """
    ensure_models_folder()

    features = get_features(df)
    X = df[features].copy()
    y = df["ConfirmedCases"]

    # Muestras aleatorias
    X_sample = X.sample(samples, random_state=42)

    # Predicción original
    preds_original = model.predict(X_sample)

    # Perturbación
    X_perturbed = X_sample * (1 + np.random.uniform(-perturbation, perturbation, X_sample.shape))

    # Predicción perturbada
    preds_perturbed = model.predict(X_perturbed)

    # Diferencia absoluta
    sensitivity = np.abs(preds_original - preds_perturbed)

    result = pd.DataFrame({
        "pred_original": preds_original,
        "pred_perturbed": preds_perturbed,
        "delta": sensitivity
    })

    # Guardar reporte
    result_path = "Project/models/sensitivity_report.csv"
    result.to_csv(result_path, index=False)

    print(f"Sensitivity Analysis guardado en: {result_path}")
    print(f"Promedio de impacto en predicción: {sensitivity.mean():.4f}")

    return sensitivity.mean()
