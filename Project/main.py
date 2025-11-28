import os
import pandas as pd

# Preprocessing pipeline
from DataCleansing import load_and_validate
from preprocessing import clean_data, feature_engineering

# Modeling (base models + CV + sensitivity)
from modeling import train_model, sensitivity_analysis

# Forecast autoregresivo
from model_xgb_forecast import (
    train_xgb_cases,
    train_xgb_fatalities,
    forecast_future
)

# Scenario 1 – Data-driven Simulation
from model_simulation import (
    train_simulation_models,
    simulation_perturbation,
    feedback_loop_simulation
)

# Scenario 2 – Cellular Automata
from cellular_automata import simulate_cellular_automata


# ============================================================
# 0. CREAR CARPETAS NECESARIAS
# ============================================================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
os.makedirs("models/plots", exist_ok=True)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("\n=== CARGANDO DATOS ===")

train_df, test_df, meta = load_and_validate(
    "data/train.csv",
    "data/test.csv",
    save_meta=True,
    meta_path="models/data_manifest.json"
)


# ============================================================
# 2. LIMPIEZA
# ============================================================
print("\n=== LIMPIANDO DATOS ===")

clean_train = clean_data(train_df)
clean_test = clean_data(test_df)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n=== GENERANDO FEATURES ===")

feat_train = feature_engineering(clean_train)
feat_test = feature_engineering(clean_test)


# ============================================================
# 4. MODELOS BASE (RF, GBR, XGB)
# ============================================================
print("\n=== ENTRENANDO MODELOS BASE ===")

model_xgb_base = train_model(feat_train, model_type="xgb")
model_rf = train_model(feat_train, model_type="rf")
model_gbr = train_model(feat_train, model_type="gbr")


# ============================================================
# 5. SENSITIVITY ANALYSIS
# ============================================================
print("\n=== ANALIZANDO SENSIBILIDAD ===")

sensitivity_analysis(model_xgb_base, feat_train)


# ============================================================
# 6. SCENARIO 1 – DATA-DRIVEN SIMULATION
# ============================================================
print("\n=== SCENARIO 1: DATA-DRIVEN SIMULATION ===")

models_sim, results_sim = train_simulation_models(feat_train)

delta_impact = simulation_perturbation(
    models_sim["random_forest"], 
    feat_train
)

loop_results = feedback_loop_simulation(
    models_sim["random_forest"], 
    feat_train
)


# ============================================================
# 7. SCENARIO 2 – EVENT-BASED SIMULATION (CELLULAR AUTOMATA)
# ============================================================
print("\n=== SCENARIO 2: CELLULAR AUTOMATA SIMULATION ===")

snapshots = simulate_cellular_automata()


# ============================================================
# 8. FORECAST AUTORREGRESIVO (PIPELINE PRINCIPAL)
# ============================================================
print("\n=== ENTRENANDO MODELOS AUTORREGRESIVOS ===")

model_cases = train_xgb_cases(feat_train)
model_fatal = train_xgb_fatalities(feat_train)

forecast_df = forecast_future(
    model_cases,
    model_fatal,
    feat_train,
    feat_test
)


# ============================================================
# 9. ARCHIVO DE SUBMISSION
# ============================================================
print("\n=== CREANDO SUBMISSION.CSV ===")

submission = forecast_df[forecast_df["ForecastId"].notna()][[
    "ForecastId", "ConfirmedCases", "Fatalities"
]]

submission_path = "submissions/submission.csv"
submission.to_csv(submission_path, index=False)


# ============================================================
# 10. FIN
# ============================================================
print("\n====================================================")
print("✔ Pipeline COMPLETO")
print("✔ Modelos base entrenados")
print("✔ Sensitivity realizado")
print("✔ Simulation Scenario 1 listo")
print("✔ Cellular Automata completo")
print("✔ Forecast generado")
print("✔ submission.csv guardado en:", submission_path)
print("✔ Reportes visuales en /models/plots/")
print("====================================================\n")
