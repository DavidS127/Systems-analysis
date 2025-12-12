import os
import pandas as pd

# Preprocessing pipeline
from DataCleansing import load_and_validate
from preprocessing import clean_data, feature_engineering

# Modeling (base models + CV + sensitivity)
from modeling import train_model, sensitivity_analysis

# Forecast autoregrssive pipeline
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

os.makedirs("Project/data", exist_ok=True)
os.makedirs("Project/models", exist_ok=True)
os.makedirs("Project/submissions", exist_ok=True)
os.makedirs("Project/models/plots", exist_ok=True)

print("\n=== CHARGING DATA ===")

train_df, test_df, meta = load_and_validate(
    "Project/data/train.csv",
    "Project/data/test.csv",
    save_meta=True,
    meta_path="Project/models/data_manifest.json"
)


print("\n=== CLEANING DATA ===")

clean_train = clean_data(train_df)
clean_test = clean_data(test_df)

print("\n=== GENERATING FEATURES ===")

feat_train = feature_engineering(clean_train)
feat_test = feature_engineering(clean_test)

print("\n=== TRAINING BASE MODELS  ===")

model_xgb_base = train_model(feat_train, model_type="xgb")
model_rf = train_model(feat_train, model_type="rf")
model_gbr = train_model(feat_train, model_type="gbr")

print("\n=== ANALAZYING SENSIBILITY ===")

sensitivity_analysis(model_xgb_base, feat_train)


# ============================================================
#  SCENARIO 1 – DATA-DRIVEN SIMULATION
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
# SCENARIO 2 – EVENT-BASED SIMULATION (CELLULAR AUTOMATA)
# ============================================================
print("\n=== SCENARIO 2: CELLULAR AUTOMATA SIMULATION ===")

snapshots = simulate_cellular_automata()


# ============================================================
# FORECAST AUTORREGRESIVO (PRINCIPAL PIPELINE )
# ============================================================
print("\n=== TRAINING AUTOREGRESSIVE MODELS ===")

model_cases = train_xgb_cases(feat_train)
model_fatal = train_xgb_fatalities(feat_train)

forecast_df = forecast_future(
    model_cases,
    model_fatal,
    feat_train,
    feat_test
)


# ============================================================
# SUBMISSION FILE CREATION
# ============================================================
print("\n=== SUBMISSION.CSV CREATING ===")

submission = forecast_df[forecast_df["ForecastId"].notna()][[
    "ForecastId", "ConfirmedCases", "Fatalities"
]]

submission_path = "Project/submissions/submission.csv"
submission.to_csv(submission_path, index=False)

print("\n====================================================")
print("Pipeline COMPLETED")
print("Base models trained")
print("Sensitivity done")
print("Simulation Scenario 1 Ready")
print("Cellular Automata completed")
print("Forecast generated")
print("submission.csv saved on:", submission_path)
print("Visual reports on: Project/models/plots/")
print("====================================================\n")
