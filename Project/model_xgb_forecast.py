import pandas as pd
import numpy as np
import xgboost as xgb
import os

def train_xgb_cases(train_df):
    os.makedirs("Project/models", exist_ok=True)

    features = [
        "ConfirmedCases", "Fatalities",
        "cases_ma7", "fatal_ma7",
        "case_growth_rate", "fatal_growth_rate",
        "dayofweek", "weekofyear"
    ]

    target = "ConfirmedCases"

    model = xgb.XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist"
    )

    model.fit(train_df[features], train_df[target])

    # Save model for cases
    model.save_model("Project/models/xgb_cases.json")

    return model

def train_xgb_fatalities(train_df):
    os.makedirs("Project/models", exist_ok=True)

    features = [
        "ConfirmedCases", "Fatalities",
        "cases_ma7", "fatal_ma7",
        "case_growth_rate", "fatal_growth_rate",
        "dayofweek", "weekofyear"
    ]

    target = "Fatalities"

    model = xgb.XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist"
    )

    model.fit(train_df[features], train_df[target])

    # Save model for fatalities
    model.save_model("Project/models/xgb_fatalities.json")

    return model

def forecast_future(model_cases, model_fatal, train_df, test_df):

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.sort_values(["region", "Date"]).reset_index(drop=True)

    features = [
        "ConfirmedCases", "Fatalities",
        "cases_ma7", "fatal_ma7",
        "case_growth_rate", "fatal_growth_rate",
        "dayofweek", "weekofyear"
    ]

    for region in df["region"].unique():

        region_df = df[df["region"] == region].copy()
        idxs = region_df.index

        future_idxs = region_df[region_df["ForecastId"].notna()].index

        for idx in future_idxs:

            # Recalculate features
            region_df.loc[idx, "cases_ma7"] = region_df.loc[:idx, "ConfirmedCases"].tail(7).mean()
            region_df.loc[idx, "fatal_ma7"] = region_df.loc[:idx, "Fatalities"].tail(7).mean()

            region_df.loc[idx, "case_growth_rate"] = (
                (region_df.loc[idx - 1, "ConfirmedCases"] - region_df.loc[idx - 2, "ConfirmedCases"]) /
                (region_df.loc[idx - 2, "ConfirmedCases"] + 1e-9)
            )

            region_df.loc[idx, "fatal_growth_rate"] = (
                (region_df.loc[idx - 1, "Fatalities"] - region_df.loc[idx - 2, "Fatalities"]) /
                (region_df.loc[idx - 2, "Fatalities"] + 1e-9)
            )

            X_input = region_df.loc[idx, features].values.reshape(1, -1)

            # Cases prediction
            pred_cases = model_cases.predict(X_input)[0]
            region_df.loc[idx, "ConfirmedCases"] = max(pred_cases, 0)

            # fatalities prediction
            pred_fatal = model_fatal.predict(X_input)[0]
            region_df.loc[idx, "Fatalities"] = max(pred_fatal, 0)

        df.loc[idxs, ["ConfirmedCases", "Fatalities"]] = region_df[["ConfirmedCases", "Fatalities"]]

    return df
