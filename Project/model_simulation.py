import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("Project/models/plots/simulation", exist_ok=True)

# Utility
def get_features(df):
    return [
        "ConfirmedCases", "Fatalities",
        "cases_ma7", "fatal_ma7",
        "case_growth_rate", "fatal_growth_rate",
        "dayofweek", "weekofyear"
    ]

# TRAIN SIMULATION MODELS
def train_simulation_models(df):
    X = df[get_features(df)]
    y = df["ConfirmedCases"]  # target base

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=12, random_state=42
        ),
        "linear_regression": LinearRegression(),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining simulation model: {name}")

        # Fit
        model.fit(X_train, y_train)

        # Prediction
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        # Saved results
        results[name] = {"mae": mae, "rmse": rmse}

        # Learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, train_sizes=np.linspace(0.1, 1, 5)
        )

        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
        plt.title(f"Learning Curve - {name}")
        plt.xlabel("Training Size")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Project/models/plots/simulation/learning_curve_{name}.png")
        plt.close()

    # Save performance table
    pd.DataFrame(results).T.to_csv("Project/models/simulation_results.csv")

    print("\nResults saved on models/simulation_results.csv")
    print("Graphics generated on models/plots/simulation/")

    return models, results

# PERTURBATION SIMULATION
def simulation_perturbation(model, df, eps=0.05):
    X = df[get_features(df)].copy()
    perturb = X * (1 + np.random.uniform(-eps, eps, X.shape))
    pred_original = model.predict(X)
    pred_perturbed = model.predict(perturb)
    delta = np.abs(pred_original - pred_perturbed)

    plt.figure(figsize=(8, 4))
    sns.histplot(delta, bins=40, kde=True)
    plt.title("Perturbation Impact Histogram")
    plt.xlabel("Prediction Change (delta)")
    plt.tight_layout()
    plt.savefig("Project/models/plots/simulation/perturbation_hist.png")
    plt.close()

    return delta.mean()


# FEEDBACK LOOP SIMULATION
def feedback_loop_simulation(model, df, steps=7):
    df_loop = df.copy()
    X = df_loop[get_features(df_loop)]

    results = []

    current_pred = model.predict(X)

    for t in range(steps):

        df_loop["ConfirmedCases"] = current_pred
        X = df_loop[get_features(df_loop)]
        current_pred = model.predict(X)

        results.append(current_pred.mean())

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(results, marker="o")
    plt.title("Feedback Loop Simulation")
    plt.xlabel("Iteration")
    plt.ylabel("Avg Predicted Cases")
    plt.tight_layout()
    plt.savefig("Project/models/plots/simulation/feedback_loop.png")
    plt.close()

    return results
