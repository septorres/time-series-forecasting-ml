"""Reconstruction (2026) of a 2023 time-series regression exercise."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["lag_1", "lag_7", "rolling_7", "day_of_week"]


def generate_data(n=500, seed=42):
    if n < 80:
        raise ValueError("n must be at least 80")
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    signal = 100 + 0.04 * days + 8 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 2, n)
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n), "value": signal})


def build_features(data):
    df = data.sort_values("date").copy()
    df["lag_1"] = df["value"].shift(1)
    df["lag_7"] = df["value"].shift(7)
    # Shift first: no observation from the target day enters its features.
    df["rolling_7"] = df["value"].shift(1).rolling(7).mean()
    df["day_of_week"] = df["date"].dt.dayofweek
    return df.dropna().reset_index(drop=True)


def run_experiment(n=500, seed=42):
    df = build_features(generate_data(n, seed))
    cutoff = int(len(df) * 0.8)
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]
    X_train, y_train = train[FEATURES], train["value"]
    X_test, y_test = test[FEATURES], test["value"]
    rf = RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1)
    rf.fit(X_train, y_train)
    # Feature selection based on training data only.
    ranked = sorted(zip(FEATURES, rf.feature_importances_), key=lambda p: p[1], reverse=True)
    selected = [name for name, _ in ranked[:3]]
    mlp = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1200, random_state=seed, early_stopping=True, n_iter_no_change=20)), transformer=StandardScaler())
    mlp.fit(X_train[selected], y_train)
    predictions = {"Random Forest": rf.predict(X_test), "MLP": mlp.predict(X_test[selected])}
    metrics = pd.DataFrame([{"model": name, "MAPE_pct": 100 * mean_absolute_percentage_error(y_test, pred), "MAE": mean_absolute_error(y_test, pred)} for name, pred in predictions.items()])
    comparison = pd.DataFrame({"date": test["date"], "actual": y_test, "random_forest": predictions["Random Forest"], "mlp": predictions["MLP"]})
    return metrics, comparison, pd.DataFrame(ranked, columns=["feature", "importance"]), selected


def main():
    parser = argparse.ArgumentParser(description="Reconstructed 2023 forecasting exercise (synthetic data)")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics, comparison, importance, selected = run_experiment(args.rows)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    comparison.to_csv(args.out / "predictions.csv", index=False)
    importance.to_csv(args.out / "feature_importance.csv", index=False)
    print("Selected features:", ", ".join(selected))
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
