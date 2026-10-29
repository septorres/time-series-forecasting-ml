from src.forecast import FEATURES, build_features, generate_data, run_experiment


def test_features_use_only_past_values():
    df = generate_data(100)
    features = build_features(df)
    first = features.iloc[0]
    assert first["lag_1"] == df.loc[6, "value"]
    assert first["lag_7"] == df.loc[0, "value"]
    assert abs(first["rolling_7"] - df.loc[:6, "value"].mean()) < 1e-9


def test_experiment_outputs():
    metrics, predictions, importance, selected = run_experiment(100)
    assert set(metrics["model"]) == {"Random Forest", "MLP"}
    assert (metrics["MAPE_pct"] >= 0).all()
    assert (metrics["MAPE_pct"] < 100).all()
    assert predictions["date"].is_monotonic_increasing
    assert len(selected) == 3 and set(selected).issubset(FEATURES)
    assert abs(importance["importance"].sum() - 1.0) < 1e-8
