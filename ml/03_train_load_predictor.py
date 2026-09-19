#!/usr/bin/env python3
"""
Entraîne un Random Forest Regressor pour prédire la charge future du cluster
à partir de son historique récent.

On agrège toutes les métriques par instant (somme du CPU/mémoire de tous les
pods, moyenne CPU/mémoire des nœuds), puis on construit une série temporelle
unique sur laquelle on prédit la valeur à N pas dans le futur à partir des
valeurs passées (lags + fenêtres glissantes).

IMPORTANT sur le choix de la cible :
Si votre workload cible est majoritairement idle en dehors des injections de
chaos (cas de "sleep infinity" + StressChaos), les métriques pod
(total_pod_cpu_usage_cores) sont quasi nulles hors anomalie et ne
constituent pas un bon signal de charge continue. Préférez alors
--target-metric cluster_node_cpu_usage_percent, plus représentatif d'une
vraie dynamique de charge.

IMPORTANT sur le split train/test :
--split-mode chronological (par défaut) réserve les derniers instants comme
test, ce qui est la pratique la plus rigoureuse (aucune fuite du futur) MAIS
peut produire un test set sans aucune anomalie si votre collecte se termine
par une longue période calme après la dernière injection - le R² y est alors
non informatif (variance quasi nulle à expliquer).
--split-mode random mélange aléatoirement les instants avant de découper,
ce qui garantit un test set représentatif de toutes les phases (utile pour
évaluer la capacité du modèle sur les anomalies), au prix d'une petite fuite
d'information tolérable pour un prototype (les features de lignes voisines
en train/test se chevauchent partiellement).

Usage:
    python3 03_train_load_predictor.py --target-metric cluster_node_cpu_usage_percent --split-mode random
    python3 03_train_load_predictor.py --horizon-steps 4 --test-fraction 0.2
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

N_LAGS = 5  # nombre de pas passés utilisés comme features

AVAILABLE_TARGETS = [
    "cluster_node_cpu_usage_percent",
    "cluster_node_memory_usage_percent",
    "total_pod_cpu_usage_cores",
    "total_pod_memory_working_set_bytes",
]


def build_cluster_timeseries(features_path):
    df = pd.read_csv(features_path, parse_dates=["timestamp_utc"])

    cluster_cols = [c for c in df.columns if c.startswith("cluster_")]

    agg = df.groupby("timestamp_utc").agg(
        total_pod_cpu_usage_cores=("pod_cpu_usage_cores", "sum"),
        total_pod_memory_working_set_bytes=("pod_memory_working_set_bytes", "sum"),
        label=("label", "first"),
        **{col: (col, "first") for col in cluster_cols},
    ).reset_index()

    agg["is_anomaly"] = (agg["label"] != "normal").astype(int)
    agg = agg.sort_values("timestamp_utc").reset_index(drop=True)
    return agg


def add_lag_and_rolling_features(df, n_lags, numeric_cols):
    for col in numeric_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll_mean"] = df[col].rolling(n_lags, min_periods=1).mean()
    return df


def run(features_path, target_metric, horizon_steps, test_fraction, split_mode,
        model_output, predictions_output):
    print(f"Chargement de {features_path}...")
    ts = build_cluster_timeseries(features_path)
    print(f"  {len(ts)} instants dans la série temporelle cluster.")
    print(f"  Cible choisie: {target_metric}")

    if target_metric not in ts.columns:
        raise SystemExit(
            f"Erreur: '{target_metric}' n'existe pas dans les colonnes disponibles.\n"
            f"Colonnes disponibles: {[c for c in ts.columns if c not in ('timestamp_utc', 'label')]}"
        )

    numeric_cols = [c for c in ts.columns if c not in ("timestamp_utc", "label", "is_anomaly")]
    ts = add_lag_and_rolling_features(ts, N_LAGS, numeric_cols)

    ts["target"] = ts[target_metric].shift(-horizon_steps)

    ts_clean = ts.dropna().reset_index(drop=True)
    print(f"  {len(ts_clean)} lignes exploitables après création des lags/cible "
          f"(horizon={horizon_steps} pas).")
    print(f"  Dont {ts_clean['is_anomaly'].sum()} lignes en période d'anomalie "
          f"({ts_clean['is_anomaly'].mean():.1%}).")

    feature_cols = [c for c in ts_clean.columns
                    if c not in ("timestamp_utc", "label", "is_anomaly", "target")]
    X = ts_clean[feature_cols].values
    y = ts_clean["target"].values
    timestamps = ts_clean["timestamp_utc"].values
    is_anomaly = ts_clean["is_anomaly"].values

    if split_mode == "chronological":
        split_idx = int(len(ts_clean) * (1 - test_fraction))
        train_idx = np.arange(0, split_idx)
        test_idx = np.arange(split_idx, len(ts_clean))
        print(f"  Split CHRONOLOGIQUE: test = derniers {test_fraction:.0%} des instants.")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(ts_clean)), test_size=test_fraction, random_state=42, shuffle=True
        )
        print(f"  Split ALÉATOIRE: test = {test_fraction:.0%} des instants tirés au hasard.")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ts_test = timestamps[test_idx]
    anomaly_test = is_anomaly[test_idx]

    print(f"  Train: {len(X_train)} lignes | Test: {len(X_test)} lignes "
          f"(dont {anomaly_test.sum()} en anomalie, {anomaly_test.mean():.1%})")

    if anomaly_test.sum() == 0:
        print("  ATTENTION: aucune anomalie dans le jeu de test -> le R² ne reflète "
              "que la capacité à prédire une période stable. Considérez --split-mode random.")

    print("\nEntraînement du Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Évaluation globale sur le jeu de test ===")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  R²   : {r2:.4f}")

    if anomaly_test.sum() > 0 and (anomaly_test == 0).sum() > 0:
        print("\n=== Évaluation détaillée par régime ===")
        for regime, mask in [("normal", anomaly_test == 0), ("anomalie", anomaly_test == 1)]:
            if mask.sum() > 1:
                r2_regime = r2_score(y_test[mask], y_pred[mask])
                rmse_regime = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                print(f"  {regime:10s}: n={mask.sum():4d}  RMSE={rmse_regime:.6f}  R²={r2_regime:.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 features les plus importantes:")
    print(importances.sort_values(ascending=False).head(10))

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "horizon_steps": horizon_steps,
         "target_metric": target_metric},
        model_output,
    )
    print(f"\nModèle sauvegardé dans {model_output}")

    predictions_df = pd.DataFrame({
        "timestamp_utc": ts_test,
        "actual_load": y_test,
        "predicted_load": y_pred,
        "is_anomaly": anomaly_test,
    })
    predictions_df.to_csv(predictions_output, index=False)
    print(f"Prédictions sauvegardées dans {predictions_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Random Forest Regressor (prédiction de charge)")
    parser.add_argument("--input", default="features_dataset.csv")
    parser.add_argument(
        "--target-metric",
        default="cluster_node_cpu_usage_percent",
        choices=AVAILABLE_TARGETS,
        help="Métrique à prédire (défaut: CPU moyen des nœuds, signal continu recommandé)",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=4,
        help="Nombre de pas de 15s dans le futur à prédire (4 = ~1 min d'avance)",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        default="random",
        choices=["chronological", "random"],
        help="'random' (défaut, garantit des anomalies dans le test) ou 'chronological' (plus rigoureux mais risque un test sans anomalie)",
    )
    parser.add_argument("--model-output", default="load_predictor_model.joblib")
    parser.add_argument("--predictions-output", default="load_predictions.csv")
    args = parser.parse_args()

    run(
        args.input,
        args.target_metric,
        args.horizon_steps,
        args.test_fraction,
        args.split_mode,
        args.model_output,
        args.predictions_output,
    )
