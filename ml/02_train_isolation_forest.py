#!/usr/bin/env python3
"""
Entraîne un Isolation Forest pour la détection d'anomalies sur les features
préparées par 01_prepare_features.py.

L'entraînement est non supervisé (l'algorithme ne voit pas la colonne label),
mais on utilise le label connu pour EVALUER la qualité de la détection
(precision/recall/F1), ce qui est possible ici car on a injecté les anomalies
nous-mêmes et on connaît la vérité terrain.

Usage:
    python3 02_train_isolation_forest.py
    python3 02_train_isolation_forest.py --input features_dataset.csv --contamination auto
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

ID_COLS = ["timestamp_utc", "pod", "label", "is_anomaly"]


def select_feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS]


def run(input_path, model_path, scored_output, contamination):
    print(f"Chargement de {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["timestamp_utc"])

    feature_cols = select_feature_columns(df)
    print(f"  {len(feature_cols)} features utilisées: {feature_cols}")

    X = df[feature_cols].fillna(0).values
    y_true = df["is_anomaly"].values

    actual_ratio = y_true.mean()
    print(f"  Proportion réelle d'anomalies dans le dataset: {actual_ratio:.3%}")

    if contamination == "auto":
        contam_value = "auto"
        print("  contamination='auto' (l'algorithme estime lui-même le seuil)")
    else:
        contam_value = float(contamination)
        print(f"  contamination={contam_value} (fournie explicitement)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nEntraînement de l'Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contam_value,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # predict() renvoie -1 pour anomalie, 1 pour normal -> on convertit en 1/0
    raw_pred = model.predict(X_scaled)
    y_pred = (raw_pred == -1).astype(int)
    scores = model.decision_function(X_scaled)  # plus la valeur est basse, plus c'est anormal

    print("\n=== Évaluation contre les vrais labels (multi-classes -> binaire) ===")
    print(classification_report(y_true, y_pred, target_names=["normal", "anomalie"]))

    print("Matrice de confusion (lignes=vérité, colonnes=prédiction):")
    print(confusion_matrix(y_true, y_pred))

    # Sauvegarde du modèle + scaler pour réutilisation
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, model_path)
    print(f"\nModèle sauvegardé dans {model_path}")

    # Sauvegarde du dataset enrichi des scores/prédictions, pour la visualisation
    df["anomaly_score"] = scores
    df["predicted_anomaly"] = y_pred
    df.to_csv(scored_output, index=False)
    print(f"Dataset scoré sauvegardé dans {scored_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Isolation Forest")
    parser.add_argument("--input", default="features_dataset.csv")
    parser.add_argument("--model-output", default="isolation_forest_model.joblib")
    parser.add_argument("--scored-output", default="features_dataset_scored.csv")
    parser.add_argument(
        "--contamination",
        default="auto",
        help="'auto' ou une valeur entre 0 et 0.5 (proportion attendue d'anomalies)",
    )
    args = parser.parse_args()

    run(args.input, args.model_output, args.scored_output, args.contamination)
